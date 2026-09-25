#!/usr/bin/env python3
"""Cut the real Qwen3-TTS layer after its input projection and verify CPU parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BASE_SHA256 = "b4c28d1393496be9ebc9815e6de32ea2f955dbfe4219206a224426b038923bca"
FIXTURE_SHA256 = "1ef435ccc70484fb4b7971df822eee1315a50294cb6f3218c1b21a1dfb89fb43"
CUT = "/transformer/input_proj/Add_output_0"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "fixture", "prefix", "suffix", "split-fixture", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.base) != BASE_SHA256 or sha256(args.fixture) != FIXTURE_SHA256:
        raise ValueError("source layer or eleven-step fixture changed")

    import onnx
    import onnxruntime as ort

    for path in (args.prefix, args.suffix, args.split_fixture, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    onnx.utils.extract_model(str(args.base), str(args.prefix), ["conv"], [CUT])
    onnx.utils.extract_model(str(args.base), str(args.suffix),
                             [CUT, "positions", "key_0", "value_0"],
                             ["/transformer/layers.0/Add_1_output_0",
                              "next_key_0", "next_value_0"])
    suffix = onnx.load(str(args.suffix))
    if suffix.graph.input[0].name != CUT:
        raise ValueError("extracted suffix input order changed")
    suffix.graph.input[0].name = "projected"
    for node in suffix.graph.node:
        for index, name in enumerate(node.input):
            if name == CUT:
                node.input[index] = "projected"
    onnx.save(suffix, str(args.suffix))
    onnx.checker.check_model(str(args.prefix))
    onnx.checker.check_model(str(args.suffix))

    with np.load(args.fixture, allow_pickle=False) as archive:
        fixture = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    base = ort.InferenceSession(str(args.base), providers=["CPUExecutionProvider"])
    prefix = ort.InferenceSession(str(args.prefix), providers=["CPUExecutionProvider"])
    suffix = ort.InferenceSession(str(args.suffix), providers=["CPUExecutionProvider"])
    if ([item.name for item in suffix.get_inputs()]
            != ["projected", "positions", "key_0", "value_0"]):
        raise ValueError("split suffix input contract changed")
    base_state = {name: fixture[name] for name in ("key_0", "value_0")}
    split_state = {name: fixture[name] for name in ("key_0", "value_0")}
    rows = []
    for index in range(11):
        start = 95 + 2 * index
        conv = fixture["conv" if index == 0 else "conv_next" if index == 1
                       else f"conv_step{index}"]
        projected = prefix.run(None, {"conv": conv})[0]
        fixture["projected" if index == 0 else "projected_next" if index == 1
                else f"projected_step{index}"] = projected
        positions = np.array([[start, start + 1]], np.int64)
        direct = base.run(None, {"conv": conv, "positions": positions, **base_state})
        split = suffix.run(None, {"projected": projected, "positions": positions,
                                  **split_state})
        errors = [relative_l2(a, b) for a, b in zip(direct, split)]
        rows.append({"start_frame": start, "max_relative_l2": max(errors)})
        if max(errors) > 1e-6:
            raise ValueError(f"CPU split changed layer output at {start}: {errors}")
        base_state = {name: direct[output_index + 1]
                      for output_index, name in enumerate(("key_0", "value_0"))}
        split_state = {name: split[output_index + 1]
                       for output_index, name in enumerate(("key_0", "value_0"))}
    np.savez_compressed(args.split_fixture, **fixture)
    report = {
        "scope": "real-weight first decoder layer split after CPU input projection; CPU composition, not NPU quality",
        "base_sha256": BASE_SHA256,
        "source_fixture_sha256": FIXTURE_SHA256,
        "prefix_sha256": sha256(args.prefix),
        "suffix_sha256": sha256(args.suffix),
        "split_fixture_sha256": sha256(args.split_fixture),
        "onnxruntime": ort.__version__,
        "rows": rows,
        "status": "eleven_step_cpu_split_parity_pass",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "suffix_sha256": report["suffix_sha256"],
                      "split_fixture_sha256": report["split_fixture_sha256"]}))


if __name__ == "__main__":
    main()
