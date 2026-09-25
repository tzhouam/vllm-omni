#!/usr/bin/env python3
"""Split the pinned Qwen3-TTS layer into CPU attention/KV and NPU-candidate MLP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_SHA256 = "5790b5a750eab743bf1c84270467157bcd9d4e4243ec9cd87c6e1da5c6dd8900"
FIXTURE_SHA256 = "c96efd647f8078789a9790b929e8e5a501a4a2399e9b35e16bdd22f6f1752304"
ATTENTION_RESIDUAL = "/transformer/layers.0/Add_output_0"
FINAL_HIDDEN = "/transformer/layers.0/Add_1_output_0"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    first, second = reference.astype(np.float64), actual.astype(np.float64)
    return float(np.linalg.norm(first - second) / max(np.linalg.norm(first), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fixture", "attention", "mlp", "mlp-fixture", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.source) != SOURCE_SHA256 or sha256(args.fixture) != FIXTURE_SHA256:
        raise ValueError("pinned source layer or eleven-step fixture changed")

    import onnx
    import onnxruntime as ort

    for path in (args.attention, args.mlp, args.mlp_fixture, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    onnx.utils.extract_model(
        str(args.source), str(args.attention),
        ["projected", "positions", "key_0", "value_0"],
        [ATTENTION_RESIDUAL, "next_key_0", "next_value_0"],
    )
    onnx.utils.extract_model(str(args.source), str(args.mlp),
                             [ATTENTION_RESIDUAL], [FINAL_HIDDEN])
    mlp = onnx.load(str(args.mlp))
    mlp.graph.input[0].name = "residual"
    for node in mlp.graph.node:
        for index, name in enumerate(node.input):
            if name == ATTENTION_RESIDUAL:
                node.input[index] = "residual"
    onnx.save(mlp, str(args.mlp))
    onnx.checker.check_model(str(args.attention))
    onnx.checker.check_model(str(args.mlp))

    with np.load(args.fixture, allow_pickle=False) as archive:
        fixture = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    source = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    attention = ort.InferenceSession(str(args.attention), providers=["CPUExecutionProvider"])
    mlp_session = ort.InferenceSession(str(args.mlp), providers=["CPUExecutionProvider"])
    if ([item.name for item in attention.get_inputs()]
            != ["projected", "positions", "key_0", "value_0"]
            or [item.name for item in mlp_session.get_inputs()] != ["residual"]):
        raise ValueError("extracted attention/MLP input contract changed")

    state = {name: fixture[name] for name in ("key_0", "value_0")}
    captured = {}
    rows = []
    for index in range(11):
        start = 95 + 2 * index
        projected = fixture["projected" if index == 0 else "projected_next" if index == 1
                            else f"projected_step{index}"]
        feeds = {"projected": projected,
                 "positions": np.array([[start, start + 1]], np.int64), **state}
        direct = source.run(None, feeds)
        residual, next_key, next_value = attention.run(None, feeds)
        hidden = mlp_session.run(None, {"residual": residual})[0]
        errors = [relative_l2(a, b) for a, b in zip(direct, (hidden, next_key, next_value))]
        if max(errors) > 1e-6:
            raise ValueError(f"CPU attention/MLP composition changed layer at {start}: {errors}")
        captured[f"residual_step{index}"] = residual
        for output_index, output in enumerate(direct):
            captured[f"cpu_step{index}_out{output_index}"] = output
        rows.append({"start_frame": start, "max_relative_l2": max(errors)})
        state = {"key_0": direct[1], "value_0": direct[2]}

    np.savez_compressed(args.mlp_fixture, **captured)
    report = {
        "scope": "pinned real-weight layer-0 CPU attention/KV plus extracted MLP; CPU composition gate only",
        "source_sha256": SOURCE_SHA256,
        "source_fixture_sha256": FIXTURE_SHA256,
        "attention_sha256": sha256(args.attention),
        "mlp_sha256": sha256(args.mlp),
        "mlp_fixture_sha256": sha256(args.mlp_fixture),
        "onnxruntime": ort.__version__,
        "rows": rows,
        "status": "eleven_step_cpu_composition_parity_pass",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "mlp_sha256": report["mlp_sha256"],
                      "mlp_fixture_sha256": report["mlp_fixture_sha256"]}))


if __name__ == "__main__":
    main()
