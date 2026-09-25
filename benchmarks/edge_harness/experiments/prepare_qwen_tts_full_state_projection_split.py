#!/usr/bin/env python3
"""Extract a source-faithful CPU input projection and eight-layer suffix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_SHA256 = "2a74a49e99ffeddd4918cade906aea0a497bc9651279475eaaa583cd386b09fd"
FIXTURE_SHA256 = (
    "1ef435ccc70484fb4b7971df822eee1315a50294cb6f3218c1b21a1dfb89fb43",
    "924d67394c1ea8c027aa7e6b7b5a34f00087759523dc061a3c2bc4fa7f2cd45b",
)
CUT = "/transformer/input_proj/Add_output_0"
STATE_NAMES = tuple(f"{kind}_{layer}" for layer in range(8) for kind in ("key", "value"))
OUTPUT_NAMES = ("hidden",) + tuple(
    f"next_{kind}_{layer}" for layer in range(8) for kind in ("key", "value")
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prefix", "suffix", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, action="append", required=True)
    parser.add_argument("--split-fixture", type=Path, action="append", required=True)
    args = parser.parse_args()
    if (len(args.fixture) != 2 or len(args.split_fixture) != 2
            or sha256(args.source) != SOURCE_SHA256
            or tuple(sha256(path) for path in args.fixture) != FIXTURE_SHA256):
        raise ValueError("pinned eight-layer graph or two utterance fixtures changed")

    import onnx
    import onnxruntime as ort

    for path in (args.prefix, args.suffix, args.report, *args.split_fixture):
        path.parent.mkdir(parents=True, exist_ok=True)
    source = onnx.load(str(args.source))
    if ([item.name for item in source.graph.input] !=
            ["conv", "positions", *STATE_NAMES]
            or [item.name for item in source.graph.output] != list(OUTPUT_NAMES)):
        raise ValueError("eight-layer ONNX input/output contract changed")
    onnx.utils.extract_model(str(args.source), str(args.prefix), ["conv"], [CUT])
    onnx.utils.extract_model(str(args.source), str(args.suffix),
                             [CUT, "positions", *STATE_NAMES], list(OUTPUT_NAMES))
    suffix_graph = onnx.load(str(args.suffix))
    if suffix_graph.graph.input[0].name != CUT:
        raise ValueError("suffix cut input order changed")
    suffix_graph.graph.input[0].name = "projected"
    for node in suffix_graph.graph.node:
        for index, name in enumerate(node.input):
            if name == CUT:
                node.input[index] = "projected"
    onnx.save(suffix_graph, str(args.suffix))
    onnx.checker.check_model(str(args.prefix))
    onnx.checker.check_model(str(args.suffix))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    source_cpu = ort.InferenceSession(str(args.source), sess_options=options,
                                      providers=["CPUExecutionProvider"])
    prefix_cpu = ort.InferenceSession(str(args.prefix), sess_options=options,
                                      providers=["CPUExecutionProvider"])
    suffix_cpu = ort.InferenceSession(str(args.suffix), sess_options=options,
                                      providers=["CPUExecutionProvider"])
    if ([item.name for item in suffix_cpu.get_inputs()] !=
            ["projected", "positions", *STATE_NAMES]):
        raise ValueError("CPU suffix input contract changed")
    rows = []
    for fixture_path, split_path in zip(args.fixture, args.split_fixture):
        with np.load(fixture_path, allow_pickle=False) as archive:
            fixture = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        source_state = {name: fixture[name] for name in STATE_NAMES}
        split_state = {name: fixture[name] for name in STATE_NAMES}
        step_rows = []
        for index in range(11):
            start = 95 + 2 * index
            conv_name = "conv" if index == 0 else "conv_next" if index == 1 else f"conv_step{index}"
            projected_name = "projected" if index == 0 else (
                "projected_next" if index == 1 else f"projected_step{index}"
            )
            projected = prefix_cpu.run(None, {"conv": fixture[conv_name]})[0]
            fixture[projected_name] = projected
            positions = np.array([[start, start + 1]], np.int64)
            direct = source_cpu.run(None, {"conv": fixture[conv_name],
                                           "positions": positions, **source_state})
            composed = suffix_cpu.run(None, {"projected": projected,
                                             "positions": positions, **split_state})
            errors = [relative_l2(a, b) for a, b in zip(direct, composed)]
            if len(errors) != 17 or max(errors) > 1e-6 or not all(
                    np.isfinite(value).all() for value in composed):
                raise ValueError(f"CPU full-state split changed outputs at frame {start}: {errors}")
            step_rows.append({"start_frame": start,
                              "hidden_relative_l2": errors[0],
                              "max_state_relative_l2": max(errors[1:])})
            source_state = dict(zip(STATE_NAMES, direct[1:]))
            split_state = dict(zip(STATE_NAMES, composed[1:]))
        np.savez_compressed(split_path, **fixture)
        rows.append({"source_fixture_sha256": sha256(fixture_path),
                     "split_fixture_sha256": sha256(split_path),
                     "steps": step_rows})
    report = {
        "scope": "source-faithful CPU input-projection plus full eight-layer rolling-state suffix on two generated utterances; no NPU or waveform result",
        "source_sha256": SOURCE_SHA256,
        "prefix_sha256": sha256(args.prefix),
        "suffix_sha256": sha256(args.suffix),
        "onnxruntime": ort.__version__,
        "rows": rows,
        "status": "two_utterance_eleven_step_cpu_split_parity_pass",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "suffix_sha256": report["suffix_sha256"],
                      "max_relative_l2": max(max(item["hidden_relative_l2"],
                                                 item["max_state_relative_l2"])
                                             for row in rows for item in row["steps"])}))


if __name__ == "__main__":
    main()
