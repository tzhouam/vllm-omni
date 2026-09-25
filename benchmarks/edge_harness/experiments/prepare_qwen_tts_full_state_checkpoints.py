#!/usr/bin/env python3
"""Expose all eight real Qwen3-TTS decoder-layer hidden outputs for NPU diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fixture", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort

    if (sha256(args.source) != args.expected_source_sha256
            or sha256(args.fixture) != args.expected_fixture_sha256):
        raise ValueError("real-weight source graph or rolling-state fixture changed")
    original = onnx.load(str(args.source))
    if (len(original.graph.input) != 18 or len(original.graph.output) != 17
            or original.graph.output[0].name != "hidden"):
        raise ValueError("eight-layer source graph contract changed")
    inferred = onnx.shape_inference.infer_shapes(original)
    values = {item.name: item for item in inferred.graph.value_info}
    names = []
    for index in range(8):
        name = f"/transformer/layers.{index}/Add_1_output_0"
        item = values.get(name)
        if item is None:
            raise ValueError(f"missing real layer-{index} hidden tensor")
        dims = [dimension.dim_value for dimension in item.type.tensor_type.shape.dim]
        if item.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or dims != [1, 2, 512]:
            raise ValueError(f"layer-{index} hidden tensor contract changed: {dims}")
        original.graph.output.add().CopyFrom(item)
        names.append(name)
    onnx.checker.check_model(original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(original, str(args.output))

    with np.load(args.fixture, allow_pickle=False) as source:
        fixture = {name: np.ascontiguousarray(source[name]) for name in source.files}
    feeds = {item.name: (np.array([[95, 96]], np.int64)
                         if item.name == "positions" else fixture[item.name])
             for item in original.graph.input}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    cpu_source = ort.InferenceSession(str(args.source), sess_options=options,
                                      providers=["CPUExecutionProvider"])
    cpu_diagnostic = ort.InferenceSession(str(args.output), sess_options=options,
                                          providers=["CPUExecutionProvider"])
    reference = cpu_source.run(None, feeds)
    candidate = cpu_diagnostic.run(None, feeds)
    if len(reference) != 17 or len(candidate) != 25:
        raise ValueError("instrumented graph outputs changed")
    original_errors = [relative_l2(a, b) for a, b in zip(reference, candidate[:17])]
    if max(original_errors) > 1e-5 or not all(np.isfinite(value).all()
                                               for value in candidate):
        raise ValueError("instrumented graph changed original CPU outputs")
    report = {
        "scope": "diagnostic eight-layer hidden-output ONNX graph; no NPU yet",
        "source_sha256": args.expected_source_sha256,
        "fixture_sha256": args.expected_fixture_sha256,
        "candidate_sha256": sha256(args.output),
        "source_outputs": 17,
        "candidate_outputs": 25,
        "checkpoint_output_names": names,
        "original_cpu_output_max_relative_l2": max(original_errors),
        "checkpoint_shapes": [list(value.shape) for value in candidate[17:]],
        "onnxruntime": ort.__version__,
        "status": "diagnostic_cpu_output_parity_pass",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "candidate_sha256": report["candidate_sha256"],
                      "original_cpu_output_max_relative_l2": max(original_errors)}))


if __name__ == "__main__":
    main()
