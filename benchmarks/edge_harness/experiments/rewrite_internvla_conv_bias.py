#!/usr/bin/env python3
"""Move the real Cosmos Conv13 bias outside the candidate quantized Conv."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_SHA = "ff2ebad2c59a472b561536fa49cdd8bc43cf8e7bcf000a8ff4420f8e3a4766c8"
FIXTURE_SHA = "7bca8b4dcafa0ab0d45387e95a4d3e844f04f296f9253a4d4ac54ab5682d068c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "boundary-fixtures", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    if sha256(args.source) != SOURCE_SHA or sha256(args.boundary_fixtures) != FIXTURE_SHA:
        raise ValueError("real Cosmos boundary or retained activations changed")
    model = onnx.load(args.source)
    if [node.op_type for node in model.graph.node] != ["Sigmoid", "Mul", "Conv"]:
        raise ValueError("expected Sigmoid, Mul and Conv boundary")
    conv = model.graph.node[-1]
    if conv.name != "node_conv2d_13" or list(conv.output) != ["conv2d_13"] or len(conv.input) != 3:
        raise ValueError("real convolution contract changed")
    original_bias_name = conv.input[2]
    original_bias = next((item for item in model.graph.initializer if item.name == original_bias_name), None)
    if original_bias is None:
        raise ValueError("real convolution bias missing")
    bias = numpy_helper.to_array(original_bias)
    if bias.shape != (256,) or bias.dtype != np.float32:
        raise ValueError("real convolution bias shape or dtype changed")
    conv.input.pop()
    conv.output[0] = "conv2d_13_without_bias"
    model.graph.node.extend([helper.make_node(
        "Add", ["conv2d_13_without_bias", "conv13_bias_nchw"],
        ["conv2d_13"], name="node_conv13_fp32_bias")])
    bias_nchw = numpy_helper.from_array(bias.reshape(1, 256, 1, 1), name="conv13_bias_nchw")
    weights = [item for item in model.graph.initializer if item.name != original_bias_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(weights)
    model.graph.initializer.extend([bias_nchw])
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)

    source_cpu = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    rewritten_cpu = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    parity = {}
    with np.load(args.boundary_fixtures, allow_pickle=False) as data:
        for case in ("ramp", "pattern"):
            activation = np.ascontiguousarray(data[f"{case}_group_norm"])
            expected = source_cpu.run(None, {"group_norm": activation})[0]
            actual = rewritten_cpu.run(None, {"group_norm": activation})[0]
            difference = actual.astype(np.float64) - expected.astype(np.float64)
            parity[case] = {
                "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12)),
                "max_absolute_error": float(np.max(np.abs(difference))),
                "finite": bool(np.isfinite(actual).all()),
            }
    report = {
        "scope": "real Cosmos Sigmoid+Mul+Conv13 with unchanged FP32 bias moved to a following Add; component only",
        "source_sha256": SOURCE_SHA,
        "boundary_fixtures_sha256": FIXTURE_SHA,
        "candidate_sha256": sha256(args.output),
        "candidate_bytes": args.output.stat().st_size,
        "rewritten_ops": [node.op_type for node in model.graph.node],
        "cpu_parity_vs_source": parity,
        "onnxruntime": ort.__version__,
        "status": "cpu_rewrite_pass" if all(item["finite"] and item["relative_l2"] <= 1e-5 for item in parity.values()) else "cpu_rewrite_fail",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "cpu_rewrite_pass":
        raise ValueError("FP32 bias rewrite changed the Cosmos boundary")


if __name__ == "__main__":
    main()
