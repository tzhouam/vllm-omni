#!/usr/bin/env python3
"""Decompose the first real-weight Cosmos GroupNormalization for VitisAI isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIXTURES_SHA = "dcbaf1391ca914d4bbccc1cb384daf05666e771d9cdae90c4f069ec8fcf05368"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    if sha256(args.source) != args.source_sha256 or sha256(args.fixtures) != FIXTURES_SHA:
        raise ValueError("source or fixtures changed")
    model = onnx.load(str(args.source), load_external_data=True)
    nodes = list(model.graph.node)
    index = next((i for i, node in enumerate(nodes) if node.name == "node_group_norm"), None)
    if index is None:
        raise ValueError("first GroupNormalization not present")
    original = nodes[index]
    if original.op_type != "GroupNormalization" or list(original.input) != [
        "conv2d_12", "inner.encoder.down.0.block.0.norm1.weight",
        "inner.encoder.down.0.block.0.norm1.bias",
    ] or list(original.output) != ["group_norm"]:
        raise ValueError("first GroupNormalization contract changed")
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in original.attribute}
    if attrs.get("num_groups") != 32 or abs(attrs.get("epsilon", 0) - 1e-6) > 1e-11:
        raise ValueError("first GroupNormalization settings changed")
    p = "gn_first_"
    replacement = [
        helper.make_node("Reshape", ["conv2d_12", p + "group_shape"], [p + "grouped"], name=p + "reshape_group"),
        helper.make_node("ReduceMean", [p + "grouped", p + "axes"], [p + "mean"], keepdims=1, name=p + "mean"),
        helper.make_node("Sub", [p + "grouped", p + "mean"], [p + "centered"], name=p + "center"),
        helper.make_node("Mul", [p + "centered", p + "centered"], [p + "squared"], name=p + "square"),
        helper.make_node("ReduceMean", [p + "squared", p + "axes"], [p + "variance"], keepdims=1, name=p + "variance"),
        helper.make_node("Add", [p + "variance", p + "epsilon"], [p + "variance_eps"], name=p + "add_epsilon"),
        helper.make_node("Sqrt", [p + "variance_eps"], [p + "std"], name=p + "sqrt"),
        helper.make_node("Div", [p + "centered", p + "std"], [p + "normalized_grouped"], name=p + "divide"),
        helper.make_node("Reshape", [p + "normalized_grouped", p + "tensor_shape"], [p + "normalized"], name=p + "reshape_tensor"),
        helper.make_node("Reshape", [original.input[1], p + "channel_shape"], [p + "gamma"], name=p + "reshape_gamma"),
        helper.make_node("Reshape", [original.input[2], p + "channel_shape"], [p + "beta"], name=p + "reshape_beta"),
        helper.make_node("Mul", [p + "normalized", p + "gamma"], [p + "scaled"], name=p + "scale"),
        helper.make_node("Add", [p + "scaled", p + "beta"], ["group_norm"], name=p + "shift"),
    ]
    nodes[index:index + 1] = replacement
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    constants = {
        p + "group_shape": np.array([6, 32, 4, 64, 64], dtype=np.int64),
        p + "tensor_shape": np.array([6, 128, 64, 64], dtype=np.int64),
        p + "channel_shape": np.array([1, 128, 1, 1], dtype=np.int64),
        p + "axes": np.array([2, 3, 4], dtype=np.int64),
        p + "epsilon": np.array(1e-6, dtype=np.float32),
    }
    for name, value in constants.items():
        model.graph.initializer.append(numpy_helper.from_array(value, name))
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    baseline = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    parity = {}
    with np.load(args.fixtures, allow_pickle=False) as fixtures:
        for name in ("ramp", "pattern"):
            pixels = np.ascontiguousarray(fixtures[f"{name}_pixels"])
            expected = baseline.run(None, {"pixels": pixels})[0]
            actual = candidate.run(None, {"pixels": pixels})[0]
            difference = actual.astype(np.float64) - expected.astype(np.float64)
            parity[name] = {
                "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12)),
                "max_abs": float(np.max(np.abs(difference))),
            }
    report = {
        "scope": "first GroupNormalization decomposition; compiler component candidate only",
        "source_sha256": args.source_sha256,
        "candidate_sha256": sha256(args.output),
        "candidate_bytes": args.output.stat().st_size,
        "fixtures_sha256": FIXTURES_SHA,
        "cpu_parity_vs_source": parity,
        "replacement": "fixed-shape group reshape, reductions, affine scale/bias",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
