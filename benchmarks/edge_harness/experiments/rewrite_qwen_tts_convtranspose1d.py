#!/usr/bin/env python3
"""Rewrite ONNX ConvTranspose1d as equivalent size-one-height ConvTranspose2d."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--expected-fixture-sha256")
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort

    source_hash = sha256(args.source)
    if source_hash != args.expected_source_sha256:
        raise ValueError("source ONNX hash changed")
    model = onnx.load(args.source)
    weights = {initializer.name: initializer for initializer in model.graph.initializer}
    existing_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    } | set(weights)
    axes_name = "__edge_convtranspose2d_axis2"
    if axes_name in existing_names:
        raise ValueError("rewrite tensor name collision")
    axes = onnx.helper.make_tensor(axes_name, onnx.TensorProto.INT64, [1], [2])
    model.graph.initializer.append(axes)

    rewritten = []
    replacements = []
    for node in model.graph.node:
        if node.op_type != "ConvTranspose":
            replacements.append(node)
            continue
        attributes = {
            attr.name: onnx.helper.get_attribute_value(attr)
            for attr in node.attribute
        }
        kernel_shape = list(attributes.get("kernel_shape", []))
        if len(kernel_shape) != 1:
            replacements.append(node)
            continue
        if len(node.input) < 2 or len(node.output) != 1:
            raise ValueError("unexpected ConvTranspose input/output contract")
        weight = weights.get(node.input[1])
        if weight is None or len(weight.dims) != 3:
            raise ValueError("ConvTranspose1d weight is not an embedded rank-three initializer")
        original_dims = list(weight.dims)
        if original_dims[2] != kernel_shape[0]:
            raise ValueError("weight kernel shape changed")
        input_2d = node.input[0] + "__edge_2d"
        output_2d = node.output[0] + "__edge_2d"
        if input_2d in existing_names or output_2d in existing_names:
            raise ValueError("rewrite tensor name collision")
        existing_names.update((input_2d, output_2d))
        attributes["kernel_shape"] = [1, kernel_shape[0]]
        for name in ("strides", "dilations", "output_padding"):
            if name in attributes:
                attributes[name] = [1 if name != "output_padding" else 0, *attributes[name]]
        if "pads" in attributes:
            before, after = attributes["pads"]
            attributes["pads"] = [0, before, 0, after]
        del weight.dims[:]
        weight.dims.extend([original_dims[0], original_dims[1], 1, original_dims[2]])
        replacements.extend([
            onnx.helper.make_node(
                "Unsqueeze", [node.input[0], axes_name], [input_2d],
                name=node.name + "__edge_unsqueeze"),
            onnx.helper.make_node(
                "ConvTranspose", [input_2d, *node.input[1:]], [output_2d],
                name=node.name + "__edge_2d", **attributes),
            onnx.helper.make_node(
                "Squeeze", [output_2d, axes_name], list(node.output),
                name=node.name + "__edge_squeeze"),
        ])
        rewritten.append({
            "node_name": node.name,
            "weight_name": node.input[1],
            "original_weight_dims": original_dims,
            "new_weight_dims": list(weight.dims),
        })
    if not rewritten:
        raise ValueError("source has no rank-one ConvTranspose to rewrite")
    del model.graph.node[:]
    model.graph.node.extend(replacements)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)

    source_session = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    rewritten_session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    source_input = source_session.get_inputs()[0]
    rewritten_input = rewritten_session.get_inputs()[0]
    if source_input.shape != rewritten_input.shape:
        raise ValueError("rewritten graph changed input shape")
    if args.fixture is None:
        shape = source_input.shape
        if any(not isinstance(size, int) or size <= 0 for size in shape):
            raise ValueError("synthetic control requires static input shape")
        data = np.random.default_rng(42).standard_normal(shape).astype(np.float32)
        input_provenance = "synthetic fixed-seed normal"
        fixture_hash = None
    else:
        if args.expected_fixture_sha256 is None:
            raise ValueError("fixture hash required")
        fixture_hash = sha256(args.fixture)
        if fixture_hash != args.expected_fixture_sha256:
            raise ValueError("fixture hash changed")
        with np.load(args.fixture, allow_pickle=False) as fixture:
            if fixture.files != ["quantized"]:
                raise ValueError("fixture schema changed")
            data = np.ascontiguousarray(fixture["quantized"])
        input_provenance = "pinned real-weight vocoder fixture"
    expected = source_session.run(None, {source_input.name: data})[0]
    actual = rewritten_session.run(None, {rewritten_input.name: data})[0]
    if expected.shape != actual.shape:
        raise ValueError("rewritten graph changed output shape")
    error = actual.astype(np.float64) - expected.astype(np.float64)
    relative_l2 = float(
        np.linalg.norm(error) /
        max(np.linalg.norm(expected.astype(np.float64)), 1e-12))
    report = {
        "source_sha256": source_hash,
        "source_file": str(args.source.resolve()),
        "rewritten_sha256": sha256(args.output),
        "rewritten_file": str(args.output.resolve()),
        "fixture_sha256": fixture_hash,
        "input_provenance": input_provenance,
        "rewritten_nodes": rewritten,
        "input_shape": list(data.shape),
        "output_shape": list(actual.shape),
        "cpu_finite": bool(np.isfinite(actual).all()),
        "cpu_relative_l2_vs_source": relative_l2,
        "cpu_maximum_absolute_error": float(np.max(np.abs(error))),
        "status": "cpu_rewrite_parity_pass" if relative_l2 <= 1e-5 else "cpu_rewrite_parity_failed",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rewritten_nodes": len(rewritten),
                      "cpu_relative_l2_vs_source": relative_l2}))
    if report["status"] != "cpu_rewrite_parity_pass":
        raise ValueError("rewritten graph changed CPU output")


if __name__ == "__main__":
    main()
