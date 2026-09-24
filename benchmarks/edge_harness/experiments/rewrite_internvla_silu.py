#!/usr/bin/env python3
"""Replace the first Cosmos Sigmoid+Mul activation with equivalent ONNX ops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    if sha256(args.source) != args.source_sha256:
        raise ValueError("source prefix hash changed")
    model = onnx.load(str(args.source), load_external_data=True)
    nodes = list(model.graph.node)
    matches = [i for i, node in enumerate(nodes)
               if node.name == "node_mul_2" and node.op_type == "Mul"
               and list(node.input) == ["group_norm", "sigmoid"]]
    if len(matches) != 1:
        raise ValueError("expected exactly one pinned first SiLU pair")
    index = matches[0]
    sigmoid = next((node for node in nodes if node.name == "node_sigmoid"), None)
    if sigmoid is None or list(sigmoid.input) != ["group_norm"] or list(sigmoid.output) != ["sigmoid"]:
        raise ValueError("first Sigmoid contract changed")
    if any("sigmoid" in node.input for node in nodes if node.name != "node_mul_2"):
        raise ValueError("first Sigmoid has additional consumers")

    # x * sigmoid(x) == x / (1 + exp(-x)). Keep the public tensor name.
    replacement = [
        helper.make_node("Neg", ["group_norm"], ["silu_first_neg"], name="silu_first_neg"),
        helper.make_node("Exp", ["silu_first_neg"], ["silu_first_exp"], name="silu_first_exp"),
        helper.make_node("Add", ["silu_first_exp", "silu_first_one"], ["silu_first_denominator"], name="silu_first_add"),
        helper.make_node("Div", ["group_norm", "silu_first_denominator"], ["mul_2"], name="silu_first_div"),
    ]
    del nodes[index]
    nodes[index:index] = replacement
    nodes = [node for node in nodes if node.name != "node_sigmoid"]
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    model.graph.initializer.append(numpy_helper.from_array(np.array(1.0, dtype=np.float32), "silu_first_one"))
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    if sha256(args.fixtures) != "dcbaf1391ca914d4bbccc1cb384daf05666e771d9cdae90c4f069ec8fcf05368":
        raise ValueError("Cosmos fixtures changed")
    original_cpu = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    rewritten_cpu = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    parity = {}
    with np.load(args.fixtures, allow_pickle=False) as fixtures:
        for name in ("ramp", "pattern"):
            pixels = np.ascontiguousarray(fixtures[f"{name}_pixels"])
            reference = original_cpu.run(None, {"pixels": pixels})[0]
            actual = rewritten_cpu.run(None, {"pixels": pixels})[0]
            difference = actual.astype(np.float64) - reference.astype(np.float64)
            parity[name] = {
                "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(reference.astype(np.float64)), 1e-12)),
                "max_abs": float(np.max(np.abs(difference))),
            }
    report = {
        "scope": "first Cosmos activation rewrite; component compiler candidate only",
        "source_sha256": args.source_sha256,
        "output_sha256": sha256(args.output),
        "candidate_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "replacement": "group_norm/(1+exp(-group_norm))",
        "source_node_count": len(onnx.load(str(args.source), load_external_data=False).graph.node),
        "candidate_node_count": len(model.graph.node),
        "fixture_sha256": sha256(args.fixtures),
        "cpu_parity_vs_source": parity,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
