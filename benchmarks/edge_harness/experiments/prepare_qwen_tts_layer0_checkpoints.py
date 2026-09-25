#!/usr/bin/env python3
"""Expose real decoder layer-0 checkpoints for CPU/NPU error localization."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


BASE_SHA256 = "b4c28d1393496be9ebc9815e6de32ea2f955dbfe4219206a224426b038923bca"
FIXTURE_SHA256 = "e132f412ee3b97a93b068548c2e52c5530a9c8226fd4d7ae268730d04e6f19dc"
CHECKPOINTS = (
    "/transformer/input_proj/MatMul_output_0",
    "/transformer/layers.0/input_layernorm/Mul_1_output_0",
    "/transformer/layers.0/self_attn/q_proj/MatMul_output_0",
    "/transformer/layers.0/self_attn/k_proj/MatMul_output_0",
    "/transformer/layers.0/self_attn/v_proj/MatMul_output_0",
    "/transformer/layers.0/self_attn/Softmax_output_0",
    "/transformer/layers.0/self_attn/MatMul_1_output_0",
    "/transformer/layers.0/self_attn/o_proj/MatMul_output_0",
    "/transformer/layers.0/Add_output_0",
    "/transformer/layers.0/post_attention_layernorm/Mul_1_output_0",
    "/transformer/layers.0/mlp/down_proj/MatMul_output_0",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "fixture", "output", "report"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if sha256(args.base) != BASE_SHA256 or sha256(args.fixture) != FIXTURE_SHA256:
        raise ValueError("source ONNX layer or fixture changed")

    import onnx
    import onnxruntime as ort

    model = onnx.load(str(args.base))
    existing = tuple(item.name for item in model.graph.output)
    if len(existing) != 3:
        raise ValueError(f"expected hidden/new-key/new-value outputs: {existing}")
    model = onnx.shape_inference.infer_shapes(model)
    available = {item.name: item for item in model.graph.value_info}
    for name in CHECKPOINTS:
        if name not in available:
            raise ValueError(f"shape inference did not resolve checkpoint {name}")
        model.graph.output.append(copy.deepcopy(available[name]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    onnx.checker.check_model(str(args.output))

    with np.load(args.fixture, allow_pickle=False) as archive:
        fixture = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    base = ort.InferenceSession(str(args.base), providers=["CPUExecutionProvider"])
    checkpoint = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    feeds = {item.name: fixture[item.name] for item in base.get_inputs()}
    base_outputs = base.run(None, feeds)
    checkpoint_outputs = checkpoint.run(None, feeds)
    parity = [relative_l2(a, b) for a, b in zip(base_outputs, checkpoint_outputs[:3])]
    if max(parity) > 1e-6:
        raise ValueError(f"diagnostic graph changed original CPU outputs: {parity}")
    report = {
        "scope": "first real Qwen3-TTS rolling-KV transformer layer, diagnostic graph outputs only",
        "base_sha256": BASE_SHA256,
        "fixture_sha256": FIXTURE_SHA256,
        "checkpoint_model_sha256": sha256(args.output),
        "onnxruntime": ort.__version__,
        "outputs": [item.name for item in checkpoint.get_outputs()],
        "output_shapes": [list(value.shape) for value in checkpoint_outputs],
        "base_output_cpu_relative_l2": parity,
        "status": "checkpoint_graph_cpu_parity_pass",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "checkpoint_model_sha256": report["checkpoint_model_sha256"]}))


if __name__ == "__main__":
    main()
