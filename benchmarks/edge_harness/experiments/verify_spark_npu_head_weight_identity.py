# SPDX-License-Identifier: Apache-2.0
"""Check that the older Spark NPU-head source matches the current BF16 head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors import safe_open


SOURCE_GRAPH_SHA = "f5f0c31eb438e9940ab242da8bf6f0f5dad6ebc9f6ef02d5475c21b744e595bc"
BF16_INDEX_SHA = "cc2b212985f5d0469bf926903e4bd7ee81856687baa0c6c6b55ff200d4cdc63f"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-graph", type=Path, required=True)
    parser.add_argument("--bf16-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.source_graph) != SOURCE_GRAPH_SHA:
        raise ValueError("source output-head ONNX hash changed")
    index = args.bf16_model / "model.safetensors.index.json"
    if digest(index) != BF16_INDEX_SHA:
        raise ValueError("BF16 checkpoint index hash changed")
    weight_map = json.loads(index.read_text())["weight_map"]
    graph = onnx.load(str(args.source_graph))
    initializers = {item.name: item for item in graph.graph.initializer}
    if set(initializers) != {"norm", "onnx::MatMul_24"}:
        raise ValueError("unexpected source-head initializers")
    pairs = (
        ("norm", "model.norm.weight", False),
        ("onnx::MatMul_24", "model.embedding.weight", True),
    )
    rows = []
    for graph_name, checkpoint_name, transpose in pairs:
        source = numpy_helper.to_array(initializers[graph_name])
        shard = args.bf16_model / weight_map[checkpoint_name]
        with safe_open(str(shard), framework="pt", device="cpu") as archive:
            bf16 = archive.get_tensor(checkpoint_name).float().numpy()
        candidate = bf16.T if transpose else bf16
        if source.shape != candidate.shape:
            raise ValueError(f"shape mismatch for {graph_name}")
        equal = bool(np.array_equal(source, candidate))
        rows.append({
            "source_initializer": graph_name,
            "checkpoint_tensor": checkpoint_name,
            "checkpoint_shard": shard.name,
            "shape": list(source.shape),
            "source_dtype": str(source.dtype),
            "checkpoint_dtype": "bfloat16 converted exactly to float32",
            "transpose_checkpoint_tensor": transpose,
            "bitwise_equal_after_conversion": equal,
        })
        if not equal:
            raise ValueError(f"weight mismatch for {graph_name}")
    report = {
        "status": "output_head_weights_equal_only",
        "source_graph_sha256": SOURCE_GRAPH_SHA,
        "source_revision": "448e61eb392c00f2c403185c5b56d5e0665bfaab",
        "bf16_checkpoint_index_sha256": BF16_INDEX_SHA,
        "bf16_revision": "14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8",
        "comparisons": rows,
        "scope_limit": "The output norm and tied projection match; no other model weights or live activations were compared.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "comparisons": len(rows)}))


if __name__ == "__main__":
    main()
