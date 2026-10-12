#!/usr/bin/env python3
"""Export contiguous real-weight Spark blocks for hosted Qualcomm replay.

The first block includes the token embedding and the last includes the tied
output head. Every neural operation in a decode step can therefore be run on
the named target device, with only sampling and explicit KV routing on host.
The default 2 + 6*4 + 2 partition keeps the two endpoint ONNX files below the
protobuf size limit despite their large embedding/head matrices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from vllm_omni.edge.spark_export import (
    SparkDecodeStep,
    SparkTokenDecodeBlock,
    config_from_spark,
    export_onnx,
    load_spark_weights,
)


BLOCKS = ((0, 2), (2, 6), (6, 10), (10, 14), (14, 18),
          (18, 22), (22, 26), (26, 28))


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--block", type=int, choices=range(len(BLOCKS)),
                    action="append", help="export only this zero-based block; repeatable")
    args = ap.parse_args()
    if args.context < 2:
        raise ValueError("context must hold at least one prior token")
    model_config = args.model / "config.json"
    model_index = args.model / "model.safetensors.index.json"
    if not model_config.is_file() or not model_index.is_file():
        raise FileNotFoundError("complete Spark checkpoint config and index required")
    hf = json.loads(model_config.read_text())
    if len(hf["layer_types"]) != 28:
        raise ValueError("block topology requires the pinned 28-layer Spark checkpoint")
    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=True)
    selected = sorted(set(args.block)) if args.block is not None else range(len(BLOCKS))
    for index in selected:
        start, stop = BLOCKS[index]
        cfg = config_from_spark(
            hf, slice(start, stop), include_lm_head=(stop == 28),
        )
        module = (SparkTokenDecodeBlock(cfg) if start == 0
                  else SparkDecodeStep(cfg))
        copied = load_spark_weights(module, args.model)
        path = args.out / f"spark_block{index}_{start}_{stop}_ctx{args.context}.onnx"
        record = export_onnx(module.eval(), args.context, path)
        record.update({
            "block": index,
            "first_layer": start,
            "last_layer_exclusive": stop,
            "on_target_embedding": start == 0,
            "on_target_lm_head": stop == 28,
            "weights_copied": copied,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "checkpoint": str(args.model),
            "checkpoint_revision": args.revision,
            "checkpoint_config_sha256": sha256(model_config),
            "checkpoint_index_sha256": sha256(model_index),
            "precision": "fp32_export; target quantization must be recorded separately",
            "batch_size": 1,
        })
        manifest = path.with_suffix(".json")
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"block": index, "path": str(path),
                          "size_bytes": record["size_bytes"],
                          "sha256": record["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
