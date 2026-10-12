#!/usr/bin/env python3
"""Compare downloaded Spark block outputs with pinned HF BF16 boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    record = json.loads(args.run.read_text())
    if record["identity"]["model"] != "Spark-X2.5":
        raise ValueError("expected a Spark replay")
    with np.load(args.reference, allow_pickle=False) as archive:
        references = {name: archive[name] for name in archive.files}
    rows = []
    for step in record["steps"]:
        if step["status"] != "completed":
            continue
        index = int(step["stage_name"].removeprefix("block"))
        path = args.run.parent / step["output_npz"]
        if sha256(path) != step["output_file_sha256"]:
            raise ValueError(f"target output file changed for block {index}")
        with np.load(path, allow_pickle=False) as archive:
            got = archive["output_0"]
        expected = references["logits" if index == 7 else f"block{index}"]
        if got.shape != expected.shape or not np.isfinite(got).all():
            raise ValueError(f"block {index} output shape/finite gate failed")
        error = got.astype(np.float64) - expected.astype(np.float64)
        row = {
            "block": index, "job_id": step["job_id"],
            "target_model_id": step["artifact"]["target_model_id"],
            "relative_l2_vs_hf_bf16": float(np.linalg.norm(error) /
                                              max(float(np.linalg.norm(expected.astype(np.float64))), 1e-12)),
            "max_abs_vs_hf_bf16": float(np.max(np.abs(error))),
        }
        if index == 7:
            row["reference_top1"] = int(np.argmax(expected))
            row["target_top1"] = int(np.argmax(got))
            row["top1_match"] = row["reference_top1"] == row["target_top1"]
        rows.append(row)
    result = {
        "depth": "C; first prompt token, 28-layer dependent hosted decode only",
        "not_e2e": True,
        "reference_sha256": sha256(args.reference),
        "completed_blocks": len(rows),
        "blocks": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
