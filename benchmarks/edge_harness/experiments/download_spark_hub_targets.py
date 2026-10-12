#!/usr/bin/env python3
"""Download compiled Spark DLCs and bind each target artifact to its bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import qai_hub as hub


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for block in range(args.first, args.last + 1):
        attempt = "trunc64" if block == 0 else "base"
        report = json.loads((args.evidence / f"block{block}_{attempt}_compile_report.json").read_text())
        submission = json.loads((args.evidence / f"block{block}_{attempt}_compile_submission.json").read_text())
        if report["status"] != "SUCCESS" or report["job_id"] != submission["job_id"]:
            raise ValueError(f"block {block} has no successful matching compile")
        job = hub.get_job(report["job_id"])
        if job.get_status().code != "SUCCESS":
            raise ValueError(f"block {block} remote compile changed status")
        target = job.get_target_model()
        if target.model_id != report["target_model_id"]:
            raise ValueError(f"block {block} target model identity changed")
        path = args.out / f"spark_block{block}_s25_qnn.dlc"
        if not path.exists():
            downloaded = job.download_target_model(str(path))
            if downloaded is not None:
                path = Path(downloaded)
        if not path.is_file():
            raise FileNotFoundError(f"block {block} compiled artifact did not download")
        record = {
            "block": block, "compile_job_id": report["job_id"],
            "target_model_id": target.model_id,
            "source_onnx_sha256": submission["onnx_sha256"],
            "compile_options": submission["options"],
            "device": submission["device"],
            "sha256": sha256(path), "size_bytes": path.stat().st_size,
            "artifact_path": str(path),
        }
        (args.evidence / f"block{block}_target_hash.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(f"block {block}: {record['target_model_id']} {record['sha256']} "
              f"{record['size_bytes']} bytes", flush=True)


if __name__ == "__main__":
    main()
