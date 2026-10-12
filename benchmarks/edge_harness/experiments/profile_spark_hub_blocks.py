#!/usr/bin/env python3
"""Collect raw AI Hub placement profiles for selected compiled Spark blocks.

These are separate stage profiles, never summed into whole-request latency.
Run after the dependent functional replay to keep one active S25 request.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import qai_hub as hub


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiles", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--blocks", nargs="+", type=int, default=[0, 7])
    args = ap.parse_args()
    for block in args.blocks:
        attempt = "trunc64" if block == 0 else "base"
        report = json.loads((args.compiles / f"block{block}_{attempt}_compile_report.json").read_text())
        if report["status"] != "SUCCESS":
            raise ValueError(f"block {block} compile is not successful")
        target = hub.get_job(report["job_id"]).get_target_model()
        if target.model_id != report["target_model_id"]:
            raise ValueError(f"block {block} target artifact changed")
        submitted = args.evidence / f"block{block}_profile_submission.json"
        if submitted.exists():
            metadata = json.loads(submitted.read_text())
            if metadata["target_model_id"] != target.model_id:
                raise ValueError("profile submission points to another artifact")
            job = hub.get_job(metadata["job_id"])
        else:
            job = hub.submit_profile_job(
                model=target, device=hub.Device("Samsung Galaxy S25"),
                name=f"spark real block{block} placement",
                options="--compute_unit npu",
            )
            save(submitted, {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "job_id": job.job_id, "url": job.url,
                "target_model_id": target.model_id,
                "device": "Samsung Galaxy S25",
                "options": "--compute_unit npu",
            })
            print(f"block {block}: submitted {job.job_id}", flush=True)
        status = job.wait()
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job.job_id, "status": status.code,
            "status_message": status.message or "",
            "target_model_id": target.model_id,
            "device": {"name": job.device.name, "os": job.device.os,
                       "attributes": list(job.device.attributes)},
            "options": job.options,
            "profile": None, "compute_unit_node_counts": None,
        }
        if status.success:
            profile = job.download_profile()
            record["profile"] = profile
            record["compute_unit_node_counts"] = dict(Counter(
                row.get("compute_unit", "unknown")
                for row in profile.get("execution_detail", [])
            ))
        save(args.evidence / f"block{block}_profile_report.json", record)
        print(f"block {block}: {status.code} {record['compute_unit_node_counts']}", flush=True)


if __name__ == "__main__":
    main()
