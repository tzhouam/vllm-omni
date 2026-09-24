#!/usr/bin/env python3
"""Preserve raw Workbench component jobs submitted in an evidence directory.

Requires qai-hub authentication through the local client configuration. The
submission JSON files identify existing jobs; this script never resubmits them.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qai_hub as hub


def device_record(device: object) -> dict:
    return {
        "name": device.name,
        "os": device.os,
        "attributes": list(device.attributes),
    }


def collect(evidence_dir: Path, kind: str, client: hub.Client) -> str:
    submission = json.loads((evidence_dir / f"{kind}_submission.json").read_text(encoding="utf-8"))
    report_path = evidence_dir / f"{kind}_report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("job_id") != submission["job_id"]:
            raise RuntimeError(f"{report_path}: job ID differs from its submission")
        return f"{kind}: {submission['job_id']} {existing['status']} (already collected)"
    job = client.get_job(submission["job_id"])
    status = job.get_status()
    if status.code not in {"SUCCESS", "FAILED", "CANCELED"}:
        return f"{kind}: {submission['job_id']} {status.code} (pending)"
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "qai_hub_version": str(hub.__version__),
        "facility": "Qualcomm AI Hub Workbench",
        "job_id": submission["job_id"],
        "url": submission["url"],
        "status": status.code,
        "status_message": status.message or "",
        "options": job.options,
        "hub_version": str(job.hub_version),
        "device": device_record(job.device),
    }
    if kind == "compile":
        report["source_model_id"] = job.model.model_id
        report["target_model_id"] = job.get_target_model().model_id if status.code == "SUCCESS" else None
    elif kind == "inference":
        report["model_id"] = job.model.model_id
        report["input_dataset_id"] = job.inputs.dataset_id
        if status.code == "SUCCESS":
            output_path = evidence_dir / "device_output.npz"
            outputs = job.download_output_data()
            arrays = {f"{name}__{i}": value for name, values in outputs.items() for i, value in enumerate(values)}
            np.savez_compressed(output_path, **arrays)
            report["outputs"] = output_path.name
    elif kind == "profile":
        report["model_id"] = job.model.model_id
        report["shapes"] = job.shapes
        if status.code == "SUCCESS":
            report["profile"] = job.download_profile()
    else:
        raise ValueError(f"unexpected job kind {kind}")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return f"{kind}: {submission['job_id']} {status.code}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dirs", nargs="+", type=Path)
    parser.add_argument("--kinds", nargs="+", choices=("compile", "inference", "profile"), default=("inference", "profile"))
    args = parser.parse_args()
    client = hub.Client()
    for directory in args.evidence_dirs:
        for kind in args.kinds:
            print(f"{directory.name}: {collect(directory, kind, client)}")


if __name__ == "__main__":
    main()
