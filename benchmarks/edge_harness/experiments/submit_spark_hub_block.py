#!/usr/bin/env python3
"""Upload and compile one real-weight Spark block on an exact AI Hub target.

The script preserves a submission record before waiting. It is an experiment,
not an E2E result: successful compilation alone does not prove execution.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import qai_hub as hub


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--device", default="Samsung Galaxy S25")
    ap.add_argument("--quantize", default="float16")
    ap.add_argument("--runtime", default="qnn_dlc",
                    choices=("qnn_dlc", "precompiled_qnn_onnx", "onnx", "tflite"))
    ap.add_argument("--truncate-64bit-io", action="store_true",
                    help="QNN compiler maps the bounded token ID input to int32")
    ap.add_argument("--attempt", default="base",
                    help="preserve separate evidence for compile retries")
    ap.add_argument("--existing-model-id", default=None,
                    help="recover an upload whose ID was confirmed separately")
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    model_path = Path(manifest["path"])
    if not model_path.is_file() or model_path.stat().st_size != manifest["size_bytes"]:
        raise ValueError("exported model missing or changed since manifest")
    key = f"block{manifest['block']}"
    if not args.attempt.replace("-", "").replace("_", "").isalnum():
        raise ValueError("attempt needs an alphanumeric tag")
    uploaded_path = args.evidence / f"{key}_upload.json"
    compiled_path = args.evidence / f"{key}_{args.attempt}_compile_submission.json"
    if uploaded_path.exists():
        uploaded = json.loads(uploaded_path.read_text())
        if uploaded["onnx_sha256"] != manifest["sha256"]:
            raise ValueError("existing upload is for another ONNX artifact")
        model_id = uploaded["model_id"]
    else:
        model = (hub.Client().get_model(args.existing_model_id)
                 if args.existing_model_id else
                 hub.upload_model(str(model_path), name=f"spark-real-{key}"))
        if model.name != f"spark-real-{key}":
            raise ValueError("existing uploaded model has another name")
        model_id = model.model_id
        write_json(uploaded_path, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sdk_version": str(hub.__version__),
            "model_id": model_id,
            "onnx_sha256": manifest["sha256"],
            "checkpoint_revision": manifest["checkpoint_revision"],
            "manifest": str(args.manifest),
        })
        print(f"uploaded {key}: {model_id}", flush=True)
    if compiled_path.exists():
        submission = json.loads(compiled_path.read_text())
        if submission["source_model_id"] != model_id or submission["device"] != args.device:
            raise ValueError("existing compile is for another model or target")
        job = hub.get_job(submission["job_id"])
    else:
        options = (f"--target_runtime {args.runtime} "
                   f"--quantize_full_type {args.quantize}" +
                   (" --truncate_64bit_io" if args.truncate_64bit_io else ""))
        job = hub.submit_compile_job(
            model=hub.Client().get_model(model_id), device=hub.Device(args.device),
            name=f"spark real {key} {args.quantize}", options=options,
        )
        write_json(compiled_path, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sdk_version": str(hub.__version__),
            "job_id": job.job_id,
            "job_url": job.url,
            "device": args.device,
            "source_model_id": model_id,
            "options": options,
            "onnx_sha256": manifest["sha256"],
        })
        print(f"compile submitted {key}: {job.job_id}", flush=True)
    if args.wait:
        status = job.wait()
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job.job_id,
            "status": status.code,
            "message": status.message or "",
            "target_model_id": (job.get_target_model().model_id if status.success else None),
        }
        write_json(args.evidence / f"{key}_{args.attempt}_compile_report.json", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
