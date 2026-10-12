#!/usr/bin/env python3
"""Real AI Hub SDK transport smoke from an already audited component job.

This intentionally submits one component inference, not a complete model
request. It verifies the new replay journal and explicit profile=False option
without promoting a device/model matrix cell. Credentials stay in SDK config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qai_hub as hub

from benchmarks.edge_harness.aihub_replay import HubArtifact, HubReplay, audit_run


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job-id", default="jp0mvkj6g")
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    source = hub.get_job(args.source_job_id)
    if source.get_status().code != "SUCCESS":
        raise RuntimeError("source component job did not succeed")
    if str(hub.__version__) != "0.56.0":
        raise RuntimeError("this smoke was checked against qai-hub 0.56.0")
    sample_data = source.inputs.download()
    if any(len(samples) != 1 for samples in sample_data.values()):
        raise ValueError("source dataset is not batch size one")
    inputs = {name: np.asarray(samples[0]) for name, samples in sample_data.items()}
    with tempfile.TemporaryDirectory(prefix="omni-aihub-artifact-") as temporary:
        artifact_file = Path(source.model.download(str(Path(temporary) / "compiled.bin")))
        artifact_sha256 = digest(artifact_file)
    artifact = HubArtifact(
        artifact_id=source.model.model_id,
        target_model_id=source.model.model_id,
        sha256=artifact_sha256,
        checkpoint=args.checkpoint,
        revision=args.revision,
        layout_version=1,
        precision="w8a16",
        target_name=source.device.name,
        runtime=str(source.model.model_type),
        shape_bucket="source-job-" + args.source_job_id,
    )
    request_id = "sdk-smoke-" + args.source_job_id + "-" + args.evidence_dir.name
    existing_journal = args.evidence_dir / "run.json"
    if existing_journal.is_file():
        # Earlier component-smoke journals used the source job alone. Keep
        # their identity stable when re-reading the same evidence directory.
        request_id = json.loads(existing_journal.read_text(encoding="utf-8"))["identity"]["request_id"]
    with HubReplay(
        args.evidence_dir, target_name=source.device.name,
        model="Spark-X2.5-attention-component-only",
        checkpoint=args.checkpoint, revision=args.revision,
        request_id=request_id,
        stage_topology={"attention_component": []},
    ) as replay:
        result = replay.execute(
            "attention-0", stage_id=0, stage_name="attention_component",
            artifact=artifact, inputs=inputs,
            input_provenance={name: "request:audited-" + args.source_job_id
                              for name in inputs},
            options=source.options,
        )
    reference = source.download_output_data()
    submitted = hub.get_job(result.job_id)
    observed_profile = submitted.download_profile()
    comparison = {}
    for name, output in result.outputs.items():
        baseline = np.asarray(reference[name][0])
        current = np.asarray(output)
        comparison[name] = {
            "shape_equal": list(baseline.shape) == list(current.shape),
            "max_abs": float(np.max(np.abs(baseline.astype(np.float64)
                                            - current.astype(np.float64)))),
        }
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one Spark attention component; not model E2E",
        "source_job_id": args.source_job_id,
        "new_job_id": result.job_id,
        "target_registration": source.device.name,
        "target_model_id": source.model.model_id,
        "compiled_artifact_sha256": artifact_sha256,
        "qai_hub_version": str(hub.__version__),
        "profile_requested": False,
        "profile_enabled_on_service": submitted.profiling,
        "profile_detail_rows": len(observed_profile.get("execution_detail", [])),
        "profile_iteration_samples": len(
            observed_profile.get("execution_summary", {}).get("all_inference_times", [])
        ),
        "comparison": comparison,
        "journal_audit": audit_run(args.evidence_dir),
    }
    report_path = args.evidence_dir / "sdk_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "new_job_id": result.job_id,
                      "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
