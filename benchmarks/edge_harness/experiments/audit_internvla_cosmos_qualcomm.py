#!/usr/bin/env python3
"""Audit one hosted InternVLA Cosmos encoder output against a pinned source fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


MODEL_SHA256 = "656a5263f584851a1457ef6dcf60b995f7c67b056b3c90a4f33c814f7caf9ec9"
FIXTURE_SHA256 = "dcbaf1391ca914d4bbccc1cb384daf05666e771d9cdae90c4f069ec8fcf05368"
COMPILE_OPTIONS = "--target_runtime qnn_dlc --qnn_options default_graph_htp_precision=FLOAT16"


def file_record(path: Path) -> dict:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-onnx", "fixture", "compile-report", "inference-submission",
        "inference-report", "device-output", "output-report",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device-name", default="Samsung Galaxy S25")
    parser.add_argument("--profile-report", type=Path)
    parser.add_argument("--expected-compile-options", default=COMPILE_OPTIONS)
    parser.add_argument("--expected-run-options", default="--compute_unit npu")
    args = parser.parse_args()

    compiled = json.loads(args.compile_report.read_text(encoding="utf-8"))
    submitted = json.loads(args.inference_submission.read_text(encoding="utf-8"))
    inferred = json.loads(args.inference_report.read_text(encoding="utf-8"))
    if compiled.get("status") != "SUCCESS" or inferred.get("status") != "SUCCESS":
        raise ValueError("compile and inference must be terminal successes")
    if (compiled.get("source_model_id") != "mn4okj3rq"
            or compiled.get("options") != args.expected_compile_options):
        raise ValueError("compile used a different source or options")
    if inferred.get("model_id") != compiled.get("target_model_id"):
        raise ValueError("inference used a different target model")
    if inferred.get("device") != compiled.get("device"):
        raise ValueError("compile/inference device metadata differs")
    if inferred.get("device", {}).get("name") != args.device_name:
        raise ValueError("inference did not use the requested exact device")
    if inferred.get("options") != args.expected_run_options:
        raise ValueError("inference did not request the expected compute unit")
    if (inferred.get("job_id") != submitted.get("job_id")
            or inferred.get("input_dataset_id") != submitted.get("input_dataset_id")):
        raise ValueError("inference used a different job or input dataset")
    if inferred.get("input_dataset_id") != "d26qo8p57":
        raise ValueError("inference did not use the pinned pattern dataset")
    if submitted.get("fixture_key", "pattern_pixels") != "pattern_pixels":
        raise ValueError("submission did not use the pattern fixture")

    files = {
        name: file_record(path)
        for name, path in {
            "source_onnx": args.source_onnx,
            "fixture": args.fixture,
            "device_output": args.device_output,
        }.items()
    }
    if files["source_onnx"]["sha256"] != MODEL_SHA256 or files["fixture"]["sha256"] != FIXTURE_SHA256:
        raise ValueError("source model or fixture hash differs")
    with np.load(args.fixture, allow_pickle=False) as fixture:
        if fixture.files != ["ramp_pixels", "ramp_reference", "pattern_pixels", "pattern_reference"]:
            raise ValueError("fixture layout differs")
        pixels = np.asarray(fixture["pattern_pixels"])
        reference = np.asarray(fixture["pattern_reference"])
        if pixels.shape != (6, 3, 256, 256) or pixels.dtype != np.float32 or not np.isfinite(pixels).all():
            raise ValueError("input pixels violate fixed source contract")
        if reference.shape != (6, 16, 32, 32) or reference.dtype != np.float32 or not np.isfinite(reference).all():
            raise ValueError("source reference violates fixed latent contract")
    with np.load(args.device_output, allow_pickle=False) as output:
        if output.files != ["output_0__0"]:
            raise ValueError("unexpected hosted output layout")
        actual = np.asarray(output["output_0__0"])
        if (actual.shape != reference.shape or actual.dtype not in (np.float16, np.float32)
                or not np.isfinite(actual).all()):
            raise ValueError("hosted output violates finite latent contract")

    source_values = reference.astype(np.float64).ravel()
    device_values = actual.astype(np.float64).ravel()
    error = source_values - device_values
    source_norm = float(np.linalg.norm(source_values))
    device_norm = float(np.linalg.norm(device_values))
    error_norm = float(np.linalg.norm(error))
    if source_norm == 0 or device_norm == 0:
        raise ValueError("zero latent norm cannot define numerical similarity")
    report = {
        "scope": "one six-frame Cosmos image encoder on synthetic pattern; no complete InternVLA policy",
        "status": "component_inference_numeric_measured",
        "device": inferred["device"],
        "source_model_id": compiled["source_model_id"],
        "target_model_id": compiled["target_model_id"],
        "compile_job_id": compiled["job_id"],
        "inference_job_id": inferred["job_id"],
        "input_dataset_id": inferred["input_dataset_id"],
        "requested_run_options": inferred["options"],
        "files": files,
        "output_dtype": str(actual.dtype),
        "source_vs_device": {
            "relative_l2": error_norm / source_norm,
            "snr_db": 20 * math.log10(source_norm / error_norm) if error_norm else None,
            "max_abs_error": float(np.max(np.abs(error))),
            "cosine": float(np.dot(source_values, device_values) / (source_norm * device_norm)),
        },
        "limits": [
            "The fixture is synthetic; no real-observation or action-quality tolerance is established.",
            "The action policy, state lifecycle, admission and complete request were not run on this device.",
        ],
    }
    if args.profile_report is not None:
        profiled = json.loads(args.profile_report.read_text(encoding="utf-8"))
        if (profiled.get("status") != "SUCCESS"
                or profiled.get("model_id") != compiled["target_model_id"]
                or profiled.get("device") != inferred["device"]
                or profiled.get("options") != args.expected_run_options):
            raise ValueError("profile did not execute the same target on the exact device")
        profile = profiled["profile"]
        summary = profile["execution_summary"]
        samples = summary["all_inference_times"]
        if not samples or any(not isinstance(value, int) or value <= 0 for value in samples):
            raise ValueError("profile has no valid inference samples")
        units = Counter(row.get("compute_unit") for row in profile["execution_detail"])
        if not units:
            raise ValueError("profile has no compute-unit detail")
        ordered = sorted(samples)
        report["profile"] = {
            "job_id": profiled["job_id"],
            "sample_count": len(samples),
            "sample_min_us": ordered[0],
            "nearest_rank_p50_us": ordered[math.ceil(0.50 * len(samples)) - 1],
            "nearest_rank_p95_us": ordered[math.ceil(0.95 * len(samples)) - 1],
            "reported_estimated_inference_time_us": summary["estimated_inference_time"],
            "reported_peak_memory_bytes": summary["estimated_inference_peak_memory"],
            "reported_first_load_time_us": summary["first_load_time"],
            "reported_warm_load_time_us": summary["warm_load_time"],
            "reported_first_load_peak_memory_bytes": summary["first_load_peak_memory"],
            "reported_warm_load_peak_memory_bytes": summary["warm_load_peak_memory"],
            "compute_unit_row_counts": dict(units),
        }
        report["status"] = "component_inference_numeric_and_placement_measured"
    else:
        report["limits"].append(
            "Requested NPU is not actual per-node placement without a device profile."
        )
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "difference": report["source_vs_device"]}, indent=2))


if __name__ == "__main__":
    main()
