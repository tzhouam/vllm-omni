#!/usr/bin/env python3
"""Audit a fixed-window Code2Wav candidate against its local CPU export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def waveform(path: Path, name: str, expected_shape: tuple[int, int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if data.files != [name]:
            raise ValueError(f"{path}: expected one output named {name}")
        value = np.asarray(data[name])
    if value.shape != expected_shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"{path}: expected finite float32{expected_shape}, got {value.shape}/{value.dtype}")
    return value


def compare(reference: np.ndarray, observed: np.ndarray) -> dict:
    a = reference.astype(np.float64).ravel()
    b = observed.astype(np.float64).ravel()
    delta = float(np.linalg.norm(a - b))
    norm = float(np.linalg.norm(a))
    return {
        "relative_l2": delta / norm,
        "snr_db": 20 * math.log10(norm / delta) if delta else None,
        "max_abs": float(np.max(np.abs(a - b))),
        "reference_rms": float(np.sqrt(np.mean(a * a))),
        "output_rms": float(np.sqrt(np.mean(b * b))),
        "output_saturated_fraction_abs_ge_0_999": float(np.mean(np.abs(b) >= 0.999)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export-report", "upload-report", "dataset-report", "fixture",
                 "cpu-output", "eager-output", "compile-report", "inference-report",
                 "device-output", "output-report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--profile-report", type=Path)
    parser.add_argument("--expected-device", default="Samsung Galaxy S25")
    parser.add_argument("--expected-compile-options", default="--target_runtime qnn_dlc --qnn_options default_graph_htp_precision=FLOAT16")
    parser.add_argument("--expected-run-options", default="--compute_unit npu")
    args = parser.parse_args()
    export = json.loads(args.export_report.read_text())
    upload = json.loads(args.upload_report.read_text())
    dataset = json.loads(args.dataset_report.read_text())
    compile_job = json.loads(args.compile_report.read_text())
    inference = json.loads(args.inference_report.read_text())
    profile = json.loads(args.profile_report.read_text()) if args.profile_report else None
    if (export["context_frames"] != 72 or not 1 <= export["chunk_frames"] <= 25
            or export["input_shape"] != [1, 512, 72 + export["chunk_frames"]]
            or export["output_shape"] != [1, 1920 * export["chunk_frames"]]):
        raise ValueError("export differs from the tested fixed-window contract")
    if (upload["source_sha256"] != export["onnx_sha256"]
            or compile_job["source_model_id"] != upload["source_model_id"]
            or compile_job["status"] != "SUCCESS"
            or compile_job["options"] != args.expected_compile_options):
        raise ValueError("compile does not use the verified short-window artifact")
    if (dataset["fixture_sha256"] != export["short_fixture_sha256"]
            or dataset["input_shape"] != export["input_shape"]
            or dataset["sample_count"] != 1
            or sha256(args.fixture) != dataset["fixture_sha256"]):
        raise ValueError("hosted input is not the retained short-window fixture")
    if (inference["status"] != "SUCCESS"
            or inference["model_id"] != compile_job["target_model_id"]
            or inference["input_dataset_id"] != dataset["dataset_id"]
            or inference["options"] != args.expected_run_options
            or inference["device"] != compile_job["device"]
            or inference["device"]["name"] != args.expected_device):
        raise ValueError("inference differs from the exact target/fixture/route")
    if (sha256(args.cpu_output) != export["ort_output_sha256"]
            or sha256(args.eager_output) != export["eager_output_sha256"]):
        raise ValueError("local CPU outputs differ from export parity evidence")
    expected_shape = tuple(export["output_shape"])
    cpu = waveform(args.cpu_output, "wav", expected_shape)
    eager = waveform(args.eager_output, "wav", expected_shape)
    hosted = waveform(args.device_output, "output_0__0", expected_shape)
    cpu_comparison = compare(cpu, hosted)
    gross_numeric_failure = (cpu_comparison["relative_l2"] >= 1.0
                             or cpu_comparison["output_saturated_fraction_abs_ge_0_999"] >= 0.1)
    report = {
        "scope": f"one {export['chunk_frames']}-frame Qwen3-TTS vocoder component on hosted {args.expected_device}; no complete stream",
        "status": "component_gross_numeric_failure" if gross_numeric_failure else "component_inference_numeric_measured",
        "device": inference["device"],
        "source_model_id": compile_job["source_model_id"],
        "target_model_id": compile_job["target_model_id"],
        "input_dataset_id": dataset["dataset_id"],
        "jobs": {"compile": compile_job["job_id"], "inference": inference["job_id"]},
        "route": {"compile_options": compile_job["options"], "run_options": inference["options"]},
        "files": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                  for name, path in {"fixture": args.fixture, "ort_cpu": args.cpu_output,
                                     "eager_cpu": args.eager_output,
                                     "device_output": args.device_output}.items()},
        "ort_cpu_vs_device": cpu_comparison,
        "eager_cpu_vs_device": compare(eager, hosted),
        "gross_numeric_failure_screen": {
            "failed": gross_numeric_failure,
            "criteria": "relative L2 >= 1.0 or fraction |sample| >= 0.999 at least 0.1; this is not a speech-quality acceptance threshold",
        },
        "limits": [
            "One fixed-window waveform has no listening-quality tolerance.",
            "The 72-frame input history, talker, predictor and stage handoff did not run on the device.",
            "No persistent state, admission, cancellation or complete TTS stream was tested.",
        ],
    }
    if profile is not None:
        if (profile["status"] != "SUCCESS"
                or profile["model_id"] != inference["model_id"]
                or profile["device"] != inference["device"]
                or profile["options"] != inference["options"]
                or profile["shapes"] != {"quantized": [export["input_shape"], "float32"]}):
            raise ValueError("placement profile did not use the audited device/artifact")
        execution = profile["profile"]
        samples = execution["execution_summary"]["all_inference_times"]
        if not samples or any(not isinstance(t, int) or t <= 0 for t in samples):
            raise ValueError("placement profile has no valid timing samples")
        units = Counter(row.get("compute_unit") for row in execution["execution_detail"])
        if not units:
            raise ValueError("placement profile has no node placement detail")
        ordered = sorted(samples)
        report["jobs"]["profile"] = profile["job_id"]
        report["profile"] = {
            "sample_count": len(samples),
            "min_us": ordered[0],
            "nearest_rank_p50_us": ordered[math.ceil(0.5 * len(samples)) - 1],
            "nearest_rank_p95_us": ordered[math.ceil(0.95 * len(samples)) - 1],
            "reported_peak_memory_bytes": execution["execution_summary"]["estimated_inference_peak_memory"],
            "compute_unit_row_counts": dict(units),
            "profile_input": "Workbench generated input with audited tensor shape; the waveform inference used the retained fixture",
        }
        report["status"] = ("component_gross_numeric_failure_with_placement_profile"
                            if gross_numeric_failure else "component_inference_numeric_and_placement_measured")
    else:
        report["limits"].append("Requested accelerator execution is not verified node placement without a profile.")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "comparison": report["ort_cpu_vs_device"],
                      "profile": report.get("profile")}, indent=2))


if __name__ == "__main__":
    main()
