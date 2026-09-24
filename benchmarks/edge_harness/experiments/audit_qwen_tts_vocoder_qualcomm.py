#!/usr/bin/env python3
"""Audit a hosted Qwen3-TTS vocoder component against a pinned ONNX CPU run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


SOURCE_SHA256 = "7c07666229c6d404894132e22f784525e3212a49efad55ddd532eda5286c2b2b"
FIXTURE_SHA256 = "dfca9e8d2a724a568a31b6cd18ff6a52fdc205fd34e13cbf1cd75f0f7937e82d"
CPU_OUTPUT_SHA256 = "8de78b94b59f01e49852f7706adbd2508a9976b8e2705cfd3b94d07bb3606cb1"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_output(path: Path, name: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if data.files != [name]:
            raise ValueError(f"{path}: expected only {name}, got {data.files}")
        output = np.asarray(data[name])
    if output.shape != (1, 48000) or output.dtype != np.float32 or not np.isfinite(output).all():
        raise ValueError(f"{path}: violates finite FP32 48,000-sample WAV contract")
    return output


def compare(reference: np.ndarray, output: np.ndarray) -> dict:
    a = reference.astype(np.float64).ravel()
    b = output.astype(np.float64).ravel()
    error_norm = float(np.linalg.norm(a - b))
    reference_norm = float(np.linalg.norm(a))
    output_norm = float(np.linalg.norm(b))
    if reference_norm == 0 or output_norm == 0:
        raise ValueError("silent reference/output cannot define waveform similarity")
    return {
        "relative_l2": error_norm / reference_norm,
        "snr_db": 20 * math.log10(reference_norm / error_norm) if error_norm else None,
        "max_abs": float(np.max(np.abs(a - b))),
        "cosine": float(np.dot(a, b) / (reference_norm * output_norm)),
        "reference_rms": float(np.sqrt(np.mean(a * a))),
        "output_rms": float(np.sqrt(np.mean(b * b))),
        "output_saturated_fraction_abs_ge_0_999": float(np.mean(np.abs(b) >= 0.999)),
    }


def nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--compile-options", required=True)
    parser.add_argument("--run-options", required=True)
    for name in (
        "source-onnx", "fixture", "cpu-output", "s25-output", "s25-inference-report", "compile-report",
        "inference-report", "device-output", "output-report",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--profile-report", type=Path)
    args = parser.parse_args()
    compile_report = json.loads(args.compile_report.read_text(encoding="utf-8"))
    inference = json.loads(args.inference_report.read_text(encoding="utf-8"))
    profile = json.loads(args.profile_report.read_text(encoding="utf-8")) if args.profile_report else None
    s25_inference = json.loads(args.s25_inference_report.read_text(encoding="utf-8"))
    if any(report.get("status") != "SUCCESS" for report in (compile_report, inference)):
        raise RuntimeError("compile and inference must both succeed")
    if profile is not None and profile.get("status") != "SUCCESS":
        raise RuntimeError("provided profile must succeed")
    if compile_report.get("source_model_id") != "mqyer339n" or compile_report.get("options") != args.compile_options:
        raise RuntimeError("compile did not use pinned Qwen3-TTS vocoder source/options")
    target_id = compile_report["target_model_id"]
    if inference.get("model_id") != target_id or (profile is not None and profile.get("model_id") != target_id):
        raise RuntimeError("inference/profile used a different target artifact")
    if inference.get("input_dataset_id") != "d7m80jl32":
        raise RuntimeError("inference did not use the retained vocoder fixture")
    if (s25_inference.get("status") != "SUCCESS" or s25_inference.get("model_id") != "mng7kwxon"
            or s25_inference.get("input_dataset_id") != inference["input_dataset_id"]
            or s25_inference.get("device", {}).get("name") != "Samsung Galaxy S25"):
        raise RuntimeError("historical S25 output lacks matching model/device/dataset provenance")
    for report in (compile_report, inference) + ((profile,) if profile is not None else ()):
        if report.get("device", {}).get("name") != args.device_name:
            raise RuntimeError("job did not use the exact requested device")
    if inference.get("options") != args.run_options or (profile is not None and profile.get("options") != args.run_options):
        raise RuntimeError("inference/profile requested different compute options")
    if profile is not None and profile.get("shapes") != {"quantized": [[1, 512, 97], "float32"]}:
        raise RuntimeError("profile did not use the pinned 97-frame input contract")
    files = {
        name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in {
            "source_onnx": args.source_onnx,
            "fixture": args.fixture,
            "cpu_output": args.cpu_output,
            "s25_output": args.s25_output,
            "device_output": args.device_output,
        }.items()
    }
    for name, expected in {
        "source_onnx": SOURCE_SHA256,
        "fixture": FIXTURE_SHA256,
        "cpu_output": CPU_OUTPUT_SHA256,
    }.items():
        if files[name]["sha256"] != expected:
            raise RuntimeError(f"{name} differs from the pinned source/fixture/reference")
    with np.load(args.fixture, allow_pickle=False) as data:
        if data.files != ["quantized"] or data["quantized"].shape != (1, 512, 97):
            raise RuntimeError("fixture input contract differs")
    cpu = read_output(args.cpu_output, "wav")
    s25 = read_output(args.s25_output, "output_0__0")
    output = read_output(args.device_output, "output_0__0")
    same_s25_output = args.s25_output.resolve() == args.device_output.resolve()
    report = {
        "scope": "one Qwen3-TTS code2wav 48,000-sample component on one retained fixture; no complete TTS stream",
        "status": "component_executed_quality_unqualified" if profile is not None else "component_inference_numeric_measured",
        "device": inference["device"],
        "compile_job_id": compile_report["job_id"],
        "inference_job_id": inference["job_id"],
        "target_model_id": target_id,
        "input_dataset_id": inference["input_dataset_id"],
        "s25_prior_inference_job_id": s25_inference["job_id"],
        "requested_run_options": args.run_options,
        "files": files,
        "cpu_source_vs_device": compare(cpu, output),
        "s25_gpu_vs_device": None if same_s25_output else compare(s25, output),
        "s25_comparison_note": ("same retained S25 output; no independent device comparison"
                                if same_s25_output else "independent hosted output comparison"),
        "limits": [
            "Historical source export does not attest its exact checkpoint revision.",
            "This fixed component waveform has no listening or task-quality tolerance.",
            "Hosted component jobs exclude talker, predictor, stage handoff, streaming, admission and sustained power/thermal behavior.",
        ],
    }
    if profile is not None:
        execution = profile["profile"]
        times_us = execution["execution_summary"]["all_inference_times"]
        if not times_us or any(not isinstance(value, int) or value <= 0 for value in times_us):
            raise RuntimeError("missing or invalid device samples")
        units = Counter(row.get("compute_unit") for row in execution["execution_detail"])
        if not units:
            raise RuntimeError("no compute-unit placement detail")
        report["profile_job_id"] = profile["job_id"]
        report["profile"] = {
            "sample_count": len(times_us),
            "sample_min_us": min(times_us),
            "nearest_rank_p50_us": nearest_rank(times_us, 0.50),
            "nearest_rank_p95_us": nearest_rank(times_us, 0.95),
            "reported_estimated_inference_time_us": execution["execution_summary"]["estimated_inference_time"],
            "reported_peak_memory_bytes": execution["execution_summary"]["estimated_inference_peak_memory"],
            "compute_unit_row_counts": dict(units),
            "profile_input": "Workbench generated input with pinned tensor shape; waveform inference used the retained fixture",
        }
    else:
        report["limits"].append("Requested compute unit is not verified node placement without a device profile.")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "profile": report.get("profile")}, indent=2))


if __name__ == "__main__":
    main()
