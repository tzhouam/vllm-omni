#!/usr/bin/env python3
"""Audit one hosted Qwen3-TTS talker decode step against pinned ONNX CPU tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


SOURCE_SHA256 = "41feb08517678e94d9301e992d0df610fc60e7c9faba5600703298c86554fd22"
FIXTURE_SHA256 = "0bf538e1b64dfc9737dc4f925d61c6e4ffa5075a5dcd7afb1e85ac3bd2d9afea"
REFERENCE_SHA256 = "ee0ead70e33c02c7d3d6bcb757c5cf0fcbb9eb6fc75c34abfbe8a4479657d5be"
COMPILE_OPTIONS = "--target_runtime qnn_context_binary --quantize_full_type float16"
INPUT_NAMES = ["x", "cos", "sin", "mask"] + [
    name for layer in range(28) for name in (f"k_cache_{layer}", f"v_cache_{layer}")
]
OUTPUT_NAMES = ["logits"] + [
    name for layer in range(28) for name in (f"k_new_{layer}", f"v_new_{layer}")
]


def file_record(path: Path) -> dict:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def require_array(data: np.lib.npyio.NpzFile, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(data[name])
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite FP32 {shape}, got {value.shape}/{value.dtype}")
    return value


def compare(reference: np.ndarray, actual: np.ndarray) -> dict:
    original = reference.astype(np.float64).ravel()
    observed = actual.astype(np.float64).ravel()
    error = original - observed
    norm = float(np.linalg.norm(original))
    delta = float(np.linalg.norm(error))
    return {
        "relative_l2": delta / norm if norm else (0.0 if delta == 0 else None),
        "max_abs_error": float(np.max(np.abs(error))),
        "reference_rms": float(np.sqrt(np.mean(original * original))),
        "output_rms": float(np.sqrt(np.mean(observed * observed))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--cpu-reference", type=Path, required=True)
    parser.add_argument("--compile-report", type=Path, required=True)
    parser.add_argument("--inference-report", type=Path, required=True)
    parser.add_argument("--device-output", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-compile-options", default=COMPILE_OPTIONS)
    parser.add_argument("--expected-run-options", default="--compute_unit npu")
    parser.add_argument("--profile-report", type=Path)
    args = parser.parse_args()

    compiled = json.loads(args.compile_report.read_text(encoding="utf-8"))
    inferred = json.loads(args.inference_report.read_text(encoding="utf-8"))
    if compiled.get("status") != "SUCCESS" or inferred.get("status") != "SUCCESS":
        raise ValueError("compile and inference must both be terminal successes")
    if compiled.get("source_model_id") != "mq3xre4rq" or compiled.get("options") != args.expected_compile_options:
        raise ValueError("compile source or options differ from the pinned talker artifact")
    if inferred.get("model_id") != compiled.get("target_model_id"):
        raise ValueError("inference did not use the audited target artifact")
    if (inferred.get("input_dataset_id") != "d7dw6pgy2"
            or inferred.get("options") != args.expected_run_options):
        raise ValueError("inference fixture or requested compute unit differs")
    if inferred.get("device") != compiled.get("device"):
        raise ValueError("compile/inference device metadata differs")

    files = {
        name: file_record(path)
        for name, path in {
            "source_onnx": args.source_onnx,
            "fixture": args.fixture,
            "cpu_reference": args.cpu_reference,
            "device_output": args.device_output,
        }.items()
    }
    for name, expected in {
        "source_onnx": SOURCE_SHA256,
        "fixture": FIXTURE_SHA256,
        "cpu_reference": REFERENCE_SHA256,
    }.items():
        if files[name]["sha256"] != expected:
            raise ValueError(f"{name} hash differs from pinned source")

    with np.load(args.fixture, allow_pickle=False) as fixture:
        if fixture.files != INPUT_NAMES:
            raise ValueError("fixture input names/order differ from the source contract")
        require_array(fixture, "x", (1, 1, 1024))
        for name in ("cos", "sin"):
            require_array(fixture, name, (1, 1, 1, 128))
        require_array(fixture, "mask", (1, 1, 1, 257))
        for name in INPUT_NAMES[4:]:
            require_array(fixture, name, (1, 8, 256, 128))

    details = {}
    with np.load(args.cpu_reference, allow_pickle=False) as reference, np.load(
        args.device_output, allow_pickle=False
    ) as output:
        if reference.files != OUTPUT_NAMES:
            raise ValueError("CPU reference output names/order differ from the source contract")
        expected_device_keys = [f"output_{index}__0" for index in range(len(OUTPUT_NAMES))]
        if output.files != expected_device_keys:
            raise ValueError("device output names/order differ from the compiled contract")
        for index, name in enumerate(OUTPUT_NAMES):
            shape = (1, 3072) if index == 0 else (1, 8, 1, 128)
            source_value = require_array(reference, name, shape)
            device_value = require_array(output, expected_device_keys[index], shape)
            details[name] = compare(source_value, device_value)
            if name == "logits":
                details[name]["cpu_top1"] = int(np.argmax(source_value[0]))
                details[name]["device_top1"] = int(np.argmax(device_value[0]))
                details[name]["top1_matches"] = bool(
                    details[name]["cpu_top1"] == details[name]["device_top1"]
                )

    report = {
        "scope": "one synthetic fixed-cache 28-layer Qwen3-TTS talker decode step; no complete TTS stream",
        "status": "component_executed_numeric_measured",
        "device": inferred["device"],
        "compile_job_id": compiled["job_id"],
        "inference_job_id": inferred["job_id"],
        "source_model_id": compiled["source_model_id"],
        "target_model_id": compiled["target_model_id"],
        "input_dataset_id": inferred["input_dataset_id"],
        "requested_run_options": inferred["options"],
        "files": files,
        "tensor_comparison": details,
        "cache_max_relative_l2": max(
            details[name]["relative_l2"] for name in OUTPUT_NAMES[1:]
        ),
        "limits": [
            "The historical source export does not attest an exact checkpoint revision.",
            "The fixture is synthetic and has no token-to-audio quality tolerance.",
            "No predictor, vocoder, persistent cache loop or complete TTS stream was tested.",
        ],
    }
    if args.profile_report is not None:
        profiled = json.loads(args.profile_report.read_text(encoding="utf-8"))
        if (profiled.get("status") != "SUCCESS"
                or profiled.get("model_id") != compiled["target_model_id"]
                or profiled.get("device") != inferred["device"]
                or profiled.get("options") != args.expected_run_options):
            raise ValueError("profile did not execute the same target, device and route")
        summary = profiled["profile"]["execution_summary"]
        samples = summary["all_inference_times"]
        if not samples or any(not isinstance(value, int) or value <= 0 for value in samples):
            raise ValueError("profile has no valid inference samples")
        units = Counter(row.get("compute_unit") for row in profiled["profile"]["execution_detail"])
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
            "reported_first_load_peak_memory_bytes": summary["first_load_peak_memory"],
            "compute_unit_row_counts": dict(units),
        }
        report["status"] = "component_inference_numeric_and_placement_measured"
    else:
        report["limits"].append("Requested compute unit is not actual placement without a device profile.")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "device": report["device"]["name"],
        "logits": details["logits"],
        "cache_max_relative_l2": report["cache_max_relative_l2"],
    }, indent=2))


if __name__ == "__main__":
    main()
