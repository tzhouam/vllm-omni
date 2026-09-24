#!/usr/bin/env python3
"""Audit one MiniCPM-o speech-head decode step against pinned ONNX CPU tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_SHA256 = "78cf64804f11ad269ee4180da61588ccc5eefe2942773227345499756e4cc229"
FIXTURE_SHA256 = "65919ea1f351feec33c1415a3e3db76d60bc9c2beb3d49a72af75dcb418e83b4"
REFERENCE_SHA256 = "902adcceadbe761d938ef2ade51cf953c8917ea1abfaee33a10c84e300c1bd96"
COMPILE_OPTIONS = {
    "--target_runtime qnn_context_binary --quantize_full_type float16",
    "--target_runtime qnn_dlc --qnn_options default_graph_htp_precision=FLOAT16",
    "--target_runtime tflite",
}
INPUT_NAMES = ["x", "cos", "sin", "mask"] + [
    name for layer in range(20) for name in (f"k_cache_{layer}", f"v_cache_{layer}")
]
OUTPUT_NAMES = ["logits"] + [
    name for layer in range(20) for name in (f"k_new_{layer}", f"v_new_{layer}")
]


def file_record(path: Path) -> dict:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def read_array(data: np.lib.npyio.NpzFile, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(data[name])
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite FP32 {shape}, got {value.shape}/{value.dtype}")
    return value


def compare(reference: np.ndarray, actual: np.ndarray) -> dict:
    source = reference.astype(np.float64).ravel()
    observed = actual.astype(np.float64).ravel()
    error = source - observed
    source_norm = float(np.linalg.norm(source))
    error_norm = float(np.linalg.norm(error))
    return {
        "relative_l2": error_norm / source_norm if source_norm else (0.0 if error_norm == 0 else None),
        "max_abs_error": float(np.max(np.abs(error))),
        "source_rms": float(np.sqrt(np.mean(source * source))),
        "device_rms": float(np.sqrt(np.mean(observed * observed))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--cpu-reference", type=Path, required=True)
    parser.add_argument("--compile-report", type=Path, required=True)
    parser.add_argument("--inference-report", type=Path, required=True)
    parser.add_argument("--device-output", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    compiled = json.loads(args.compile_report.read_text(encoding="utf-8"))
    inferred = json.loads(args.inference_report.read_text(encoding="utf-8"))
    if compiled.get("status") != "SUCCESS" or inferred.get("status") != "SUCCESS":
        raise ValueError("compile and inference must be terminal successes")
    if compiled.get("source_model_id") != "mn4gjjpvn" or compiled.get("options") not in COMPILE_OPTIONS:
        raise ValueError("compile source/options differ from the pinned speech-head artifact")
    if inferred.get("model_id") != compiled.get("target_model_id"):
        raise ValueError("inference used a different target artifact")
    if compiled["options"] == "--target_runtime tflite":
        allowed_run_options = {"--compute_unit cpu"}
    elif compiled["options"].startswith("--target_runtime qnn_dlc"):
        allowed_run_options = {"--compute_unit npu"}
    else:
        # Historical S24 context-binary inference did not record a run option.
        allowed_run_options = {"", "--compute_unit npu"}
    if inferred.get("options") not in allowed_run_options:
        raise ValueError("inference compute request differs from the compiled target route")
    if inferred.get("input_dataset_id") not in {"d7gwxe3y2", "d7zn8rk57"}:
        raise ValueError("inference did not use a retained matching fixture dataset")
    if (compiled.get("device", {}).get("name") != args.device_name
            or inferred.get("device", {}).get("name") != args.device_name):
        raise ValueError("jobs did not use the exact requested device")

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
            raise ValueError(f"{name} differs from the pinned artifact")

    with np.load(args.fixture, allow_pickle=False) as fixture:
        if fixture.files != INPUT_NAMES:
            raise ValueError("fixture input names/order differ")
        read_array(fixture, "x", (1, 1, 768))
        for name in ("cos", "sin"):
            read_array(fixture, name, (1, 1, 1, 64))
        read_array(fixture, "mask", (1, 1, 1, 257))
        for name in INPUT_NAMES[4:]:
            read_array(fixture, name, (1, 12, 256, 64))

    comparisons = {}
    with np.load(args.cpu_reference, allow_pickle=False) as reference, np.load(
        args.device_output, allow_pickle=False
    ) as output:
        if reference.files != OUTPUT_NAMES:
            raise ValueError("CPU output names/order differ")
        expected_keys = [f"output_{index}__0" for index in range(len(OUTPUT_NAMES))]
        if output.files != expected_keys:
            raise ValueError("device output names/order differ")
        for index, name in enumerate(OUTPUT_NAMES):
            shape = (1, 6562) if index == 0 else (1, 12, 1, 64)
            source_value = read_array(reference, name, shape)
            device_value = read_array(output, expected_keys[index], shape)
            comparisons[name] = compare(source_value, device_value)
            if name == "logits":
                comparisons[name]["cpu_top1"] = int(np.argmax(source_value[0]))
                comparisons[name]["device_top1"] = int(np.argmax(device_value[0]))
                comparisons[name]["top1_matches"] = bool(
                    comparisons[name]["cpu_top1"] == comparisons[name]["device_top1"]
                )

    report = {
        "scope": "one fixed-cache 20-layer MiniCPM-o speech-head decode step; no complete multimodal stream",
        "status": "component_inference_numeric_measured",
        "device": inferred["device"],
        "compile_job_id": compiled["job_id"],
        "inference_job_id": inferred["job_id"],
        "source_model_id": compiled["source_model_id"],
        "target_model_id": compiled["target_model_id"],
        "input_dataset_id": inferred["input_dataset_id"],
        "requested_run_options": inferred["options"],
        "files": files,
        "tensor_comparison": comparisons,
        "cache_max_relative_l2": max(comparisons[name]["relative_l2"] for name in OUTPUT_NAMES[1:]),
        "limits": [
            "The historical ONNX export does not attest the exact source checkpoint revision.",
            "One retained decode fixture cannot establish speech-token or generated-audio quality.",
            "Actual CPU/NPU placement requires a device profile; a request or compiled artifact alone is insufficient.",
            "Thinker, audio/vision encoders, vocoder, state loop and complete Omni request were not tested.",
        ],
    }
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "device": report["device"]["name"],
        "logits": comparisons["logits"],
        "cache_max_relative_l2": report["cache_max_relative_l2"],
    }, indent=2))


if __name__ == "__main__":
    main()
