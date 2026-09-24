#!/usr/bin/env python3
"""Bind a generated-code S25 waveform to its pinned TFLite target and CPU references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def waveform(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if data.files != [key]:
            raise ValueError(f"{path}: expected only {key}")
        value = np.asarray(data[key])
    if value.shape != (1, 48000) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"{path}: invalid 25-frame waveform {value.shape}/{value.dtype}")
    return value


def compare(reference: np.ndarray, observed: np.ndarray) -> dict:
    a = reference.astype(np.float64).ravel()
    b = observed.astype(np.float64).ravel()
    error = float(np.linalg.norm(a - b))
    norm = float(np.linalg.norm(a))
    return {
        "relative_l2": error / norm,
        "snr_db": 20 * math.log10(norm / error) if error else None,
        "max_abs": float(np.max(np.abs(a - b))),
        "reference_rms": float(np.sqrt(np.mean(a * a))),
        "device_rms": float(np.sqrt(np.mean(b * b))),
        "device_saturated_fraction_abs_ge_0_999": float(np.mean(np.abs(b) >= 0.999)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--pinned-compile", type=Path, required=True)
    parser.add_argument("--pinned-export", type=Path, required=True)
    args = parser.parse_args()
    p = args.evidence_dir
    local = json.loads((p / "report.json").read_text())
    dataset = json.loads((p / "dataset_submission.json").read_text())
    submitted = json.loads((p / "inference_submission.json").read_text())
    inference = json.loads((p / "inference_report.json").read_text())
    compiled = json.loads(args.pinned_compile.read_text())
    exported = json.loads(args.pinned_export.read_text())
    if (local["model_revision"] != exported["model_snapshot_revision"]
            or local["decoder_weight_sha256"] != exported["model_weight_sha256"]
            or local["onnx_sha256"] != exported["onnx_sha256"]):
        raise ValueError("generated codes and compiled source are not from the pinned export")
    if (local["files"]["fixture"]["sha256"] != dataset["fixture_sha256"]
            or dataset["input_shape"] != [1, 512, 97]
            or sha256(p / "fixture.npz") != dataset["fixture_sha256"]):
        raise ValueError("device input differs from real generated-code fixture")
    if (submitted["job_id"] != inference["job_id"]
            or inference["status"] != "SUCCESS"
            or inference["model_id"] != compiled["target_model_id"]
            or inference["input_dataset_id"] != dataset["dataset_id"]
            or inference["options"] != "--compute_unit gpu"
            or inference["device"] != compiled["device"]
            or inference["device"]["name"] != "Samsung Galaxy S25"):
        raise ValueError("device result does not bind to pinned S25 GPU target")
    for name in ("eager", "ort", "full_reference"):
        entry = local["files"][name]
        if sha256(p / entry["filename"]) != entry["sha256"]:
            raise ValueError(f"local {name} reference changed")
    cpu = waveform(p / "ort_output.npz", "wav")
    eager = waveform(p / "eager_output.npz", "wav")
    full = waveform(p / "full_reference_segment.npz", "wav")
    hosted = waveform(p / "device_output.npz", "output_0__0")
    comparison = compare(cpu, hosted)
    gross_failure = (comparison["relative_l2"] >= 1.0
                     or comparison["device_saturated_fraction_abs_ge_0_999"] >= 0.1)
    report = {
        "scope": "one generated-code 25-frame vocoder window on hosted S25; no device-local talker or complete stream",
        "status": "component_gross_numeric_failure" if gross_failure else "component_numeric_measured_no_speech_quality_gate",
        "device": inference["device"],
        "source_model_id": compiled["source_model_id"],
        "target_model_id": compiled["target_model_id"],
        "input_dataset_id": dataset["dataset_id"],
        "jobs": {"compile": compiled["job_id"], "inference": inference["job_id"]},
        "files": {name: {"sha256": sha256(p / filename), "bytes": (p / filename).stat().st_size}
                  for name, filename in {"fixture": "fixture.npz", "device": "device_output.npz",
                                         "ort": "ort_output.npz", "eager": "eager_output.npz",
                                         "full": "full_reference_segment.npz"}.items()},
        "ort_cpu_vs_device": comparison,
        "eager_cpu_vs_device": compare(eager, hosted),
        "full_fp32_decoder_segment_vs_device": compare(full, hosted),
        "gross_numeric_failure_screen": {
            "failed": gross_failure,
            "criteria": "relative L2 >= 1.0 or >=10% saturated samples; not a speech-quality tolerance",
        },
        "limits": [
            "One generated-code window is not a speech listening or intelligibility test.",
            "Talker, predictor, co-resident handoff, state, cancellation and memory admission did not run on S25.",
            "The separate placement profile used Workbench-generated same-shape input, not this fixture.",
        ],
    }
    (p / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "ort_cpu_vs_device": comparison}, indent=2))


if __name__ == "__main__":
    main()
