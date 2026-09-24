#!/usr/bin/env python3
"""Validate real-weight MiniCPM-o speech-head placement and one-step parity on AMD NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(expected: np.ndarray, actual: np.ndarray) -> dict:
    if expected.shape != actual.shape:
        raise ValueError("output tensor shape changed")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "finite": bool(np.isfinite(actual).all()),
        "relative_l2": float(np.linalg.norm(delta) / max(
            np.linalg.norm(expected.astype(np.float64)), 1e-12)),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "fixture", "reference", "ep-dir", "report", "profile-prefix"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("model", "fixture", "reference"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--candidate-cpu-relative-l2-gate", type=float, default=1e-4)
    args = parser.parse_args()
    import onnxruntime as ort

    artifact_hashes = {name: sha256(getattr(args, name))
                       for name in ("model", "fixture", "reference")}
    for name, actual in artifact_hashes.items():
        if actual != getattr(args, f"expected_{name}_sha256"):
            raise ValueError(f"{name} hash changed")
    candidate = None
    if args.candidate_report is not None:
        candidate = json.loads(args.candidate_report.read_text(encoding="utf-8"))
        if candidate.get("model_sha256") != "78cf64804f11ad269ee4180da61588ccc5eefe2942773227345499756e4cc229" or candidate.get("candidate_sha256") != artifact_hashes["model"]:
            raise ValueError("candidate does not derive from pinned MiniCPM-o source")
    report = {
        "scope": "real-weight one-step MiniCPM-o 4.5 speech head, fixed 256-token cache; not complete MiniCPM-o",
        "os": platform.platform(),
        "onnxruntime_version": ort.__version__,
        "artifact_sha256": artifact_hashes,
        "candidate_quantization": candidate.get("nodes_to_quantize") if candidate else None,
        "candidate_cpu_relative_l2_gate": args.candidate_cpu_relative_l2_gate,
        "status": "started",
    }
    write_report(args.report, report)
    with np.load(args.fixture, allow_pickle=False) as fixture:
        inputs = {name: np.ascontiguousarray(fixture[name]) for name in fixture.files}
    with np.load(args.reference, allow_pickle=False) as source:
        expected = {name: np.asarray(source[name]) for name in source.files}
    if len(inputs) != 44 or len(expected) != 41:
        raise ValueError("speech-head fixture/reference schema changed")
    started = time.perf_counter()
    cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    report["cpu_session_create_s"] = time.perf_counter() - started
    input_names = [item.name for item in cpu.get_inputs()]
    output_names = [item.name for item in cpu.get_outputs()]
    if set(input_names) != set(inputs) or set(output_names) != set(expected):
        raise ValueError("ONNX speech-head input/output names differ from pinned fixture")
    started = time.perf_counter()
    cpu_outputs = dict(zip(output_names, cpu.run(None, inputs), strict=True))
    report["cpu_one_inference_s"] = time.perf_counter() - started
    report["cpu_vs_retained_reference"] = {
        name: compare(expected[name], cpu_outputs[name]) for name in output_names}
    report["cpu_top1"] = int(np.argmax(cpu_outputs["logits"]))
    cpu_cache_max = max(
        values["relative_l2"] for name, values in report["cpu_vs_retained_reference"].items()
        if name != "logits")
    if (not all(values["finite"] for values in report["cpu_vs_retained_reference"].values())
            or report["cpu_vs_retained_reference"]["logits"]["relative_l2"] > args.candidate_cpu_relative_l2_gate
            or cpu_cache_max > args.candidate_cpu_relative_l2_gate):
        raise ValueError("local CPU speech-head output differs from retained reference")
    report["status"] = "cpu_reference_pass"
    write_report(args.report, report)
    del cpu
    if args.cpu_only:
        return

    ep_dir = args.ep_dir.resolve(strict=True)
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library(
        "vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
    devices = [device for device in ort.get_ep_devices()
               if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
    report["vitisai_npu_device_count"] = len(devices)
    report["status"] = "vitisai_session_creation_started"
    write_report(args.report, report)
    if not devices:
        raise RuntimeError("VitisAI EP did not expose an NPU device")
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    options.enable_profiling = True
    options.profile_file_prefix = str(args.profile_prefix)
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options)
    report["vitisai_session_create_s"] = time.perf_counter() - started
    report["status"] = "vitisai_session_created"
    write_report(args.report, report)
    started = time.perf_counter()
    actual = dict(zip(output_names, session.run(None, inputs), strict=True))
    report["vitisai_one_inference_s"] = time.perf_counter() - started
    report["vitisai_vs_cpu"] = {
        name: compare(cpu_outputs[name], actual[name]) for name in output_names}
    report["vitisai_top1"] = int(np.argmax(actual["logits"]))
    report["vitisai_cache_max_relative_l2"] = max(
        values["relative_l2"] for name, values in report["vitisai_vs_cpu"].items()
        if name != "logits")
    profile = Path(session.end_profiling())
    report["profile_file"] = str(profile)
    events = json.loads(profile.read_text(encoding="utf-8"))
    report["node_providers"] = dict(Counter(
        (event.get("args") or {}).get("provider") for event in events
        if event.get("cat") == "Node"))
    report["status"] = (
        "component_npu_numeric_pass"
        if report["node_providers"].get("vitisai", 0) > 0
        and report["vitisai_top1"] == report["cpu_top1"]
        and all(values["finite"] for values in report["vitisai_vs_cpu"].values())
        and report["vitisai_vs_cpu"]["logits"]["relative_l2"] <= 0.01
        and report["vitisai_cache_max_relative_l2"] <= 0.01
        else "component_not_qualified"
    )
    write_report(args.report, report)
    print(json.dumps({"status": report["status"],
                      "npu_nodes": report["node_providers"].get("vitisai", 0),
                      "top1_match": report["vitisai_top1"] == report["cpu_top1"]}))


if __name__ == "__main__":
    main()
