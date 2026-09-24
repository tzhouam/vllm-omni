#!/usr/bin/env python3
"""Check actual AMD NPU placement and waveform parity for one Qwen3-TTS vocoder window."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare(reference: np.ndarray, actual: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f"waveform shape changed: {reference.shape} versus {actual.shape}")
    error = actual - reference
    denom = max(float(np.linalg.norm(reference)), 1e-12)
    relative_l2 = float(np.linalg.norm(error) / denom)
    return {
        "finite": bool(np.isfinite(actual).all()),
        "relative_l2": relative_l2,
        "snr_db": float(-20 * np.log10(max(relative_l2, 1e-12))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "saturation_fraction": float(np.mean(np.abs(actual) >= 0.999)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "fixture", "reference", "ep-dir", "report", "profile-prefix"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--context-frames", type=int, default=97)
    parser.add_argument("--output-samples", type=int, default=48000)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    import onnxruntime as ort

    manifest_hashes = {
        "model": args.expected_model_sha256,
        "fixture": args.expected_fixture_sha256,
        "reference": args.expected_reference_sha256,
    }
    report = {
        "scope": "one real-weight Qwen3-TTS Code2Wav component window; not a complete TTS stream",
        "os": platform.platform(),
        "onnxruntime_version": ort.__version__,
        "input_shape": [1, 512, args.context_frames],
        "output_shape": [1, args.output_samples],
        "model_file": str(args.model.resolve()),
        "fixture_file": str(args.fixture.resolve()),
        "reference_file": str(args.reference.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.profile_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        report["artifact_sha256"] = {
            "model": sha256(args.model),
            "fixture": sha256(args.fixture),
            "reference": sha256(args.reference),
        }
        for name, expected in manifest_hashes.items():
            if report["artifact_sha256"][name] != expected:
                raise ValueError(f"{name} differs from pinned historical bundle")
        with np.load(args.fixture, allow_pickle=False) as source:
            if source.files != ["quantized"]:
                raise ValueError("vocoder fixture schema changed")
            quantized = np.ascontiguousarray(source["quantized"])
        with np.load(args.reference, allow_pickle=False) as source:
            if source.files != ["wav"]:
                raise ValueError("CPU reference schema changed")
            reference = np.asarray(source["wav"])
        if quantized.shape != (1, 512, args.context_frames) or quantized.dtype != np.float32:
            raise ValueError("vocoder input shape or dtype changed")
        if reference.shape != (1, args.output_samples) or reference.dtype != np.float32:
            raise ValueError("vocoder reference shape or dtype changed")

        cpu_started = time.perf_counter()
        cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
        report["cpu_load_s"] = time.perf_counter() - cpu_started
        inputs = cpu.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 512, args.context_frames]:
            raise ValueError("ONNX vocoder input contract changed")
        input_name = inputs[0].name
        cpu_started = time.perf_counter()
        cpu_wave = cpu.run(None, {input_name: quantized})[0]
        report["cpu_inference_s"] = time.perf_counter() - cpu_started
        report["cpu_vs_retained_reference"] = compare(reference, cpu_wave)
        if (not report["cpu_vs_retained_reference"]["finite"]
                or report["cpu_vs_retained_reference"]["relative_l2"] > 1e-4):
            raise ValueError("current ONNX CPU output differs from the retained reference")
        report["status"] = "cpu_reference_pass"
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        del cpu
        if args.cpu_only:
            return

        ep_dir = args.ep_dir.resolve(strict=True)
        os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(ep_dir))
        ort.register_execution_provider_library("vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
        devices = [device for device in ort.get_ep_devices()
                   if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
        report["vitisai_npu_device_count"] = len(devices)
        if not devices:
            raise RuntimeError("VitisAI EP did not expose an NPU device")
        options = ort.SessionOptions()
        options.add_provider_for_devices(devices, {})
        options.enable_profiling = True
        options.profile_file_prefix = str(args.profile_prefix)
        report["status"] = "vitisai_session_creation_started"
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        npu_started = time.perf_counter()
        npu = ort.InferenceSession(str(args.model), sess_options=options)
        report["npu_load_s"] = time.perf_counter() - npu_started
        npu_started = time.perf_counter()
        npu_wave = npu.run(None, {input_name: quantized})[0]
        report["npu_inference_s"] = time.perf_counter() - npu_started
        report["npu_vs_cpu"] = compare(cpu_wave, npu_wave)
        report["npu_vs_retained_reference"] = compare(reference, npu_wave)
        profile_path = Path(npu.end_profiling())
        report["profile_file"] = str(profile_path)
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        providers = [(event.get("args") or {}).get("provider") for event in profile
                     if event.get("cat") == "Node"]
        report["placement_events"] = {
            "vitisai": providers.count("vitisai"),
            "cpu": providers.count("CPUExecutionProvider"),
        }
        report["status"] = (
            "component_npu_numeric_pass"
            if (report["placement_events"]["vitisai"] > 0
                and report["npu_vs_cpu"]["finite"]
                and report["npu_vs_cpu"]["relative_l2"] <= 0.01
                and report["npu_vs_cpu"]["saturation_fraction"] < 0.01)
            else "component_not_qualified"
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "error": report.get("error"),
                          "placement_events": report.get("placement_events")}))


if __name__ == "__main__":
    main()
