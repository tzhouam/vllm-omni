#!/usr/bin/env python3
"""Probe the real Qwen3-TTS rolling-KV ONNX step on a native AMD NPU."""

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
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def run_steps(session, inputs: dict[str, np.ndarray], *, layer0: bool, step_count: int):
    names = [item.name for item in session.get_inputs()]
    expected = 4 if layer0 else 18
    if names[:2] != ["conv", "positions"] or len(names) != expected:
        raise ValueError(f"stateful ONNX input contract changed: {names}")
    state = {name: inputs[name] for name in names[2:]}
    steps = []
    for index in range(step_count):
        start = 95 + 2 * index
        conv_name = "conv" if index == 0 else "conv_next" if index == 1 else f"conv_step{index}"
        conv = inputs[conv_name]
        feeds = {"conv": conv, "positions": np.array([[start, start + 1]], np.int64), **state}
        begun = time.perf_counter()
        outputs = session.run(None, feeds)
        steps.append({"elapsed_s": time.perf_counter() - begun,
                      "outputs": outputs})
        state = {name: outputs[index + 1] for index, name in enumerate(names[2:])}
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "fixture", "ep-dir", "report", "profile-prefix"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--capture-dir", type=Path,
                        help="Retain both consecutive CPU and NPU output tuples")
    parser.add_argument("--layer0", action="store_true",
                        help="Probe the extracted first transformer layer only")
    parser.add_argument("--steps", type=int, default=2,
                        help="Number of consecutive two-frame steps beginning at frame 95")
    args = parser.parse_args()
    if not 1 <= args.steps <= 11:
        parser.error("--steps must be 1..11 for the retained 117-frame utterance")

    import onnxruntime as ort

    report = {
        "scope": ("real-weight first transformer layer with rolling KV"
                  if args.layer0 else "real-weight 8-layer rolling-KV pre-transformer step")
                 + "; not complete TTS or waveform quality",
        "os": platform.platform(),
        "onnxruntime": ort.__version__,
        "artifact_sha256": {"model": sha256(args.model), "fixture": sha256(args.fixture)},
        "status": "started",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.profile_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        if (report["artifact_sha256"]["model"] != args.expected_model_sha256
                or report["artifact_sha256"]["fixture"] != args.expected_fixture_sha256):
            raise ValueError("stateful graph or source fixture changed")
        with np.load(args.fixture, allow_pickle=False) as archive:
            inputs = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        cpu_started = time.perf_counter()
        cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
        report["cpu_session_create_s"] = time.perf_counter() - cpu_started
        cpu_steps = run_steps(cpu, inputs, layer0=args.layer0, step_count=args.steps)
        report["cpu_step_s"] = [step["elapsed_s"] for step in cpu_steps]
        report["status"] = "cpu_reference_pass"
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

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
        started = time.perf_counter()
        npu = ort.InferenceSession(str(args.model), sess_options=options)
        report["npu_session_create_s"] = time.perf_counter() - started
        npu_steps = run_steps(npu, inputs, layer0=args.layer0, step_count=args.steps)
        report["npu_step_s"] = [step["elapsed_s"] for step in npu_steps]
        if args.capture_dir is not None:
            args.capture_dir.mkdir(parents=True, exist_ok=True)
            captured = {}
            for index, (reference, candidate) in enumerate(zip(cpu_steps, npu_steps)):
                for kind, step in (("cpu", reference), ("npu", candidate)):
                    for output_index, value in enumerate(step["outputs"]):
                        captured[f"{kind}_step{index}_out{output_index}"] = value
            capture_path = args.capture_dir / ("layer0_outputs.npz" if args.layer0
                                                else "full_state_outputs.npz")
            np.savez_compressed(capture_path, **captured)
            report["capture"] = {"filename": capture_path.name,
                                 "sha256": sha256(capture_path)}
        report["comparison"] = []
        for index, (reference, candidate) in enumerate(zip(cpu_steps, npu_steps)):
            errors = [relative_l2(a, b) for a, b in zip(reference["outputs"], candidate["outputs"])]
            report["comparison"].append({"start_frame": 95 + 2 * index,
                                         "hidden_relative_l2": errors[0],
                                         "max_state_relative_l2": max(errors[1:]),
                                         "finite": all(np.isfinite(value).all() for value in candidate["outputs"])})
        profile_path = Path(npu.end_profiling())
        report["profile_file"] = str(profile_path)
        events = json.loads(profile_path.read_text(encoding="utf-8"))
        providers = [(event.get("args") or {}).get("provider") for event in events
                     if event.get("cat") == "Node"]
        report["node_providers"] = {"vitisai": providers.count("vitisai"),
                                    "cpu": providers.count("CPUExecutionProvider")}
        report["status"] = (
            "npu_stateful_component_numeric_pass"
            if report["node_providers"]["vitisai"] > 0
            and all(row["finite"] and row["hidden_relative_l2"] <= 0.01
                    and row["max_state_relative_l2"] <= 0.01 for row in report["comparison"])
            else "npu_stateful_component_not_qualified"
        )
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "error": report.get("error"),
                          "node_providers": report.get("node_providers")}))


if __name__ == "__main__":
    main()
