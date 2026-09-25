#!/usr/bin/env python3
"""Measure the real Qwen3-TTS layer-0 MLP on HX370 NPU and retain replay outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    first, second = reference.astype(np.float64), actual.astype(np.float64)
    return float(np.linalg.norm(first - second) / max(np.linalg.norm(first), 1e-12))


def rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlp", "fixture", "ep-dir", "profile-prefix", "capture", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-mlp-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    args = parser.parse_args()
    if (sha256(args.mlp) != args.expected_mlp_sha256
            or sha256(args.fixture) != args.expected_fixture_sha256):
        raise ValueError("MLP candidate or eleven-step fixture changed")

    import onnxruntime as ort

    with np.load(args.fixture, allow_pickle=False) as archive:
        fixture = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    expected = {f"residual_step{index}" for index in range(11)} | {
        f"cpu_step{index}_out{output}" for index in range(11) for output in range(3)}
    if set(fixture) != expected:
        raise ValueError("MLP fixture fields differ from declared eleven-step contract")

    cpu = ort.InferenceSession(str(args.mlp), providers=["CPUExecutionProvider"])
    if [item.name for item in cpu.get_inputs()] != ["residual"] or len(cpu.get_outputs()) != 1:
        raise ValueError("MLP input/output contract changed")
    cpu_outputs = []
    for index in range(11):
        output = cpu.run(None, {"residual": fixture[f"residual_step{index}"]})[0]
        reference = fixture[f"cpu_step{index}_out0"]
        if relative_l2(reference, output) > 1e-6:
            raise ValueError(f"CPU MLP differs from pinned complete layer at step {index}")
        cpu_outputs.append(output)
    cpu.run(None, {"residual": fixture["residual_step0"]})  # Excluded warmup.
    cpu_times = []
    for index in range(11):
        started = time.perf_counter()
        output = cpu.run(None, {"residual": fixture[f"residual_step{index}"]})[0]
        cpu_times.append(time.perf_counter() - started)
        if relative_l2(cpu_outputs[index], output) > 1e-6:
            raise ValueError(f"CPU MLP repeat changed step {index}")

    ep_dir = args.ep_dir.resolve(strict=True)
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(ep_dir))
    ep_dll = ep_dir / "onnxruntime_vitisai_ep.dll"
    ort.register_execution_provider_library("vitisai", str(ep_dll))
    devices = [device for device in ort.get_ep_devices()
               if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
    if not devices:
        raise RuntimeError("VitisAI EP did not expose an NPU")
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    options.enable_profiling = True
    args.profile_prefix.parent.mkdir(parents=True, exist_ok=True)
    options.profile_file_prefix = str(args.profile_prefix)
    report = {
        "scope": "real checkpoint layer-0 CPU attention/KV plus NPU MLP, eleven captured code steps; component only",
        "os": platform.platform(), "onnxruntime": ort.__version__,
        "artifact_sha256": {"mlp": sha256(args.mlp), "fixture": sha256(args.fixture),
                            "ep_dll": sha256(ep_dll)},
        "cpu_call_s": cpu_times,
        "nearest_rank_p50_cpu_call_s": rank(cpu_times, .5),
        "nearest_rank_p95_cpu_call_s": rank(cpu_times, .95),
        "npu_device_count": len(devices), "status": "session_creation_started",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    npu = ort.InferenceSession(str(args.mlp), sess_options=options)
    report["session_creation_s"] = time.perf_counter() - started
    npu.run(None, {"residual": fixture["residual_step0"]})  # Excluded warmup.
    captures = {}
    rows = []
    for index in range(11):
        residual = fixture[f"residual_step{index}"]
        started = time.perf_counter()
        output = npu.run(None, {"residual": residual})[0]
        elapsed = time.perf_counter() - started
        reference = cpu_outputs[index]
        if output.shape != reference.shape or output.dtype != np.float32 or not np.isfinite(output).all():
            raise ValueError(f"NPU MLP returned an invalid hidden tensor at step {index}")
        rows.append({"start_frame": 95 + 2 * index, "npu_call_s": elapsed,
                     "hidden_relative_l2_vs_cpu": relative_l2(reference, output),
                     "finite": True})
        for output_index in range(3):
            captures[f"cpu_step{index}_out{output_index}"] = fixture[f"cpu_step{index}_out{output_index}"]
            captures[f"npu_step{index}_out{output_index}"] = (
                output if output_index == 0 else fixture[f"cpu_step{index}_out{output_index}"])
    args.capture.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.capture, **captures)
    profile = Path(npu.end_profiling())
    events = json.loads(profile.read_text(encoding="utf-8"))
    providers = [(event.get("args") or {}).get("provider") for event in events if event.get("cat") == "Node"]
    report.update({
        "rows": rows,
        "nearest_rank_p50_npu_call_s": rank([row["npu_call_s"] for row in rows], .5),
        "nearest_rank_p95_npu_call_s": rank([row["npu_call_s"] for row in rows], .95),
        "node_providers": {"vitisai": providers.count("vitisai"),
                           "cpu": providers.count("CPUExecutionProvider")},
        "profile_file": str(profile), "profile_sha256": sha256(profile),
        "capture_sha256": sha256(args.capture),
    })
    report["status"] = ("npu_mlp_component_numeric_pass"
                        if report["node_providers"]["vitisai"] >= 12
                        and all(row["hidden_relative_l2_vs_cpu"] <= .01 for row in rows)
                        else "npu_mlp_component_not_qualified")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "node_providers": report["node_providers"],
                      "worst_hidden_relative_l2": max(row["hidden_relative_l2_vs_cpu"] for row in rows)}))


if __name__ == "__main__":
    main()
