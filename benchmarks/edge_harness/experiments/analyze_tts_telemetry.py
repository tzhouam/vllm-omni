#!/usr/bin/env python3
"""Join a complete sustained TTS request run with device-wide GPU telemetry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from analyze_tts_sustained import WINDOWS_MIN, nearest_rank, summarize


METRICS = (
    "gpu_temperature_c",
    "gpu_power_mw",
    "gpu_clock_sm_mhz",
    "gpu_clock_memory_mhz",
    "gpu_utilization_pct",
    "gpu_memory_used_bytes",
    "host_available_bytes",
    "host_cpu_utilization_pct",
)


def analyze(report_path: Path, request_path: Path, telemetry_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("sustained_wall_s", 0) < 1800:
        raise ValueError("A terminal 30-minute sustained report is required")
    if report.get("gpu_telemetry", {}).get("error"):
        raise ValueError("GPU telemetry reported an error")
    if report.get("gpu_telemetry", {}).get("samples", 0) < 1500:
        raise ValueError("Too few device telemetry samples for a 30-minute run")
    start = report["sustained_start_unix"]
    base = summarize(request_path)
    telemetry = [json.loads(line) for line in telemetry_path.open(encoding="utf-8")]
    if len(telemetry) != report["gpu_telemetry"]["samples"]:
        raise ValueError("Telemetry sample count differs from the final report")
    for (lo, hi), window in zip(WINDOWS_MIN, base["windows"], strict=True):
        samples = [row for row in telemetry if lo * 60 <= row["unix"] - start < hi * 60]
        window["device_telemetry_samples"] = len(samples)
        window["device_telemetry"] = {}
        for name in METRICS:
            values = [row[name] for row in samples if row.get(name) is not None]
            window["device_telemetry"][name] = {
                "p50": nearest_rank(values, 0.5),
                "p95": nearest_rank(values, 0.95),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "available_samples": len(values),
            }
        window["gpu_throttle_reason_counts"] = dict(sorted(Counter(
            str(row["gpu_throttle_reasons"]) for row in samples
        ).items()))
    return {
        "scope": "complete-model Qwen3-TTS stream; simulated playback; telemetry covers whole GPU/host, not just this process",
        "report": str(report_path),
        "requests": str(request_path),
        "telemetry": str(telemetry_path),
        "loaded_omni_path": report.get("loaded_omni_path"),
        "cwd": report.get("cwd"),
        "start_unix": start,
        "sustained_wall_s": report["sustained_wall_s"],
        "windows": base["windows"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.run_dir / "report.json",
        args.run_dir / "requests.jsonl",
        args.run_dir / "gpu_telemetry.jsonl",
    )
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
