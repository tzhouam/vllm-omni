#!/usr/bin/env python3
"""Summarize an interrupted TTS run without treating it as a sustained pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from analyze_tts_sustained import nearest_rank, required_prebuffer_ms


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _power_limit(text: str, label: str) -> float:
    match = re.search(rf"{label}\s*:\s*([0-9.]+) W", text)
    if match is None:
        raise ValueError(f"missing GPU {label}")
    return float(match.group(1))


def _phase(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no complete requests in phase")
    for row in rows:
        if not row["finished"] or not row["chunks_all"]:
            raise ValueError(f"incomplete request: {row['request_id']}")
        if not all(chunk["finite"] for chunk in row["chunks_all"]):
            raise ValueError(f"nonfinite audio: {row['request_id']}")
        chunks = row["chunks_all"]
        if (sum(bool(chunk["terminal"]) for chunk in chunks) != 1
                or not chunks[-1]["terminal"]
                or [chunk["idx"] for chunk in chunks] != list(range(len(chunks)))
                or any(chunk["request_id"] != row["request_id"] for chunk in chunks)):
            raise ValueError(f"chunk order or terminal event differs: {row['request_id']}")
    return {
        "count": len(rows),
        "request_wall_s_p50": nearest_rank([row["total_wall_s"] for row in rows], .5),
        "request_wall_s_p95": nearest_rank([row["total_wall_s"] for row in rows], .95),
        "rtf_p50": nearest_rank([row["rtf_total"] for row in rows], .5),
        "rtf_p95": nearest_rank([row["rtf_total"] for row in rows], .95),
        "ttfa_ms_p50": nearest_rank([row["ttfa_ms"] for row in rows], .5),
        "ttfa_ms_p95": nearest_rank([row["ttfa_ms"] for row in rows], .95),
        "requests_with_simulated_underrun": sum(bool(row["underruns"]) for row in rows),
        "required_extra_prebuffer_ms_p95": nearest_rank(
            [required_prebuffer_ms(row) for row in rows], .95),
    }


def analyze(run_dir: Path) -> dict:
    names = (
        "report.json", "requests.jsonl", "gpu_telemetry.jsonl",
        "gpu_power_state_during_sustained.txt",
        "windows_power_state_during_sustained.json",
    )
    paths = {name: run_dir / name for name in names}
    report = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    if report["status"] != "running" or report.get("end_unix") is None:
        raise ValueError("expected a stopped run with the harness finally block completed")
    if report.get("sustained_wall_s") != 0 or report.get("sustained_start_unix") is None:
        raise ValueError("the run is not the interrupted sustained phase")
    telemetry_status = report.get("gpu_telemetry", {})
    if telemetry_status.get("error"):
        raise ValueError("GPU telemetry failed")
    rows = [json.loads(line) for line in paths["requests.jsonl"].open(encoding="utf-8")]
    telemetry = [json.loads(line) for line in paths["gpu_telemetry.jsonl"].open(encoding="utf-8")]
    if telemetry_status.get("samples") != len(telemetry):
        raise ValueError("GPU telemetry sample count differs from the harness report")
    phases = {phase: _phase([row for row in rows if row["phase"] == phase])
              for phase in ("measured", "sustained")}
    start = report["sustained_start_unix"]
    completed_sustained = [row for row in rows if row["phase"] == "sustained"]
    active = [row for row in telemetry if start <= row["unix"] <= report["end_unix"]]
    if not active:
        raise ValueError("no GPU samples in the interrupted sustained phase")
    gpu = {}
    for name in ("gpu_power_mw", "gpu_clock_sm_mhz", "gpu_clock_memory_mhz",
                 "gpu_utilization_pct", "gpu_temperature_c", "gpu_memory_used_bytes"):
        values = [row[name] for row in active if row.get(name) is not None]
        gpu[name] = {
            "samples": len(values),
            "p50": nearest_rank(values, .5),
            "p95": nearest_rank(values, .95),
            "max": max(values) if values else None,
        }
    power_text = paths["gpu_power_state_during_sustained.txt"].read_text(encoding="utf-8")
    windows = json.loads(paths["windows_power_state_during_sustained.json"].read_text(encoding="utf-8-sig"))
    return {
        "status": "interrupted_after_failed_realtime_gate",
        "scope": "WSL2 HX370 + RTX 5090 Laptop, exact-state Omni Qwen3-TTS; simulated playback only",
        "stop_reason": "operator SIGINT after every complete measured/sustained request missed the playback schedule",
        "not_a_30_minute_pass": True,
        "start_unix": report["start_unix"],
        "sustained_start_unix": start,
        "end_unix": report["end_unix"],
        "last_complete_sustained_request_s": completed_sustained[-1]["finished_unix"] - start,
        "start_to_harness_end_s": report["end_unix"] - start,
        "phases": phases,
        "gpu_telemetry": {
            "scope": "device-wide sampled values; not process-attributed",
            "sustained_samples": len(active),
            "metrics": gpu,
        },
        "power_condition": {
            "observed_gpu_limit_w": _power_limit(power_text, "Current Power Limit"),
            "gpu_default_limit_w": _power_limit(power_text, "Default Power Limit"),
            "software_power_cap_active": bool(re.search(r"SW Power Cap\s*:\s*Active", power_text)),
            "windows": windows,
        },
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "phases": result["phases"],
                      "observed_gpu_limit_w": result["power_condition"]["observed_gpu_limit_w"]}))


if __name__ == "__main__":
    main()
