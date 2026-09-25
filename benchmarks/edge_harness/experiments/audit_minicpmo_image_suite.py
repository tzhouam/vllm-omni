#!/usr/bin/env python3
"""Audit same-input CPU and CPU+NPU MiniCPM-o image/audio suites and placement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_path(report: Path, recorded_path: str) -> Path:
    as_run = Path(recorded_path)
    if as_run.is_file():
        return as_run
    adjacent = report.parent / as_run.parent.name / as_run.name
    if adjacent.is_file():
        return adjacent
    raise FileNotFoundError(f"waveform archive missing: {recorded_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-report", type=Path, required=True)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--npu-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    npu = json.loads(args.npu_report.read_text(encoding="utf-8"))
    cpu = json.loads(args.cpu_report.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in args.npu_events.read_text(encoding="utf-8").splitlines()]
    request_count = len(npu["requests"])
    if (npu["model"] != cpu["model"] or request_count < 2
            or len(cpu["requests"]) != request_count
            or npu.get("audio_sha256") != cpu.get("audio_sha256")):
        raise ValueError("suites must use one checkpoint and matching input requests")
    placements = [row["placement"] for row in events if row["phase"] == "placement"]
    runs = [row for row in events if row["phase"] == "run"]
    completed = [row for row in events if row["phase"] == "request_complete"]
    closes = [row for row in events if row["phase"] == "close"]
    if (len(placements) != 1 or not runs or len(closes) != 1
            or placements[0]["ep"] != "vitisai" or placements[0]["target_nodes"] < 1
            or closes[0]["calls"] != len(runs)):
        raise ValueError("missing verified NPU placement/calls or clean close")
    if completed:
        if [row["request"] for row in completed] != list(range(1, request_count + 1)):
            raise ValueError("missing or unordered tiled NPU requests")
        grouped_runs = [[row for row in runs if row.get("request") == index + 1]
                        for index in range(request_count)]
        for index, (group, done) in enumerate(zip(grouped_runs, completed)):
            if (len(group) != done["tiles"]
                    or [row["tile"] for row in group] != list(range(len(group)))
                    or sum(row["valid_tokens"] for row in group)
                    != done["input_shape"][0] * done["input_shape"][1]):
                raise ValueError(f"incomplete or unordered NPU tiles for request {index}")
        if sum(map(len, grouped_runs)) != len(runs):
            raise ValueError("unattributed NPU graph calls")
    elif len(runs) == request_count and all("request" not in row for row in runs):
        grouped_runs = [[row] for row in runs]
    else:
        raise ValueError("NPU calls do not cover complete requests")

    comparison = []
    for index, (a, b) in enumerate(zip(npu["requests"], cpu["requests"])):
        if (a["image_sha256"] != b["image_sha256"]
                or a.get("audio_sha256") != b.get("audio_sha256")
                or a["index"] != index
                or b["index"] != index
                or not a["text"] or not b["text"] or not a["token_ids"] or not b["token_ids"]
                or a["audio_samples"] <= 0 or b["audio_samples"] <= 0
                or a["audio_rms_float"] <= 0 or b["audio_rms_float"] <= 0):
            raise ValueError(f"incomplete or unmatched image request {index}")
        waveform = None
        if bool(a.get("audio_archive_path")) != bool(b.get("audio_archive_path")):
            raise ValueError(f"waveform archive missing on one route for request {index}")
        if a.get("audio_archive_path") and b.get("audio_archive_path"):
            npu_wave_path = archive_path(args.npu_report, a["audio_archive_path"])
            cpu_wave_path = archive_path(args.cpu_report, b["audio_archive_path"])
            if (sha256(npu_wave_path) != a["audio_archive_sha256"]
                    or sha256(cpu_wave_path) != b["audio_archive_sha256"]):
                raise ValueError(f"waveform archive hash mismatch for request {index}")
            npu_wave = np.load(npu_wave_path, allow_pickle=False)
            cpu_wave = np.load(cpu_wave_path, allow_pickle=False)
            if (npu_wave.size != a["audio_samples"] or cpu_wave.size != b["audio_samples"]
                    or not np.isfinite(npu_wave).all() or not np.isfinite(cpu_wave).all()):
                raise ValueError(f"invalid archived waveform for request {index}")
            if npu_wave.shape == cpu_wave.shape:
                difference = np.linalg.norm(npu_wave.astype(np.float64) - cpu_wave.astype(np.float64))
                reference = np.linalg.norm(cpu_wave.astype(np.float64))
                waveform = {
                    "same_shape": True,
                    "exact_equal": bool(np.array_equal(npu_wave, cpu_wave)),
                    "relative_l2_vs_cpu": float(difference / reference),
                    "snr_db_vs_cpu": (None if difference == 0 else
                                       float(20 * np.log10(reference / difference))),
                    "npu_sha256": a["audio_archive_sha256"],
                    "cpu_sha256": b["audio_archive_sha256"],
                }
            else:
                waveform = {"same_shape": False, "npu_sha256": a["audio_archive_sha256"],
                            "cpu_sha256": b["audio_archive_sha256"]}
        comparison.append({
            "index": index,
            "image_sha256": a["image_sha256"],
            "audio_sha256": a.get("audio_sha256"),
            "npu_wall_s": a["wall_s"],
            "cpu_wall_s": b["wall_s"],
            "token_ids_equal": a["token_ids"] == b["token_ids"],
            "npu_text": a["text"],
            "cpu_text": b["text"],
            "npu_audio_samples": a["audio_samples"],
            "cpu_audio_samples": b["audio_samples"],
            "npu_audio_rms": a["audio_rms_float"],
            "cpu_audio_rms": b["audio_rms_float"],
            "npu_max_wsl_used_bytes": a["sampled_memory"].get("max_wsl_used_bytes"),
            "cpu_max_wsl_used_bytes": b["sampled_memory"].get("max_wsl_used_bytes"),
            "npu_max_swap_used_bytes": a["sampled_memory"].get("max_swap_used_bytes"),
            "cpu_max_swap_used_bytes": b["sampled_memory"].get("max_swap_used_bytes"),
            "npu_graph_calls": len(grouped_runs[index]),
            "npu_graph_round_trip_s": sum(row["timing"]["round_trip_s"]
                                          for row in grouped_runs[index]),
            "npu_projection_relative_l2": (completed[index].get("projection_relative_l2")
                                           if completed else None),
            "waveform_comparison": waveform,
        })

    output = {
        "scope": ("serial audio+image" if npu.get("audio_sha256") else "serial image-only")
        + " complete requests in separate Omni CPU and CPU+NPU sessions; no paired speedup or speech-quality claim",
        "npu_report_sha256": sha256(args.npu_report),
        "cpu_report_sha256": sha256(args.cpu_report),
        "npu_events_sha256": sha256(args.npu_events),
        "model": npu["model"],
        "npu_startup_s": npu["startup_s"],
        "cpu_startup_s": cpu["startup_s"],
        "npu_placement": placements[0],
        "npu_worker_peak_rss_bytes": closes[0]["worker_stats"]["peak_rss_bytes"],
        "token_identical_count": sum(row["token_ids_equal"] for row in comparison),
        "complete_request_count": len(comparison),
        "requests": comparison,
        "status": "scoped_complete_suite_pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "complete": len(comparison),
                      "token_identical": output["token_identical_count"]}))


if __name__ == "__main__":
    main()
