#!/usr/bin/env python3
"""Audit same-input CPU and CPU+NPU MiniCPM-o image/audio suites and placement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    closes = [row for row in events if row["phase"] == "close"]
    if (len(placements) != 1 or len(runs) != request_count or len(closes) != 1
            or placements[0]["ep"] != "vitisai" or placements[0]["target_nodes"] < 1
            or closes[0]["calls"] != request_count):
        raise ValueError("missing verified NPU placement/calls or clean close")

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
            "npu_graph_round_trip_s": runs[index]["timing"]["round_trip_s"],
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
