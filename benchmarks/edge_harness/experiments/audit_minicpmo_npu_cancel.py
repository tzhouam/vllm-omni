#!/usr/bin/env python3
"""Audit a distinct-image MiniCPM-o CPU+NPU abort and fresh request."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in args.events.read_text(encoding="utf-8").splitlines()]
    required = {
        "fresh_events", "late_events", "fresh_image_sha256", "image_sha256",
        "cancel_before", "abort_terminal", "late_audio_samples",
        "fresh_audio_samples",
    }
    if missing := required - report.keys():
        raise ValueError(f"abort/recovery report lacks {sorted(missing)}")
    placement = [row["placement"] for row in events if row["phase"] == "placement"]
    runs = [row for row in events if row["phase"] == "run"]
    requests = [row for row in events if row["phase"] == "request_complete"]
    closes = [row for row in events if row["phase"] == "close"]
    fresh = report["fresh_events"]
    late = report["late_events"]
    if (report["status"] != "cancel_then_fresh_complete_pass"
            or report["image_sha256"] == report["fresh_image_sha256"]
            or report["cancel_before"]["event"]["finished"]
            or not report["cancel_before"]["event"]["text"]
            or report["abort_terminal"] != "stream_ended"
            or report["late_audio_samples"] != 0
            or any(row["audio_samples"] for row in late)
            or not fresh or not fresh[-1]["finished"]
            or not any(row["text"] for row in fresh)
            or sum(row["audio_samples"] for row in fresh) != report["fresh_audio_samples"]
            or report["fresh_audio_samples"] <= 0
            or any(row["audio_finite"] is False for row in fresh)
            or len(placement) != 1 or placement[0]["ep"] != "vitisai"
            or placement[0]["node_counts"].get("vitisai") != 1
            or [row["request"] for row in requests] != [1, 2]
            or [row["valid_tokens"] for row in runs] != [1014, 1024, 1024, 1024, 33]
            or [row["request"] for row in runs] != [1, 2, 2, 2, 2]
            or len(closes) != 1 or closes[0]["calls"] != 5
            or any(row["projection_relative_l2"] > .01 for row in requests)):
        raise ValueError("abort/recovery or NPU execution evidence failed")
    output = {
        "status": "scoped_abort_then_fresh_npu_request_pass",
        "report_sha256": sha256(args.report),
        "events_sha256": sha256(args.events),
        "abort_ack_s": report["abort_ack_s"],
        "late_events": len(late),
        "late_audio_samples": report["late_audio_samples"],
        "fresh_wall_s": report["fresh_wall_s"],
        "fresh_audio_samples": report["fresh_audio_samples"],
        "npu_graph_calls": len(runs),
        "npu_request_numbers": [row["request"] for row in requests],
        "projection_relative_l2": [row["projection_relative_l2"] for row in requests],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
