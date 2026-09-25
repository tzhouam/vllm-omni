#!/usr/bin/env python3
"""Audit public Qwen3-TTS Windows CPU abort and fresh-stage restart evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_PCM = "c8a49b5c73efd642a1eb04be973d6aefdc8ed9cfd18b34ff67690b8464906bed"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def worker_props(raw: bytes) -> dict:
    lines = [line for line in raw.splitlines() if b"qwen-tts-cpu-worker ready port=" in line]
    if len(lines) != 1:
        raise ValueError("expected one ready CPU worker per log")
    return json.loads(lines[0].split(b" props=", 1)[1].decode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "first-worker-log", "restart-worker-log", "driver-log", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    first_log = args.first_worker_log.read_bytes()
    restart_log = args.restart_worker_log.read_bytes()
    driver = args.driver_log.read_text(encoding="utf-8", errors="replace")
    first = report["outputs"]
    recovered = report["recovery"]["outputs"]
    abort = report["abort"]
    if (report["status"] != "passed"
            or report["entrypoint"] != "AsyncOmni.generate"
            or report["pipeline"] != "qwen3_tts_cpu_whole"
            or report["recovery"]["mode"] != "fresh_stage_after_inflight_abort"
            or len(first) != 1 or len(recovered) != 1
            or any(row["frames"] != 109440 or row["sample_rate"] != 24000
                   or row["pcm_sha256"] != EXPECTED_PCM
                   or not row["stage_event"]["terminal"]
                   or row["stage_event"]["kind"] != "audio"
                   for row in first + recovered)
            or first[0]["stage_event"]["worker_generation"]
            == recovered[0]["stage_event"]["worker_generation"]
            or abort["worker_started"] is not True
            or abort["worker_start_marker"].encode() not in first_log
            or len(abort["outputs_after_start"]) != 1
            or abort["outputs_after_start"][0]["request_id"] != "public-cpu-tts-abort"
            or abort["outputs_after_start"][0]["has_audio"]
            or first_log.count(b"tts request duration_s=") != 1
            or restart_log.count(b"tts request duration_s=") != 1
            or driver.count("[StagePool] Stage 0 replica 0 shut down") != 2):
        raise ValueError("CPU TTS public abort/restart behavior did not pass")
    for props in (worker_props(first_log), worker_props(restart_log)):
        if (props["placement"] != "cpu" or props["torch"] != "2.13.0+cu130"
                or props["transformers"] != "4.57.3" or props["qwen_tts"] != "0.1.1"
                or props["dtype"] != "bfloat16" or props["attention"] != "sdpa"
                or props["sample_rate"] != 24000 or props["max_new_tokens"] != 64):
            raise ValueError("CPU TTS worker placement or runtime drifted")
    result = {
        "status": "scoped_cpu_tts_fresh_stage_recovery_pass",
        "report_sha256": sha256(args.report),
        "first_worker_log_sha256": sha256(args.first_worker_log),
        "restart_worker_log_sha256": sha256(args.restart_worker_log),
        "driver_log_sha256": sha256(args.driver_log),
        "pcm_sha256": EXPECTED_PCM,
        "abort_ack_s": abort["ack_s"],
        "abort_delivered_audio": 0,
        "first_startup_s": report["startup_s"],
        "restart_startup_s": report["recovery"]["startup_s"],
        "first_request_wall_s": report["request_wall_s"],
        "restart_request_wall_s": report["recovery"]["request_wall_s"],
        "worker_generations": [
            first[0]["stage_event"]["worker_generation"],
            recovered[0]["stage_event"]["worker_generation"],
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
