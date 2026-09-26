#!/usr/bin/env python3
"""Audit explicit CPU TTS fallback on an HX370 host with a detected AMD NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path


TALKER_SHA256 = "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb"
TOKENIZER_SHA256 = "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
FIRST_PCM_SHA256 = "c8a49b5c73efd642a1eb04be973d6aefdc8ed9cfd18b34ff67690b8464906bed"
SECOND_PCM_SHA256 = "4edc7a62f749b1fde5ea39bedbc15a35f291ab1db8d24d7b7eb3df7270885e16"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pcm_sha256(path: Path) -> tuple[str, int]:
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 24000):
            raise ValueError(f"{path}: output is not mono PCM16 at 24 kHz")
        frames = source.getnframes()
        return hashlib.sha256(source.readframes(frames)).hexdigest(), frames


def check_plan(report: dict) -> None:
    if report["status"] != "passed" or "Windows" not in report["host_os"]:
        raise ValueError("native Windows complete-request report did not pass")
    inventory = report.get("accelerator_inventory", [])
    if not any(item.get("pci_id") == "VEN_1022&DEV_17F0"
               and item.get("status") == "OK" for item in inventory):
        raise ValueError("healthy HX370 AMD NPU was not detected")
    policy = report.get("placement_policy", {})
    plan = report["execution_plan"]
    if (policy.get("selected_device") != "cpu" or policy.get("npu_execution_claimed") is not False
            or plan["backend"] != "external.qwen_tts.cpu.v1"
            or plan["worker_props"]["placement"] != "cpu"
            or plan["worker_props"]["dtype"] != "bfloat16"):
        raise ValueError("the fallback was not actually CPU BF16")
    hashes = plan["artifact_sha256"]
    if (hashes["talker_sha256"] != TALKER_SHA256
            or hashes["tokenizer_sha256"] != TOKENIZER_SHA256):
        raise ValueError("checkpoint differs from the pinned CPU baseline")
    budget = report["memory_budget"]["demands"]["host_ram"]
    if report["sampled_max_private_bytes"] > budget:
        raise ValueError("sampled process-tree private peak exceeded the reservation")
    if report["ledger_after_shutdown"]["reserved"]["host_ram"] != 0:
        raise ValueError("host RAM reservation remained after shutdown")
    if report["standalone_pcm_parity"] != [True, True]:
        raise ValueError("named output hashes differ from the pinned standalone checkpoint")
    if [row["pcm_sha256"] for row in report["checks"]] != [
        FIRST_PCM_SHA256, SECOND_PCM_SHA256
    ]:
        raise ValueError("named request PCM hashes changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "wav", "abort-report", "abort-wav", "audit"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    run, abort = load(args.report), load(args.abort_report)
    check_plan(run)
    check_plan(abort)
    if run["execution_plan"]["artifact_sha256"] != abort["execution_plan"]["artifact_sha256"]:
        raise ValueError("profile and abort did not use the same artifacts")
    if (run["warmup_count"] != 1 or run["measured_count"] != 3
            or len(run["measured"]) != 3 or not run["all_same_pcm_sha256"]
            or any(row["pcm_sha256"] != FIRST_PCM_SHA256 for row in run["measured"])):
        raise ValueError("three-request serial profile did not reproduce pinned PCM")
    if (abort["warmup_count"] != 0 or abort["measured_count"] != 1
            or abort["measured"][0]["pcm_sha256"] != FIRST_PCM_SHA256):
        raise ValueError("abort fixture did not first complete a known request")
    lifecycle = abort.get("abort_check", {})
    if (lifecycle.get("stale_output") is not False
            or lifecycle.get("worker_exited") is not True
            or lifecycle["ledger_after_abort"]["reserved"]["host_ram"] != 0):
        raise ValueError("abort left a stale output, worker, or reservation")
    for path in (args.wav, args.abort_wav):
        pcm_hash, frames = pcm_sha256(path)
        if pcm_hash != FIRST_PCM_SHA256 or frames != 109440:
            raise ValueError(f"{path}: WAV differs from the pinned first request")
    audit = {
        "scope": "native Windows HX370 NPU-present Qwen3-TTS complete-request CPU fallback; no NPU execution",
        "status": "cpu_fallback_scoped_e2e_pass",
        "raw_sha256": {key: file_sha256(path) for key, path in (
            ("report", args.report), ("wav", args.wav),
            ("abort_report", args.abort_report), ("abort_wav", args.abort_wav))},
        "named_pcm_parity": [True, True],
        "warmups": run["warmup_count"],
        "measured_requests": run["measured_count"],
        "nearest_rank_p50_wall_s": run["nearest_rank_p50_wall_s"],
        "nearest_rank_p95_wall_s": run["nearest_rank_p95_wall_s"],
        "sampled_max_private_bytes": run["sampled_max_private_bytes"],
        "reserved_host_ram_bytes": run["memory_budget"]["demands"]["host_ram"],
        "abort": lifecycle,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in (
        "status", "measured_requests", "nearest_rank_p50_wall_s",
        "nearest_rank_p95_wall_s")}, indent=2))


if __name__ == "__main__":
    main()
