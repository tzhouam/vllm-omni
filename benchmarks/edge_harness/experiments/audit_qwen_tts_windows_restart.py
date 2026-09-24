#!/usr/bin/env python3
"""Audit the pinned native-Windows Qwen3-TTS in-flight cancel/restart probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path


ARTIFACT_HASHES = {
    "server_sha256": "999c6af252077195adb723b96da9ee5d3cd291dc7c7a4b32dc493ae148035529",
    "talker_sha256": "5227dcbc4df7c5533341d111cc469fa491a48e722b23dd10f553181b52dff2d9",
    "codec_sha256": "70dc95dbfdd9aa5d9d406236ff771d061bf17b0cda02a72513953355606e719b",
    "punc_sha256": "faf4a43e3135bc307a66194685af00f756e6f4c28c7d9e2dd8f3517cddca5c45",
}
PREFILL = "qwen3_tts[customvoice]: prefill role="
PCM_SHA256 = "57f60228e2b456f6dfefc7e49df862c7bff6ba7cf39231ef1824f615948a1539"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_ledger(value: dict) -> bool:
    return (value["reserved"] == {"host_ram": 0}
            and value["owners"] == [] and value["quarantined"] == [])


def audit(root: Path) -> dict:
    report_path = root / "tts_restart_report.json"
    first_log = root / "tts_restart_server.log"
    second_log = root / "tts_restart_server_restart.log"
    wav_path = root / "tts_restart.wav"
    hardware_path = root / "hardware.json"
    run = json.loads(report_path.read_text(encoding="utf-8"))
    before = run["execution_plan"]
    after = run["restart_check"]["fresh_execution_plan"]
    abort = run["abort_check"]
    restart = run["restart_check"]
    if run["status"] != "passed" or run["measured_count"] != 1:
        raise ValueError("TTS probe did not pass the declared single-request recovery scope")
    if any(plan["artifact_sha256"] != ARTIFACT_HASHES for plan in (before, after)):
        raise ValueError("TTS artifact hashes changed between sessions")
    if any(plan["placement"] != "Radeon Vulkan0 talker/codec + CPU FP32 code predictor"
           or plan["expected_gpu_name"] != "AMD Radeon(TM) 890M Graphics"
           or plan["ggml_vk_visible_devices"] != "1" for plan in (before, after)):
        raise ValueError("TTS execution placement changed")
    if any(row["pcm_sha256"] != PCM_SHA256 or row["frames"] != 61440
           for row in (run["checks"][0], run["measured"][0], restart["fresh_request"])):
        raise ValueError("TTS fresh request differs from the pinned complete-WAV output")
    if (not restart["same_pcm_as_before_abort"] or restart["late_output"]
            or abort["stale_output"] or not abort["worker_exited"]
            or abort["prefills_after_abort_request_start"] <= abort["prefills_before_abort_request"]):
        raise ValueError("TTS cancellation or fresh-stage recovery failed")
    if not all(clean_ledger(value) for value in (
        abort["ledger_after_abort"], restart["first_runtime_ledger_after_shutdown"],
        run["ledger_after_shutdown"],
    )):
        raise ValueError("TTS RAM reservation or quarantine remained")
    if (before["worker_generation"] == after["worker_generation"]
            or restart["fresh_request"]["stage_event"]["worker_generation"]
            != after["worker_generation"]):
        raise ValueError("TTS fresh request reused the cancelled worker")
    logs = [path.read_text(encoding="utf-8", errors="replace") for path in (first_log, second_log)]
    if (logs[0].count(PREFILL) < abort["prefills_after_abort_request_start"]
            or logs[1].count(PREFILL) < 1
            or any("desc=AMD Radeon(TM) 890M Graphics" not in log
                   or "code_pred CPU-pinned" not in log
                   or "codec: GPU default - loading weights onto Vulkan0" not in log
                   for log in logs)):
        raise ValueError("TTS raw server logs do not verify prefill and hybrid placement")
    with wave.open(str(wav_path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 24000:
            raise ValueError("saved WAV format differs from the declared output")
        if wav.getnframes() != 61440 or hashlib.sha256(wav.readframes(61440)).hexdigest() != PCM_SHA256:
            raise ValueError("saved WAV payload differs from the stage-reported PCM")
    files = [report_path, first_log, second_log, wav_path, hardware_path]
    return {
        "status": "passed",
        "scope": "one native Windows hybrid TTS cancellation after backend prefill and fresh StageRuntime request",
        "old_generation": before["worker_generation"],
        "new_generation": after["worker_generation"],
        "first_server_prefills": logs[0].count(PREFILL),
        "fresh_server_prefills": logs[1].count(PREFILL),
        "fresh_pcm_sha256": PCM_SHA256,
        "ledger_cleared": True,
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
        "limits": [
            "The stage returns a complete WAV, not incremental playable audio.",
            "A fresh StageRuntime starts after cancellation; no same-session state migration is claimed.",
            "One recovery case is not concurrency or sustained reliability qualification.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.evidence_dir)
    (args.evidence_dir / "audit_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "fresh_pcm_sha256": result["fresh_pcm_sha256"]}))


if __name__ == "__main__":
    main()
