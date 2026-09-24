#!/usr/bin/env python3
"""Audit complete-WAV RTX hybrid TTS evidence and Radeon route regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path


ARTIFACT_HASHES = {
    "server_sha256": "999c6af252077195adb723b96da9ee5d3cd291dc7c7a4b32dc493ae148035529",
    "talker_sha256": "5227dcbc4df7c5533341d111cc469fa491a48e722b23dd10f553181b52dff2d9",
    "codec_sha256": "70dc95dbfdd9aa5d9d406236ff771d061bf17b0cda02a72513953355606e719b",
    "punc_sha256": "faf4a43e3135bc307a66194685af00f756e6f4c28c7d9e2dd8f3517cddca5c45",
}
RTX_PCM = "d3d071531e2574b933c8acf18e2a0344c8961493bd17c63662307270a99d265c"
RADEON_PCM = "57f60228e2b456f6dfefc7e49df862c7bff6ba7cf39231ef1824f615948a1539"
PREFILL = "qwen3_tts[customvoice]: prefill role="


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_ledger(value: dict, *, rtx: bool) -> bool:
    empty = {"host_ram": 0, "gpu_vram": 0} if rtx else {"host_ram": 0}
    return value["reserved"] == empty and value["owners"] == [] and value["quarantined"] == []


def check_wav(path: Path, frames: int, pcm_hash: str) -> None:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) != (1, 2, 24000, frames):
            raise ValueError(f"{path.name}: unexpected WAV format or frame count")
        if hashlib.sha256(wav.readframes(frames)).hexdigest() != pcm_hash:
            raise ValueError(f"{path.name}: PCM differs from the stage output")


def check_log(path: Path, device: str, started: int = 1) -> int:
    log = path.read_text(encoding="utf-8", errors="replace")
    if (f"0 = {device}" not in log or "using preferred GPU backend: Vulkan0" not in log
            or "code_pred CPU-pinned" not in log
            or "codec: GPU default - loading weights onto Vulkan0" not in log
            or "falling back to CPU" in log):
        raise ValueError(f"{path.name}: hybrid device placement not verified")
    count = log.count(PREFILL)
    if count < started:
        raise ValueError(f"{path.name}: expected TTS prefill absent")
    return count


def audit(root: Path) -> dict:
    profile = json.loads((root / "tts_rtx_profile20_report.json").read_text(encoding="utf-8"))
    smoke = json.loads((root / "tts_rtx_smoke_report.json").read_text(encoding="utf-8"))
    postguard = json.loads((root / "tts_rtx_postguard_report.json").read_text(encoding="utf-8"))
    radeon = json.loads((root / "tts_radeon_regression_report.json").read_text(encoding="utf-8"))
    public = json.loads((root / "tts_rtx_public_report.json").read_text(encoding="utf-8"))
    asr = json.loads((root / "tts_rtx_smoke_asr.json").read_text(encoding="utf-8"))
    refused = json.loads((root / "tts_rtx_refused_3gib.json").read_text(encoding="utf-8"))
    hardware = json.loads((root / "hardware.json").read_text(encoding="utf-8"))
    if any(report["status"] != "passed" for report in (profile, smoke, postguard, radeon, public, refused)):
        raise ValueError("one of the TTS runs failed")
    for report, device, gpu, pcm in (
        (profile, "NVIDIA GeForce RTX 5090 Laptop GPU", True, RTX_PCM),
        (smoke, "NVIDIA GeForce RTX 5090 Laptop GPU", True, RTX_PCM),
        (postguard, "NVIDIA GeForce RTX 5090 Laptop GPU", True, RTX_PCM),
        (radeon, "AMD Radeon(TM) 890M Graphics", False, RADEON_PCM),
    ):
        plan = report["execution_plan"]
        if (plan["artifact_sha256"] != ARTIFACT_HASHES or plan["expected_gpu_name"] != device
                or plan["ggml_vk_visible_devices"] != ("0" if gpu else "1")
                or ("gpu_vram" in plan["reserved_bytes"]) != gpu):
            raise ValueError("TTS artifact, GPU identity or memory pools changed")
        if (report["checks"][0]["pcm_sha256"] != pcm
                or report["measured"][0]["pcm_sha256"] != pcm
                or not clean_ledger(report["ledger_after_shutdown"], rtx=gpu)):
            raise ValueError("TTS complete WAV or release changed")
    if (profile["warmup_count"] != 1 or profile["measured_count"] != 20
            or len(profile["measured"]) != 20 or not profile["all_same_pcm_sha256"]
            or any(row["pcm_sha256"] != RTX_PCM or row["frames"] != 65280
                   for row in profile["measured"])):
        raise ValueError("RTX serial profile did not return 20 identical complete WAVs")
    walls = sorted(row["wall_s"] for row in profile["measured"])
    if (profile["nearest_rank_p50_wall_s"] != walls[math.ceil(20 * .5) - 1]
            or profile["nearest_rank_p95_wall_s"] != walls[math.ceil(20 * .95) - 1]):
        raise ValueError("RTX p50/p95 do not match the retained 20 samples")
    for report, gpu in ((profile, True), (radeon, False)):
        abort = report["abort_check"]
        restart = report["restart_check"]
        old_plan, fresh_plan = report["execution_plan"], restart["fresh_execution_plan"]
        if (abort["prefills_after_abort_request_start"] <= abort["prefills_before_abort_request"]
                or abort["stale_output"] or not abort["worker_exited"]
                or restart["late_output"] or not restart["same_pcm_as_before_abort"]
                or old_plan["worker_generation"] == fresh_plan["worker_generation"]
                or fresh_plan["artifact_sha256"] != ARTIFACT_HASHES
                or fresh_plan["expected_gpu_name"] != old_plan["expected_gpu_name"]
                or fresh_plan["reserved_bytes"] != old_plan["reserved_bytes"]
                or restart["fresh_request"]["stage_event"]["worker_generation"] != fresh_plan["worker_generation"]
                or not all(clean_ledger(value, rtx=gpu) for value in (
                    abort["ledger_after_abort"], restart["first_runtime_ledger_after_shutdown"]
                ))):
            raise ValueError("in-flight cancellation or fresh-stage recovery failed")
    if (len(public["outputs"]) != 1 or public["outputs"][0]["pcm_sha256"] != RTX_PCM
            or public["outputs"][0]["frames"] != 65280 or public["outputs"][0]["sample_rate"] != 24000):
        raise ValueError("public AsyncOmni audio differs from the RTX stage output")
    if (asr["word_error_rate"] != 0.0 or asr["transcript"] != "Hello from the local computer."
            or asr["wav_sha256"] != sha256(root / "tts_rtx_smoke.wav")):
        raise ValueError("independent ASR proxy or WAV identity failed")
    if (refused["server_log_created"] or "GPU-VRAM reservation" not in refused["error"]
            or not clean_ledger(refused["ledger_after_refusal"], rtx=True)
            or (root / "tts_rtx_refused_should_not_exist.log").exists()):
        raise ValueError("undersized GPU reservation launched or retained a worker")
    if not any(gpu["Name"] == "NVIDIA GeForce RTX 5090 Laptop GPU" for gpu in hardware["gpus"]):
        raise ValueError("RTX hardware identity absent")
    counts = {
        "smoke": check_log(root / "tts_rtx_smoke_server.log", "NVIDIA GeForce RTX 5090 Laptop GPU", 3),
        "postguard": check_log(root / "tts_rtx_postguard_server.log", "NVIDIA GeForce RTX 5090 Laptop GPU", 3),
        "profile": check_log(root / "tts_rtx_profile20_server.log", "NVIDIA GeForce RTX 5090 Laptop GPU", 24),
        "restart": check_log(root / "tts_rtx_profile20_server_restart.log", "NVIDIA GeForce RTX 5090 Laptop GPU"),
        "public": check_log(root / "tts_rtx_public_server.log", "NVIDIA GeForce RTX 5090 Laptop GPU"),
        "radeon": check_log(root / "tts_radeon_regression_server.log", "AMD Radeon(TM) 890M Graphics"),
        "radeon_restart": check_log(root / "tts_radeon_regression_server_restart.log", "AMD Radeon(TM) 890M Graphics"),
    }
    check_wav(root / "tts_rtx_smoke.wav", 65280, RTX_PCM)
    check_wav(root / "tts_rtx_postguard.wav", 65280, RTX_PCM)
    check_wav(root / "tts_rtx_profile20.wav", 65280, RTX_PCM)
    check_wav(root / "tts_radeon_regression.wav", 61440, RADEON_PCM)
    files = [path for path in root.iterdir() if path.is_file() and path.name not in ("README.md", "audit_report.json")]
    return {
        "status": "passed",
        "scope": "RTX complete-request TTS, 20 serial outputs, public API, in-flight cancel/restart, GPU admission refusal and Radeon regression",
        "profile_nearest_rank_p50_wall_s": profile["nearest_rank_p50_wall_s"],
        "profile_nearest_rank_p95_wall_s": profile["nearest_rank_p95_wall_s"],
        "rtx_pcm_sha256": RTX_PCM,
        "radeon_pcm_sha256": RADEON_PCM,
        "server_prefill_counts": counts,
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
        "limits": [
            "ASR exact transcription is an intelligibility proxy, not speaker or perceptual quality qualification.",
            "The backend emits complete WAVs and has no playable chunk stream.",
            "Declared GPU-VRAM admission is not a measured device-memory loading peak.",
            "One fresh-stage restart is not same-session continuation or sustained recovery qualification.",
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
    print(json.dumps({"status": result["status"], "profile_p50": result["profile_nearest_rank_p50_wall_s"]}))


if __name__ == "__main__":
    main()
