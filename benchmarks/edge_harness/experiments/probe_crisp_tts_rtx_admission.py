#!/usr/bin/env python3
"""Verify that an undersized discrete-GPU TTS reservation fails before load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "server_bin", "server_sha256", "talker_file", "talker_sha256", "codec_file",
        "codec_sha256", "punc_file", "punc_sha256", "server_log", "output_report",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args()

    from vllm_omni.engine.backends.crisp_tts import CrispTTSStageClient
    from vllm_omni.engine.resource_ledger import ResourceLedger, ResourceUnavailable

    log_path = Path(args.server_log)
    if log_path.exists():
        raise FileExistsError("refusal probe needs a fresh server-log path")
    ledger = ResourceLedger({"host_ram": 16 << 30, "gpu_vram": 16 << 30})
    reservation = ledger.reserve("undersized-rtx-tts", {"host_ram": 8 << 30, "gpu_vram": 3 << 30})
    backend = {
        "server_bin": args.server_bin,
        "server_sha256": args.server_sha256,
        "talker_file": args.talker_file,
        "talker_sha256": args.talker_sha256,
        "codec_file": args.codec_file,
        "codec_sha256": args.codec_sha256,
        "punc_file": args.punc_file,
        "punc_sha256": args.punc_sha256,
        "log_file": str(log_path),
        "expected_gpu_name": "NVIDIA GeForce RTX 5090 Laptop GPU",
        "ggml_vk_visible_devices": "0",
        "memory_overhead_bytes": 5 << 30,
        "gpu_memory_pool": "gpu_vram",
        "gpu_memory_overhead_bytes": 2 << 30,
    }
    error = None
    try:
        CrispTTSStageClient(SimpleNamespace(stage_id=0), backend, ledger, reservation)
    except ResourceUnavailable as exc:
        error = f"{type(exc).__name__}: {exc}"
    if error is None:
        raise RuntimeError("undersized GPU-VRAM reservation was admitted")
    snapshot = ledger.snapshot()
    if snapshot["reserved"] != {"host_ram": 0, "gpu_vram": 0} or snapshot["quarantined"] or log_path.exists():
        raise RuntimeError("GPU-VRAM refusal left a reservation or launched the server")
    report = {
        "status": "passed",
        "scope": "RTX TTS 3 GiB GPU-VRAM demand is refused before server launch",
        "artifact_sha256": {name: getattr(args, name) for name in (
            "server_sha256", "talker_sha256", "codec_sha256", "punc_sha256"
        )},
        "declared_gpu_reservation_bytes": 3 << 30,
        "declared_gpu_overhead_bytes": 2 << 30,
        "error": error,
        "server_log_created": False,
        "ledger_after_refusal": snapshot,
    }
    destination = Path(args.output_report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "error": report["error"]}))


if __name__ == "__main__":
    main()
