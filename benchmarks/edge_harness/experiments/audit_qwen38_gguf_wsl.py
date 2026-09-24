#!/usr/bin/env python3
"""Audit pinned Qwen3.8 GGUF Omni WSL CPU complete-request profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MODEL_SHA = "c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747"
MMPROJ_SHA = "2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb"
SERVER_SHA = "0d6bb6ade8da8f3231917ce23261da5331333df0007dc4bda74c00e0774e092c"
REVISION = "efbb3b1f70a21d97fd4495240648405f7228554f"


def read(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def digest(path: Path) -> dict[str, str | int]:
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_dir
    for name in ("smoke.json", "smoke2.json"):
        failed = read(root, name)
        if failed["status"] != "failed" or "available native host RAM" not in failed["error"]:
            raise ValueError(f"{name}: low-available-RAM pre-admission refusal missing")
    repack = read(root, "smoke3.json")
    if repack["status"] != "failed" or "RSS exceeds stage reservation" not in repack["error"]:
        raise ValueError("repacked model did not fail the loaded-RSS admission gate")
    smoke = read(root, "smoke4.json")
    profile = read(root, "profile20.json")
    if smoke["status"] != "completed" or profile["status"] != "completed":
        raise ValueError("WSL no-repack smoke/profile did not complete")
    plan = profile["execution_plan"]
    if (plan["backend"] != "external.llamacpp.multimodal.v1"
            or plan["model_sha256"] != MODEL_SHA
            or plan["mmproj_sha256"] != MMPROJ_SHA
            or plan["server_sha256"] != SERVER_SHA
            or plan["requested_device"] != "cpu"
            or plan["offloaded_layers"] is not None
            or not plan["cpu_only_layout"]
            or plan["model_buffer_devices"]
            or plan["all_model_buffers"] != ["CPU_Mapped"]
            or plan["vision_backends"] != ["CPU"]
            or not plan["disable_repack"]
            or plan["loaded_rss_bytes"] > profile["reserved_host_ram_bytes"]
            or profile["available_host_ram_before_bytes"] < profile["reserved_host_ram_bytes"]):
        raise ValueError("WSL artifact, explicit no-repack CPU placement or admission changed")
    ledger = profile["ledger_after_shutdown"]
    if ledger["reserved"].get("host_ram") != 0 or ledger["owners"] or ledger["quarantined"]:
        raise ValueError("WSL stage did not release its reservation")
    events = [*profile["checks"]]
    timing = {}
    for kind, expected in (("text", "ready"), ("image", "Red")):
        rows = profile[f"{kind}_measured"]
        if (len(rows) != 20 or profile[f"{kind}_warmup"]["answer"] != expected
                or any(row["kind"] != kind or row["answer"] != expected for row in rows)):
            raise ValueError(f"WSL {kind} sample count/output changed")
        events.extend([profile[f"{kind}_warmup"], *rows])
        walls = [row["wall_s"] for row in rows]
        p50, p95 = rank(walls, 0.5), rank(walls, 0.95)
        if (profile[f"nearest_rank_p50_{kind}_wall_s"] != p50
                or profile[f"nearest_rank_p95_{kind}_wall_s"] != p95):
            raise ValueError(f"WSL {kind} percentile arithmetic changed")
        timing[kind] = {"n": 20, "p50_s": p50, "p95_s": p95,
                        "min_s": min(walls), "max_s": max(walls)}
    if (sorted(row["stage_event"]["epoch"] for row in events) != list(range(1, 45))
            or any(not row["stage_event"]["terminal"] or row["stage_event"]["seq"] != 1
                   or row["stage_event"]["worker_generation"] != plan["worker_generation"]
                   for row in events)):
        raise ValueError("WSL terminal event epochs missing or duplicated")
    log = (root / "profile20.server.log").read_text(encoding="utf-8", errors="replace")
    if ("warning: no usable GPU found" not in log
            or "CPU_Mapped model buffer size" not in log
            or "CPU_REPACK model buffer size" in log
            or "CLIP using CPU backend" not in log):
        raise ValueError("WSL server log does not confirm CPU/no-repack/vision placement")
    files = sorted(path for path in root.iterdir() if path.suffix in (".json", ".log")
                   and path.name != "audit_report.json")
    audit = {
        "status": "audited_scoped_e2e",
        "evidence_depth": "P: WSL CPU Omni complete text and synthetic-image requests",
        "checkpoint": "ggml-org/Qwen3.8-27B-GGUF",
        "revision": REVISION,
        "quantization": "Q4_K_M language model + Q8_0 vision projector",
        "model_sha256": MODEL_SHA,
        "mmproj_sha256": MMPROJ_SHA,
        "server_sha256": SERVER_SHA,
        "host": "Ryzen AI 9 HX 370, Ubuntu 26.04 on WSL2, 30.91 GiB WSL RAM quota",
        "runtime": "Omni checkout through Linux vLLM 0.29; llama.cpp b10849 CPU-only build, --no-repack",
        "scope": "2048 context, 128 max new tokens, concurrency one, one 96x96 red PNG and one short text instruction",
        "startup_s": profile["startup_s"],
        "reserved_host_ram_bytes": profile["reserved_host_ram_bytes"],
        "loaded_rss_bytes": plan["loaded_rss_bytes"],
        "timing": timing,
        "files": {path.name: digest(path) for path in files},
        "limits": [
            "The prompts produce two completion tokens; long-context, 128-token output and broad quality remain unqualified.",
            "No public AsyncOmni, cancellation/restart, sampled loading/request memory peak, concurrency or sustained power/thermal profile was run.",
            "The WSL quota, no-repack option and Linux llama.cpp b10849 differ from the native Windows CPU setup; latencies are not a paired OS comparison.",
        ],
    }
    (root / "audit_report.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "timing": timing}))


if __name__ == "__main__":
    main()
