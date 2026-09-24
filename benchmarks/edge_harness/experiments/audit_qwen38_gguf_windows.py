#!/usr/bin/env python3
"""Audit pinned Qwen3.8 GGUF Omni CPU/Radeon text and image profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MODEL_SHA = "c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747"
MMPROJ_SHA = "2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb"
SERVER_SHA = "9ffc5919acb4cb43c7be5f3053b59014cce70a87d3a801fcad49c4f459984f52"
REVISION = "efbb3b1f70a21d97fd4495240648405f7228554f"


def rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def read(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def digest(path: Path) -> dict[str, str | int]:
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def check(root: Path, device: str) -> dict:
    report = read(root, "profile20.json")
    smoke = read(root, "attempt2.json")
    placement_verify = read(root, "placement_verify.json")
    if any(item["status"] != "completed" for item in (report, smoke, placement_verify)):
        raise ValueError(f"{device}: smoke/profile did not complete")
    plan = report["execution_plan"]
    latest_plan = placement_verify["execution_plan"]
    if (latest_plan["vision_backends"] != (["CPU"] if device == "cpu" else ["Vulkan1"])
            or latest_plan["model_sha256"] != MODEL_SHA
            or latest_plan["mmproj_sha256"] != MMPROJ_SHA
            or any(row["answer"] != expected for row, expected in zip(
                placement_verify["checks"], ("ready", "Red"), strict=True
            ))):
        raise ValueError(f"{device}: current projector placement enforcement failed")
    if (plan["backend"] != "external.llamacpp.multimodal.v1"
            or plan["requested_device"] != device
            or plan["model_sha256"] != MODEL_SHA
            or plan["mmproj_sha256"] != MMPROJ_SHA
            or plan["server_sha256"] != SERVER_SHA
            or plan["request_capacity"] != 1
            or plan["loaded_rss_bytes"] > report["reserved_host_ram_bytes"]
            or report["available_host_ram_before_bytes"] < report["reserved_host_ram_bytes"]):
        raise ValueError(f"{device}: artifact, placement or admission record changed")
    if device == "cpu":
        if plan["offloaded_layers"] != ["0", "65"] or plan["model_buffer_devices"] or any(
            not item.startswith("CPU") for item in plan["all_model_buffers"]
        ):
            raise ValueError("CPU model buffers or layer placement changed")
    elif (plan["expected_device_name"] != "AMD Radeon(TM) 890M Graphics"
          or plan["offloaded_layers"] != ["65", "65"]
          or plan["model_buffer_devices"] != ["Vulkan1"]):
        raise ValueError("Radeon model was not fully offloaded to Vulkan1")
    ledger = report["ledger_after_shutdown"]
    if ledger["reserved"].get("host_ram") != 0 or ledger["owners"] or ledger["quarantined"]:
        raise ValueError(f"{device}: stage reservation was not released")
    events = []
    timing = {}
    for kind, expected in (("text", "ready"), ("image", "Red")):
        rows = report[f"{kind}_measured"]
        if len(rows) != 20 or report[f"{kind}_warmup"]["answer"] != expected:
            raise ValueError(f"{device}: {kind} warmup or sample count changed")
        if any(row["answer"] != expected or row["kind"] != kind for row in rows):
            raise ValueError(f"{device}: {kind} output changed")
        events.extend([report[f"{kind}_warmup"], *rows])
        walls = [row["wall_s"] for row in rows]
        p50, p95 = rank(walls, 0.5), rank(walls, 0.95)
        if (report[f"nearest_rank_p50_{kind}_wall_s"] != p50
                or report[f"nearest_rank_p95_{kind}_wall_s"] != p95):
            raise ValueError(f"{device}: {kind} percentile arithmetic changed")
        timing[kind] = {"n": 20, "p50_s": p50, "p95_s": p95,
                        "min_s": min(walls), "max_s": max(walls)}
    events = [*report["checks"], *events]
    if any(not row["stage_event"]["terminal"]
           or row["stage_event"]["seq"] != 1
           or row["stage_event"]["worker_generation"] != plan["worker_generation"]
           for row in events):
        raise ValueError(f"{device}: terminal event contract changed")
    if sorted(row["stage_event"]["epoch"] for row in events) != list(range(1, 45)):
        raise ValueError(f"{device}: request epochs are missing or duplicated")
    log = (root / "profile20.server.log").read_text(encoding="utf-8", errors="replace")
    if device == "cpu":
        if "offloaded 0/65 layers to GPU" not in log or "CLIP using CPU backend" not in log:
            raise ValueError("CPU server placement line missing")
    elif ("using device Vulkan1 (AMD Radeon(TM) 890M Graphics)" not in log
          or "offloaded 65/65 layers to GPU" not in log
          or "CLIP using Vulkan1 backend" not in log):
        raise ValueError("Radeon server placement line missing")
    return {"startup_s": report["startup_s"], "reserved_host_ram_bytes": report["reserved_host_ram_bytes"],
            "loaded_rss_bytes": plan["loaded_rss_bytes"], "timing": timing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_root
    cpu = root / "qwen38_gguf_windows_cpu"
    radeon = root / "qwen38_gguf_windows_radeon"
    failed = read(cpu, "attempt1.json")
    if failed["status"] != "failed" or "RSS exceeds stage reservation" not in failed["error"]:
        raise ValueError("undersized CPU reservation failure is missing")
    mistyped = read(radeon, "attempt1.json")
    if mistyped["status"] != "failed" or "differs from the declared artifact hash" not in mistyped["error"]:
        raise ValueError("mistyped Radeon digest refusal is missing")
    regression = read(cpu, "spark_text_regression.json")
    if (regression["status"] != "passed"
            or regression["execution_plan"]["backend"] != "external.llamacpp.text.v1"
            or [(row["case"], row["answer"]) for row in regression["checks"]]
            != [("short", "Paris"), ("inventory_120", "1511")]):
        raise ValueError("existing Spark llama.cpp text route regressed")
    summary = {"cpu": check(cpu, "cpu"), "radeon": check(radeon, "Vulkan1")}
    files = sorted(path for folder in (cpu, radeon) for path in folder.iterdir()
                   if path.suffix in (".json", ".log") and path.name != "audit_report.json")
    audit = {
        "status": "audited_scoped_e2e",
        "evidence_depth": "P: native Windows Omni complete text and synthetic-image requests on CPU and Radeon 890M",
        "checkpoint": "ggml-org/Qwen3.8-27B-GGUF",
        "revision": REVISION,
        "quantization": "Q4_K_M language model + Q8_0 vision projector",
        "model_sha256": MODEL_SHA,
        "mmproj_sha256": MMPROJ_SHA,
        "server_sha256": SERVER_SHA,
        "host": "Ryzen AI 9 HX 370, Windows 11 build 26200, Radeon 890M; AC power",
        "runtime": "Omni checkout through native vLLM 0.29 StageRuntime; llama.cpp b11124 Vulkan build",
        "scope": "2048 context, 128 max new tokens, concurrency one, one 96x96 red PNG and one short text instruction",
        "results": summary,
        "files": {str(path.relative_to(root)): digest(path) for path in files},
        "limits": [
            "The prompts produce two completion tokens; no long-context, 128-token output or broad quality claim follows.",
            "No public AsyncOmni, cancellation/restart, sampled loading/runtime memory peak, power, thermal, concurrency or sustained profile was run for this GGUF route.",
            "Windows CPU and Radeon use different actual placement, so timings do not establish an isolated accelerator speedup.",
        ],
    }
    (root / "qwen38_gguf_audit_report.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "results": summary}))


if __name__ == "__main__":
    main()
