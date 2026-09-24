#!/usr/bin/env python3
"""Audit the scoped native-Windows Qwen3.8-27B FP8 recovery and serial profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"


def digest(path: Path) -> dict[str, int | str]:
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_dir
    names = [
        "fp8_copy_integrity.json",
        "fp8_triton_probe.json",
        "fp8_triton_probe.driver.log",
        "qwen38_image_fp8_attempt1.json",
        "qwen38_image_fp8_attempt1.driver.log",
        "qwen38_image_fp8_triton_attempt1.json",
        "qwen38_image_fp8_triton_attempt1.driver.log",
        "qwen38_image_fp8_triton_attempt2.json",
        "qwen38_image_fp8_triton_attempt2.driver.log",
        "qwen38_image_fp8_triton_ntfs_attempt1.json",
        "qwen38_image_fp8_triton_ntfs_attempt1.driver.log",
        "qwen38_fp8_triton_ntfs_profile20.json",
        "qwen38_fp8_triton_ntfs_profile20.driver.log",
        "qwen38_fp8_public_async.json",
        "qwen38_fp8_public_async.driver.log",
        "qwen38_fp8_public_async_abort.json",
        "qwen38_fp8_public_async_abort.driver.log",
        "qwen38_fp8_public_async_abort_timed.json",
        "qwen38_fp8_public_async_abort_timed.driver.log",
        "qwen38_fp8_public_async_abort_terminal.json",
        "qwen38_fp8_public_async_abort_terminal.driver.log",
        "qwen38_fp8_public_async_abort_recovery.json",
        "qwen38_fp8_public_async_abort_recovery.driver.log",
    ]
    paths = {name: root / name for name in names}
    reports = {name: json.loads(paths[name].read_text(encoding="utf-8")) for name in names if name.endswith(".json")}
    logs = {name: paths[name].read_text(encoding="utf-8", errors="replace") for name in names if name.endswith(".log")}
    probe = reports["fp8_triton_probe.json"]
    copy = reports["fp8_copy_integrity.json"]
    if (copy["status"] != "checked" or copy["file_count"] != 81
            or copy["weight_files_matching_manifest"] != 66
            or set(copy["stale_manifest_nonweight_files"]) != {
                "chat_template.jinja", "generation_config.json", "tokenizer_config.json"
            }):
        raise ValueError("native checkpoint copy integrity record is incomplete")
    if probe["status"] != "pass" or probe["capability"] != [12, 0] or probe["value"] != 128:
        raise ValueError("FP8 Triton operation did not pass on SM120")
    if reports["qwen38_image_fp8_attempt1.json"]["status"] != "failed" or "No compiled cutlass_scaled_mm" not in logs["qwen38_image_fp8_attempt1.driver.log"]:
        raise ValueError("default CUTLASS failure is missing")
    if reports["qwen38_image_fp8_triton_attempt1.json"]["status"] != "failed" or "no structured config owner: linear_backend" not in logs["qwen38_image_fp8_triton_attempt1.driver.log"]:
        raise ValueError("Omni config ownership failure is missing")
    if reports["qwen38_image_fp8_triton_attempt2.json"]["status"] != "failed" or "0.33 GiB KV cache is needed" not in logs["qwen38_image_fp8_triton_attempt2.driver.log"]:
        raise ValueError("undersized KV-cache refusal is missing")
    single = reports["qwen38_image_fp8_triton_ntfs_attempt1.json"]
    profile = reports["qwen38_fp8_triton_ntfs_profile20.json"]
    for report in (single, profile):
        cfg = report["config"]
        if (report["status"] != "completed" or report["outputs"] != ["Red"]
                or cfg["linear_backend"] != "triton" or cfg["cpu_offload_gb"] != 12
                or cfg["gpu_memory_utilization"] != 0.8 or cfg["max_model_len"] != 512
                or "Qwen3.8-27B-FP8" not in report["model"]):
            raise ValueError("successful report does not match the bounded FP8 image case")
    for name in ("qwen38_image_fp8_triton_ntfs_attempt1.driver.log", "qwen38_fp8_triton_ntfs_profile20.driver.log"):
        if not all(token in logs[name] for token in (
            "Selected TritonFp8BlockScaledMMKernel",
            "Loading safetensors checkpoint shards: 100% Completed | 66/66",
        )):
            raise ValueError(f"successful log lacks kernel/load/cache evidence: {name}")
        cache_match = re.search(r"GPU KV cache size: ([\d,]+) tokens", logs[name])
        if cache_match is None or int(cache_match.group(1).replace(",", "")) < 512:
            raise ValueError(f"successful log lacks enough KV cache for one request: {name}")
    if profile["text_output"] != ["ready"]:
        raise ValueError("text warmup answer changed")
    public = reports["qwen38_fp8_public_async.json"]
    recovery = reports["qwen38_fp8_public_async_abort_recovery.json"]
    for report in (public, recovery):
        if report["status"] != "completed" or report["entrypoint"] != "AsyncOmni.generate":
            raise ValueError("public AsyncOmni request did not complete")
        cases = {case["kind"]: case for case in report["cases"]}
        if set(cases) != {"text", "image"} or any(
            cases[kind]["outputs"][-1]["text"] != expected
            or ("finished" in cases[kind]["outputs"][-1] and not cases[kind]["outputs"][-1]["finished"])
            for kind, expected in (("text", "ready"), ("image", "Red"))
        ):
            raise ValueError("public text/image final output changed")
    if any(reports[name]["status"] != "failed" for name in (
        "qwen38_fp8_public_async_abort.json",
        "qwen38_fp8_public_async_abort_timed.json",
        "qwen38_fp8_public_async_abort_terminal.json",
    )):
        raise ValueError("retained abort-probe assertion failures are missing")
    abort = recovery["abort_check"]
    if (not abort["task_active_before_abort"] or abort["remaining_request_states"] != 0
            or abort["fresh_outputs"][-1] != "ready"):
        raise ValueError("abort did not clear state and recover")
    terminal = [row for row in abort["emissions"] if row["at_monotonic"] > abort["abort_completed_monotonic"]]
    if len(terminal) != 1 or terminal[0]["finish_reason"] != "abort" or not terminal[0]["finished"]:
        raise ValueError("abort did not emit one terminal marker")
    timing = {}
    for kind, expected in (("image", "Red"), ("text", "ready")):
        rows = [row for row in profile["profile"] if row["kind"] == kind]
        if len(rows) != 20 or {row["index"] for row in rows} != set(range(20)) or any(row["output"] != expected for row in rows):
            raise ValueError(f"{kind} serial profile is incomplete or changed output")
        values = [row["wall_s"] for row in rows]
        timing[kind] = {"n": len(values), "nearest_rank_p50_s": percentile(values, 0.5),
                        "nearest_rank_p95_s": percentile(values, 0.95), "min_s": min(values), "max_s": max(values)}
    result = {
        "status": "audited_scoped_e2e",
        "evidence_depth": "P: complete text and synthetic-image requests on native Windows RTX, plus public AsyncOmni and same-engine abort/recovery; no long context or broad quality claim",
        "artifact": "Qwen/Qwen3.8-27B-FP8",
        "checkpoint_revision": REVISION,
        "host": "Windows 11 build 26200; Ryzen AI 9 HX 370; RTX 5090 Laptop; driver 610.71",
        "runtime": "native vLLM 0.29.0+cu134; PyTorch 2.13.0+cu130; Triton FP8 block linear backend",
        "configuration": profile["config"],
        "warmups": {"image": 1, "text": 1},
        "timing": timing,
        "files": {name: digest(path) for name, path in paths.items()},
        "limits": [
            "The FP8 checkpoint and Triton backend are distinct from the NVFP4 Marlin failure.",
            "The image is a synthetic 96x96 red square and the text prompt is a single-word response; no broad multimodal or long-context quality claim follows.",
            "The 12 GiB host offload and 0.80 GPU-memory cap are declared settings, not measured loading-peak admission or a sampled power profile.",
            "Cancellation used a terminal abort marker and a fresh request in the same engine; server-started cancellation, separate stage restart, concurrency, video, sustained power/thermal behavior and task quality remain unverified.",
        ],
    }
    (root / "fp8_audit_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "timing": timing}))


if __name__ == "__main__":
    main()
