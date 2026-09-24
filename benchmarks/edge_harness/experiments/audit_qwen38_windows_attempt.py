#!/usr/bin/env python3
"""Audit the retained native-Windows Qwen3.8 NVFP4 startup failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT_CAUSE = "CUDA error: the provided PTX was compiled with an unsupported toolchain"
MODEL = "Qwen3.8-27B-NVFP4"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_dir
    attempts = [root / f"qwen38_image_unc_attempt{i}.json" for i in (1, 2)]
    log_path = root / "qwen38_image_unc_attempt2.driver.log"
    hardware_path = root / "hardware_runtime.json"
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in attempts]
    log = log_path.read_text(encoding="utf-8", errors="replace")
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    if any(report["status"] != "failed" or MODEL not in report["model"]
           or report["config"]["max_model_len"] != 512
           or report["config"]["gpu_memory_utilization"] != 0.85
           or report["config"]["cpu_offload_gb"] != 2 for report in reports):
        raise ValueError("retained reports do not describe the bounded Windows NVFP4 attempt")
    if not all(token in log for token in (
        "Loading safetensors checkpoint shards: 100% Completed | 3/3",
        "process_weights_after_loading",
        "prepare_fp4_layer_for_marlin",
        "marlin_permute_scales",
        ROOT_CAUSE,
    )):
        raise ValueError("captured worker log lacks the post-load Marlin/PTX root cause")
    if ("610.71" not in hardware["nvidia_smi_header"]
            or "CUDA UMD Version: 13.3" not in hardware["nvidia_smi_header"]
            or "+cu134" not in hardware["installed_vllm_wheel"]
            or hardware["nvcc"] != "13.4.59"):
        raise ValueError("recorded driver/toolkit versions differ from the observed combination")
    files = [*attempts, log_path, hardware_path]
    result = {
        "status": "audited_failure",
        "evidence_depth": "backend load failure; no model inference or E2E output",
        "artifact": MODEL,
        "checkpoint_revision": "dbb8f445b3145f8a4c18ddc769f032d57d32867c",
        "root_cause": ROOT_CAUSE,
        "failure_location": "vLLM NVFP4 Marlin scale permutation during post-load weight processing",
        "driver": "NVIDIA 610.71 / CUDA UMD 13.3",
        "vllm_wheel": hardware["installed_vllm_wheel"],
        "inference_completed": False,
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
        "limits": [
            "The error is specific to the tested native-Windows wheel, driver and NVFP4 artifact.",
            "The previously verified WSL text/image route is a different execution environment.",
            "The driver/toolchain relationship is inferred from versions and NVIDIA's PTX compatibility guidance; no upgraded-driver control has run.",
        ],
    }
    (root / "audit_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "root_cause": result["root_cause"]}))


if __name__ == "__main__":
    main()
