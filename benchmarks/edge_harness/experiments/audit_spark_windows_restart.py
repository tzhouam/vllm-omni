#!/usr/bin/env python3
"""Audit native-Windows Spark cancellation/restart reports and server logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MODEL_SHA256 = "902bde2522394954ac17821b3e5fd0df02defbc6944f122253f2580acf0503f4"
SERVER_SHA256 = "9ffc5919acb4cb43c7be5f3053b59014cce70a87d3a801fcad49c4f459984f52"
TASK_START = re.compile(r"slot launch_slot_: id\s+\d+ \| task \d+ \| processing task")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_ledger(value: dict) -> bool:
    return (value["reserved"] == {"host_ram": 0}
            and value["owners"] == [] and value["quarantined"] == [])


def audit(root: Path, route: str) -> dict:
    prefix = "spark_rtx" if route == "rtx" else f"{route}_v4"
    report_path = root / f"{prefix}_report.json"
    first_log = root / f"{prefix}_server.log"
    second_log = root / f"{prefix}_server_restart.log"
    run = json.loads(report_path.read_text(encoding="utf-8"))
    before = run["execution_plan"]
    after = run["restart_check"]["fresh_execution_plan"]
    abort = run["abort_check"]
    restart = run["restart_check"]
    expected_device = {"cpu": "cpu", "radeon": "Vulkan1", "rtx": "Vulkan0"}[route]
    if run["status"] != "passed" or run["device"] != expected_device:
        raise ValueError(f"{route}: runtime outcome or device differs")
    if route == "rtx" and (run["warmup_count"] != 1 or run["measured_count"] != 20):
        raise ValueError("RTX route did not complete its declared 20-request profile")
    if any(plan["model_sha256"] != MODEL_SHA256 or plan["server_sha256"] != SERVER_SHA256
           for plan in (before, after)):
        raise ValueError(f"{route}: model or server binary changed between sessions")
    expected_offload = ["0", "29"] if route == "cpu" else ["29", "29"]
    expected_buffers = {"cpu": [], "radeon": ["Vulkan1"], "rtx": ["Vulkan0"]}[route]
    if any(plan["offloaded_layers"] != expected_offload
           or plan["model_buffer_devices"] != expected_buffers for plan in (before, after)):
        raise ValueError(f"{route}: execution placement differs from the intended hardware")
    if route == "cpu" and any(plan["all_model_buffers"] != ["CPU_Mapped", "CPU_REPACK"]
                              for plan in (before, after)):
        raise ValueError("CPU route had a non-CPU model buffer")
    if route == "radeon" and any(plan["expected_device_name"] != "AMD Radeon(TM) 890M Graphics"
                                 for plan in (before, after)):
        raise ValueError("Radeon route did not pin the 890M by name")
    if route == "rtx" and any(plan["expected_device_name"] != "NVIDIA GeForce RTX 5090 Laptop GPU"
                              for plan in (before, after)):
        raise ValueError("RTX route did not pin the 5090 by name")
    if ([row["answer"] for row in run["checks"]] != ["Paris", "1511"]
            or not all(row["correct"] for row in run["checks"] + run["measured"])
            or restart["fresh_request"]["answer"] != "Paris"
            or not restart["fresh_request"]["correct"]):
        raise ValueError(f"{route}: named complete-request checks failed")
    if route == "rtx" and (not run["context_check"]["rejected"]
                           or run["context_check"]["recovery"]["answer"] != "Paris"):
        raise ValueError("RTX route did not reject overflow and recover")
    if (abort["abort_case"] != "inventory_120"
            or abort["server_tasks_after_abort_request_start"]
            <= abort["server_tasks_before_abort_request"]
            or abort["stale_output"] or not abort["worker_exited"]
            or restart["late_output"]):
        raise ValueError(f"{route}: cancellation was not verified after server task start")
    if not all(clean_ledger(value) for value in (
        abort["ledger_after_abort"], restart["first_runtime_ledger_after_shutdown"],
        run["ledger_after_shutdown"],
    )):
        raise ValueError(f"{route}: reservation or quarantine remained")
    if (before["worker_generation"] == after["worker_generation"]
            or restart["fresh_request"]["stage_event"]["worker_generation"]
            != after["worker_generation"]):
        raise ValueError(f"{route}: resumed request did not use a fresh worker")
    first_starts = len(TASK_START.findall(first_log.read_text(encoding="utf-8", errors="replace")))
    second_starts = len(TASK_START.findall(second_log.read_text(encoding="utf-8", errors="replace")))
    if first_starts < abort["server_tasks_after_abort_request_start"] or second_starts < 1:
        raise ValueError(f"{route}: server logs do not contain the recorded tasks")
    if route == "rtx":
        for path in (first_log, second_log):
            log = path.read_text(encoding="utf-8", errors="replace")
            if ("using device Vulkan0 (NVIDIA GeForce RTX 5090 Laptop GPU)" not in log
                    or "offloaded 29/29 layers to GPU" not in log
                    or "Vulkan0 model buffer size" not in log):
                raise ValueError("RTX raw server log does not verify GPU placement")
        public_path = root / "spark_rtx_public_report.json"
        public_log_path = root / "spark_rtx_public_server.log"
        public = json.loads(public_path.read_text(encoding="utf-8"))
        if (public["status"] != "passed" or public["device"] != "Vulkan0"
                or len(public["outputs"]) != 1
                or public["outputs"][0]["text"].strip() != "Paris"
                or public["outputs"][0]["error"] is not None):
            raise ValueError("RTX public AsyncOmni request did not pass")
        public_log = public_log_path.read_text(encoding="utf-8", errors="replace")
        if ("using device Vulkan0 (NVIDIA GeForce RTX 5090 Laptop GPU)" not in public_log
                or "offloaded 29/29 layers to GPU" not in public_log):
            raise ValueError("RTX public request did not use the RTX")
    files = [report_path, first_log, second_log]
    if route == "rtx":
        files += [public_path, public_log_path, root / "hardware.json"]
    return {
        "status": "passed",
        "device": run["device"],
        "old_generation": before["worker_generation"],
        "new_generation": after["worker_generation"],
        "first_server_task_starts": first_starts,
        "second_server_task_starts": second_starts,
        "started_before_cancel": True,
        "fresh_answer": restart["fresh_request"]["answer"],
        "ledger_cleared": True,
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in files},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--routes", nargs="+", choices=("cpu", "radeon", "rtx"),
                        default=("cpu", "radeon"))
    args = parser.parse_args()
    result = {
        "scope": "one in-flight cancel and fresh-stage restart per selected native Windows Spark placement",
        "checkpoint": "Spark-X2.5-1.7B Q4_K_M",
        "model_sha256": MODEL_SHA256,
        "llama_server_sha256": SERVER_SHA256,
        "routes": {route: audit(args.evidence_dir, route) for route in args.routes},
        "limits": [
            "A fresh StageRuntime is created after cancellation; no same-session state migration is claimed.",
            "The backend returns complete requests rather than incremental token events.",
            "This is one restart per placement, not concurrent or sustained recovery qualification.",
        ],
    }
    path = args.evidence_dir / "audit_report.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({route: row["status"] for route, row in result["routes"].items()}, indent=2))


if __name__ == "__main__":
    main()
