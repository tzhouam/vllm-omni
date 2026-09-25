#!/usr/bin/env python3
"""Audit joint CPU+NPU+Radeon InternVLA abort and fresh-stage recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def worker_props(raw: bytes) -> dict:
    lines = [line for line in raw.splitlines() if b"internvla-worker ready port=" in line]
    if len(lines) != 1:
        raise ValueError("expected exactly one ready worker in each log")
    return json.loads(lines[0].split(b" props=", 1)[1].decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "reference-public", "first-worker-log",
                 "restart-worker-log", "driver-log", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_public.read_text(encoding="utf-8"))
    first_log = args.first_worker_log.read_bytes()
    second_log = args.restart_worker_log.read_bytes()
    driver = args.driver_log.read_text(encoding="utf-8", errors="replace")
    first_props, second_props = worker_props(first_log), worker_props(second_log)
    first, second = report["outputs"], report["recovery"]["outputs"]
    abort = report["abort"]
    if (report["status"] != "passed"
            or report["entrypoint"] != "AsyncOmni.generate"
            or report["placement"] != "amd-npu-radeon-cosmos"
            or report["recovery"]["mode"] != "fresh_stage_after_inflight_abort"
            or len(first) != 1 or len(second) != 1
            or first[0]["action_shape"] != [1, 50, 32]
            or second[0]["action_shape"] != [1, 50, 32]
            or not first[0]["finite"] or not second[0]["finite"]
            or first[0]["action_sha256"] != second[0]["action_sha256"]
            or first[0]["action_sha256"] != reference["outputs"][0]["action_sha256"]
            or first[0]["stage_event"]["worker_generation"]
            == second[0]["stage_event"]["worker_generation"]
            or not first[0]["stage_event"]["terminal"]
            or not second[0]["stage_event"]["terminal"]
            or first[0]["metadata"]["control_ready"] is not False
            or second[0]["metadata"]["control_ready"] is not False
            or abort["worker_started"] is not True
            or abort["internal_request_id"].encode() not in first_log
            or any(row["has_actions"] for row in abort["outputs_after_start"])
            or b"internvla-worker request-start id=" not in second_log
            or driver.count("[StagePool] Stage 0 replica 0 shut down") < 2):
        raise ValueError("joint policy abort/restart behavior did not pass")
    for props in (first_props, second_props):
        external = props.get("external_load") or {}
        npu = props.get("npu_load") or {}
        if (props.get("placement") != "amd-npu-radeon-cosmos"
                or props.get("runtime_mode") != "real_checkpoint_loaded"
                or props.get("policy_device") != "cpu"
                or props.get("cosmos_dtype") != "float32"
                or external.get("ep") != "dml"
                or external.get("device_id_requested") != 1
                or external.get("node_counts", {}).get("DmlExecutionProvider", 0) < 1
                or npu.get("provider") != "vitisai"
                or npu.get("warmup_npu_node_events", 0) < 1):
            raise ValueError("restarted joint policy lost verified accelerator placement")
    output = {
        "status": "scoped_joint_abort_fresh_stage_pass",
        "report_sha256": sha256(args.report),
        "first_worker_log_sha256": sha256(args.first_worker_log),
        "restart_worker_log_sha256": sha256(args.restart_worker_log),
        "driver_log_sha256": sha256(args.driver_log),
        "reference_public_sha256": sha256(args.reference_public),
        "action_sha256": first[0]["action_sha256"],
        "abort_ack_s": abort["ack_s"],
        "abort_delivered_actions": 0,
        "first_startup_s": report["startup_s"],
        "restart_startup_s": report["recovery"]["startup_s"],
        "first_request_wall_s": report["request_wall_s"],
        "restart_request_wall_s": report["recovery"]["wall_s"],
        "worker_generations": [
            first[0]["stage_event"]["worker_generation"],
            second[0]["stage_event"]["worker_generation"],
        ],
        "npu_warmup_nodes": [
            first_props["npu_load"]["warmup_npu_node_events"],
            second_props["npu_load"]["warmup_npu_node_events"],
        ],
        "dml_suffix_nodes": [
            first_props["external_load"]["node_counts"]["DmlExecutionProvider"],
            second_props["external_load"]["node_counts"]["DmlExecutionProvider"],
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
