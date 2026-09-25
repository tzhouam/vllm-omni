#!/usr/bin/env python3
"""Audit same-fixture InternVLA CPU, AMD NPU, Radeon and joint profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("cpu", "npu", "radeon", "joint", "joint_after_radeon"):
        parser.add_argument(f"--{label}-report", type=Path, required=True)
        parser.add_argument(f"--{label}-actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {label: (getattr(args, f"{label}_report"), getattr(args, f"{label}_actions"))
             for label in ("cpu", "npu", "radeon", "joint", "joint_after_radeon")}
    expected_placements = {"cpu": "cpu", "npu": "amd-npu-conv13",
                           "radeon": "radeon-cosmos",
                           "joint": "amd-npu-radeon-cosmos",
                           "joint_after_radeon": "amd-npu-radeon-cosmos"}
    reports = {}
    actions = {}
    for label, (report_path, action_path) in paths.items():
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        action = np.load(action_path, allow_pickle=False)
        if (report["status"] != "passed" or report["placement"] != expected_placements[label]
                or report["repeats"] != 5 or report["warmups"] != 1
                or len(report["measured"]) != 5 or action.shape != (1, 50, 32)
                or action.dtype != np.float32 or not np.isfinite(action).all()):
            raise ValueError(f"{label} profile or action contract differs")
        action_hash = hashlib.sha256(action.tobytes()).hexdigest()
        if (report["unique_action_hashes"] != [action_hash]
                or report["nearest_rank_p50_wall_s"] != rank(
                    [item["wall_s"] for item in report["measured"]], .5)
                or report["nearest_rank_p95_wall_s"] != rank(
                    [item["wall_s"] for item in report["measured"]], .95)):
            raise ValueError(f"{label} measured actions or percentiles differ")
        if report["ledger_after_shutdown"]["reserved"]["host_ram"] != 0:
            raise ValueError(f"{label} did not release its host-RAM reservation")
        reports[label], actions[label] = report, action
    shared_keys = ("model", "model_config", "train_config", "stats", "cosmos_encoder",
                   "cosmos_decoder", "processor_tokenizer", "processor_config")
    for label in ("npu", "radeon", "joint", "joint_after_radeon"):
        if (reports[label]["fixture_sha256"] != reports["cpu"]["fixture_sha256"]
                or reports[label]["fixture_manifest"] != reports["cpu"]["fixture_manifest"]
                or any(reports[label]["artifact_sha256"][key] !=
                       reports["cpu"]["artifact_sha256"][key] for key in shared_keys)):
            raise ValueError(f"{label} did not use the same input and base model")
        abort = reports[label].get("abort_check", {})
        if abort.get("stale_output") is not False or abort.get("ledger", {}).get("reserved", {}).get("host_ram") != 0:
            raise ValueError(f"{label} cancellation did not release cleanly")
    for label in ("npu", "joint", "joint_after_radeon"):
        props = reports[label]["execution_plan_start"]["worker_props"]
        if props["npu_load"]["warmup_npu_node_events"] < 1:
            raise ValueError(f"{label} has no NPU node event")
    radeon_props = reports["radeon"]["execution_plan_start"]["worker_props"]
    radeon_load = radeon_props["external_load"]
    if (radeon_load["device_name"] != "AMD Radeon(TM) 890M Graphics"
            or radeon_load["node_counts"].get("dml", 0) < 1
            or radeon_load["placement_granularity"] != "output_device"):
        raise ValueError("Radeon profile lacks its declared output-device placement")
    joint_props = reports["joint"]["execution_plan_start"]["worker_props"]
    dml = joint_props["external_load"]
    if dml["node_counts"].get("DmlExecutionProvider", 0) < 1 or dml["device_id_requested"] != 1:
        raise ValueError("joint profile has no requested adapter-1 DirectML node")
    repeat_dml = reports["joint_after_radeon"]["execution_plan_start"]["worker_props"]["external_load"]
    if (repeat_dml["node_counts"] != dml["node_counts"]
            or repeat_dml["device_id_requested"] != dml["device_id_requested"]
            or reports["joint_after_radeon"]["unique_action_hashes"] != reports["joint"]["unique_action_hashes"]):
        raise ValueError("reverse-order joint placement or action differs")
    manifest = reports["cpu"]["fixture_manifest"]
    if manifest["observation_kind"] != "recorded-camera-with-synthetic-checkpoint-mean-state":
        raise ValueError("fixture is not the declared out-of-domain stress case")

    summary = {
        "scope": "same simulated-camera fixture and synthetic state; whole policy only, no robot-task quality",
        "fixture_sha256": reports["cpu"]["fixture_sha256"],
        "source": manifest["source"],
        "source_task": manifest["source_task"],
        "checkpoint_task": manifest["task"],
        "profiles": {},
        "pairwise_action_relative_l2": {
            "npu_vs_cpu": relative_l2(actions["cpu"], actions["npu"]),
            "radeon_vs_cpu": relative_l2(actions["cpu"], actions["radeon"]),
            "joint_vs_cpu": relative_l2(actions["cpu"], actions["joint"]),
            "joint_vs_npu": relative_l2(actions["npu"], actions["joint"]),
            "joint_vs_radeon": relative_l2(actions["radeon"], actions["joint"]),
            "joint_repeat_vs_joint": relative_l2(actions["joint"], actions["joint_after_radeon"]),
        },
    }
    for label, (report_path, action_path) in paths.items():
        report = reports[label]
        summary["profiles"][label] = {
            "report_sha256": sha256(report_path),
            "actions_sha256": sha256(action_path),
            "action_tensor_sha256": report["unique_action_hashes"][0],
            "startup_s": report["startup_s"],
            "n": len(report["measured"]),
            "p50_wall_s": report["nearest_rank_p50_wall_s"],
            "p95_wall_s": report["nearest_rank_p95_wall_s"],
            "loaded_rss_bytes": report["execution_plan_start"]["loaded_rss_bytes"],
        }
    summary["npu_warmup_node_events"] = {
        label: reports[label]["execution_plan_start"]["worker_props"]["npu_load"]["warmup_npu_node_events"]
        for label in ("npu", "joint", "joint_after_radeon")}
    summary["joint_suffix_provider_node_counts"] = dml["node_counts"]
    summary["joint_order_check"] = {
        "first_joint_p50_wall_s": reports["joint"]["nearest_rank_p50_wall_s"],
        "joint_after_radeon_p50_wall_s": reports["joint_after_radeon"]["nearest_rank_p50_wall_s"],
        "same_action_tensor": True,
        "interpretation": "run-order variation prevents a reliable speedup conclusion",
    }
    summary["radeon_full_encoder_placement"] = {
        "device_name": radeon_load["device_name"],
        "placement_granularity": radeon_load["placement_granularity"],
        "node_counts": radeon_load["node_counts"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
