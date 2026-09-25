#!/usr/bin/env python3
"""Audit real-camera A2D whole-policy probes without claiming task accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1e-12))


def check_report(root: Path, name: str, manifest: dict, expected_placement: str,
                 action_mean: np.ndarray, action_std: np.ndarray,
                 raw_state: np.ndarray, reference_actions: np.ndarray) -> tuple[dict, np.ndarray]:
    report_path = root / f"{name}_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    actions_path = root / f"{name}_actions.npy"
    physical_path = root / f"{name}_physical_actions.npy"
    actions = np.load(actions_path, allow_pickle=False)
    physical = np.load(physical_path, allow_pickle=False)
    measured = report["measured"]
    plan = report["execution_plan_end"]
    props = plan["worker_props"]
    if (report["status"] != "passed" or report["placement"] != expected_placement
            or report["fixture_sha256"]["input_fixture"] != manifest["fixture_sha256"]
            or report["artifact_sha256"]["model"] != manifest["checkpoint_model_sha256"]
            or report["artifact_sha256"]["stats"] != manifest["checkpoint_stats_sha256"]
            or report["budget"]["demands"]["host_ram"] != (16 << 30)
            or report["ledger_after_shutdown"]["reserved"]["host_ram"] != 0
            or report["ledger_after_shutdown"]["quarantined"]
            or len(measured) != 1 or report["warmups"] != 0
            or actions.shape != (1, 50, 32) or actions.dtype != np.float32
            or physical.shape != (1, 50, 16) or physical.dtype != np.float32
            or not np.isfinite(actions).all() or not np.isfinite(physical).all()
            or props["runtime_mode"] != "real_checkpoint_loaded"
            or props["policy_device"] != ("cuda" if expected_placement == "cuda" else "cpu")):
        raise ValueError(f"{name}: request, artifact, action or admission contract failed")
    event = measured[0]["stage_event"]
    metadata = measured[0]["action_metadata"]
    if (not event["terminal"] or event["kind"] != "action" or event["seq"] != 1
            or len(event["buffers"]) != 2 or metadata["control_ready"]
            or metadata["physical_action_shape"] != [1, 50, 16]
            or measured[0]["action_sha256"] != hashlib.sha256(actions.tobytes()).hexdigest()
            or measured[0]["physical_action_sha256"] != hashlib.sha256(physical.tobytes()).hexdigest()):
        raise ValueError(f"{name}: terminal event or action digest failed")
    decoded = np.ascontiguousarray(actions[:, :, :16] * action_std + action_mean,
                                   dtype=np.float32)
    decoded[:, :, :14] += raw_state[:, None, :14]
    if not np.array_equal(decoded, physical):
        raise ValueError(f"{name}: physical actions differ from independently decoded A2D fields")
    comparison = report["single_sample_reference_action_comparison"]
    if (comparison["reference_shape"] != [1, 50, 16]
            or abs(comparison["relative_l2"] - relative_l2(physical, reference_actions)) > 1e-6):
        raise ValueError(f"{name}: dataset reference-action comparison changed")
    if expected_placement in {"amd-npu-conv13", "amd-npu-radeon-cosmos"}:
        npu = props.get("npu_load") or {}
        if npu.get("provider") != "vitisai" or npu.get("warmup_npu_node_events", 0) < 1:
            raise ValueError(f"{name}: VitisAI placement is unverified")
        if expected_placement == "amd-npu-radeon-cosmos":
            profiles = list(root.glob("joint_worker_npu_profile_*.json"))
            if len(profiles) != 1:
                raise ValueError("joint: expected one retained NPU profile")
            profile = json.loads(profiles[0].read_text(encoding="utf-8"))
            providers = [event.get("args", {}).get("provider") for event in profile
                         if event.get("cat") == "Node"]
            if providers.count("vitisai") < 1:
                raise ValueError("joint: retained profile has no VitisAI node")
    if expected_placement == "amd-npu-radeon-cosmos":
        dml = props.get("external_load") or {}
        if (dml.get("node_counts", {}).get("DmlExecutionProvider", 0) < 1
                or dml.get("device_id_requested") != 1):
            raise ValueError("joint: Radeon DirectML placement is unverified")
    if expected_placement == "radeon-cosmos":
        dml = props.get("external_load") or {}
        if "Radeon" not in str(dml.get("device_name", "")):
            raise ValueError("radeon: selected output device is unverified")
    if expected_placement == "cuda" and "5090" not in str(props.get("cuda_device_name", "")):
        raise ValueError("cuda: actual device is unverified")
    result = {
        "name": name,
        "placement": expected_placement,
        "report_sha256": sha256(report_path),
        "actions_sha256": sha256(actions_path),
        "physical_actions_sha256": sha256(physical_path),
        "request_wall_s": measured[0]["wall_s"],
        "reference_physical_relative_l2": comparison["relative_l2"],
        "action_sha256": measured[0]["action_sha256"],
        "physical_action_sha256": measured[0]["physical_action_sha256"],
    }
    if expected_placement == "amd-npu-radeon-cosmos":
        result["npu_profile_sha256"] = sha256(profiles[0])
        result["npu_profile_node_providers"] = {
            provider: providers.count(provider) for provider in sorted(set(providers))
        }
    return result, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--include-radeon", action="store_true")
    parser.add_argument("--include-cuda", action="store_true")
    parser.add_argument("--include-windows-cuda", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_dir
    manifest_path = root / "observation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture_path = root / "observation.npz"
    train = json.loads((args.model_dir / "train_config.json").read_text(encoding="utf-8"))
    if (manifest["source"] != "pepijn223/task_374@dade034a5780c23de5ab98fd236fee99f0758021"
            or manifest["sample_index"] != 15 or manifest["episode_index"] != 0
            or manifest["fixture_sha256"] != sha256(fixture_path)
            or manifest["checkpoint_model_sha256"] != sha256(args.model_dir / "model.safetensors")
            or manifest["checkpoint_train_config_sha256"] != sha256(args.model_dir / "train_config.json")
            or train["dataset"]["repo_id"] == manifest["source"].split("@")[0]
            or "Sort laundry" not in manifest["task"]):
        raise ValueError("real-camera fixture pin or out-of-task distinction failed")
    with np.load(fixture_path, allow_pickle=False) as fixture:
        raw_state = fixture["state_raw"]
        reference_actions = fixture["reference_actions"]
        camera_change = {f"image{i}": float(np.mean(np.abs(
            fixture[f"image{i}"][:, 0] - fixture[f"image{i}"][:, 1]))) for i in range(3)}
    if any(value <= 0 for value in camera_change.values()):
        raise ValueError("real camera histories do not contain two distinct frames")
    stats = json.loads((args.model_dir / "stats.json").read_text(encoding="utf-8"))["a2d"]
    action_keys = ("actions.joint.position", "actions.effector.position")
    action_mean = np.asarray(sum((stats[key]["mean"] for key in action_keys), []), dtype=np.float32)
    action_std = np.asarray(sum((stats[key]["std"] for key in action_keys), []), dtype=np.float32)
    rows = []
    tensors = {}
    names = [("cpu", "cpu"), ("windows_cpu", "cpu"),
             ("npu", "amd-npu-conv13"), ("joint", "amd-npu-radeon-cosmos")]
    if args.include_radeon:
        names.append(("radeon", "radeon-cosmos"))
    if args.include_cuda:
        names.append(("cuda", "cuda"))
    if args.include_windows_cuda:
        names.append(("windows_cuda", "cuda"))
    for name, placement in names:
        row, actions = check_report(root, name, manifest, placement,
                                    action_mean, action_std, raw_state,
                                    reference_actions)
        rows.append(row)
        tensors[name] = actions
    native_cpu = tensors["windows_cpu"]
    for row in rows:
        row["relative_l2_vs_windows_cpu_actions"] = relative_l2(
            tensors[row["name"]], native_cpu)
    source_files = {}
    if args.dataset_dir:
        paths = ["meta/info.json", "meta/stats.json", "meta/tasks.parquet",
                 "meta/episodes/chunk-000/file-000.parquet",
                 "data/chunk-000/file-000.parquet"]
        paths += [f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
                  for camera in ("head", "hand_left", "hand_right")]
        source_files = {path: sha256(args.dataset_dir / path) for path in paths}
        if (source_files["meta/info.json"] != manifest["dataset_info_sha256"]
                or source_files["data/chunk-000/file-000.parquet"]
                != manifest["dataset_data_sha256"]):
            raise ValueError("downloaded A2D dataset files differ from the fixture")
    output = {
        "status": "real_camera_execution_passed_task_accuracy_unqualified",
        "fixture_sha256": sha256(fixture_path),
        "manifest_sha256": sha256(manifest_path),
        "source": manifest["source"],
        "source_task": manifest["task"],
        "checkpoint_task": train["dataset"],
        "camera_frame_mean_absolute_change": camera_change,
        "source_files_sha256": source_files,
        "placements": rows,
        "limits": "One out-of-task A2D real observation, one complete request per placement; no Place_Markpen task-quality, paired latency, action units or robot-control claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "placements": [r["name"] for r in rows]}))


if __name__ == "__main__":
    main()
