#!/usr/bin/env python3
"""Audit a bounded Spark RTX decoder plus HX370 NPU output-head profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_dir
    report_path = root / "report.json"
    worker_path = root / "npu_worker.json"
    spec_path = root / "npu_spec.json"
    trace_path = root / "npu_raw_profile.json"
    driver_path = root / "driver.log"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if (report["status"] != "scoped_joint_complete_quality_pass"
            or report["model"] != str(args.model_dir.resolve())
            or report["model_index_sha256"] != sha256(args.model_dir / "model.safetensors.index.json")
            or report["graph_sha256"] != spec["graph_sha256"]
            or report["graph_sha256"] != sha256(Path(spec["graph"]))
            or report["as_run_spec_sha256"] != sha256(spec_path)
            or report["npu_worker_report_sha256"] != sha256(worker_path)
            or report["cuda_plan"]["selected_device"]["device_id"] != "cuda:0"
            or not report["cuda_plan"]["admitted"] or not report["npu_plan"]["admitted"]
            or report["npu_plan"]["budget_bytes"] < worker["worker_stats"]["peak_rss_bytes"]
            or report["capacity"]["npu_plan_shared_ram_bytes"]
            + report["capacity"]["cuda_host_process_reserve_bytes"]
            > min(report["capacity"]["windows_available_before_bytes"],
                  report["capacity"]["wsl_available_before_bytes"])):
        raise ValueError("model, artifact, plan, memory admission or report status changed")
    if (worker["placement"]["ep"] != "vitisai"
            or worker["placement"]["node_counts"].get("vitisai", 0) < 1
            or worker["input_layout"] != "post_final_norm"
            or worker["cpu_refine_top_k"] != 64
            or worker["sampling_contract"] != "greedy-only"
            or worker["worker_stats"]["runs"] != report["npu_worker_runs"]
            or len(worker["calls"]) != report["npu_worker_runs"]
            or len(worker["refinement_calls"]) != report["npu_refinement_calls"]
            or any(row["refine_device"] != "cuda:0" for row in worker["refinement_calls"])):
        raise ValueError("NPU execution or CUDA refinement contract failed")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    providers = [row.get("args", {}).get("provider") for row in trace
                 if row.get("cat") == "Node"]
    if providers.count("vitisai") < 1:
        raise ValueError("retained raw trace lacks an NPU node")
    if (len(report["phases"]) != 2
            or [phase["mode"] for phase in report["phases"]] != ["cuda", "cuda_npu"]):
        raise ValueError("RTX-only and joint phases are missing")
    expected_count = report["warmups_per_phase"] + report["measured_per_phase"]
    reference = report["phases"][0]["rows"][0]["token_ids"]
    for phase in report["phases"]:
        rows = phase["rows"]
        measured = [row for row in rows if not row["warmup"]]
        if (len(rows) != expected_count or len(measured) != report["measured_per_phase"]
                or any(row["token_ids"] != reference for row in rows)
                or any(row["output_tokens"] != report["max_new_tokens"] for row in rows)
                or not phase["all_tokens_match_cuda_reference"]
                or phase["nearest_rank_p50_wall_s"] != rank([row["wall_s"] for row in measured], .5)
                or phase["nearest_rank_p95_wall_s"] != rank([row["wall_s"] for row in measured], .95)):
            raise ValueError(f"{phase['mode']}: complete request or token check failed")
    output_tokens = expected_count * report["max_new_tokens"]
    if report["npu_worker_runs"] - output_tokens != report["npu_non_request_calls"]:
        raise ValueError("NPU startup/request call accounting failed")
    driver = driver_path.read_text(encoding="utf-8", errors="replace")
    if driver.count("Stage 0 replica 0 shut down") != 2:
        raise ValueError("both stages did not shut down")
    cpu_path = root / "cpu_regression_report.json"
    cpu_worker_path = root / "cpu_regression_worker.json"
    cpu_trace_path = root / "cpu_regression_raw_profile.json"
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    cpu_worker = json.loads(cpu_worker_path.read_text(encoding="utf-8"))
    cpu_trace = json.loads(cpu_trace_path.read_text(encoding="utf-8"))
    cpu_providers = [row.get("args", {}).get("provider") for row in cpu_trace
                     if row.get("cat") == "Node"]
    if (cpu["status"] != "completed" or not cpu["all_runs_match_reference"]
            or cpu["npu_worker_runs"] != report["max_new_tokens"]
            or cpu_worker["placement"]["ep"] != "vitisai"
            or cpu_providers.count("vitisai") < 1
            or len(cpu_worker["refinement_calls"]) != report["max_new_tokens"]
            or any(row["refine_device"] != "cpu" for row in cpu_worker["refinement_calls"])):
        raise ValueError("existing CPU+NPU route regressed after CUDA support")
    p50_cuda, p50_joint = [phase["nearest_rank_p50_wall_s"] for phase in report["phases"]]
    output = {
        "status": "scoped_joint_complete_quality_pass_no_benefit_on_tested_workload",
        "report_sha256": sha256(report_path),
        "worker_sha256": sha256(worker_path),
        "spec_sha256": sha256(spec_path),
        "driver_sha256": sha256(driver_path),
        "raw_trace_sha256": sha256(trace_path),
        "raw_node_providers": {name: providers.count(name) for name in sorted(set(providers))},
        "cpu_regression_report_sha256": sha256(cpu_path),
        "cpu_regression_worker_sha256": sha256(cpu_worker_path),
        "cpu_regression_raw_trace_sha256": sha256(cpu_trace_path),
        "checkpoint_index_sha256": report["model_index_sha256"],
        "graph_sha256": report["graph_sha256"],
        "npu_worker_peak_rss_bytes": worker["worker_stats"]["peak_rss_bytes"],
        "npu_budget_bytes": report["npu_plan"]["budget_bytes"],
        "per_phase_measured_requests": report["measured_per_phase"],
        "per_request_output_tokens": report["max_new_tokens"],
        "reference_token_ids_sha256": report["phases"][0]["rows"][0]["token_ids_sha256"],
        "cuda_p50_s": p50_cuda, "joint_p50_s": p50_joint,
        "joint_to_cuda_p50_ratio": p50_joint / p50_cuda,
        "npu_total_calls": report["npu_worker_runs"],
        "npu_extra_calls_vs_output_tokens": report["npu_non_request_calls"],
        "limits": "One greedy prompt, serial fixed 64-token output, CUDA then joint order; no broad quality, non-greedy, long-context, paired-order speedup, fault recovery or sustained power qualification.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "p50_ratio": output["joint_to_cuda_p50_ratio"]}))


if __name__ == "__main__":
    main()
