#!/usr/bin/env python3
"""Audit four serial Spark CPU/NPU phases and the native ORT traces."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_dir
    paired_path = root / "paired_report.json"
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    names = ["01_cpu", "02_npu", "03_npu", "04_cpu"]
    if (paired["status"] != "completed"
            or [p["name"] for p in paired["phases"]] != names
            or paired["phase_order"] != ["cpu", "npu", "npu", "cpu"]
            or paired["warmups_per_phase"] != 1
            or paired["measured_requests_per_phase"] != 20
            or paired["max_new_tokens"] != 64):
        raise ValueError("paired workload or phase order changed")
    expected = paired["expected_token_ids_sha256"]
    rows = []
    for name, phase in zip(names, paired["phases"]):
        path = root / f"{name}_report.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (phase["exit_code"] != 0 or phase["report_sha256"] != sha256(path)
                or phase["driver_log_sha256"] != sha256(root / f"{name}_driver.log")
                or raw["warmup_count"] != 1 or raw["measured_count"] != 20
                or raw["max_new_tokens"] != 64):
            raise ValueError(f"{name}: raw report/log or workload changed")
        warmup = [r for r in raw["runs"] if r["kind"] == "warmup"]
        measured = [r for r in raw["runs"] if r["kind"] == "measured"]
        if (len(warmup) != 1 or len(measured) != 20
                or any(r["token_ids_sha256"] != expected or r["output_tokens"] != 64
                       or not math.isfinite(r["wall_s"]) or r["wall_s"] <= 0
                       for r in warmup + measured)):
            raise ValueError(f"{name}: incomplete or non-identical text requests")
        p50 = rank([r["wall_s"] for r in measured], .5)
        p95 = rank([r["wall_s"] for r in measured], .95)
        if (abs(p50 - phase["nearest_rank_p50_wall_s"]) > 1e-9
                or abs(p95 - phase["nearest_rank_p95_wall_s"]) > 1e-9):
            raise ValueError(f"{name}: reported latency statistic changed")
        row = {"name": name, "mode": phase["mode"], "measured": 20,
               "p50_wall_s": p50, "p95_wall_s": p95,
               "report_sha256": sha256(path)}
        if phase["mode"] == "npu":
            worker_path = root / f"{name}_worker.json"
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            trace_path = root / f"{name}_ort_profile.json"
            events = json.loads(trace_path.read_text(encoding="utf-8"))
            providers = [(e.get("args") or {}).get("provider") for e in events
                         if e.get("cat") == "Node"]
            placement = raw["npu_placement"]
            if (raw["status"] != "completed" or raw["all_runs_match_reference"] is not True
                    or worker["worker_stats"]["runs"] < 21 * 64
                    or worker["cpu_refine_top_k"] != 64
                    or worker["sampling_contract"] != "greedy-only"
                    or placement["ep"] != "vitisai"
                    or placement["node_counts"] != {"CPUExecutionProvider": 6, "vitisai": 1}
                    or providers.count("vitisai") != 1
                    or providers.count("CPUExecutionProvider") != 6):
                raise ValueError(f"{name}: NPU execution, re-rank or placement unverified")
            row.update({"worker_sha256": sha256(worker_path),
                        "ort_profile_sha256": sha256(trace_path),
                        "npu_worker_runs": worker["worker_stats"]["runs"],
                        "node_counts": placement["node_counts"]})
        elif (raw["status"] != "scoped_e2e_profiled"
              or raw["plan"]["selected_device"]["device_id"] != "cpu"):
            raise ValueError(f"{name}: unsplit CPU execution unverified")
        rows.append(row)
    cpu = [row["p50_wall_s"] for row in rows if row["mode"] == "cpu"]
    npu = [row["p50_wall_s"] for row in rows if row["mode"] == "npu"]
    ratios = [n / c for n, c in zip(npu, cpu)]
    if (any(abs(a - b) > 1e-9 for a, b in
            zip(ratios, paired["npu_vs_cpu_phase_p50_ratios"]))):
        raise ValueError("paired ratio changed")
    result = {
        "status": "paired_order_complete_no_npu_latency_benefit_on_fixture"
        if all(ratio > 1 for ratio in ratios) else "paired_order_complete_mixed_latency",
        "paired_report_sha256": sha256(paired_path),
        "reference_token_ids_sha256": expected,
        "phases": rows,
        "npu_to_cpu_p50_ratios": ratios,
        "scope": "one 28-prompt-token/64-output-token greedy fixture, serial 20 measured requests per phase; no package/NPU power or sustained thermal audit",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "npu_to_cpu_p50_ratios": ratios}))


if __name__ == "__main__":
    main()
