#!/usr/bin/env python3
"""Alternate whole Spark CPU and CPU+AMD NPU phases on one HX370 host."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output-dir must be fresh; prior raw phases are immutable evidence")
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.source_spec.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_profile.read_text(encoding="utf-8"))
    expected = reference["runs"][0]["token_ids_sha256"]
    report = {
        "status": "running",
        "scope": "paired-order serial complete Spark BF16 greedy text, CPU versus CPU decoder plus AMD NPU projection and BF16 CPU top-64 re-rank",
        "host": platform.platform(),
        "model": str(args.model.resolve(strict=True)),
        "checkpoint_index_sha256": sha256(args.model / "model.safetensors.index.json"),
        "source_spec_sha256": sha256(args.source_spec),
        "graph_sha256": spec["graph_sha256"],
        "reference_profile_sha256": sha256(args.reference_profile),
        "expected_token_ids_sha256": expected,
        "phase_order": ["cpu", "npu", "npu", "cpu"],
        "warmups_per_phase": 1,
        "measured_requests_per_phase": 20,
        "prompt": "acceptance_prompts()[0]",
        "max_new_tokens": 64,
        "concurrency": 1,
        "power_condition": "observed host condition; package/NPU power and thermals not sampled",
        "phases": [],
    }
    save(output / "paired_report.json", report)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    for index, mode in enumerate(report["phase_order"], 1):
        name = f"{index:02d}_{mode}"
        phase_report = output / f"{name}_report.json"
        log_file = output / f"{name}_driver.log"
        if mode == "cpu":
            command = [
                sys.executable,
                str(root / "benchmarks/edge_harness/experiments/profile_spark_bf16_omni_cpu.py"),
                "--model", str(args.model), "--report", str(phase_report),
                "--warmups", "1", "--repeats", "20", "--max-tokens", "64",
            ]
        else:
            phase_spec = dict(spec)
            phase_spec["report_path"] = str(output / f"{name}_worker.json")
            # ORT runs in native Windows. Only /mnt/c paths are translated by
            # the existing worker transport; a WSL-only path yields no trace.
            profile_dir = Path(spec["profile_dir"]) / f"paired_{name}"
            profile_dir.mkdir(parents=True, exist_ok=True)
            phase_spec["profile_dir"] = str(profile_dir)
            spec_path = output / f"{name}_spec.json"
            save(spec_path, phase_spec)
            command = [
                sys.executable,
                str(root / "benchmarks/edge_harness/experiments/probe_spark_live_npu_head.py"),
                "--model", str(args.model), "--spec", str(spec_path),
                "--reference-profile", str(args.reference_profile),
                "--report", str(phase_report), "--warmup-requests", "1",
                "--measured-requests", "20", "--max-tokens", "64",
            ]
        started = time.time()
        print(f"starting {name}", flush=True)
        with log_file.open("wb") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        phase = {
            "name": name, "mode": mode, "started_unix": started,
            "ended_unix": time.time(), "exit_code": result.returncode,
            "command": command, "driver_log": str(log_file),
            "driver_log_sha256": sha256(log_file),
        }
        report["phases"].append(phase)
        if result.returncode != 0 or not phase_report.is_file():
            report["status"] = "failed"
            report["error"] = f"{name} did not complete; inspect its retained driver log"
            save(output / "paired_report.json", report)
            raise RuntimeError(report["error"])
        raw = json.loads(phase_report.read_text(encoding="utf-8"))
        measured = [row for row in raw["runs"] if row["kind"] == "measured"]
        hashes = {row["token_ids_sha256"] for row in measured}
        if (len(measured) != 20 or hashes != {expected}
                or raw["max_new_tokens"] != 64
                or (mode == "cpu" and raw["status"] != "scoped_e2e_profiled")
                or (mode == "npu" and (raw["status"] != "completed"
                                      or not raw["all_runs_match_reference"]
                                      or raw["npu_placement"]["ep"] != "vitisai"
                                      or raw["npu_placement"]["target_nodes"] < 1))):
            report["status"] = "failed"
            report["error"] = f"{name} lost workload, token or placement fidelity"
            save(output / "paired_report.json", report)
            raise RuntimeError(report["error"])
        walls = [row["wall_s"] for row in measured]
        phase.update({
            "report": str(phase_report), "report_sha256": sha256(phase_report),
            "worker_report": str(output / f"{name}_worker.json") if mode == "npu" else None,
            "measured_count": len(measured),
            "nearest_rank_p50_wall_s": rank(walls, .5),
            "nearest_rank_p95_wall_s": rank(walls, .95),
            "mean_wall_s": sum(walls) / len(walls),
            "min_wall_s": min(walls), "max_wall_s": max(walls),
            "token_ids_sha256": expected,
            "npu_placement": raw.get("npu_placement"),
        })
        save(output / "paired_report.json", report)
        print(f"completed {name}: p50={phase['nearest_rank_p50_wall_s']:.3f}s", flush=True)
    by_mode = {
        mode: [phase["nearest_rank_p50_wall_s"] for phase in report["phases"]
               if phase["mode"] == mode]
        for mode in ("cpu", "npu")
    }
    report["phase_p50_wall_s"] = by_mode
    report["npu_vs_cpu_phase_p50_ratios"] = [
        npu / cpu for npu, cpu in zip(by_mode["npu"], by_mode["cpu"])
    ]
    report["status"] = "completed"
    save(output / "paired_report.json", report)
    print(json.dumps({"status": report["status"],
                      "phase_p50_wall_s": by_mode,
                      "npu_vs_cpu_phase_p50_ratios": report["npu_vs_cpu_phase_p50_ratios"]}))


if __name__ == "__main__":
    main()
