# SPDX-License-Identifier: Apache-2.0
"""Compare unsplit Spark BF16 with a live NPU+CPU head on unseen prompts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import psutil

from benchmarks.edge_harness.experiments.probe_spark_live_npu_head import windows_available_bytes
from vllm_omni.edge.hardware_probe import load_profile
from vllm_omni.edge.local.capabilities import FORMAT_ONNX_A16W8, enumerate_devices
from vllm_omni.edge.local.engine import LocalTextEngine
from vllm_omni.edge.local.external.stage import plan_external_stage
from vllm_omni.edge.local.manifest import build_graph_artifact, runtime_versions
from vllm_omni.edge.local.plan import plan_text_session


async def run_requests(plan: object, prompts: list[dict[str, str]], tokens: int) -> tuple[list[dict], dict]:
    rows = []
    async with LocalTextEngine(plan) as engine:
        for item in prompts:
            session = engine.open_session()
            try:
                started = time.perf_counter()
                request_id, stream = await engine.submit(
                    session, item["prompt"], max_tokens=tokens,
                    temperature=0.0, ignore_eos=True,
                )
                async for _ in stream:
                    pass
                record = engine.records[request_id]
                if record.error or record.cancelled or not record.finished or record.output_tokens != tokens:
                    raise RuntimeError(f"held-out request {item['name']} did not complete")
                ids = list(record.output_token_ids)
                rows.append({
                    "name": item["name"], "prompt_tokens": record.prompt_tokens,
                    "output_token_ids": ids, "text": record.text,
                    "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    "wall_s": time.perf_counter() - started,
                    "ttft_s": record.ttft_s,
                })
            finally:
                engine.close_session(session.session_id)
        metadata = {"placement": engine.report_placement(), "usage": engine.report_usage()}
    return rows, metadata


async def probe(args: argparse.Namespace) -> dict:
    prompts = json.loads(args.prompts.read_text())
    if (not prompts or any(set(item) != {"name", "prompt"} for item in prompts)
        or len({item["name"] for item in prompts}) != len(prompts)):
        raise ValueError("expected uniquely named held-out prompts")
    spec = json.loads(args.spec.read_text())
    graph = build_graph_artifact(
        spec["graph"], fmt=FORMAT_ONNX_A16W8, opset=21,
        source_model="XHToken/Spark-X2.5-1.7B",
        source_revision="448e61eb392c00f2c403185c5b56d5e0665bfaab",
        component="spark_output_head",
        exporter=("probe_spark_amd_npu_lm_head.py --composite-shards"
                  if spec.get("input_layout") == "post_final_norm"
                  else "probe_spark_amd_npu_lm_head.py --composite-with-norm"),
    )
    if graph.sha256 != spec["graph_sha256"] or spec.get("cpu_refine_top_k") != 64:
        raise ValueError("held-out hybrid run requires the pinned 64-candidate NPU graph")
    cpu_plan = plan_text_session(
        str(args.model), max_model_len=4096, max_num_seqs=1,
        max_num_batched_tokens=2048, enforce_eager=True,
    )
    if not cpu_plan.admitted or cpu_plan.selected is None or cpu_plan.selected.device_id != "cpu":
        raise RuntimeError("BF16 CPU reference plan refused")
    npu_plan = plan_external_stage(
        graph, enumerate_devices(load_profile(use_torch=False)),
        require="npu:amd", min_fraction_on_target=0.125,
        worker_peak_rss_hint_bytes=int(spec["worker_peak_rss_hint_bytes"]),
    )
    if not npu_plan.admitted:
        raise RuntimeError(npu_plan.summary())
    os.environ.pop("VLLM_OMNI_SPARK_EXTERNAL_HEAD_SPEC", None)
    reference, reference_metadata = await run_requests(cpu_plan, prompts, args.max_tokens)
    capacity = {
        "windows_available_before_hybrid_bytes": windows_available_bytes(),
        "wsl_available_before_hybrid_bytes": psutil.virtual_memory().available,
        "combined_budget_bytes": cpu_plan.peak_bytes + npu_plan.budget_bytes,
    }
    if capacity["combined_budget_bytes"] > min(
        capacity["windows_available_before_hybrid_bytes"],
        capacity["wsl_available_before_hybrid_bytes"],
    ):
        raise RuntimeError(f"joint shared-RAM budget refused: {capacity}")
    worker_report = Path(spec["report_path"])
    worker_report.unlink(missing_ok=True)
    os.environ["VLLM_OMNI_SPARK_EXTERNAL_HEAD_SPEC"] = str(args.spec.resolve())
    try:
        hybrid, hybrid_metadata = await run_requests(cpu_plan, prompts, args.max_tokens)
    finally:
        os.environ.pop("VLLM_OMNI_SPARK_EXTERNAL_HEAD_SPEC", None)
    if not worker_report.exists():
        raise RuntimeError("NPU worker did not write placement and refinement evidence")
    worker = json.loads(worker_report.read_text())
    placement = worker["placement"]
    refinements = worker["refinement_calls"]
    if (placement.get("ep") != "vitisai" or placement.get("target_nodes", 0) < 1
        or len(refinements) < len(prompts) * args.max_tokens):
        raise RuntimeError("NPU placement or refinement count failed")
    rows = []
    for original, actual in zip(reference, hybrid, strict=True):
        rows.append({
            "name": original["name"],
            "reference": original, "hybrid": actual,
            "token_ids_equal": original["output_token_ids"] == actual["output_token_ids"],
            "first_divergence_index": next(
                (i for i, (a, b) in enumerate(zip(
                    original["output_token_ids"], actual["output_token_ids"], strict=True
                )) if a != b), None,
            ),
        })
    report = {
        "scope": "held-out serial 128-token greedy requests; unsplit BF16 CPU then NPU top-64/CPU sparse BF16 re-rank",
        "status": "completed", "model": str(args.model),
        "runtime": runtime_versions().to_dict(),
        "prompt_file_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "spec": str(args.spec), "graph_sha256": graph.sha256,
        "cpu_plan": cpu_plan.to_dict(), "npu_plan": npu_plan.to_dict(),
        "capacity": capacity, "reference_metadata": reference_metadata,
        "hybrid_metadata": hybrid_metadata,
        "npu_placement": placement,
        "npu_worker_peak_rss_bytes": worker["worker_stats"]["peak_rss_bytes"],
        "npu_worker_runs": worker["worker_stats"]["runs"],
        "refinement_calls": len(refinements),
        "cpu_top1_in_candidates": sum(bool(row.get("cpu_top1_in_candidates")) for row in refinements),
        "refined_matches_full_cpu": sum(bool(row.get("refined_top1_matches_full_cpu")) for row in refinements),
        "diagnostic_refinement_count": sum("refined_top1_matches_full_cpu" in row for row in refinements),
        "exact_match_count": sum(row["token_ids_equal"] for row in rows),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    report = asyncio.run(probe(args))
    print(json.dumps({k: report[k] for k in (
        "status", "exact_match_count", "npu_worker_runs", "refinement_calls",
        "cpu_top1_in_candidates", "refined_matches_full_cpu",
    )}, indent=2))


if __name__ == "__main__":
    main()
