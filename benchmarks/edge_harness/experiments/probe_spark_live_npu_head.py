# SPDX-License-Identifier: Apache-2.0
"""Profile complete Spark text requests with the opt-in AMD NPU output head."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import psutil

from vllm_omni.edge.hardware_probe import load_profile
from vllm_omni.edge.local.capabilities import FORMAT_ONNX_A16W8, enumerate_devices
from vllm_omni.edge.local.engine import LocalTextEngine
from vllm_omni.edge.local.external.stage import plan_external_stage
from vllm_omni.edge.local.manifest import build_graph_artifact, runtime_versions
from vllm_omni.edge.local.plan import plan_text_session
from vllm_omni.edge.local.prompts import acceptance_prompts


def windows_available_bytes() -> int:
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "[long](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024"],
        text=True, capture_output=True, timeout=30, check=True,
    )
    return int(result.stdout.strip())


async def probe(args: argparse.Namespace) -> dict:
    if args.measured_requests < 1 or args.warmup_requests < 0 or args.max_tokens < 1:
        raise ValueError("request counts and max tokens must be positive")
    spec = json.loads(args.spec.read_text())
    worker_report = Path(spec["report_path"])
    graph = build_graph_artifact(
        spec["graph"], fmt=FORMAT_ONNX_A16W8, opset=21,
        source_model="XHToken/Spark-X2.5-1.7B",
        source_revision="448e61eb392c00f2c403185c5b56d5e0665bfaab",
        component="spark_output_head", exporter="probe_spark_amd_npu_lm_head.py --composite-with-norm",
    )
    if graph.sha256 != spec["graph_sha256"]:
        raise ValueError("NPU graph changed")
    cpu = plan_text_session(
        str(args.model), max_model_len=4096, max_num_seqs=1,
        max_num_batched_tokens=2048, enforce_eager=True,
    )
    npu = plan_external_stage(
        graph, enumerate_devices(load_profile(use_torch=False)),
        require="npu:amd", min_fraction_on_target=0.125,
        worker_peak_rss_hint_bytes=int(spec["worker_peak_rss_hint_bytes"]),
    )
    if not cpu.admitted or cpu.selected is None or cpu.selected.device_id != "cpu":
        raise RuntimeError("BF16 Spark CPU model was not admitted")
    if not npu.admitted:
        raise RuntimeError(npu.summary())
    capacity = {
        "windows_available_before_bytes": windows_available_bytes(),
        "wsl_available_before_bytes": psutil.virtual_memory().available,
        "cpu_plan_peak_bytes": cpu.peak_bytes,
        "npu_plan_budget_bytes": npu.budget_bytes,
    }
    capacity["combined_budget_bytes"] = cpu.peak_bytes + npu.budget_bytes
    if capacity["combined_budget_bytes"] > min(
        capacity["windows_available_before_bytes"], capacity["wsl_available_before_bytes"]
    ):
        raise RuntimeError(f"joint shared-RAM budget refused: {capacity}")
    reference = json.loads(args.reference_profile.read_text())
    prompts = acceptance_prompts()
    if args.prompt_set in ("all", "selected"):
        if (args.max_tokens != reference.get("acceptance", {}).get("min_tokens_required")
            or len(reference.get("requests", [])) != len(prompts)
            or args.measured_requests != (len(prompts) if args.prompt_set == "all" else 1)
            or args.warmup_requests != 0
            or not 0 <= args.prompt_index < len(prompts)):
            raise ValueError("acceptance runs require 128 tokens, one run per selected prompt and no warmup")
        reference_ids = [row["output_token_ids"] for row in reference["requests"]]
    else:
        if args.max_tokens != reference.get("max_new_tokens"):
            raise ValueError("output length must match the CPU reference profile")
        reference_ids = None
    worker_report.unlink(missing_ok=True)
    expected_hash = reference["runs"][0]["token_ids_sha256"] if reference_ids is None else None
    report = {
        "status": "running", "scope": "serial complete text requests with live vLLM decoder and AMD NPU output head",
        "started_unix": time.time(), "model": str(args.model),
        "spec": str(args.spec), "runtime": runtime_versions().to_dict(),
        "cpu_plan": cpu.to_dict(), "npu_plan": npu.to_dict(), "capacity": capacity,
        "reference_token_ids_sha256": expected_hash,
        "prompt_set": args.prompt_set,
        "warmup_count": args.warmup_requests,
        "measured_count": args.measured_requests,
        "max_new_tokens": args.max_tokens,
        "runs": [],
    }
    os.environ["VLLM_OMNI_SPARK_EXTERNAL_HEAD_SPEC"] = str(args.spec.resolve())
    engine: LocalTextEngine | None = None
    try:
        engine = LocalTextEngine(cpu)
        await engine.start()
        for run_index in range(args.warmup_requests + args.measured_requests):
            prompt_index = (
                run_index if args.prompt_set == "all"
                else args.prompt_index if args.prompt_set == "selected"
                else 0
            )
            name, prompt = prompts[prompt_index]
            run_expected_hash = (
                hashlib.sha256(json.dumps(reference_ids[prompt_index]).encode()).hexdigest()
                if reference_ids is not None else expected_hash
            )
            session = engine.open_session()
            try:
                started = time.perf_counter()
                request_id, stream = await engine.submit(
                    session, prompt, max_tokens=args.max_tokens,
                    temperature=0.0, ignore_eos=True,
                )
                async for _ in stream:
                    pass
                record = engine.records[request_id]
                row = {
                    "kind": "warmup" if run_index < args.warmup_requests else "measured",
                    "prompt_name": name,
                    "prompt_tokens": record.prompt_tokens,
                    "output_tokens": record.output_tokens,
                    "output_token_ids": record.output_token_ids,
                    "text": record.text,
                    "token_ids_sha256": hashlib.sha256(json.dumps(record.output_token_ids).encode()).hexdigest(),
                    "reference_token_ids_sha256": run_expected_hash,
                    "wall_s": time.perf_counter() - started,
                    "ttft_s": record.ttft_s,
                    "error": record.error,
                    "cancelled": record.cancelled,
                    "finished": record.finished,
                }
                row["token_ids_match_reference"] = row["token_ids_sha256"] == run_expected_hash
                if reference_ids is not None:
                    expected = reference_ids[prompt_index]
                    actual = record.output_token_ids
                    row["token_agreement_fraction"] = sum(a == b for a, b in zip(actual, expected)) / len(expected)
                    row["first_divergence_index"] = next(
                        (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                        None,
                    )
                report["runs"].append(row)
                if record.error or record.cancelled or not record.finished or record.output_tokens != args.max_tokens:
                    raise RuntimeError(f"live split request {run_index} did not complete")
            finally:
                engine.close_session(session.session_id)
        measured = [row for row in report["runs"] if row["kind"] == "measured"]

        def nearest_rank(values: list[float], fraction: float) -> float:
            ordered = sorted(values)
            return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]
        if args.prompt_set == "first":
            report["nearest_rank_p50_wall_s"] = nearest_rank([row["wall_s"] for row in measured], .5)
            report["nearest_rank_p95_wall_s"] = nearest_rank([row["wall_s"] for row in measured], .95)
            report["nearest_rank_p50_ttft_s"] = nearest_rank([row["ttft_s"] for row in measured], .5)
            report["nearest_rank_p95_ttft_s"] = nearest_rank([row["ttft_s"] for row in measured], .95)
        report["all_runs_match_reference"] = all(row["token_ids_match_reference"] for row in report["runs"])
        report["exact_match_count"] = sum(row["token_ids_match_reference"] for row in measured)
        if reference_ids is not None:
            report["mean_token_agreement_fraction"] = sum(
                row["token_agreement_fraction"] for row in measured
            ) / len(measured)
        if len(report["runs"]) == 1:
            row = report["runs"][0]
            report.update({
                "request_wall_s": row["wall_s"], "prompt_tokens": row["prompt_tokens"],
                "output_tokens": row["output_tokens"], "output_token_ids": row["output_token_ids"],
                "output_token_ids_sha256": row["token_ids_sha256"],
                "token_ids_match_reference": row["token_ids_match_reference"],
                "ttft_s": row["ttft_s"], "record_error": row["error"],
                "record_cancelled": row["cancelled"], "record_finished": row["finished"],
            })
        report["cpu_placement"] = engine.report_placement()
        report["cpu_usage"] = engine.report_usage()
        report["cpu_measured_peak"] = engine.measured_peak()
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if engine is not None:
            await engine.close()
        if worker_report.exists():
            worker = json.loads(worker_report.read_text())
            report["npu_worker_runs"] = worker.get("worker_stats", {}).get("runs")
            report["npu_placement"] = worker.get("placement")
            report["npu_worker_peak_rss_bytes"] = worker.get("worker_stats", {}).get("peak_rss_bytes")
        if report["status"] == "completed":
            placement = report.get("npu_placement") or {}
            if ((report.get("npu_worker_runs") or 0) < sum(
                row["output_tokens"] for row in report["runs"]
            ) or placement.get("target_nodes", 0) < 1
                or placement.get("ep") != "vitisai"):
                report["status"] = "failed"
                report["error"] = "NPU run count or placement does not match all generated tokens"
        os.environ.pop("VLLM_OMNI_SPARK_EXTERNAL_HEAD_SPEC", None)
        report["ended_unix"] = time.time()
        report["windows_available_after_close_bytes"] = windows_available_bytes()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "completed":
        raise RuntimeError(report["error"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-set", choices=("first", "all", "selected"), default="first")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--measured-requests", type=int, default=1)
    args = parser.parse_args()
    result = asyncio.run(probe(args))
    print(json.dumps({key: result.get(key) for key in (
        "status", "measured_count", "nearest_rank_p50_wall_s", "nearest_rank_p95_wall_s",
        "all_runs_match_reference", "exact_match_count", "mean_token_agreement_fraction",
        "npu_worker_runs"
    )}, indent=2))


if __name__ == "__main__":
    main()
