#!/usr/bin/env python3
"""Compare complete Spark BF16 RTX requests with an opt-in HX370 NPU head."""

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


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def windows_available_bytes() -> int:
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "[long](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024"],
        text=True, capture_output=True, timeout=30, check=True,
    )
    return int(result.stdout.strip())


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


async def profile(args: argparse.Namespace) -> dict:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be fresh to preserve raw evidence")
    output.mkdir(parents=True, exist_ok=True)
    source_spec = json.loads(args.source_spec.read_text(encoding="utf-8"))
    spec = dict(source_spec)
    spec["report_path"] = str(output / "npu_worker.json")
    spec["capture_path"] = None
    # Native Windows ORT needs a path under /mnt/c for its raw placement trace.
    profile_dir = Path(source_spec["profile_dir"]) / output.name
    profile_dir.mkdir(parents=True, exist_ok=True)
    spec["profile_dir"] = str(profile_dir)
    spec_path = output / "npu_spec.json"
    save(spec_path, spec)
    graph = build_graph_artifact(
        spec["graph"], fmt=FORMAT_ONNX_A16W8, opset=21,
        source_model="XHToken/Spark-X2.5-1.7B",
        source_revision="448e61eb392c00f2c403185c5b56d5e0665bfaab",
        component="spark_output_head",
        exporter="probe_spark_amd_npu_lm_head.py --composite-shards",
    )
    if graph.sha256 != spec["graph_sha256"]:
        raise ValueError("NPU graph changed")
    cuda_plan = plan_text_session(
        str(args.model), max_model_len=4096, max_num_seqs=1,
        max_num_batched_tokens=2048, enforce_eager=True, mask="cpu",
    )
    npu_plan = plan_external_stage(
        graph, enumerate_devices(load_profile(use_torch=False)),
        require="npu:amd", min_fraction_on_target=.125,
        worker_peak_rss_hint_bytes=int(spec["worker_peak_rss_hint_bytes"]),
    )
    if (not cuda_plan.admitted or cuda_plan.selected is None
            or cuda_plan.selected.device_id != "cuda:0" or not npu_plan.admitted):
        raise RuntimeError("CUDA decoder or AMD NPU head was not admitted")
    capacity = {
        "windows_available_before_bytes": windows_available_bytes(),
        "wsl_available_before_bytes": psutil.virtual_memory().available,
        "cuda_plan_vram_peak_bytes": cuda_plan.peak_bytes,
        "npu_plan_shared_ram_bytes": npu_plan.budget_bytes,
        "cuda_host_process_reserve_bytes": 2 << 30,
    }
    if npu_plan.budget_bytes + (2 << 30) > min(
        capacity["windows_available_before_bytes"], capacity["wsl_available_before_bytes"]
    ):
        raise RuntimeError(f"joint host-RAM budget refused: {capacity}")
    name, prompt = acceptance_prompts()[0]
    report = {
        "status": "running",
        "scope": "one-prompt BF16 Spark RTX complete greedy text versus RTX+HX370 NPU output head",
        "model": str(args.model.resolve(strict=True)),
        "model_index_sha256": sha256(args.model / "model.safetensors.index.json"),
        "source_spec_sha256": sha256(args.source_spec),
        "as_run_spec_sha256": sha256(spec_path),
        "graph_sha256": graph.sha256,
        "runtime": runtime_versions().to_dict(),
        "cuda_plan": cuda_plan.to_dict(), "npu_plan": npu_plan.to_dict(),
        "capacity": capacity, "prompt_name": name,
        "max_new_tokens": args.max_tokens,
        "warmups_per_phase": args.warmups, "measured_per_phase": args.repeats,
        "phases": [], "started_unix": time.time(),
    }
    save(output / "report.json", report)
    reference_ids = None
    try:
        for mode in ("cuda", "cuda_npu"):
            if mode == "cuda_npu":
                os.environ["VLLM_OMNI_SPARK_EXTERNAL_HEAD_SPEC"] = str(spec_path)
            else:
                os.environ.pop("VLLM_OMNI_SPARK_EXTERNAL_HEAD_SPEC", None)
            rows = []
            async with LocalTextEngine(cuda_plan) as engine:
                for index in range(args.warmups + args.repeats):
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
                        if (record.error or record.cancelled or not record.finished
                                or record.output_tokens != args.max_tokens):
                            raise RuntimeError(f"{mode} request {index} did not complete")
                        ids = record.output_token_ids
                        rows.append({
                            "index": index,
                            "warmup": index < args.warmups,
                            "token_ids": ids,
                            "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                            "wall_s": time.perf_counter() - started,
                            "ttft_s": record.ttft_s,
                            "output_tokens": record.output_tokens,
                            "text": record.text,
                        })
                    finally:
                        engine.close_session(session.session_id)
                placement = engine.report_placement()
                usage = engine.report_usage()
                measured_peak = engine.measured_peak()
                load_s = engine.load_seconds
            hashes = {row["token_ids_sha256"] for row in rows}
            if len(hashes) != 1:
                raise RuntimeError(f"{mode} generated different greedy token sequences")
            if reference_ids is None:
                reference_ids = rows[0]["token_ids"]
            measured = rows[args.warmups:]
            phase = {
                "mode": mode, "rows": rows, "placement": placement,
                "usage": usage, "measured_peak": measured_peak,
                "load_s_not_request_timing": load_s,
                "nearest_rank_p50_wall_s": rank([row["wall_s"] for row in measured], .5),
                "nearest_rank_p95_wall_s": rank([row["wall_s"] for row in measured], .95),
                "all_tokens_match_cuda_reference": all(row["token_ids"] == reference_ids for row in rows),
            }
            report["phases"].append(phase)
            save(output / "report.json", report)
            print(f"{mode}: {len(measured)} complete requests; p50={phase['nearest_rank_p50_wall_s']:.3f}s", flush=True)
        worker_path = Path(spec["report_path"])
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        report["npu_worker_report_sha256"] = sha256(worker_path)
        report["npu_worker_runs"] = worker.get("worker_stats", {}).get("runs")
        report["npu_placement"] = worker.get("placement")
        report["npu_refinement_calls"] = len(worker.get("refinement_calls", []))
        generated_tokens = (args.warmups + args.repeats) * args.max_tokens
        report["npu_non_request_calls"] = report["npu_worker_runs"] - generated_tokens
        if (not 0 <= report["npu_non_request_calls"] <= 8
                or report["npu_refinement_calls"] != report["npu_worker_runs"]
                or report["npu_placement"].get("ep") != "vitisai"
                or report["npu_placement"].get("target_nodes", 0) < 1):
            raise RuntimeError("NPU run count, refinement count or placement failed")
        report["status"] = ("scoped_joint_complete_quality_pass"
                            if report["phases"][1]["all_tokens_match_cuda_reference"]
                            else "scoped_joint_complete_token_quality_failed")
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        os.environ.pop("VLLM_OMNI_SPARK_EXTERNAL_HEAD_SPEC", None)
        report["ended_unix"] = time.time()
        save(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1 or args.max_tokens < 1:
        parser.error("warmups, repeats and tokens must be positive")
    result = asyncio.run(profile(args))
    print(json.dumps({"status": result["status"],
                      "phase_p50": [row["nearest_rank_p50_wall_s"] for row in result["phases"]],
                      "npu_worker_runs": result["npu_worker_runs"]}))


if __name__ == "__main__":
    main()
