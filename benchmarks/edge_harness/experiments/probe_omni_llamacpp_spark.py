#!/usr/bin/env python3
"""Exercise a pinned Spark GGUF through Omni StageRuntime and StagePool.

Runs complete, serial text requests. This does not claim incremental streaming,
quality beyond the named inputs, or device-memory peak attribution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace


def _rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--server-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--ggml-vk-visible-devices")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--capacity-gib", type=int, default=12)
    parser.add_argument("--reserve-gib", type=int, default=4)
    parser.add_argument("--abort-check", action="store_true")
    parser.add_argument("--context-check", action="store_true")
    parser.add_argument("--restart-check", action="store_true",
                        help="After in-flight cancellation, start a fresh stage and verify a request")
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0 or args.capacity_gib < args.reserve_gib or args.reserve_gib < 4:
        parser.error("invalid serial profile or explicit memory budget")
    if args.restart_check and not args.abort_check:
        parser.error("--restart-check requires --abort-check")

    from vllm_omni.config.stage_config import (
        DeployConfig,
        StageDeployConfig,
        merge_pipeline_deploy,
    )
    from vllm_omni.engine.stage_runtime import StageRuntime
    from vllm_omni.model_executor.models.spark2_5.pipeline import SPARK2_5_GGUF_TEXT_PIPELINE

    inventory = "Here is an inventory listing. " + " ".join(
        f"Item {i}: the code for warehouse district {i} is {1000 + 7 * i}."
        for i in range(1, 121)
    ) + " What is the code for warehouse district 73? Answer with just the number."
    cases = [
        ("short", "What is the capital of France? Answer in one word.", "Paris"),
        ("inventory_120", inventory, "1511"),
    ]
    pipeline = SPARK2_5_GGUF_TEXT_PIPELINE
    backend = {
        "name": "external.llamacpp.text.v1",
        "model_file": str(args.model_file),
        "model_sha256": args.model_sha256,
        "server_bin": str(args.server_bin),
        "server_sha256": args.server_sha256,
        "log_file": str(args.server_log),
        "device": args.device,
        "expected_device_name": args.expected_device_name,
        "ggml_vk_visible_devices": args.ggml_vk_visible_devices,
        "context_tokens": 4096,
        "max_new_tokens": 96,
        "max_io_bytes": 1 << 20,
        "memory_overhead_bytes": 2 << 30,
        "start_timeout_s": 120,
        "request_timeout_s": 120,
    }
    deploy = DeployConfig(
        async_chunk=False,
        stages=[StageDeployConfig(
            stage_id=0,
            backend=backend,
            resource_budget={
                "capacities": {"host_ram": args.capacity_gib << 30},
                "demands": {"host_ram": args.reserve_gib << 30},
            },
        )],
    )
    configs = [stage.to_omegaconf() for stage in merge_pipeline_deploy(pipeline, deploy)]
    runtime = StageRuntime(configs, "local-spark-gguf", "", stage_init_timeout=120, async_chunk=False)
    report = {
        "scope": "complete Spark GGUF text requests through Omni StageRuntime/StagePool and external llama.cpp backend",
        "model_file": str(args.model_file.resolve()),
        "device": args.device,
        "memory_budget": deploy.stages[0].resource_budget,
        "warmup_count": args.warmups,
        "measured_count": args.repeats,
    }
    try:
        started = time.perf_counter()
        runtime.initialize()
        report["startup_s"] = time.perf_counter() - started
        pool = runtime.stage_pools[0]
        report["execution_plan"] = pool.stage_client.execution_plan
        state = SimpleNamespace(sampling_params_list=[None])

        async def request_one(name: str, prompt: str, expected: str) -> dict:
            request_id = f"{name}-{time.monotonic_ns()}"
            started = time.perf_counter()
            await pool.submit_initial(request_id, state, {"text": prompt, "max_tokens": 96})
            deadline = time.monotonic() + 120
            while True:
                output = pool.poll_graph_output(0)
                if output is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Omni llama.cpp stage request timed out")
                await asyncio.sleep(0.002)
            wall_s = time.perf_counter() - started
            try:
                if output.error:
                    raise RuntimeError(output.error)
                answer = output.outputs[0].text.strip()
                row = {
                    "case": name,
                    "answer": answer,
                    "expected": expected,
                    "correct": answer == expected,
                    "wall_s": wall_s,
                    "stage_event": output.custom_output["stage_event"],
                    "metrics": output.metrics,
                }
                if not row["correct"]:
                    raise RuntimeError(f"Spark answer differed: {row}")
                return row
            finally:
                output.release_stage_buffers()
                await asyncio.sleep(0)

        report["checks"] = [await request_one(*case) for case in cases]
        profile_case = cases[1]
        report["warmups"] = [await request_one(*profile_case) for _ in range(args.warmups)]
        report["measured"] = [await request_one(*profile_case) for _ in range(args.repeats)]
        values = [row["wall_s"] for row in report["measured"]]
        report["nearest_rank_p50_wall_s"] = _rank(values, 0.5)
        report["nearest_rank_p95_wall_s"] = _rank(values, 0.95)
        if args.context_check:
            try:
                await pool.submit_initial(
                    f"overflow-{time.monotonic_ns()}", state,
                    {"text": "inventory " * 6000, "max_tokens": 96},
                )
            except Exception as exc:
                report["context_check"] = {"rejected": True, "error": f"{type(exc).__name__}: {exc}"}
            else:
                report["context_check"] = {"rejected": False}
                raise RuntimeError("llama.cpp accepted a prompt beyond the declared context")
            report["context_check"]["recovery"] = await request_one(*cases[0])
        if args.abort_check:
            def started_server_tasks() -> int:
                log = args.server_log.read_text(encoding="utf-8", errors="replace")
                return len(re.findall(
                    r"slot launch_slot_: id\s+\d+ \| task \d+ \| processing task",
                    log,
                ))

            tasks_before_abort = started_server_tasks()
            request_id = f"abort-{time.monotonic_ns()}"
            await pool.submit_initial(
                request_id, state,
                {"text": inventory, "max_tokens": 96},
            )
            deadline = time.monotonic() + 10
            while started_server_tasks() <= tasks_before_abort:
                if time.monotonic() >= deadline:
                    raise TimeoutError("abort request did not reach the owned llama.cpp server")
                await asyncio.sleep(0.02)
            premature = pool.poll_graph_output(0)
            if premature is not None:
                premature.release_stage_buffers()
                raise RuntimeError("abort request completed before in-flight cancellation")
            await pool.abort_requests([request_id])
            await asyncio.sleep(0.1)
            stale = pool.poll_graph_output(0)
            report["abort_check"] = {
                "abort_case": "inventory_120",
                "server_tasks_before_abort_request": tasks_before_abort,
                "server_tasks_after_abort_request_start": started_server_tasks(),
                "stale_output": stale is not None,
                "worker_exited": pool.stage_client._proc.poll() is not None,
                "ledger_after_abort": runtime.resource_ledger.snapshot(),
            }
            if stale is not None:
                stale.release_stage_buffers()
            if stale is not None or not report["abort_check"]["worker_exited"]:
                raise RuntimeError("llama.cpp cancel did not drain worker and stale output")
            if args.restart_check:
                old_generation = report["execution_plan"]["worker_generation"]
                runtime.shutdown()
                released = runtime.resource_ledger.snapshot()
                if released["reserved"]["host_ram"] or released["quarantined"]:
                    raise RuntimeError("cancelled stage retained a host-RAM reservation")
                restart_log = args.server_log.with_name(
                    args.server_log.stem + "_restart" + args.server_log.suffix
                )
                restart_deploy = DeployConfig(
                    async_chunk=False,
                    stages=[StageDeployConfig(
                        stage_id=0,
                        backend={**backend, "log_file": str(restart_log)},
                        resource_budget=deploy.stages[0].resource_budget,
                    )],
                )
                restart_configs = [
                    stage.to_omegaconf()
                    for stage in merge_pipeline_deploy(pipeline, restart_deploy)
                ]
                runtime = StageRuntime(
                    restart_configs, "local-spark-gguf-restart", "",
                    stage_init_timeout=120, async_chunk=False,
                )
                restarted_at = time.perf_counter()
                runtime.initialize()
                fresh_startup_s = time.perf_counter() - restarted_at
                pool = runtime.stage_pools[0]
                state = SimpleNamespace(sampling_params_list=[None])
                new_plan = pool.stage_client.execution_plan
                if new_plan["worker_generation"] == old_generation:
                    raise RuntimeError("restart reused the cancelled worker generation")
                restarted = await request_one(*cases[0])
                late = pool.poll_graph_output(0)
                report["restart_check"] = {
                    "first_runtime_ledger_after_shutdown": released,
                    "fresh_startup_s": fresh_startup_s,
                    "fresh_server_log": str(restart_log),
                    "fresh_execution_plan": new_plan,
                    "fresh_request": restarted,
                    "late_output": late is not None,
                }
                if late is not None:
                    late.release_stage_buffers()
                    raise RuntimeError("restart emitted an extra output")
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        runtime.shutdown()
        if runtime.resource_ledger is not None:
            report["ledger_after_shutdown"] = runtime.resource_ledger.snapshot()
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in {"measured", "warmups"}}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
