#!/usr/bin/env python3
"""Profile bounded Qwen3.8 GGUF text and image requests through one Omni stage."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace


def nearest_rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model_file", "mmproj_file", "server_bin", "model_sha256", "mmproj_sha256",
                "server_sha256", "device", "server_log", "output_report"):
        parser.add_argument("--" + key.replace("_", "-"), required=True)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--ggml-vk-visible-devices")
    parser.add_argument("--reserve-gib", type=int, default=28)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.reserve_gib < 23 or args.repeats < 1:
        parser.error("Qwen3.8 GGUF requires an explicit >=23 GiB reservation and positive repeats")

    import psutil
    from PIL import Image

    from vllm_omni.config.stage_config import DeployConfig, StageDeployConfig, merge_pipeline_deploy
    from vllm_omni.engine.stage_runtime import StageRuntime
    from vllm_omni.model_executor.models.qwen3_8_gguf.pipeline import QWEN3_8_GGUF_MULTIMODAL_PIPELINE

    available = psutil.virtual_memory().available
    reserve = args.reserve_gib << 30
    report = {
        "scope": "one pinned Qwen3.8 27B Q4_K_M GGUF + Q8 vision projector through Omni whole-session llama.cpp",
        "device": args.device,
        "available_host_ram_before_bytes": available,
        "reserved_host_ram_bytes": reserve,
        "repeats": args.repeats,
        "model_file": str(Path(args.model_file).resolve()),
        "mmproj_file": str(Path(args.mmproj_file).resolve()),
    }
    runtime = None
    try:
        if available < reserve:
            raise MemoryError("available native host RAM is below the declared Qwen3.8 reservation")
        backend = {
            "name": "external.llamacpp.multimodal.v1",
            "model_file": args.model_file,
            "mmproj_file": args.mmproj_file,
            "server_bin": args.server_bin,
            "model_sha256": args.model_sha256,
            "mmproj_sha256": args.mmproj_sha256,
            "server_sha256": args.server_sha256,
            "log_file": args.server_log,
            "device": args.device,
            "expected_device_name": args.expected_device_name,
            "ggml_vk_visible_devices": args.ggml_vk_visible_devices,
            "context_tokens": 2048,
            "max_new_tokens": 128,
            "max_io_bytes": 1 << 20,
            "max_image_bytes": 256 << 10,
            "image_token_reserve": 512,
            "memory_overhead_bytes": 4 << 30,
            "start_timeout_s": 300,
            "request_timeout_s": 300,
        }
        deploy = DeployConfig(async_chunk=False, stages=[StageDeployConfig(
            stage_id=0, backend=backend,
            resource_budget={"capacities": {"host_ram": available}, "demands": {"host_ram": reserve}},
        )])
        configs = [stage.to_omegaconf() for stage in merge_pipeline_deploy(
            QWEN3_8_GGUF_MULTIMODAL_PIPELINE, deploy
        )]
        runtime = StageRuntime(configs, "local-qwen38-gguf", "", stage_init_timeout=300, async_chunk=False)
        started = time.perf_counter()
        runtime.initialize()
        report["startup_s"] = time.perf_counter() - started
        pool = runtime.stage_pools[0]
        report["execution_plan"] = pool.stage_client.execution_plan
        state = SimpleNamespace(sampling_params_list=[None])
        image = Image.new("RGB", (96, 96), (255, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        cases = (
            ("text", {"text": "Reply with the single word ready.", "max_tokens": 128}, "ready"),
            ("image", {"text": "What is the dominant color in this image? Answer in one word.",
                       "image_data_url": image_data_url, "max_tokens": 128}, "Red"),
        )

        async def one(kind: str, prompt: dict, expected: str) -> dict:
            request_id = f"{kind}-{time.monotonic_ns()}"
            started = time.perf_counter()
            await pool.submit_initial(request_id, state, prompt)
            deadline = time.monotonic() + 300
            while True:
                output = pool.poll_graph_output(0)
                if output is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{kind} request exceeded 300 s")
                await asyncio.sleep(0.01)
            try:
                if output.error:
                    raise RuntimeError(output.error)
                answer = output.outputs[0].text.strip()
                row = {"kind": kind, "wall_s": time.perf_counter() - started,
                       "answer": answer, "expected": expected,
                       "stage_event": output.custom_output.get("stage_event"),
                       "metrics": output.metrics}
                if answer.casefold() != expected.casefold():
                    raise RuntimeError(f"unexpected Qwen3.8 {kind} answer: {answer!r}")
                return row
            finally:
                output.release_stage_buffers()
                await asyncio.sleep(0)

        report["checks"] = [await one(*case) for case in cases]
        for kind, prompt, expected in cases:
            report[f"{kind}_warmup"] = await one(kind, prompt, expected)
            measured = [await one(kind, prompt, expected) for _ in range(args.repeats)]
            report[f"{kind}_measured"] = measured
            walls = [row["wall_s"] for row in measured]
            report[f"nearest_rank_p50_{kind}_wall_s"] = nearest_rank(walls, 0.5)
            report[f"nearest_rank_p95_{kind}_wall_s"] = nearest_rank(walls, 0.95)
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if runtime is not None:
            runtime.shutdown()
            report["ledger_after_shutdown"] = runtime.resource_ledger.snapshot()
        out = Path(args.output_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "error": report.get("error")}))


if __name__ == "__main__":
    asyncio.run(main())
