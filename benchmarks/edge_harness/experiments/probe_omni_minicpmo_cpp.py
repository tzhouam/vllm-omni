#!/usr/bin/env python3
"""Run a pinned audio+image→text+speech MiniCPM-o GGUF request through Omni.

The single C++ process owns the whole session. This probe records complete
request latency; it does not claim incremental speech streaming.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
import wave
from pathlib import Path
from types import SimpleNamespace


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "cli-bin", "reference-wav", "audio-wav", "image-jpeg",
                 "artifact-report", "work-root", "log-root", "output-report", "output-wav"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cli-sha256", required=True)
    parser.add_argument("--placement", choices=("cpu", "radeon-hybrid"), required=True)
    parser.add_argument("--reserve-gib", type=int, default=24)
    parser.add_argument("--capacity-gib", type=int, default=30)
    parser.add_argument("--abort-check", action="store_true")
    parser.add_argument("--expect-admission-refusal", action="store_true")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--expected-text-any", action="append",
                        help="accept one of these case-insensitive output terms; defaults to red")
    parser.add_argument("--max-output-audio-s", type=int, default=20)
    parser.add_argument("--max-wav-bytes", type=int, default=2 << 20)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats must be positive")
    if args.reserve_gib < 1 or args.capacity_gib < args.reserve_gib:
        parser.error("invalid explicit host-RAM budget")
    expected_text_any = args.expected_text_any or ["red"]
    if any(not term.strip() for term in expected_text_any):
        parser.error("expected text terms must be nonempty")
    if not 0 < args.max_output_audio_s <= 120 or not 0 < args.max_wav_bytes <= 64 << 20:
        parser.error("invalid bounded output audio or WAV size")

    import psutil

    host_available_bytes = psutil.virtual_memory().available
    if args.capacity_gib << 30 > host_available_bytes:
        raise RuntimeError("declared host-RAM capacity exceeds OS available RAM before load")

    from vllm_omni.config.stage_config import DeployConfig, StageDeployConfig, merge_pipeline_deploy
    from vllm_omni.engine.stage_runtime import StageRuntime
    from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_GGUF_WHOLE_PIPELINE

    source = json.loads(args.artifact_report.read_text(encoding="utf-8"))
    reference_hash = hashlib.sha256(args.reference_wav.read_bytes()).hexdigest()
    inputs = {"audio_wav": args.audio_wav.read_bytes(), "image_jpeg": args.image_jpeg.read_bytes()}
    backend = {
        "name": "external.minicpmo.gguf.v1",
        "placement": args.placement,
        "ggml_vk_visible_devices": "1" if args.placement == "radeon-hybrid" else None,
        "expected_gpu_name": "AMD Radeon(TM) 890M Graphics" if args.placement == "radeon-hybrid" else None,
        "model_dir": str(args.model_dir),
        "model_revision": source["model_revision"],
        "artifact_sha256": {name: data["sha256"] for name, data in source["artifacts"].items()},
        "cli_bin": str(args.cli_bin), "cli_sha256": args.cli_sha256,
        "reference_wav": str(args.reference_wav), "reference_sha256": reference_hash,
        "work_root": str(args.work_root), "log_root": str(args.log_root),
        "memory_overhead_bytes": 12 << 30,
        "context_tokens": 2048, "max_new_tokens": 96,
        "max_input_bytes": 2 << 20, "max_wav_bytes": args.max_wav_bytes,
        "max_text_bytes": 16 << 10, "max_work_bytes": 64 << 20,
        "max_input_audio_s": 10, "max_output_audio_s": args.max_output_audio_s,
        "request_timeout_s": 240, "t2w_wait_seconds": 180,
    }
    budget = {"capacities": {"host_ram": args.capacity_gib << 30},
              "demands": {"host_ram": args.reserve_gib << 30}}
    deploy = DeployConfig(async_chunk=False, stages=[StageDeployConfig(
        stage_id=0, backend=backend, resource_budget=budget,
    )])
    configs = [item.to_omegaconf() for item in merge_pipeline_deploy(
        MINICPMO_4_5_GGUF_WHOLE_PIPELINE, deploy,
    )]
    runtime = StageRuntime(configs, "local-minicpmo-gguf", "", stage_init_timeout=180,
                           async_chunk=False)
    report = {"scope": "MiniCPM-o GGUF complete audio+image→text+speech through Omni StageRuntime/StagePool",
              "placement": args.placement, "model_revision": source["model_revision"],
              "artifact_sha256": backend["artifact_sha256"],
              "cli_sha256": args.cli_sha256, "reference_sha256": reference_hash,
              "input_sha256": {key: hashlib.sha256(value).hexdigest() for key, value in inputs.items()},
              "expected_text_any": expected_text_any,
              "output_limits": {"max_output_audio_s": args.max_output_audio_s,
                                "max_wav_bytes": args.max_wav_bytes},
              "memory_budget": budget, "profile_count": args.repeats,
              "warmup_count": args.warmups,
              "host_available_bytes_before": host_available_bytes}
    try:
        started = time.perf_counter()
        if args.expect_admission_refusal:
            try:
                runtime.initialize()
            except Exception as exc:
                if "exceed reservation" not in str(exc):
                    raise
                report["admission_refusal"] = f"{type(exc).__name__}: {exc}"
                report["status"] = "passed"
                return
            raise AssertionError("MiniCPM-o accepted an insufficient memory reservation")
        runtime.initialize()
        report["startup_s"] = time.perf_counter() - started
        pool = runtime.stage_pools[0]
        report["execution_plan_at_start"] = dict(pool.stage_client.execution_plan)
        state = SimpleNamespace(sampling_params_list=[None])

        async def request_one(request_id: str):
            started = time.perf_counter()
            await pool.submit_initial(request_id, state, inputs)
            deadline = time.monotonic() + 250
            while True:
                output = pool.poll_graph_output(0)
                if output is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("MiniCPM-o Omni output timed out")
                await asyncio.sleep(0.01)
            try:
                if output.error:
                    raise RuntimeError(output.error)
                completion = output.outputs[0]
                metadata = output.custom_output["audio_metadata"]
                pcm = (completion.multimodal_output["audio"].numpy() * 32768).astype("<i2").tobytes()
                assert completion.multimodal_output["sr"] == 24000
                assert len(pcm) // 2 == metadata["pcm_frames"]
                assert hashlib.sha256(pcm).hexdigest() == metadata["pcm_sha256"]
                assert completion.text.strip(), "MiniCPM-o returned empty text"
                assert any(term.casefold() in completion.text.casefold()
                           for term in expected_text_any), completion.text
                assert output.custom_output["stage_event"]["terminal"] is True
                return {"text": completion.text, "wall_s": time.perf_counter() - started,
                        "audio_metadata": metadata, "stage_event": output.custom_output["stage_event"],
                        "metrics": output.metrics, "pcm": pcm}
            finally:
                output.release_stage_buffers()
                await asyncio.sleep(0)

        report["warmups"] = []
        for index in range(args.warmups):
            item = await request_one(f"minicpmo-warmup-{index}")
            item.pop("pcm")
            report["warmups"].append(item)
        measured = []
        report["measured"] = measured
        for index in range(args.repeats):
            item = await request_one(f"minicpmo-measured-{index}")
            if index == 0:
                pcm = item["pcm"]
                report["complete_request"] = {key: value for key, value in item.items() if key != "pcm"}
            item.pop("pcm")
            measured.append(item)
        walls = sorted(item["wall_s"] for item in measured)
        report["nearest_rank_p50_wall_s"] = walls[math.ceil(0.50 * len(walls)) - 1]
        report["nearest_rank_p95_wall_s"] = walls[math.ceil(0.95 * len(walls)) - 1]
        args.output_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(args.output_wav), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(24000)
            out.writeframes(pcm)
        report["output_wav"] = str(args.output_wav)
        report["execution_plan_after_request"] = dict(pool.stage_client.execution_plan)
        if args.abort_check:
            await pool.submit_initial("minicpmo-abort", state, inputs)
            await asyncio.sleep(0.05)
            await pool.abort_requests(["minicpmo-abort"])
            await asyncio.sleep(0.1)
            report["abort_check"] = {
                "stale_output": pool.poll_graph_output(0) is not None,
                "ledger": runtime.resource_ledger.snapshot(),
            }
            assert not report["abort_check"]["stale_output"]
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
        args.output_report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "artifact_sha256"},
                         indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
