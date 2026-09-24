#!/usr/bin/env python3
"""Profile pinned Qwen3-TTS CustomVoice through the Omni CrispASR stage."""

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

import numpy as np


def _rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def _save_wav(path: Path, audio) -> None:
    pcm = np.rint(audio.numpy() * 32768).astype("<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "server_bin", "server_sha256", "talker_file", "talker_sha256", "codec_file",
        "codec_sha256", "server_log", "output_report", "output_wav",
        "punc_file", "punc_sha256",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--expected-gpu-name", default="AMD Radeon(TM) 890M Graphics")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--abort-check", action="store_true")
    parser.add_argument("--restart-check", action="store_true",
                        help="After in-flight cancellation, start a fresh stage and verify audio")
    parser.add_argument("--capacity-gib", type=int, default=16)
    parser.add_argument("--reserve-gib", type=int, default=8)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0 or args.reserve_gib <= 0 or args.capacity_gib < args.reserve_gib:
        parser.error("invalid warmup/repeat count or memory budget")
    if args.restart_check and not args.abort_check:
        parser.error("--restart-check requires --abort-check")

    from vllm_omni.config.stage_config import DeployConfig, StageDeployConfig, merge_pipeline_deploy
    from vllm_omni.engine.stage_runtime import StageRuntime
    from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_CRISP_HYBRID_PIPELINE

    backend = {
        "name": "external.crisp.tts.v1",
        "server_bin": args.server_bin,
        "server_sha256": args.server_sha256,
        "talker_file": args.talker_file,
        "talker_sha256": args.talker_sha256,
        "codec_file": args.codec_file,
        "codec_sha256": args.codec_sha256,
        "punc_file": args.punc_file,
        "punc_sha256": args.punc_sha256,
        "log_file": args.server_log,
        "expected_gpu_name": args.expected_gpu_name,
        "ggml_vk_visible_devices": "1",
        "memory_overhead_bytes": 5 << 30,
        "max_text_bytes": 4096,
        "max_wav_bytes": 4 << 20,
        "max_audio_s": 10,
        "request_timeout_s": 180,
        "start_timeout_s": 180,
        "voice": "ryan",
        "seed": 42,
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
    pipeline = QWEN3_TTS_CRISP_HYBRID_PIPELINE
    configs = [stage.to_omegaconf() for stage in merge_pipeline_deploy(pipeline, deploy)]
    runtime = StageRuntime(configs, "local-qwen-tts-crisp", "", stage_init_timeout=180, async_chunk=False)
    report = {
        "scope": "complete resident Qwen3-TTS CustomVoice text-to-WAV through Omni StageRuntime/StagePool",
        "placement": "Radeon Vulkan talker/codec + CPU FP32 code predictor",
        "warmup_count": args.warmups,
        "measured_count": args.repeats,
        "memory_budget": deploy.stages[0].resource_budget,
    }
    try:
        started = time.perf_counter()
        runtime.initialize()
        report["startup_s"] = time.perf_counter() - started
        pool = runtime.stage_pools[0]
        report["execution_plan"] = pool.stage_client.execution_plan
        state = SimpleNamespace(sampling_params_list=[None])

        async def request_one(text: str, *, save: bool = False) -> dict:
            request_id = f"tts-{time.monotonic_ns()}"
            started = time.perf_counter()
            await pool.submit_initial(request_id, state, {"text": text, "voice": "ryan", "seed": 42})
            deadline = time.monotonic() + 180
            while True:
                output = pool.poll_graph_output(0)
                if output is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Omni CrispASR stage request timed out")
                await asyncio.sleep(0.002)
            wall_s = time.perf_counter() - started
            try:
                if output.error:
                    raise RuntimeError(output.error)
                payload = output.multimodal_output
                audio, sample_rate = payload["audio"], payload["sr"]
                metadata = output.custom_output["audio_metadata"]
                assert sample_rate == 24000 and audio.ndim == 1
                assert len(audio) == metadata["pcm_frames"] and len(audio) > 0
                assert float(audio.square().mean().sqrt()) > 1e-5
                if save:
                    _save_wav(Path(args.output_wav), audio)
                return {
                    "text": text,
                    "wall_s": wall_s,
                    "frames": len(audio),
                    "duration_s": metadata["duration_s"],
                    "pcm_sha256": metadata["pcm_sha256"],
                    "stage_event": output.custom_output["stage_event"],
                }
            finally:
                output.release_stage_buffers()
                await asyncio.sleep(0)

        first = "Hello from the local computer."
        second = "The blue car is parked beside the library."
        report["checks"] = [await request_one(first), await request_one(second)]
        report["warmups"] = [await request_one(first) for _ in range(args.warmups)]
        report["measured"] = [await request_one(first, save=i == 0) for i in range(args.repeats)]
        walls = [row["wall_s"] for row in report["measured"]]
        report["nearest_rank_p50_wall_s"] = _rank(walls, 0.5)
        report["nearest_rank_p95_wall_s"] = _rank(walls, 0.95)
        report["all_same_pcm_sha256"] = len({row["pcm_sha256"] for row in report["measured"]}) == 1
        if args.abort_check:
            def started_tts_prefills() -> int:
                log = Path(args.server_log).read_text(encoding="utf-8", errors="replace")
                return log.count("qwen3_tts[customvoice]: prefill role=")

            prefills_before_abort = started_tts_prefills()
            request_id = f"tts-abort-{time.monotonic_ns()}"
            await pool.submit_initial(request_id, state, {"text": second, "voice": "ryan", "seed": 42})
            deadline = time.monotonic() + 10
            while started_tts_prefills() <= prefills_before_abort:
                if time.monotonic() >= deadline:
                    raise TimeoutError("abort request did not reach the owned TTS server")
                await asyncio.sleep(0.02)
            premature = pool.poll_graph_output(0)
            if premature is not None:
                premature.release_stage_buffers()
                raise RuntimeError("TTS request completed before in-flight cancellation")
            await pool.abort_requests([request_id])
            await asyncio.sleep(0.1)
            stale = pool.poll_graph_output(0)
            report["abort_check"] = {
                "prefills_before_abort_request": prefills_before_abort,
                "prefills_after_abort_request_start": started_tts_prefills(),
                "stale_output": stale is not None,
                "worker_exited": pool.stage_client._proc.poll() is not None,
                "ledger_after_abort": runtime.resource_ledger.snapshot(),
            }
            if stale is not None:
                stale.release_stage_buffers()
            if report["abort_check"]["stale_output"] or not report["abort_check"]["worker_exited"]:
                raise RuntimeError("CrispASR cancellation left stale output or live worker")
            if args.restart_check:
                old_generation = report["execution_plan"]["worker_generation"]
                runtime.shutdown()
                released = runtime.resource_ledger.snapshot()
                if released["reserved"]["host_ram"] or released["quarantined"]:
                    raise RuntimeError("cancelled TTS stage retained a host-RAM reservation")
                server_log = Path(args.server_log)
                restart_log = server_log.with_name(
                    server_log.stem + "_restart" + server_log.suffix
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
                    restart_configs, "local-qwen-tts-crisp-restart", "",
                    stage_init_timeout=180, async_chunk=False,
                )
                restarted_at = time.perf_counter()
                runtime.initialize()
                fresh_startup_s = time.perf_counter() - restarted_at
                pool = runtime.stage_pools[0]
                state = SimpleNamespace(sampling_params_list=[None])
                fresh_plan = pool.stage_client.execution_plan
                if fresh_plan["worker_generation"] == old_generation:
                    raise RuntimeError("restart reused the cancelled TTS worker generation")
                restarted = await request_one(first)
                late = pool.poll_graph_output(0)
                report["restart_check"] = {
                    "first_runtime_ledger_after_shutdown": released,
                    "fresh_startup_s": fresh_startup_s,
                    "fresh_server_log": str(restart_log),
                    "fresh_execution_plan": fresh_plan,
                    "fresh_request": restarted,
                    "same_pcm_as_before_abort": restarted["pcm_sha256"] == report["checks"][0]["pcm_sha256"],
                    "late_output": late is not None,
                }
                if late is not None:
                    late.release_stage_buffers()
                if late is not None or not report["restart_check"]["same_pcm_as_before_abort"]:
                    raise RuntimeError("fresh TTS stage emitted extra or changed audio")
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        runtime.shutdown()
        if runtime.resource_ledger is not None:
            report["ledger_after_shutdown"] = runtime.resource_ledger.snapshot()
        destination = Path(args.output_report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
