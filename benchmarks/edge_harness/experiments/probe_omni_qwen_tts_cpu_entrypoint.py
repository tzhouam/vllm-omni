#!/usr/bin/env python3
"""Verify public AsyncOmni WAV delivery from the isolated CPU Qwen3-TTS worker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "python_bin", "python_sha256", "model_dir", "overlay_dir", "talker_sha256",
        "tokenizer_sha256", "server_log", "output_report",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--overlay-marker-sha256")
    parser.add_argument("--abort-recovery", action="store_true",
                        help="abort after worker start, then verify a fresh-stage WAV")
    parser.add_argument("--expected-frames", type=int, default=109440)
    parser.add_argument(
        "--expected-pcm-sha256",
        default="c8a49b5c73efd642a1eb04be973d6aefdc8ed9cfd18b34ff67690b8464906bed",
    )
    args = parser.parse_args()

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_CPU_WHOLE_PIPELINE

    backend = {
        "name": "external.qwen_tts.cpu.v1",
        "python_bin": args.python_bin,
        "python_sha256": args.python_sha256,
        "model_dir": args.model_dir,
        "overlay_dir": args.overlay_dir,
        "overlay_marker_sha256": args.overlay_marker_sha256,
        "talker_sha256": args.talker_sha256,
        "tokenizer_sha256": args.tokenizer_sha256,
        "log_file": args.server_log,
        "memory_overhead_bytes": 6 << 30,
        "expected_torch": "2.13.0+cu130",
        "max_text_bytes": 4096,
        "max_wav_bytes": 4 << 20,
        "max_audio_s": 10,
        "request_timeout_s": 180,
        "start_timeout_s": 180,
    }
    report = {"entrypoint": "AsyncOmni.generate", "pipeline": QWEN3_TTS_CPU_WHOLE_PIPELINE.model_type}
    engine = None
    try:
        with tempfile.TemporaryDirectory(prefix="omni-qwen-cpu-tts-") as temporary:
            deployment = Path(temporary) / "deploy.yaml"
            deployment.write_text(yaml.safe_dump({
                "pipeline": QWEN3_TTS_CPU_WHOLE_PIPELINE.model_type,
                "async_chunk": False,
                "stages": [{
                    "stage_id": 0,
                    "backend": backend,
                    "resource_budget": {
                        "capacities": {"host_ram": 16 << 30},
                        "demands": {"host_ram": 10 << 30},
                    },
                }],
            }), encoding="utf-8")
            started = time.perf_counter()
            engine = AsyncOmni(
                model=str(Path(args.model_dir).resolve(strict=True)),
                deploy_config=str(deployment), stage_init_timeout=180, init_timeout=240,
            )
            report["startup_s"] = time.perf_counter() - started
            outputs = []
            started = time.perf_counter()
            async for output in engine.generate(
                {"text": "Hello from the local computer.", "voice": "Ryan", "seed": 42},
                request_id="public-cpu-tts-1",
            ):
                if output.error:
                    raise RuntimeError(output.error)
                audio = output.multimodal_output["audio"]
                pcm = np.rint(audio.numpy() * 32768).astype("<i2").tobytes()
                outputs.append({
                    "request_id": output.request_id,
                    "sample_rate": output.multimodal_output["sr"],
                    "frames": len(audio),
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "stage_event": output.custom_output.get("stage_event"),
                })
            report["request_wall_s"] = time.perf_counter() - started
            report["outputs"] = outputs
            assert len(outputs) == 1 and outputs[0]["request_id"] == "public-cpu-tts-1"
            assert outputs[0]["sample_rate"] == 24000 and outputs[0]["frames"] == args.expected_frames
            assert outputs[0]["pcm_sha256"] == args.expected_pcm_sha256
            if args.abort_recovery:
                abort_id = "public-cpu-tts-abort"
                abort_text = "The blue car is parked beside the library."
                abort_outputs = []

                async def consume_abort():
                    async for output in engine.generate(
                        {"text": abort_text, "voice": "Ryan", "seed": 42},
                        request_id=abort_id,
                    ):
                        abort_outputs.append({
                            "request_id": output.request_id,
                            "error": str(output.error) if output.error else None,
                            "has_audio": bool(output.multimodal_output and
                                              "audio" in output.multimodal_output),
                        })

                pending = asyncio.create_task(consume_abort())
                marker = ("tts request-start text_sha256=" +
                          hashlib.sha256(abort_text.encode()).hexdigest()).encode()

                async def wait_for_worker_start():
                    while True:
                        if Path(args.server_log).is_file() and marker in Path(args.server_log).read_bytes():
                            return
                        if pending.done():
                            raise RuntimeError("abort candidate finished before worker start")
                        await asyncio.sleep(0.05)

                await asyncio.wait_for(wait_for_worker_start(), timeout=30)
                started = time.perf_counter()
                await engine.abort(abort_id)
                report["abort"] = {
                    "worker_started": True,
                    "worker_start_marker": marker.decode(),
                    "ack_s": time.perf_counter() - started,
                }
                await asyncio.wait_for(pending, timeout=30)
                report["abort"]["outputs_after_start"] = abort_outputs
                if (len(abort_outputs) != 1
                        or abort_outputs[0]["request_id"] != abort_id
                        or abort_outputs[0]["has_audio"]):
                    raise RuntimeError("aborted request delivered stale audio")

                engine.shutdown()
                engine = None
                restart_log = Path(args.server_log).with_name(
                    Path(args.server_log).stem + "_restarted" + Path(args.server_log).suffix
                )
                restarted_backend = {**backend, "log_file": str(restart_log)}
                restarted_deployment = Path(temporary) / "restart-deploy.yaml"
                restarted_deployment.write_text(yaml.safe_dump({
                    "pipeline": QWEN3_TTS_CPU_WHOLE_PIPELINE.model_type,
                    "async_chunk": False,
                    "stages": [{
                        "stage_id": 0,
                        "backend": restarted_backend,
                        "resource_budget": {
                            "capacities": {"host_ram": 16 << 30},
                            "demands": {"host_ram": 10 << 30},
                        },
                    }],
                }), encoding="utf-8")
                started = time.perf_counter()
                engine = AsyncOmni(
                    model=str(Path(args.model_dir).resolve(strict=True)),
                    deploy_config=str(restarted_deployment),
                    stage_init_timeout=180, init_timeout=240,
                )
                restart_startup_s = time.perf_counter() - started
                recovered = []
                started = time.perf_counter()
                async for output in engine.generate(
                    {"text": "Hello from the local computer.", "voice": "Ryan", "seed": 42},
                    request_id="public-cpu-tts-after-abort",
                ):
                    if output.error:
                        raise RuntimeError(output.error)
                    audio = output.multimodal_output["audio"]
                    pcm = np.rint(audio.numpy() * 32768).astype("<i2").tobytes()
                    recovered.append({
                        "request_id": output.request_id,
                        "sample_rate": output.multimodal_output["sr"],
                        "frames": len(audio),
                        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                        "stage_event": output.custom_output.get("stage_event"),
                    })
                report["recovery"] = {
                    "mode": "fresh_stage_after_inflight_abort",
                    "startup_s": restart_startup_s,
                    "request_wall_s": time.perf_counter() - started,
                    "worker_log": str(restart_log),
                    "outputs": recovered,
                }
                if (len(recovered) != 1 or recovered[0]["sample_rate"] != 24000
                        or recovered[0]["frames"] != args.expected_frames
                        or recovered[0]["pcm_sha256"] != args.expected_pcm_sha256
                        or recovered[0]["stage_event"]["worker_generation"]
                        == outputs[0]["stage_event"]["worker_generation"]):
                    raise RuntimeError("fresh-stage PCM recovery differs from baseline")
            report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        destination = Path(args.output_report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
