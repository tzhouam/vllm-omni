#!/usr/bin/env python3
"""Profile the isolated official Qwen3-TTS CPU wrapper through Omni."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import subprocess
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
        "python_bin", "python_sha256", "model_dir", "overlay_dir", "talker_sha256",
        "tokenizer_sha256", "server_log", "output_report", "output_wav",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--reserve-gib", type=int, default=10)
    parser.add_argument("--abort-check", action="store_true")
    parser.add_argument("--allow-platform-variation", action="store_true")
    parser.add_argument("--overlay-marker-sha256")
    parser.add_argument("--expect-admission-refusal", action="store_true")
    parser.add_argument("--require-amd-npu-present", action="store_true",
                        help="qualify explicit CPU fallback on an HX370 host with its NPU detected")
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0 or not 0 < args.reserve_gib <= 16:
        parser.error("invalid profile or budget")

    from vllm_omni.config.stage_config import DeployConfig, StageDeployConfig, merge_pipeline_deploy
    from vllm_omni.engine.stage_runtime import StageRuntime
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
    deploy = DeployConfig(
        async_chunk=False,
        stages=[StageDeployConfig(
            stage_id=0,
            backend=backend,
            resource_budget={
                "capacities": {"host_ram": 16 << 30},
                "demands": {"host_ram": args.reserve_gib << 30},
            },
        )],
    )
    pipeline = QWEN3_TTS_CPU_WHOLE_PIPELINE
    configs = [stage.to_omegaconf() for stage in merge_pipeline_deploy(pipeline, deploy)]
    runtime = StageRuntime(configs, "local-qwen-tts-cpu", "", stage_init_timeout=180, async_chunk=False)
    report = {
        "scope": "complete resident local CPU Qwen3-TTS text-to-WAV through Omni",
        "host_os": platform.platform(),
        "allow_platform_variation": args.allow_platform_variation,
        "warmup_count": args.warmups,
        "measured_count": args.repeats,
        "memory_budget": deploy.stages[0].resource_budget,
    }
    if args.require_amd_npu_present:
        if platform.system() != "Windows":
            raise RuntimeError("AMD NPU host fallback probe requires native Windows")
        command = (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.PNPDeviceID -like 'PCI\\VEN_1022&DEV_17F0*' } | "
            "Select-Object Name,PNPDeviceID,Status | ConvertTo-Json -Compress"
        )
        detected = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, check=True,
        )
        raw = detected.stdout.strip()
        devices = json.loads(raw) if raw else []
        if isinstance(devices, dict):
            devices = [devices]
        ready = [device for device in devices if device.get("Status") == "OK"]
        if not ready:
            raise RuntimeError("HX370 AMD NPU was not detected as healthy")
        report["accelerator_inventory"] = [
            {"name": device["Name"], "status": device["Status"],
             "pci_id": "VEN_1022&DEV_17F0", "evidence": "D"}
            for device in ready
        ]
        report["placement_policy"] = {
            "requested_backend": "external.qwen_tts.cpu.v1",
            "selected_device": "cpu",
            "npu_execution_claimed": False,
            "reason": "NPU MLP component passes offline waveform checks but fails the measured benefit gate",
        }
    sample_task = None
    stop_sampling = asyncio.Event()
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
            raise AssertionError("Qwen CPU TTS accepted an insufficient reservation")
        runtime.initialize()
        report["startup_s"] = time.perf_counter() - started
        pool = runtime.stage_pools[0]
        report["execution_plan"] = pool.stage_client.execution_plan
        if args.require_amd_npu_present:
            plan = report["execution_plan"]
            if (plan["backend"] != "external.qwen_tts.cpu.v1"
                    or plan["worker_props"]["placement"] != "cpu"):
                raise RuntimeError("CPU fallback plan executed on an unexpected backend")
        state = SimpleNamespace(sampling_params_list=[None])
        import psutil

        memory_samples = []

        async def sample_memory() -> None:
            while not stop_sampling.is_set():
                try:
                    root = psutil.Process(pool.stage_client._proc.pid)
                    members = [root, *root.children(recursive=True)]
                    memories = [process.memory_full_info() for process in members]
                    memory_samples.append({
                        "unix": time.time(),
                        "rss_bytes": sum(int(memory.rss) for memory in memories),
                        "private_bytes": sum(
                            int(getattr(memory, "private", memory.rss)) for memory in memories
                        ),
                        "host_available_bytes": psutil.virtual_memory().available,
                        "pids": [process.pid for process in members],
                    })
                except psutil.Error:
                    break
                await asyncio.sleep(0.25)

        sample_task = asyncio.create_task(sample_memory())

        async def request_one(text: str, *, save: bool = False) -> dict:
            request_id = f"cpu-tts-{time.monotonic_ns()}"
            started = time.perf_counter()
            await pool.submit_initial(request_id, state, {"text": text, "voice": "Ryan", "seed": 42})
            deadline = time.monotonic() + 180
            while True:
                output = pool.poll_graph_output(0)
                if output is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Omni Qwen CPU TTS stage request timed out")
                await asyncio.sleep(0.005)
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
                pcm = np.rint(audio.numpy() * 32768).astype("<i2").tobytes()
                assert hashlib.sha256(pcm).hexdigest() == metadata["pcm_sha256"]
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
        expected = (
            "c8a49b5c73efd642a1eb04be973d6aefdc8ed9cfd18b34ff67690b8464906bed",
            "4edc7a62f749b1fde5ea39bedbc15a35f291ab1db8d24d7b7eb3df7270885e16",
        )
        report["standalone_pcm_parity"] = [
            row["pcm_sha256"] == digest for row, digest in zip(report["checks"], expected)
        ]
        if report["standalone_pcm_parity"] != [True, True] and not args.allow_platform_variation:
            raise RuntimeError("CPU Omni PCM differs from the same checkpoint's standalone output")
        report["warmups"] = [await request_one(first) for _ in range(args.warmups)]
        report["measured"] = [await request_one(first, save=i == 0) for i in range(args.repeats)]
        walls = [row["wall_s"] for row in report["measured"]]
        report["nearest_rank_p50_wall_s"] = _rank(walls, 0.5)
        report["nearest_rank_p95_wall_s"] = _rank(walls, 0.95)
        report["all_same_pcm_sha256"] = len({row["pcm_sha256"] for row in report["measured"]}) == 1
        report["memory_samples"] = memory_samples
        report["sampled_max_rss_bytes"] = max((row["rss_bytes"] for row in memory_samples), default=None)
        report["sampled_max_private_bytes"] = max(
            (row["private_bytes"] for row in memory_samples), default=None
        )
        if args.abort_check:
            request_id = f"cpu-tts-abort-{time.monotonic_ns()}"
            await pool.submit_initial(request_id, state, {"text": second, "voice": "Ryan", "seed": 42})
            await asyncio.sleep(0.01)
            await pool.abort_requests([request_id])
            await asyncio.sleep(0.1)
            report["abort_check"] = {
                "stale_output": pool.poll_graph_output(0) is not None,
                "worker_exited": pool.stage_client._proc.poll() is not None,
                "ledger_after_abort": runtime.resource_ledger.snapshot(),
            }
            if report["abort_check"]["stale_output"] or not report["abort_check"]["worker_exited"]:
                raise RuntimeError("CPU TTS cancellation left stale output or live worker")
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_sampling.set()
        if sample_task is not None:
            await sample_task
        runtime.shutdown()
        if runtime.resource_ledger is not None:
            report["ledger_after_shutdown"] = runtime.resource_ledger.snapshot()
        destination = Path(args.output_report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
