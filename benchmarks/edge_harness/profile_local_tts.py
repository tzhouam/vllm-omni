# SPDX-License-Identifier: Apache-2.0
"""Profile real Qwen3-TTS streams through AsyncOmni, retaining terminal audio separately."""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from gpu_telemetry import GpuTelemetry
from profile_local_text import HOST_ENVIRONMENT, save


async def run(args):
    import torch
    from transformers import AutoTokenizer

    from vllm_omni import AsyncOmni
    from vllm_omni.edge.local.engine import MemorySampler
    from vllm_omni.edge.local.manifest import runtime_versions
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tts"))
    from stream_latency_bench import _cat_audio, _sample_rate
    from stream_latency_metrics import request_metrics, underruns_if_start_at

    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "argv": sys.argv,
        "python": sys.executable,
        "host_environment": HOST_ENVIRONMENT,
        "platform": platform.platform(),
        "runtime": runtime_versions().to_dict(),
        "settings": vars(args),
        "start_unix": time.time(),
        "requests": 0,
        "cache_condition": "existing disk/JIT caches; not cold disk",
        "quality_gate": "finite PCM checks only; ASR/speaker/listening reference gates pending",
        "playback_scope": "arrival-based simulated playback; no sound-device measurement",
        "terminal_audio": (
            "DELTA output required; terminal audio is included once in complete playback metrics "
            "and archived separately"
        ),
    }
    save(args.out / "report.json", report)
    sampler = MemorySampler()
    sampler.start()
    gpu_telemetry = None
    if args.gpu_telemetry_interval_s > 0:
        gpu_telemetry = GpuTelemetry(args.out / "gpu_telemetry.jsonl", args.gpu_telemetry_interval_s)
        gpu_telemetry.start()
    omni = None
    try:
        cfg = Qwen3TTSConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        texts = {
            "short": "Hello, this is a speech test.",
            "medium": (
                "The garden has trees, flowers, and a small pond. We walk along the path and listen to the birds."
            ),
            "long": (
                "Today we are testing a local speech system. "
                "It should speak clearly and keep the audio flowing smoothly. " * 4
            ),
        }
        prompts = {}
        for name, text in texts.items():
            info = {"task_type": ["CustomVoice"], "text": [text], "language": ["English"], "speaker": ["Vivian"]}
            length = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                additional_information=info,
                task_type="CustomVoice",
                tokenize_prompt=lambda x: tokenizer(x, padding=False)["input_ids"],
                codec_language_id=cfg.talker_config.codec_language_id,
                spk_is_dialect=cfg.talker_config.spk_is_dialect,
            )
            prompts[name] = {"prompt_token_ids": [0] * length, "additional_information": info}
        report["prompts"] = prompts
        start = time.perf_counter()
        omni = AsyncOmni(model=args.model, deploy_profile="edge", stage_init_timeout=180)
        report["startup_s"] = time.perf_counter() - start
        from vllm.sampling_params import RequestOutputKind

        sampling_params = omni.resolve_sampling_params_list(None, allow_delta_coercion=True)
        report["sampling_params"] = [repr(p) for p in sampling_params]
        if sampling_params[-1].output_kind != RequestOutputKind.DELTA:
            raise RuntimeError("Final TTS stage must emit DELTA audio for this playback benchmark")
        report["memory_after_load"] = sampler.peaks()
        save(args.out / "report.json", report)
        counter = 0
        samples = (args.out / "requests.jsonl").open("a", encoding="utf-8")

        async def request(name, concurrency, phase, slow=False):
            nonlocal counter
            counter += 1
            rid = f"{phase}-{name}-{concurrency}-{counter}"
            chunks, buffers = [], []
            submitted_unix = time.time()
            start = time.perf_counter()
            finished, sr = False, 24000
            async for output in omni.generate(prompts[name], request_id=rid, sampling_params_list=sampling_params):
                if output.request_id != rid:
                    raise RuntimeError(f"Unexpected output request ID: {output.request_id}; expected {rid}")
                elapsed = (time.perf_counter() - start) * 1000
                mm = output.outputs[0].multimodal_output if output.outputs else None
                sr = _sample_rate(mm, sr)
                audio = _cat_audio(mm.get("audio") if mm else None)
                raw = audio.numpy().tobytes() if audio is not None else b""
                finite = bool(torch.isfinite(audio).all()) if audio is not None else True
                chunks.append(
                    {
                        "idx": len(chunks),
                        "t_ms": elapsed,
                        "samples": len(raw) // 4,
                        "sr": sr,
                        "request_id": output.request_id,
                        "terminal": output.finished,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "finite": finite,
                    }
                )
                if phase == "warmup":
                    buffers.append(raw)
                finished = finished or output.finished
                if not finite:
                    raise RuntimeError(f"Nonfinite audio: {rid}")
                if slow:
                    await asyncio.sleep(0.2)
            wall = time.perf_counter() - start
            streamed = [c for c in chunks if not c["terminal"]]
            terminal = [c for c in chunks if c["terminal"]]
            rec = {
                "request_id": rid,
                "phase": phase,
                "length_band": name,
                "submitted_unix": submitted_unix,
                "finished_unix": time.time(),
                "concurrency": concurrency,
                "sr": sr,
                "chunks_all": chunks,
                "finished": finished,
                "terminal_audio_s": sum(c["samples"] for c in terminal) / sr,
                "prefix_metrics": request_metrics(streamed, sr, wall),
                **request_metrics(chunks, sr, wall),
            }
            rec["underruns"] = underruns_if_start_at(chunks, sr, rec["ttfa_ms"]) if rec["ttfa_ms"] is not None else []
            # Preserve each raw warmup chunk, including the terminal event, without guessing concatenation semantics.
            if buffers:
                audio_dir = args.out / "audio_chunks"
                audio_dir.mkdir(exist_ok=True)
                for i, raw in enumerate(buffers):
                    (audio_dir / f"{rid}-{i}.f32").write_bytes(raw)
            samples.write(json.dumps(rec) + "\n")
            samples.flush()
            if not finished or not any(c["samples"] for c in chunks):
                raise RuntimeError(f"Missing completion/audio: {rid}")
            report["requests"] = counter
            save(args.out / "report.json", report)

        for concurrency in (args.concurrency,) if args.concurrency is not None else (1, 2, 4):
            for name in (args.length_band,) if args.length_band else prompts:
                await asyncio.gather(*(request(name, concurrency, "warmup") for _ in range(concurrency)))
                for i in range(0, args.repeats, concurrency):
                    await asyncio.gather(
                        *(request(name, concurrency, "measured") for _ in range(min(concurrency, args.repeats - i)))
                    )
                print(f"profiled {name} concurrency={concurrency}", flush=True)
        report["sustained_wall_s"] = 0.0
        if args.sustained_seconds > 0:
            start = time.perf_counter()
            report["sustained_start_unix"] = time.time()
            while time.perf_counter() - start < args.sustained_seconds:
                await request("medium", 1, "sustained")
            report["sustained_wall_s"] = time.perf_counter() - start
        await request("medium", 1, "slow_consumer", slow=True)
        stream = omni.generate(prompts["long"], request_id="cancel-probe", sampling_params_list=sampling_params)
        async for output in stream:
            if output.outputs:
                break
        start = time.perf_counter()
        await omni.abort("cancel-probe")
        await stream.aclose()
        report["cancel"] = {
            "abort_ack_s": time.perf_counter() - start,
            "scope": "abort acknowledgement and subsequent request; late-event gate not verified",
        }
        await request("short", 1, "after_cancel")
        samples.close()
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if omni is not None:
            omni.shutdown()
        sampler.stop()
        report["memory"] = sampler.peaks()
        if gpu_telemetry is not None:
            report["gpu_telemetry"] = gpu_telemetry.stop()
        report["end_unix"] = time.time()
        save(args.out / "report.json", report)
    return report["status"] == "completed"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--repeats", default=20, type=int)
    p.add_argument("--sustained-seconds", default=1800, type=float)
    p.add_argument("--gpu-telemetry-interval-s", default=0.0, type=float,
                   help="Sample device-wide NVML and host telemetry; 0 disables sampling.")
    p.add_argument(
        "--length-band",
        choices=("short", "medium", "long"),
        help="Select one band for a separate instrumentation diagnostic.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        choices=(1, 2, 4),
        help="Select one concurrency for a separate instrumentation diagnostic.",
    )
    args = p.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm_omni.windows.aio import install_selector_policy

    install_selector_policy()
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
