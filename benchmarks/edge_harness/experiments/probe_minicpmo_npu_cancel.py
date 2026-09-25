#!/usr/bin/env python3
"""Abort a live MiniCPM-o audio+image request, then complete a fresh one."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import vllm

from examples.offline_inference.minicpmo.end2end import get_audio_image_query
from vllm_omni.entrypoints.async_omni import AsyncOmni


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def summarize(output) -> dict:
    multimodal = getattr(output.outputs[0], "multimodal_output", None) if output.outputs else None
    audio = multimodal.get("audio") if multimodal else None
    if isinstance(audio, list):
        audio = torch.cat([torch.as_tensor(part).flatten() for part in audio])
    if audio is not None:
        audio = torch.as_tensor(audio).detach().float().cpu().numpy().reshape(-1)
    return {
        "request_id": output.request_id,
        "kind": output.final_output_type,
        "finished": bool(output.finished),
        "text": output.outputs[0].text if output.outputs and output.final_output_type == "text" else None,
        "token_ids": (list(output.outputs[0].token_ids)
                      if output.outputs and output.final_output_type == "text" else None),
        "audio_samples": int(audio.size) if audio is not None else 0,
        "audio_finite": bool(np.isfinite(audio).all()) if audio is not None else None,
        "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest() if audio is not None else None,
    }


async def run(args: argparse.Namespace) -> dict:
    model = args.model.resolve(strict=True)
    config = args.deploy_config.resolve(strict=True)
    image = args.image.resolve(strict=True)
    fresh_image = args.fresh_image.resolve(strict=True)
    recorded_audio = args.audio.resolve(strict=True)
    query = get_audio_image_query(audio_path=str(recorded_audio), image_path=str(image), use_tts=True)
    prompt = {**query.inputs, "modalities": ["text", "audio"]}
    fresh_query = get_audio_image_query(
        audio_path=str(recorded_audio), image_path=str(fresh_image), use_tts=True
    )
    fresh_prompt = {**fresh_query.inputs, "modalities": ["text", "audio"]}
    report = {
        "scope": "one cancelled and one fresh complete natural audio+image MiniCPM-o request",
        "model": str(model),
        "deploy_config": str(config),
        "deploy_config_sha256": sha256(config),
        "image_sha256": sha256(image),
        "fresh_image_sha256": sha256(fresh_image),
        "input_audio_sha256": sha256(recorded_audio),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "environment": {key: os.environ.get(key) for key in (
            "VLLM_TARGET_DEVICE", "VLLM_CPU_KVCACHE_SPACE", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "VLLM_ENABLE_V1_MULTIPROCESSING",
        )},
        "status": "started",
    }
    engine = None
    try:
        started = time.perf_counter()
        engine = AsyncOmni(
            model=str(model), deploy_config=str(config), trust_remote_code=True,
            init_timeout=args.init_timeout, stage_init_timeout=args.stage_init_timeout,
        )
        report["startup_s"] = time.perf_counter() - started
        stream = engine.generate(prompt, request_id="minicpmo-cancel", output_modalities=["text", "audio"])
        first = None
        started = time.perf_counter()
        async for output in stream:
            row = summarize(output)
            if row["request_id"] != "minicpmo-cancel":
                raise RuntimeError("cancel stream mixed request IDs")
            if row["kind"] == "text" and row["text"]:
                first = row
                break
        if first is None:
            raise RuntimeError("cancel candidate finished without a text event")
        report["cancel_before"] = {"first_text_s": time.perf_counter() - started, "event": first}
        started = time.perf_counter()
        await engine.abort("minicpmo-cancel")
        report["abort_ack_s"] = time.perf_counter() - started
        late = []
        try:
            async def drain():
                async for output in stream:
                    late.append(summarize(output))
            await asyncio.wait_for(drain(), timeout=args.drain_timeout)
            report["abort_terminal"] = "stream_ended"
        except (asyncio.CancelledError, asyncio.TimeoutError) as exc:
            report["abort_terminal"] = type(exc).__name__
        finally:
            await stream.aclose()
        report["late_events"] = late
        report["late_audio_samples"] = sum(row["audio_samples"] for row in late)
        if report["late_audio_samples"]:
            raise RuntimeError("cancelled request emitted late audio")

        fresh = []
        started = time.perf_counter()
        async for output in engine.generate(
            fresh_prompt, request_id="minicpmo-after-cancel", output_modalities=["text", "audio"]
        ):
            row = summarize(output)
            if row["request_id"] != "minicpmo-after-cancel":
                raise RuntimeError("fresh stream mixed request IDs")
            fresh.append(row)
        report["fresh_wall_s"] = time.perf_counter() - started
        report["fresh_events"] = fresh
        report["fresh_text_events"] = sum(bool(row["text"]) for row in fresh)
        report["fresh_audio_samples"] = sum(row["audio_samples"] for row in fresh)
        if (not report["fresh_text_events"] or report["fresh_audio_samples"] <= 0
                or any(row["audio_finite"] is False for row in fresh)):
            raise RuntimeError("fresh request lacked nonempty text or finite audio")
        report["status"] = "cancel_then_fresh_complete_pass"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        if args.npu_events.is_file():
            events = [json.loads(line) for line in args.npu_events.read_text().splitlines()]
            placements = [row["placement"] for row in events if row["phase"] == "placement"]
            completed = [row for row in events if row["phase"] == "request_complete"]
            runs = [row for row in events if row["phase"] == "run"]
            closes = [row for row in events if row["phase"] == "close"]
            report["npu_audit"] = {
                "placement_target_nodes": placements[0]["target_nodes"] if len(placements) == 1 else None,
                "complete_request_numbers": [row["request"] for row in completed],
                "graph_calls": len(runs),
                "close_calls": closes[0]["calls"] if len(closes) == 1 else None,
                "projection_relative_l2": [row.get("projection_relative_l2") for row in completed],
            }
            if (report["status"] == "cancel_then_fresh_complete_pass"
                    and (len(placements) != 1 or placements[0]["target_nodes"] < 1
                         or [row["request"] for row in completed] != [1, 2]
                         or len(runs) != args.expected_graph_calls
                         or len(closes) != 1 or closes[0]["calls"] != len(runs)
                         or any(row.get("projection_relative_l2", 1) > .01
                                for row in completed))):
                report["status"] = "failed"
                report["error"] = "fresh request lacked two verified NPU request groups"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "deploy-config", "image", "fresh-image", "audio", "output", "npu-events"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-graph-calls", type=int, required=True)
    parser.add_argument("--init-timeout", type=int, default=900)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument("--drain-timeout", type=float, default=15)
    args = parser.parse_args()
    report = asyncio.run(run(args))
    print(json.dumps({key: report.get(key) for key in (
        "status", "startup_s", "abort_ack_s", "late_audio_samples",
        "fresh_wall_s", "fresh_text_events", "fresh_audio_samples",
    )}), flush=True)
    if report["status"] != "cancel_then_fresh_complete_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
