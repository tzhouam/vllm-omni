#!/usr/bin/env python3
"""Measure serial real-weight MiniCPM-o input-to-text+speech requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import vllm

from examples.offline_inference.minicpmo.end2end import (
    get_audio_query,
    get_image_query,
    get_text_query,
)
from vllm_omni.entrypoints.omni import Omni


class MemorySampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.scope = "wsl_guest" if "microsoft" in platform.release().lower() else "native_os"
        self.stop = threading.Event()
        self.samples: list[dict[str, int | float]] = []
        self.thread = threading.Thread(target=self._run, name="memory-sampler", daemon=True)

    def _run(self) -> None:
        parent = psutil.Process()
        while not self.stop.is_set():
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            rss_sum = 0
            for proc in [parent, *parent.children(recursive=True)]:
                try:
                    rss_sum += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.samples.append(
                {
                    "monotonic_s": time.perf_counter(),
                    "system_used_bytes": memory.used,
                    "system_available_bytes": memory.available,
                    "swap_used_bytes": swap.used,
                    "process_tree_rss_sum_bytes": rss_sum,
                }
            )
            self.stop.wait(self.interval_s)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop.set()
        self.thread.join()


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _run_request(omni: Omni, prompt: dict, memory_sampler: MemorySampler) -> dict:
    started = time.perf_counter()
    first_text_s = None
    first_audio_s = None
    text = None
    token_ids = None
    audio_samples = None
    audio_rms = None
    request_ids = set()

    for output in omni.generate([prompt], None):
        request_ids.add(output.request_id)
        if output.final_output_type == "text":
            first_text_s = time.perf_counter() - started
            text = output.outputs[0].text.strip()
            token_ids = list(output.outputs[0].token_ids)
        elif output.final_output_type == "audio":
            first_audio_s = time.perf_counter() - started
            audio = output.outputs[0].multimodal_output["audio"]
            if isinstance(audio, list):
                audio = torch.cat(
                    [
                        part.flatten() if isinstance(part, torch.Tensor) else torch.as_tensor(part).flatten()
                        for part in audio
                    ]
                )
            wave = audio.detach().float().cpu().numpy().reshape(-1)
            if not np.isfinite(wave).all():
                raise RuntimeError("MiniCPM-o generated non-finite audio")
            audio_samples = int(wave.size)
            audio_rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))

    completed = time.perf_counter()
    if not text or not token_ids or not audio_samples or audio_rms == 0:
        raise RuntimeError("MiniCPM-o did not produce both nonempty text and nonzero audio")
    samples = [sample for sample in memory_sampler.samples if started <= sample["monotonic_s"] <= completed]
    sampled_memory = {
        "scope": memory_sampler.scope,
        "n": len(samples),
        "max_system_used_bytes": max((s["system_used_bytes"] for s in samples), default=None),
        "min_system_available_bytes": min((s["system_available_bytes"] for s in samples), default=None),
        "max_swap_used_bytes": max((s["swap_used_bytes"] for s in samples), default=None),
        # Sum of RSS can double-count shared pages; it is not unique RAM.
        "max_process_tree_rss_sum_bytes": max((s["process_tree_rss_sum_bytes"] for s in samples), default=None),
    }
    if memory_sampler.scope == "wsl_guest":
        sampled_memory["max_wsl_used_bytes"] = sampled_memory["max_system_used_bytes"]
        sampled_memory["min_wsl_available_bytes"] = sampled_memory["min_system_available_bytes"]
    return {
        "request_ids": sorted(request_ids),
        "wall_s": completed - started,
        "first_text_s": first_text_s,
        "first_audio_s": first_audio_s,
        "text": text,
        "token_ids": token_ids,
        "audio_samples": audio_samples,
        "audio_duration_s_at_24khz": audio_samples / 24000,
        "audio_rms_float": audio_rms,
        "sampled_memory": sampled_memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--deploy-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--init-timeout", type=int, default=600)
    parser.add_argument("--stage-init-timeout", type=int, default=300)
    parser.add_argument("--query-type", choices=("text", "image", "audio"), default="text")
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--audio-path", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats must be positive")
    if args.query_type != "image" and args.image_path is not None:
        parser.error("--image-path is only valid for image queries")
    if args.query_type != "audio" and args.audio_path is not None:
        parser.error("--audio-path is only valid for audio queries")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.query_type == "image":
        if args.image_path is None:
            parser.error("--image-path is required for image queries")
        query = get_image_query(image_path=str(args.image_path), use_tts=True)
    elif args.query_type == "audio":
        if args.audio_path is None:
            parser.error("--audio-path is required for audio queries")
        query = get_audio_query(audio_path=str(args.audio_path), use_tts=True)
    else:
        query = get_text_query(use_tts=True)
    prompt = {**query.inputs, "modalities": ["text", "audio"]}
    with MemorySampler() as sampler:
        started = time.perf_counter()
        omni = Omni(
            model=str(args.model.resolve()),
            deploy_config=str(args.deploy_config.resolve()),
            trust_remote_code=True,
            init_timeout=args.init_timeout,
            stage_init_timeout=args.stage_init_timeout,
        )
        startup_s = time.perf_counter() - started
        try:
            rows = []
            with args.output.with_suffix(".jsonl").open("w", encoding="utf-8") as raw:
                for index in range(args.warmups + args.repeats):
                    row = _run_request(omni, prompt, sampler)
                    row["index"] = index
                    row["warmup"] = index < args.warmups
                    rows.append(row)
                    raw.write(json.dumps(row) + "\n")
                    raw.flush()
                    print(f"request={index} warmup={row['warmup']} wall_s={row['wall_s']:.3f}", flush=True)
        finally:
            omni.close()

    measured = [row for row in rows if not row["warmup"]]
    times = [row["wall_s"] for row in measured]
    output = {
        "scope": (
            f"serial one-request-at-a-time MiniCPM-o {args.query_type}-to-text+speech; "
            "no speech-quality claim"
        ),
        "query_type": args.query_type,
        "input_sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in (("image", args.image_path), ("audio", args.audio_path))
            if path is not None
        },
        "model": str(args.model.resolve()),
        "deploy_config": str(args.deploy_config.resolve()),
        "deploy_config_sha256": hashlib.sha256(args.deploy_config.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "memory_scope": sampler.scope,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "VLLM_TARGET_DEVICE",
                "VLLM_CPU_KVCACHE_SPACE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
            )
        },
        "init_timeout_s": args.init_timeout,
        "stage_init_timeout_s": args.stage_init_timeout,
        "startup_s": startup_s,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "wall_s_p50_nearest_rank": _nearest_rank(times, 0.5),
        "wall_s_p95_nearest_rank": _nearest_rank(times, 0.95),
        "requests": rows,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
