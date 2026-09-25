#!/usr/bin/env python3
"""Run a pinned serial MiniCPM-o image-to-text+speech suite in one Omni session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch
import vllm

from benchmarks.edge_harness.profile_minicpmo_text_speech import MemorySampler, _run_request
from examples.offline_inference.minicpmo.end2end import (
    get_audio_image_query,
    get_image_query,
)
from vllm_omni.entrypoints.omni import Omni


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--deploy-config", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--audio", type=Path,
                        help="add the same spoken-audio input to each image request")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-timeout", type=int, default=900)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    args = parser.parse_args()
    if len(args.image) < 2 or any(not path.is_file() for path in args.image):
        parser.error("provide at least two existing --image files")
    if args.audio is not None and not args.audio.is_file():
        parser.error("--audio must name an existing WAV file")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    images = [
        {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prompt": {
                **(get_audio_image_query(audio_path=str(args.audio), image_path=str(path),
                                         use_tts=True).inputs if args.audio is not None else
                   get_image_query(image_path=str(path), use_tts=True).inputs),
                "modalities": ["text", "audio"],
            },
        }
        for path in args.image
    ]
    rows = []
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
            with args.output.with_suffix(".jsonl").open("w", encoding="utf-8") as raw:
                for index, item in enumerate(images):
                    row = _run_request(omni, item["prompt"], sampler)
                    row.update(index=index, image_path=item["path"], image_sha256=item["sha256"])
                    if args.audio is not None:
                        row["audio_sha256"] = hashlib.sha256(args.audio.read_bytes()).hexdigest()
                    rows.append(row)
                    raw.write(json.dumps(row) + "\n")
                    raw.flush()
                    print(f"image={index} wall_s={row['wall_s']:.3f} text={row['text']!r}", flush=True)
        finally:
            omni.close()

    output = {
        "scope": ("serial audio+image MiniCPM-o input to text+speech in one Omni session"
                  if args.audio is not None else
                  "serial image-only MiniCPM-o input to text+speech in one Omni session"),
        "audio_path": str(args.audio.resolve()) if args.audio is not None else None,
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest()
        if args.audio is not None else None,
        "model": str(args.model.resolve()),
        "deploy_config": str(args.deploy_config.resolve()),
        "deploy_config_sha256": hashlib.sha256(args.deploy_config.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "memory_scope": sampler.scope,
        "environment": {
            name: os.environ.get(name)
            for name in ("VLLM_TARGET_DEVICE", "VLLM_CPU_KVCACHE_SPACE", "OMP_NUM_THREADS",
                         "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "VLLM_ENABLE_V1_MULTIPROCESSING")
        },
        "startup_s": startup_s,
        "serial_requests": len(rows),
        "requests": rows,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
