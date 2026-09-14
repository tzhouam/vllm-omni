# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Arguments and event loading for the spatial VAE benchmark."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_CAMERA_ACTION_SCHEMA = "lingbot.camera_actions.v1"
_FRAMES_PER_BLOCK = 3
_MAX_REALTIME_TICKS = 10


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LingBot-World 2.0 realtime TP/SP/compile benchmark.")
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--session-id", default="lingbot-world-bench")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--ulysses-degree", type=int, default=4)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--label", default="lingbot_bench")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--port", type=int, default=29500)
    p.add_argument("--case", choices=["D", "E"], default="D")
    args = p.parse_args(argv)
    if args.height % 16 or args.width % 16 or min(args.height, args.width) <= 0:
        p.error("height and width must be positive multiples of 16")
    if not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= 1:
        p.error("gpu-memory-fraction must be in (0, 1]")
    if args.epochs < 2:
        p.error("at least two epochs are required (warmup and measured)")
    if args.tensor_parallel_size != 1 or args.ulysses_degree not in (2, 4):
        p.error("this benchmark supports TP1 with Ulysses2 or Ulysses4")
    if not 1024 <= args.port <= 65525:
        p.error("port must be between 1024 and 65525")
    return args


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        value = json.loads(line)
        frames = value.get("frames")
        if frames is not None and (not isinstance(frames, list) or len(frames) != _FRAMES_PER_BLOCK):
            raise ValueError(f"events line {line_number}: frames must contain exactly three lists.")
        events.append({"event_id": int(value["event_id"]), "prompt": value.get("prompt"), "frames": frames})
    if not events or len(events) > _MAX_REALTIME_TICKS:
        raise ValueError("events file must contain 1..10 events.")
    return events
