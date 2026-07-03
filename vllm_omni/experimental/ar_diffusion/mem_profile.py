# SPDX-License-Identifier: Apache-2.0
"""Env-gated GPU memory profiling for the AR-Diffusion runner.

Set ``AR_DIFFUSION_MEM_PROFILE=<out_dir>`` to record, per worker rank:

- ``mem_checkpoints_rank<N>.json`` — allocated/reserved at each lifecycle stage
  (weights loaded, KV pool built, warm-up done, first forwards) plus a
  per-component weight/buffer breakdown of the loaded pipeline.
- ``mem_snapshot_load_rank<N>.pickle`` / ``mem_snapshot_run_rank<N>.pickle`` —
  ``torch.cuda.memory._dump_snapshot`` captures (allocation stacks; view at
  https://pytorch.org/memory_viz or with torch.cuda._memory_viz).

Zero overhead when the env var is unset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_ENV = "AR_DIFFUSION_MEM_PROFILE"


class ARDiffusionMemProfiler:
    """Checkpointed allocator stats + allocation-history snapshots for one rank."""

    def __init__(self, rank: int) -> None:
        out = os.environ.get(_ENV)
        self.enabled = bool(out) and torch.cuda.is_available()
        self.rank = int(rank)
        self._checkpoints: list[dict] = []
        self._components: dict[str, float] = {}
        self._forwards_seen = 0
        if not self.enabled:
            return
        self.out_dir = Path(out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
        except Exception as exc:  # snapshot API is best-effort across torch versions
            logger.warning("AR-Diffusion mem-profile: record_memory_history failed (%s)", exc)
        logger.info("AR-Diffusion mem-profile: recording to %s (rank %d)", out, rank)

    def checkpoint(self, name: str) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        self._checkpoints.append(
            {
                "name": name,
                "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "device_used_gib": (total - free) / 2**30,
            }
        )
        self._write()

    def component_breakdown(self, pipeline) -> None:
        """Sum parameter + buffer bytes per top-level pipeline component."""
        if not self.enabled or pipeline is None:
            return
        for name in ("transformer", "text_encoder", "image_encoder", "vae", "action_head"):
            mod = getattr(pipeline, name, None)
            if not isinstance(mod, torch.nn.Module):
                continue
            n = sum(p.numel() * p.element_size() for p in mod.parameters())
            n += sum(b.numel() * b.element_size() for b in mod.buffers())
            self._components[name] = n / 2**30
        self._write()

    def note(self, name: str, gib: float) -> None:
        """Record a known allocation (e.g. the paged KV pool) by size."""
        if not self.enabled:
            return
        self._components[name] = float(gib)
        self._write()

    def dump_snapshot(self, tag: str) -> None:
        if not self.enabled:
            return
        try:
            torch.cuda.memory._dump_snapshot(str(self.out_dir / f"mem_snapshot_{tag}_rank{self.rank}.pickle"))
        except Exception as exc:
            logger.warning("AR-Diffusion mem-profile: dump_snapshot(%s) failed (%s)", tag, exc)

    def on_forward_done(self) -> None:
        """Checkpoint the first few forwards, then dump the run snapshot and stop."""
        if not self.enabled:
            return
        self._forwards_seen += 1
        if self._forwards_seen <= 4:
            self.checkpoint(f"forward{self._forwards_seen}")
        if self._forwards_seen == 4:
            self.dump_snapshot("run")
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass

    def _write(self) -> None:
        payload = {
            "rank": self.rank,
            "device": torch.cuda.current_device(),
            "component_weights_gib": self._components,
            "checkpoints": self._checkpoints,
        }
        (self.out_dir / f"mem_checkpoints_rank{self.rank}.json").write_text(json.dumps(payload, indent=1))
