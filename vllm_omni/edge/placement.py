# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Where each parameter lives when the checkpoint is bigger than the VRAM.

:mod:`vllm_omni.edge.weight_path` picks a *format* from the device. This picks
a *placement*, which is a different question and only arises when the two
numbers do not fit: Qwen3.8-27B-FP8 is 30.89 GB of weights against a 24 GiB
card. vLLM will do this for you -- ``cpu_offload_gb`` maps host memory into the
GPU's address space and the kernels read it over PCIe -- but it offloads
"non-selectively until the memory limit is reached", and that is the expensive
way to do it.

**The cost of getting it wrong, measured.** On this laptop (RTX 5090 Laptop
24 GiB, PCIe 4.0 x8, 13.9 GB/s pinned H2D, 472 GB/s VRAM) Qwen3.8-27B-FP8 runs
at **1.05 tok/s** with the smallest bulk offload that fits (10 GB; 9 GB fails
to leave room for the cache). Decode reads every weight once per token, so a
byte on the host side of a 13.9 GB/s link costs 34x what the same byte costs in
472 GB/s VRAM. At 10 GB offloaded that is 719 ms of a ~950 ms step: **the
offload is 75% of the decode time and the other 18.8 GB of weights are 4%.**

**Which bytes those are is a choice.** Not every parameter is read once per
token, and the ones that are not are free to evict. Qwen3.8-27B-FP8's
``outside.safetensors`` and ``mtp.safetensors`` hold 6.5 GB, of which:

    lm_head             2.543 GB   HOT  -- one 248320 x 5120 GEMM per token
    embed_tokens        2.543 GB   COLD -- a gather; one row, ~10 KB per token
    visual.*            0.922 GB   COLD -- unless the prompt has an image
    mtp.*               0.478 GB   COLD -- unless speculative decoding is on

``embed_tokens`` and ``lm_head`` are the same shape and the same dtype, and one
of them is the single best parameter in the model to offload while the other is
among the worst. A size-ordered, name-blind policy cannot tell them apart;
``cpu_offload_params`` can, because it matches name segments.

So the plan here is: close the deficit out of the cold set first, and only
spend the hot set when the cold set runs out. On this checkpoint the cold set
is **3.94 GB**, which is most of the 8.4 GB deficit, and the arithmetic says
the remainder costs ~324 ms/token instead of 719.

**Two limits, both found the hard way (2026-09-15).**

*The cold set is not actually offloadable in vLLM today.* The plan below sorts
parameters into hot and cold and emits ``cpu_offload_params`` for the cold ones.
That flag exists and matches name segments as documented -- but the offloader is
only ever **offered decoder layers**: ``wrap_modules`` is called from inside
``make_layers`` (``vllm/model_executor/models/utils.py:812``) and nowhere else.
``embed_tokens``, ``lm_head``, the vision tower and the MTP head are built
outside it and can never be reached, whatever the flag says. Measured: asking
for ``{embed_tokens, visual, mtp}`` on a 21.92 GB checkpoint with a 4.31 GB cold
set moved **~0.5 GiB**, and the run failed identically to one with no offload at
all. The classification here is still right about which bytes are cheap to move;
the *mechanism* is missing, and closing it means an upstream change. Until then
treat ``cold_offload`` plans as advisory.

*Fewer bytes is not always faster.* This module prices weights by traffic, which
assumes decode is bandwidth-bound. That held for Qwen3.8-27B-FP8 to within 4.3%
(see the test). It does **not** hold for the NVFP4 build of the same model:
21.95 GB of NVFP4 measured **0.67 tok/s** against 30.87 GB of FP8 at
**1.05 tok/s** -- slower while 30% smaller, and 4.7x off what the traffic model
predicts. Enabling torch.compile and CUDA graphs moved it to 0.72, so it is not
an eager-mode artifact: that path is compute-bound, not bandwidth-bound, and a
model of bytes cannot see it. Never recommend a format change on this module's
estimate alone -- measure it.

Estimates here are arithmetic, not measurements: bytes over a bandwidth. They
are labelled ``est_`` and the measured points they were checked against are in
the table above.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GiB = 2**30

# Name segments that are read once per decode step, and segments that are not.
# Matching is on dot-separated segments, the same rule vLLM's
# ``cpu_offload_params`` uses, so what is planned here is what it will do.
COLD_SEGMENTS: tuple[str, ...] = (
    "embed_tokens",   # a gather: one row per token, not the whole table
    "visual",         # vision tower: only on a prompt that carries an image
    "vision_tower",
    "mtp",            # multi-token-prediction head: only under speculation
    "audio_tower",
)
"""Parameters a text decode step does not read. Offloading these is close to
free -- ``embed_tokens`` still costs one row's worth of PCIe per token, which
at 5120 x 2 bytes is 10 KiB against a 13.9 GB/s link, i.e. under a microsecond."""

HOT_ALWAYS: tuple[str, ...] = ("lm_head",)
"""Named so the planner can never classify them cold by accident: ``lm_head``
is the same shape and dtype as ``embed_tokens`` and the opposite decision."""


@dataclass(frozen=True)
class ParamGroup:
    name: str
    bytes: int
    cold: bool


@dataclass(frozen=True)
class Placement:
    strategy: str
    """``resident`` | ``cold_offload`` | ``hot_offload`` | ``infeasible``."""
    device_bytes: int
    host_bytes: int
    host_read_per_token: int
    """Bytes crossing the link every decode step. This, not ``host_bytes``, is
    what costs time -- offloading a parameter nothing reads costs nothing."""
    deficit_bytes: int
    """How far the checkpoint overran the device budget before any offload."""
    est_link_s_per_token: float
    est_vram_s_per_token: float
    engine_kwargs: dict[str, Any]
    rationale: str
    notes: list[str] = field(default_factory=list)

    @property
    def est_decode_tok_s(self) -> float:
        """Weight-traffic bound only: no attention, no kernel launch, no
        sampling. An upper bound, and on the measured point it was 1.24x the
        observed rate."""
        total = self.est_link_s_per_token + self.est_vram_s_per_token
        return 1.0 / total if total > 0 else float("inf")

    def summary(self) -> str:
        return (
            f"{self.strategy}: {self.device_bytes / 1e9:.1f} GB on device, "
            f"{self.host_bytes / 1e9:.1f} GB on host of which "
            f"{self.host_read_per_token / 1e9:.1f} GB is read per token; "
            f"est {self.est_decode_tok_s:.1f} tok/s on weight traffic alone"
        )


def is_cold(name: str) -> bool:
    """Segment match, so ``model.visual.blocks.0`` is cold and ``visualise``
    is not. ``lm_head`` is never cold whatever else it matches."""
    segments = set(name.split("."))
    if segments & set(HOT_ALWAYS):
        return False
    return bool(segments & set(COLD_SEGMENTS))


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def group_key(name: str, top_segments: int = 3) -> str:
    """The unit the planner is allowed to move: a prefix, but never coarser
    than one decoder layer.

    A plain 3-segment prefix puts all 64 of Qwen3.8's decoder layers in one
    24.4 GB group, and a planner that can only take groups whole then has a
    choice between offloading no layers and offloading every layer. Keeping the
    layer index makes each layer its own 0.38 GB group, which is also the
    granularity vLLM's prefetch offloader works at.
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part.isdigit() and i > 0 and parts[i - 1].endswith("layers"):
            return ".".join(parts[: i + 1])
    return ".".join(parts[:top_segments])


def checkpoint_param_groups(model_dir: str | Path, top_segments: int = 3) -> list[ParamGroup]:
    """Bytes per parameter group, read from the safetensors headers.

    Reads headers only -- a few kilobytes per shard -- so this is cheap enough
    to run at deploy-planning time on a 31 GB checkpoint. Walks every
    ``*.safetensors``, which is how Qwen3.8's per-layer shards are laid out.
    """
    root = Path(model_dir)
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under {root}")

    totals: dict[str, int] = {}
    cold: dict[str, bool] = {}
    for shard in shards:
        for key, meta in _safetensors_header(shard).items():
            if key == "__metadata__":
                continue
            offs = meta.get("data_offsets")
            if not offs:
                continue
            prefix = group_key(key, top_segments)
            totals[prefix] = totals.get(prefix, 0) + (offs[1] - offs[0])
            # A group is cold only if every tensor in it is.
            cold[prefix] = cold.get(prefix, True) and is_cold(key)
    return sorted(
        (ParamGroup(k, v, cold[k]) for k, v in totals.items()),
        key=lambda g: -g.bytes,
    )


def _engine_kwargs(
    offloaded: list[ParamGroup], hot: list[ParamGroup], freed: int
) -> dict[str, Any]:
    """Translate a plan into flags vLLM actually has.

    The cold set is expressible exactly: ``cpu_offload_params`` matches name
    segments, and every cold group is named by one. The hot set is not -- it is
    "these 12 specific layers", and the only knob for that is the prefetch
    offloader's ``offload_group_size`` / ``offload_num_in_group``, which takes
    the last N of every group of M. So the hot half is emitted as the closest
    (M, N) whose count matches, and the caller is told it is an approximation.
    """
    segments = sorted(
        {
            seg
            for g in offloaded
            if g.cold
            for seg in g.name.split(".")
            if seg in COLD_SEGMENTS
        }
    )
    kwargs: dict[str, Any] = {
        "cpu_offload_gb": round(freed / GiB + 0.5, 1),
        "cpu_offload_params": segments,
    }
    if hot:
        n_layers = sum(1 for g in hot if group_key(g.name) == g.name and
                       any(p.isdigit() for p in g.name.split(".")))
        n_layers = n_layers or len(hot)
        # Offload 1 layer in every ceil(total/n) -- the coarsest pattern that
        # reaches the count. The caller re-plans if the fit is not exact.
        kwargs["offload_group_size"] = max(2, round(64 / max(n_layers, 1)))
        kwargs["offload_num_in_group"] = 1
    return kwargs


def plan_placement(
    groups: list[ParamGroup],
    *,
    device_bytes_available: int,
    link_bytes_per_s: float,
    vram_bytes_per_s: float,
    embedding_row_bytes: int = 0,
) -> Placement:
    """Decide what goes to the host, cheapest bytes first.

    ``device_bytes_available`` is what is left for *weights* after the KV
    cache, the recurrent state, activations and the driver's own reservation --
    :func:`vllm_omni.edge.kv_budget.kv_budget_for` sizes the first two.
    """
    weights = sum(g.bytes for g in groups)
    deficit = weights - device_bytes_available

    if deficit <= 0:
        return Placement(
            strategy="resident",
            device_bytes=weights, host_bytes=0, host_read_per_token=0,
            deficit_bytes=0,
            est_link_s_per_token=0.0,
            est_vram_s_per_token=weights / vram_bytes_per_s,
            engine_kwargs={},
            rationale=(
                "The checkpoint fits. Every byte is read at VRAM speed and "
                "nothing crosses the link, which is the only placement that "
                "does not pay 34x for some of its weights."
            ),
        )

    # Cheapest first: parameters a decode step never reads.
    offloaded: list[ParamGroup] = []
    freed = 0
    for g in sorted((g for g in groups if g.cold), key=lambda g: -g.bytes):
        if freed >= deficit:
            break
        offloaded.append(g)
        freed += g.bytes

    cold_only = freed >= deficit
    hot_offloaded: list[ParamGroup] = []
    if not cold_only:
        # Still short. Spend hot parameters, largest first -- they all cost the
        # same per byte, so the only thing that matters is hitting the target
        # in as few groups as possible.
        for g in sorted((g for g in groups if not g.cold), key=lambda g: -g.bytes):
            if freed >= deficit:
                break
            hot_offloaded.append(g)
            offloaded.append(g)
            freed += g.bytes

    if freed < deficit:
        return Placement(
            strategy="infeasible",
            device_bytes=device_bytes_available, host_bytes=freed,
            host_read_per_token=sum(g.bytes for g in hot_offloaded),
            deficit_bytes=deficit - freed,
            est_link_s_per_token=float("inf"), est_vram_s_per_token=0.0,
            engine_kwargs={},
            rationale=(
                f"{(deficit - freed) / 1e9:.1f} GB short even with every "
                "parameter on the host. A smaller checkpoint is the only "
                "remaining lever."
            ),
        )

    hot_bytes = sum(g.bytes for g in hot_offloaded)
    # A cold embedding still pays one row per token; everything else cold is 0.
    read_per_token = hot_bytes + (
        embedding_row_bytes
        if any("embed_tokens" in g.name for g in offloaded)
        else 0
    )
    device_bytes = weights - freed
    notes = [
        f"cold offload: {', '.join(g.name for g in offloaded if g.cold)}"
        or "cold offload: none",
    ]
    if hot_offloaded:
        notes.append(
            "hot offload (read every token): "
            + ", ".join(g.name for g in hot_offloaded)
        )
    return Placement(
        strategy="cold_offload" if cold_only else "hot_offload",
        device_bytes=device_bytes,
        host_bytes=freed,
        host_read_per_token=read_per_token,
        deficit_bytes=deficit,
        est_link_s_per_token=read_per_token / link_bytes_per_s,
        est_vram_s_per_token=device_bytes / vram_bytes_per_s,
        engine_kwargs=_engine_kwargs(offloaded, hot_offloaded, freed),
        rationale=(
            f"{deficit / 1e9:.1f} GB over budget. "
            + (
                "Closed entirely out of parameters a decode step does not "
                "read, so the link is idle between steps."
                if cold_only
                else f"The cold set gave {(freed - hot_bytes) / 1e9:.1f} GB; "
                f"the remaining {hot_bytes / 1e9:.1f} GB is read every token "
                f"and is what the decode rate is bound by."
            )
        ),
        notes=notes,
    )
