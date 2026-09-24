# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration for AR-Diffusion engine-level KV cache management."""

from __future__ import annotations

import os
from dataclasses import dataclass

KV_GATHER_ENV = "VLLM_OMNI_AR_DIFFUSION_KV_GATHER"


# The worker's resolved choice (deployment config, else the pipeline's KV spec), set once by the runner.
_contiguous_kv_gather: bool = False


def set_contiguous_kv_gather(enabled: bool) -> None:
    """Record the runner-resolved gather choice for this worker process (see :func:`contiguous_kv_gather_enabled`)."""
    global _contiguous_kv_gather
    _contiguous_kv_gather = bool(enabled)


def contiguous_kv_gather_enabled() -> bool:
    """Whether the contiguous-K/V gather attention path is on.

    ``VLLM_OMNI_AR_DIFFUSION_KV_GATHER=1`` / ``=0`` forces it on or off. Unset, the runner's resolved
    ``ARDiffusionKVConfig.contiguous_kv_gather`` applies: the deployment value, else the pipeline's
    ``ARDiffusionKVCacheSpec`` default (off unless a pipeline opts in).

    The one place the switch is read: the KV manager consults it when it budgets and allocates the history
    staging buffers, and the attention dispatch when it picks the path, so the two cannot disagree. It is the
    only consumer of ``ARDiffusionKVConfig.reuse_history_staging``.
    """
    forced = os.environ.get(KV_GATHER_ENV)
    if forced in ("0", "1"):
        return forced == "1"
    return _contiguous_kv_gather


@dataclass
class ARDiffusionKVConfig:
    """Settings for the AR-Diffusion paged KV cache.

    Disabled by default: when ``enable`` is False the AR-Diffusion engine behaves exactly
    like the base ``DiffusionEngine`` (no pool, no paged KV).
    """

    enable: bool = False
    # Persistent KV tokens materialized per paged cache block.
    chunk_size: int = 0
    # Resident window in chunks. ``None`` means full attention (no eviction).
    window_chunks: int | None = None
    # Protected leading chunks (attention sink); never evicted.
    sink_chunks: int = 0
    # Boundary reset vs. sliding replacement.
    reset_at_boundary: bool = False
    # Fraction of free device memory used to admit additional resident
    # sessions. One session is admitted whenever it fits actual free memory.
    gpu_memory_fraction: float = 0.1
    # When CUDA graph / torch.compile is on (not enforce_eager), pre-capture the
    # DiT graphs for every window-fill shape at load time via a synthetic rollout,
    # so the serving run is fast from the first chunk. No effect when eager.
    warmup_cudagraph: bool = True
    # Keep one contiguous K/V staging buffer per layer and refresh only the tokens that changed.
    # Within one AR block every forward attends the same history and differs only in the current
    # chunk, so re-gathering the whole visible window per forward re-copies bytes that did not move.
    # It costs two max-sequence buffers per layer, per rank. See ARDiffusionPagedForwardContext.history_staging.
    # ``None`` takes the pipeline's ARDiffusionKVCacheSpec default (off unless the pipeline opts in).
    reuse_history_staging: bool | None = None
    # Attend a contiguous gather of the visible K/V instead of FA3's paged path (see contiguous_kv_gather_enabled).
    # ``None`` takes the pipeline's ARDiffusionKVCacheSpec default; the environment variable still overrides.
    contiguous_kv_gather: bool | None = None
    # Also capture the post-window-boundary (reset-cycle) forward during warm-up.
    warmup_capture_reset: bool = False

    @property
    def sliding_window(self) -> int | None:
        """Window size in tokens, or ``None`` for full attention."""
        if self.window_chunks is None:
            return None
        return self.window_chunks * self.chunk_size
