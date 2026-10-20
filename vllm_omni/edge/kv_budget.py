# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Size the KV cache from the model instead of from a server default.

``VLLM_CPU_KVCACHE_SPACE`` defaults to a server-shaped budget and is parsed
with ``int()``, so it cannot even express an edge-sized cache -- the smallest
non-zero value it accepts is 1 GiB.

**Retraction (2026-09-13).** This module used to claim that right-sizing the
budget took resident memory from 8329 MB to 5254 MB. That number was wrong
twice over: it was read from ``VmRSS``, which counts the mmap'd checkpoint's
pages and moved 1-2 GB between identical runs, and the two arms were not the
same model file. Re-measured on ``RssAnon``, which repeats to 0.2%, the saving
is **188 MB**.

**Re-measured 2026-09-14, and the retraction needs its own retraction.** The
188 MB was itself measured through a path that ignored the knob: this platform
silently overwrote ``kv_cache_memory_bytes`` from ``VLLM_CPU_KVCACHE_SPACE``,
so both arms ran the same cache. With the budget actually honoured, and both
arms filling a 1536-token context, 3 interleaved passes each:

    1 GiB budget    3264 MB resident
    192 MiB budget  2421 MB resident   ->  843 MB

The cache is **fully resident**, not allocated-and-untouched: ``mincore(2)``
reports 1024.1 MB in core of 1023.8 MB allocated, and 191.1 of 190.8 when
sized down. Sizing this correctly is the largest single memory lever measured
on this model -- larger than the weight layout, the repack, or the RoPE table.

A hybrid-attention model does not *need* a full-length cache in every layer.
Spark-X2.5 is 3 sliding layers (window 512) to 1 full layer over 28 layers, so
at a 32 k context its real cache is 3.8x smaller than a flat budget:

    flat      28 x 32768               = 917 504 token-layers
    hybrid    21 x 512 + 7 x 32768     = 240 128 token-layers

But vLLM's CPU path did not take that discount when this was measured: for a
4 GiB budget it reported 71 527 tokens, i.e. 56 KiB/token, which is the flat
figure. That observation now needs re-checking rather than trusting --
``CpuPlatform.support_hybrid_kv_cache()`` returns ``True`` in vLLM 0.28
(``vllm/platforms/cpu.py:545``), so either the model's attention groups are not
being reported as hybrid, or the budget is applied before that is known. Until
someone establishes which, the conservative number is the one to pass. So
``bytes_total`` is the **flat** number -- the one that is safe to pass today --
and the hybrid number is reported alongside it as ``bytes_total_hybrid``, for
a backend that accounts per layer type. A budget that is too small is not a
graceful degradation: vLLM preempts by recompute, which on a single-stream
edge deployment is a stall.

**Linear-attention layers (2026-09-14).** Spark's hybrid is sliding-vs-full,
which is still a KV cache in every layer, only a shorter one. Qwen3.5/3.6/3.8
are a different hybrid: 3 of every 4 layers are *gated DeltaNet*, which keeps a
fixed-size recurrent state and **no KV cache at all**, at any context length.
Charging those layers per token is not conservative, it is wrong -- the cache
they need is zero. Qwen3.8-27B is 48 linear layers to 16 full ones, so a flat
64-layer price over-counts its KV by 4x:

    flat over 64 layers    256 KiB/token
    16 full layers only     64 KiB/token

What the linear layers cost instead is a constant per *sequence*, and it is not
small -- vLLM allocates it from the same pool as the KV cache, so a budget that
omits it is short by that amount before the first token:

    conv state      (kernel-1) x (2 x k_heads x k_dim + v_heads x v_dim)
    temporal state  v_heads x v_dim x k_dim, in ``mamba_ssm_dtype``

At Qwen3.8-27B's geometry that is 60 KiB + 3 MiB per layer per sequence, i.e.
**147 MiB per sequence** across the 48 linear layers, against 64 KiB/token of
KV. The two cross at about 2350 tokens: below that the state costs more than
the cache does. The shapes follow vLLM's own
``MambaStateShapeCalculator.gated_delta_net_state_shape``
(``vllm/model_executor/layers/mamba/mamba_utils.py:262``) rather than a
re-derivation, so they stay right if that changes.

The nesting matters too. A multimodal checkpoint puts all of this under
``text_config`` -- ``Qwen3_5Config`` has no ``num_hidden_layers`` of its own --
so this module resolves the text config first. Passed the outer config it used
to raise "cannot read num_hidden_layers", which reads as "this model is
unsupported" when the real answer was one attribute deeper.
"""

from dataclasses import dataclass
from typing import Any

_DTYPE_BYTES = {
    "float32": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "half": 2,
    "float8_e4m3fn": 1, "fp8": 1, "int8": 1,
}

# Layer-type spellings that mean "recurrent state, no KV cache". vLLM's own
# configs use "linear_attention" (Qwen3.5/3.6/3.8, Qwen3-Next) and "mamba"
# (Bamba, Zamba); the substring test catches "mamba2" and friends.
_LINEAR_LAYER_MARKERS = ("linear_attention", "mamba", "recurrent", "gated_delta")


def dtype_bytes(dtype: Any) -> int:
    name = getattr(dtype, "name", None) or str(dtype).replace("torch.", "")
    return _DTYPE_BYTES.get(name, 2)


def text_config(hf_config: Any) -> Any:
    """The sub-config that carries the decoder geometry.

    Transformers puts a multimodal model's decoder under ``text_config`` and
    exposes ``get_text_config()`` to reach it. Both are tried, and a config
    that is already the text config is returned unchanged, so callers can pass
    whichever one they have.
    """
    getter = getattr(hf_config, "get_text_config", None)
    if callable(getter):
        try:
            resolved = getter()
        except Exception:
            resolved = None
        if resolved is not None and getattr(resolved, "num_hidden_layers", None):
            return resolved
    nested = getattr(hf_config, "text_config", None)
    if nested is not None and getattr(nested, "num_hidden_layers", None):
        return nested
    return hf_config


def _is_linear(layer_type: Any) -> bool:
    name = str(layer_type).lower()
    return any(marker in name for marker in _LINEAR_LAYER_MARKERS)


@dataclass(frozen=True)
class LinearState:
    """Per-sequence recurrent state of the gated-DeltaNet layers.

    Zero on a model that has none, so callers never have to branch.
    """

    layers: int
    conv_bytes_per_layer: int
    temporal_bytes_per_layer: int

    @property
    def bytes_per_seq(self) -> int:
        return self.layers * (self.conv_bytes_per_layer + self.temporal_bytes_per_layer)


@dataclass(frozen=True)
class KVBudget:
    bytes_total: int
    """Flat budget: safe to pass to vLLM today. Includes the recurrent state."""
    bytes_total_hybrid: int
    """What the model's own cache needs, if the backend accounts per layer type."""
    bytes_per_token_flat: int
    token_layers_hybrid: int
    token_layers_flat: int
    sliding_layers: int
    full_layers: int
    window: int | None
    linear_layers: int = 0
    """Layers with a recurrent state and no KV cache. They are excluded from
    both token-layer counts, because their cache is zero at every context."""
    state_bytes_per_seq: int = 0
    state_bytes_total: int = 0
    """``state_bytes_per_seq * max_num_seqs``, already inside both totals."""

    @property
    def saving_vs_flat(self) -> float:
        """What a layer-type-aware backend would save; 1.0 when there is none."""
        return (
            self.token_layers_flat / self.token_layers_hybrid
            if self.token_layers_hybrid
            else 1.0
        )

    @property
    def state_crossover_tokens(self) -> float:
        """Context at which the KV cache first costs more than the fixed state.

        ``inf`` when there is no state, 0 when there are no attention layers.
        """
        if not self.state_bytes_per_seq:
            return float("inf")
        if not self.bytes_per_token_flat:
            return 0.0
        return self.state_bytes_per_seq / self.bytes_per_token_flat

    def summary(self) -> str:
        extra = ""
        if self.sliding_layers:
            extra = (
                f"; {self.bytes_total_hybrid / 2**20:.0f} MiB if the backend "
                f"accounts {self.sliding_layers} sliding@{self.window} layers "
                f"separately ({self.saving_vs_flat:.2f}x)"
            )
        state = ""
        if self.linear_layers:
            state = (
                f"; includes {self.state_bytes_total / 2**20:.0f} MiB of "
                f"recurrent state for {self.linear_layers} linear layers, "
                f"which hold no KV at any context"
            )
        return f"{self.bytes_total / 2**20:.0f} MiB flat{extra}{state}"


def linear_state_for(
    hf_config: Any,
    *,
    model_dtype: Any = "bfloat16",
    n_linear_layers: int,
) -> LinearState:
    """Per-sequence gated-DeltaNet state, in vLLM's own shapes.

    Mirrors ``MambaStateShapeCalculator.gated_delta_net_state_shape`` at
    ``tp_world_size=1``: a conv state of ``(kernel - 1, conv_dim)`` and a
    temporal state of ``(v_heads, v_dim, k_dim)``. The temporal state is the
    one that matters -- it is quadratic in head dim and, by default, fp32.
    """
    cfg = text_config(hf_config)
    if n_linear_layers <= 0:
        return LinearState(0, 0, 0)
    k_heads = int(getattr(cfg, "linear_num_key_heads", 0) or 0)
    v_heads = int(getattr(cfg, "linear_num_value_heads", 0) or 0)
    k_dim = int(getattr(cfg, "linear_key_head_dim", 0) or 0)
    v_dim = int(getattr(cfg, "linear_value_head_dim", 0) or 0)
    kernel = int(getattr(cfg, "linear_conv_kernel_dim", 0) or 0)
    if not all((k_heads, v_heads, k_dim, v_dim, kernel)):
        raise ValueError(
            f"{n_linear_layers} linear-attention layers, but the config does "
            "not carry the linear_* geometry needed to size their state "
            "(linear_num_key_heads, linear_num_value_heads, "
            "linear_key_head_dim, linear_value_head_dim, "
            "linear_conv_kernel_dim)"
        )
    conv_dim = k_dim * k_heads * 2 + v_dim * v_heads
    conv_elems = (kernel - 1) * conv_dim
    temporal_elems = v_heads * v_dim * k_dim
    ssm_dtype = getattr(cfg, "mamba_ssm_dtype", None) or model_dtype
    return LinearState(
        layers=n_linear_layers,
        conv_bytes_per_layer=conv_elems * dtype_bytes(model_dtype),
        temporal_bytes_per_layer=temporal_elems * dtype_bytes(ssm_dtype),
    )


def kv_budget_for(
    hf_config: Any,
    *,
    max_model_len: int,
    max_num_seqs: int = 1,
    kv_dtype: Any = "bfloat16",
    block_size: int = 16,
    headroom: float = 1.25,
) -> KVBudget:
    """Bytes of KV cache a deployment of this model actually needs.

    Reads ``layer_types``/``sliding_window`` when present, so hybrid-attention
    models are charged for their real cache and not for the longest layer, and
    linear-attention layers are charged a per-sequence state instead of a
    per-token cache. Accepts a multimodal config and resolves its text config.
    """
    cfg = text_config(hf_config)
    n_layers = int(
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", 0)
    )
    if n_layers <= 0:
        raise ValueError("cannot read num_hidden_layers from the config")
    n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or n_heads or 1)
    head_dim = int(
        getattr(cfg, "head_dim", 0)
        or (getattr(cfg, "hidden_size", 0) // max(n_heads, 1))
    )
    if head_dim <= 0:
        raise ValueError("cannot read head_dim from the config")

    layer_types = list(getattr(cfg, "layer_types", None) or [])
    window = getattr(cfg, "sliding_window", None)
    linear = sum(1 for t in layer_types if _is_linear(t))
    sliding = sum(1 for t in layer_types if "sliding" in str(t)) if window else 0
    full = n_layers - sliding - linear

    state = linear_state_for(
        hf_config, model_dtype=kv_dtype, n_linear_layers=linear
    )
    state_total = state.bytes_per_seq * max_num_seqs

    def rounded(tokens: int) -> int:
        return ((tokens + block_size - 1) // block_size) * block_size

    per_seq_full = rounded(max_model_len)
    # A sliding layer still has to hold the window, and never more than the
    # context: a 512-window layer at a 128-token context caches 128.
    per_seq_sliding = rounded(min(int(window), max_model_len)) if sliding else 0

    token_layers_hybrid = max_num_seqs * (
        full * per_seq_full + sliding * per_seq_sliding
    )
    # "Flat" means every *attention* layer at full length. A linear layer is
    # not a shorter cache, it is no cache, so it is out of both counts rather
    # than conservatively included.
    token_layers_flat = max_num_seqs * (n_layers - linear) * per_seq_full
    bytes_per_token_layer = kv_heads * head_dim * 2 * dtype_bytes(kv_dtype)

    return KVBudget(
        bytes_total=int(token_layers_flat * bytes_per_token_layer * headroom)
        + state_total,
        bytes_total_hybrid=int(token_layers_hybrid * bytes_per_token_layer * headroom)
        + state_total,
        bytes_per_token_flat=(n_layers - linear) * bytes_per_token_layer,
        token_layers_hybrid=token_layers_hybrid,
        token_layers_flat=token_layers_flat,
        sliding_layers=sliding,
        full_layers=full,
        window=int(window) if sliding else None,
        linear_layers=linear,
        state_bytes_per_seq=state.bytes_per_seq,
        state_bytes_total=state_total,
    )
