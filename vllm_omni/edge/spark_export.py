# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export a Spark-X2.5 decode step for phone accelerators (QNN / TFLite / ORT).

The graph is one autoregressive step with the KV cache passed in and out as
explicit tensors, which is the shape every mobile NPU runtime wants: static
shapes, no control flow, no in-place cache mutation.

What makes Spark worth exporting this way is its hybrid attention. Three of
every four layers attend to a 512-token sliding window, so their cache is a
fixed 512 entries *no matter how long the conversation is*; only the seven
full-attention layers grow with context. At a 4k context that is 81 MiB of
KV instead of the 235 MiB a uniform-attention model of the same shape would
carry, and the gap widens linearly from there. The two layer types are
therefore exported with separately sized caches, masks and rotary angles --
sliding layers rotate all 256 head dims with theta 1e4, full layers only the
first 64 with theta 5e6.

``python -m vllm_omni.edge.spark_export --help`` exports one layer of each
type, or the whole 28-layer step, to ONNX.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm_omni.edge.qnn_export import sanitize_onnx

__all__ = [
    "SparkStepConfig",
    "SparkDecodeStep",
    "SparkDecodeCache",
    "config_from_spark",
    "example_inputs",
    "input_names",
    "output_names",
    "export_onnx",
    "load_spark_weights",
    "kv_cache_bytes",
]

SLIDING = "sliding_attention"
FULL = "full_attention"


@dataclass
class SparkStepConfig:
    """Shape of one Spark-X2.5 decode step."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    sliding_window: int
    layer_types: tuple[str, ...]
    rope_theta: dict[str, float]
    partial_rotary_factor: dict[str, float]
    rms_norm_eps: float = 1e-6
    include_lm_head: bool = True
    gelu_mode: str = "exact"
    """``exact`` is the erf GELU the model was trained with (and the only one
    its own config accepts). ``tanh`` and ``sigmoid`` are the usual cheaper
    approximations, exposed because the erf form is 11-13% of every layer on
    the Hexagon NPU -- adopt one only if its fidelity is measured."""
    cache_layout: str = "auto"
    """``ring``: the cache is a fixed ring buffer the runtime writes into and
    the graph returns only the new K/V entry. ``roll``: the graph concatenates
    the new entry, re-slices the window and hands the whole window back.

    ``auto`` (the default) picks per layer type, which is what the device
    measurements say to do. Ring takes 12.2% off a sliding layer, because roll
    made it hand a 512-entry window back every token. A full-attention layer
    gains nothing (1.567 vs 1.559 ms: the concat ring saves is spent on its
    extra score matmul) and ring additionally fails QNN's context-binary
    converter at w4a16 there, so full layers keep roll."""
    first_layer: int = 0
    """Index of the first exported layer, so a subset can be profiled on its own."""

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads {self.num_attention_heads} is not a multiple "
                f"of num_key_value_heads {self.num_key_value_heads}"
            )
        bad = set(self.layer_types) - {SLIDING, FULL}
        if bad:
            raise ValueError(f"unknown layer types: {sorted(bad)}")
        if self.cache_layout not in ("auto", "ring", "roll"):
            raise ValueError(f"unknown cache_layout: {self.cache_layout}")
        if self.gelu_mode not in ("exact", "tanh", "sigmoid"):
            raise ValueError(f"unknown gelu_mode: {self.gelu_mode}")

    @property
    def group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)

    def rotary_dim(self, layer_type: str) -> int:
        return int(self.head_dim * self.partial_rotary_factor[layer_type])

    def layout_for(self, layer_type: str) -> str:
        """Effective cache layout for a layer of this type."""
        if self.cache_layout != "auto":
            return self.cache_layout
        return "ring" if layer_type == SLIDING else "roll"

    def cache_len(self, layer_type: str, context: int) -> int:
        """KV entries a layer of this type holds at ``context`` tokens.

        A 512-token window counts the current token, so a ring buffer holds
        511 previous entries and the graph scores those plus the new one.
        The roll layout instead keeps 512 and drops the oldest after
        appending. (transformers' own sliding cache holds 511, same reason.)
        """
        if layer_type != SLIDING:
            return context
        ring = self.layout_for(SLIDING) == "ring"
        return min(context, self.sliding_window - 1 if ring else self.sliding_window)


def _gelu(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "exact":
        return torch.nn.functional.gelu(x)
    if mode == "tanh":
        return torch.nn.functional.gelu(x, approximate="tanh")
    return x * torch.sigmoid(1.702 * x)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Neox-style rotation of the leading ``cos.shape[-1]`` dims; the rest pass through."""
    rot = cos.shape[-1]
    if rot < x.shape[-1]:
        x_rot, x_pass = x[..., :rot], x[..., rot:]
    else:
        x_rot, x_pass = x, None
    half = rot // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    out = x_rot * cos + torch.cat([-x2, x1], dim=-1) * sin
    return out if x_pass is None else torch.cat([out, x_pass], dim=-1)


class _Layer(nn.Module):
    def __init__(self, cfg: SparkStepConfig, layer_type: str):
        super().__init__()
        self.cfg = cfg
        self.layer_type = layer_type
        self.cache_layout = cfg.layout_for(layer_type)
        h, d = cfg.hidden_size, cfg.head_dim
        q_dim = cfg.num_attention_heads * d
        kv_dim = cfg.num_key_value_heads * d
        self.input_layernorm = nn.Parameter(torch.ones(h))
        self.post_attention_layernorm = nn.Parameter(torch.ones(h))
        self.q_k_v_proj = nn.Linear(h, q_dim + 2 * kv_dim, bias=False)
        self.g_proj = nn.Linear(h, cfg.num_attention_heads, bias=False)
        self.out_proj = nn.Linear(q_dim, h, bias=False)
        self.gate_proj = nn.Linear(h, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(h, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, h, bias=False)

    def forward(self, x, cos, sin, k_cache, v_cache, mask):
        cfg = self.cfg
        heads, kv, d, g = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.group_size,
        )
        h = _rms(x, self.input_layernorm, cfg.rms_norm_eps)
        qkv = self.q_k_v_proj(h)
        q = qkv[..., : heads * d].view(1, heads, 1, d)
        k = qkv[..., heads * d : (heads + kv) * d].view(1, kv, 1, d)
        v = qkv[..., (heads + kv) * d :].view(1, kv, 1, d)
        gate = torch.sigmoid(self.g_proj(h)).view(1, heads, 1, 1)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)

        # GQA without materialising expanded K/V: fold the query heads that
        # share a kv head into their own axis.
        scale = 1.0 / math.sqrt(d)
        qg = q.view(1, kv, g, d)

        if self.cache_layout == "ring":
            # The cache is a ring buffer the runtime writes into, so the graph
            # never has to hand a window back: it returns the one new entry and
            # the runtime drops it into the slot for this position, evicting
            # the oldest. That removes both the cache-sized Output (13% of the
            # layer on device) and the window Slice (7.4%).
            #
            # Ring order is arbitrary, which is fine: softmax over keys is
            # order-independent and each cached key already carries its own
            # rotary phase. Only slots not yet written need masking.
            #
            # Attention itself stays exactly as roll computes it -- concat the
            # keys, one fused Softmax. Splitting it into an online softmax over
            # two key sets removes the concat too, and is numerically identical
            # (140.3 dB), but the explicit exp/max/div it needs cannot be
            # quantized: QNN's context-binary converter exits 14 at w4a16,
            # while the fused Softmax form compiles.
            keys = torch.cat([k_cache, k], dim=2)
            vals = torch.cat([v_cache, v], dim=2)
            scores = torch.matmul(qg, keys.transpose(2, 3)) * scale + mask
            attn = torch.matmul(torch.softmax(scores, dim=-1), vals)
            k_out, v_out = k, v
        else:
            # Roll the window: the oldest entry falls off the front.
            keys = torch.cat([k_cache, k], dim=2)
            vals = torch.cat([v_cache, v], dim=2)
            if self.layer_type == SLIDING:
                keys, vals = keys[:, :, 1:], vals[:, :, 1:]
            scores = torch.matmul(qg, keys.transpose(2, 3)) * scale + mask
            attn = torch.matmul(torch.softmax(scores, dim=-1), vals)
            k_out = keys if self.layer_type == SLIDING else k
            v_out = vals if self.layer_type == SLIDING else v

        attn = (attn.view(1, heads, 1, d) * gate).view(1, 1, heads * d)

        x = x + self.out_proj(attn)
        h = _rms(x, self.post_attention_layernorm, cfg.rms_norm_eps)
        x = x + self.down_proj(_gelu(self.gate_proj(h), cfg.gelu_mode) * self.up_proj(h))
        return x, k_out, v_out


class SparkDecodeStep(nn.Module):
    """One decode step.

    Under the default ``ring`` cache layout every layer returns just the new
    K/V entry and the runtime drops it into the ring slot for this position.
    Under ``roll`` a sliding layer hands back its whole recomputed window,
    which costs a fifth of the layer on device.
    """

    def __init__(self, cfg: SparkStepConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(_Layer(cfg, t) for t in cfg.layer_types)
        if cfg.include_lm_head:
            self.norm = nn.Parameter(torch.ones(cfg.hidden_size))
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, x, cos_sw, sin_sw, cos_full, sin_full, mask_sw, mask_full, *caches):
        cfg = self.cfg
        if len(caches) != 2 * cfg.num_layers:
            raise ValueError(
                f"expected {2 * cfg.num_layers} cache tensors, got {len(caches)}"
            )
        new: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            sliding = cfg.layer_types[i] == SLIDING
            cos, sin = (cos_sw, sin_sw) if sliding else (cos_full, sin_full)
            mask = mask_sw if sliding else mask_full
            x, k, v = layer(x, cos, sin, caches[2 * i], caches[2 * i + 1], mask)
            new += [k, v]
        if not cfg.include_lm_head:
            return (x, *new)
        logits = self.lm_head(_rms(x, self.norm, cfg.rms_norm_eps))[:, 0]
        return (logits, *new)


class SparkDecodeCache:
    """CPU reference for the fixed-shape KV contract of ``SparkDecodeStep``.

    Sliding layers keep the preceding ``window - 1`` entries in a ring; full
    layers keep every preceding entry. The graph sees fixed-size buffers and
    masks unused slots. This is a state-contract reference, not a mobile
    device allocator or an Omni request scheduler.
    """

    def __init__(
        self,
        cfg: SparkStepConfig,
        max_context: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if max_context < 1:
            raise ValueError("max_context must be positive")
        if cfg.layout_for(SLIDING) != "ring" or cfg.layout_for(FULL) != "roll":
            raise ValueError("SparkDecodeCache requires ring sliding and roll full layout")
        self.cfg = cfg
        self.max_context = max_context
        self.position = 0
        self._seeded = False
        self._closed = False
        self.buffers: list[torch.Tensor] = []
        for layer_type in cfg.layer_types:
            capacity = cfg.sliding_window - 1 if layer_type == SLIDING else max_context
            if capacity < 1:
                raise ValueError("sliding_window must be at least 2")
            for _ in ("k", "v"):
                self.buffers.append(
                    torch.zeros(
                        1, cfg.num_key_value_heads, capacity, cfg.head_dim,
                        dtype=dtype,
                    )
                )

    def seed(self, layer_caches: list[tuple[torch.Tensor, torch.Tensor]], position: int) -> None:
        """Seed from a verified prefill; full layers must contain its entire past."""
        if self._closed or self._seeded or position < 0 or position > self.max_context:
            raise ValueError("seed requires an empty cache and valid position")
        if len(layer_caches) != self.cfg.num_layers:
            raise ValueError("prefill layer count does not match the export")
        # Reject the whole seed before changing any buffer.
        for i, ((keys, values), layer_type) in enumerate(
            zip(layer_caches, self.cfg.layer_types)
        ):
            capacity = self.buffers[2 * i].shape[2]
            expected = min(position, capacity) if layer_type == SLIDING else position
            for source in (keys, values):
                if source.shape != (1, self.cfg.num_key_value_heads,
                                     expected, self.cfg.head_dim):
                    raise ValueError(f"prefill cache shape mismatch at layer {i}")
        for i, ((keys, values), layer_type) in enumerate(
            zip(layer_caches, self.cfg.layer_types)
        ):
            capacity = self.buffers[2 * i].shape[2]
            expected = min(position, capacity) if layer_type == SLIDING else position
            for source, dest in ((keys, self.buffers[2 * i]),
                                 (values, self.buffers[2 * i + 1])):
                if layer_type == FULL:
                    dest[:, :, :position].copy_(source.to(dest.dtype))
                else:
                    for offset in range(expected):
                        absolute = position - expected + offset
                        dest[:, :, absolute % capacity].copy_(
                            source[:, :, offset].to(dest.dtype)
                        )
        self.position = position
        self._seeded = True

    def step_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Inputs for one token at the absolute ``position`` of this state."""
        if self._closed or not self._seeded:
            raise ValueError("decode state is not active")
        if self.position >= self.max_context:
            raise ValueError("decode would exceed max_context")
        if x.shape != (1, 1, self.cfg.hidden_size):
            raise ValueError("decode embedding shape mismatch")
        if x.dtype != self.buffers[0].dtype:
            raise ValueError("decode embedding dtype mismatch")
        cfg = self.cfg
        cos_sw, sin_sw = _angles(
            self.position, cfg.rotary_dim(SLIDING), cfg.rope_theta[SLIDING],
            x.dtype,
        )
        cos_full, sin_full = _angles(
            self.position, cfg.rotary_dim(FULL), cfg.rope_theta[FULL], x.dtype,
        )
        sw_capacity = cfg.sliding_window - 1
        mask_sw = torch.full(
            (1, 1, 1, sw_capacity + 1), float("-inf"), dtype=x.dtype
        )
        mask_sw[:, :, :, :min(self.position, sw_capacity)] = 0
        mask_sw[:, :, :, -1] = 0
        mask_full = torch.full(
            (1, 1, 1, self.max_context + 1), float("-inf"), dtype=x.dtype
        )
        mask_full[:, :, :, :self.position] = 0
        mask_full[:, :, :, -1] = 0
        return (
            x, cos_sw, sin_sw, cos_full, sin_full, mask_sw, mask_full,
            *self.buffers,
        )

    def commit(self, outputs: tuple[torch.Tensor, ...], *, expected_position: int) -> None:
        """Publish a completed decode step once, after all graph outputs exist."""
        if self._closed or not self._seeded:
            raise ValueError("decode state is not active")
        if expected_position != self.position:
            raise ValueError("stale or repeated decode output")
        if self.position >= self.max_context:
            raise ValueError("decode would exceed max_context")
        if len(outputs) != 1 + 2 * self.cfg.num_layers:
            raise ValueError("decode output count mismatch")
        # Validate every output before writing any K/V slot.
        first_shape = ((1, self.cfg.vocab_size) if self.cfg.include_lm_head
                       else (1, 1, self.cfg.hidden_size))
        if outputs[0].shape != first_shape or not torch.isfinite(outputs[0]).all():
            raise ValueError("decode logits or hidden output is invalid")
        for i in range(self.cfg.num_layers):
            for kv in (0, 1):
                source = outputs[1 + 2 * i + kv]
                dest = self.buffers[2 * i + kv]
                if source.shape != (1, self.cfg.num_key_value_heads, 1,
                                    self.cfg.head_dim):
                    raise ValueError(f"decode output shape mismatch at layer {i}")
                if source.dtype != dest.dtype or source.device != dest.device:
                    raise ValueError(f"decode output placement mismatch at layer {i}")
                if not torch.isfinite(source).all():
                    raise ValueError(f"non-finite decode cache at layer {i}")
        for i, layer_type in enumerate(self.cfg.layer_types):
            for kv in (0, 1):
                source = outputs[1 + 2 * i + kv]
                dest = self.buffers[2 * i + kv]
                slot = self.position % dest.shape[2] if layer_type == SLIDING else self.position
                dest[:, :, slot:slot + 1].copy_(source)
        self.position += 1

    def clear(self) -> None:
        """Retire this reference state and erase its cached tensors."""
        for buffer in self.buffers:
            buffer.zero_()
        self.position = 0
        self._closed = True


def config_from_spark(
    hf_config: dict[str, Any],
    layer_slice: slice | None = None,
    include_lm_head: bool = True,
    cache_layout: str = "auto",
    gelu_mode: str = "exact",
) -> SparkStepConfig:
    """Build a step config from a Spark ``config.json``.

    ``layer_slice`` selects a contiguous subset of layers, which is how a
    single sliding or full layer gets profiled on its own.
    """
    types = list(hf_config["layer_types"])
    first = 0
    if layer_slice is not None:
        first = layer_slice.start or 0
        types = types[layer_slice]
    rope = hf_config["rope_parameters"]
    return SparkStepConfig(
        hidden_size=hf_config["hidden_size"],
        num_attention_heads=hf_config["num_attention_heads"],
        num_key_value_heads=hf_config["num_key_value_heads"],
        head_dim=hf_config["head_dim"],
        intermediate_size=hf_config["intermediate_size"],
        vocab_size=hf_config["vocab_size"],
        sliding_window=hf_config["sliding_window"],
        layer_types=tuple(types),
        rope_theta={k: float(v.get("rope_theta", 10000)) for k, v in rope.items()},
        partial_rotary_factor={
            k: float(v.get("partial_rotary_factor", 1.0)) for k, v in rope.items()
        },
        rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
        include_lm_head=include_lm_head,
        cache_layout=cache_layout,
        gelu_mode=gelu_mode,
        first_layer=first,
    )


def input_names(cfg: SparkStepConfig) -> list[str]:
    return [
        "x",
        "cos_sw",
        "sin_sw",
        "cos_full",
        "sin_full",
        "mask_sw",
        "mask_full",
    ] + [f"{kv}_cache_{i}" for i in range(cfg.num_layers) for kv in ("k", "v")]


def output_names(cfg: SparkStepConfig) -> list[str]:
    head = "logits" if cfg.include_lm_head else "hidden"
    return [head] + [
        f"{kv}_new_{i}" for i in range(cfg.num_layers) for kv in ("k", "v")
    ]


def _angles(position: int, rotary_dim: int, theta: float, dtype: torch.dtype):
    inv = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    ang = position * inv
    cos = torch.cat([ang.cos(), ang.cos()]).view(1, 1, 1, rotary_dim).to(dtype)
    sin = torch.cat([ang.sin(), ang.sin()]).view(1, 1, 1, rotary_dim).to(dtype)
    return cos, sin


def example_inputs(
    cfg: SparkStepConfig,
    context: int,
    position: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """Concrete inputs for tracing at a given context length."""
    pos = context if position is None else position
    sw_len = cfg.cache_len(SLIDING, context)
    full_len = cfg.cache_len(FULL, context)
    cos_sw, sin_sw = _angles(pos, cfg.rotary_dim(SLIDING), cfg.rope_theta[SLIDING], dtype)
    cos_full, sin_full = _angles(pos, cfg.rotary_dim(FULL), cfg.rope_theta[FULL], dtype)
    # Both layouts score the same number of keys: ring scores its cache plus
    # the new token, roll scores the window it just rebuilt (which already
    # contains it). Ring's sliding cache is one shorter, so the widths match.
    mask_sw = torch.zeros(1, 1, 1, sw_len + (1 if cfg.layout_for(SLIDING) == "ring" else 0), dtype=dtype)
    mask_full = torch.zeros(1, 1, 1, full_len + 1, dtype=dtype)
    x = (torch.randn(1, 1, cfg.hidden_size) * 0.1).to(dtype)
    caches: list[torch.Tensor] = []
    for t in cfg.layer_types:
        n = cfg.cache_len(t, context)
        for _ in range(2):
            caches.append(
                torch.randn(1, cfg.num_key_value_heads, n, cfg.head_dim, dtype=dtype)
                * 0.5
            )
    return (x, cos_sw, sin_sw, cos_full, sin_full, mask_sw, mask_full, *caches)


def kv_cache_bytes(cfg: SparkStepConfig, context: int, bytes_per_elem: int = 2) -> dict[str, int]:
    """KV footprint of the hybrid layout versus an all-full-attention model."""
    per_entry = 2 * cfg.num_key_value_heads * cfg.head_dim * bytes_per_elem
    hybrid = sum(cfg.cache_len(t, context) for t in cfg.layer_types) * per_entry
    uniform = cfg.num_layers * context * per_entry
    return {"hybrid": hybrid, "uniform": uniform, "saved": uniform - hybrid}


def load_spark_weights(
    module: SparkDecodeStep, weights_dir: str | Path, dtype: torch.dtype = torch.float32
) -> int:
    """Load real Spark weights into the exported step. Returns tensors copied."""
    from safetensors.torch import load_file

    weights_dir = Path(weights_dir)
    shards = sorted(weights_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {weights_dir}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(shard)))

    cfg = module.cfg
    copied = 0
    with torch.no_grad():
        for i, layer in enumerate(module.layers):
            src = f"model.layers.{cfg.first_layer + i}."
            for dst_name, key in (
                ("input_layernorm", "input_layernorm.weight"),
                ("post_attention_layernorm", "post_attention_layernorm.weight"),
            ):
                getattr(layer, dst_name).copy_(state[src + key].to(dtype))
                copied += 1
            for mod, key in (
                (layer.q_k_v_proj, "self_attn.q_k_v_proj.weight"),
                (layer.g_proj, "self_attn.g_proj.weight"),
                (layer.out_proj, "self_attn.out_proj.weight"),
                (layer.gate_proj, "mlp.gate_proj.weight"),
                (layer.up_proj, "mlp.up_proj.weight"),
                (layer.down_proj, "mlp.down_proj.weight"),
            ):
                mod.weight.copy_(state[src + key].to(dtype))
                copied += 1
        if cfg.include_lm_head:
            module.norm.copy_(state["model.norm.weight"].to(dtype))
            # Spark ties the output projection to the input embedding.
            emb = state.get("lm_head.weight", state["model.embedding.weight"])
            module.lm_head.weight.copy_(emb.to(dtype))
            copied += 2
    return copied


def export_onnx(
    module: SparkDecodeStep, context: int, path: str | Path, opset: int = 18
) -> dict[str, Any]:
    """Trace to ONNX at a fixed context length and sanitize for Qualcomm tools."""
    cfg = module.cfg
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    args = example_inputs(cfg, context)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            args,
            str(path),
            input_names=input_names(cfg),
            output_names=output_names(cfg),
            opset_version=opset,
            dynamo=False,
        )
    stats = sanitize_onnx(path)
    return {
        "path": str(path),
        "context": context,
        "layers": cfg.num_layers,
        "layer_types": list(cfg.layer_types),
        "include_lm_head": cfg.include_lm_head,
        "sanitized": stats,
        "kv_bytes": kv_cache_bytes(cfg, context),
    }


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to Spark config.json")
    ap.add_argument("--weights", default=None, help="directory of safetensors shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument(
        "--layers",
        default="all",
        help="'all', 'sliding', 'full', or START:STOP",
    )
    ap.add_argument("--no-lm-head", action="store_true")
    ap.add_argument("--cache-layout", default="auto", choices=["auto", "ring", "roll"])
    ap.add_argument("--gelu", default="exact", choices=["exact", "tanh", "sigmoid"])
    a = ap.parse_args(argv)

    hf = json.loads(Path(a.config).read_text())
    types = hf["layer_types"]
    if a.layers == "all":
        sl = None
    elif a.layers == "sliding":
        i = types.index(SLIDING)
        sl = slice(i, i + 1)
    elif a.layers == "full":
        i = types.index(FULL)
        sl = slice(i, i + 1)
    else:
        start, stop = (int(v) for v in a.layers.split(":"))
        sl = slice(start, stop)

    cfg = config_from_spark(hf, sl, include_lm_head=not a.no_lm_head,
                            cache_layout=a.cache_layout, gelu_mode=a.gelu)
    step = SparkDecodeStep(cfg)
    if a.weights:
        print(f"loaded {load_spark_weights(step, a.weights)} tensors")
    print(json.dumps(export_onnx(step, a.context, a.out), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
