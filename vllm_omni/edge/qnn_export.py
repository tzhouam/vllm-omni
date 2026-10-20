# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Export the Qwen3-TTS talker / code-predictor **decode step** for phone accelerators (QNN, TFLite, ONNX Runtime).

Companion to :mod:`vllm_omni.edge.decoder_export`, which exports the Code2Wav vocoder. Together the two cover the
whole on-device path served by the codec-token stream (WP5): the talker and its code predictor produce codec tokens,
the vocoder turns them into audio.

Layout
------
One token in, one token out, with an explicit static KV cache::

    inputs   x [1, 1, H], cos [1, 1, 1, D], sin [1, 1, 1, D], mask [1, 1, 1, L+1],
             k_cache_i / v_cache_i [1, KV, L, D] for each layer i
    outputs  logits [1, vocab], k_new_i / v_new_i [1, KV, 1, D] for each layer i

Everything that is cheap on a CPU and awkward on an accelerator stays on the host: embedding lookups, the rotary
angles (passed in as ``cos``/``sin``, which also keeps M-RoPE's interleaved sections out of the graph), sampling, and
appending ``k_new``/``v_new`` into the caches.

Two details are what make this fast, and both were measured on real devices (Snapdragon 8 Gen 3 / 8 Elite via
Qualcomm AI Hub, 2026-09-10; see ``analysis/mobile_htp_simulation.md`` §5):

* **per-layer cache tensors**, not one stacked ``[num_layers, ...]`` tensor that each layer slices. The stacked form
  spent ~70 % of its NPU cycles in cache ``select``/``Reshape`` and fp32→fp16 conversion of the whole cache;
* **grouped-query attention by reshaping the query** to ``[1, KV, G, D]`` instead of expanding keys and values to
  ``[1, H, L+1, D]``. Expanding multiplies the attention tensors by ``G`` before the matmul.

On a Snapdragon 8 Elite this took the 0.6B talker step from 36.9 ms to 15.8 ms (fp16, 256-token cache); the code
predictor step is 2.25 ms, so a full 80 ms frame (talker + 15 predictor sub-steps) costs ≈50 ms on the NPU.

Placement
---------
Measured, same phone, same graphs (per 80 ms frame, fp16):

===================  ===========  ==========  =========
component            Hexagon NPU  Adreno GPU  phone CPU
===================  ===========  ==========  =========
talker step            15.8 ms      29.9 ms    29.8 ms
predictor step x15     33.9 ms      65.0 ms    72.3 ms
Code2Wav vocoder      182 ms        21.7 ms   109 ms
===================  ===========  ==========  =========

So run **this** graph on the NPU and the **vocoder on the GPU**: 8.4x for the vocoder, ~2x for the transformer steps,
and the two stages pipeline the way they already do in the server deployment. Sending everything to the NPU is 3.3x
slower than the split. Integer quantization is *not* the lever it appears to be: every automatic int8 build measured
was numerically broken (predictor 3.1 dB, vocoder -4.3 dB vs the fp32 reference), while fp16 is faithful
(39-48 dB on the NPU).

ONNX for the Qualcomm toolchains
--------------------------------
:func:`sanitize_onnx` fixes the three things the dynamo exporter emits that ``qairt-converter`` and AI Hub reject:
``IsNaN``/``Where`` guards, ``Reshape`` with ``allowzero=1``, and rank-0 ``axes`` initializers on
``Unsqueeze``/``Squeeze`` (which crash AI Hub's shape inference).

CLI::

    python -m vllm_omni.edge.qnn_export --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \\
        --component talker --cache-len 256 --out-dir ./qnn_export
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

__all__ = [
    "DecodeStepConfig",
    "DecodeStep",
    "config_from_qwen3_tts",
    "config_from_minicpmo",
    "load_decoder_weights",
    "load_qwen3_tts_weights",
    "load_minicpmo_tts_weights",
    "example_inputs",
    "input_names",
    "output_names",
    "export_onnx",
    "sanitize_onnx",
    "PLACEMENT_MS_PER_FRAME",
]

#: Measured on Samsung Galaxy S25 (Snapdragon 8 Elite), fp16, per 80 ms audio frame. See the module docstring.
PLACEMENT_MS_PER_FRAME = {
    "talker_step": {"npu": 15.8, "gpu": 29.9, "cpu": 29.8},
    "predictor_step_x15": {"npu": 33.9, "gpu": 65.0, "cpu": 72.3},
    "code2wav_chunk": {"npu": 182.0, "gpu": 21.7, "cpu": 109.0},
}


@dataclass(frozen=True)
class DecodeStepConfig:
    """Shape of one autoregressive step. Mirrors the fields of a Qwen3 decoder config that the graph needs."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    qk_norm: bool = True
    """Qwen3-style per-head RMSNorm on q and k. Llama-style decoders (MiniCPM-o's TTS head) do not have it."""

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(f"num_attention_heads {self.num_attention_heads} is not a multiple of num_key_value_heads {self.num_key_value_heads}")

    @property
    def group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class _Layer(nn.Module):
    def __init__(self, cfg: DecodeStepConfig):
        super().__init__()
        self.cfg = cfg
        h, d = cfg.hidden_size, cfg.head_dim
        self.input_layernorm = nn.Parameter(torch.ones(h))
        self.post_attention_layernorm = nn.Parameter(torch.ones(h))
        self.q_norm = nn.Parameter(torch.ones(d))
        self.k_norm = nn.Parameter(torch.ones(d))
        self.q_proj = nn.Linear(h, cfg.num_attention_heads * d, bias=False)
        self.k_proj = nn.Linear(h, cfg.num_key_value_heads * d, bias=False)
        self.v_proj = nn.Linear(h, cfg.num_key_value_heads * d, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * d, h, bias=False)
        self.gate_proj = nn.Linear(h, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(h, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, h, bias=False)

    def forward(self, x, cos, sin, k_cache, v_cache, mask):
        cfg = self.cfg
        kv, d, g = cfg.num_key_value_heads, cfg.head_dim, cfg.group_size
        h = _rms(x, self.input_layernorm, cfg.rms_norm_eps)
        q = self.q_proj(h).view(1, cfg.num_attention_heads, 1, d)
        k = self.k_proj(h).view(1, kv, 1, d)
        if cfg.qk_norm:
            q = _rms(q, self.q_norm, cfg.rms_norm_eps)
            k = _rms(k, self.k_norm, cfg.rms_norm_eps)
        v = self.v_proj(h).view(1, kv, 1, d)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        keys = torch.cat([k_cache, k], dim=2)
        vals = torch.cat([v_cache, v], dim=2)
        # GQA without expanding keys/values: group the query heads that share a kv head.
        scores = torch.matmul(q.view(1, kv, g, d), keys.transpose(2, 3)) * (1.0 / math.sqrt(d)) + mask
        attn = torch.matmul(torch.softmax(scores, dim=-1), vals).view(1, 1, cfg.num_attention_heads * d)
        x = x + self.o_proj(attn)
        h = _rms(x, self.post_attention_layernorm, cfg.rms_norm_eps)
        x = x + self.down_proj(torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))
        return x, k, v


class DecodeStep(nn.Module):
    """One decode step with an explicit per-layer KV cache; see the module docstring for the tensor contract."""

    def __init__(self, cfg: DecodeStepConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(_Layer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = nn.Parameter(torch.ones(cfg.hidden_size))
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, x, cos, sin, mask, *caches):
        if len(caches) != 2 * self.cfg.num_hidden_layers:
            raise ValueError(f"expected {2 * self.cfg.num_hidden_layers} cache tensors, got {len(caches)}")
        new: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, cos, sin, caches[2 * i], caches[2 * i + 1], mask)
            new += [k, v]
        logits = self.lm_head(_rms(x, self.norm, self.cfg.rms_norm_eps))[:, 0]
        return (logits, *new)


def input_names(cfg: DecodeStepConfig) -> list[str]:
    return ["x", "cos", "sin", "mask"] + [f"{kv}_cache_{i}" for i in range(cfg.num_hidden_layers) for kv in ("k", "v")]


def output_names(cfg: DecodeStepConfig) -> list[str]:
    return ["logits"] + [f"{kv}_new_{i}" for i in range(cfg.num_hidden_layers) for kv in ("k", "v")]


def example_inputs(cfg: DecodeStepConfig, cache_len: int, position: int | None = None, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, ...]:
    """Concrete inputs for tracing: a random hidden state, the rotary angles for ``position`` and a causal mask."""
    pos = cache_len // 2 if position is None else position
    d = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    ang = pos * inv
    cos = torch.cat([ang.cos(), ang.cos()]).view(1, 1, 1, d).to(dtype)
    sin = torch.cat([ang.sin(), ang.sin()]).view(1, 1, 1, d).to(dtype)
    mask = torch.zeros(1, 1, 1, cache_len + 1, dtype=dtype)
    mask[..., pos:cache_len] = torch.finfo(torch.float16).min / 4  # positions not yet written
    x = (torch.randn(1, 1, cfg.hidden_size) * 0.1).to(dtype)
    caches = [torch.randn(1, cfg.num_key_value_heads, cache_len, d, dtype=dtype) * 0.5 for _ in range(2 * cfg.num_hidden_layers)]
    return (x, cos, sin, mask, *caches)


def config_from_minicpmo(hf_config: dict[str, Any]) -> DecodeStepConfig:
    """Build a :class:`DecodeStepConfig` for MiniCPM-o 4.5's TTS head from its ``config.json`` dict.

    A Llama-style decoder (no q/k norm), single codebook (``num_vq == 1``), speech tokens at 25 Hz.
    """
    cfg = hf_config["tts_config"] if "tts_config" in hf_config else hf_config
    heads = cfg["num_attention_heads"]
    return DecodeStepConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=heads,
        num_key_value_heads=cfg.get("num_key_value_heads", heads),
        head_dim=cfg.get("head_dim", cfg["hidden_size"] // heads),
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["num_audio_tokens"],
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
        rope_theta=cfg.get("rope_theta", 10000.0),
        qk_norm=False,
    )


def load_decoder_weights(
    module: DecodeStep,
    safetensors_path: str | Path,
    prefix: str,
    head_key: str,
    *,
    layer_prefix: str = "model.layers.",
    norm_key: str = "model.norm.weight",
    head_is_weight_normed: bool = False,
) -> int:
    """Copy decoder weights into ``module`` from one safetensors file.

    ``prefix`` is prepended to every key (``"talker."`` for Qwen3-TTS, ``"tts."`` for MiniCPM-o). ``head_key`` names
    the output projection. ``head_is_weight_normed`` reconstructs a ``torch.nn.utils.parametrizations.weight_norm``
    head from its stored ``original0`` (magnitude) and ``original1`` (direction) tensors.
    """
    from safetensors import safe_open

    loaded = 0
    with safe_open(str(safetensors_path), "pt") as st:
        get = lambda key: st.get_tensor(prefix + key).float()  # noqa: E731
        for i, layer in enumerate(module.layers):
            p = f"{layer_prefix}{i}."
            for name, param in (("input_layernorm", layer.input_layernorm), ("post_attention_layernorm", layer.post_attention_layernorm)):
                param.data.copy_(get(p + name + ".weight"))
                loaded += 1
            if module.cfg.qk_norm:
                for name, param in (("q_norm", layer.q_norm), ("k_norm", layer.k_norm)):
                    param.data.copy_(get(p + "self_attn." + name + ".weight"))
                    loaded += 1
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(layer, name).weight.data.copy_(get(p + "self_attn." + name + ".weight"))
                loaded += 1
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(layer, name).weight.data.copy_(get(p + "mlp." + name + ".weight"))
                loaded += 1
        module.norm.data.copy_(get(norm_key))
        loaded += 1
        if head_is_weight_normed:
            g = get(head_key + ".parametrizations.weight.original0")
            v = get(head_key + ".parametrizations.weight.original1")
            module.lm_head.weight.data.copy_(g * v / v.norm(dim=tuple(range(1, v.dim())), keepdim=True))
        else:
            module.lm_head.weight.data.copy_(get(head_key))
        loaded += 1
    return loaded


def config_from_qwen3_tts(hf_config: dict[str, Any], component: str) -> DecodeStepConfig:
    """Build a :class:`DecodeStepConfig` from a Qwen3-TTS ``config.json`` dict. ``component``: talker | predictor."""
    talker = hf_config["talker_config"] if "talker_config" in hf_config else hf_config
    cfg = talker if component == "talker" else talker["code_predictor_config"]
    return DecodeStepConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_theta=cfg.get("rope_theta", 1e6),
    )


def load_qwen3_tts_weights(module: DecodeStep, safetensors_path: str | Path, component: str, lm_head_index: int = 0) -> int:
    """Copy real weights into ``module``. ``component``: ``talker`` (codec head) or ``predictor`` (``lm_head.<i>``)."""
    return load_decoder_weights(
        module,
        safetensors_path,
        prefix="talker." if component == "talker" else "talker.code_predictor.",
        head_key="codec_head.weight" if component == "talker" else f"lm_head.{lm_head_index}.weight",
    )


def load_minicpmo_tts_weights(module: DecodeStep, safetensors_path: str | Path) -> int:
    """Copy MiniCPM-o 4.5 TTS-head weights (``tts.*``) into ``module``; its code head is weight-normalised."""
    return load_decoder_weights(module, safetensors_path, prefix="tts.", head_key="head_code.0", head_is_weight_normed=True)


def sanitize_onnx(path: str | Path) -> dict[str, int]:
    """Make a dynamo-exported ONNX file digestible by ``qairt-converter`` and Qualcomm AI Hub, in place.

    Removes ``IsNaN``/``Where`` guard pairs (the converter has no ``IsNaN``), clears ``Reshape``'s ``allowzero``
    attribute (rejected), and promotes rank-0 ``axes`` initializers on ``Unsqueeze``/``Squeeze`` to rank 1 (AI Hub's
    shape inference raises "Inferred shape and existing shape differ in rank" on them). Returns the counts.
    """
    import onnx

    path = Path(path)
    # Whether the file on disk keeps its weights beside it. ``onnx.load``
    # materialises them either way, so this has to be read before the load or
    # the information is gone -- and saving a 1.8 GB tower back inline would
    # both orphan the ``.data`` file and run at the 2 GB protobuf ceiling.
    external = any(
        init.data_location == onnx.TensorProto.EXTERNAL
        for init in onnx.load(str(path), load_external_data=False).graph.initializer
    )
    model = onnx.load(str(path))
    graph = model.graph
    produced = {out: node for node in graph.node for out in node.output}
    renamed: dict[str, str] = {}
    for node in list(graph.node):
        if node.op_type != "Where" or node.input[0] not in produced:
            continue
        guard = produced[node.input[0]]
        if guard.op_type != "IsNaN":
            continue
        source = guard.input[0]
        keep = source if source in (node.input[1], node.input[2]) else None
        if keep is None:
            continue
        renamed[node.output[0]] = keep
        graph.node.remove(node)
        graph.node.remove(guard)
    initializers = {init.name: init for init in graph.initializer}
    constants = {node.output[0]: node for node in graph.node if node.op_type == "Constant"}
    stats = {"nan_guards_removed": len(renamed), "allowzero_cleared": 0, "scalar_axes_fixed": 0}
    scalar_axes: list[str] = []
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in renamed:
                node.input[i] = renamed[name]
        if node.op_type == "Reshape":
            for attr in node.attribute:
                if attr.name == "allowzero" and attr.i:
                    attr.i = 0
                    stats["allowzero_cleared"] += 1
        if node.op_type in ("Unsqueeze", "Squeeze") and len(node.input) > 1:
            axes = node.input[1]
            if axes in initializers and not initializers[axes].dims:
                initializers[axes].dims.extend([1])
                scalar_axes.append(axes)
            elif axes in constants:
                for attr in constants[axes].attribute:
                    if attr.name == "value" and not attr.t.dims:
                        attr.t.dims.extend([1])
                        scalar_axes.append(axes)
    stats["scalar_axes_fixed"] = len(scalar_axes)
    for collection in (graph.value_info, graph.input):
        for info in list(collection):
            if info.name in scalar_axes:
                collection.remove(info)  # stale rank-0 shape annotation
    for out in graph.output:
        if out.name in renamed:
            out.name = renamed[out.name]
    if external:
        # Rewrite the pair. ``convert_model_to_external_data`` only sets the
        # location metadata; ``onnx.save`` is what writes the blob, and the old
        # one has to go first or stale tensors survive under the same name.
        data_file = path.name + ".data"
        stale = path.with_name(data_file)
        if stale.exists():
            stale.unlink()
        onnx.save(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
        )
    else:
        onnx.save(model, str(path))
    return stats


def export_onnx(module: DecodeStep, cache_len: int, path: str | Path, opset: int = 18) -> dict[str, Any]:
    """Export ``module`` to ONNX at ``path`` (dynamo exporter) and sanitize it for the Qualcomm toolchains."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = module.cfg
    args = example_inputs(cfg, cache_len)
    program = torch.onnx.export(
        module,
        args,
        input_names=input_names(cfg),
        output_names=output_names(cfg),
        opset_version=opset,
        dynamo=True,
        optimize=True,
    )
    program.save(str(path))
    stats = sanitize_onnx(path)
    return {"path": str(path), "cache_len": cache_len, "config": asdict(cfg), "sanitize": stats}


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--component", choices=("talker", "predictor"), default="talker")
    parser.add_argument("--cache-len", type=int, default=256)
    parser.add_argument("--lm-head-index", type=int, default=0, help="predictor only: which residual head to attach")
    parser.add_argument("--out-dir", default="./qnn_export")
    parser.add_argument("--no-weights", action="store_true", help="export the graph shape only (random weights)")
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(args.model, allow_patterns=["config.json", "model.safetensors"]))
    cfg = config_from_qwen3_tts(json.loads((snapshot / "config.json").read_text()), args.component)
    module = DecodeStep(cfg).eval()
    if not args.no_weights:
        loaded = load_qwen3_tts_weights(module, snapshot / "model.safetensors", args.component, args.lm_head_index)
        print(f"loaded {loaded} tensors ({sum(p.numel() for p in module.parameters()) / 1e6:.0f}M params)", flush=True)
    out = Path(args.out_dir) / f"{args.component}_step_L{args.cache_len}.onnx"
    manifest = export_onnx(module, args.cache_len, out)
    manifest["model"] = args.model
    manifest["component"] = args.component
    (Path(args.out_dir) / f"{args.component}_step_L{args.cache_len}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
