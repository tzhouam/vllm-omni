# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The XHToken team. All rights reserved.
"""Inference-only Spark-X2.5 model compatible with HuggingFace weights.

Spark-X2.5 (XHToken/iFlytek) is a small on-device model built for agentic
tool use. Three traits separate it from a Llama-style decoder and drive the
implementation below:

* **Hybrid attention.** ``layer_types`` interleaves three 512-token
  ``sliding_attention`` layers with one ``full_attention`` layer, which is what
  keeps the 1M-token context affordable on a phone.
* **Per-layer-type RoPE.** Sliding layers rotate every one of the 256 head
  dims with theta 1e4; full layers rotate only the first quarter with theta
  5e6. Both come out of the nested ``rope_parameters`` dict.
* **Head-wise attention output gate.** A ``g_proj`` of shape
  ``[hidden, num_heads]`` reads the *same* post-norm input as QKV and scales
  each head's attention output by a sigmoid before ``out_proj``.

The checkpoint stores QKV pre-fused as a single ``q_k_v_proj``; load_weights
splits it so vLLM's sharded ``QKVParallelLinear`` can take it.
"""

import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op

from .configuration_spark2_5 import Spark2_5Config


class Spark2_5MLP(nn.Module):
    def __init__(
        self,
        config: Spark2_5Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        # Spark's config validator rejects anything but exact GELU.
        if config.hidden_act != "gelu":
            raise ValueError(
                f"Spark-X2.5 only supports hidden_act='gelu', got {config.hidden_act!r}"
            )
        self.act_fn = get_act_and_mul_fn(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Spark2_5Attention(nn.Module):
    def __init__(
        self,
        config: Spark2_5Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # One scalar gate per attention head, so this projection is only
        # [hidden, num_heads] wide. It is deliberately a plain parameter
        # rather than a ColumnParallelLinear: vLLM's CPU backend routes any
        # narrow linear to the sgl-kernel packed GEMM, whose fake tensor rule
        # assumes the packed weight keeps N in dim 0. VNNI packing does not
        # for N this small, so the shape it reports is K rather than N and
        # torch.compile dies tracing the gate multiply. A 2048x8 matmul has
        # nothing to gain from the packed AMX kernel anyway.
        self.headwise_attn_output_gate = config.headwise_attn_output_gate
        if self.headwise_attn_output_gate:
            if config.attention_bias:
                raise NotImplementedError(
                    "Spark-X2.5 with attention_bias and a head-wise gate is "
                    "not supported"
                )
            if config.gate_attn_act_mode not in ("sigmoid", "silu"):
                raise ValueError(
                    f"Unsupported gate_attn_act_mode: {config.gate_attn_act_mode}"
                )
            self.gate_attn_act_mode = config.gate_attn_act_mode
            self.g_weight = nn.Parameter(
                torch.empty(
                    self.num_heads, config.hidden_size, dtype=torch.get_default_dtype()
                )
            )
            set_weight_attrs(self.g_weight, {"weight_loader": self._load_g_weight})
        else:
            self.g_weight = None

        # Only CPU pays a thread barrier per op that a 2048x8 projection
        # cannot amortize; on GPU the plain matmul is already free. The
        # override exists so the choice can be ablated on a new machine
        # rather than assumed.
        override = os.environ.get("VLLM_OMNI_SPARK_GATE_REDUCTION", "auto")
        self.gate_via_reduction = (
            current_platform.is_cpu() if override == "auto" else override == "1"
        )

        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = config.sliding_window if self.is_sliding else None

        # Sliding and full layers are trained with different theta and
        # different partial-rotary factors; both live under rope_parameters.
        rope_parameters = dict(config.rope_parameters[layer_type])
        rope_parameters.setdefault("rope_type", "default")

        # Spark advertises a 1 048 576-token context, and vLLM builds the RoPE
        # cos/sin table over the whole of it -- `torch.arange(max_position)` --
        # regardless of how long a sequence the engine will actually accept.
        # Measured with mincore on a 2048-token deployment: 512 MB resident for
        # the full-attention layers (head_dim 256) and 128 MB for the sliding
        # ones (partial rotary, 64 wide), 640 MB for two tables that need
        # 1.25 MB between them. The table only has to cover positions the
        # scheduler can produce, which vLLM caps at max_model_len.
        max_position = config.max_position_embeddings
        if os.environ.get("VLLM_OMNI_SPARK_ROPE_FULL", "0") == "0":
            vllm_config = get_current_vllm_config_or_none()
            model_config = (
                getattr(vllm_config, "model_config", None) if vllm_config else None
            )
            max_model_len = getattr(model_config, "max_model_len", None)
            if max_model_len:
                max_position = min(max_position, int(max_model_len))

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def _load_g_weight(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Take just the gates belonging to this rank's query heads."""
        tp_rank = get_tensor_model_parallel_rank()
        shard = loaded_weight.narrow(0, tp_rank * self.num_heads, self.num_heads)
        param.data.copy_(shard)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)

        if self.g_weight is not None:
            if self.gate_via_reduction:
                gate_score = torch.ops.vllm.spark_head_gate(
                    hidden_states, self.g_weight
                )
            else:
                gate_score = torch.nn.functional.linear(
                    hidden_states, self.g_weight
                ).float()
            if self.gate_attn_act_mode == "sigmoid":
                gate = torch.sigmoid(gate_score)
            else:
                gate = nn.functional.silu(gate_score)
            gate = gate.to(attn_output.dtype)
            # Reshape only the feature dim: a view(-1, ...) here would make
            # torch.compile re-derive the symbolic token count and fail.
            attn_output = attn_output.unflatten(
                -1, (self.num_heads, self.head_dim)
            ) * gate.unsqueeze(-1)
            attn_output = attn_output.flatten(-2)

        output, _ = self.out_proj(attn_output)
        return output


class Spark2_5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Spark2_5Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Spark2_5Attention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Spark2_5MLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Spark2_5Model(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "model.embedding.": "model.embed_tokens.",
            ".self_attn.g_proj.weight": ".self_attn.g_weight",
        },
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "",
        defer_final_norm: bool = False,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.defer_final_norm = defer_final_norm

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Spark2_5DecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.defer_final_norm:
            # The explicitly selected external output-head graph owns this
            # final RMS norm. Keep vLLM's decoder/KV and return its exact
            # pre-norm residual sum; a graph failure must fail the request.
            return hidden_states + residual if residual is not None else hidden_states
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def _head_gate_score(
    hidden_states: torch.Tensor, g_weight: torch.Tensor
) -> torch.Tensor:
    """Head-gate projection, routed by token count.

    Eight outputs is far too narrow for a GEMM to pay for its thread fan-out:
    at one token `aten::mm` costs 34 us on 24 cores, 0.9 ms of a decode step,
    nearly all of it barrier. The same arithmetic as a broadcast-and-reduce
    costs 8 us, because ATen's grain size keeps a 16 K-element reduction on
    one thread. Above a couple of tokens the GEMM wins again.

    It has to be an opaque op. Written inline the reduction is traced by
    `@support_torch_compile`, and inductor's generated loop is far slower than
    either eager form -- measured 44.5 tok/s against 78.8.
    """
    if hidden_states.shape[0] <= _GATE_REDUCE_MAX_TOKENS:
        return (hidden_states.unsqueeze(-2).float() * g_weight.float()).sum(-1)
    return torch.nn.functional.linear(hidden_states, g_weight).float()


def _head_gate_score_fake(
    hidden_states: torch.Tensor, g_weight: torch.Tensor
) -> torch.Tensor:
    return torch.empty(
        (hidden_states.shape[0], g_weight.shape[0]),
        dtype=torch.float32,
        device=hidden_states.device,
    )


_GATE_REDUCE_MAX_TOKENS = 2

direct_register_custom_op(
    op_name="spark_head_gate",
    op_func=_head_gate_score,
    mutates_args=[],
    fake_impl=_head_gate_score_fake,
)


def _split_fused_qkv(
    weights: Iterable[tuple[str, torch.Tensor]],
    config: Spark2_5Config,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Split the checkpoint's fused ``q_k_v_proj`` into q/k/v.

    vLLM's ``QKVParallelLinear`` loads one shard at a time so that tensor
    parallelism can slice heads; the Spark checkpoint ships the three
    projections already concatenated, so undo that here.
    """
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    for name, loaded in weights:
        if ".self_attn.q_k_v_proj." in name:
            q, k, v = loaded.split([q_dim, kv_dim, kv_dim], dim=0)
            for shard, tensor in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
                yield name.replace("q_k_v_proj", shard), tensor
        else:
            yield name, loaded


class Spark2_5ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    hf_to_vllm_mapper = Spark2_5Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        external_head_spec = os.environ.get("VLLM_OMNI_SPARK_EXTERNAL_HEAD_SPEC")
        if external_head_spec and (
            vllm_config.parallel_config.tensor_parallel_size != 1
            or vllm_config.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("the experimental Spark external head requires TP=PP=1")

        self.model = Spark2_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
            defer_final_norm=bool(external_head_spec),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self._external_head = None
        if external_head_spec:
            from vllm_omni.edge.local.spark_external_head import SparkExternalOutputHead

            self._external_head = SparkExternalOutputHead(
                vllm_config.model_config.model, external_head_spec
            )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        # Omni's AR model runner hands every stage model ``sampling_metadata``,
        # ``logits_index`` and ``sampler``; the omni-native models take them in
        # ``**kwargs``. Spark samples through vLLM's own sampler, and the
        # decoder stack reads none of them, so they stop here rather than
        # widening ``Spark2_5Model.forward`` to a signature it ignores.
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self._external_head is not None:
            logits = self._external_head.compute_logits(hidden_states)
            if len(self._external_head.reference_comparisons) < self._external_head.reference_compare_limit:
                normalized = self.model.norm(hidden_states)
                reference = self.logits_processor(self.lm_head, normalized)
                self._external_head.compare_reference(logits, reference)
            return logits
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # vLLM 0.29.0 dropped AutoWeightsLoader's ``skip_prefixes``: the loader now
        # detects tied parameters itself (``_get_tied_embedding_params``) and loads
        # the first qualname only, so the tied ``lm_head.`` needs no explicit skip.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            _split_fused_qkv(weights, self.config), mapper=self.hf_to_vllm_mapper
        )
