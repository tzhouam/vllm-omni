# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The XHToken team and the HuggingFace Inc. team.
"""Spark-X2.5 configuration, vendored so the model loads without
``trust_remote_code``.

Mirrors ``configuration_spark.py`` shipped with XHToken/Spark-X2.5-*.
"""

from transformers import PretrainedConfig


class Spark2_5Config(PretrainedConfig):
    model_type = "spark2_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="gelu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        headwise_attn_output_gate=False,
        gate_attn_act_mode="sigmoid",
        sliding_window=None,
        layer_types=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible "
                f"by num_key_value_heads ({num_key_value_heads})"
            )
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.gate_attn_act_mode = gate_attn_act_mode
        self.sliding_window = sliding_window
        self.rope_parameters = rope_parameters

        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) must match "
                f"num_hidden_layers ({num_hidden_layers})"
            )
        self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_rope_theta(self, layer_type):
        return (self.rope_parameters or {}).get(layer_type, {}).get("rope_theta", 10000)

    def get_partial_rotary_factor(self, layer_type):
        return (self.rope_parameters or {}).get(layer_type, {}).get(
            "partial_rotary_factor", 1.0
        )


__all__ = ["Spark2_5Config"]
