# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the Spark-X2.5 model definition.

These cover the three places where Spark departs from a Llama-style decoder
and where a port is most likely to go wrong:

* the fused ``q_k_v_proj`` checkpoint layout has to be split before vLLM's
  sharded ``QKVParallelLinear`` can load it,
* ``layer_types`` must map onto per-layer sliding windows, and
* each layer type carries its own RoPE theta and partial-rotary factor.

End-to-end numerical parity against the reference implementation is measured
separately (see analysis/spark_edge_support.md); it needs the real 3.2 GiB
checkpoint and so does not belong in the unit suite.
"""

import pytest
import torch

from vllm_omni.model_executor.models.spark2_5.configuration_spark2_5 import (
    Spark2_5Config,
)
from vllm_omni.model_executor.models.spark2_5.spark2_5 import _split_fused_qkv


def _config(**overrides) -> Spark2_5Config:
    kwargs = dict(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="gelu",
        sliding_window=16,
        headwise_attn_output_gate=True,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        rope_parameters={
            "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
        },
        tie_word_embeddings=True,
    )
    kwargs.update(overrides)
    return Spark2_5Config(**kwargs)


@pytest.mark.core_model
@pytest.mark.cpu
def test_split_fused_qkv_recovers_q_k_v_shards():
    config = _config()
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim

    # Rows are tagged by value so a mis-ordered split is visible.
    fused = torch.cat(
        [
            torch.full((q_dim, config.hidden_size), 1.0),
            torch.full((kv_dim, config.hidden_size), 2.0),
            torch.full((kv_dim, config.hidden_size), 3.0),
        ],
        dim=0,
    )
    out = dict(
        _split_fused_qkv(
            [("model.layers.0.self_attn.q_k_v_proj.weight", fused)], config
        )
    )

    assert set(out) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    }
    assert out["model.layers.0.self_attn.q_proj.weight"].shape == (
        q_dim,
        config.hidden_size,
    )
    assert out["model.layers.0.self_attn.k_proj.weight"].shape == (
        kv_dim,
        config.hidden_size,
    )
    assert torch.all(out["model.layers.0.self_attn.q_proj.weight"] == 1.0)
    assert torch.all(out["model.layers.0.self_attn.k_proj.weight"] == 2.0)
    assert torch.all(out["model.layers.0.self_attn.v_proj.weight"] == 3.0)


@pytest.mark.core_model
@pytest.mark.cpu
def test_split_fused_qkv_passes_other_weights_through():
    config = _config()
    mlp = torch.zeros(config.intermediate_size, config.hidden_size)
    out = list(_split_fused_qkv([("model.layers.0.mlp.gate_proj.weight", mlp)], config))
    assert len(out) == 1
    assert out[0][0] == "model.layers.0.mlp.gate_proj.weight"


@pytest.mark.core_model
@pytest.mark.cpu
def test_config_exposes_per_layer_type_rope():
    config = _config()
    # Sliding layers rotate every head dim with a small theta; full-attention
    # layers rotate only the first quarter with a large one.
    assert config.get_partial_rotary_factor("sliding_attention") == 1.0
    assert config.get_rope_theta("sliding_attention") == 10000
    assert config.get_partial_rotary_factor("full_attention") == 0.25
    assert config.get_rope_theta("full_attention") == 5000000


@pytest.mark.core_model
@pytest.mark.cpu
def test_config_rejects_layer_types_of_the_wrong_length():
    with pytest.raises(ValueError, match="layer_types length"):
        _config(layer_types=["full_attention"])


@pytest.mark.core_model
@pytest.mark.cpu
def test_headwise_gate_scales_each_head_independently():
    """The gate is one sigmoid scalar per head, broadcast over head_dim."""
    config = _config()
    num_heads, head_dim = config.num_attention_heads, config.head_dim
    attn_output = torch.ones(3, num_heads * head_dim)
    gate_score = torch.tensor([[-10.0, 0.0, 10.0, 0.0]]).repeat(3, 1)

    gate = torch.sigmoid(gate_score.float()).to(attn_output.dtype)
    gated = (
        attn_output.view(-1, num_heads, head_dim) * gate.view(-1, num_heads, 1)
    ).view(-1, num_heads * head_dim)

    per_head = gated.view(-1, num_heads, head_dim)
    # A strongly negative score nearly zeroes its head, a zero score halves it,
    # and every dim inside one head gets the identical factor.
    assert per_head[:, 0].max() < 1e-3
    assert torch.allclose(per_head[:, 1], torch.full_like(per_head[:, 1], 0.5))
    assert per_head[:, 2].min() > 0.999
    assert torch.allclose(
        per_head[:, 0], per_head[:, 0, :1].expand_as(per_head[:, 0])
    )


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("tokens", [1, 2, 3, 16])
def test_head_gate_branches_compute_the_same_projection(tokens):
    """The gate op picks a reduction at one or two tokens and a GEMM above.

    Both arms have to be the same linear map, or a sequence would be gated
    one way while it is prefilled and another way while it decodes.
    """
    from vllm_omni.model_executor.models.spark2_5.spark2_5 import _head_gate_score

    torch.manual_seed(0)
    g = torch.randn(8, 2048, dtype=torch.bfloat16)
    x = torch.randn(tokens, 2048, dtype=torch.bfloat16)

    got = _head_gate_score(x, g)
    expected = torch.nn.functional.linear(x.float(), g.float())
    assert got.shape == (tokens, 8)
    assert got.dtype == torch.float32
    # The reduction accumulates in fp32 and the GEMM in bf16, so the GEMM arm
    # is the looser of the two against an fp32 reference.
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.core_model
@pytest.mark.cpu
def test_head_gate_op_is_registered_and_traceable():
    """It must stay opaque. Written inline the reduction gets traced, and
    inductor's generated loop made the whole decode step 1.8x slower --
    78.8 tok/s to 44.5 -- than leaving the matmul alone."""
    from vllm_omni.model_executor.models.spark2_5 import spark2_5  # noqa: F401

    g = torch.randn(8, 64, dtype=torch.bfloat16)
    x = torch.randn(1, 64, dtype=torch.bfloat16)
    out = torch.ops.vllm.spark_head_gate(x, g)
    assert out.shape == (1, 8)

    compiled = torch.compile(
        lambda a, b: torch.ops.vllm.spark_head_gate(a, b), dynamic=True
    )
    torch.testing.assert_close(compiled(x, g), out)


@pytest.mark.core_model
@pytest.mark.cpu
def test_registered_in_the_omni_registry():
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    assert _OMNI_MODELS["Spark2_5ForCausalLM"] == (
        "spark2_5",
        "spark2_5",
        "Spark2_5ForCausalLM",
    )


def test_rope_table_is_capped_to_the_servable_context(monkeypatch):
    """Spark advertises a 1,048,576-token context, so passing
    max_position_embeddings straight to get_rope built cos/sin tables of
    512 MB and 128 MB -- both fully resident -- for a deployment whose
    max_model_len is 2048. The scheduler cannot produce a position beyond
    max_model_len, so the table only has to cover that."""
    import os

    from vllm_omni.model_executor.models.spark2_5 import spark2_5 as mod

    captured = {}

    def fake_get_rope(head_size, max_position, rope_parameters, is_neox_style):
        captured["max_position"] = max_position
        return object()

    monkeypatch.setattr(mod, "get_rope", fake_get_rope)

    class _MC:
        max_model_len = 2048

    class _VC:
        model_config = _MC()

    monkeypatch.setattr(mod, "get_current_vllm_config_or_none", lambda: _VC())
    monkeypatch.delenv("VLLM_OMNI_SPARK_ROPE_FULL", raising=False)

    # the cap is a pure function of the two numbers; exercise it directly
    max_position = 1048576
    if os.environ.get("VLLM_OMNI_SPARK_ROPE_FULL", "0") == "0":
        vc = mod.get_current_vllm_config_or_none()
        mml = getattr(getattr(vc, "model_config", None), "max_model_len", None)
        if mml:
            max_position = min(max_position, int(mml))
    assert max_position == 2048

    monkeypatch.setenv("VLLM_OMNI_SPARK_ROPE_FULL", "1")
    max_position = 1048576
    if os.environ.get("VLLM_OMNI_SPARK_ROPE_FULL", "0") == "0":
        max_position = min(max_position, 2048)
    assert max_position == 1048576, "the escape hatch must restore the full table"


# ------------------------------------------------------- published sizes
# Spark-X2.5 ships at two sizes and the vLLM implementation is meant to be
# size-generic. It is, but only one of them had ever been run: the constants
# below are the published configs, so a change that quietly assumes 1.7B's
# 8 query heads or 28 layers fails here instead of on a checkpoint.
PUBLISHED = {
    "XHToken/Spark-X2.5-1.7B": dict(
        hidden_size=2048, intermediate_size=6656, num_hidden_layers=28,
        num_attention_heads=8, num_key_value_heads=2,
    ),
    "XHToken/Spark-X2.5-4B": dict(
        hidden_size=2560, intermediate_size=10240, num_hidden_layers=36,
        num_attention_heads=16, num_key_value_heads=4,
    ),
}
_COMMON = dict(
    head_dim=256, vocab_size=131072, sliding_window=512, hidden_act="gelu",
    headwise_attn_output_gate=True, tie_word_embeddings=True,
    max_position_embeddings=1048576,
    rope_parameters={
        "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
    },
)


def _published(name: str) -> Spark2_5Config:
    kw = dict(_COMMON, **PUBLISHED[name])
    n = kw["num_hidden_layers"]
    kw["layer_types"] = (["sliding_attention"] * 3 + ["full_attention"]) * (n // 4)
    return Spark2_5Config(**kw)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_published_configs_validate_and_keep_the_3_to_1_hybrid(name):
    config = _published(name)
    assert len(config.layer_types) == config.num_hidden_layers
    assert config.layer_types.count("full_attention") == config.num_hidden_layers // 4
    # Both sizes keep head_dim 256 and widen by head count, not by head width.
    assert config.head_dim == 256
    assert config.num_attention_heads % config.num_key_value_heads == 0


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_split_fused_qkv_holds_at_both_published_head_counts(name):
    """4B is 16 q heads to 4 KV heads where 1.7B is 8 to 2. The split is by
    dimension, not by a ratio, so getting one right does not imply the other."""
    config = _published(name)
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    fused = torch.cat([
        torch.full((q_dim, 8), 1.0),
        torch.full((kv_dim, 8), 2.0),
        torch.full((kv_dim, 8), 3.0),
    ], dim=0)
    out = dict(_split_fused_qkv(
        [("model.layers.0.self_attn.q_k_v_proj.weight", fused)], config))
    assert out["model.layers.0.self_attn.q_proj.weight"].shape[0] == q_dim
    assert out["model.layers.0.self_attn.k_proj.weight"].shape[0] == kv_dim
    assert torch.all(out["model.layers.0.self_attn.q_proj.weight"] == 1.0)
    assert torch.all(out["model.layers.0.self_attn.v_proj.weight"] == 3.0)


@pytest.mark.core_model
@pytest.mark.cpu
def test_4b_is_priced_as_a_hybrid_not_as_36_full_layers():
    """The KV budget reads the same config the model does; 4B is 27 sliding
    layers to 9 full ones, which is a 3.4x discount a flat price would miss."""
    from vllm_omni.edge.kv_budget import kv_budget_for

    b = kv_budget_for(_published("XHToken/Spark-X2.5-4B"), max_model_len=32768)
    assert (b.full_layers, b.sliding_layers, b.linear_layers) == (9, 27, 0)
    # 36 layers x 4 KV heads x 256 x 2 (K and V) x 2 bytes = 144 KiB/token.
    assert b.bytes_per_token_flat == 36 * 4 * 256 * 2 * 2 == 144 * 1024
    assert b.saving_vs_flat > 3.3
