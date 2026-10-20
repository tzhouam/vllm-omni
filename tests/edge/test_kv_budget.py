# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-sized KV budgets, including hybrid attention."""

import pytest

from vllm_omni.edge.kv_budget import dtype_bytes, kv_budget_for

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _spark(**kw) -> _Cfg:
    """Spark-X2.5-1.7B: 28 layers, 3 sliding (512) to 1 full, 2 KV heads, 256."""
    layer_types = ["sliding_attention"] * 3 + ["full_attention"]
    base = dict(
        num_hidden_layers=28, num_attention_heads=8, num_key_value_heads=2,
        head_dim=256, hidden_size=2048, sliding_window=512,
        layer_types=layer_types * 7,
    )
    base.update(kw)
    return _Cfg(**base)


def test_flat_bytes_per_token_matches_the_hand_calculation():
    # 28 layers x 2 KV heads x 256 head_dim x 2 (K and V) x 2 bytes = 56 KiB.
    b = kv_budget_for(_spark(), max_model_len=2048)
    assert b.bytes_per_token_flat == 28 * 2 * 256 * 2 * 2
    assert b.bytes_per_token_flat == 56 * 1024


def test_hybrid_attention_is_charged_for_its_real_cache():
    b = kv_budget_for(_spark(), max_model_len=32768)
    assert (b.full_layers, b.sliding_layers, b.window) == (7, 21, 512)
    assert b.token_layers_hybrid == 7 * 32768 + 21 * 512
    assert b.token_layers_flat == 28 * 32768
    # The recommended budget stays flat, because vLLM's CPU accounting is flat.
    assert b.bytes_total > b.bytes_total_hybrid
    assert b.saving_vs_flat == pytest.approx(917504 / 240128, rel=1e-3)


def test_a_short_context_never_pays_for_a_longer_window():
    """A 512-window layer at a 128-token context caches 128, not 512."""
    b = kv_budget_for(_spark(), max_model_len=128)
    assert b.token_layers_hybrid == 28 * 128


def test_budget_scales_with_sequences_and_dtype():
    one = kv_budget_for(_spark(), max_model_len=2048, max_num_seqs=1)
    four = kv_budget_for(_spark(), max_model_len=2048, max_num_seqs=4)
    assert four.bytes_total == 4 * one.bytes_total
    fp8 = kv_budget_for(_spark(), max_model_len=2048, kv_dtype="fp8")
    assert fp8.bytes_total * 2 == one.bytes_total


def test_edge_sized_budget_is_far_under_the_server_default():
    """The measured point: 2048 tokens needs ~112 MiB, not the 4 GiB default."""
    b = kv_budget_for(_spark(), max_model_len=2048)
    mib = b.bytes_total / 2**20
    assert 130 < mib < 150, mib          # flat 56 KiB/token x 2048 x 1.25
    assert b.bytes_total < 4 * 2**30 / 25  # vs the 4 GiB default
    assert b.bytes_total_hybrid / 2**20 < 70


def test_plain_model_without_layer_types_uses_a_flat_budget():
    cfg = _Cfg(num_hidden_layers=32, num_attention_heads=32,
               num_key_value_heads=8, hidden_size=4096)
    b = kv_budget_for(cfg, max_model_len=4096)
    assert b.sliding_layers == 0 and b.window is None
    assert b.token_layers_hybrid == b.token_layers_flat
    assert b.saving_vs_flat == 1.0
    assert b.bytes_total == b.bytes_total_hybrid
    # head_dim inferred from hidden_size / heads.
    assert b.bytes_per_token_flat == 32 * 8 * 128 * 2 * 2


def test_sliding_window_without_layer_types_is_not_assumed():
    """A config with a window but no per-layer map might apply it everywhere or
    nowhere; charging the smaller cache would risk preemption."""
    cfg = _Cfg(num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=8,
               head_dim=64, sliding_window=256)
    b = kv_budget_for(cfg, max_model_len=4096)
    assert b.sliding_layers == 0
    assert b.token_layers_hybrid == b.token_layers_flat


def test_unreadable_config_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="num_hidden_layers"):
        kv_budget_for(_Cfg(), max_model_len=2048)
    with pytest.raises(ValueError, match="head_dim"):
        kv_budget_for(_Cfg(num_hidden_layers=4), max_model_len=2048)


def test_dtype_bytes_covers_the_kv_dtypes_vllm_accepts():
    assert dtype_bytes("bfloat16") == 2
    assert dtype_bytes("float32") == 4
    assert dtype_bytes("fp8") == 1
    assert dtype_bytes("something_unknown") == 2  # conservative default


def test_summary_names_the_shape_it_priced():
    s = kv_budget_for(_spark(), max_model_len=32768).summary()
    assert "flat" in s and "21 sliding@512" in s and "MiB" in s


def _qwen3_8(**kw) -> _Cfg:
    """Qwen3.8-27B: 64 layers, 3 linear (gated DeltaNet) to 1 full, 4 KV
    heads x 256. The linear layers hold a recurrent state and no KV."""
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    base = dict(
        num_hidden_layers=64, num_attention_heads=24, num_key_value_heads=4,
        head_dim=256, hidden_size=5120, layer_types=layer_types * 16,
        linear_num_key_heads=16, linear_num_value_heads=48,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, mamba_ssm_dtype="float32",
    )
    base.update(kw)
    return _Cfg(**base)


class _Outer:
    """A multimodal config: the decoder geometry lives one level down."""

    def __init__(self, text):
        self.text_config = text
        self.model_type = "qwen3_5"

    def get_text_config(self):
        return self.text_config


def test_linear_attention_layers_are_charged_no_kv_at_all():
    """Not "a shorter cache" -- none. A flat 64-layer price is 4x the truth."""
    b = kv_budget_for(_qwen3_8(), max_model_len=8192)
    assert (b.linear_layers, b.full_layers, b.sliding_layers) == (48, 16, 0)
    # 16 full layers x 4 KV heads x 256 x 2 (K and V) x 2 bytes = 64 KiB.
    assert b.bytes_per_token_flat == 16 * 4 * 256 * 2 * 2 == 64 * 1024
    assert b.token_layers_flat == 16 * 8192


def test_the_recurrent_state_is_sized_from_vllms_own_shapes():
    """conv (kernel-1, 2*k_heads*k_dim + v_heads*v_dim); temporal
    (v_heads, v_dim, k_dim) in mamba_ssm_dtype."""
    b = kv_budget_for(_qwen3_8(), max_model_len=8192)
    conv_dim = 128 * 16 * 2 + 128 * 48
    per_layer = (4 - 1) * conv_dim * 2 + 48 * 128 * 128 * 4
    assert b.state_bytes_per_seq == 48 * per_layer
    assert b.state_bytes_per_seq / 2**20 == pytest.approx(147.0, abs=0.5)
    assert b.state_bytes_total == b.state_bytes_per_seq  # max_num_seqs=1


def test_the_state_is_inside_the_budget_and_scales_with_sequences():
    """It comes out of the same pool as the KV cache, so a budget that
    reports only the cache is short before the first token."""
    one = kv_budget_for(_qwen3_8(), max_model_len=8192, max_num_seqs=1)
    two = kv_budget_for(_qwen3_8(), max_model_len=8192, max_num_seqs=2)
    assert two.state_bytes_total == 2 * one.state_bytes_total
    assert one.bytes_total > one.state_bytes_total
    kv_only = int(one.token_layers_flat * one.bytes_per_token_flat / 16 * 1.25)
    assert one.bytes_total == kv_only + one.state_bytes_total


def test_the_state_dominates_a_short_context():
    """147 MiB of state against 64 KiB/token: the cache only outgrows it past
    ~2350 tokens, so over an edge deployment's range the constant is the
    bigger half. The crossover is on raw bytes -- ``headroom`` is a safety
    factor on the cache, not a cost the state has to beat."""
    assert kv_budget_for(
        _qwen3_8(), max_model_len=2048
    ).state_crossover_tokens == pytest.approx(2352, abs=8)
    short = kv_budget_for(_qwen3_8(), max_model_len=1024)
    assert short.state_bytes_total > short.bytes_total - short.state_bytes_total
    long = kv_budget_for(_qwen3_8(), max_model_len=32768)
    assert long.state_bytes_total < long.bytes_total - long.state_bytes_total


def test_a_model_without_linear_layers_reports_no_state():
    b = kv_budget_for(_spark(), max_model_len=2048)
    assert b.linear_layers == 0
    assert b.state_bytes_per_seq == 0
    assert b.state_crossover_tokens == float("inf")


def test_a_multimodal_config_is_resolved_to_its_text_config():
    """Qwen3_5Config has no num_hidden_layers; this used to raise."""
    nested = kv_budget_for(_Outer(_qwen3_8()), max_model_len=8192)
    direct = kv_budget_for(_qwen3_8(), max_model_len=8192)
    assert nested == direct


def test_linear_layers_without_their_geometry_raise_rather_than_guess():
    cfg = _qwen3_8()
    del cfg.linear_num_value_heads
    with pytest.raises(ValueError, match="linear_num_value_heads"):
        kv_budget_for(cfg, max_model_len=8192)


def test_summary_names_the_state_it_priced():
    s = kv_budget_for(_qwen3_8(), max_model_len=8192).summary()
    assert "recurrent state for 48 linear layers" in s
