# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the Spark-X2.5 decode-step exporter.

The point of the export is the hybrid cache layout, so that is what these
check: sliding layers stay at a fixed 512 entries however long the context
gets, full layers grow, and the window actually rolls.
"""

import pytest
import torch

from vllm_omni.edge.spark_export import (
    FULL,
    SLIDING,
    SparkDecodeCache,
    SparkDecodeStep,
    SparkStepConfig,
    config_from_spark,
    example_inputs,
    export_onnx,
    input_names,
    kv_cache_bytes,
    output_names,
)

HF_CONFIG = {
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 256,
    "sliding_window": 8,
    "rms_norm_eps": 1e-6,
    "layer_types": [SLIDING, SLIDING, SLIDING, FULL],
    "rope_parameters": {
        "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
    },
}


def _cfg(**kw) -> SparkStepConfig:
    return config_from_spark(HF_CONFIG, **kw)


@pytest.mark.core_model
@pytest.mark.cpu
def test_sliding_cache_is_capped_but_full_cache_grows():
    cfg = _cfg(cache_layout="roll")
    assert cfg.cache_len(SLIDING, 4) == 4  # shorter than the window
    assert cfg.cache_len(SLIDING, 1024) == 8  # capped at sliding_window
    assert cfg.cache_len(SLIDING, 1_000_000) == 8
    assert cfg.cache_len(FULL, 1024) == 1024


@pytest.mark.core_model
@pytest.mark.cpu
def test_rotary_dim_differs_by_layer_type():
    cfg = _cfg()
    assert cfg.rotary_dim(SLIDING) == 16  # all head dims
    assert cfg.rotary_dim(FULL) == 4  # first quarter only


@pytest.mark.core_model
@pytest.mark.cpu
def test_kv_bytes_beats_uniform_attention_and_the_gap_grows():
    cfg = _cfg()
    near = kv_cache_bytes(cfg, 16)
    far = kv_cache_bytes(cfg, 4096)
    assert near["hybrid"] < near["uniform"]
    assert far["hybrid"] < far["uniform"]
    # The saving is a growing fraction, not a constant one.
    assert far["saved"] / far["uniform"] > near["saved"] / near["uniform"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_example_inputs_match_the_declared_signature():
    cfg = _cfg()
    args = example_inputs(cfg, context=1024)
    names = input_names(cfg)
    assert len(args) == len(names)
    caches = args[7:]
    assert len(caches) == 2 * cfg.num_layers
    for i, layer_type in enumerate(cfg.layer_types):
        expected = cfg.cache_len(layer_type, 1024)
        assert caches[2 * i].shape == (1, cfg.num_key_value_heads, expected, cfg.head_dim)
    assert len(output_names(cfg)) == 1 + 2 * cfg.num_layers


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("layout", ["ring", "roll"])
def test_every_layer_returns_what_the_runtime_must_write(layout):
    cfg = _cfg(cache_layout=layout)
    step = SparkDecodeStep(cfg).eval()
    context = 1024
    with torch.no_grad():
        out = step(*example_inputs(cfg, context))
    logits, caches = out[0], out[1:]
    assert logits.shape == (1, cfg.vocab_size)
    for i, layer_type in enumerate(cfg.layer_types):
        k_new = caches[2 * i]
        if layout == "ring":
            # One entry per layer, dropped into the ring slot for this
            # position -- nothing cache-sized crosses the graph boundary.
            assert k_new.shape[2] == 1
        elif layer_type == SLIDING:
            assert k_new.shape[2] == cfg.cache_len(SLIDING, context)
        else:
            assert k_new.shape[2] == 1


@pytest.mark.core_model
@pytest.mark.cpu
def test_ring_and_roll_agree_on_the_same_history():
    """Both layouts must attend over the same window of tokens.

    Roll keeps `sliding_window` entries and drops the oldest after appending;
    ring keeps one fewer and appends in score space. Given a shared history
    they have to produce identical output, or the device numbers measure a
    different model.
    """
    torch.manual_seed(0)
    ring = _cfg(cache_layout="ring", layer_slice=slice(0, 1))
    roll = _cfg(cache_layout="roll", layer_slice=slice(0, 1))
    step_ring = SparkDecodeStep(ring).eval()
    step_roll = SparkDecodeStep(roll).eval()
    step_roll.load_state_dict(step_ring.state_dict())

    context = 64
    args_roll = list(example_inputs(roll, context))
    k_hist, v_hist = args_roll[7], args_roll[8]
    # Ring sees the same history minus the entry roll is about to discard.
    args_ring = list(args_roll)
    args_ring[5] = torch.zeros(1, 1, 1, ring.cache_len(SLIDING, context) + 1)
    args_ring[7] = k_hist[:, :, 1:]
    args_ring[8] = v_hist[:, :, 1:]

    with torch.no_grad():
        out_ring = step_ring(*args_ring)
        out_roll = step_roll(*args_roll)
    assert torch.allclose(out_ring[0], out_roll[0], atol=2e-4, rtol=2e-4)


@pytest.mark.core_model
@pytest.mark.cpu
def test_roll_layout_drops_the_oldest_entry():
    """Roll's rolled window is the old cache shifted by one with the new key
    appended -- what makes a masked pad slot safe to prepend."""
    cfg = _cfg(cache_layout="roll", layer_slice=slice(0, 1))
    step = SparkDecodeStep(cfg).eval()
    args = list(example_inputs(cfg, context=1024))
    k_cache = args[7]
    with torch.no_grad():
        out = step(*args)
    k_new = out[1]
    assert torch.allclose(k_new[:, :, :-1], k_cache[:, :, 1:])


@pytest.mark.core_model
@pytest.mark.cpu
def test_ring_sliding_cache_holds_one_less_than_the_window():
    ring, roll = _cfg(cache_layout="ring"), _cfg(cache_layout="roll")
    assert ring.cache_len(SLIDING, 1024) == HF_CONFIG["sliding_window"] - 1
    assert roll.cache_len(SLIDING, 1024) == HF_CONFIG["sliding_window"]
    # Full-attention layers are unaffected by the layout.
    assert ring.cache_len(FULL, 1024) == roll.cache_len(FULL, 1024) == 1024


@pytest.mark.core_model
@pytest.mark.cpu
def test_layer_slice_selects_one_layer_type():
    types = HF_CONFIG["layer_types"]
    sliding = _cfg(layer_slice=slice(0, 1))
    assert sliding.layer_types == (SLIDING,)
    full = _cfg(layer_slice=slice(types.index(FULL), types.index(FULL) + 1))
    assert full.layer_types == (FULL,)
    assert full.first_layer == types.index(FULL)


@pytest.mark.core_model
@pytest.mark.cpu
def test_unknown_layer_type_is_rejected():
    bad = dict(HF_CONFIG, layer_types=[SLIDING, "linear_attention", SLIDING, FULL])
    with pytest.raises(ValueError, match="unknown layer types"):
        config_from_spark(bad)


@pytest.mark.core_model
@pytest.mark.cpu
def test_gelu_modes_stay_close_to_the_exact_form():
    """The approximations are offered because erf GELU is ~12% of a layer on
    device, but they are only adoptable if they track the trained activation."""
    x = torch.linspace(-6, 6, 4096)
    exact = torch.nn.functional.gelu(x)
    from vllm_omni.edge.spark_export import _gelu

    tanh_err = (_gelu(x, "tanh") - exact).abs().max().item()
    sigmoid_err = (_gelu(x, "sigmoid") - exact).abs().max().item()
    assert torch.equal(_gelu(x, "exact"), exact)
    assert tanh_err < 2e-3
    # The sigmoid form is markedly looser; it is kept for measurement only.
    assert 1e-2 < sigmoid_err < 3e-2
    assert tanh_err < sigmoid_err


@pytest.mark.core_model
@pytest.mark.cpu
def test_unknown_gelu_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown gelu_mode"):
        _cfg(gelu_mode="relu")


@pytest.mark.core_model
@pytest.mark.cpu
def test_unknown_cache_layout_is_rejected():
    with pytest.raises(ValueError, match="unknown cache_layout"):
        _cfg(cache_layout="linked_list")


@pytest.mark.core_model
@pytest.mark.cpu
def test_auto_layout_picks_ring_for_sliding_and_roll_for_full():
    """Ring pays off only on sliding layers, and only there does it compile at
    w4a16, so the default mixes them."""
    cfg = _cfg()  # default cache_layout="auto"
    assert cfg.layout_for(SLIDING) == "ring"
    assert cfg.layout_for(FULL) == "roll"
    assert cfg.cache_len(SLIDING, 1024) == HF_CONFIG["sliding_window"] - 1
    forced = _cfg(cache_layout="roll")
    assert forced.layout_for(SLIDING) == forced.layout_for(FULL) == "roll"


@pytest.mark.core_model
@pytest.mark.cpu
def test_auto_layout_returns_one_entry_for_sliding_and_a_window_for_full():
    cfg = _cfg()
    step = SparkDecodeStep(cfg).eval()
    with torch.no_grad():
        out = step(*example_inputs(cfg, 1024))
    caches = out[1:]
    for i, layer_type in enumerate(cfg.layer_types):
        # Both layouts hand back exactly one new entry here: ring by design,
        # roll because a full layer only ever appends.
        assert caches[2 * i].shape[2] == 1, layer_type


@pytest.mark.core_model
@pytest.mark.cpu
def test_fixed_decode_cache_crosses_window_and_rejects_stale_commit():
    cfg = _cfg()
    state = SparkDecodeCache(cfg, max_context=20)
    prefill = []
    for layer_type in cfg.layer_types:
        length = 7 if layer_type == SLIDING else 10
        values = torch.arange(10 - length, 10, dtype=torch.float32)
        cache = values.view(1, 1, length, 1).expand(
            1, cfg.num_key_value_heads, length, cfg.head_dim
        ).clone()
        prefill.append((cache, cache.clone()))
    state.seed(prefill, position=10)
    assert state.position == 10
    # The ring is physical, so the oldest entry occupies the next write slot.
    ring_slot = 10 % 7
    assert state.buffers[0][0, 0, ring_slot, 0] == 3
    args = state.step_inputs(torch.zeros(1, 1, cfg.hidden_size))
    assert args[5].shape[-1] == 8
    assert args[6].shape[-1] == 21
    assert torch.isneginf(args[6][0, 0, 0, 10:20]).all()
    step = SparkDecodeStep(cfg).eval()
    with torch.no_grad():
        outputs = step(*args)
    before = [buffer.clone() for buffer in state.buffers]
    bad = list(outputs)
    bad[-1] = torch.empty(1, 1, 2, 1)
    with pytest.raises(ValueError, match="output shape mismatch"):
        state.commit(tuple(bad), expected_position=10)
    assert state.position == 10
    assert all(torch.equal(a, b) for a, b in zip(before, state.buffers))
    state.commit(outputs, expected_position=10)
    assert state.position == 11
    assert torch.equal(state.buffers[0][:, :, ring_slot:ring_slot + 1], outputs[1])
    with pytest.raises(ValueError, match="stale or repeated"):
        state.commit(outputs, expected_position=10)
    state.clear()
    assert all(not buffer.any() for buffer in state.buffers)
    with pytest.raises(ValueError, match="not active"):
        state.step_inputs(torch.zeros(1, 1, cfg.hidden_size))


@pytest.mark.core_model
@pytest.mark.cpu
def test_full_cache_bucket_growth_preserves_owned_state():
    cfg = _cfg()
    prefill = []
    for layer_type in cfg.layer_types:
        length = 7 if layer_type == SLIDING else 10
        data = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1)
        cache = data.expand(1, cfg.num_key_value_heads, length, cfg.head_dim).clone()
        prefill.append((cache, cache.clone()))
    state = SparkDecodeCache(cfg, max_context=10)
    direct = SparkDecodeCache(cfg, max_context=12)
    state.seed(prefill, position=10)
    direct.seed(prefill, position=10)
    before = [buffer.clone() for buffer in state.buffers]
    with pytest.raises(ValueError, match="larger capacity"):
        state.grow_full_capacity(10)
    assert all(torch.equal(a, b) for a, b in zip(before, state.buffers))
    state.grow_full_capacity(12)
    assert state.max_context == 12 and state.position == 10
    assert all(torch.equal(a, b) for a, b in zip(state.buffers, direct.buffers))
    x = torch.zeros(1, 1, cfg.hidden_size)
    assert all(torch.equal(a, b) for a, b in zip(state.step_inputs(x),
                                                 direct.step_inputs(x)))
    step = SparkDecodeStep(cfg).eval()
    with torch.no_grad():
        outputs = step(*state.step_inputs(x))
    state.commit(outputs, expected_position=10)
    direct.commit(outputs, expected_position=10)
    assert state.position == direct.position == 11
    assert all(torch.equal(a, b) for a, b in zip(state.buffers, direct.buffers))
    state.clear()
    with pytest.raises(ValueError, match="active state"):
        state.grow_full_capacity(14)


@pytest.mark.core_model
@pytest.mark.cpu
def test_fixed_decode_cache_masks_unfilled_ring_slots():
    cfg = _cfg()
    state = SparkDecodeCache(cfg, max_context=16)
    prefill = [
        (torch.zeros(1, cfg.num_key_value_heads, 2, cfg.head_dim),
         torch.zeros(1, cfg.num_key_value_heads, 2, cfg.head_dim))
        for _ in cfg.layer_types
    ]
    state.seed(prefill, position=2)
    args = state.step_inputs(torch.zeros(1, 1, cfg.hidden_size))
    assert torch.equal(args[5][0, 0, 0, :2], torch.zeros(2))
    assert torch.isneginf(args[5][0, 0, 0, 2:7]).all()
    assert args[5][0, 0, 0, -1] == 0


@pytest.mark.core_model
@pytest.mark.cpu
def test_roll_decode_cache_keeps_chronological_window_across_boundary():
    cfg = _cfg(cache_layout="roll")
    state = SparkDecodeCache(cfg, max_context=20)
    prefill = []
    for layer_type in cfg.layer_types:
        length = 6
        values = torch.arange(length, dtype=torch.float32)
        cache = values.view(1, 1, length, 1).expand(
            1, cfg.num_key_value_heads, length, cfg.head_dim
        ).clone()
        prefill.append((cache, cache.clone()))
    state.seed(prefill, position=6)
    assert state.buffers[0][0, 0, :, 0].tolist() == [0, 0, 0, 1, 2, 3, 4, 5]
    args = state.step_inputs(torch.zeros(1, 1, cfg.hidden_size))
    assert torch.isneginf(args[5][0, 0, 0, 0])
    assert torch.equal(args[5][0, 0, 0, 1:], torch.zeros(7))
    step = SparkDecodeStep(cfg).eval()
    with torch.no_grad():
        first = step(*args)
    assert first[1].shape[2] == cfg.sliding_window
    state.commit(first, expected_position=6)
    assert torch.equal(state.buffers[0], first[1])
    second_args = state.step_inputs(torch.zeros(1, 1, cfg.hidden_size))
    assert torch.equal(second_args[5], torch.zeros_like(second_args[5]))
    with torch.no_grad():
        second = step(*second_args)
    state.commit(second, expected_position=7)
    assert torch.equal(state.buffers[0][:, :, :-1], first[1][:, :, 1:])
    assert state.position == 8


@pytest.mark.core_model
@pytest.mark.cpu
def test_bf16_reference_step_uses_fp32_residual_and_bf16_cache(tmp_path):
    cfg = _cfg(arithmetic_mode="hf_bf16_reference")
    state = SparkDecodeCache(cfg, max_context=16, dtype=torch.bfloat16)
    prefill = [
        (torch.zeros(1, cfg.num_key_value_heads, 2, cfg.head_dim,
                     dtype=torch.bfloat16),
         torch.zeros(1, cfg.num_key_value_heads, 2, cfg.head_dim,
                     dtype=torch.bfloat16))
        for _ in cfg.layer_types
    ]
    state.seed(prefill, position=2)
    step = SparkDecodeStep(cfg).eval()
    with torch.no_grad():
        out = step(*state.step_inputs(torch.zeros(1, 1, cfg.hidden_size)))
    assert out[0].dtype == torch.bfloat16
    assert all(cache.dtype == torch.bfloat16 for cache in out[1:])
    state.commit(out, expected_position=2)
    assert state.position == 3
    with pytest.raises(ValueError, match="not a qualified mobile export"):
        export_onnx(step, 4, tmp_path / "unqualified.onnx")


@pytest.mark.core_model
@pytest.mark.cpu
def test_unknown_arithmetic_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown arithmetic_mode"):
        _cfg(arithmetic_mode="silent_precision_change")
