# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the cpu_int4 W4A16 method."""

import pytest
import torch

from vllm_omni.model_executor.layers.quantization.cpu_int4 import (
    CPUInt4Config,
    _to_tinygemm,
    dequantize,
    pack_nibbles,
    quantize_groupwise,
    to_split_halves,
    unpack_nibbles,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _weights(n=64, k=256, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, k)


def test_nibble_pack_round_trips():
    q = torch.randint(0, 16, (32, 128), dtype=torch.int32)
    assert torch.equal(unpack_nibbles(pack_nibbles(q)), q)


def test_quantized_weights_stay_on_the_four_bit_grid():
    w = _weights()
    packed, scale, zero = quantize_groupwise(w, 64)
    codes = unpack_nibbles(packed)
    assert codes.min() >= 0 and codes.max() <= 15
    assert packed.dtype == torch.uint8
    assert packed.shape == (64, 128)
    assert scale.shape == zero.shape == (64, 4)


def test_group_extremes_are_reproduced_exactly():
    """min/max grid: the two ends of each group must dequantize back to
    themselves, which is what makes the offset form worth its extra bytes."""
    w = _weights(n=16, k=128)
    group = 64
    packed, scale, zero = quantize_groupwise(w, group)
    codes = unpack_nibbles(packed).float().reshape(16, 128 // group, group)
    deq = (codes - 8.0) * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    wg = w.reshape(16, 128 // group, group)
    torch.testing.assert_close(deq.amax(-1), wg.amax(-1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(deq.amin(-1), wg.amin(-1), rtol=1e-5, atol=1e-5)


def test_split_halves_is_a_relabelling_of_the_same_codes():
    w = _weights()
    packed, _, _ = quantize_groupwise(w, 64)
    halves = to_split_halves(packed)
    codes = unpack_nibbles(packed)
    k = codes.shape[1]
    assert torch.equal((halves & 0x0F).to(torch.int32), codes[:, : k // 2])
    assert torch.equal((halves >> 4).to(torch.int32), codes[:, k // 2 :])


@pytest.mark.parametrize("group", [32, 64, 128])
def test_dequantize_path_agrees_with_the_tinygemm_kernel(group):
    """The two GEMM paths must be the same function of the weights.

    Prefill takes the dequantize-then-AMX path and decode the tinygemm one;
    a mismatch here would show up as a model that generates fine but answers
    differently depending on prompt length, which no throughput benchmark
    would catch.
    """
    w = _weights(n=64, k=256)
    packed, scale, zero = quantize_groupwise(w, group)
    inner, sz = _to_tinygemm(packed, scale, zero)
    halves = to_split_halves(packed)
    sc, ze = scale.to(torch.bfloat16), zero.to(torch.bfloat16)

    x = torch.randn(8, 256, dtype=torch.bfloat16)
    via_kernel = torch.ops.aten._weight_int4pack_mm_for_cpu(x, inner, group, sz)
    via_dequant = torch.nn.functional.linear(x, dequantize(halves, sc, ze, group))

    # Scale-free: both sides accumulate in bf16, so the gap is bf16 rounding
    # on outputs whose magnitude grows with K, not a layout disagreement.
    a, b = via_kernel.float(), via_dequant.float()
    snr = 20 * torch.log10(a.norm() / (a - b).norm())
    assert snr > 40.0, f'paths disagree beyond bf16 rounding: {snr:.1f} dB'



def test_four_bit_error_is_small_enough_to_be_useful():
    w = _weights(n=256, k=1024)
    packed, scale, zero = quantize_groupwise(w, 64)
    codes = unpack_nibbles(packed).float().reshape(256, 1024 // 64, 64)
    deq = ((codes - 8.0) * scale.unsqueeze(-1) + zero.unsqueeze(-1)).reshape(256, 1024)
    snr = 20 * torch.log10(w.norm() / (w - deq).norm())
    assert snr > 19.0, snr


def test_config_round_trips_and_reports_bf16_only():
    cfg = CPUInt4Config.from_config(
        {"quant_method": "cpu_int4", "group_size": 128, "prefill_dequant": False}
    )
    assert cfg.group_size == 128
    assert cfg.prefill_dequant is False
    assert cfg.get_supported_act_dtypes() == [torch.bfloat16]
    assert cfg.get_name() == "cpu_int4"


def test_registered_into_vllms_quantization_registry():
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )

    import vllm_omni.model_executor.layers.quantization  # noqa: F401

    assert "cpu_int4" in QUANTIZATION_METHODS
    assert get_quantization_config("cpu_int4") is CPUInt4Config


def test_ignore_list_falls_back_to_full_precision():
    cfg = CPUInt4Config(group_size=64, ignore=["lm_head"])
    from vllm.model_executor.layers.linear import (
        ReplicatedLinear,
        UnquantizedLinearMethod,
    )

    layer = ReplicatedLinear.__new__(ReplicatedLinear)
    assert isinstance(cfg.get_quant_method(layer, "lm_head"), UnquantizedLinearMethod)
    assert not isinstance(
        cfg.get_quant_method(layer, "model.layers.0.mlp.down_proj"),
        UnquantizedLinearMethod,
    )


# ------------------------------------------------- direct dequant (one copy)

def test_direct_dequant_reproduces_the_row_major_path_exactly():
    """The prefill path used to need a second packed copy of every weight --
    740 MB on Spark -- purely because the tinygemm layout was treated as
    opaque. Reading it directly has to give bit-identical weights, or the
    saving is bought with an accuracy change nobody asked for."""
    from vllm_omni.model_executor.layers.quantization.cpu_int4 import (
        _dequantize_from_int4pack,
        _dequantize_halves,
        _to_tinygemm,
        quantize_groupwise,
        to_split_halves,
    )

    torch.manual_seed(0)
    for n, k, group in ((2048, 2048, 128), (3072, 2048, 128), (256, 512, 64)):
        w = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
        packed, scale, zero = quantize_groupwise(w, group)
        int4pack, sz = _to_tinygemm(packed, scale, zero)
        ref = _dequantize_halves(
            to_split_halves(packed), scale.to(torch.bfloat16),
            zero.to(torch.bfloat16), group,
        )
        got = _dequantize_from_int4pack(int4pack, sz, group)
        assert torch.equal(got, ref), f"N={n} K={k} g={group}"


def test_the_packed_layout_this_depends_on_is_what_torch_actually_emits():
    """Pins the measured layout: [N/64][K][32 bytes], byte d holding output
    channel d in its low nibble and d+32 in its high one. A torch release that
    repacks differently must fail here rather than silently return wrong
    weights."""
    from vllm_omni.model_executor.layers.quantization.cpu_int4 import (
        INNER_K_TILES,
        TINYGEMM_N_BLOCK,
    )

    n, k = 128, 64
    codes = torch.randint(0, 16, (n, k), dtype=torch.int32)
    packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(codes, INNER_K_TILES)
    blocks = packed.reshape(n // TINYGEMM_N_BLOCK, k, TINYGEMM_N_BLOCK // 2)
    lo = (blocks & 0x0F).to(torch.int32)
    hi = (blocks >> 4).to(torch.int32)
    rebuilt = torch.cat([lo, hi], dim=2).permute(0, 2, 1).reshape(n, k)
    assert torch.equal(rebuilt, codes)


def test_shapes_that_do_not_tile_the_block_keep_the_second_copy():
    """N=80 packs as a 64 block plus a 16 tail, which the reshape would
    mis-read. Those layers must fall back, not be quietly wrong."""
    from vllm_omni.model_executor.layers.quantization.cpu_int4 import (
        supports_direct_dequant,
    )

    assert supports_direct_dequant(2048)
    assert supports_direct_dequant(3072)
    assert supports_direct_dequant(13312)
    assert supports_direct_dequant(131072)
    assert supports_direct_dequant(32)        # below the block, tiles exactly
    assert not supports_direct_dequant(80)    # 64 + 16 tail
    assert not supports_direct_dequant(144)


def test_a_layer_with_prefill_dequant_no_longer_allocates_a_second_copy():
    from vllm_omni.model_executor.layers.quantization.cpu_int4 import (
        _finalize_int4,
        quantize_groupwise,
    )

    n, k, group = 512, 512, 128
    w = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    packed, scale, zero = quantize_groupwise(w, group)
    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale.to(torch.bfloat16), requires_grad=False)
    layer.weight_zero = torch.nn.Parameter(zero.to(torch.bfloat16), requires_grad=False)
    _finalize_int4(layer, keep_row_major=False, prefill_dequant=True)

    assert layer.cpu_int4_direct_dequant is True
    assert layer.weight_halves is None
    assert layer.weight_group_scale is None
    assert layer.weight_group_zero is None
    # and the prefill path still works off the remaining layout
    x = torch.randn(8, k, dtype=torch.bfloat16)
    out = torch.ops.vllm.cpu_int4_linear(
        x, layer.weight_int4pack, layer.weight_scales_and_zeros,
        None, None, None, group, 2, True,
    )
    assert out.shape == (8, n) and torch.isfinite(out).all()


def test_the_escape_hatch_restores_the_second_copy(monkeypatch):
    """The two paths trade 615 MB against 1.9x prefill, so both have to remain
    reachable -- and the A/B that measured them needs identical code."""
    from vllm_omni.model_executor.layers.quantization import cpu_int4

    n, k, group = 512, 512, 128
    w = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    packed, scale, zero = cpu_int4.quantize_groupwise(w, group)

    def build():
        layer = torch.nn.Module()
        layer.weight_packed = torch.nn.Parameter(packed, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            scale.to(torch.bfloat16), requires_grad=False)
        layer.weight_zero = torch.nn.Parameter(
            zero.to(torch.bfloat16), requires_grad=False)
        cpu_int4._finalize_int4(layer, keep_row_major=False, prefill_dequant=True)
        return layer

    monkeypatch.setenv("VLLM_OMNI_CPU_INT4_DIRECT_DEQUANT", "0")
    old = build()
    assert old.cpu_int4_direct_dequant is False
    assert old.weight_halves is not None

    monkeypatch.setenv("VLLM_OMNI_CPU_INT4_DIRECT_DEQUANT", "1")
    new = build()
    assert new.cpu_int4_direct_dequant is True
    assert new.weight_halves is None

    # and both produce the same answer, which is what makes the A/B meaningful
    x = torch.randn(8, k, dtype=torch.bfloat16)
    a = torch.ops.vllm.cpu_int4_linear(
        x, old.weight_int4pack, old.weight_scales_and_zeros,
        old.weight_halves, old.weight_group_scale, old.weight_group_zero,
        group, 2, False)
    b = torch.ops.vllm.cpu_int4_linear(
        x, new.weight_int4pack, new.weight_scales_and_zeros,
        None, None, None, group, 2, True)
    assert torch.equal(a, b)


def test_the_path_choice_lives_in_the_config_so_it_reaches_the_cache_key():
    """The two prefill paths build different graphs -- one passes
    weight_halves into the custom op, the other passes None -- so they must not
    share a compiled artifact.

    An env-var-only switch does not reach vLLM's compile-cache key, and a warm
    cache then loads the wrong graph: three passes of a benchmark failed with
    `KeyError: 'weight_halves'` inside the AOT-loaded forward, after the first
    pass had saved an artifact for the other path. `direct_dequant` is a config
    field so it is hashed with the rest of the model config.
    """
    from vllm_omni.model_executor.layers.quantization.cpu_int4 import CPUInt4Config

    on = CPUInt4Config.from_config({"group_size": 128, "direct_dequant": True})
    off = CPUInt4Config.from_config({"group_size": 128, "direct_dequant": False})
    assert on.direct_dequant is True
    assert off.direct_dequant is False
    # and it shows up in the repr, which is what lands in logs and configs
    assert "direct_dequant=True" in repr(on)
    assert "direct_dequant=False" in repr(off)
    # default stays on: this path exists for memory-constrained devices
    assert CPUInt4Config.from_config({"group_size": 128}).direct_dequant is True


def test_env_override_still_wins_for_experiments(monkeypatch):
    from vllm_omni.model_executor.layers.quantization.cpu_int4 import (
        direct_dequant_enabled,
    )

    monkeypatch.delenv("VLLM_OMNI_CPU_INT4_DIRECT_DEQUANT", raising=False)
    assert direct_dequant_enabled(True) is True
    assert direct_dequant_enabled(False) is False

    monkeypatch.setenv("VLLM_OMNI_CPU_INT4_DIRECT_DEQUANT", "0")
    assert direct_dequant_enabled(True) is False
    monkeypatch.setenv("VLLM_OMNI_CPU_INT4_DIRECT_DEQUANT", "1")
    assert direct_dequant_enabled(False) is True


def test_blocked_repacking_matches_packing_the_whole_matrix():
    """unpack_nibbles expands 4-bit codes to int32 -- 8x the packed form -- and
    doing that for a whole matrix makes the transient the largest thing in the
    process: 1024 MiB for the 131072-row tied embedding. n-blocks of the packed
    layout are independent, so it can be chunked; the result must be identical
    or the weights are silently wrong."""
    from vllm_omni.model_executor.layers.quantization.cpu_int4 import (
        INNER_K_TILES,
        _pack_blocked,
        quantize_groupwise,
        unpack_nibbles,
    )

    torch.manual_seed(0)
    for n, k in ((1024, 512), (2048, 1024), (640, 256)):
        w = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
        packed, _, _ = quantize_groupwise(w, 128)
        whole = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            unpack_nibbles(packed), INNER_K_TILES
        )
        assert torch.equal(_pack_blocked(packed), whole), f"N={n} K={k}"


def test_blocked_repacking_handles_a_ragged_last_chunk():
    """N not a multiple of the chunk size must still round-trip."""
    from vllm_omni.model_executor.layers.quantization.cpu_int4 import (
        INNER_K_TILES,
        _pack_blocked,
        quantize_groupwise,
        unpack_nibbles,
    )

    n, k = 576, 256           # 576 = 512 + 64, one full chunk plus a short one
    w = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    packed, _, _ = quantize_groupwise(w, 128)
    whole = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
        unpack_nibbles(packed), INNER_K_TILES
    )
    assert torch.equal(_pack_blocked(packed), whole)


def test_split_halves_still_round_trips_after_chunking():
    from vllm_omni.model_executor.layers.quantization.cpu_int4 import (
        _dequantize_halves,
        quantize_groupwise,
        to_split_halves,
    )

    n, k, group = 1088, 512, 128          # not a multiple of the chunk size
    w = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    packed, scale, zero = quantize_groupwise(w, group)
    deq = _dequantize_halves(to_split_halves(packed),
                             scale.to(torch.bfloat16), zero.to(torch.bfloat16), group)
    err = (deq.float() - w.float())
    snr = 10 * torch.log10((w.float() ** 2).mean() / err.pow(2).mean())
    assert snr > 20, f"SNR {snr:.1f} dB"
