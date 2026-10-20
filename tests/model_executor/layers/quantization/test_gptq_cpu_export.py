# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The GPTQ W4A8 export convention, checked against the kernel that reads it."""

import pytest
import torch

from vllm_omni.model_executor.layers.quantization.gptq_cpu_export import (
    PACK_FACTOR,
    dequantize_symmetric,
    gptq_tensors,
    pack_int32,
    quantize_symmetric,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_pack_round_trips_through_vllms_unpacker():
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        unpack_quantized_values_into_int32,
    )
    from vllm.scalar_type import scalar_types

    codes = torch.randint(0, 16, (64, 32), dtype=torch.int32)
    packed = pack_int32(codes, dim=0)
    assert packed.shape == (64 // PACK_FACTOR, 32)
    back = unpack_quantized_values_into_int32(packed, scalar_types.uint4, 0)
    assert torch.equal(back, codes)

    packed1 = pack_int32(codes, dim=1)
    assert packed1.shape == (64, 32 // PACK_FACTOR)
    back1 = unpack_quantized_values_into_int32(packed1, scalar_types.uint4, 1)
    assert torch.equal(back1, codes)


def test_codes_stay_in_range_and_shapes_match_the_loader():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    qweight, qzeros, scales, g_idx, snr = gptq_tensors(w, 64)
    assert qweight.shape == (256 // 8, 64) and qweight.dtype == torch.int32
    assert qzeros.shape == (256 // 64, 64 // 8)
    assert scales.shape == (256 // 64, 64) and scales.dtype == torch.bfloat16
    assert g_idx.shape == (256,) and g_idx.max().item() == 3
    assert snr > 18.0


def test_scale_search_beats_the_fixed_grid():
    torch.manual_seed(1)
    w = torch.randn(128, 256)
    w[:, 0] *= 40.0  # one outlier per row is what a fixed scale handles worst
    err = {}
    for search in (False, True):
        codes, scale = quantize_symmetric(w, 64, search=search)
        deq = dequantize_symmetric(codes, scale, 64)
        err[search] = (w - deq).norm().item()
    assert err[True] < err[False]


@pytest.mark.skipif(
    not torch.cpu._is_amx_tile_supported(), reason="W4A8 kernel needs AMX tiles"
)
def test_kernel_reads_back_what_we_wrote():
    """End-to-end against `int4_scaled_mm_cpu`, through the exact repack
    `CPUWNA16LinearKernel._process_gptq_weights_w4a8` performs.

    This is the guard that matters: a checkpoint written to a different
    zero-point convention still loads and still generates text, just wrong
    text, so nothing short of running the kernel catches it.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32,
        unpack_quantized_values_into_int32,
    )
    from vllm.scalar_type import scalar_types

    wt = scalar_types.uint4
    torch.manual_seed(0)
    n, k, group = 64, 256, 64
    w = torch.randn(n, k)
    qweight, qzeros, scales, _, _ = gptq_tensors(w, group)
    codes, scale = quantize_symmetric(w, group)
    deq = dequantize_symmetric(codes, scale, group)

    weight = unpack_quantized_values_into_int32(qweight, wt, 0)
    weight = weight.view(k, n // 8, 8)[:, :, (0, 2, 4, 6, 1, 3, 5, 7)].reshape(k, n)
    weight = pack_quantized_values_into_int32(weight, wt, 1).contiguous()
    fake_zp = torch.ones(k // group, n // 8, dtype=torch.int32) * -2004318072
    zp = unpack_quantized_values_into_int32(fake_zp, wt, 1)
    zp = zp.view(k // group, n // 8, 8)[:, :, (0, 2, 4, 6, 1, 3, 5, 7)].reshape(
        k // group, n
    )
    zp = pack_quantized_values_into_int32(zp, wt, 1).contiguous()
    bw, bzp, bs = ops.convert_weight_packed_scale_zp(
        weight, zp, scales, ops.CPUQuantAlgo.AWQ
    )

    x = torch.randn(4, k, dtype=torch.bfloat16)
    got = ops.int4_scaled_mm_cpu(x=x, w=bw, w_zeros=bzp, w_scales=bs, bias=None).float()
    exp = x.float() @ deq.t()
    snr = 20 * torch.log10(exp.norm() / (got - exp).norm())
    # The kernel quantizes activations to int8, so ~40 dB is the ceiling here;
    # a convention mismatch lands far below zero.
    assert snr > 30.0, f"kernel disagrees with our grid: {snr:.1f} dB"
