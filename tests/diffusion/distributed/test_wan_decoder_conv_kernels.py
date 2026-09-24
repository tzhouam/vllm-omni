# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Triton 96-channel Wan decoder conv: installer routing (CPU) and closeness to cuDNN (CUDA)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders.wan_decoder_conv_kernels import (
    can_use_triton_conv3d,
    install_triton_conv3d_96,
    pack_conv3d_weight,
    triton_conv3d_3x3x3,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
def test_installer_routes_only_96_to_96_3x3x3_and_falls_back_off_cuda():
    root = nn.Sequential(nn.Conv3d(96, 96, 3), nn.Conv3d(96, 3, 3), nn.Conv3d(192, 192, 3), nn.Conv3d(96, 96, 1))
    assert install_triton_conv3d_96(root) == 1
    x = torch.randn(1, 96, 4, 6, 6)
    conv = root[0]
    assert not can_use_triton_conv3d(x, conv.weight, conv)
    torch.testing.assert_close(conv(x), F.conv3d(x, conv.weight, conv.bias), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("narrow", [False, True])
def test_kernel_is_within_one_bf16_ulp_of_fp64_and_no_worse_than_cudnn(narrow: bool):
    torch.manual_seed(0)
    cl = torch.channels_last_3d
    x = torch.randn(1, 96, 6, 34, 42, device="cuda", dtype=torch.bfloat16).contiguous(memory_format=cl)
    if narrow:  # the spatial shard's trim shift hands the conv a narrowed view
        x = x.narrow(4, 1, 40)
    w = (torch.randn(96, 96, 3, 3, 3, device="cuda") * 0.02).to(torch.bfloat16).contiguous(memory_format=cl)
    b = torch.randn(96, device="cuda", dtype=torch.bfloat16)
    exact = F.conv3d(x.double(), w.double(), b.double())
    cudnn = F.conv3d(x, w, b)
    got = triton_conv3d_3x3x3(x, pack_conv3d_weight(w), b)
    assert got.shape == cudnn.shape and got.is_contiguous(memory_format=cl)
    # Not bit-identical to cuDNN (K reduction order), so compare both against fp64 ground truth.
    err = (got.double() - exact).abs()
    assert bool((err <= torch.finfo(torch.bfloat16).eps * exact.abs().clamp(min=1.0)).all())
    assert err.max() <= (cudnn.double() - exact).abs().max()
