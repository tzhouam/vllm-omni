# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One-pass FP8 input quant: installer wiring (CPU) and bitwise parity with the served inductor kernels (CUDA)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.models.lingbot_world.fp8_input_quant import (
    OnePassFP8InputQuant,
    install_one_pass_fp8_input_quant,
    quant_fp8_per_token_reference,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _served_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM ``QuantFP8.forward_native``, dynamic per token, exactly as the served graph traces it."""
    x_max, _ = x.abs().max(dim=-1)
    scale = (x_max.unsqueeze(-1).to(torch.float32) / _FP8_MAX).clamp(min=1.0 / (_FP8_MAX * 512.0))
    out = (x.to(torch.float32) * scale.reciprocal()).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return out, scale


class _Kernel:
    def __init__(self, static: bool = False) -> None:
        self.quant_fp8 = SimpleNamespace(
            static=static, num_token_padding=None, group_shape=SimpleNamespace(is_per_token=lambda: True)
        )


_Kernel.__name__ = "CutlassFP8ScaledMMLinearKernel"


def _linear(kernel: _Kernel | None) -> nn.Module:
    layer = nn.Linear(4, 4)
    layer.quant_method = SimpleNamespace(fp8_linear=kernel)
    return layer


def _block(*, static_ffn: bool = False) -> nn.Module:
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.o = _linear(_Kernel())
    block.cross_attn = nn.Module()
    block.cross_attn.o = _linear(_Kernel())
    block.ffn = nn.Sequential(_linear(_Kernel()), nn.GELU(approximate="tanh"), _linear(_Kernel(static=static_ffn)))
    return block


@pytest.mark.cpu
def test_installer_swaps_only_dynamic_attention_out_and_ffn_down():
    transformer = nn.Module()
    transformer.blocks = nn.ModuleList([_block(), _block(static_ffn=True)])
    counts = install_one_pass_fp8_input_quant(transformer)
    assert counts == {"attn_out": 4, "ffn_down_gelu": 1, "skipped": 1}
    first, second = transformer.blocks
    assert first.ffn[2].quant_method.fp8_linear.quant_fp8.gelu is True
    assert isinstance(first.ffn[1], nn.Identity)
    assert not isinstance(first.ffn[0].quant_method.fp8_linear.quant_fp8, OnePassFP8InputQuant)
    assert isinstance(second.ffn[1], nn.GELU)
    assert not isinstance(second.ffn[2].quant_method.fp8_linear.quant_fp8, OnePassFP8InputQuant)


@pytest.mark.cpu
@pytest.mark.parametrize("gelu", [False, True])
def test_reference_matches_served_expression_on_cpu(gelu: bool):
    x = (torch.randn(5, 64) * 4).to(torch.bfloat16)
    served_input = F.gelu(x.float(), approximate="tanh") if gelu else x
    q_ref, s_ref = quant_fp8_per_token_reference(x, gelu)
    q, s = _served_quant(served_input)
    assert torch.equal(s_ref, s)
    assert torch.equal(q_ref.view(torch.uint8), q.view(torch.uint8))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(("gelu", "cols"), [(False, 5120), (True, 13824)])
def test_kernel_is_bit_identical_to_compiled_served_kernel(gelu: bool, cols: int):
    """The served graph is regionally torch.compile'd, so the reference is inductor's fused kernel."""

    def served(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _served_quant(F.gelu(x, approximate="tanh") if gelu else x)

    compiled = torch.compile(served, dynamic=True)
    generator = torch.Generator(device="cuda").manual_seed(0)
    base = torch.randn(257, cols, device="cuda", generator=generator)
    cases = [
        base * 2,
        base * 30,
        base * 1e-6,
        torch.zeros_like(base),
        base.index_fill(1, torch.tensor([7], device="cuda"), 5e3),
    ]
    for case in cases:
        x = case.to(torch.bfloat16)
        q, s = torch.ops.vllm_omni.lingbot_quant_fp8_per_token(x, gelu)
        q_ref, s_ref = compiled(x)
        assert torch.equal(s, s_ref)
        assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
