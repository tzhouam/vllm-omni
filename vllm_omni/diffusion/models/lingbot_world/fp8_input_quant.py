# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One-pass dynamic per-token FP8 input quantization for LingBot-World's FP8 linears.

Under regional ``torch.compile`` vLLM's ``QuantFP8.forward_native`` is fused by inductor with whatever produced
its input. The rows here are 5120 or 13824 wide, above inductor's 1024-element persistent-reduction limit, so
the generated kernels loop over every row two or three times (recomputing tanh-GELU, dragging the unused
argmax of ``max(dim=-1)`` through the reduction) and reach about a fifth of HBM bandwidth.

:func:`quant_fp8_per_token` keeps each row in registers and reads it once. Its arithmetic is op-for-op the
inductor expression it replaces, so the FP8 payload and the FP32 scales are bit-identical:

    v     = x                                                           (plain)
    v     = (0.5 * x) * (tanh(0.7978845608028654 * (x + 0.044715 * x*x*x)) + 1)   (GELU, fp32, not rounded)
    scale = max(amax(|v|) / 448, 1 / (448 * 512))
    q     = clamp(v * (1 / scale), -448, 448) -> float8_e4m3fn

:func:`install_one_pass_fp8_input_quant` swaps the kernel's quantizer only on the linears whose producer is an
elementwise op inductor cannot fuse into the reduction anyway: the self- and cross-attention output
projections, and the FFN down projection, where the tanh-GELU moves into the quantizer and ``ffn[1]``
becomes an identity. The CUTLASS GEMM, its scales and its bias are untouched.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

try:  # Triton is optional: CPU tests and non-CUDA platforms use the torch reference.
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only without Triton
    _HAS_TRITON = False

_FP8_MAX = 448.0
_FP8_MIN_SCALE = 1.0 / (_FP8_MAX * 512.0)

if _HAS_TRITON:

    @triton.jit
    def _quant_fp8_row_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        n_cols,
        stride_x,
        stride_q,
        fp8_max,
        gelu: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, block_n)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        if gelu:
            v = (x * 0.5) * (libdevice.tanh((x + ((x * x) * x) * 0.044715) * 0.7978845608028654) + 1.0)
        else:
            v = x
        amax = tl.max(tl.where(mask, tl.abs(v), 0.0), axis=0)
        # True division by a runtime fp32 448, as the inductor kernel does (its divisor is an fp64 graph
        # input cast to fp32); a folded constant or a reciprocal multiply can round differently.
        scale = tl.maximum(amax / fp8_max, 4.359654017857143e-06)
        q = tl.minimum(tl.maximum(v * (1.0 / scale), -448.0), 448.0)
        tl.store(q_ptr + row * stride_q + cols, q.to(tl.float8e4nv), mask=mask)
        tl.store(s_ptr + row, scale)


def quant_fp8_per_token_reference(x: torch.Tensor, gelu: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference of the kernel (the served ``QuantFP8.forward_native`` expression, GELU in fp32)."""
    v = x.float()
    if gelu:
        v = (v * 0.5) * (torch.tanh((v + ((v * v) * v) * 0.044715) * 0.7978845608028654) + 1.0)
    scale = (v.abs().amax(dim=-1, keepdim=True) / _FP8_MAX).clamp(min=_FP8_MIN_SCALE)
    q = (v * scale.reciprocal()).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


@torch.library.custom_op("vllm_omni::lingbot_quant_fp8_per_token", mutates_args=())
def quant_fp8_per_token(x: torch.Tensor, gelu: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token FP8 quantization of a 2-D ``[M, N]`` tensor, optionally of ``gelu_tanh(x)``."""
    if x.dim() != 2:
        raise ValueError(f"expected a 2-D input, got shape {tuple(x.shape)}")
    if not (_HAS_TRITON and x.is_cuda):
        return quant_fp8_per_token_reference(x, gelu)
    if x.stride(-1) != 1:
        x = x.contiguous()
    rows, cols = x.shape
    q = torch.empty((rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    if rows:
        block = triton.next_power_of_2(cols)
        # Measured on H200 at 2340 rows: 4 warps for 5120-wide plain rows, 32 for 13824-wide GELU rows.
        num_warps = 32 if gelu and block >= 16384 else (16 if block >= 16384 else 4)
        _quant_fp8_row_kernel[(rows,)](
            x, q, scale, cols, x.stride(0), q.stride(0), _FP8_MAX, gelu=gelu, block_n=block, num_warps=num_warps
        )
    return q, scale


@quant_fp8_per_token.register_fake
def _(x: torch.Tensor, gelu: bool) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    return (
        x.new_empty((rows, cols), dtype=torch.float8_e4m3fn),
        x.new_empty((rows, 1), dtype=torch.float32),
    )


class OnePassFP8InputQuant:
    """Drop-in for a CUTLASS FP8 linear kernel's dynamic per-token ``QuantFP8`` (``quant_fp8(x_2d, None, None)``)."""

    def __init__(self, gelu: bool) -> None:
        self.gelu = gelu

    def __call__(
        self, x: torch.Tensor, scale: torch.Tensor | None = None, scale_ub: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scale is not None or scale_ub is not None:
            raise RuntimeError("one-pass FP8 input quant replaces dynamic per-token quantization only")
        return torch.ops.vllm_omni.lingbot_quant_fp8_per_token(x, self.gelu)


def _dynamic_per_token_cutlass_kernel(layer: nn.Module):
    """The layer's CUTLASS FP8 kernel if it quantizes its input dynamically per token, else ``None``."""
    kernel = getattr(getattr(layer, "quant_method", None), "fp8_linear", None)
    if kernel is None or type(kernel).__name__ != "CutlassFP8ScaledMMLinearKernel":
        return None
    quant = getattr(kernel, "quant_fp8", None)
    if quant is None or getattr(quant, "static", True) or getattr(quant, "num_token_padding", None) is not None:
        return None
    if not getattr(getattr(quant, "group_shape", None), "is_per_token", lambda: False)():
        return None
    return kernel


def install_one_pass_fp8_input_quant(transformer: nn.Module) -> dict[str, int]:
    """Swap the input quantizer of each block's attention output projections and FFN down projection."""
    counts = {"attn_out": 0, "ffn_down_gelu": 0, "skipped": 0}
    for block in getattr(transformer, "blocks", ()):
        for attn in (block.self_attn, block.cross_attn):
            kernel = _dynamic_per_token_cutlass_kernel(attn.o)
            if kernel is None:
                counts["skipped"] += 1
                continue
            kernel.quant_fp8 = OnePassFP8InputQuant(gelu=False)
            counts["attn_out"] += 1
        ffn = block.ffn
        activation = ffn[1]
        kernel = _dynamic_per_token_cutlass_kernel(ffn[2])
        if kernel is None or not (isinstance(activation, nn.GELU) and activation.approximate == "tanh"):
            counts["skipped"] += 1
            continue
        kernel.quant_fp8 = OnePassFP8InputQuant(gelu=True)
        ffn[1] = nn.Identity()
        counts["ffn_down_gelu"] += 1
    return counts
