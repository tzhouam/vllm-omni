# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Triton implicit-GEMM 3x3x3 convolution for the Wan decoder's 96-channel full-resolution stage.

cuDNN has no Hopper kernel for a 96-wide output tile, so the decoder's 96 -> 96 residual convs and the 96 -> 3
``conv_out`` fall back to an sm80 ``256x32`` implicit GEMM at roughly a quarter of the tensor-core peak, while
the 192/384-channel convs of the same decoder get sm90 kernels at about twice that. This kernel is the plain
implicit GEMM: M = output pixels, K = 27 taps x Cin (tap-major, 32 channels per step), N = Cout padded to a
power of two and masked on store, fp32 accumulation, bias in fp32, bf16 result.

It reads a stride-general channels_last_3d input (channel stride 1, any T/H/W strides, so the width- or
height-narrowed views the spatial shard produces need no copy) and writes a dense channels_last_3d output,
the layout cuDNN returns for channels_last operands. Only the 96 -> 96 convs are routed here; for the 96 -> 3
``conv_out`` cuDNN is faster (550 vs 640 us per rank on H200). The input must already carry the causal and spatial
padding (the Wan causal conv wrappers call the conv with ``padding=0``). Values are not bit-identical to
cuDNN: the K reduction order differs (typically <= 1 bf16 ulp), so it is gated with the fused decode level.
"""

from __future__ import annotations

from types import MethodType

import torch

try:  # Triton is optional: CPU tests and non-CUDA platforms keep cuDNN / the CPU conv.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only without Triton
    _HAS_TRITON = False

_CIN = 96

if _HAS_TRITON:

    @triton.jit
    def _conv3d_3x3x3_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        n_t,
        n_h,
        n_w,
        sxt,
        sxh,
        sxw,
        cout,
        cin: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        m = tl.program_id(0) * block_m + tl.arange(0, block_m)
        m_mask = m < n_t * n_h * n_w
        w_i = m % n_w
        h_i = (m // n_w) % n_h
        t_i = m // (n_w * n_h)
        base = t_i.to(tl.int64) * sxt + h_i.to(tl.int64) * sxh + w_i.to(tl.int64) * sxw
        rk = tl.arange(0, 32)
        n = tl.arange(0, block_n)
        n_mask = n < cout
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        for tap in range(27):
            tap_off = (tap // 9) * sxt + ((tap // 3) % 3) * sxh + (tap % 3) * sxw
            for k0 in tl.static_range(0, cin, 32):
                a = tl.load(x_ptr + base[:, None] + tap_off + (k0 + rk)[None, :], mask=m_mask[:, None], other=0.0)
                b = tl.load(w_ptr + (tap * cin + k0 + rk)[:, None] * cout + n[None, :], mask=n_mask[None, :], other=0.0)
                acc = tl.dot(a, b, acc)
        acc += tl.load(b_ptr + n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        out = m.to(tl.int64)[:, None] * cout + n[None, :]
        tl.store(y_ptr + out, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def pack_conv3d_weight(weight: torch.Tensor) -> torch.Tensor:
    """``[Cout, Cin, 3, 3, 3]`` -> ``[27, Cin, Cout]`` contiguous (tap-major K, Cout fastest)."""
    cout, cin = weight.shape[:2]
    return weight.permute(2, 3, 4, 1, 0).reshape(27, cin, cout).contiguous()


def can_use_triton_conv3d(x: torch.Tensor, weight: torch.Tensor, conv: torch.nn.Conv3d) -> bool:
    """Shapes and layouts the kernel supports; everything else stays on cuDNN."""
    return (
        _HAS_TRITON
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.dim() == 5
        and x.shape[0] == 1
        and x.shape[1] == _CIN
        and x.stride(1) == 1
        and tuple(weight.shape[1:]) == (_CIN, 3, 3, 3)
        and weight.shape[0] == _CIN
        and tuple(conv.stride) == (1, 1, 1)
        and tuple(conv.padding) == (0, 0, 0)
        and tuple(conv.dilation) == (1, 1, 1)
        and conv.groups == 1
        and all(s >= 3 for s in x.shape[2:])
    )


def triton_conv3d_3x3x3(x: torch.Tensor, packed_weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """``F.conv3d(x, w, bias)`` for an already-padded channels-last input; see the module docstring."""
    _, cin, tp, hp, wp = x.shape
    cout = packed_weight.shape[2]
    t, h, w = tp - 2, hp - 2, wp - 2
    y = torch.empty((1, cout, t, h, w), device=x.device, dtype=torch.bfloat16, memory_format=torch.channels_last_3d)
    if bias is None:
        bias = torch.zeros(cout, device=x.device, dtype=torch.bfloat16)
    block_n = max(16, triton.next_power_of_2(cout))
    # Measured on H200 for 1 x 96 x 6 x 482 x 418: 256 x 128 tile, 8 warps, 3 stages (989 us vs cuDNN 1591 us).
    block_m, num_warps, num_stages = (256, 8, 3) if block_n >= 128 else (128, 4, 3)
    grid = (triton.cdiv(t * h * w, block_m),)
    _conv3d_3x3x3_kernel[grid](
        x,
        packed_weight,
        bias,
        y,
        t,
        h,
        w,
        x.stride(2),
        x.stride(3),
        x.stride(4),
        cout,
        cin=cin,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


def _triton_conv_forward(self: torch.nn.Conv3d, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
    if can_use_triton_conv3d(x, weight, self):
        packed = self._triton_packed_weight
        if packed is None or self._triton_packed_version != (weight.data_ptr(), weight._version):
            packed = pack_conv3d_weight(weight)
            self._triton_packed_weight = packed
            self._triton_packed_version = (weight.data_ptr(), weight._version)
        return triton_conv3d_3x3x3(x, packed, bias)
    return torch.nn.Conv3d._conv_forward(self, x, weight, bias)


def install_triton_conv3d_96(module: torch.nn.Module) -> int:
    """Route every 96 -> 96 3x3x3 causal conv under ``module`` through :func:`triton_conv3d_3x3x3`."""
    count = 0
    for conv in module.modules():
        if not isinstance(conv, torch.nn.Conv3d) or tuple(conv.weight.shape) != (_CIN, _CIN, 3, 3, 3):
            continue
        conv._triton_packed_weight = None
        conv._triton_packed_version = None
        conv._conv_forward = MethodType(_triton_conv_forward, conv)
        count += 1
    return count
