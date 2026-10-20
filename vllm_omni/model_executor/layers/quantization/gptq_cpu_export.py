# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build GPTQ-format 4-bit tensors for vLLM's CPU W4A8 kernel.

`CPUWNA16LinearKernel` routes a GPTQ checkpoint to `int4_scaled_mm_cpu`, an
AMX W4A8 GEMM, when the activations are bf16 and the checkpoint carries no
activation reordering.  That kernel builds its own zero point -- a constant 8
-- and ignores the checkpoint's `qzeros`, so the grid written here has to be
symmetric about 8:

    w = (code - 8) * scale,   code in [0, 15]

Keeping that assumption in the package, with a test that checks it against the
kernel itself, is the point of this module: it is the one thing that can go
silently wrong -- a checkpoint quantized to a different convention still loads,
still generates, and is simply wrong.
"""

import torch

PACK_FACTOR = 8
# Scale candidates as a fraction of amax/8. 0.875 puts +amax exactly on code
# 15; larger values trade clipping the extreme for a finer step, and which
# wins depends on the group, so it is searched rather than assumed.
DEFAULT_SHRINKS = (1.06, 1.0, 0.95, 0.9, 0.875, 0.85, 0.8, 0.75)


def pack_int32(codes: torch.Tensor, dim: int) -> torch.Tensor:
    """Pack 4-bit codes into int32 words, eight per word, along ``dim``."""
    codes = codes.to(torch.int32)
    if dim == 0:
        out = torch.zeros(
            codes.shape[0] // PACK_FACTOR, codes.shape[1], dtype=torch.int32
        )
        for i in range(PACK_FACTOR):
            out |= (codes[i::PACK_FACTOR] & 0xF) << (4 * i)
    else:
        out = torch.zeros(
            codes.shape[0], codes.shape[1] // PACK_FACTOR, dtype=torch.int32
        )
        for i in range(PACK_FACTOR):
            out |= (codes[:, i::PACK_FACTOR] & 0xF) << (4 * i)
    return out


def quantize_symmetric(w: torch.Tensor, group: int, search: bool = True):
    """[N, K] float -> (codes [N, K] in [0, 15], scales [N, K // group])."""
    n, k = w.shape
    if k % group:
        raise ValueError(f"input dim {k} is not a multiple of group size {group}")
    wf = w.float().reshape(n, k // group, group)
    amax = wf.abs().amax(-1).clamp(min=1e-9)

    best_err = best = None
    for f in (DEFAULT_SHRINKS if search else (0.875,)):
        scale = amax * f / 8.0
        codes = (wf / scale.unsqueeze(-1)).round().add_(8).clamp_(0, 15)
        err = ((codes - 8.0) * scale.unsqueeze(-1) - wf).pow_(2).sum(-1)
        if best_err is None:
            best_err, best = err, (codes, scale)
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best = (
                torch.where(take.unsqueeze(-1), codes, best[0]),
                torch.where(take, scale, best[1]),
            )
    codes, scale = best
    return codes.reshape(n, k).to(torch.int32), scale


def dequantize_symmetric(codes: torch.Tensor, scale: torch.Tensor, group: int):
    n, k = codes.shape
    c = codes.float().reshape(n, k // group, group)
    return ((c - 8.0) * scale.unsqueeze(-1)).reshape(n, k)


def gptq_tensors(w: torch.Tensor, group: int, search: bool = True):
    """[N, K] float -> the four tensors AutoGPTQLinearMethod expects.

    Returns ``(qweight [K//8, N], qzeros [K//g, N//8], scales [K//g, N],
    g_idx [K], snr_db)``.
    """
    n, k = w.shape
    codes, scale = quantize_symmetric(w, group, search)
    deq = dequantize_symmetric(codes, scale, group)
    wf = w.float()
    snr = float(20 * torch.log10(wf.norm() / (wf - deq).norm().clamp(min=1e-12)))

    qweight = pack_int32(codes.t().contiguous(), dim=0)
    scales = scale.t().contiguous().to(torch.bfloat16)
    qzeros = pack_int32(torch.full((k // group, n), 8, dtype=torch.int32), dim=1)
    g_idx = torch.arange(k, dtype=torch.int32) // group
    return qweight, qzeros, scales, g_idx, snr
