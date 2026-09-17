# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact fast path for the Wan VAE decoder on a streaming (frame-at-a-time) decode.

A kernel ledger of the LingBot-World served path (4xH200, 480x832, width-sharded bf16 decode) put the
decoder at 91 ms of GPU time per chunk per rank, half of it in ~2,300 launches of glue around the
convolutions: autocast re-casting every conv weight and bias on every call (the autocast weight cache only
serves leaves that require grad, so under ``torch.no_grad`` it never hits), the nearest upsample's fp32
round trip, and a freshly allocated, zero-filled, concatenated input tensor per conv call. None of that
touches a value that reaches the convolution, so it can go without changing a single output bit:

- ``conv_dtype``: the decoder's convolution parameters are cast once to the dtype the decode already runs
  under (``decode_autocast_dtype``), the same cast autocast applied per call. Norm parameters stay in their
  own dtype, so the normalisation arithmetic and its fp32 promotion are untouched.
- ``WanUpsample`` runs its nearest-exact gather on the activation dtype directly instead of via
  ``x.float()`` and ``type_as``: a gather moves values, so the result is identical.
- The spatially sharded conv wrappers keep one input buffer per conv across calls
  (``reuse_input_buffer``), writing only the activation interior and the halo slots into it.

Numerics: bit-identical to the plain path (``tests/diffusion/distributed/test_wan_decoder_fast_path.py``).
Memory: one persistent input buffer per sharded conv (about the size of that conv's input).
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard import WanDistCausalConv3d, WanDistConv2d

logger = init_logger(__name__)

_INSTALLED_ATTR = "_vllm_omni_wan_decoder_fast_path"


def _upsample_forward(self: nn.Upsample, x: torch.Tensor) -> torch.Tensor:
    # ``WanUpsample.forward`` is ``super().forward(x.float()).type_as(x)``; a nearest gather never changes a
    # value, so running it on ``x`` itself is the same tensor without the two casts.
    return nn.Upsample.forward(self, x)


def _is_nearest_upsample(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "WanUpsample"
        and isinstance(module, nn.Upsample)
        and str(getattr(module, "mode", "")) in ("nearest", "nearest-exact")
    )


def install_wan_decoder_fast_path(vae: Any, *, conv_dtype: torch.dtype | None) -> dict[str, int]:
    """Install the exact fast path on ``vae``'s decoder (and ``post_quant_conv``); idempotent.

    ``conv_dtype`` must be the dtype the decode runs under (``decode_autocast_dtype``): with ``None`` the
    parameters are left alone, because casting them without autocast would change the convolution's
    arithmetic rather than remove a cast.
    """
    installed = getattr(vae, _INSTALLED_ATTR, None)
    if installed is not None:
        return installed
    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        raise ValueError("Wan decoder fast path requires a decoder module.")
    if conv_dtype is not None and conv_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"conv_dtype must be float16, bfloat16 or None; got {conv_dtype!r}")
    roots = [decoder]
    post_quant_conv = getattr(vae, "post_quant_conv", None)
    if isinstance(post_quant_conv, nn.Module):
        roots.append(post_quant_conv)
    counts = {"conv_params_cast": 0, "upsamples": 0, "persistent_input_buffers": 0}
    for root in roots:
        for module in root.modules():
            if conv_dtype is not None and isinstance(module, (nn.Conv2d, nn.Conv3d)):
                if any(p.dtype != conv_dtype for p in module.parameters(recurse=False)):
                    module.to(dtype=conv_dtype)
                    counts["conv_params_cast"] += 1
            if _is_nearest_upsample(module) and not getattr(module, "_vllm_omni_fast_path_upsample", False):
                module.forward = MethodType(_upsample_forward, module)
                module._vllm_omni_fast_path_upsample = True
                counts["upsamples"] += 1
            if isinstance(module, (WanDistCausalConv3d, WanDistConv2d)):
                module.reuse_input_buffer = True
                counts["persistent_input_buffers"] += 1
    setattr(vae, _INSTALLED_ATTR, counts)
    logger.info(
        "Installed the exact Wan decoder fast path: %d conv modules cast to %s, %d upsamples, "
        "%d persistent conv input buffers.",
        counts["conv_params_cast"],
        conv_dtype,
        counts["upsamples"],
        counts["persistent_input_buffers"],
    )
    return counts
