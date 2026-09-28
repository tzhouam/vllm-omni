# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Graph-safe text-only prompt encoding for TaoMate-H3 prompt updates.

A prompt update re-encodes the prompt on the workers with the MiniMax-H3
Qwen3-VL text encoder. For a few dozen tokens the 50 retained decoder layers
are launch-bound (about 0.15-0.27 s per prompt, measured locally, for a few
milliseconds of device work), and the encode runs inside the streaming step
loop, so every update costs the stream that much wall time.

``GraphSafeTextEncode`` re-states the encoder's text-only forward without
host synchronizations: the vocab-parallel embedding's boolean-mask writes
become ``torch.where``/``masked_fill``, and the mRoPE positions of a text-only
prompt are ``arange`` on three axes (what ``_get_rope_index`` yields when
every token is text). Everything else is the encoder's own modules, so the
result matches the eager encode exactly. The wrapper is captured by
``GraphedForward`` per token count.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def cudnn_sdp_enabled(enabled: bool = True) -> Iterator[None]:
    """Mirror ``MiniMaxH3Qwen3VLEncoder.encode_ids``, which selects cuDNN SDPA for the encode."""
    if not torch.cuda.is_available():
        yield
        return
    previous = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.enable_cudnn_sdp(previous)


def embed_tokens_graph_safe(embed: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """The vocab-parallel embedding of ``ids`` ([1, L]) without boolean-mask indexing."""
    tp_size = int(getattr(embed, "_tp_size", 1))
    if tp_size == 1:
        return F.embedding(ids, embed.weight)
    per_partition = int(embed.num_embeddings_per_partition)
    start = int(embed._tp_rank) * per_partition
    end = start + per_partition
    outside = (ids < start) | (ids >= end)
    local = torch.where(outside, torch.zeros_like(ids), ids - start)
    output = F.embedding(local, embed.weight)
    output = output.masked_fill(outside.unsqueeze(-1), 0.0)
    return embed.group.all_reduce(output)


def text_only_positions(ids: torch.Tensor) -> torch.Tensor:
    """mRoPE positions [3, 1, L] of a prompt without image or video tokens."""
    length = int(ids.shape[-1])
    return torch.arange(length, device=ids.device, dtype=ids.dtype).view(1, 1, -1).expand(3, 1, -1)


class GraphSafeTextEncode(nn.Module):
    """``encoder._encode`` for text-only prompts, written for CUDA-graph capture."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, *, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        text_model = self.encoder.text_model
        ids = input_ids.view(1, -1).to(torch.long)
        inputs_embeds = embed_tokens_graph_safe(text_model.embed_tokens, ids)
        hidden = text_model(inputs_embeds, text_only_positions(ids))[0]
        return hidden.to(torch.bfloat16)


__all__ = ["GraphSafeTextEncode", "cudnn_sdp_enabled", "embed_tokens_graph_safe", "text_only_positions"]
