# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests of the graph-safe text-only prompt encode."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.taomate_h3.text_encode import (
    GraphSafeTextEncode,
    embed_tokens_graph_safe,
    text_only_positions,
)


class _Group:
    def __init__(self) -> None:
        self.calls = 0

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return tensor


def _upstream_embedding(embed: SimpleNamespace, ids: torch.Tensor) -> torch.Tensor:
    """The encoder's vocab-parallel embedding with its boolean-mask writes (tensor-parallel path)."""
    start_idx = embed._tp_rank * embed.num_embeddings_per_partition
    end_idx = start_idx + embed.num_embeddings_per_partition
    input_mask = (ids < start_idx) | (ids >= end_idx)
    masked_input = ids.clone() - start_idx
    masked_input[input_mask] = 0
    output = F.embedding(masked_input, embed.weight)
    output[input_mask, :] = 0.0
    embed.group.all_reduce(output)
    return output


def test_graph_safe_embedding_matches_the_upstream_mask_writes() -> None:
    torch.manual_seed(0)
    vocab, dim = 16, 4
    for tp_rank in (0, 1):
        embed = SimpleNamespace(
            _tp_size=2,
            _tp_rank=tp_rank,
            num_embeddings_per_partition=vocab // 2,
            weight=torch.randn(vocab // 2, dim),
            group=_Group(),
        )
        ids = torch.tensor([[0, 3, 7, 8, 12, 15, 9]])
        assert torch.equal(embed_tokens_graph_safe(embed, ids), _upstream_embedding(embed, ids))
        assert embed.group.calls == 2
    single = SimpleNamespace(_tp_size=1, weight=torch.randn(vocab, dim))
    assert torch.equal(embed_tokens_graph_safe(single, ids), F.embedding(ids, single.weight))


def test_text_only_positions_are_arange_on_three_axes() -> None:
    ids = torch.zeros(1, 9, dtype=torch.long)
    positions = text_only_positions(ids)
    assert tuple(positions.shape) == (3, 1, 9) and positions.dtype == torch.long
    for axis in range(3):
        assert torch.equal(positions[axis, 0], torch.arange(9))


def test_graph_safe_text_encode_runs_the_text_model_on_the_embedding() -> None:
    seen: dict[str, torch.Tensor] = {}

    def text_model(inputs_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        seen["embeds"] = inputs_embeds
        seen["positions"] = positions
        return inputs_embeds * 2.0

    text_model.embed_tokens = SimpleNamespace(_tp_size=1, weight=torch.arange(8.0).view(4, 2))  # type: ignore[attr-defined]
    module = GraphSafeTextEncode(SimpleNamespace(text_model=text_model))
    hidden = module(input_ids=torch.tensor([3, 1, 0]))
    assert hidden.dtype == torch.bfloat16 and tuple(hidden.shape) == (3, 2)
    assert torch.equal(hidden.float(), torch.tensor([[12.0, 14.0], [4.0, 6.0], [0.0, 2.0]]))
    assert tuple(seen["positions"].shape) == (3, 1, 3)
