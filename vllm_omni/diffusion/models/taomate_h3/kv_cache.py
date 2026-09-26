# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Persistent clean audio/video KV cache for TaoMate-H3 streaming.

TaoMate-H3 conditions every new chunk on the *clean* (sigma = 0) keys and
values of earlier chunks. After the three noisy denoise forwards of a chunk, a
fourth forward at sigma = 0 recomputes the chunk's K/V from its clean latents
and appends them to this cache. Retention keeps the first chunk's video rows
as a permanent sink plus the two most recent complete audio/video chunks; the
audio history is dropped every twelve requests.

The cache is model-owned and dense: it stores, per transformer layer, the K
and V rows *after* the Ulysses all-to-all, so each sequence-parallel rank keeps
the full row history for its own head shard (56 / ulysses_degree heads).
Chunk sizes are irregular (12/10/10/5 video latents plus a variable number of
audio latents) and the sink is video-only, which is why this does not use the
runner's paged block pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

VIDEO_TOKEN_TAG = 0
TEXT_TOKEN_TAG = 1
AUDIO_TOKEN_TAG = 2


@dataclass(frozen=True)
class KVContract:
    """Shape contract for the rank-local attention shard."""

    num_layers: int
    local_heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(f"blocks.{index}.attn" for index in range(self.num_layers))

    def bytes_for_rows(self, rows: int) -> int:
        """Bytes of K plus V for ``rows`` rows across all layers."""
        element = torch.empty((), dtype=self.dtype).element_size()
        return 2 * self.num_layers * rows * self.local_heads * self.head_dim * element


@dataclass(frozen=True)
class AVKV:
    """Dense clean audio/video keys and values for one attention layer."""

    key: torch.Tensor
    value: torch.Tensor


def _validate_av_pair(pair: AVKV, contract: KVContract, *, layer_name: str) -> None:
    expected_tail = (contract.local_heads, contract.head_dim)
    if pair.key.ndim != 3 or tuple(pair.key.shape[1:]) != expected_tail:
        raise RuntimeError(
            f"{layer_name}: expected key shape [tokens, {expected_tail[0]}, {expected_tail[1]}], "
            f"got {tuple(pair.key.shape)}"
        )
    if pair.value.shape != pair.key.shape:
        raise RuntimeError(f"{layer_name}: key/value shape mismatch")
    if pair.key.dtype != contract.dtype or pair.value.dtype != contract.dtype:
        raise RuntimeError(f"{layer_name}: persistent KV must be {contract.dtype}")


class CleanAVKVCache:
    """Transactional persistent cache holding clean media KV only.

    A clean commit stages one chunk layer by layer (``begin_clean_commit`` ->
    ``stage`` x layers -> ``commit``); ``rollback`` discards a partial stage.
    ``retain_sink_and_recent_commits`` applies TaoMate's retention policy after
    each commit and ``drop_audio_history`` implements the periodic audio reset.
    """

    def __init__(self, contract: KVContract) -> None:
        self.contract = contract
        self._layer_names = set(contract.layer_names)
        self._history: dict[str, AVKV] = {}
        self._staged: dict[str, AVKV] = {}
        self._staged_block: int | None = None
        self._staged_tags: tuple[int, ...] | None = None
        self._staged_indices: torch.Tensor | None = None
        self._block_index = 0
        self._commit_token_counts: list[int] = []
        self._commit_token_tags: list[tuple[int, ...]] = []

    @property
    def committed_blocks(self) -> int:
        return self._block_index

    @property
    def history_tokens(self) -> int:
        if not self._history:
            return 0
        return int(next(iter(self._history.values())).key.shape[0])

    @property
    def history_audio_tokens(self) -> int:
        return sum(tag == AUDIO_TOKEN_TAG for tags in self._commit_token_tags for tag in tags)

    @property
    def history_video_tokens(self) -> int:
        return sum(tag == VIDEO_TOKEN_TAG for tags in self._commit_token_tags for tag in tags)

    @property
    def clean_commit_active(self) -> bool:
        return self._staged_block is not None

    def history(self, layer_name: str) -> AVKV | None:
        self._validate_layer_name(layer_name)
        return self._history.get(layer_name)

    def begin_clean_commit(self, block_index: int) -> None:
        if self.clean_commit_active:
            raise RuntimeError("a clean KV commit is already active")
        if block_index != self._block_index:
            raise RuntimeError(f"expected clean chunk {self._block_index}, got {block_index}")
        self._staged_block = block_index
        self._staged.clear()
        self._staged_tags = None
        self._staged_indices = None

    def stage(
        self,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        token_tags: torch.Tensor,
        commit_mask: torch.Tensor,
    ) -> None:
        """Stage the clean audio/video rows produced by one transformer layer.

        ``key``/``value`` are the full post-all-to-all rows of the current
        chunk sequence ``[seq_len, local_heads, head_dim]``; ``token_tags`` and
        ``commit_mask`` are the full-sequence modality tags and the mask of
        rows that enter the persistent history (audio and video targets).
        """
        if not self.clean_commit_active:
            raise RuntimeError("begin_clean_commit() must be called before stage()")
        self._validate_layer_name(layer_name)
        if layer_name in self._staged:
            raise RuntimeError(f"{layer_name}: KV was staged more than once")
        if key.shape != value.shape or key.ndim != 3:
            raise RuntimeError(f"{layer_name}: key/value must have matching rank-3 shapes")
        if token_tags.ndim != 1 or commit_mask.ndim != 1:
            raise RuntimeError("token_tags and commit_mask must be rank-1")
        if token_tags.shape[0] != key.shape[0] or commit_mask.shape[0] != key.shape[0]:
            raise RuntimeError(f"{layer_name}: packed metadata length does not match KV rows")
        if self._staged_indices is None:
            indices = torch.nonzero(commit_mask.to(torch.bool), as_tuple=False).flatten()
            if indices.numel() == 0:
                raise RuntimeError("clean commit did not contain audio/video rows")
            selected_tags = token_tags.index_select(0, indices)
            tags = tuple(int(tag) for tag in selected_tags.detach().cpu().tolist())
            if frozenset(tags) != frozenset((VIDEO_TOKEN_TAG, AUDIO_TOKEN_TAG)):
                raise RuntimeError("each clean chunk must contain both video and audio KV")
            self._staged_indices = indices
            self._staged_tags = tags
        indices = self._staged_indices
        assert indices is not None
        pair = AVKV(
            key=key.index_select(0, indices).detach().to(self.contract.dtype),
            value=value.index_select(0, indices).detach().to(self.contract.dtype),
        )
        _validate_av_pair(pair, self.contract, layer_name=layer_name)
        self._staged[layer_name] = pair

    def commit(self) -> None:
        """Atomically append all staged layers to the persistent history."""
        if not self.clean_commit_active:
            raise RuntimeError("no clean KV commit is active")
        missing = [name for name in self.contract.layer_names if name not in self._staged]
        if missing:
            raise RuntimeError(f"clean KV commit is missing {len(missing)} transformer layers")
        assert self._staged_tags is not None
        token_count = len(self._staged_tags)
        # One layer at a time, so old and new histories are never both resident
        # for every layer at the steady-state memory peak.
        for layer_name in self.contract.layer_names:
            current = self._staged.pop(layer_name)
            previous = self._history.pop(layer_name, None)
            if previous is None:
                combined = current
            else:
                combined = AVKV(
                    key=torch.cat((previous.key, current.key), dim=0),
                    value=torch.cat((previous.value, current.value), dim=0),
                )
            self._history[layer_name] = combined
        self._commit_token_counts.append(token_count)
        self._commit_token_tags.append(self._staged_tags)
        self._block_index += 1
        self._clear_staging()

    def retain_sink_and_recent_commits(self) -> None:
        """Keep the first chunk's video sink and the two most recent clean chunks."""
        if self.clean_commit_active:
            raise RuntimeError("cannot trim persistent KV during an active clean commit")
        block_count = len(self._commit_token_counts)
        # As soon as three clean chunks exist, age chunk zero to a video-only
        # sink and keep the two most recent chunks as complete AV recents.
        if block_count <= 2:
            return
        recent_start = max(1, block_count - 2)
        selection = [(0, True)] + [(index, False) for index in range(recent_start, block_count)]
        self._retain_commit_rows(selection)

    def drop_audio_history(self) -> int:
        """Remove audio rows while preserving every retained video row."""
        if self.clean_commit_active:
            raise RuntimeError("cannot drop audio history during an active clean commit")
        removed = self.history_audio_tokens
        if not self._commit_token_counts:
            return 0
        self._retain_commit_rows([(index, True) for index in range(len(self._commit_token_counts))])
        return removed

    def rollback(self) -> None:
        self._clear_staging()

    def clear(self) -> None:
        self._history.clear()
        self._commit_token_counts.clear()
        self._commit_token_tags.clear()
        self._block_index = 0
        self._clear_staging()

    def _retain_commit_rows(self, selection: list[tuple[int, bool]]) -> None:
        offsets = [0]
        for count in self._commit_token_counts:
            offsets.append(offsets[-1] + count)
        selected_rows: list[int] = []
        selected_counts: list[int] = []
        selected_tags: list[tuple[int, ...]] = []
        for block_index, video_only in selection:
            tags = self._commit_token_tags[block_index]
            start = offsets[block_index]
            local_rows = [i for i, tag in enumerate(tags) if not video_only or tag == VIDEO_TOKEN_TAG]
            if not local_rows:
                continue
            selected_rows.extend(start + row for row in local_rows)
            kept_tags = tuple(tags[row] for row in local_rows)
            selected_counts.append(len(kept_tags))
            selected_tags.append(kept_tags)
        if not selected_rows:
            self._history.clear()
            self._commit_token_counts.clear()
            self._commit_token_tags.clear()
            return
        next_history: dict[str, AVKV] = {}
        indices: torch.Tensor | None = None
        for layer_name, pair in self._history.items():
            if indices is None or indices.device != pair.key.device:
                indices = torch.tensor(selected_rows, dtype=torch.long, device=pair.key.device)
            next_history[layer_name] = AVKV(
                key=pair.key.index_select(0, indices),
                value=pair.value.index_select(0, indices),
            )
        self._history = next_history
        self._commit_token_counts = selected_counts
        self._commit_token_tags = selected_tags

    def _clear_staging(self) -> None:
        self._staged.clear()
        self._staged_block = None
        self._staged_tags = None
        self._staged_indices = None

    def _validate_layer_name(self, layer_name: str) -> None:
        if layer_name not in self._layer_names:
            raise KeyError(f"unexpected transformer layer: {layer_name}")


__all__ = ["AUDIO_TOKEN_TAG", "AVKV", "CleanAVKVCache", "KVContract", "TEXT_TOKEN_TAG", "VIDEO_TOKEN_TAG"]
