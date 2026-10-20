# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Edge scheduling helpers (WP8): playback slack, prefill throttling, no-preemption admission.

Pure functions so the policy can be unit-tested without an engine. They are
consumed by the hardware adaptation layer (``vllm_omni.edge.adapt``) and are the
reference for the measurement-gated scheduler changes described in
``analysis/edge_branch_plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "fp8": 1, "float8_e4m3fn": 1, "auto": 2}


# --------------------------------------------------------------- playback slack
@dataclass(frozen=True)
class ChunkArrival:
    t_ms: float  # arrival time on the consumer clock
    audio_ms: float  # audio duration carried by the chunk


def playback_slack_ms(chunks: list[ChunkArrival], now_ms: float, start_ms: float | None = None) -> float:
    """Audio still buffered ahead of a player that started at ``start_ms``.

    The player starts at the first chunk's arrival unless ``start_ms`` is
    given. Slack < 0 means the player has already underrun.
    """
    if not chunks:
        return 0.0
    start = chunks[0].t_ms if start_ms is None else start_ms
    buffered = sum(c.audio_ms for c in chunks if c.t_ms <= now_ms)
    return buffered - max(0.0, now_ms - start)


def should_throttle_prefills(slack_ms: float, next_chunk_eta_ms: float, safety_margin_ms: float) -> bool:
    """Throttle other requests' prefills when this request's next chunk would land too late.

    ``next_chunk_eta_ms`` is the expected time until the next chunk is
    delivered (steps x step_ms + hop). Throttling is worthwhile only when the
    buffered audio would not cover that wait plus the safety margin.
    """
    return slack_ms - next_chunk_eta_ms < safety_margin_ms


# --------------------------------------------------------------- KV admission
def kv_bytes_per_token(
    num_layers: int, num_kv_heads: int, head_dim: int, dtype: str = "bfloat16", kv_cache_dtype: str = "auto"
) -> int:
    """Bytes of KV cache one token occupies (K and V, all layers)."""
    d = kv_cache_dtype if kv_cache_dtype != "auto" else dtype
    b = _DTYPE_BYTES.get(d, 2)
    return 2 * num_layers * num_kv_heads * head_dim * b


def kv_bytes_for_seqs(max_num_seqs: int, max_model_len: int, bytes_per_token: int, block_size: int = 16) -> int:
    """KV bytes needed so ``max_num_seqs`` sequences of ``max_model_len`` never preempt."""
    blocks_per_seq = -(-max_model_len // block_size)
    return max_num_seqs * blocks_per_seq * block_size * bytes_per_token


def max_seqs_without_preemption(
    kv_cache_bytes: int, max_model_len: int, bytes_per_token: int, block_size: int = 16
) -> int:
    """Largest ``max_num_seqs`` the KV budget admits with no recompute preemption (>= 0)."""
    per_seq = kv_bytes_for_seqs(1, max_model_len, bytes_per_token, block_size)
    return int(kv_cache_bytes // per_seq) if per_seq > 0 else 0


def admission_allows_no_preemption(
    kv_cache_bytes: int, max_num_seqs: int, max_model_len: int, bytes_per_token: int, block_size: int = 16
) -> bool:
    return max_seqs_without_preemption(kv_cache_bytes, max_model_len, bytes_per_token, block_size) >= max_num_seqs


@dataclass(frozen=True)
class KVGeometry:
    num_layers: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_hf_config(cls, cfg: dict) -> KVGeometry | None:
        """Talker geometry from a Qwen3-TTS (or plain decoder) HF config dict."""
        t = cfg.get("talker_config") if isinstance(cfg.get("talker_config"), dict) else cfg
        try:
            layers = int(t["num_hidden_layers"])
            kv_heads = int(t.get("num_key_value_heads") or t["num_attention_heads"])
            head_dim = int(t.get("head_dim") or (int(t["hidden_size"]) // int(t["num_attention_heads"])))
        except (KeyError, TypeError, ValueError):
            return None
        return cls(layers, kv_heads, head_dim)

    def bytes_per_token(self, dtype: str = "bfloat16") -> int:
        return kv_bytes_per_token(self.num_layers, self.num_kv_heads, self.head_dim, dtype)
