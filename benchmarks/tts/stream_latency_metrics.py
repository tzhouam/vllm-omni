"""Pure-Python metrics for streaming TTS latency (no torch / engine imports).

A streamed utterance is a list of chunk records ``{"t_ms": float, "samples": int}``
where ``t_ms`` is the arrival time of the chunk relative to request submission and
``samples`` is the number of PCM samples it carries at sample rate ``sr``.

Definitions
-----------
TTFA
    Arrival time of the first chunk that carries audio.
playback_start_ms
    The earliest time a player could start such that it never underruns given the
    observed arrivals: ``max_i (t_i - C_i)`` where ``C_i`` is the audio duration
    (ms) accumulated before chunk ``i``. Always ``>= TTFA``.
stall_at_ttfa_ms
    Total stall a player would suffer if it started at TTFA instead of
    ``playback_start_ms`` (this is the "1 -> 25 frame cliff" measured on the
    default Qwen3-TTS schedule).
RTF
    Wall time divided by generated audio seconds; lower is faster than real time.
"""

from __future__ import annotations

import math
from typing import Any


def _audio_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in chunks if int(c.get("samples", 0)) > 0]


def chunk_durations_ms(chunks: list[dict[str, Any]], sr: int) -> list[float]:
    """Audio duration in ms of each audio-carrying chunk, in arrival order."""
    return [1000.0 * int(c["samples"]) / sr for c in _audio_chunks(chunks)]


def cumulative_audio_before_ms(chunks: list[dict[str, Any]], sr: int) -> list[float]:
    """``C_i``: audio (ms) delivered before chunk ``i`` (audio chunks only)."""
    out: list[float] = []
    acc = 0.0
    for d in chunk_durations_ms(chunks, sr):
        out.append(acc)
        acc += d
    return out


def ttfa_ms(chunks: list[dict[str, Any]]) -> float | None:
    audio = _audio_chunks(chunks)
    return float(audio[0]["t_ms"]) if audio else None


def playback_start_ms(chunks: list[dict[str, Any]], sr: int) -> float | None:
    """Earliest underrun-free playback start; ``None`` when no audio arrived."""
    audio = _audio_chunks(chunks)
    if not audio:
        return None
    before = cumulative_audio_before_ms(chunks, sr)
    return max(float(c["t_ms"]) - b for c, b in zip(audio, before))


def underruns_if_start_at(chunks: list[dict[str, Any]], sr: int, start_ms: float) -> list[tuple[int, float]]:
    """Stalls a player starting at ``start_ms`` would suffer: ``[(chunk_index, stall_ms)]``.

    After each stall the playback cursor shifts by the stall, so later chunks are
    judged against the shifted schedule (this models a real ring-buffer player).
    """
    audio = _audio_chunks(chunks)
    before = cumulative_audio_before_ms(chunks, sr)
    stalls: list[tuple[int, float]] = []
    cursor = float(start_ms)
    for i, (c, b) in enumerate(zip(audio, before)):
        needed_at = cursor + b
        t = float(c["t_ms"])
        if t > needed_at + 1e-9:
            stall = t - needed_at
            stalls.append((i, stall))
            cursor += stall
    return stalls


def stall_at_ttfa_ms(chunks: list[dict[str, Any]], sr: int) -> float | None:
    t0 = ttfa_ms(chunks)
    if t0 is None:
        return None
    return sum(s for _, s in underruns_if_start_at(chunks, sr, t0))


def audio_seconds(chunks: list[dict[str, Any]], sr: int) -> float:
    return sum(int(c["samples"]) for c in _audio_chunks(chunks)) / sr


def rtf(total_wall_s: float, audio_s: float) -> float | None:
    if audio_s <= 0:
        return None
    return total_wall_s / audio_s


def inter_chunk_gaps_ms(chunks: list[dict[str, Any]]) -> list[float]:
    """Arrival gaps between consecutive audio chunks (excludes the first chunk)."""
    audio = _audio_chunks(chunks)
    return [float(b["t_ms"]) - float(a["t_ms"]) for a, b in zip(audio, audio[1:])]


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (``q`` in [0, 1]); ``None`` for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _stats(values: list[float]) -> dict[str, float | None]:
    vals = [float(v) for v in values if v is not None]
    return {
        "n": len(vals),
        "p50": percentile(vals, 0.50),
        "p90": percentile(vals, 0.90),
        "max": max(vals) if vals else None,
        "mean": (sum(vals) / len(vals)) if vals else None,
    }


def request_metrics(
    chunks: list[dict[str, Any]], sr: int, total_wall_s: float, tail_audio_s: float = 0.0
) -> dict[str, Any]:
    """Derive all per-request metrics from a chunk timeline."""
    streamed_s = audio_seconds(chunks, sr)
    t_last = max((float(c["t_ms"]) for c in _audio_chunks(chunks)), default=None)
    return {
        "ttfa_ms": ttfa_ms(chunks),
        "playback_start_ms": playback_start_ms(chunks, sr),
        "stall_at_ttfa_ms": stall_at_ttfa_ms(chunks, sr),
        "n_chunks": len(_audio_chunks(chunks)),
        "streamed_audio_s": streamed_s,
        "tail_audio_s": tail_audio_s,
        "total_wall_s": total_wall_s,
        "rtf_streamed": rtf(t_last / 1000.0, streamed_s) if t_last is not None else None,
        "rtf_total": rtf(total_wall_s, streamed_s + tail_audio_s),
        "gap_ms": inter_chunk_gaps_ms(chunks),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-request metric dicts (as produced by :func:`request_metrics`)."""
    gaps: list[float] = []
    for r in records:
        gaps.extend(r.get("gap_ms") or [])
    return {
        "n_requests": len(records),
        "ttfa_ms": _stats([r.get("ttfa_ms") for r in records]),
        "playback_start_ms": _stats([r.get("playback_start_ms") for r in records]),
        "stall_at_ttfa_ms": _stats([r.get("stall_at_ttfa_ms") for r in records]),
        "rtf_streamed": _stats([r.get("rtf_streamed") for r in records]),
        "rtf_total": _stats([r.get("rtf_total") for r in records]),
        "gap_ms": _stats(gaps),
        "total_audio_s": sum(float(r.get("streamed_audio_s") or 0.0) for r in records),
        "total_wall_s": sum(float(r.get("total_wall_s") or 0.0) for r in records),
    }
