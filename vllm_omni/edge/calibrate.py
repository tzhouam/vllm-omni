"""Calibration of the codec chunk schedule from measured chunk timings.

Given the chunk arrivals of a warm-up request (as recorded by
``benchmarks/tts/stream_latency_bench.py``), derive parameters for the runtime
adaptive chunk controller (``codec_chunk_adaptive`` / ``codec_chunk_min_frames``
/ ``codec_chunk_safety_margin_ms``) and a static ramp that would have played
back without underruns. Results are cached per hardware fingerprint so the
next start can apply them without re-measuring.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from vllm_omni.edge.hardware_probe import HardwareProfile, hardware_class

FRAME_MS = 80.0  # 12.5 Hz codec


def hardware_fingerprint(profile: HardwareProfile, model: str) -> str:
    key = json.dumps(
        {
            "class": hardware_class(profile),
            "cpu": profile.cpu_model,
            "flags": sorted(profile.cpu_flags),
            "gpu": profile.gpu_name,
            "gpu_mem": profile.gpu_mem_bytes // (256 * 2**20),
            "cores": profile.n_physical,
            "model": model,
        },
        sort_keys=True,
    )
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def cache_dir() -> Path:
    root = os.environ.get("VLLM_CACHE_ROOT") or os.path.join(os.path.expanduser("~"), ".cache", "vllm")
    return Path(root) / "omni_edge"


def cache_path(fingerprint: str) -> Path:
    return cache_dir() / f"{fingerprint}.json"


def load_calibration(fingerprint: str) -> dict[str, Any] | None:
    p = cache_path(fingerprint)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_calibration(fingerprint: str, data: dict[str, Any]) -> Path:
    p = cache_path(fingerprint)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return p


def frames_per_second_from_chunks(chunks: list[dict[str, Any]], sr: int, frame_ms: float = FRAME_MS) -> float | None:
    """Steady-state generation rate (codec frames per wall second) from chunk arrivals."""
    audio = [c for c in chunks if int(c.get("samples", 0)) > 0]
    if len(audio) < 2:
        return None
    first, last = audio[0], audio[-1]
    frames = sum(int(c["samples"]) for c in audio[1:]) / sr * 1000.0 / frame_ms
    wall_s = (float(last["t_ms"]) - float(first["t_ms"])) / 1000.0
    if wall_s <= 0:
        return None
    return frames / wall_s


def derive_chunk_schedule(
    chunks: list[dict[str, Any]],
    sr: int,
    *,
    steady_frames: int = 25,
    frame_ms: float = FRAME_MS,
    jitter_ms: float = 40.0,
) -> dict[str, Any]:
    """Derive adaptive-controller parameters and a static ramp from one measured request.

    Rule: the audio buffered before waiting for chunk ``i+1`` must cover the
    time to produce it. With generation rate ``r`` frames/s (real-time factor
    ``rtf = 12.5 / r``), producing ``n`` frames takes ``n / r`` seconds while
    ``n`` frames play for ``n * 0.08`` seconds, so a chunk of size ``n`` is safe
    once ``n * 0.08 >= n / r + margin``; the first chunk therefore sets the
    playback lead. We choose ``min_frames`` so the lead covers ``margin`` and
    build a doubling ramp up to ``steady_frames``.
    """
    rate = frames_per_second_from_chunks(chunks, sr, frame_ms)
    audio = [c for c in chunks if int(c.get("samples", 0)) > 0]
    ttfa_ms = float(audio[0]["t_ms"]) if audio else None
    if rate is None or rate <= 0:
        return {
            "measured": False,
            "codec_chunk_adaptive": True,
            "codec_chunk_min_frames": 2,
            "codec_chunk_safety_margin_ms": 50.0,
            "codec_chunk_ramp": [2, 4, 8, 16, steady_frames],
            "ttfa_ms": ttfa_ms,
        }
    rtf = (1000.0 / frame_ms) / rate  # wall seconds per audio second
    # Time to produce one frame vs its playback duration.
    produce_ms_per_frame = 1000.0 / rate
    lead_needed_ms = jitter_ms + max(0.0, produce_ms_per_frame - frame_ms) * steady_frames
    min_frames = max(1, int(-(-lead_needed_ms // frame_ms)))  # ceil
    min_frames = min(min_frames, steady_frames)
    margin_ms = max(20.0, min(400.0, jitter_ms + produce_ms_per_frame))
    ramp: list[int] = []
    n = min_frames
    while n < steady_frames:
        ramp.append(n)
        n *= 2
    ramp.append(steady_frames)
    if len(ramp) < 2:
        ramp = [max(1, steady_frames // 2), steady_frames]
    return {
        "measured": True,
        "rate_frames_per_s": rate,
        "rtf": rtf,
        "ttfa_ms": ttfa_ms,
        "codec_chunk_adaptive": rtf > 0.5,  # only worth it when generation is not far faster than real time
        "codec_chunk_min_frames": min_frames,
        "codec_chunk_safety_margin_ms": margin_ms,
        "codec_chunk_ramp": ramp,
    }


def connector_extra_from_calibration(cal: dict[str, Any]) -> dict[str, Any]:
    out = {
        "codec_chunk_ramp": list(cal.get("codec_chunk_ramp", [2, 4, 8, 16, 25])),
        "codec_chunk_min_frames": int(cal.get("codec_chunk_min_frames", 2)),
        "codec_chunk_safety_margin_ms": float(cal.get("codec_chunk_safety_margin_ms", 50.0)),
    }
    if cal.get("codec_chunk_adaptive"):
        out["codec_chunk_adaptive"] = True
    return out
