"""Tests for vllm_omni.edge.calibrate (CPU)."""

from __future__ import annotations

import pytest

from vllm_omni.edge import calibrate as cal
from vllm_omni.edge.hardware_probe import HardwareProfile

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SR = 24000
FRAME = 1920


def _chunks(rate_frames_per_s: float, sizes: list[int], ttfa_ms: float = 30.0):
    """Synthesize arrivals for a generator producing ``rate`` frames/s."""
    chunks = []
    t = ttfa_ms
    for i, n in enumerate(sizes):
        if i > 0:
            t += n * 1000.0 / rate_frames_per_s
        chunks.append({"t_ms": t, "samples": n * FRAME})
    return chunks


def test_rate_estimate_from_measured_shape():
    # measured on L20X: 25 frames every ~160 ms -> ~156 frames/s
    chunks = [
        {"t_ms": 30.0, "samples": FRAME},
        {"t_ms": 190.0, "samples": 25 * FRAME},
        {"t_ms": 350.0, "samples": 25 * FRAME},
    ]
    rate = cal.frames_per_second_from_chunks(chunks, SR)
    assert rate == pytest.approx(50 / 0.32, rel=1e-3)
    assert cal.frames_per_second_from_chunks(chunks[:1], SR) is None


def test_fast_generator_keeps_small_first_chunk_and_no_adaptive():
    sched = cal.derive_chunk_schedule(_chunks(150.0, [1, 25, 25, 25]), SR)
    assert sched["measured"] is True
    assert sched["rtf"] < 0.2
    assert sched["codec_chunk_adaptive"] is False
    assert sched["codec_chunk_min_frames"] == 1
    assert sched["codec_chunk_ramp"][-1] == 25 and sched["codec_chunk_ramp"][0] == 1


def test_slow_generator_needs_lead_and_adaptive():
    # 10 frames/s < 12.5 real time: each 25-frame chunk takes 2.5 s to make but plays 2.0 s
    sched = cal.derive_chunk_schedule(_chunks(10.0, [2, 4, 8, 16, 25, 25]), SR)
    assert sched["rtf"] > 1.0
    assert sched["codec_chunk_adaptive"] is True
    assert sched["codec_chunk_min_frames"] >= 7  # (40 + 20*25) / 80 -> 6.75 -> 7
    assert sched["codec_chunk_ramp"][-1] == 25
    assert all(a < b for a, b in zip(sched["codec_chunk_ramp"], sched["codec_chunk_ramp"][1:]))
    extra = cal.connector_extra_from_calibration(sched)
    assert extra["codec_chunk_adaptive"] is True and extra["codec_chunk_ramp"][-1] == 25


def test_unmeasurable_falls_back_to_defaults():
    sched = cal.derive_chunk_schedule([{"t_ms": 30.0, "samples": FRAME}], SR)
    assert sched["measured"] is False and sched["codec_chunk_ramp"] == [2, 4, 8, 16, 25]


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    prof = HardwareProfile(arch="x86_64", cpu_model="x", n_physical=8, gpu_name="", gpu_mem_bytes=0)
    fp = cal.hardware_fingerprint(prof, "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    assert len(fp) == 16
    assert cal.load_calibration(fp) is None
    p = cal.save_calibration(fp, {"codec_chunk_ramp": [1, 25]})
    assert p.parent == tmp_path / "omni_edge"
    assert cal.load_calibration(fp) == {"codec_chunk_ramp": [1, 25]}
    # different model -> different fingerprint
    assert cal.hardware_fingerprint(prof, "other") != fp
