"""Tests for benchmarks/tts/stream_latency_metrics.py (pure Python, CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "tts"))
import stream_latency_metrics as m  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SR = 24000
FRAME = 1920  # one 12.5 Hz codec frame at 24 kHz = 80 ms


def test_measured_default_schedule_has_80ms_cliff():
    # Shape measured on L20X with the default 1 -> 25 frame schedule.
    chunks = [
        {"t_ms": 30.0, "samples": FRAME},
        {"t_ms": 190.0, "samples": 25 * FRAME},
        {"t_ms": 348.0, "samples": 25 * FRAME},
    ]
    assert m.ttfa_ms(chunks) == 30.0
    assert m.playback_start_ms(chunks, SR) == pytest.approx(110.0)
    stalls = m.underruns_if_start_at(chunks, SR, 30.0)
    assert stalls == [(1, pytest.approx(80.0))]
    assert m.stall_at_ttfa_ms(chunks, SR) == pytest.approx(80.0)


def test_fast_arrivals_start_at_ttfa():
    chunks = [
        {"t_ms": 30.0, "samples": 4 * FRAME},
        {"t_ms": 100.0, "samples": 8 * FRAME},
        {"t_ms": 200.0, "samples": 25 * FRAME},
    ]
    assert m.playback_start_ms(chunks, SR) == 30.0
    assert m.underruns_if_start_at(chunks, SR, 30.0) == []
    assert m.stall_at_ttfa_ms(chunks, SR) == 0.0


def test_late_middle_chunk_dominates_and_shifts_cursor():
    chunks = [{"t_ms": 10.0, "samples": FRAME}, {"t_ms": 500.0, "samples": FRAME}, {"t_ms": 520.0, "samples": FRAME}]
    # C = [0, 80, 160]; t - C = [10, 420, 360] -> 420
    assert m.playback_start_ms(chunks, SR) == pytest.approx(420.0)
    stalls = m.underruns_if_start_at(chunks, SR, 10.0)
    # chunk 1 needed at 90, arrives 500 -> stall 410; cursor 420; chunk 2 needed at 580, arrives 520 -> ok
    assert stalls == [(1, pytest.approx(410.0))]


def test_zero_sample_chunks_are_ignored():
    chunks = [{"t_ms": 5.0, "samples": 0}, {"t_ms": 30.0, "samples": FRAME}, {"t_ms": 40.0, "samples": 0}]
    assert m.ttfa_ms(chunks) == 30.0
    assert m.chunk_durations_ms(chunks, SR) == [pytest.approx(80.0)]
    assert m.inter_chunk_gaps_ms(chunks) == []
    assert m.playback_start_ms([], SR) is None
    assert m.stall_at_ttfa_ms([{"t_ms": 1.0, "samples": 0}], SR) is None


def test_request_metrics_and_summary():
    chunks = [{"t_ms": 30.0, "samples": FRAME}, {"t_ms": 190.0, "samples": 25 * FRAME}]
    r = m.request_metrics(chunks, SR, total_wall_s=0.4, tail_audio_s=0.4)
    assert r["n_chunks"] == 2
    assert r["streamed_audio_s"] == pytest.approx(26 * 0.08)
    assert r["rtf_streamed"] == pytest.approx(0.190 / (26 * 0.08))
    assert r["rtf_total"] == pytest.approx(0.4 / (26 * 0.08 + 0.4))
    assert r["gap_ms"] == [pytest.approx(160.0)]
    s = m.summarize([r, r, r])
    assert s["n_requests"] == 3
    assert s["ttfa_ms"]["p50"] == 30.0
    assert s["gap_ms"]["n"] == 3
    assert s["stall_at_ttfa_ms"]["max"] == pytest.approx(80.0)


def test_percentile_nearest_rank():
    assert m.percentile([], 0.5) is None
    assert m.percentile([1.0], 0.9) == 1.0
    assert m.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert m.percentile([1.0, 2.0, 3.0, 4.0], 0.9) == 4.0
