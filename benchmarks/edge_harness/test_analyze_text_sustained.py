# SPDX-License-Identifier: Apache-2.0
"""Protocol regressions for archived sequential text profiles."""

from analyze_text_sustained import analyze
from single_request_protocol import metadata


def _run():
    rows = []
    now = 1000.0
    for band in ("short", "medium", "long"):
        for phase in ["warmup", *(["measured"] * 20)]:
            rows.append(_row(len(rows), phase, band, now, 1.0))
            now += 1.1
    # Roughly Spark-sized requests form a continuously busy 30-minute window.
    start = now
    for _ in range(600):
        rows.append(_row(len(rows), "sustained", "medium", now, 3.0))
        now += 3.0
    return ({"status": "completed", "profile_protocol": metadata(),
             "settings": {"batch_size": 1, "concurrency": 1}, "requests": len(rows),
             "sustained_start_unix": start, "sustained_wall_s": 1800.0}, rows)


def _row(index, phase, band, submitted, wall):
    first, last = submitted + 0.1, submitted + wall - 0.1
    return {"request_id": str(index), "phase": phase, "length_band": band,
            "batch_size": 1, "concurrency": 1, "submitted_unix": submitted,
            "wall_s": wall, "ttft_s": first - submitted,
            "decode_tok_per_s": 127 / (last - first),
            "first_token_unix": first, "last_token_unix": last, "token_unix": [first, last],
            "prompt_tokens": {"short": 56, "medium": 488, "long": 1928}[band],
            "output_tokens": 128, "output_token_ids": [1] * 128,
            "finished": True, "error": None,
            "arrivals": [{"sequence": 0, "epoch": 1, "kind": "done"}],
            "stream": {"high_water_chunks": 1, "max_chunks": 1,
                       "high_water_bytes": 16, "max_bytes": 16}}


def test_complete_raw_window_and_output_metadata_pass():
    report, rows = _run()
    result = analyze(report, rows, [])
    assert result["timing_protocol_complete"]
    assert result["sustained"]["observed_request_window_s"] >= 1799.99
    assert all(result["measured"][band]["requests"] == 20 for band in ("short", "medium", "long"))


def test_reported_duration_cannot_hide_overlap_or_short_raw_window():
    report, rows = _run()
    rows[-1]["submitted_unix"] = rows[-2]["submitted_unix"] + 2.0
    result = analyze(report, rows, [])
    assert not result["timing_protocol_complete"]
    assert any("overlapping_request_interval" in violation for violation in result["violations"])
    rows[-1]["submitted_unix"] = rows[-2]["submitted_unix"] + 3.0
    rows[-1]["wall_s"] = 2.0
    result = analyze(report, rows, [])
    assert "sustained_request_window_under_30_minutes" in result["violations"]


def test_long_idle_gap_cannot_count_as_sustained_generation():
    report, rows = _run()
    rows[-1]["submitted_unix"] = rows[-2]["submitted_unix"] + 100.0
    first = rows[-1]["submitted_unix"] + 0.1
    last = rows[-1]["submitted_unix"] + 2.9
    rows[-1].update(first_token_unix=first, last_token_unix=last,
                    token_unix=[first, last], ttft_s=first - rows[-1]["submitted_unix"],
                    decode_tok_per_s=127 / (last - first))
    result = analyze(report, rows, [])
    assert result["sustained"]["observed_request_window_s"] >= 1800
    assert "sustained_active_fraction_low" in result["violations"]


def test_late_warmup_and_equal_prompt_lengths_do_not_qualify():
    report, rows = _run()
    rows[0]["phase"] = "measured"
    rows[1]["phase"] = "warmup"
    for row in rows:
        row["prompt_tokens"] = 56
    result = analyze(report, rows, [])
    assert "warmup_after_measurement:short" in result["violations"]
    assert "nonincreasing_prompt_length_bands" in result["violations"]


def test_invalid_or_unverified_raw_timing_cannot_qualify():
    report, rows = _run()
    rows[1]["ttft_s"] = -1
    rows[2]["decode_tok_per_s"] = 100000
    result = analyze(report, rows, [])
    assert not result["timing_protocol_complete"]
    assert any("ttft_s" in gap["fields"] for gap in result["metric_gaps"])
    assert any("decode_rate_disagrees_with_raw_timestamps" in gap["fields"] for gap in result["metric_gaps"])
