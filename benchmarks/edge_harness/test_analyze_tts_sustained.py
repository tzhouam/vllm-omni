# SPDX-License-Identifier: Apache-2.0
"""Pure synthetic checks for the read-only TTS sustained analyzer."""

import unittest

from analyze_tts_sustained import analyze
from single_request_protocol import metadata


def record(number, phase, band, submitted, wall=1.0, ttfa=100.0):
    rid = f"{phase}-{band}-{number}"
    return {
        "request_id": rid,
        "phase": phase,
        "length_band": band,
        "submitted_unix": submitted,
        "finished_unix": submitted + wall,
        "batch_size": 1,
        "concurrency": 1,
        "finished": True,
        "sr": 24000,
        "chunks_all": [
            {
                "idx": 0,
                "t_ms": ttfa,
                "samples": 24000,
                "sr": 24000,
                "request_id": rid,
                "terminal": False,
                "finite": True,
            },
            {
                "idx": 1,
                "t_ms": 600.0,
                "samples": 24000,
                "sr": 24000,
                "request_id": rid,
                "terminal": True,
                "finite": True,
            },
        ],
        "ttfa_ms": ttfa,
        "playback_start_ms": ttfa,
        "stall_at_ttfa_ms": 0.0,
        "total_wall_s": wall,
        "rtf_total": wall / 2,
        "underruns": [],
    }


def complete_fixture():
    rows = []
    t = 1000.0
    for band in ("short", "medium", "long"):
        rows.append(record(len(rows), "warmup", band, t))
        t += 2
        for index in range(20):
            rows.append(record(len(rows), "measured", band, t, wall=(index + 1) / 10))
            t += 3
    sustained_start = t
    for index in range(200):
        rows.append(record(len(rows), "sustained", "medium", t, wall=(index + 1) / 100, ttfa=100 + index))
        t += 10
    report = {
        "status": "completed",
        "settings": {"batch_size": 1, "concurrency": 1},
        "profile_protocol": metadata(),
        "requests": len(rows),
        "sustained_start_unix": sustained_start,
        "sustained_wall_s": 2000,
        "gpu_telemetry": {"error": None},
    }
    telemetry = [
        {"unix": sustained_start + i * 100, "gpu_power_mw": 50000 + i, "gpu_temperature_c": 60 + i}
        for i in range(20)
    ]
    return report, rows, telemetry


class TtsSustainedAnalysisTests(unittest.TestCase):
    def test_complete_batch1_profile_and_nearest_rank_percentiles(self):
        report, rows, telemetry = complete_fixture()
        result = analyze(report, rows, telemetry)
        self.assertTrue(result["timing_protocol_complete"])
        self.assertEqual(result["measured"]["short"]["metrics"]["total_wall_s"]["p50"], 1.0)
        self.assertEqual(result["measured"]["short"]["metrics"]["total_wall_s"]["p95"], 1.9)
        self.assertEqual(result["sustained"]["first_vs_last"]["first_n"], 100)
        self.assertEqual(result["sustained"]["first_vs_last"]["last_n"], 100)
        self.assertEqual(result["sustained"]["first_vs_last"]["first_ttfa_ms"]["p50"], 149)
        self.assertEqual(result["sustained"]["first_vs_last"]["last_ttfa_ms"]["p50"], 249)
        self.assertEqual(result["telemetry"]["sustained_window"]["ranges"]["gpu_power_mw"]["min"], 50000)
        self.assertEqual(result["telemetry"]["sustained_window"]["ranges"]["gpu_temperature_c"]["max"], 79)
        self.assertIn("audio reference", result["qualification"])

    def test_batch_overlap_and_short_duration_cannot_pass(self):
        report, rows, telemetry = complete_fixture()
        report["settings"]["concurrency"] = 2
        report["sustained_wall_s"] = 1799
        rows[1]["concurrency"] = 2
        rows[2]["submitted_unix"] = rows[1]["finished_unix"] - 0.5
        result = analyze(report, rows, telemetry)
        self.assertFalse(result["timing_protocol_complete"])
        self.assertIn("settings_not_batch1_concurrency1", result["violations"])
        self.assertIn("sustained_wall_under_30_minutes", result["violations"])
        self.assertIn(f"overlapping_requests:{rows[2]['request_id']}", result["violations"])

    def test_finite_audio_output_and_playback_failures_are_visible(self):
        report, rows, telemetry = complete_fixture()
        bad = rows[-1]
        bad["chunks_all"][0]["finite"] = False
        bad["ttfa_ms"] = float("nan")
        bad["playback_start_ms"] = 650
        bad["underruns"] = [[1, 50.0]]
        result = analyze(report, rows, telemetry)
        self.assertFalse(result["timing_protocol_complete"])
        self.assertIn("recorded_output_invariants_failed", result["violations"])
        self.assertIn("measured_or_sustained_metric_gaps", result["violations"])
        self.assertEqual(result["sustained"]["playback"]["simulated_underrun_count"], 1)
        self.assertEqual(result["sustained"]["playback"]["simulated_underrun_max_ms"], 50.0)
        self.assertEqual(result["output"]["findings"][0]["request_id"], bad["request_id"])
        self.assertEqual(result["finite_metric_gaps"][0]["request_id"], bad["request_id"])
        self.assertEqual(result["failures"]["nonfinite_audio_requests"], 1)
        self.assertEqual(result["failures"]["nonfinite_or_missing_metric_requests"], 1)

    def test_startup_buffer_uses_complete_chunk_timeline(self):
        report, rows, telemetry = complete_fixture()
        measured = next(row for row in rows if row["phase"] == "measured")
        measured["playback_start_ms"] = 250
        measured["stall_at_ttfa_ms"] = 150
        measured["underruns"] = [[1, 150.0]]
        result = analyze(report, rows, telemetry)
        playback = result["measured"]["short"]["playback"]
        self.assertEqual(playback["minimum_underrun_free_startup_buffer_ms"]["maximum"], 150)
        self.assertEqual(playback["requests_with_simulated_underruns"], 1)
        self.assertEqual(playback["simulated_underrun_total_ms"], 150)


if __name__ == "__main__":
    unittest.main()
