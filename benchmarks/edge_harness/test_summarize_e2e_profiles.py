# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path

from summarize_e2e_profiles import (
    blocker_codes,
    delivery_spacing,
    diagnostic,
    output_checks,
    profile,
    read_lines,
    stats,
)
from single_request_protocol import metadata, validate_settings


class ProfilingEvidenceTests(unittest.TestCase):
    def test_timeout_cannot_leave_a_stale_running_snapshot_in_the_summary(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "run"
            path.mkdir()
            raw = json.dumps({"status": "running"})
            (path / "report.json").write_text(raw)
            host = Path(root) / "run-host"
            host.mkdir()
            (host / "status.json").write_text(json.dumps({"status": "exited", "returncode": 124}))
            self.assertEqual(profile(path)["status"], "failed")
            self.assertIn("124", diagnostic(path, "trace")["error"])
            self.assertEqual((path / "report.json").read_text(), raw)

    def test_missing_checkpoint_is_not_reported_as_an_executed_cpu_kernel_failure(self):
        for model in ("minicpm", "vla"):
            self.assertEqual(
                blocker_codes({"device": "pc_cpu_windows", "model": model}),
                ["COMPLETE_MATCHING_CHECKPOINT_MISSING"],
            )

    def test_delivery_intervals_do_not_cross_requests_or_count_empty_pcm(self):
        rows = [
            {
                "token_unix": [0, 0.01],
                "output_tokens": 4,
                "chunks_all": [{"t_ms": 0, "samples": 10}, {"t_ms": 5, "samples": 0}, {"t_ms": 20, "samples": 10}],
                "ttft_s": 2,
                "prompt_tokens": 100,
            },
            {"token_unix": [100, 100.03], "output_tokens": 2},
        ]
        result = delivery_spacing(rows)
        self.assertEqual(result["text_output_update_gap_ms"]["n"], 2)
        self.assertAlmostEqual(result["text_output_update_gap_ms"]["p95"], 30)
        self.assertEqual(result["audio_nonempty_chunk_gap_ms"]["p50"], 20)
        self.assertEqual(result["effective_prefill_tokens_per_s"]["p50"], 50)
        self.assertEqual(result["text_delivery_counts"][0], {"updates": 2, "output_tokens": 4})

    def test_terminated_launcher_without_report_is_not_pending(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "run"
            self.assertEqual(profile(path)["status"], "pending")
            host = Path(root) / "run-host"
            host.mkdir()
            (host / "status.json").write_text(json.dumps({"status": "exited", "returncode": 1}))
            self.assertEqual(profile(path)["status"], "failed")
            self.assertIn("code 1", profile(path)["error"])

    def test_batch_comparison_detects_consistent_but_different_sequences(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "report.json").write_text(json.dumps({"status": "running"}))
            rows = [
                {
                    "request_id": str(c),
                    "phase": "measured",
                    "length_band": "long",
                    "concurrency": c,
                    "output_token_ids": [c],
                }
                for c in (1, 2)
            ]
            (path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            groups = profile(path)["measurements"]
            self.assertEqual(groups[1]["distinct_token_sequences_for_fixed_prompt"], 1)
            self.assertEqual(groups[1]["requests_with_sequences_not_seen_at_concurrency_1"], 1)

    def test_output_checks_expose_duplicate_and_stale_metadata(self):
        row = {
            "request_id": "a",
            "finished": True,
            "output_tokens": 128,
            "output_token_ids": [1] * 128,
            "arrivals": [{"sequence": 1, "epoch": 0, "kind": "token"}, {"sequence": 1, "epoch": 1, "kind": "done"}],
            "stream": {"high_water_chunks": 5, "max_chunks": 4},
        }
        result = output_checks([row, row])
        self.assertFalse(result["recorded_invariants_pass"])
        self.assertIn("nonincreasing_sequence", result["findings"][0]["errors"])
        self.assertIn("missing_or_mixed_epochs", result["findings"][0]["errors"])
        self.assertIn("queue_bound_exceeded_chunks", result["findings"][0]["errors"])
        self.assertIn("duplicate_request_record", result["findings"][1]["errors"])

    def test_audio_terminal_tail_is_counted_but_wrong_request_rejected(self):
        row = {
            "request_id": "a",
            "finished": True,
            "sr": 24000,
            "chunks_all": [
                {"idx": 0, "samples": 0, "finite": True, "terminal": False},
                {"idx": 1, "samples": 2400, "finite": True, "terminal": True},
            ],
        }
        self.assertTrue(output_checks([row])["recorded_invariants_pass"])
        row["chunks_all"][-1]["request_id"] = "b"
        self.assertIn("wrong_request_id", output_checks([row])["findings"][0]["errors"])

    def test_diagnostic_completion_does_not_hide_failed_checks(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "report.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "oversized_plan_refused": False,
                        "saturation_reached": True,
                        "owned_processes_requiring_harness_cleanup": [{"pid": 123}],
                    }
                )
            )
            result = diagnostic(path, "reliability")
            self.assertFalse(result["oversized_plan_refused"])
            self.assertEqual(len(result["owned_processes_requiring_harness_cleanup"]), 1)
            self.assertNotIn("passed", result)

    def test_incomplete_last_record_is_not_counted(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "requests.jsonl"
            path.write_text('{"ok": true}\n{"partial":')
            self.assertEqual(read_lines(path), [{"ok": True}])
            path.write_text('{"partial":\n{"ok": true}\n')
            with self.assertRaises(json.JSONDecodeError):
                read_lines(path)
            path.write_text('{"partial":\n')
            with self.assertRaises(json.JSONDecodeError):
                read_lines(path)

    def test_percentiles_exclude_unavailable_values(self):
        result = stats([None, float("nan"), *range(1, 21)])
        self.assertEqual(result["n"], 20)
        self.assertEqual(result["p50"], 10)
        self.assertEqual(result["p95"], 19)
        self.assertIsNone(stats([])["p95"])

    def test_total_count_cannot_replace_per_configuration_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "report.json").write_text(json.dumps({"status": "completed", "sustained_wall_s": 1801}))
            row = {"phase": "measured", "length_band": "short", "concurrency": 1, "ttft_s": 0.1}
            (path / "requests.jsonl").write_text((json.dumps(row) + "\n") * 180)
            result = profile(path)
            self.assertFalse(result["minimum_20_per_configuration"])
            self.assertFalse(result["profile_protocol_complete"])

    def test_warmups_do_not_count_and_short_thermal_run_is_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "report.json").write_text(json.dumps({"status": "completed", "sustained_wall_s": 1799}))
            rows = []
            for length in ("short", "medium", "long"):
                for concurrency in (1, 2, 4):
                    rows.extend(
                        {"phase": "measured", "length_band": length, "concurrency": concurrency} for _ in range(20)
                    )
                    rows.append({"phase": "warmup", "length_band": length, "concurrency": concurrency})
            (path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            result = profile(path)
            self.assertTrue(result["minimum_20_per_configuration"])
            self.assertFalse(result["profile_protocol_complete"])
            self.assertEqual([m["n"] for m in result["measurements"]], [20] * 9)
            (path / "report.json").write_text(json.dumps({"status": "completed", "sustained_wall_s": 1801}))
            result = profile(path)
            self.assertFalse(result["required_metrics_complete"])
            self.assertFalse(result["profile_protocol_complete"])
            self.assertEqual(len(result["metric_gaps"]), 180)
            for row in rows:
                row.update(wall_s=2, ttft_s=0.1, decode_tok_per_s=10)
            (path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            result = profile(path)
            self.assertTrue(result["historical_protocol_complete"])
            self.assertFalse(result["profile_protocol_complete"])

    def test_batch1_protocol_requires_only_three_serial_groups_and_records_its_scope(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "report.json").write_text(json.dumps({
                "status": "completed", "sustained_wall_s": 1801,
                "profile_protocol": metadata(),
            }))
            rows = []
            for length in ("short", "medium", "long"):
                rows.append({"phase": "warmup", "length_band": length, "concurrency": 1, "batch_size": 1})
                rows.extend({
                    "phase": "measured", "length_band": length,
                    "concurrency": 1, "batch_size": 1,
                    "wall_s": 2, "ttft_s": 0.1, "decode_tok_per_s": 10,
                } for _ in range(20))
            rows.append({"phase": "sustained", "length_band": "medium", "concurrency": 1, "batch_size": 1})
            for index, row in enumerate(rows):
                row.update({
                    "request_id": f"request-{index}",
                    "finished": True,
                    "output_tokens": 128,
                    "output_token_ids": [1] * 128,
                    "arrivals": [{"sequence": 1, "epoch": 0, "kind": "done"}],
                    "stream": {"high_water_chunks": 1, "max_chunks": 1,
                               "high_water_bytes": 1, "max_bytes": 1},
                })
            requests = path / "requests.jsonl"
            requests.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = profile(path)
            self.assertTrue(result["profile_protocol_complete"])
            self.assertEqual(result["profile_protocol"], "single_request_batch1_v1")
            self.assertEqual([m["concurrency"] for m in result["measurements"]], [1, 1, 1])
            self.assertFalse(result["historical_protocol_complete"])

            rows[1]["finished"] = False
            requests.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = profile(path)
            self.assertFalse(result["profile_protocol_complete"])
            self.assertFalse(result["recorded_output_checks"]["recorded_invariants_pass"])
            rows[1]["finished"] = True

            rows[1]["batch_size"] = 2
            requests.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = profile(path)
            self.assertFalse(result["profile_protocol_complete"])
            self.assertTrue(any("non_single_request_sample" in item for item in result["protocol_violations"]))

    def test_batch1_settings_reject_multiple_active_requests(self):
        validate_settings(1, 1)
        with self.assertRaisesRegex(ValueError, "batch size 1 and concurrency 1"):
            validate_settings(2, 1)
        with self.assertRaisesRegex(ValueError, "batch size 1 and concurrency 1"):
            validate_settings(1, 2)


if __name__ == "__main__":
    unittest.main()
