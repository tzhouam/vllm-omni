# SPDX-License-Identifier: Apache-2.0
"""The rolling ledger records timing depth without promoting release support."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from experiments.verify_e2e_summary_table import verify, verify_timing_evidence


class RollingBatch1EvidenceTests(unittest.TestCase):
    def test_spark_and_tts_timing_sources_are_raw_and_release_stays_open(self):
        summary = Path(__file__).parent / "results/e2e_profiling_20260922/evidence/summary"
        result = verify((summary / "README.md").read_text(encoding="utf-8"))
        self.assertEqual(result["counts"]["batch1_timing_verified"], 2)
        verified = [case for case in result["cases"] if case["batch1_protocol"] == "verified_timing_only"]
        self.assertEqual(len(verified), 2)
        self.assertEqual({case["model"] for case in verified}, {"Spark-X2.5", "Qwen3-TTS 0.6B CustomVoice"})
        for case in verified:
            self.assertEqual(case["device_configuration"], "PC HX370 + RTX 5090 Laptop / WSL")
            self.assertEqual(case["release_qualification"], "not_qualified")
            self.assertEqual(case["qualification_outcome"], "open")
            raw = summary / case["batch1_protocol_evidence"]["raw_requests"]
            analysis = summary / case["batch1_protocol_evidence"]["analysis"]
            self.assertTrue(raw.is_file())
            self.assertTrue(analysis.is_file())
            self.assertTrue(json.loads(analysis.read_text(encoding="utf-8"))["timing_protocol_complete"])
        verify_timing_evidence(result, summary / "README.md")
        self.assertEqual(result["counts"]["release_qualified"], 0)

    def test_stale_raw_requests_cannot_regenerate_verified_timing_cell(self):
        summary = Path(__file__).parent / "results/e2e_profiling_20260922/evidence/summary"
        result = verify((summary / "README.md").read_text(encoding="utf-8"))
        case = next(case for case in result["cases"] if case["model"] == "Spark-X2.5"
                    and case["batch1_protocol"] == "verified_timing_only")
        source = (summary / case["batch1_protocol_evidence"]["raw_requests"]).parent
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "timing"
            dest.mkdir()
            for name in ("requests.jsonl", "analysis.json", "report.json",
                         "gpu_telemetry.jsonl", "power_snapshot_during.json"):
                shutil.copyfile(source / name, dest / name)
            copied = copy.deepcopy(result)
            copied_case = next(case for case in copied["cases"] if case["model"] == "Spark-X2.5"
                               and case["batch1_protocol"] == "verified_timing_only")
            copied["cases"] = [copied_case]
            copied_case["batch1_protocol_evidence"] = {
                "raw_requests": "timing/requests.jsonl",
                "analysis": "timing/analysis.json",
            }
            raw = dest / "requests.jsonl"
            raw.write_text("\n".join(raw.read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
                           encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "request counts disagree"):
                verify_timing_evidence(copied, Path(tmp) / "README.md")
            (dest / "analysis.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing or empty timing evidence"):
                verify_timing_evidence(copied, Path(tmp) / "README.md")


if __name__ == "__main__":
    unittest.main()
