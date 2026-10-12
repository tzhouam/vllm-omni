# SPDX-License-Identifier: Apache-2.0
"""The rolling ledger records timing depth without promoting release support."""

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from experiments.verify_e2e_summary_table import (
    HOSTED_REQUIRED_CHECKS, apply_hosted_evidence, verify,
    verify_hosted_model_acceptance, verify_timing_evidence,
)


class RollingBatch1EvidenceTests(unittest.TestCase):
    def test_hosted_acceptance_rejects_missing_stage_or_changed_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proof = root / "quality.json"
            proof.write_text('{"comparison":"pinned"}', encoding="utf-8")
            pcm = root / "speech.pcm"
            pcm.write_bytes(b"\x00\x01\x02\x03")
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            final = root / "final_output.json"
            final.write_text(json.dumps({
                "pcm_path": "speech.pcm", "pcm_sha256": sha(pcm),
                "sample_rate": 24000, "samples": 2, "tail_flushed": True,
            }), encoding="utf-8")
            model = "Qwen3-TTS 0.6B CustomVoice"
            record = {
                "identity": {"model": model, "revision": "pinned-revision"},
                "completion": {
                    "final_output_path": "final_output.json",
                    "final_output_sha256": sha(final),
                    "quality": {"reference": "checkpoint@pinned-revision", "checks": [
                        {"name": name, "passed": True, "evidence_path": "quality.json",
                         "sha256": sha(proof)} for name in HOSTED_REQUIRED_CHECKS[model]
                    ]},
                },
            }
            verify_hosted_model_acceptance(record, root)
            missing = copy.deepcopy(record)
            missing["completion"]["quality"]["checks"].pop()
            with self.assertRaisesRegex(ValueError, "omits whole-model"):
                verify_hosted_model_acceptance(missing, root)
            pcm.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                verify_hosted_model_acceptance(record, root)

    def test_hosted_index_never_promotes_a_component_report(self):
        summary = Path(__file__).parent / "results/e2e_profiling_20260922/evidence/summary"
        ledger = verify((summary / "README.md").read_text(encoding="utf-8"))
        apply_hosted_evidence(ledger, summary / "hosted_e2e_index.json")
        self.assertEqual(ledger["counts"]["hosted_replay_verified"], 0)
        self.assertEqual(ledger["counts"]["hosted_external_blocker"], 10)
        self.assertEqual(sum(case["hosted_full_replay"] == "not_yet_verified"
                             for case in ledger["cases"]), 15)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            (root / "component.json").write_text('{"status":"SUCCESS"}', encoding="utf-8")
            index = root / "index.json"
            index.write_text(json.dumps({
                "schema_version": 1,
                "entries": [{
                    "device_configuration": "Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy",
                    "model": "Spark-X2.5",
                    "outcome": "hosted_functional_replay_pending_device",
                    "evidence_path": "component.json",
                }],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be run.json"):
                apply_hosted_evidence(verify((summary / "README.md").read_text(encoding="utf-8")),
                                      index)

    def test_named_external_blocker_keeps_release_unqualified(self):
        summary = Path(__file__).parent / "results/e2e_profiling_20260922/evidence/summary"
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            report = {
                "status": "external_blocker", "model": "InternVLA-A1",
                "target_name": "Samsung Galaxy S25",
                "reason": "matching Place_Markpen reference data requires approval",
                "next_step": "retry the pinned dataset access probe after approval",
                "source_evidence": "access_denial.json",
            }
            (root / "access_denial.json").write_text(
                '{"status":"HTTP 403","repository":"InternRobotics/InternData-A1"}',
                encoding="utf-8")
            report["source_sha256"] = hashlib.sha256(
                (root / "access_denial.json").read_bytes()).hexdigest()
            (root / "blocker.json").write_text(json.dumps(report), encoding="utf-8")
            index = root / "index.json"
            index.write_text(json.dumps({
                "schema_version": 1,
                "entries": [{
                    "device_configuration": "Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy",
                    "model": "InternVLA-A1", "outcome": "external_blocker",
                    "evidence_path": "blocker.json",
                }],
            }), encoding="utf-8")
            ledger = verify((summary / "README.md").read_text(encoding="utf-8"))
            apply_hosted_evidence(ledger, index)
            case = next(case for case in ledger["cases"] if
                        case["device_configuration"].startswith("Mobile Galaxy S25")
                        and case["model"] == "InternVLA-A1")
            self.assertEqual(case["qualification_outcome"], "external_blocker")
            self.assertEqual(case["release_qualification"], "not_qualified")
            self.assertEqual(ledger["counts"]["hosted_external_blocker"], 1)
            (root / "access_denial.json").write_text('{"status":"approved"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash is missing or changed"):
                apply_hosted_evidence(verify((summary / "README.md").read_text(encoding="utf-8")),
                                      index)

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
