"""Review the committed memory-quality aggregate for private-data leakage."""

from __future__ import annotations

import json
import re
from pathlib import Path


PUBLIC = (
    Path(__file__).resolve().parents[2]
    / "benchmarks" / "edge_agent" / "public_evidence"
    / "agent_memory_quality_20261006.json"
)
TOP_KEYS = {
    "schema", "reviewed_date", "review_method", "scope", "batch_size",
    "concurrency", "requests_per_run", "qualification_status", "hardware",
    "route", "runs", "failed_check_note", "limits",
}
HARDWARE_KEYS = {"os", "cpu", "gpu", "gpu_driver", "power_condition"}
ROUTE_KEYS = {
    "route_id", "model", "backend", "reported_placement", "precision",
    "checkpoint_revision", "model_sha256", "projector_sha256", "server_sha256",
    "native_config_sha256", "lineage_sha256",
}
RUN_KEYS = {
    "run_id", "result", "code_state", "loaded_runtime_sha256", "raw_sha256",
    "index_sha256", "cases", "passed", "per_case",
}
CASE_KEYS = {"case_id", "passed", "failed_checks"}


def test_public_memory_quality_contains_only_reviewed_aggregate_fields() -> None:
    aggregate = json.loads(PUBLIC.read_text(encoding="utf-8"))
    assert set(aggregate) == TOP_KEYS
    assert set(aggregate["hardware"]) == HARDWARE_KEYS
    assert set(aggregate["route"]) == ROUTE_KEYS
    assert aggregate["batch_size"] == aggregate["concurrency"] == 1
    assert aggregate["requests_per_run"] == 16
    assert "unsigned" in aggregate["qualification_status"]
    assert [run["passed"] for run in aggregate["runs"]] == [3, 4]
    assert {run["cases"] for run in aggregate["runs"]} == {4}
    assert len({run["run_id"] for run in aggregate["runs"]}) == 2
    for run in aggregate["runs"]:
        assert set(run) == RUN_KEYS
        assert re.fullmatch(r"memory_quality_[0-9a-f]{32}", run["run_id"])
        assert run["passed"] == sum(case["passed"] for case in run["per_case"])
        assert {case["case_id"] for case in run["per_case"]} == {
            "en_en", "zh_zh", "en_zh", "zh_en",
        }
        for case in run["per_case"]:
            assert set(case) == CASE_KEYS
            assert case["passed"] == (not case["failed_checks"])
        for key in ("loaded_runtime_sha256", "raw_sha256", "index_sha256"):
            assert re.fullmatch(r"[0-9a-f]{64}", run[key])
    for key in (
        "model_sha256", "projector_sha256", "server_sha256",
        "native_config_sha256", "lineage_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", aggregate["route"][key])
    assert re.fullmatch(r"[0-9a-f]{40}", aggregate["route"]["checkpoint_revision"])

    published = json.dumps(aggregate, ensure_ascii=False)
    assert not re.search(r"(?:[A-Z]:\\|\\\\wsl|/home/|/tmp/|file://)", published)
    assert not re.search(r"\bID[0-9A-F]{12}\b", published)
    assert not any(field in published for field in (
        '"fixture_seed_hex"', '"prompt"', '"answer"', '"source_event_id"',
        '"distractor_event_id"', '"events"', '"payload"', '"memory_file"',
    ))
