# SPDX-License-Identifier: Apache-2.0
"""The public Agent report must not copy private index or retest fields."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.edge_agent.public_report import (
    _failure_code,
    _safe_label,
    _structured_raw_verified,
    _submission_mode,
    summarize_index,
    summarize_load_refusal,
    summarize_loader_log,
    summarize_navigation_retest,
    summarize_qwen_host_mapped_smoke,
)
from benchmarks.edge_agent.native_profile import (
    MODEL_PROMPT_IDENTITY_CAPTURE, STRUCTURED_READ_URL_MODE,
    STRUCTURED_READ_URL_SUITE_ID, _input_contract,
)
from benchmarks.edge_agent.paired_suite import FixtureSite, build_paired_cases


_HASH = "a" * 64
_SECRET = r"C:\Users\private\secret.txt"


def _structured_index(origin: str) -> dict:
    cases = build_paired_cases(origin, 10)["browser_text"]
    return {
        "conditions": {"suite_id": STRUCTURED_READ_URL_SUITE_ID},
        "submission_mode": STRUCTURED_READ_URL_MODE,
        "protocol": "smoke_incomplete",
        "fixture_origin": origin,
        "results": [{"task_class": "browser_text",
                     "submission_mode": STRUCTURED_READ_URL_MODE}],
        "case_input_contracts": {
            "browser_text": {
                length: [_input_contract(case, structured_read_url=True,
                                         fixture_origin=origin) for case in examples]
                for length, examples in cases.items()
            },
        },
        "case_prompt_sha256": {
            "browser_text": {
                length: [case.metadata["prompt_sha256"] for case in examples]
                for length, examples in cases.items()
            },
        },
    }


def test_structured_suite_requires_exact_mode_contract_and_full_prompt_capture():
    with FixtureSite() as site:
        index = _structured_index(site.origin)
        assert _submission_mode(index) == STRUCTURED_READ_URL_MODE
        index["results"][0]["protocol_compliant"] = True
        with pytest.raises(ValueError, match="backend model prompt identity"):
            _submission_mode(index)
        index["results"][0]["protocol_compliant"] = False
        index["protocol"] = "full_20x3_and_30m"
        with pytest.raises(ValueError, match="backend model prompt identity"):
            _submission_mode(index)
        index["model_prompt_identity_capture"] = MODEL_PROMPT_IDENTITY_CAPTURE
        assert _submission_mode(index) == STRUCTURED_READ_URL_MODE
        index["case_input_contracts"]["browser_text"]["short"][0]["explicit_read_url"] = _SECRET
        with pytest.raises(ValueError, match="input contracts"):
            _submission_mode(index)
        index = _structured_index("http://127.0.0.1:99999")
        with pytest.raises(ValueError, match="exact loopback fixture origin"):
            _submission_mode(index)


def test_structured_raw_prompt_identity_is_required_only_when_declared(tmp_path):
    with FixtureSite() as site:
        case = build_paired_cases(site.origin, 10)["browser_text"]["short"][0]
        proof = {"step": 0, "sha256": _HASH, "utf8_bytes": 123, "chars": 120}
        row = {
            "record_type": "request", "case": asdict(case),
            "fixture_setup": {"input_contract": _input_contract(
                case, structured_read_url=True, fixture_origin=site.origin)},
            "events": [{"kind": "model_prompt_identity",
                        "payload": {"first": proof, "model_steps": 1}}],
            "result": {"complete_agent_trace": True, "placement_evidence": {
                "first_model_prompt_identity": proof, "model_prompt_step_count": 1}},
        }
        raw = tmp_path / "samples.jsonl"
        raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert _structured_raw_verified(raw, site.origin,
                                        require_model_prompt_identity=True)
        row["events"] = []
        raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert not _structured_raw_verified(raw, site.origin,
                                            require_model_prompt_identity=True)
        assert _structured_raw_verified(raw, site.origin,
                                        require_model_prompt_identity=False)
        row["events"] = [{"kind": "model_prompt_identity", "payload": {
            "first": {**proof, "utf8_bytes": True}, "model_steps": 1}}]
        raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert not _structured_raw_verified(raw, site.origin,
                                            require_model_prompt_identity=True)


def test_failed_index_exports_only_known_fields(tmp_path: Path) -> None:
    index_path = tmp_path / ("native_" + "b" * 32) / "index.json"
    index_path.parent.mkdir()
    index = {
        "scope": "whole_agent_batch1_paired_local_fixture",
        "protocol": "smoke_incomplete",
        "source_config_sha256": _HASH,
        "lineage_sha256": _HASH,
        "source_config": _SECRET,
        "hardware": {
            "cpu": "AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD",
            "gpu_name": "NVIDIA GeForce RTX 5090 Laptop GPU",
            "host_ram_total_bytes": 67773509632,
            "host_ram_available_bytes": 27726852096,
            "vram_available_bytes": 19957633024,
            "os": "Windows-11-10.0.26200-SP0",
            "power_condition": "AC",
        },
        "conditions": {
            "os_version": "Windows-11-10.0.26200-SP0",
            "power_condition": "AC",
            "suite_id": "edge-agent-fixed-local-fixtures-v1",
            "environment_fingerprint": _HASH,
            "runtime_versions": {
                "python": "3.12.10",
                "vllm_omni": "0.29.0rc2.dev71",
                "vllm_omni_source": _SECRET,
            },
            "driver_versions": {"nvidia": "610.71"},
        },
        "artifact_provenance": {
            "public-route": {
                "checkpoint_revision": "unverified-local-gguf-origin",
                "lineage_verified": False,
                "model_sha256": _HASH,
                "server_sha256": _HASH,
                "precision": "Q4_K_M",
                "model_file": _SECRET,
            },
        },
        "results": [{
            "task_class": "browser_text",
            "route_id": "public-route",
            "status": "failed_or_blocked",
            "error": f"TargetClosedError: page closed at {_SECRET}",
            "raw_prompt": _SECRET,
        }],
    }
    index_path.write_text(json.dumps(index), encoding="utf-8")

    public = summarize_index(index_path)
    serialized = json.dumps(public)
    assert _SECRET not in serialized
    assert public["results"][0]["failure_code"] == "browser_target_closed"
    assert public["routes"]["public-route"]["checkpoint_revision"] is None
    runtime = public["conditions"]["runtime_versions"]
    assert runtime["vllm_omni_installed_distribution"] == "0.29.0rc2.dev71"
    assert runtime["vllm_omni_loaded_source_version"] is None
    assert runtime["vllm_omni_imported_source_sha256"] is None
    assert runtime["agent_runtime_identity_sha256"] is None

    index["results"][0]["error"] = (
        "ResourceUnavailable: profile route public-route refused before model load: "
        "host_ram: declared demand 22000000000 bytes exceeds the native "
        "controller ceiling 20758487040 bytes"
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    public = summarize_index(index_path)
    refusal = public["results"][0]
    assert refusal["failure_code"] == "preload_physical_pool_capacity_refusal"
    assert refusal["admission"] == {
        "physical_pool": "host_ram", "declared_demand_bytes": 22000000000,
        "native_controller_ceiling_bytes": 20758487040,
        "model_load_attempted": False, "agent_request_attempted": False,
    }
    index["results"][0]["error"] += f" at {_SECRET}"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    public = summarize_index(index_path)
    assert public["results"][0]["failure_code"] == "unclassified_failure"
    assert "admission" not in public["results"][0]
    assert _SECRET not in json.dumps(public)
    index["results"][0]["error"] = (
        "ResourceUnavailable: profile route public-route refused before model load: "
        "host_ram: declared demand 100 bytes exceeds the native controller "
        "ceiling 200 bytes"
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    public = summarize_index(index_path)
    assert public["results"][0]["failure_code"] == "unclassified_failure"
    assert "admission" not in public["results"][0]
    index["results"][0]["task_class"] = "basic"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    assert summarize_index(index_path)["results"][0]["task_class"] == "basic"


def test_new_profile_distinguishes_loaded_source_from_distribution(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / ("native_" + "c" * 32) / "index.json"
    index_path.parent.mkdir()
    index_path.write_text(json.dumps({
        "scope": "whole_agent_batch1_paired_local_fixture",
        "protocol": "smoke_incomplete",
        "source_config_sha256": _HASH,
        "lineage_sha256": _HASH,
        "hardware": {
            "cpu": "Test CPU", "gpu_name": None,
            "host_ram_total_bytes": 1000,
            "host_ram_available_bytes": 500,
            "vram_available_bytes": None,
            "os": "Test OS", "power_condition": "AC",
        },
        "conditions": {
            "os_version": "Test OS", "power_condition": "AC",
            "suite_id": "edge-agent-fixed-local-fixtures-v1",
            "environment_fingerprint": _HASH,
            "driver_versions": {},
            "runtime_versions": {
                "vllm_omni": "0.29.0rc2.dev72",
                "vllm_omni_installed_distribution": "0.29.0rc2.dev71",
                "vllm_omni_imported_source_sha256": _HASH,
                "agent_runtime_identity_sha256": "b" * 64,
                "vllm_omni_source": _SECRET,
            },
        },
        "artifact_provenance": {}, "results": [],
    }), encoding="utf-8")
    public = summarize_index(index_path)
    runtime = public["conditions"]["runtime_versions"]
    assert runtime["vllm_omni_loaded_source_version"] == "0.29.0rc2.dev72"
    assert runtime["vllm_omni_installed_distribution"] == "0.29.0rc2.dev71"
    assert runtime["vllm_omni_imported_source_sha256"] == _HASH
    assert runtime["agent_runtime_identity_sha256"] == "b" * 64
    assert _SECRET not in json.dumps(public)


def test_public_labels_reject_local_paths() -> None:
    with pytest.raises(ValueError, match="safe public label"):
        _safe_label(_SECRET, "hardware")


def test_foreground_fixture_failure_has_narrow_public_code() -> None:
    assert _failure_code(
        "RuntimeError: Edge fixture window could not be verified in foreground"
    ) == "fixture_foreground_unverified"
    assert _failure_code(
        f"RuntimeError: Edge fixture window failed at {_SECRET}"
    ) == "unclassified_failure"
    assert _failure_code("ValueError: browser_read requires exactly: []") == (
        "browser_read_tool_arguments_invalid"
    )


def test_single_request_retest_is_allowlisted(tmp_path: Path) -> None:
    path = tmp_path / "retest.json"
    path.write_text(json.dumps({
        "scope": "one_request_navigation_fix_smoke_unqualified",
        "config_sha256": _HASH,
        "case_prompt_sha256": _HASH,
        "duration_s": 62.5,
        "final_answer_present": True,
        "reference_matched": True,
        "tool_sequence": ["browser_open", "browser_read"],
        "navigation_auto_read_recorded": True,
        "seq_ordered": True,
        "raw_payload": _SECRET,
    }), encoding="utf-8")
    public = summarize_navigation_retest(path)
    assert _SECRET not in json.dumps(public)
    assert public["release_qualified"] is False


def test_vision_retest_has_distinct_tool_scope(tmp_path: Path) -> None:
    path = tmp_path / "vision_retest.json"
    path.write_text(json.dumps({
        "scope": "one_request_navigation_fix_smoke_unqualified",
        "task_class": "browser_vision",
        "config_sha256": _HASH,
        "case_prompt_sha256": _HASH,
        "duration_s": 81.5,
        "final_answer_present": True,
        "reference_matched": True,
        "tool_sequence": ["browser_open", "browser_screenshot"],
        "navigation_auto_observation_recorded": True,
        "seq_ordered": True,
        "raw_payload": _SECRET,
    }), encoding="utf-8")
    public = summarize_navigation_retest(path)
    assert public["task_class"] == "browser_vision"
    assert public["navigation_auto_observation_recorded"] is True
    assert _SECRET not in json.dumps(public)


def test_load_refusal_exports_only_failure_code(tmp_path: Path) -> None:
    path = tmp_path / "qwen_budget_smoke.jsonl"
    rows = [
        {"record_type": "manifest", "scope": "one_complete_agent_request_smoke",
         "batch_size": 1, "concurrency": 1, "config_sha256": _HASH,
         "config_snapshot": _SECRET},
        {"kind": "user_observation", "payload": {"prompt": _SECRET}},
        {"kind": "error", "payload": {
            "type": "ResourceUnavailable",
            "message": "actual hybrid model buffers exceed the declared per-pool weight budgets",
            "local_log": _SECRET,
        }},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8")
    public = summarize_load_refusal(path)
    assert _SECRET not in json.dumps(public)
    assert public["failure_code"] == "declared_weight_budget_exceeded"


def test_loader_log_rejects_non_allowlisted_text(tmp_path: Path) -> None:
    path = tmp_path / "loader.log"
    path.write_text("\n".join((
        "load_tensors: CPU model buffer size = 16499.72 MiB",
        "load_tensors: Vulkan0 model buffer size = 1921.34 MiB",
    )) + "\n", encoding="utf-8")
    assert summarize_loader_log(path)["cpu_model_buffer_mib"] == 16499.72
    path.write_text(path.read_text(encoding="utf-8") + _SECRET + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="non-allowlisted"):
        summarize_loader_log(path)


def test_qwen_host_mapped_smoke_exports_only_narrow_outcome(tmp_path: Path) -> None:
    path = tmp_path / "host_mapped_smoke.jsonl"
    events = [
        {"kind": "user_observation", "payload": {"prompt": _SECRET}},
        {"kind": "route", "payload": {
            "route_id": "qwen-host-mapped", "model": "Qwen3.6-35B-A3B UD-IQ4_XS",
            "backend": "external.llamacpp.multimodal.v1",
            "actual_placement": "Vulkan_Host+Vulkan0", "experimental": True,
        }},
        {"kind": "text_delta", "payload": {"text": "ready"}},
        {"kind": "model_metrics", "payload": {}},
        {"kind": "final", "payload": {
            "answer": "ready", "experimental": True, "streamed": True,
        }},
    ]
    for seq, event in enumerate(events, 1):
        event["seq"] = seq
    manifest = {"record_type": "manifest",
                "scope": "one_complete_agent_request_smoke",
                "batch_size": 1, "concurrency": 1,
                "config_sha256": _HASH, "model_path": _SECRET}
    path.write_text("\n".join(json.dumps(row) for row in [manifest, *events]) + "\n",
                    encoding="utf-8")
    public = summarize_qwen_host_mapped_smoke(path)
    assert public["completed_model_request"] is True
    assert public["expert_storage_and_compute_verified"] is False
    assert _SECRET not in json.dumps(public)
