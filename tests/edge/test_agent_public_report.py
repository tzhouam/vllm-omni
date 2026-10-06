# SPDX-License-Identifier: Apache-2.0
"""The public Agent report must not copy private index or retest fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.edge_agent.public_report import (
    _failure_code,
    _safe_label,
    summarize_index,
    summarize_load_refusal,
    summarize_loader_log,
    summarize_navigation_retest,
    summarize_qwen_host_mapped_smoke,
)


_HASH = "a" * 64
_SECRET = r"C:\Users\private\secret.txt"


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
