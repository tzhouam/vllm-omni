# SPDX-License-Identifier: Apache-2.0
"""The unsigned assembler must not mistake a diagnostic for a qualified gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.edge_agent.experiments.assemble_gemma_gate_evidence import (
    EvidenceBlocked, PROFILE_BINDING_SCHEMA, _ordered_cancel_events, _same_route, assemble,
    identity_binding_gaps, memory_observations, quality_observations,
)
from benchmarks.edge_agent.experiments.native_memory_admission import compare_samples


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(tmp_path: Path) -> dict:
    route = {
        "route_id": "gemma-test", "artifact_id": "artifact",
        "model_sha256": "a" * 64, "server_sha256": "b" * 64,
        "mmproj_sha256": "c" * 64, "placement": "cpu+Vulkan0",
        "memory_demands": {"host_ram": 50, "vram": 25},
        "log_file": "original.log",
    }
    config = _write(tmp_path / "source.json", {"routes": [route]})
    index = _write(tmp_path / "index.json", {})
    hardware = _hardware()
    stable = {key: hardware[key] for key in ("os", "machine", "cpu", "gpu_name", "gpu_driver")}
    stable["loaded_agent_runtime_sha256"] = "0" * 64
    fingerprint = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    return {
        "route": route,
        "index_path": index,
        "index": {"source_config_sha256": _sha(config), "hardware": hardware},
        "identity": {
            "route_id": "gemma-test", "artifact_sha256": "a" * 64,
            "placement": "cpu+Vulkan0", "power_condition": "AC",
            "profile_raw_sha256": "d" * 64,
            "environment_fingerprint": fingerprint,
        },
        "summary": {"conditions": {
            "hardware_id": "test CPU | test GPU | RAM 400 bytes",
            "os_version": "Windows test", "driver_versions": {"nvidia": "test-driver"},
            "runtime_versions": {
                "vllm_omni_imported_source_sha256": "f" * 64,
                "agent_runtime_identity_sha256": "0" * 64,
            },
        }},
    }


def _hardware() -> dict:
    return {
        "cpu": "test CPU", "machine": "AMD64", "gpu_name": "test GPU",
        "host_ram_total_bytes": 400,
        "os": "Windows test", "gpu_driver": "test-driver", "power_condition": "AC",
        "host_ram_available_bytes": 100, "vram_available_bytes": 50,
    }


def _memory_record(tmp_path: Path, ctx: dict) -> Path:
    route = ctx["route"]
    probe_route = dict(route, log_file="private.log")
    probe_config = _write(tmp_path / "probe.json", {"routes": [probe_route]})
    snapshot = {"resident_route": route["route_id"], "ledger": {
        "owners": [route["route_id"]], "quarantined": [],
        "reserved": route["memory_demands"],
    }}
    released = {"resident_route": None, "ledger": {
        "owners": [], "quarantined": [],
        "reserved": {"host_ram": 0, "vram": 0},
    }}
    samples = [
        {"record_type": "sample", "phase": "pre_load", "monotonic_ns": 10,
         "values": {"ram_used_bytes": 100, "vram_used_bytes": 10}},
        {"record_type": "sample", "phase": "cold_load", "monotonic_ns": 10,
         "values": {"ram_used_bytes": 120, "vram_used_bytes": 20}},
        {"record_type": "sample", "phase": "complete_request", "monotonic_ns": 11,
         "values": {"ram_used_bytes": 130, "vram_used_bytes": 30}},
    ]
    rows = [{
        "record_type": "manifest", "scope": "independent_memory_admission_probe",
        "batch_size": 1, "concurrency": 1, "route_id": route["route_id"],
        "source_config_sha256": ctx["index"]["source_config_sha256"],
        "probe_config": str(probe_config), "probe_config_sha256": _sha(probe_config),
        "model_sha256": route["model_sha256"],
        "server_sha256": route["server_sha256"],
        "mmproj_sha256": route["mmproj_sha256"],
    }, {"record_type": "hardware", "hardware": _hardware()}]
    for pool, ceiling in (("host_ram", 100), ("vram", 50)):
        rows.append({
            "record_type": "over_ceiling_refusal", "pool": pool,
            "ceiling_bytes": ceiling, "declared_bytes": ceiling + 1,
            "physical_allocation_attempted": False,
            "refusal": f"{pool} over ceiling", "ledger_after_refusal": released["ledger"],
        })
    rows.extend([
        {"record_type": "pre_load_admission", "admitted": True},
        {"record_type": "loaded_plan", "plan": {
            "requested_device": route["placement"],
            "reserved_bytes": route["memory_demands"]}},
        {"record_type": "host_ledger_after_load", "snapshot": snapshot},
        {"record_type": "host_ledger_after_request", "snapshot": snapshot},
        {"record_type": "host_ledger_after_release", "snapshot": released},
        *samples,
        {"record_type": "sampled_comparison",
         "comparison": compare_samples(samples, route["memory_demands"])},
        {"record_type": "outcome", "result": "one_complete_agent_turn_measured",
         "error": None, "sample_count": len(samples),
         "event_kinds": ["user_observation", "route", "model_metrics", "final"],
         "answer_sha256": "1" * 64},
    ])
    record = tmp_path / "memory.jsonl"
    record.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return record


def test_old_probe_is_diagnostic_even_when_route_and_hardware_match(tmp_path):
    ctx = _context(tmp_path)
    manifest = {"route_id": ctx["identity"]["route_id"]}
    assert identity_binding_gaps(manifest, _hardware(), ctx) == ["profile_binding"]
    manifest["profile_binding"] = {
        "schema": PROFILE_BINDING_SCHEMA,
        "profile_index_sha256": _sha(ctx["index_path"]),
        "profile_raw_sha256": "d" * 64,
        "source_config_sha256": ctx["index"]["source_config_sha256"],
        "route_id": "gemma-test",
        "environment_fingerprint": ctx["identity"]["environment_fingerprint"],
        "imported_omni_source_sha256": "f" * 64,
        "loaded_runtime_sha256": "0" * 64,
        "hardware_identity": {
            key: _hardware()[key] for key in
            ("os", "machine", "cpu", "host_ram_total_bytes", "gpu_name", "gpu_driver",
             "power_condition")},
    }
    assert identity_binding_gaps(manifest, _hardware(), ctx) == []
    changed = dict(_hardware(), gpu_driver="other-driver")
    assert identity_binding_gaps(manifest, changed, ctx) == [
        "hardware_identity", "profile_hardware_identity", "hardware_os_driver_power",
        "recomputed_environment_fingerprint",
    ]
    manifest["profile_binding"]["loaded_runtime_sha256"] = "1" * 64
    assert "loaded_runtime_sha256" in identity_binding_gaps(manifest, _hardware(), ctx)


def test_memory_reads_raw_samples_including_equal_clock_ticks(tmp_path):
    ctx = _context(tmp_path)
    path = _memory_record(tmp_path, ctx)
    observations = memory_observations(path, ctx)
    assert observations["route_incremental_peak_bytes"] == {"host_ram": 30, "vram": 20}
    assert observations["measurement_scope"].startswith("sampled_whole_host")
    assert "outcome" not in observations


def test_memory_rejects_a_stored_peak_not_recomputed_from_samples(tmp_path):
    ctx = _context(tmp_path)
    path = _memory_record(tmp_path, ctx)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    next(row for row in rows if row.get("record_type") == "sampled_comparison")[
        "comparison"]["host_ram"]["sampled_global_incremental_peak_bytes"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(EvidenceBlocked, match="raw samples"):
        memory_observations(path, ctx)


def test_route_comparison_does_not_waive_budget_changes():
    original = {"model_sha256": "a", "memory_demands": {"vram": 20},
                "log_file": "original.log"}
    assert _same_route(dict(original, log_file="private.log"), original)
    assert not _same_route({**original, "memory_demands": {"vram": 19}}, original)


def test_cancel_raw_order_must_not_interleave_requests():
    rows = [
        {"request_id": "first", "epoch": 1, "seq": 1, "kind": "user_observation"},
        {"request_id": "first", "epoch": 1, "seq": 2, "kind": "cancelled"},
        {"request_id": "second", "epoch": 2, "seq": 1, "kind": "user_observation"},
        {"request_id": "second", "epoch": 2, "seq": 2, "kind": "final"},
    ]
    _ordered_cancel_events(rows)
    with pytest.raises(EvidenceBlocked, match="interleaved"):
        _ordered_cancel_events([rows[0], rows[2], rows[1], rows[3]])
    with pytest.raises(EvidenceBlocked, match="duplicated or out of order"):
        _ordered_cancel_events([rows[0], rows[1], rows[2], rows[2]])


def test_unbound_observations_stay_diagnostic_not_gate_source(tmp_path, monkeypatch):
    import benchmarks.edge_agent.experiments.assemble_gemma_gate_evidence as assembler

    ctx = _context(tmp_path)
    index = ctx["index_path"]
    summary = _write(tmp_path / "summary.json", {})
    raw = _write(tmp_path / "samples.jsonl", {})
    source = _write(tmp_path / "memory-probe.jsonl", {})
    ctx.update({
        "index_path": index, "summary_path": summary,
        "audit": SimpleNamespace(raw_jsonl=raw, measured_attempts=60,
                                 measured_successes=60),
    })
    monkeypatch.setattr(assembler, "profile_context", lambda _: ctx)
    monkeypatch.setattr(assembler, "memory_observations", lambda *_: {"measured": 1})
    monkeypatch.setattr(assembler, "source_binding_gaps",
                        lambda gate, *_: ["profile_raw_sha256"] if gate == "memory_admission" else [])
    monkeypatch.setattr(assembler, "imported_omni_source_sha256", lambda: "f" * 64)
    monkeypatch.setattr(assembler, "loaded_runtime_sha256", lambda: "0" * 64)
    report_path = assemble(profile_index=index, output_root=tmp_path / "private",
                           memory_probe=source)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gate_drafts"] == {}
    assert list(report["diagnostics"]) == ["memory_admission"]
    diagnostic = json.loads(Path(report["diagnostics"]["memory_admission"]["path"])
                            .read_text(encoding="utf-8"))
    assert diagnostic["not_gate_evidence"] is True
    assert diagnostic["record_type"] != "memory_admission_evidence_v1"
    assert report["router_qualification_created"] is False
    assert all(item.get("outcome") != "pass" for item in (report, diagnostic))


def test_quality_rechecks_answer_hash_and_reference_not_only_pass_flags(tmp_path, monkeypatch):
    import benchmarks.edge_agent.experiments.assemble_gemma_gate_evidence as assembler

    ctx = _context(tmp_path)
    ctx["identity"]["suite_id"] = "timing-suite"
    monkeypatch.setattr(assembler, "loaded_runtime_sha256", lambda: "current-runtime")
    folder = tmp_path / "quality"
    folder.mkdir()
    cases = []
    for language in ("en-US", "zh-CN"):
        reference = "answer-" + language
        digest = hashlib.sha256(reference.encode()).hexdigest()
        cases.append({
            "record_type": "quality_case", "case_id": language,
            "language": language, "source_event_id": "source-" + language,
            "distractor_event_id": "distractor-" + language,
            "reference": reference, "reference_sha256": digest,
            "recall": {"answer": reference, "answer_sha256": digest,
                       "trace_ok": True,
                       "final_derived_from": ["source-" + language,
                                              "distractor-" + language]},
            "before_retrieved_event_ids": ["source-" + language,
                                           "distractor-" + language],
            "after_delete": {"trace_ok": True, "answer": "UNKNOWN"},
            "deletion": {"after_retrieved_event_ids": ["distractor-" + language],
                         "source_absent": True, "distractor_retained": True,
                         "derived_final_absent": True, "answer_index_absent": True},
            "checks": {"passed": True, "reference_answer_exact": True},
        })
    manifest = {"record_type": "manifest", "batch_size": 1, "concurrency": 1,
                "route": {"route_id": ctx["identity"]["route_id"],
                          "artifact_sha256": ctx["identity"]["artifact_sha256"]},
                "native_config_sha256": ctx["index"]["source_config_sha256"],
                "suite_id": "independent-quality", "loaded_runtime_sha256": "current-runtime"}
    summary = {"record_type": "summary", "cases": 2, "passed": 2,
               "hardware_by_phase": {}}
    raw = folder / "raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in [manifest, *cases, summary]),
                   encoding="utf-8")
    index = _write(folder / "index.json", {
        "raw_sha256": _sha(raw), "status": "completed",
        "review_status": "unreviewed_private_experiment", "qualifies_default": False,
        "summary": {"cases": 2, "passed": 2},
    })
    observations = quality_observations(index, ctx)
    assert len(observations["cases"]) == 2
    assert {case["language"] for case in observations["cases"]} == {"en-US", "zh-CN"}
    assert "answer" not in observations["cases"][0]
    cases[0]["recall"]["answer"] = "wrong answer"
    raw.write_text("".join(json.dumps(row) + "\n" for row in [manifest, *cases, summary]),
                   encoding="utf-8")
    _write(index, {"raw_sha256": _sha(raw), "status": "completed",
                   "review_status": "unreviewed_private_experiment",
                   "qualifies_default": False, "summary": {"cases": 2, "passed": 2}})
    with pytest.raises(EvidenceBlocked, match="raw answer"):
        quality_observations(index, ctx)
