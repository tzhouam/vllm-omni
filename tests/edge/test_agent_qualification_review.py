# SPDX-License-Identifier: Apache-2.0
"""A reviewed native route cannot be promoted from claims or short smoke data."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from benchmarks.edge_agent import profile
from benchmarks.edge_agent.evidence import (
    GATE_SCHEMA, PROMOTION_SCHEMA, REQUIRED_GATES, audit_summary,
    bundle_signing_bytes,
    load_reviewed_qualification,
)
from benchmarks.edge_agent.native_profile import SUITE_ID
from benchmarks.edge_agent.paired_suite import build_paired_cases, evaluate_case
from benchmarks.edge_agent.profile import (
    AgentCase, AgentRunResult, Preparation, ProfileConditions, ProfileConfig,
    ProfileRoute, _telemetry_summary, run_profile,
)
from vllm_omni.edge.agent.native_app import _qualifications
from vllm_omni.edge.agent.qualification import (
    _verify_gate_observations, verify_fixed_suite_cases,
)
from vllm_omni.edge.agent.runtime_identity import (
    imported_omni_source_sha256, loaded_runtime_sha256,
)


def test_wheel_package_discovery_includes_verifier_without_benchmark_data():
    from setuptools import find_namespace_packages

    root = Path(__file__).resolve().parents[2]
    selected = set(find_namespace_packages(
        where=str(root), include=["vllm_omni*", "omni_stage_contracts*"],
    ))
    assert "vllm_omni.edge.agent" in selected
    assert "benchmarks.edge_agent" not in selected
    verifier = root / "vllm_omni" / "edge" / "agent" / "qualification.py"
    assert verifier.is_file()
    assert "benchmarks.edge_agent" not in verifier.read_text(encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _ref(path):
    return {"path": str(path), "sha256": _sha(path)}


def _qualified_fixture(tmp_path, monkeypatch, *, measured_successes=60):
    route = ProfileRoute(
        route_id="reviewed-cpu", model_id="test-model", artifact_id="artifact",
        checkpoint_revision="0123456789abcdef0123456789abcdef01234567",
        artifact_sha256="a" * 64, precision="Q4_K_M",
        backend="external.llamacpp.text.v1", expected_placement="cpu",
    )
    native_route = {
        "route_id": route.route_id, "model": route.model_id,
        "artifact_id": route.artifact_id, "model_sha256": route.artifact_sha256,
        "server_sha256": "b" * 64, "placement": "cpu",
        "model_file": "model.gguf", "server_bin": "llama-server.exe",
        "memory_demands": {"host_ram": 100}, "memory_overhead_bytes": 20,
    }
    config_path = _write(tmp_path / "native.json", {"routes": [native_route]})
    conditions = ProfileConditions(
        hardware_id="Windows test host", os_version="Windows test",
        driver_versions={}, runtime_versions={
            "llama_server_sha256": "b" * 64,
            "vllm_omni_imported_source_sha256": imported_omni_source_sha256(),
            "agent_runtime_identity_sha256": loaded_runtime_sha256(),
        },
        power_condition="AC", suite_id=SUITE_ID,
        environment_fingerprint="fixed-test-host",
    )
    cases = build_paired_cases("http://127.0.0.1:12345", 10)["code_tools"]

    measured_seen = 0

    async def fake_request(*, run_id, route, case, phase, repetition, **_):
        nonlocal measured_seen
        if phase == "measured":
            measured_seen += 1
        answer = ("14" if phase != "measured" or
                  measured_seen <= measured_successes else "wrong")
        agent_events = [
            {"seq": 1, "request_id": "r", "epoch": 1,
             "kind": "user_observation", "payload": {"text": case.prompt}},
            {"seq": 2, "request_id": "r", "epoch": 1, "kind": "route",
             "payload": {"route_id": route.route_id, "model": route.model_id,
                         "artifact_id": route.artifact_id,
                         "backend": route.backend,
                         "actual_placement": route.expected_placement}},
            {"seq": 3, "request_id": "r", "epoch": 1,
             "kind": "model_metrics", "payload": {}},
            {"seq": 4, "request_id": "r", "epoch": 1,
             "kind": "final", "payload": {"answer": answer}},
        ]
        result = AgentRunResult(
            final_answer=answer, complete_agent_trace=True,
            model_id=route.model_id, artifact_id=route.artifact_id,
            actual_placement="cpu", backend=route.backend,
            placement_evidence={"runtime_log": "pinned"},
        )
        samples = [{"ram_used_bytes": 10}]
        return {
            "record_type": "request", "run_id": run_id, "phase": phase,
            "repetition": repetition, "route_id": route.route_id,
            "case": asdict(case), "batch_size": 1, "concurrency": 1,
            "ttft_s": .2, "answer_latency_s": 1800.0 if phase == "endurance" else 1.0,
            "events": [
                *({"kind": "agent_event", "offset_s": 0.0, "payload": event}
                  for event in agent_events),
                {"kind": "assistant_text_delta", "offset_s": .2, "payload": answer},
            ],
            "result": asdict(result), "evaluation": asdict(evaluate_case(case, result)),
            "e2e_complete": True, "placement_matches": True,
            "telemetry": {"raw_samples": samples, "errors": [],
                          "summary": _telemetry_summary(samples)},
            "error": None,
        }

    monkeypatch.setattr(profile, "_one_request", fake_request)
    # The synthetic endurance row represents 1800 s of active time. Model
    # execution is mocked, so supply the matching wall-clock interval too.
    ticks = iter((0.0, 1800.0))
    monkeypatch.setattr(profile.time, "perf_counter", lambda: next(ticks))
    summary = asyncio.run(run_profile(
        routes=[route], cases_by_length=cases,
        runner=lambda *_: None, evaluator=evaluate_case,
        conditions=conditions, output_dir=tmp_path,
        config=ProfileConfig(warmups_per_length=1, measured_per_length=20,
                             endurance_seconds=1800),
        prepare=lambda route: Preparation(True, route.artifact_id, "cpu", {}),
        before_request=lambda *_: {"private_memory_reset": True},
    ))
    summary_path = summary.run_directory / "summary.json"
    index_path = _write(tmp_path / "index.json", {
        "automatic_qualification_export": False,
        "source_config": str(config_path), "source_config_sha256": _sha(config_path),
        "artifact_provenance": {route.route_id: {
            "lineage_verified": True, "checkpoint_revision": route.checkpoint_revision,
            "model_sha256": route.artifact_sha256, "server_sha256": "b" * 64,
            "mmproj_sha256": None,
        }},
        "results": [{"route_id": route.route_id, "task_class": "code_tools",
                     "raw_sha256": summary.raw_sha256,
                     "summary": str(summary_path)}],
    })
    identity = {
        "route_id": route.route_id, "artifact_id": route.artifact_id,
        "artifact_sha256": route.artifact_sha256,
        "checkpoint_revision": route.checkpoint_revision,
        "backend": route.backend, "placement": route.expected_placement,
        "suite_id": conditions.suite_id,
        "environment_fingerprint": conditions.environment_fingerprint,
        "power_condition": conditions.power_condition,
        "profile_raw_sha256": summary.raw_sha256,
    }
    startup_log = tmp_path / "startup.log"
    startup_log.write_text("offloaded 0/29 layers to GPU\nmodel loaded\n")
    observations = {
        "memory_admission": {
            "admitted_demand_bytes": {"host_ram": 100},
            "pool_used_baseline_bytes": {"host_ram": 1000},
            "pool_used_peak_bytes": {"host_ram": 1090},
            "route_incremental_peak_bytes": {"host_ram": 90},
            "available_ceiling_bytes": {"host_ram": 200},
            "refused_demand_bytes": {"host_ram": 201},
            "refusal_reason": "host RAM ceiling",
        },
        "cancel_recovery": {"events": [
            {"request_id": "cancel", "epoch": 1, "seq": 1, "kind": "cancelled"},
            {"request_id": "cancel", "epoch": 1, "seq": 2, "kind": "state_released",
             "payload": {
                 "request_id": "cancel", "release_mode": "worker_shutdown",
                 "worker_pid_before": 1234, "worker_exit_code": 0,
                 "backend_request_state_verified": True,
                 "graph_gate_released": True, "tool_request_finished": True,
                 "worker_exit_confirmed": True, "stage_ledger_empty": True,
                 "host_claim_released": True, "host_ledger_empty": True,
             }},
            {"request_id": "recover", "epoch": 2, "seq": 1, "kind": "final"},
        ]},
        "runtime_placement": {
            "reported_placement": "cpu", "artifact_sha256": route.artifact_sha256,
            "startup_log": _ref(startup_log),
        },
        "checkpoint_lineage": {
            "checkpoint_revision": route.checkpoint_revision,
            "artifact_sha256": route.artifact_sha256,
            "license": "test", "source_repo": "https://example.invalid/test-model",
        },
        "reference_quality": {
            "task_class": "code_tools", "reference_suite_id": "independent-v1",
            "cases": [
                {"language": language, "task_class": "code_tools",
                 "passed": True, "comparison": "exact_sha256",
                 "reference_sha256": "c" * 64,
                 "answer_sha256": "c" * 64}
                for language in ("en-US", "zh-CN")
            ],
        },
    }
    gates = {}
    sources = {}
    for gate in REQUIRED_GATES:
        source = _write(tmp_path / f"{gate}.raw.json", {
            "record_type": f"{gate}_evidence_v1", "identity": identity,
            "observations": observations[gate],
        })
        sources[gate] = source
        receipt = _write(tmp_path / f"{gate}.review.json", {
            "schema": GATE_SCHEMA, "gate": gate, "identity": identity,
            "outcome": "pass", "source": _ref(source),
        })
        gates[gate] = _ref(receipt)
    key = Ed25519PrivateKey.generate()
    trusted_keys = {"reviewer-1": base64.b64encode(key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )).decode()}
    bundle = {
        "schema": PROMOTION_SCHEMA, "identity": identity,
        "profile_summary": _ref(summary_path), "profile_index": _ref(index_path),
        "gates": gates,
        "review": {"key_id": "reviewer-1", "reviewer": "independent reviewer",
                   "reviewed_at": "2026-10-05T00:00:00Z"},
    }

    def sign():
        bundle["review"]["signature_ed25519"] = base64.b64encode(
            key.sign(bundle_signing_bytes(bundle)),
        ).decode()
        _write(tmp_path / "promotion.json", bundle)
        return tmp_path / "promotion.json"

    return sign, summary_path, sources, native_route, trusted_keys, bundle


def test_native_rejects_claim_only_qualification_json(tmp_path):
    path = _write(tmp_path / "claims.json", [{"route_id": "x", "memory_pass": True}])
    with pytest.raises(ValueError, match="claim-only"):
        _qualifications({"qualification_file": str(path), "routes": []})


def test_host_mapped_startup_override_cannot_pass_runtime_placement_gate(tmp_path):
    startup = tmp_path / "host_startup.log"
    startup.write_text(
        "using device Vulkan0 (RTX 5090 Laptop)\n"
        "offloaded 41/41 layers to GPU\n"
        "model loaded\n",
        encoding="utf-8",
    )
    identity = {"placement": "Vulkan_Host+Vulkan0", "artifact_sha256": "a" * 64}
    observations = {
        "reported_placement": identity["placement"],
        "artifact_sha256": identity["artifact_sha256"],
    }
    with pytest.raises(ValueError, match="cannot be release-qualified from startup logs"):
        _verify_gate_observations("runtime_placement", observations, identity, tmp_path)


def test_cpu_expert_placement_receipt_requires_every_override(tmp_path):
    startup = tmp_path / "cpu-experts.log"
    identity = {"placement": "cpu+Vulkan0", "artifact_sha256": "a" * 64}
    observations = {
        "reported_placement": identity["placement"],
        "artifact_sha256": identity["artifact_sha256"],
    }
    lines = ["offloaded 3/3 layers to GPU", "model loaded"]
    lines += [f"tensor blk.{layer}.ffn_{tensor}_exps.weight (q4) buffer type overridden to CPU"
              for layer in range(2) for tensor in ("up", "down", "gate")]
    startup.write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")
    observations["startup_log"] = _ref(startup)
    with pytest.raises(ValueError, match="exact CPU expert"):
        _verify_gate_observations("runtime_placement", observations, identity,
                                  tmp_path, {"cpu_moe_layers": 2})
    startup.write_text("\n".join(lines) + "\n", encoding="utf-8")
    observations["startup_log"] = _ref(startup)
    _verify_gate_observations("runtime_placement", observations, identity,
                              tmp_path, {"cpu_moe_layers": 2})


def test_cancel_receipt_rejects_release_kind_without_worker_proof(tmp_path):
    observations = {"events": [
        {"request_id": "cancel", "epoch": 1, "seq": 1, "kind": "cancelled"},
        {"request_id": "cancel", "epoch": 1, "seq": 2, "kind": "state_released"},
        {"request_id": "recovered", "epoch": 2, "seq": 1, "kind": "final"},
    ]}
    with pytest.raises(ValueError, match="verified worker shutdown"):
        _verify_gate_observations("cancel_recovery", observations, {}, tmp_path)


def test_reference_gate_rejects_inconsistent_signed_quality_claim(tmp_path):
    cases = [
        {"language": language, "task_class": "code_tools", "passed": True,
         "comparison": "exact_sha256", "reference_sha256": "c" * 64,
         "answer_sha256": "c" * 64}
        for language in ("en-US", "zh-CN")
    ]
    observations = {"task_class": "code_tools",
                    "reference_suite_id": "independent-v1", "cases": cases}
    identity = {"suite_id": SUITE_ID}
    _verify_gate_observations("reference_quality", observations, identity, tmp_path)
    cases[1]["answer_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="reference quality evidence is incomplete"):
        _verify_gate_observations("reference_quality", observations, identity, tmp_path)
    cases[1]["answer_sha256"] = "C" * 64
    with pytest.raises(ValueError, match="reference quality evidence is incomplete"):
        _verify_gate_observations("reference_quality", observations, identity, tmp_path)


def test_signed_profile_and_separate_gates_are_required(tmp_path, monkeypatch):
    sign, _, _, route, keys, _ = _qualified_fixture(tmp_path, monkeypatch)
    bundle = sign()
    qualification = load_reviewed_qualification(
        bundle, trusted_keys=keys, native_config={"routes": [route]},
    )
    assert qualification.qualified
    assert qualification.attempts == 60
    assert qualification.sustained_seconds == 1800
    assert _qualifications({"qualification_bundles": [str(bundle)],
                            "trusted_review_keys": keys, "routes": [route]}) == [qualification]
    assert _qualifications({"qualification_bundles": [bundle.name],
                            "trusted_review_keys": keys, "routes": [route]},
                           config_dir=tmp_path) == [qualification]
    with pytest.raises(ValueError, match="signature"):
        load_reviewed_qualification(bundle, trusted_keys={"reviewer-1": base64.b64encode(b"0" * 32).decode()},
                                    native_config={"routes": [route]})
    changed_route = {**route, "memory_demands": {"host_ram": 200}}
    with pytest.raises(ValueError, match="current native Agent configuration differs"):
        load_reviewed_qualification(bundle, trusted_keys=keys,
                                    native_config={"routes": [changed_route]})
    with pytest.raises(ValueError, match="current native Agent configuration differs"):
        load_reviewed_qualification(bundle, trusted_keys=keys,
                                    native_config={"routes": [route],
                                                   "limits": {"max_model_steps": 9}})


def test_fixed_suite_release_check_binds_reference_prompt_and_observation(
        tmp_path, monkeypatch):
    _, summary, _, _, _, _ = _qualified_fixture(tmp_path, monkeypatch)
    audit = audit_summary(summary)
    verify_fixed_suite_cases(audit)
    original = [json.loads(line) for line in audit.raw_jsonl.read_text(encoding="utf-8").splitlines()]
    first_request = next(i for i, row in enumerate(original)
                         if row.get("record_type") == "request")
    for mutation in ("reference", "prompt", "observed_prompt", "schedule"):
        rows = json.loads(json.dumps(original))
        row = rows[first_request]
        if mutation == "reference":
            row["case"]["reference"] = "incorrect"
        elif mutation == "prompt":
            row["case"]["prompt"] = "a substituted benchmark task"
        elif mutation == "observed_prompt":
            next(event for event in row["events"]
                 if event.get("kind") == "agent_event" and
                 event.get("payload", {}).get("kind") == "user_observation")["payload"]["payload"]["text"] = "different model input"
        else:
            row["repetition"] = 99
        raw = tmp_path / f"tampered_{mutation}.jsonl"
        raw.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
        with pytest.raises(ValueError, match="fixed suite"):
            verify_fixed_suite_cases(replace(audit, raw_jsonl=raw))


@pytest.mark.parametrize("measured_successes", [0, 59])
def test_signed_review_rejects_partial_fixed_suite_even_with_all_gate_receipts(
        tmp_path, monkeypatch, measured_successes):
    sign, summary, _, route, keys, _ = _qualified_fixture(
        tmp_path, monkeypatch, measured_successes=measured_successes,
    )
    audited = audit_summary(summary)
    assert audited.internally_valid
    assert audited.trace_verified
    assert audited.protocol_compliant
    assert audited.measured_attempts == 60
    assert audited.measured_successes == measured_successes
    with pytest.raises(ValueError, match="every measured request to succeed"):
        load_reviewed_qualification(
            sign(), trusted_keys=keys, native_config={"routes": [route]},
        )


def test_reviewed_profile_refuses_changed_loaded_source(tmp_path, monkeypatch):
    sign, _, _, route, keys, _ = _qualified_fixture(tmp_path, monkeypatch)
    bundle = sign()
    for source in (
        "imported_omni_source_sha256",
        "imported_vllm_source_sha256",
        "imported_stage_contract_source_sha256",
    ):
        with monkeypatch.context() as patched:
            patched.setattr(
                f"vllm_omni.edge.agent.runtime_identity.{source}",
                lambda: "0" * 64,
            )
            with pytest.raises(ValueError, match="profiled Agent/Omni source"):
                load_reviewed_qualification(
                    bundle, trusted_keys=keys, native_config={"routes": [route]},
                )


def test_tampered_profile_or_gate_source_never_promotes(tmp_path, monkeypatch):
    sign, summary, sources, route, keys, manifest = _qualified_fixture(tmp_path, monkeypatch)
    bundle = sign()
    original_summary = summary.read_bytes()
    summary_data = json.loads(summary.read_text())
    summary_data["routes"][route["route_id"]]["measured_successes"] = 999
    _write(summary, summary_data)
    with pytest.raises(ValueError, match="evidence SHA-256 mismatch"):
        load_reviewed_qualification(bundle, trusted_keys=keys,
                                    native_config={"routes": [route]})
    # Even a newly signed summary reference cannot turn fabricated counts into
    # a verified raw profile: the auditor recomputes the profile from requests.
    manifest["profile_summary"] = _ref(summary)
    bundle = sign()
    with pytest.raises(ValueError, match="route profile differs"):
        load_reviewed_qualification(bundle, trusted_keys=keys,
                                    native_config={"routes": [route]})
    # Restore the summary and alter one independent gate source instead.
    summary.write_bytes(original_summary)
    manifest["profile_summary"] = _ref(summary)
    bundle = sign()
    source = sources["memory_admission"]
    source.write_text(source.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence SHA-256 mismatch"):
        load_reviewed_qualification(bundle, trusted_keys=keys,
                                    native_config={"routes": [route]})
    # The reviewer can rebind an edited gate file, but contradictory numbers
    # still fail the gate-specific check.
    source_data = json.loads(source.read_text())
    source_data["observations"]["pool_used_peak_bytes"]["host_ram"] = 1999
    source_data["observations"]["route_incremental_peak_bytes"]["host_ram"] = 999
    _write(source, source_data)
    gate_path = tmp_path / "memory_admission.review.json"
    receipt = json.loads(gate_path.read_text())
    receipt["source"] = _ref(source)
    _write(gate_path, receipt)
    manifest["gates"]["memory_admission"] = _ref(gate_path)
    bundle = sign()
    with pytest.raises(ValueError, match="overcommitted pool"):
        load_reviewed_qualification(bundle, trusted_keys=keys,
                                    native_config={"routes": [route]})


def test_memory_receipt_rejects_peak_above_route_claim_or_missing_increment(tmp_path, monkeypatch):
    sign, _, sources, route, keys, bundle_data = _qualified_fixture(tmp_path, monkeypatch)
    source = sources["memory_admission"]
    receipt_path = tmp_path / "memory_admission.review.json"

    def rebind_and_sign(observations):
        raw = json.loads(source.read_text(encoding="utf-8"))
        raw["observations"] = observations
        _write(source, raw)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["source"] = _ref(source)
        _write(receipt_path, receipt)
        bundle_data["gates"]["memory_admission"] = _ref(receipt_path)
        return sign()

    original = json.loads(source.read_text(encoding="utf-8"))["observations"]
    # A 150-byte route increase is below the 200-byte live ceiling but above
    # the 100-byte reservation, so physical headroom alone is insufficient.
    over_claim = json.loads(json.dumps(original))
    over_claim["pool_used_peak_bytes"]["host_ram"] = 1150
    over_claim["route_incremental_peak_bytes"]["host_ram"] = 150
    with pytest.raises(ValueError, match="route peak exceeds admitted demand"):
        load_reviewed_qualification(rebind_and_sign(over_claim), trusted_keys=keys,
                                    native_config={"routes": [route]})

    # A legacy receipt that has only a system-wide total has no route peak.
    legacy = json.loads(json.dumps(original))
    legacy.pop("route_incremental_peak_bytes")
    with pytest.raises(ValueError, match="lacks admitted, baseline, route peak"):
        load_reviewed_qualification(rebind_and_sign(legacy), trusted_keys=keys,
                                    native_config={"routes": [route]})

    inconsistent = json.loads(json.dumps(original))
    inconsistent["route_incremental_peak_bytes"]["host_ram"] = 80
    with pytest.raises(ValueError, match="differs from baseline-adjusted"):
        load_reviewed_qualification(rebind_and_sign(inconsistent), trusted_keys=keys,
                                    native_config={"routes": [route]})
