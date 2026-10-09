# SPDX-License-Identifier: Apache-2.0
"""Reviewed qualification preserves its exact Agent consumer contract.

These are constructor/routing tests, not model or release qualification.
The raw-profile, Strata placement/release and independent-observation
validators are boundary fakes. Actual Ed25519 signing, file references,
configuration/provenance comparisons, contract validation, Qualification and
router selection run normally. Existing full verifier tests remain required.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vllm_omni.edge.agent import qualification as q
from vllm_omni.edge.agent import runtime_identity
from vllm_omni.edge.agent.model_output import AgentOutputContract
from vllm_omni.edge.agent.router import Admission, Route, select_route


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _fixture(tmp_path, monkeypatch, *, mode=None, invalid_workspace=False):
    contract = AgentOutputContract(mode) if mode is not None else None
    backend = {
        "name": q.STRATA_BACKEND,
        "weight_tier_plan": {"budget": {
            "host_workspace_bytes": 2 << 20, "host_transfer_bytes": 65536,
        }},
    }
    base = "strata:" + _hash(backend)
    entry = {
        "route_id": "synthetic-reviewed-route", "model": "synthetic-model",
        "backend": q.STRATA_BACKEND, "backend_config": backend,
        "artifact_id": base, "placement": "cpu", "memory_demands": {"host_ram": 2 << 20},
        "max_io_bytes": 65536,
    }
    binding = {"synthetic_engine_binding": "boundary-only"}
    if contract is not None:
        consumer = contract.consumer_identity(base)
        entry.update(
            artifact_id="strata-agent:" + consumer["identity_sha256"],
            base_artifact_id=base, model_output_contract=contract.to_dict(),
            model_output_consumer_identity=consumer,
            model_output_workspace_bytes=0 if invalid_workspace else contract.workspace_budget_bytes,
        )
        binding.update(base_engine_artifact_id=base, model_output_consumer_identity=consumer)
    samples = {name: [1.0] * 20 for name in ("short", "medium", "long")}
    conditions = {
        "suite_id": q.FIXED_SUITE_ID, "environment_fingerprint": "synthetic-host", "power_condition": "AC",
        "runtime_versions": {
            "vllm_omni_imported_source_sha256": "0" * 64,
            "agent_runtime_identity_sha256": "1" * 64, "strata_runtime_manifest_sha256": "2" * 64,
        },
    }
    profile_route = {
        "route_id": entry["route_id"], "model_id": entry["model"], "artifact_id": entry["artifact_id"],
        "artifact_sha256": "3" * 64, "checkpoint_revision": "4" * 40,
        "backend": entry["backend"], "expected_placement": entry["placement"], "backend_identity": binding,
    }
    stored = {
        "route": profile_route, "task_class": "basic", "measured_successes": 60, "measured_attempts": 60,
        "answer_latency_s": samples, "ttft_s": samples,
        "warmups_per_length": {name: 1 for name in samples}, "sustained_seconds": 1800,
        **{key: True for key in (
            "correctness_pass", "e2e_trace_pass", "tool_safety_pass", "telemetry_present",
            "stability_pass", "protocol_compliant", "actual_placement_matches", "placement_evidence_present",
        )},
    }
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"record_type": "request", "result": {"placement_evidence": {}}}) + "\n")
    summary_path = _write(tmp_path / "summary.json", {"conditions": conditions, "routes": {entry["route_id"]: stored}})
    audit = q.EvidenceAudit(summary_path, raw, _ref(raw)["sha256"], entry["route_id"], 60, 60, True, True, ())
    monkeypatch.setattr(q, "audit_summary", lambda *_: audit)
    fixed_checks, release_checks, gates_checked = [], [], []
    monkeypatch.setattr(q, "verify_fixed_suite_cases", lambda actual: fixed_checks.append(actual))
    monkeypatch.setattr(q, "strata_route_binding", lambda actual: copy.deepcopy(binding))
    monkeypatch.setattr(q, "_verify_strata_release_observations", lambda actual: release_checks.append(actual))
    monkeypatch.setattr(runtime_identity, "imported_omni_source_sha256", lambda: "0" * 64)
    monkeypatch.setattr(runtime_identity, "loaded_runtime_sha256", lambda: "1" * 64)

    def check_gate(name, observations, identity, root, current_route):
        assert observations == {"task_class": "basic", "synthetic_boundary": name}
        assert current_route == entry
        gates_checked.append(name)

    monkeypatch.setattr(q, "_verify_gate_observations", check_gate)
    native_config = {"routes": [entry]}
    config_path = _write(tmp_path / "config.json", native_config)
    identity = q._identity(profile_route, conditions, audit.raw_sha256)
    provenance = {
        "lineage_verified": True, "checkpoint_revision": profile_route["checkpoint_revision"],
        "artifact_manifest_sha256": profile_route["artifact_sha256"], "runtime_manifest_sha256": "2" * 64,
        **binding,
    }
    index = _write(tmp_path / "index.json", {
        "automatic_qualification_export": False, "source_config": str(config_path),
        "source_config_sha256": _ref(config_path)["sha256"],
        "artifact_provenance": {entry["route_id"]: provenance},
        "results": [{"route_id": entry["route_id"], "task_class": "basic", "raw_sha256": audit.raw_sha256,
                     "summary": str(summary_path)}],
    })
    gates = {}
    for name in sorted(q.REQUIRED_GATES):
        source = _write(tmp_path / f"{name}.raw.json", {
            "record_type": f"{name}_evidence_v1", "identity": identity,
            "observations": {"task_class": "basic", "synthetic_boundary": name},
        })
        receipt = _write(tmp_path / f"{name}.review.json", {
            "schema": q.GATE_SCHEMA, "gate": name, "identity": identity, "outcome": "pass", "source": _ref(source),
        })
        gates[name] = _ref(receipt)
    key = Ed25519PrivateKey.generate()
    trusted = {"fixture-reviewer": base64.b64encode(key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw,
    )).decode()}
    bundle = {
        "schema": q.PROMOTION_SCHEMA, "identity": identity, "profile_summary": _ref(summary_path),
        "profile_index": _ref(index), "gates": gates,
        "review": {"key_id": "fixture-reviewer", "reviewer": "synthetic reviewer", "reviewed_at": "synthetic-time"},
    }
    bundle_path = tmp_path / "promotion.json"

    def sign():
        bundle["review"]["signature_ed25519"] = base64.b64encode(key.sign(q.bundle_signing_bytes(bundle))).decode()
        _write(bundle_path, bundle)

    sign()
    route = Route(entry["route_id"], entry["artifact_id"], entry["model"], entry["backend"],
                  frozenset({"text"}), "cpu", entry["memory_demands"],
                  model_output_contract=contract, base_artifact_id=base if contract else None)
    return SimpleNamespace(
        contract=contract, route=route, entry=entry, bundle=bundle, sign=sign, identity=identity,
        bundle_path=bundle_path, native_config=native_config, trusted=trusted,
        fixed_checks=fixed_checks, release_checks=release_checks, gates_checked=gates_checked,
    )


def _load(fixture):
    return q.load_reviewed_qualification(
        fixture.bundle_path, trusted_keys=fixture.trusted, native_config=fixture.native_config,
    )


def _select(fixture, qualification):
    return select_route(
        "basic", [fixture.route], [qualification], suite_id=q.FIXED_SUITE_ID,
        environment_fingerprint="synthetic-host", power_condition="AC",
        admit=lambda _: Admission(True, "synthetic capacity", actual_placement="cpu"),
    )


@pytest.mark.parametrize("mode", ["strict_raw_agent_json_v1", "strict_outer_json_fence_agent_v1"])
def test_reviewed_constructor_binds_contract_and_router_accepts_only_that_contract(tmp_path, monkeypatch, mode):
    fixture = _fixture(tmp_path, monkeypatch, mode=mode)
    qualification = _load(fixture)
    assert qualification.output_contract_sha256 == fixture.contract.identity_sha256
    assert qualification.qualified
    decision = _select(fixture, qualification)
    assert decision.route is fixture.route and decision.qualification is qualification and not decision.experimental
    assert fixture.fixed_checks and fixture.release_checks
    assert fixture.gates_checked == sorted(q.REQUIRED_GATES)
    for wrong_hash in (None, "f" * 64):
        refused = _select(fixture, replace(qualification, output_contract_sha256=wrong_hash))
        assert refused.route is None
        assert refused.refusals[fixture.route.route_id] == "no current whole-Agent batch-1 qualification"


def test_no_contract_keeps_legacy_review_identity_and_router_match(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    qualification = _load(fixture)
    assert qualification.output_contract_sha256 is None
    assert _select(fixture, qualification).route is fixture.route
    assert set(fixture.identity) == {
        "route_id", "artifact_id", "artifact_sha256", "checkpoint_revision", "backend", "placement",
        "suite_id", "environment_fingerprint", "power_condition", "profile_raw_sha256",
    }


def test_constructor_revalidates_explicit_workspace_instead_of_copying_claimed_hash(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, mode="strict_raw_agent_json_v1", invalid_workspace=True)
    with pytest.raises(ValueError, match="consumer identity or explicit workspace"):
        _load(fixture)
    assert fixture.gates_checked == sorted(q.REQUIRED_GATES)


def test_invalid_signature_never_reaches_constructor_or_observation_gates(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, mode="strict_raw_agent_json_v1")
    fixture.bundle["review"]["signature_ed25519"] = base64.b64encode(bytes(64)).decode()
    _write(fixture.bundle_path, fixture.bundle)
    with pytest.raises(ValueError, match="signature"):
        _load(fixture)
    assert not fixture.fixed_checks and not fixture.gates_checked


def test_missing_independent_gate_is_not_repaired_by_contract_binding(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, mode="strict_raw_agent_json_v1")
    del fixture.bundle["gates"]["checkpoint_lineage"]
    fixture.sign()
    with pytest.raises(ValueError, match="all five independent qualification gates"):
        _load(fixture)
    assert not fixture.gates_checked


@pytest.mark.parametrize("gate", sorted(q.REQUIRED_GATES))
def test_independent_gate_failure_still_refuses_qualification(tmp_path, monkeypatch, gate):
    fixture = _fixture(tmp_path, monkeypatch, mode="strict_raw_agent_json_v1")

    def reject(name, *_):
        fixture.gates_checked.append(name)
        if name == gate:
            raise ValueError("synthetic independent gate refusal")

    monkeypatch.setattr(q, "_verify_gate_observations", reject)
    with pytest.raises(ValueError, match="synthetic independent gate refusal"):
        _load(fixture)
    assert fixture.gates_checked[-1] == gate


def test_strata_release_gate_still_precedes_contract_qualification(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, mode="strict_raw_agent_json_v1")

    def reject(_):
        raise ValueError("synthetic Strata release refusal")

    monkeypatch.setattr(q, "_verify_strata_release_observations", reject)
    with pytest.raises(ValueError, match="synthetic Strata release refusal"):
        _load(fixture)
    assert not fixture.gates_checked
