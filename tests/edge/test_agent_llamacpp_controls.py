# SPDX-License-Identifier: Apache-2.0
"""Portable Agent/Stage control tests using real package imports and tiny files.

All native execution is forbidden. Synthetic manifest fingerprint substitution
only admits the tiny selected runtime fixture, never an installed runtime.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from benchmarks.edge_agent.native_profile import load_profile_routes
from vllm_omni.edge.agent import native_app, qualification
from vllm_omni.edge.agent.llamacpp_route import (
    llamacpp_route_binding,
    validate_llamacpp_profile_binding,
)
from vllm_omni.edge.agent.omni_backend import OmniLlamaBackend, llama_config_from_entry
from vllm_omni.edge.agent.placement import (
    evidence_sha256,
    preparation_placement_matches,
    result_placement_matches,
)
from vllm_omni.engine.backends import llamacpp_controls as controls
from vllm_omni.engine.resource_ledger import ResourceLedger, ResourceUnavailable
from vllm_omni.engine.stage_runtime import StageRuntime
from vllm_omni.engine.weight_tiers import ArtifactFile, ArtifactManifest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controlled(**changes):
    return {
        "schema": controls.SCHEMA,
        "runtime_profile": controls.RUNTIME_PROFILE,
        "kv_type_k": "f16",
        "kv_type_v": "f16",
        "context_shift": False,
        "speculative_decoding": "none",
        "load_mode": "mmap",
        "lazy_mode": "on",
        "enable_thinking": False,
    } | changes


@pytest.fixture
def entry(tmp_path, monkeypatch):
    rows = []
    for index, name in enumerate(("llama-server.exe", "llama-server-impl.dll", "llama.dll")):
        path = tmp_path / name
        path.write_bytes(("synthetic runtime " + str(index)).encode())
        rows.append(ArtifactFile(name, path.stat().st_size, digest(path), role="auxiliary"))
    runtime = ArtifactManifest("test/runtime", "fixed-revision", "test", tuple(rows))
    monkeypatch.setattr(controls, "RUNTIME_MANIFEST_SHA256", runtime.manifest_sha256)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"tiny synthetic weights")
    return {
        "route_id": "controlled",
        "artifact_id": "original-artifact",
        "model": "original/model",
        "backend": "external.llamacpp.text.v1",
        "placement": "cpu",
        "model_file": str(model),
        "model_sha256": digest(model),
        "server_bin": str(tmp_path / "llama-server.exe"),
        "server_sha256": digest(tmp_path / "llama-server.exe"),
        "log_file": str(tmp_path / "log"),
        "memory_demands": {"host_ram": 1000},
        "memory_overhead_bytes": 100,
        "context_tokens": 4096,
        "max_new_tokens": 8,
        "max_io_bytes": 100,
        "launch_controls": controlled(),
        "launch_controls_runtime_manifest": runtime.to_dict(),
    }


def config(value):
    return llama_config_from_entry(value, capacities=value["memory_demands"])


def plan_for(value):
    cfg = config(value)
    binding = llamacpp_route_binding(value)
    plan = {
        key: binding[key]
        for key in (
            "context_tokens",
            "max_new_tokens",
            "max_io_bytes",
            "server_sha256",
            "model_sha256",
            "mmproj_sha256",
        )
    }
    plan.update(
        requested_device=cfg.placement,
        observed_model_placement=cfg.placement,
        artifact_manifest_sha256=binding["source_artifact_manifest_sha256"],
        launch_controls={
            "requested": dict(cfg.launch_controls),
            "runtime_artifact_manifest_sha256": binding["runtime_artifact_manifest_sha256"],
            "memory_claim_discounted": False,
        },
    )
    return plan


def test_json_round_trip_reaches_actual_stage_config_detached(entry):
    decoded = json.loads(json.dumps(entry))
    cfg = config(decoded)
    backend = OmniLlamaBackend(cfg)._stage_backend_config()
    assert backend["launch_controls"] == controlled()
    assert backend["launch_controls_runtime_manifest"] == entry["launch_controls_runtime_manifest"]
    decoded["launch_controls"]["kv_type_k"] = "q8_0"
    decoded["launch_controls_runtime_manifest"]["files"][0]["sha256"] = "0" * 64
    assert OmniLlamaBackend(cfg)._stage_backend_config() == backend
    assert cfg.demands == entry["memory_demands"]
    assert entry["artifact_id"] == "original-artifact"


def test_none_loading_lazy_on_reaches_stage_without_changing_admission(entry):
    value = entry | {"launch_controls": controlled(load_mode="none", lazy_mode="on")}
    cfg = config(value)
    backend = OmniLlamaBackend(cfg)
    stage = backend._stage_backend_config()
    parsed = controls.parse_launch_controls(stage["launch_controls"])
    args = parsed.arguments()
    assert args[args.index("--load-mode") + 1] == "none"
    assert args[args.index("--lazy-mode") + 1] == "on"
    assert cfg.demands == entry["memory_demands"]
    plan = plan_for(value)
    backend._validate_loaded_plan(plan)
    assert plan["launch_controls"]["memory_claim_discounted"] is False
    original, current = llamacpp_route_binding(entry), llamacpp_route_binding(value)
    assert current["backend_config_sha256"] != original["backend_config_sha256"]
    assert current["model_sha256"] == original["model_sha256"]
    assert current["source_artifact_manifest_sha256"] == original["source_artifact_manifest_sha256"]
    assert current["runtime_artifact_manifest_sha256"] == original["runtime_artifact_manifest_sha256"]


@pytest.mark.parametrize("mutation", ["missing", "null", "invalid", "backend", "launcher", "typed"])
def test_invalid_controlled_entries_refuse_before_load(entry, mutation):
    value = copy.deepcopy(entry)
    if mutation == "missing":
        value.pop("launch_controls_runtime_manifest")
    if mutation == "null":
        value["launch_controls"] = None
    if mutation == "invalid":
        value["launch_controls"]["context_shift"] = 0
    if mutation == "backend":
        value["backend"] = "external.strata.text.v1"
    if mutation == "launcher":
        value["server_sha256"] = "0" * 64
    if mutation == "typed":
        value["context_tokens"] = 4096.0
    with pytest.raises(ValueError):
        llamacpp_route_binding(value)


def test_native_loader_uses_same_typed_config_translation():
    # This is the exact imported converter, not a duplicate kwargs projection.
    assert native_app.llama_config_from_entry is llama_config_from_entry


def test_legacy_cpu_hybrid_and_projector_configs_omit_controls(entry, tmp_path):
    base = {key: value for key, value in entry.items() if not key.startswith("launch_controls")}
    variants = [
        base,
        base
        | {
            "placement": "cpu+Vulkan1",
            "memory_demands": {"host_ram": 1000, "vram": 1000},
            "gpu_layers": 1,
            "cpu_weight_budget_bytes": 100,
            "gpu_weight_budget_bytes": 100,
            "vram_overhead_bytes": 100,
        },
    ]
    projector = tmp_path / "projector.gguf"
    projector.write_bytes(b"synthetic projector")
    variants.append(
        base
        | {
            "mmproj_file": str(projector),
            "mmproj_sha256": digest(projector),
            "max_image_bytes": 100,
            "image_token_reserve": 10,
        }
    )
    for value in variants:
        backend = OmniLlamaBackend(config(value))._stage_backend_config()
        assert "launch_controls" not in backend
        assert "launch_controls_runtime_manifest" not in backend
        assert llamacpp_route_binding(value) is None
    assert OmniLlamaBackend(config(variants[1]))._stage_backend_config()["gpu_layers"] == 1
    assert OmniLlamaBackend(config(variants[2]))._stage_backend_config()["mmproj_sha256"] == digest(projector)


def test_control_identity_is_stable_and_preserves_checkpoint(entry):
    original = llamacpp_route_binding(entry)
    moved = entry | {
        "server_bin": "elsewhere/llama-server.exe",
        "model_file": "elsewhere/model.gguf",
        "log_file": "new-log",
        "port": 12345,
        "nonce": "changed",
        "available_ram": 10,
        "removed_environment_variable_names": ["LLAMA_ARG_X"],
    }
    assert llamacpp_route_binding(moved) == original
    changed = entry | {"launch_controls": controlled(lazy_mode="off")}
    assert llamacpp_route_binding(changed)["backend_config_sha256"] != original["backend_config_sha256"]
    assert entry["artifact_id"] == "original-artifact"
    assert original["model_sha256"] == entry["model_sha256"]


def test_profile_carries_controls_without_changing_artifact_lineage(entry):
    lineage = {
        "routes": {
            entry["route_id"]: {
                "model_sha256": entry["model_sha256"],
                "mmproj_sha256": None,
                "checkpoint_revision": "fixed-weight-revision",
                "precision": "test-mixed",
                "lineage_verified": False,
            }
        }
    }
    routes, provenance = load_profile_routes({"routes": [entry]}, lineage)
    profile = asdict(routes[0])
    assert profile["artifact_id"] == entry["artifact_id"]
    assert profile["artifact_sha256"] == entry["model_sha256"]
    assert profile["checkpoint_revision"] == "fixed-weight-revision"
    assert profile["backend_identity"] == llamacpp_route_binding(entry)
    validate_llamacpp_profile_binding(profile, entry, provenance[entry["route_id"]])
    conditions = {"suite_id": "test", "environment_fingerprint": "test", "power_condition": "AC"}
    identity = qualification._identity(profile, conditions, "a" * 64)
    assert identity["artifact_id"] == entry["artifact_id"]
    assert identity["llamacpp_launch_controls_sha256"] == profile["backend_identity"]["backend_config_sha256"]
    legacy = profile | {"backend_identity": {}}
    assert "llamacpp_launch_controls_sha256" not in qualification._identity(legacy, conditions, "a" * 64)


@pytest.mark.parametrize("mutation", ["old_profile", "drop", "changed", "provenance", "runtime"])
def test_reviewed_qualification_binding_rejects_control_mismatch(entry, mutation):
    binding = llamacpp_route_binding(entry)
    profile, provenance, current = {"backend_identity": binding}, copy.deepcopy(binding), copy.deepcopy(entry)
    if mutation == "old_profile":
        profile = {"backend_identity": {}}
    if mutation == "drop":
        current.pop("launch_controls")
        current.pop("launch_controls_runtime_manifest")
    if mutation == "changed":
        current["launch_controls"]["lazy_mode"] = "off"
    if mutation == "provenance":
        provenance["backend_config_sha256"] = "0" * 64
    if mutation == "runtime":
        current["launch_controls_runtime_manifest"]["files"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_llamacpp_profile_binding(profile, current, provenance)


@pytest.mark.parametrize("mutation", ["drop", "kv", "runtime", "context", "source", "discount"])
def test_agent_and_profile_refuse_loaded_plan_dropped_or_changed_controls(entry, mutation):
    plan = plan_for(entry)
    backend = OmniLlamaBackend(config(entry))
    backend._validate_loaded_plan(plan)
    if mutation == "drop":
        plan.pop("launch_controls")
    if mutation == "kv":
        plan["launch_controls"]["requested"]["kv_type_k"] = "q8_0"
    if mutation == "runtime":
        plan["launch_controls"]["runtime_artifact_manifest_sha256"] = "0" * 64
    if mutation == "context":
        plan["context_tokens"] = 4096.0
    if mutation == "source":
        plan["artifact_manifest_sha256"] = "0" * 64
    if mutation == "discount":
        plan["launch_controls"]["memory_claim_discounted"] = True
    with pytest.raises(ValueError):
        backend._validate_loaded_plan(plan)
    route = {
        "backend": entry["backend"],
        "expected_placement": "cpu",
        "backend_identity": llamacpp_route_binding(entry),
        "model_id": entry["model"],
        "artifact_id": entry["artifact_id"],
    }
    evidence = {"execution_plan": plan, "loaded_plan_sha256": evidence_sha256(plan)}
    result = {
        "actual_placement": "cpu",
        "placement_evidence": evidence,
        "model_id": entry["model"],
        "artifact_id": entry["artifact_id"],
        "backend": entry["backend"],
    }
    preparation = {
        "actual_placement": "cpu",
        "details": evidence,
        "artifact_id": entry["artifact_id"],
        "cold_start_confirmed": True,
    }
    assert not result_placement_matches(result, route)
    assert not preparation_placement_matches(preparation, route)
    clean = plan_for(entry)
    clean_evidence = {"execution_plan": clean, "loaded_plan_sha256": evidence_sha256(clean)}
    assert result_placement_matches(result | {"placement_evidence": clean_evidence}, route)
    assert preparation_placement_matches(preparation | {"details": clean_evidence}, route)


def test_real_stage_factory_corrupt_selected_dll_refuses_and_releases_exact_lease(entry, monkeypatch):
    from pathlib import Path

    import vllm_omni.engine.backends.llamacpp as adapter

    cfg = OmniLlamaBackend(config(entry))._stage_backend_config()
    target = Path(entry["server_bin"]).with_name("llama-server-impl.dll")
    target.write_bytes(b"x" * target.stat().st_size)
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda *a, **k: pytest.fail("native launch forbidden"))
    ledger = ResourceLedger({"host_ram": 1000})
    token = ledger.reserve("test-stage", {"host_ram": 1000})
    runtime = StageRuntime.__new__(StageRuntime)
    runtime._resource_started = set()
    runtime.resource_ledger = ledger
    runtime._resource_reservations = {(0, 0): token}
    plan = SimpleNamespace(
        metadata=SimpleNamespace(stage_type="graph", stage_id=0),
        replica_id=0,
        launch_mode="local",
        stage_cfg=SimpleNamespace(engine_args={"backend": cfg}),
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        runtime._initialize_replica(plan, stage_init_timeout=1)
    assert runtime._resource_started == {(0, 0)}
    assert ledger.was_released(token)
    assert not ledger.owns(token)
    assert ledger.snapshot()["reserved"] == {"host_ram": 0}
    assert ledger.snapshot()["quarantined"] == []


@pytest.mark.parametrize("load_mode", ["mmap", "none"])
def test_real_stage_factory_lazy_cannot_discount_capacity(entry, monkeypatch, load_mode):
    import vllm_omni.engine.backends.llamacpp as adapter

    value = entry | {"launch_controls": controlled(load_mode=load_mode, lazy_mode="on")}
    cfg = OmniLlamaBackend(config(value))._stage_backend_config()
    ledger = ResourceLedger({"host_ram": 100})
    token = ledger.reserve("test-stage", {"host_ram": 100})
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda *a, **k: pytest.fail("native launch forbidden"))
    with pytest.raises(ResourceUnavailable, match="declared KV/workspace"):
        from vllm_omni.engine.backends import create_graph_client

        create_graph_client(SimpleNamespace(stage_id=0), {"backend": cfg}, ledger, token)
    assert ledger.was_released(token)
    assert ledger.snapshot()["owners"] == []


@pytest.mark.parametrize("capacity", [10, 1000])
def test_actual_native_config_loader_validates_and_translates_before_capacity(entry, tmp_path, monkeypatch, capacity):
    class ConfigReachedError(Exception):
        pass

    captured = []

    def backend(cfg):
        captured.append(cfg)
        raise ConfigReachedError

    monkeypatch.setattr(native_app.sys, "platform", "win32")
    monkeypatch.setattr(
        native_app,
        "_hardware_snapshot",
        lambda: {
            "host_ram_available_bytes": capacity,
            "vram_available_bytes": None,
            "gpu_name": None,
        },
    )
    monkeypatch.setattr(native_app, "_artifact_disk_capacity", lambda routes: None)
    monkeypatch.setattr(native_app, "OmniLlamaBackend", backend)
    path = tmp_path / "native.json"
    if capacity == 10:
        invalid = copy.deepcopy(entry)
        invalid.pop("launch_controls_runtime_manifest")
        path.write_text(json.dumps({"routes": [invalid]}), encoding="utf-8")
        with pytest.raises(ValueError, match="both explicit"):
            native_app.build_controller(path)
        assert captured == []
    else:
        path.write_text(json.dumps({"routes": [entry]}), encoding="utf-8")
        with pytest.raises(ConfigReachedError):
            native_app.build_controller(path)
        assert dict(captured[0].launch_controls) == controlled()
        assert captured[0].demands == entry["memory_demands"]
        assert captured[0].route_id == entry["route_id"]


def test_partial_profile_marker_never_falls_through_legacy_placement(entry):
    route = {
        "backend": entry["backend"],
        "expected_placement": "cpu",
        "model_id": entry["model"],
        "artifact_id": entry["artifact_id"],
        "backend_identity": {"launch_controls": controlled()},
    }
    result = {
        "model_id": entry["model"],
        "artifact_id": entry["artifact_id"],
        "backend": entry["backend"],
        "actual_placement": "cpu",
        "placement_evidence": {"execution_plan": plan_for(entry)},
    }
    result["placement_evidence"]["loaded_plan_sha256"] = evidence_sha256(result["placement_evidence"]["execution_plan"])
    assert not result_placement_matches(result, route)
    assert not preparation_placement_matches(
        {
            "artifact_id": entry["artifact_id"],
            "cold_start_confirmed": True,
            "actual_placement": "cpu",
            "details": result["placement_evidence"],
        },
        route,
    )


@pytest.fixture
def consumer_entry(entry, tmp_path, monkeypatch):
    import subprocess

    from vllm_omni.edge.agent.llamacpp_route import llamacpp_agent_config_with_output_contract
    from vllm_omni.edge.agent.model_output import AgentOutputContract

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process forbidden"))
    # Reuse tiny runtime fixture and add two synthetic source-weight records.
    first = __import__("pathlib").Path(entry["model_file"])
    second = tmp_path / "second.gguf"
    second.write_bytes(b"second tiny synthetic shard")
    source = ArtifactManifest(
        "fixture/source", "fixture-source-revision", "test",
        tuple(ArtifactFile(path.name, path.stat().st_size, digest(path),
                           role="weights", quantization="synthetic-Q2", layout="GGUF")
              for path in (first, second)),
    )
    base = entry | {
        "artifact_root": str(tmp_path), "artifact_manifest": source.to_dict(),
        "memory_demands": {"host_ram": 1000, "windows_commit": 2000},
        "max_io_bytes": 65536, "launch_controls": controlled(load_mode="none", lazy_mode="on"),
    }
    return llamacpp_agent_config_with_output_contract(
        base, AgentOutputContract("strict_outer_json_fence_agent_v1")
    )


def consumer_config(value):
    from vllm_omni.edge.agent.llamacpp_route import llamacpp_memory_demands

    return llama_config_from_entry(value, capacities=llamacpp_memory_demands(value))


def consumer_plan(value):
    binding = llamacpp_route_binding(value)
    cfg = consumer_config(value)
    return {
        **{key: binding[key] for key in (
            "context_tokens", "max_new_tokens", "max_io_bytes",
            "server_sha256", "model_sha256", "mmproj_sha256",
        )},
        "backend": "external.llamacpp.text.v1", "stage_id": 0,
        "worker_generation": "generation", "worker_pid": 123,
        "requested_device": cfg.placement, "observed_model_placement": cfg.placement,
        "placement_evidence_level": "startup_log",
        "artifact_manifest_sha256": binding["source_artifact_manifest_sha256"],
        "reserved_bytes": dict(cfg.demands), "memory_overhead_bytes": cfg.memory_overhead_bytes,
        "expert_compute_verified": None, "expert_final_storage_verified": None,
        "launch_controls": {
            "requested": dict(cfg.launch_controls),
            "runtime_artifact_manifest_sha256": binding["runtime_artifact_manifest_sha256"],
            "memory_claim_discounted": False,
        },
    }


def consumer_lineage(value):
    source = ArtifactManifest.from_dict(value["artifact_manifest"])
    return {"routes": {value["route_id"]: {
        "checkpoint_revision": source.revision, "model_sha256": value["model_sha256"],
        "mmproj_sha256": None, "artifact_manifest_sha256": source.manifest_sha256,
        "precision": "synthetic-Q2", "lineage_verified": False,
    }}}


def test_consumer_workspace_charged_once_and_source_identity_preserved(consumer_entry):
    from dataclasses import replace

    from vllm_omni.edge.agent.llamacpp_route import (
        llamacpp_agent_config_with_output_contract,
        llamacpp_memory_demands,
    )
    from vllm_omni.edge.agent.model_output import validate_output_contract_entry

    original = copy.deepcopy(consumer_entry)
    contract = validate_output_contract_entry(consumer_entry)
    cfg = consumer_config(consumer_entry)
    effective = {"host_ram": 1000 + contract.workspace_budget_bytes,
                 "windows_commit": 2000 + contract.workspace_budget_bytes}
    assert dict(cfg.demands) == effective == llamacpp_memory_demands(consumer_entry)
    assert cfg.memory_overhead_bytes == 100 + contract.workspace_budget_bytes
    assert consumer_config(consumer_entry).demands == cfg.demands
    assert replace(cfg).demands == cfg.demands  # typed copies never recharge
    assert replace(cfg).memory_overhead_bytes == cfg.memory_overhead_bytes
    assert consumer_entry == original
    assert consumer_entry["memory_demands"] == {"host_ram": 1000, "windows_commit": 2000}
    assert cfg.artifact_manifest == consumer_entry["artifact_manifest"]
    assert cfg.model_sha256 == consumer_entry["model_sha256"]
    assert consumer_entry["model"] == "original/model"
    with pytest.raises(ValueError, match="twice"):
        llamacpp_agent_config_with_output_contract(consumer_entry, contract)
    ledger = ResourceLedger(effective)
    token = ledger.reserve("one-consumer", effective)
    backend = OmniLlamaBackend(cfg)
    backend.bind_output_contract(contract, base_artifact_id=consumer_entry["base_artifact_id"])
    backend.bind_resource_lease(ledger, token)
    assert backend._shared_reservation is token
    assert ledger.snapshot()["reserved"] == effective
    assert ledger.owns(token)


def test_consumer_workspace_refuses_base_capacity_before_any_load(consumer_entry):
    with pytest.raises(ValueError, match="ceiling"):
        llama_config_from_entry(consumer_entry, capacities=consumer_entry["memory_demands"])


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("base_artifact_id"),
    lambda value: value.update(model_output_workspace_bytes=True),
    lambda value: value.update(model_output_workspace_bytes=0),
    lambda value: value.update(memory_overhead_bytes=True),
    lambda value: value["memory_demands"].pop("windows_commit"),
    lambda value: value.update(base_artifact_id="strata:" + "a" * 64),
    lambda value: value.update(artifact_id="original-artifact"),
    lambda value: value.update(execution_observation={"status": "complete"}),
    lambda value: value.update(expert_compute_verified=True),
    lambda value: value.update(cpu_compute_verified=True),
    lambda value: value.update(backend="external.llamacpp.multimodal.v1"),
    lambda value: value.update(placement="Vulkan_Host+Vulkan1"),
])
def test_invalid_consumer_metadata_refuses_without_loading(consumer_entry, mutation):
    from vllm_omni.edge.agent.model_output import validate_output_contract_entry

    value = copy.deepcopy(consumer_entry)
    mutation(value)
    with pytest.raises((ValueError, KeyError)):
        validate_output_contract_entry(value)


def test_consumer_profile_binds_all_shards_without_minting_lineage(consumer_entry):
    from vllm_omni.edge.agent.consumer_trace import consumer_trace_requested

    profiles, provenance = load_profile_routes({"routes": [consumer_entry]}, consumer_lineage(consumer_entry))
    route = asdict(profiles[0])
    assert consumer_trace_requested(route)
    assert route["backend_identity"] == llamacpp_route_binding(consumer_entry)
    assert route["artifact_sha256"] == consumer_entry["model_sha256"]
    assert provenance[consumer_entry["route_id"]]["lineage_verified"] is False
    assert provenance[consumer_entry["route_id"]]["source_artifact_manifest_sha256"] == (
        ArtifactManifest.from_dict(consumer_entry["artifact_manifest"]).manifest_sha256
    )
    validate_llamacpp_profile_binding(route, consumer_entry, provenance[consumer_entry["route_id"]])
    stale = copy.deepcopy(route)
    stale["backend_identity"].pop("model_output_consumer_identity")
    with pytest.raises(ValueError):
        validate_llamacpp_profile_binding(stale, consumer_entry, provenance[consumer_entry["route_id"]])


@pytest.mark.parametrize("mutation", [
    lambda row: row["routes"]["controlled"].update(precision="wrong"),
    lambda row: row["routes"]["controlled"].update(checkpoint_revision="wrong"),
    lambda row: row["routes"]["controlled"].pop("artifact_manifest_sha256"),
])
def test_consumer_profile_rejects_wrong_all_shard_metadata(consumer_entry, mutation):
    lineage = consumer_lineage(consumer_entry)
    mutation(lineage)
    with pytest.raises(ValueError, match="all-shard"):
        load_profile_routes({"routes": [consumer_entry]}, lineage)


def llama_trace(consumer_entry, kind="final"):
    # Reuse existing source-owned fixture and actual unchanged parser proofs.
    from tests.edge.test_agent_consumer_trace import scenario

    events, _, evidence, trusted = scenario(kind, observed=False)
    profiles, _ = load_profile_routes({"routes": [consumer_entry]}, consumer_lineage(consumer_entry))
    route = asdict(profiles[0])
    consumer = route["backend_identity"]["model_output_consumer_identity"]
    route_event = next(event["payload"] for event in events if event["kind"] == "route")
    route_event.update(
        route_id=route["route_id"], model=route["model_id"], backend=route["backend"],
        artifact_id=route["artifact_id"], model_output_consumer_identity=consumer,
        model_output_contract=consumer["contract"], actual_placement=route["expected_placement"],
    )
    for event in events:
        if event["kind"] == "model_output_contract":
            event["payload"]["consumer_identity_sha256"] = consumer["identity_sha256"]
            event["payload"]["stage_event"]["seq"] = 1
        elif event["kind"] == "model_metrics":
            event["payload"]["metrics"]["stage_event"]["seq"] = 1
    plan = consumer_plan(consumer_entry)
    evidence.update(
        execution_plan=plan, loaded_plan_sha256=evidence_sha256(plan),
        placement_verification_scope="loaded_configuration_and_terminal_stage_identity",
    )
    return events, route, evidence, trusted


@pytest.mark.parametrize("kind", ["final", "tool", "read_url"])
def test_llama_full_consumer_trace_uses_generic_proofs_and_not_strata(consumer_entry, kind, monkeypatch):
    from vllm_omni.edge.agent import consumer_trace as trace

    monkeypatch.setattr(trace, "validate_strata_request_evidence",
                        lambda *a, **k: pytest.fail("Strata placement diagnostics are forbidden for llama"))
    monkeypatch.setattr(trace, "_native_io",
                        lambda *a, **k: pytest.fail("Strata I/O diagnostics are forbidden for llama"))
    events, route, evidence, trusted = llama_trace(consumer_entry, kind)
    trace.validate_consumer_trace(events, "READY", route, evidence, trusted_task_binding=trusted)


@pytest.mark.parametrize("field,value", [
    ("worker_generation", "wrong"), ("worker_pid", 0),
    ("expert_compute_verified", True),
    ("gpu_observer_identity", {"status": "verified"}),
    ("reserved_bytes", {"host_ram": 1, "windows_commit": 1}),
])
def test_llama_consumer_trace_rejects_unowned_or_fabricated_provenance(consumer_entry, field, value):
    from vllm_omni.edge.agent.consumer_trace import validate_consumer_trace

    events, route, evidence, trusted = llama_trace(consumer_entry)
    evidence["execution_plan"][field] = value
    evidence["loaded_plan_sha256"] = evidence_sha256(evidence["execution_plan"])
    with pytest.raises(ValueError):
        validate_consumer_trace(events, "READY", route, evidence, trusted_task_binding=trusted)


def test_llama_consumer_trace_rejects_replay_and_native_strata_io(consumer_entry):
    from vllm_omni.edge.agent.consumer_trace import validate_consumer_trace

    for replay in (True, False):
        events, route, evidence, trusted = llama_trace(consumer_entry, "tool")
        if replay:
            for event in events:
                if event["kind"] == "model_output_contract":
                    event["payload"]["stage_event"]["epoch"] = 11
                if event["kind"] == "model_metrics":
                    event["payload"]["metrics"]["stage_event"]["epoch"] = 11
        else:
            metrics = next(event["payload"]["metrics"] for event in events if event["kind"] == "model_metrics")
            metrics["backend_metrics"] = {"runtime_telemetry": {"native_io_observation": {"status": "complete"}}}
        with pytest.raises(ValueError):
            validate_consumer_trace(events, "READY", route, evidence, trusted_task_binding=trusted)


def test_llama_consumer_trace_cannot_authorize_another_navigation(consumer_entry):
    from tests.edge.test_agent_consumer_trace import rebind_tool_command
    from vllm_omni.edge.agent.consumer_trace import validate_consumer_trace

    sample = llama_trace(consumer_entry, "tool")
    rebind_tool_command(sample, "browser_open", {"url": "https://unauthorized.test/"})
    events, route, evidence, trusted = sample
    with pytest.raises(ValueError):
        validate_consumer_trace(events, "READY", route, evidence, trusted_task_binding=trusted)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_typed_llama_adapter_uses_actual_parser_and_terminal_cleanup(consumer_entry, finish_reason):
    from tests.edge.test_agent_model_output_integration import ActualAdapterProtocol
    from vllm_omni.edge.agent.model_output import (
        AgentOutputBuffer,
        collect_agent_command,
        validate_output_contract_entry,
    )

    original, _, output, counts = ActualAdapterProtocol().adapter('{"final":"done"}', finish_reason)
    contract = validate_output_contract_entry(consumer_entry)
    backend = OmniLlamaBackend(consumer_config(consumer_entry))
    backend.bind_output_contract(contract, base_artifact_id=consumer_entry["base_artifact_id"])
    backend._pool = original._pool
    buffer = AgentOutputBuffer(
        contract, request_id="id", worker_generation="generation", stage_id=0, permitted_tools=frozenset(),
    )
    if finish_reason == "stop":
        command, value, proof, _ = await collect_agent_command(
            backend.generate("prompt", request_id="id", max_tokens=8), buffer=buffer, cancel=backend.cancel,
        )
        assert (command, value) == ("final", "done")
        assert backend.last_model_output()["text"] is output.outputs[0].text
        assert proof["stage_event"]["worker_generation"] == "generation"
        assert counts == {"ack": 1, "abort": 0}
    else:
        with pytest.raises(ValueError):
            await collect_agent_command(
                backend.generate("prompt", request_id="id", max_tokens=8), buffer=buffer, cancel=backend.cancel,
            )
        assert counts["ack"] == 1
        assert counts["abort"] >= 1
        assert backend._active is None
    assert backend.close()
    assert backend.last_model_output() is None


def test_consumer_real_stage_factory_corrupt_dll_releases_same_charged_lease(consumer_entry, monkeypatch):
    from pathlib import Path

    import vllm_omni.engine.backends.llamacpp as adapter
    from vllm_omni.edge.agent.model_output import validate_output_contract_entry

    cfg = consumer_config(consumer_entry)
    backend = OmniLlamaBackend(cfg)
    backend.bind_output_contract(validate_output_contract_entry(consumer_entry),
                                 base_artifact_id=consumer_entry["base_artifact_id"])
    target = Path(cfg.server_bin).with_name("llama-server-impl.dll")
    target.write_bytes(b"x" * target.stat().st_size)
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda *a, **k: pytest.fail("native launch forbidden"))
    ledger = ResourceLedger(dict(cfg.demands))
    token = ledger.reserve("consumer-stage", dict(cfg.demands))
    backend.bind_resource_lease(ledger, token)
    runtime = StageRuntime.__new__(StageRuntime)
    runtime._resource_started = set()
    runtime.resource_ledger = ledger
    runtime._resource_reservations = {(0, 0): token}
    plan = SimpleNamespace(
        metadata=SimpleNamespace(stage_type="graph", stage_id=0), replica_id=0, launch_mode="local",
        stage_cfg=SimpleNamespace(engine_args={"backend": backend._stage_backend_config()}),
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        runtime._initialize_replica(plan, stage_init_timeout=1)
    assert ledger.was_released(token)
    assert not ledger.owns(token)
    assert ledger.snapshot()["reserved"] == {"host_ram": 0, "windows_commit": 0}
    assert ledger.snapshot()["quarantined"] == []


def test_actual_native_factory_binds_consumer_and_charged_route(consumer_entry, tmp_path, monkeypatch):
    from vllm_omni.edge.agent.model_output import validate_output_contract_entry

    class FactoryReachedError(Exception):
        pass

    captured = {}
    real_backend = native_app.OmniLlamaBackend

    def backend(cfg):
        result = real_backend(cfg)
        captured["backend"] = result
        return result

    def controller(**kwargs):
        captured.update(kwargs)
        raise FactoryReachedError

    monkeypatch.setattr(native_app.sys, "platform", "win32")
    monkeypatch.setattr(native_app, "_hardware_snapshot", lambda: {
        "host_ram_available_bytes": 8 << 20, "windows_commit_available_bytes": 8 << 20,
        "vram_available_bytes": None, "gpu_name": None, "power_condition": "AC",
    })
    monkeypatch.setattr(native_app, "_artifact_disk_capacity", lambda routes: None)
    monkeypatch.setattr(native_app, "_qualifications", lambda *a, **k: [])
    monkeypatch.setattr(native_app, "_fingerprint", lambda hardware: "fixture-environment")
    monkeypatch.setattr(native_app, "EncryptedMemoryStore", lambda path: object())
    monkeypatch.setattr(native_app, "WindowsToolBoundary", lambda: object())
    monkeypatch.setattr(native_app, "OmniLlamaBackend", backend)
    monkeypatch.setattr(native_app, "AgentController", controller)
    path = tmp_path / "consumer-native.json"
    path.write_text(json.dumps({"routes": [consumer_entry]}), encoding="utf-8")
    with pytest.raises(FactoryReachedError):
        native_app.build_controller(path)
    actual = captured["backend"]
    route = captured["routes"][0]
    assert dict(route.memory_demands) == dict(actual.config.demands)
    assert route.memory_demands["host_ram"] == 1000 + consumer_entry["model_output_workspace_bytes"]
    assert route.memory_demands["windows_commit"] == 2000 + consumer_entry["model_output_workspace_bytes"]
    assert actual.model_output_contract_identity == consumer_entry["model_output_consumer_identity"]
    assert route.model_output_contract == validate_output_contract_entry(consumer_entry)
    assert actual.execution_plan is None
    assert captured["qualifications"] == []  # No eligibility or default route is minted.


def test_absent_consumer_preserves_original_legacy_claim_translation(entry):
    value = entry | {"memory_overhead_bytes": "100"}
    cfg = config(value)
    assert cfg.demands is value["memory_demands"]
    assert cfg.memory_overhead_bytes == "100"
    assert cfg.model_output_workspace_bytes == 0
    stage = OmniLlamaBackend(cfg)._stage_backend_config()
    assert stage["memory_overhead_bytes"] == "100"
    assert "model_output_workspace_bytes" not in stage
    assert "model_output_consumer_identity" not in llamacpp_route_binding(value)


@pytest.mark.parametrize("kind", ["final", "tool", "read_url"])
def test_actual_profile_bridge_wrapper_accepts_only_complete_llama_consumer_trace(consumer_entry, kind, monkeypatch):
    from benchmarks.edge_agent import native_profile as profile
    from benchmarks.edge_agent.profile import ProfileRoute
    from vllm_omni.edge.agent import consumer_trace as trace

    monkeypatch.setattr(trace, "validate_strata_request_evidence",
                        lambda *a, **k: pytest.fail("llama Bridge cannot claim Strata placement"))
    events, route, evidence, trusted = llama_trace(consumer_entry, kind)
    actual_route = ProfileRoute(**route)
    assert profile._trace_complete(
        events, "READY", actual_route, evidence, trusted_task_binding=trusted,
    )
    wrong_stop = copy.deepcopy(events)
    next(event for event in wrong_stop if event["kind"] == "model_metrics")["payload"]["metrics"][
        "finish_reason"
    ] = "length"
    assert not profile._trace_complete(
        wrong_stop, "READY", actual_route, evidence, trusted_task_binding=trusted,
    )
    assert not profile._trace_complete(
        events[:-1], "READY", actual_route, evidence, trusted_task_binding=trusted,
    )
    assert not profile._trace_complete(
        events, "READY", actual_route, evidence,
        trusted_task_binding={"text": "another task", "read_url": None, "mode": "ordinary"},
    )


@pytest.mark.parametrize("layer", ["backend_metrics", "runtime_telemetry"])
@pytest.mark.parametrize("marker,value", [
    ("cpu_compute_verified", True),
    ("expert_compute_verified", True),
    ("cpu_expert_compute_verified", True),
    ("expert_final_storage_verified", True),
    ("execution_observation", {"status": "complete"}),
    ("observation_runtime", {"identity_sha256": "a" * 64}),
    ("gpu_observer_identity", {"status": "verified"}),
    ("native_io_observation", {"status": "complete"}),
    ("weight_tier_plan", {"budget": {}}),
    ("cpu_compute_verified", False),
    ("execution_observation", None),
])
def test_llama_metrics_provenance_markers_rejected_by_shared_and_actual_wrapper(consumer_entry, layer, marker, value):
    from benchmarks.edge_agent import native_profile as profile
    from benchmarks.edge_agent.profile import ProfileRoute
    from vllm_omni.edge.agent.consumer_trace import validate_consumer_trace

    events, route, evidence, trusted = llama_trace(consumer_entry)
    metrics = next(event["payload"]["metrics"] for event in events if event["kind"] == "model_metrics")
    backend_metrics = metrics.setdefault("backend_metrics", {})
    target = backend_metrics if layer == "backend_metrics" else backend_metrics.setdefault("runtime_telemetry", {})
    target[marker] = value
    with pytest.raises(ValueError, match="unknown compute provenance"):
        validate_consumer_trace(events, "READY", route, evidence, trusted_task_binding=trusted)
    assert not profile._trace_complete(
        events, "READY", ProfileRoute(**route), evidence, trusted_task_binding=trusted,
    )
