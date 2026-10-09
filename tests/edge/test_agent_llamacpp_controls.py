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
