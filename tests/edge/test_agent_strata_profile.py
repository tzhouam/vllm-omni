# SPDX-License-Identifier: Apache-2.0
"""Evidence-contract fixtures only; these tests do not qualify a model/device."""

from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import sys
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.edge_agent import native_profile, profile
from benchmarks.edge_agent.native_profile import load_profile_routes
from benchmarks.edge_agent.paired_suite import build_paired_cases, evaluate_case
from vllm_omni.edge.agent import qualification
from vllm_omni.edge.agent.placement import (
    STRATA_BACKEND,
    STRATA_REVISION,
    evidence_sha256,
    preparation_placement_matches,
    result_placement_matches,
    strata_route_binding,
    validate_strata_profile_plan,
    validate_strata_request_evidence,
)
from vllm_omni.engine.backends.strata import (
    PINNED_STRATA_VERSION,
    _native_compute_observation,
    _verify_cache_bounds,
    _verify_load_configuration,
)
from vllm_omni.engine.weight_tiers import ArtifactFile, ArtifactManifest, WeightTierBudget, WeightTierPlan

MIB = 1 << 20


@pytest.fixture
def fixture():
    source = ArtifactManifest("fixture/Qwen", "a" * 40, "MIT", (
        ArtifactFile("model-00001-of-00002.gguf", 100, "b" * 64),
        ArtifactFile("model-00002-of-00002.gguf", 200, "c" * 64),
    ))
    runtime = ArtifactManifest("fixture/Strata", STRATA_REVISION, "MIT", (
        ArtifactFile("engine/strata.exe", 100, "d" * 64, role="runtime"),
    ))
    packed = ArtifactManifest("fixture/pack", "a" * 40 + "+pack", "MIT", (
        ArtifactFile("native_experts.txt", 100, "e" * 64, role="prepared_pack"),
    ))
    budget = WeightTierBudget(cpu_expert_cache_bytes=8 * MIB, gpu_expert_cache_bytes=4 * MIB)
    tier = WeightTierPlan("strata", source.manifest_sha256, STRATA_BACKEND, STRATA_REVISION, budget)
    config = {
        "name": STRATA_BACKEND, "runtime_revision": STRATA_REVISION,
        "artifact_manifest": source.to_dict(), "runtime_manifest": runtime.to_dict(),
        "prepared_pack_manifest": packed.to_dict(), "weight_tier_plan": tier.to_dict(),
        "conversion_manifest": {"complete": True, "source_manifest_sha256": source.manifest_sha256,
                                "prepared_manifest_sha256": packed.manifest_sha256,
                                "tool_revision": STRATA_REVISION, "conversions": []},
        "engine_file": "engine/strata.exe", "python_sha256": "f" * 64,
        "python_environment": {"executable_sha256": "f" * 64, "sys_version": "fixture Python 3.12",
                               "dependencies": {name: "fixture-1.0" for name in
                                                ("numpy", "jinja2", "regex", "PyYAML", "psutil", "Pillow", "gguf")}},
        "expert_ram_budget_bytes": 8 * MIB, "gpu_budget_bytes": 16 * MIB,
        "gpu_total_bytes": 32 * MIB, "context_tokens": 4096, "kv_type": "fp16",
        "expert_profile_file": None,
    }
    entry = {"backend": STRATA_BACKEND, "backend_config": config, "route_id": "strata",
             "model": source.checkpoint, "artifact_id": "strata:" + evidence_sha256(config),
             "placement": "cpu+cuda:0"}
    lineage = {"routes": {"strata": {"artifact_manifest_sha256": source.manifest_sha256,
               "checkpoint_revision": source.revision, "precision": "Q4", "lineage_verified": False}}}
    routes, provenance = load_profile_routes({"routes": [entry]}, lineage)
    route = routes[0]
    binding = route.backend_identity
    control = {"schema": "omni-strata-explicit-cache-v2", "layout_sha256": "e" * 64,
               "layout_version": 4, "layers": 48, "max_blob_bytes": MIB, "alignment_bytes": 256,
               "aligned_max_blob_bytes": MIB, "budget_bytes": 4 * MIB,
               "requested_slots": 4, "allocation_upper_bytes": 4 * MIB}
    info = {"engine": PINNED_STRATA_VERSION, "context": 4096, "kv": "fp16", "spec": 2, "lookup": 0,
            "conversation_cache_mib": 0, "conversation_cache_slots": 0, "pool_workers": 2,
            "expert_slots": 4, "expert_cache_mib": 4, "arena_mib": 5}
    memory = {"gpu_name_sha256": "1" * 64}
    load = _verify_load_configuration(
        {"gpu": {"local_index": 0, "name_sha256": "1" * 64},
         "cpu_pool": {"tasks_per_phase": 2, "participating_threads": 3},
         "expert_workers": {"workers": 2, "host_thread": True}, "native_pack": True, "native_starts": 1},
        info, memory, gpu=0, context=4096, kv="fp16", verify_window=2,
    )
    controls = {**binding["expected_controls"], "gpu_expert_cache": control}
    plan = {
        "backend": STRATA_BACKEND, "stage_id": 0, "worker_generation": "worker-generation",
        "runtime_revision": STRATA_REVISION, "requested_device": "cpu+cuda:0",
        "observed_model_placement": None, "observed_compute_units": None,
        "verified_execution_configuration": "cpu+cuda:0", "placement_evidence_level": "native_loaded_configuration",
        "execution_configuration_evidence": load, "fresh_memory_admission": memory,
        "context_tokens": 4096, "kv_type": "fp16", "native_verify_window": 2, "spec_tokens": 0,
        "artifact_manifest_sha256": source.manifest_sha256,
        "runtime_manifest_sha256": runtime.manifest_sha256, "prepared_manifest_sha256": packed.manifest_sha256,
        "conversion_manifest": config["conversion_manifest"], "python_environment": config["python_environment"],
        "weight_tier_plan": tier.to_dict(), "route_controls": controls,
        "route_controls_sha256": evidence_sha256(controls, ascii=True), "gpu_expert_cache_control": control,
        "gpu_budget_bytes": 16 * MIB, "expert_ram_budget_bytes": 8 * MIB,
        "expert_cache_component_bounds": _verify_cache_bounds(info, control, 8 * MIB, profiled=False),
        "gpu_aggregate_hard_cap_verified": False, "three_tier_memory_qualified": False, "file_cache_peak_bytes": None,
    }
    counters = {"generated": 2, "prompt_tokens": 18, "hits": 216, "lookups": 915,
                "offloaded": 45, "prompt_read": 0}
    terminal = {"step": 0, "metrics": {
        "stage_event": {"request_id": "request-step-0", "stage_id": 0, "epoch": 1, "seq": 3,
                        "kind": "text", "worker_generation": "worker-generation", "terminal": True},
        "backend_metrics": {"runtime_telemetry": {
            "native_compute": _native_compute_observation(counters, verified_native_pack=True, gpu=0),
            "physical_ssd_read_bytes": None,
        }},
    }}
    evidence = {"execution_plan": plan, "loaded_plan_sha256": evidence_sha256(plan),
                "terminal_model_metrics": [terminal], "model_prompt_step_count": 1}
    events = [
        {"request_id": "request", "epoch": 1, "seq": 1, "kind": "user_observation", "payload": {}},
        {"request_id": "request", "epoch": 1, "seq": 2, "kind": "route", "payload": {
            "route_id": route.route_id, "model": route.model_id, "artifact_id": route.artifact_id,
            "backend": STRATA_BACKEND, "actual_placement": None,
            "verified_execution_configuration": "cpu+cuda:0", "execution_configuration_evidence": load,
        }},
        {"request_id": "request", "epoch": 1, "seq": 3, "kind": "model_metrics", "payload": terminal},
        {"request_id": "request", "epoch": 1, "seq": 4, "kind": "final", "payload": {"answer": "14"}},
    ]
    result = profile.AgentRunResult("14", True, route.model_id, route.artifact_id, None, STRATA_BACKEND, evidence)
    prep = profile.Preparation(True, route.artifact_id, None,
                               {"execution_plan": plan, "loaded_plan_sha256": evidence_sha256(plan)})
    return SimpleNamespace(entry=entry, lineage=lineage, route=route, provenance=provenance, plan=plan,
                           evidence=evidence, events=events, result=result, prep=prep)


def test_scoped_strata_trace_keeps_whole_model_placement_unknown(fixture):
    route = asdict(fixture.route)
    validate_strata_request_evidence(fixture.evidence, route, events=fixture.events)
    assert fixture.provenance["strata"]["lineage_verified"] is False
    assert preparation_placement_matches(asdict(fixture.prep), route)
    assert result_placement_matches(asdict(fixture.result), route)
    assert native_profile._trace_complete(fixture.events, "14", fixture.route, fixture.evidence)
    assert qualification._trace_complete(fixture.events, "14", route, fixture.evidence)
    assert fixture.result.actual_placement is None and fixture.prep.actual_placement is None
    assert fixture.evidence["terminal_model_metrics"][0]["metrics"]["backend_metrics"]["runtime_telemetry"][
        "native_compute"]["scope"] == "routed_decode_experts_only"


@pytest.mark.parametrize("field", ["artifact_manifest_sha256", "runtime_manifest_sha256", "prepared_manifest_sha256",
                                 "runtime_revision", "python_environment", "conversion_manifest", "weight_tier_plan"])
def test_source_runtime_and_conversion_tampering_fails_closed(fixture, field):
    fixture.plan[field] = "0" * 64
    with pytest.raises((ValueError, RuntimeError)):
        validate_strata_profile_plan(fixture.plan, asdict(fixture.route))


@pytest.mark.parametrize("change", ["missing", "hash", "budget", "zero_slots", "layout", "info", "context"])
def test_cache_identity_and_observed_component_bounds_are_checked(fixture, change):
    plan = fixture.plan
    if change == "missing":
        plan.pop("route_controls_sha256")
    elif change == "hash":
        plan["route_controls_sha256"] = "0" * 64
    elif change == "info":
        plan["expert_cache_component_bounds"]["gpu_expert_cache"]["allocation_upper_bytes"] += 1
    elif change == "context":
        plan["context_tokens"] = 8192
    else:
        control = plan["route_controls"]["gpu_expert_cache"]
        key, value = {"budget": ("budget_bytes", 8 * MIB), "zero_slots": ("requested_slots", 0),
                      "layout": ("layout_sha256", "0" * 64)}[change]
        control[key] = value
        plan["route_controls_sha256"] = evidence_sha256(plan["route_controls"], ascii=True)
    with pytest.raises((ValueError, RuntimeError)):
        validate_strata_profile_plan(plan, asdict(fixture.route))


@pytest.mark.parametrize("change", ["unknown", "scope", "wrong_count", "negative", "bool_count", "hits",
                                   "wrong_gpu", "request", "worker", "step", "step_count", "plan_hash", "io_scope"])
def test_per_request_native_done_evidence_is_strictly_scoped_and_bound(fixture, change):
    evidence = fixture.evidence
    terminal = evidence["terminal_model_metrics"][0]
    native = terminal["metrics"]["backend_metrics"]["runtime_telemetry"]["native_compute"]
    if change == "unknown":
        native["counters"] = None
    elif change == "scope":
        native["scope"] = "whole_model"
    elif change == "wrong_count":
        native["cpu_expert_entries"] += 1
    elif change in {"negative", "bool_count", "hits"}:
        key, value = {"negative": ("generated", -1), "bool_count": ("lookups", True), "hits": ("hits", 999)}[change]
        native["counters"][key] = value
    elif change == "wrong_gpu":
        native["units"] = ["cpu", "cuda:1"]
    elif change in {"request", "worker"}:
        terminal["metrics"]["stage_event"]["request_id" if change == "request" else "worker_generation"] = "other"
    elif change == "step":
        terminal["step"] = 1
    elif change == "step_count":
        evidence["model_prompt_step_count"] = True
    elif change == "io_scope":
        terminal["metrics"]["backend_metrics"]["runtime_telemetry"]["logical_file_read_bytes"] = 100000
    else:
        evidence["loaded_plan_sha256"] = "0" * 64
    with pytest.raises((ValueError, RuntimeError)):
        validate_strata_request_evidence(evidence, asdict(fixture.route), events=fixture.events)


def test_generic_llama_matching_cannot_use_strata_proof(fixture):
    route = {**asdict(fixture.route), "backend": "external.llamacpp.text.v1"}
    result = {**asdict(fixture.result), "backend": route["backend"]}
    assert not result_placement_matches(result, route)
    assert not preparation_placement_matches(asdict(fixture.prep), route)
    result["actual_placement"] = route["expected_placement"]
    assert result_placement_matches(result, route)
    assert not result_placement_matches(asdict(fixture.result), route)


def test_strata_profile_lineage_requires_all_shard_manifest_and_runtime_identity(fixture):
    lineage = copy.deepcopy(fixture.lineage)
    lineage["routes"]["strata"]["artifact_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="all-shard"):
        load_profile_routes({"routes": [fixture.entry]}, lineage)
    entry = copy.deepcopy(fixture.entry)
    entry["backend_config"]["runtime_revision"] = "0" * 40
    with pytest.raises(ValueError, match="runtime revision"):
        strata_route_binding(entry)


def test_current_functional_strata_evidence_cannot_be_promoted(fixture):
    with pytest.raises(ValueError, match="aggregate memory"):
        qualification._verify_strata_release_observations(fixture.evidence)
    plan = fixture.plan
    plan.update(three_tier_memory_qualified=True, gpu_aggregate_hard_cap_verified=True, file_cache_peak_bytes=0)
    with pytest.raises(ValueError, match="physical SSD"):
        qualification._verify_strata_release_observations(fixture.evidence)


def test_independent_strata_gate_binds_raw_plan_metrics_and_scope(fixture, tmp_path):
    def bound(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        import hashlib
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    identity = {"backend": STRATA_BACKEND, "artifact_sha256": fixture.route.artifact_sha256,
                "route_id": fixture.route.route_id, "artifact_id": fixture.route.artifact_id,
                "checkpoint_revision": fixture.route.checkpoint_revision, "placement": fixture.route.expected_placement}
    observations = {"reported_placement": None, "artifact_sha256": fixture.route.artifact_sha256,
                    "scope": "loaded_configuration_and_per_request_routed_decode_experts",
                    "loaded_plan": bound("plan.json", fixture.plan),
                    "request_evidence": bound("request.json", fixture.evidence)}
    qualification._verify_gate_observations("runtime_placement", observations, identity, tmp_path, fixture.entry)
    observations["scope"] = "whole_model"
    with pytest.raises(ValueError, match="scoped"):
        qualification._verify_gate_observations("runtime_placement", observations, identity, tmp_path, fixture.entry)


def test_native_bridge_preparation_and_result_preserve_null_and_full_plan(fixture, tmp_path, monkeypatch):
    import vllm_omni.edge.agent.native_app as native_app

    backend = SimpleNamespace(execution_plan=None)
    backend.start = lambda: setattr(backend, "execution_plan", fixture.plan)
    listeners = []

    def submit(_prompt):
        for event in fixture.events:
            for listener in listeners:
                listener(event)
        future = Future()
        future.set_result("14")
        return future

    controller = SimpleNamespace(
        routes=[SimpleNamespace(route_id=fixture.route.route_id)], backends={fixture.route.route_id: backend},
        admit=lambda _: SimpleNamespace(admitted=True), tools=SimpleNamespace(close=lambda: None),
        add_listener=listeners.append, submit=submit,
    )
    monkeypatch.setattr(native_app, "build_controller", lambda _: (controller, {"fixture": True}))
    monkeypatch.setattr(native_profile, "_FixtureForegroundScreen", lambda: object())
    monkeypatch.setattr(native_profile, "ManagedEdgeBrowser", lambda **_: object())
    monkeypatch.setattr(native_profile, "ReadOnlyFixtureTools", lambda *_, **__: object())
    bridge = native_profile.NativeProfileBridge(
        native_config={"routes": [fixture.entry]}, config_root=tmp_path / "configs", private_root=tmp_path,
        fixture_origin="http://127.0.0.1:1234", telemetry=SimpleNamespace(sample=lambda: {"ram_used_bytes": 123}),
    )
    prep = asyncio.run(bridge.prepare(fixture.route))
    assert prep.actual_placement is None
    assert prep.details["execution_plan"] == fixture.plan
    assert prep.details["loaded_plan_sha256"] == evidence_sha256(fixture.plan)
    # The fixture records one real generate invocation identity without
    # claiming to execute a neural model during this bridge unit test.
    bridge._prompt_backend.identities = lambda: [{"step": 0, "sha256": "a" * 64, "utf8_bytes": 1, "chars": 1}]
    case = build_paired_cases("http://127.0.0.1:1234", 10)["code_tools"]["short"][0]
    result = asyncio.run(bridge.run(fixture.route, case, lambda *_: None))
    assert result.actual_placement is None and result.complete_agent_trace
    assert result.placement_evidence["execution_plan"]["route_controls_sha256"] == fixture.plan["route_controls_sha256"]


def test_profile_and_independent_audit_reconstruct_nullable_strata_evidence(fixture, tmp_path):
    counter = 0

    async def runner(route, case, emit):
        nonlocal counter
        counter += 1
        events, evidence = copy.deepcopy((fixture.events, fixture.evidence))
        request_id = f"request-{counter}"
        for event in events:
            event["request_id"] = request_id
        stage = evidence["terminal_model_metrics"][0]["metrics"]["stage_event"]
        stage.update(request_id=request_id + "-step-0", epoch=counter)
        events[2]["payload"] = evidence["terminal_model_metrics"][0]
        events[0]["payload"] = {"text": case.prompt}
        for event in events:
            emit("agent_event", event)
        emit("assistant_text_delta", "14")
        return profile.AgentRunResult("14", True, route.model_id, route.artifact_id, None, STRATA_BACKEND, evidence)

    conditions = profile.ProfileConditions("fixture", "Windows fixture", {}, {}, "AC", qualification.SUITE_ID, "fixture")
    cases = build_paired_cases("http://127.0.0.1:1234", 10)["code_tools"]
    summary = asyncio.run(profile.run_profile(
        routes=[fixture.route], cases_by_length=cases, runner=runner, evaluator=evaluate_case, conditions=conditions,
        output_dir=tmp_path, config=profile.ProfileConfig(0, 2, 0), prepare=lambda _: fixture.prep,
        telemetry=lambda: {"ram_used_bytes": 1},
        before_request=lambda *_: {"private_memory_reset": True},
    ))
    audited = qualification.audit_summary(summary.run_directory / "summary.json")
    assert audited.internally_valid, audited.errors
    assert audited.trace_verified and not audited.protocol_compliant
    assert summary.routes[fixture.route.route_id].actual_placement_matches
    assert summary.routes[fixture.route.route_id].cold_start_s is not None
    assert not summary.routes[fixture.route.route_id].qualification_fields(
        conditions=conditions, raw_evidence=str(summary.raw_jsonl), task_class="code_tools",
        memory_pass=True, cancel_recovery_pass=True, actual_placement_verified=True,
    )["actual_placement_verified"]


def test_process_gpu_samples_keep_unknown_startup_and_separate_generation_peaks(monkeypatch):
    import vllm_omni.edge.windows_gpu_memory as memory

    telemetry = native_profile.WindowsTelemetry.__new__(native_profile.WindowsTelemetry)
    telemetry.expected_power = "AC"
    telemetry._psutil = SimpleNamespace(
        sensors_battery=lambda: SimpleNamespace(power_plugged=True),
        virtual_memory=lambda: SimpleNamespace(total=1000, available=800),
    )
    telemetry._nvml = telemetry._gpu = None
    telemetry.clear_native_process()
    assert "native_process_gpu_memory" not in telemetry.sample()
    telemetry.bind_native_process(None)
    startup = telemetry.sample()
    assert startup["native_process_gpu_memory"]["status"] == "unknown"
    assert startup["native_process_gpu_memory"]["local_current_usage_bytes"] is None
    identity = {"status": "verified", "pid": 123, "creation_filetime_100ns": 456,
                "worker_generation": "one", "gpu": {"uuid": "GPU-fixture", "pci_bus_id": "0000:01:00.0",
                                                       "name_sha256": "a" * 64}}

    def observer(owner):
        return SimpleNamespace(sample=lambda: {
            **memory.unknown_observation("fixture", owner), "status": "observed",
            "local_current_usage_bytes": 120, "nonlocal_current_usage_bytes": 30,
        })

    monkeypatch.setattr(memory, "WindowsProcessGpuObserver", observer)
    telemetry.bind_native_process(identity)
    resident = telemetry.sample()
    telemetry.bind_native_process({**identity, "worker_generation": "two", "creation_filetime_100ns": 789})
    recovered = telemetry.sample()
    summary = profile._telemetry_summary([startup, resident, recovered])
    assert summary == qualification._telemetry_summary([startup, resident, recovered])
    scoped = summary["native_process_gpu_memory"]
    assert scoped["unknown_samples"] == 1 and scoped["sample_count"] == 3
    assert scoped["hard_cap_verified"] is False and scoped["nonlocal_is_additional_ram_pool"] is False
    peaks = scoped["sampled_peaks_by_generation"]
    assert len(peaks) == 2
    assert all(row["sampled_local_peak_bytes"] == 120 and row["sampled_nonlocal_peak_bytes"] == 30 for row in peaks)
    telemetry.clear_native_process()
    assert "native_process_gpu_memory" not in telemetry.sample()


def test_generic_profile_import_does_not_load_omni_or_optional_stage_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import benchmarks.edge_agent.profile; "
         "assert 'vllm_omni' not in sys.modules; assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
