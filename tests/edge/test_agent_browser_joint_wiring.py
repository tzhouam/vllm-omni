# SPDX-License-Identifier: Apache-2.0
"""Opt-in wiring source candidate: fake OS/backend, no neural/process launch.

Amounts below are synthetic ledger units, not browser allowance decisions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.edge_agent import native_profile
from benchmarks.edge_agent.paired_suite import ReadOnlyFixtureTools
from benchmarks.edge_agent.profile import ProfileRoute
from vllm_omni.edge.agent import browser_resources as br
from vllm_omni.edge.agent import native_app
from vllm_omni.edge.agent.omni_backend import OmniCompleteModelBackend
from vllm_omni.edge.agent.tools import ToolAction
from vllm_omni.engine.local_plan import CompanionResourceSpec, LocalPlanManager
from vllm_omni.engine.resource_ledger import ResourceLedger, ResourceUnavailable


def envelope_file(tmp_path: Path) -> dict[str, str]:
    evidence = tmp_path / "fixture-measured.json"
    evidence.write_text('{"unit_test_fixture":true}', encoding="utf-8")
    declaration = {"schema": "omni-browser-resource-envelope-v1", "purpose_id": "fixture-browser",
        "reviewed_for_joint_admission": True, "coverage": "exact_retained_handle_bound_set_only",
        "hard_process_cap": False, "all_descendants_covered": False,
        "memory_demands": {"host_ram": 20, "windows_commit": 30},
        "measurement_evidence": [{"path": evidence.name,
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}]}
    path = tmp_path / "fixture-envelope.json"
    path.write_text(json.dumps(declaration), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def resolved() -> br.ResolvedBrowserEnvelope:
    return br.ResolvedBrowserEnvelope(
        CompanionResourceSpec("fixture-browser", {"host_ram": 20, "windows_commit": 30},
                              "a" * 64, "fixture-envelope.json"),
        (("fixture-measurement.json", "b" * 64),),
    )


def test_native_optional_resolver_is_off_without_declaration(tmp_path: Path) -> None:
    assert native_app._browser_companion_from_config({}, config_dir=tmp_path) is None
    with pytest.raises(ValueError, match="explicit reviewed"):
        native_app._browser_companion_from_config({}, config_dir=tmp_path, browser_factory=lambda **kwargs: None)


def test_native_resolves_exact_relative_envelope_without_starting_registry(tmp_path: Path) -> None:
    declaration = envelope_file(tmp_path)
    declaration["path"] = Path(declaration["path"]).name
    calls: list[Any] = []
    companion = native_app._browser_companion_from_config(
        {"browser_resource_envelope": declaration}, config_dir=tmp_path,
        browser_factory=lambda **kwargs: calls.append(kwargs),
    )
    assert companion.is_cold() and companion.resource_spec.envelope_sha256 == declaration["sha256"]
    assert companion._registry is None and companion._browser is None and calls == []
    declaration["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="hash differs"):
        native_app._browser_companion_from_config({"browser_resource_envelope": declaration}, config_dir=tmp_path)


def test_native_build_attaches_companion_before_route_admission_and_final_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = {"route_id": "r", "artifact_id": "artifact", "model": "fixture-model",
             "backend": "external.llamacpp.text.v1", "placement": "cpu",
             "memory_demands": {"host_ram": 50, "windows_commit": 40}}
    config = {"routes": [entry], "browser_resource_envelope": envelope_file(tmp_path)}
    path = tmp_path / "native.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    hardware = {"host_ram_available_bytes": 100, "vram_available_bytes": None,
                "windows_commit_available_bytes": 100, "gpu_name": None, "power_condition": "AC"}
    captured: dict[str, Any] = {}
    class Controller:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
    monkeypatch.setattr(native_app.sys, "platform", "win32")
    monkeypatch.setattr(native_app, "_hardware_snapshot", lambda **kwargs: hardware)
    monkeypatch.setattr(native_app, "_fingerprint", lambda value: "fixture-runtime")
    monkeypatch.setattr(native_app, "llamacpp_route_binding", lambda value: None)
    monkeypatch.setattr(native_app, "llama_config_from_entry", lambda value, **kwargs: object())
    monkeypatch.setattr(native_app, "OmniLlamaBackend", lambda value: SimpleNamespace(resident=False))
    monkeypatch.setattr(native_app, "EncryptedMemoryStore", lambda value: object())
    monkeypatch.setattr(native_app, "AgentController", Controller)
    controller, _ = native_app.build_controller(path)
    companion = controller.browser_resource_companion
    assert isinstance(companion, br.WindowsBrowserResourceCompanion) and companion.is_cold()
    assert captured["tools"] is companion._tool_boundary
    assert captured["tools"]._browser is None
    assert captured["admit"].__self__._companion is companion
    assert captured["resource_finalize"].__self__ is captured["admit"].__self__
    assert captured["backends"]["r"]._coordinator._companion is companion
    assert captured["routes"][0].route_id == "r" and captured["routes"][0].artifact_id == "artifact"
    assert captured["resource_finalize"]()["ledger"]["owners"] == []


def actual_adapter_manager(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    """Keep real adapter construction and StageRuntime reservation lifecycle.

    Override only planning/launch/finalization with fake graph clients; do not
    replace StageRuntime._reserve_stage_resources or the shared resource ledger.
    """
    from vllm_omni.engine import stage_runtime as sr
    capacities = {"host_ram": 100, "windows_commit": 100}
    demands = {"host_ram": 50, "windows_commit": 40}
    config = SimpleNamespace(route_id="r", placement="cpu", capacities=capacities,
                             demands=demands, start_timeout_s=1)
    backend = OmniCompleteModelBackend(config)
    backend._stage_backend_config = lambda: {"name": "fixture.graph"}
    runtime_instances: list[Any] = []
    client_instances: list[Any] = []
    def plans(runtime: Any) -> list[Any]:
        runtime_instances.append(runtime)
        configured = runtime._stage_configs[0].engine_args["resource_budget"]
        assert dict(configured["capacities"]) == capacities
        assert dict(configured["demands"]) == demands
        replica = SimpleNamespace(metadata=SimpleNamespace(stage_id=0, stage_type="graph",
            engine_input_source=[], final_output=True), replica_id=0, launch_mode="local",
            stage_cfg=SimpleNamespace(engine_args={"resource_budget": {
                "capacities": capacities, "demands": demands}}))
        return [SimpleNamespace(stage_idx=0, replicas=[replica])]
    def initialize(runtime: Any, _plans: Any, _timeout: Any) -> dict[int, list[Any]]:
        token = runtime._resource_reservations[(0, 0)]
        assert token is backend._shared_reservation and runtime.resource_ledger is backend._shared_ledger
        assert len(runtime.resource_ledger.snapshot()["owners"]) == 2
        class Client:
            stage_type = "graph"
            stage_id = replica_id = 0
            def __init__(self) -> None:
                self.closed = False
                self._proc = SimpleNamespace(pid=987654, poll=lambda: 0 if self.closed else None)
                self.execution_plan = {"requested_device": "cpu", "observed_model_placement": "cpu",
                                       "reserved_bytes": demands}
            def check_health(self) -> None:
                if self.closed:
                    raise RuntimeError("fake graph client retired")
            def shutdown(self) -> None:
                self.closed = True
                assert runtime.resource_ledger.release(token, drained=True) or runtime.resource_ledger.was_released(token)
        client = Client()
        client_instances.append(client)
        return {0: [client]}
    def finalize(runtime: Any, _plans: Any, clients: Any) -> None:
        runtime.stage_pools = [SimpleNamespace(stage_client=clients[0][0], clients=clients[0])]
    monkeypatch.setattr(sr.StageRuntime, "_prepare_stage_plans", plans)
    monkeypatch.setattr(sr.StageRuntime, "_before_initialize_stage_replicas", lambda *args: None)
    monkeypatch.setattr(sr.StageRuntime, "_initialize_stage_replicas", initialize)
    monkeypatch.setattr(sr.StageRuntime, "_finalize_initialized_stages", finalize)
    companion = br.WindowsBrowserResourceCompanion(resolved(),
        browser_factory=lambda **kwargs: pytest.fail("no browser launch is needed for the stage lease test"),
        registry_factory=lambda generation: pytest.fail("no registry construction is needed"))
    ledger = ResourceLedger(capacities)
    route = SimpleNamespace(route_id="r", placement="cpu", memory_demands=demands)
    manager = LocalPlanManager(routes=[route], backends={"r": backend}, capacities=capacities,
        free_bytes=lambda: dict(capacities), resource_ledger=ledger, companion=companion)
    return manager, backend, companion, ledger, runtime_instances, client_instances


def test_actual_complete_model_stage_adopts_exact_model_only_token_with_tool_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, backend, companion, ledger, runtimes, clients = actual_adapter_manager(monkeypatch)
    manager.start("r")
    model_token, tool_token = backend._shared_reservation, companion._reservation
    assert backend.resident and len(runtimes) == len(clients) == 1
    assert runtimes[0]._resource_reservations[(0, 0)] is model_token
    assert dict(model_token.demands) == {"host_ram": 50, "windows_commit": 40}
    assert ledger.snapshot()["reserved"] == {"host_ram": 70, "windows_commit": 70}
    assert backend.close() is True
    assert ledger.was_released(model_token) and ledger.owns(tool_token)
    assert ledger.snapshot()["reserved"] == {"host_ram": 20, "windows_commit": 30}
    assert manager.finalize_close()["ledger"]["owners"] == []


def test_actual_complete_model_cancel_receipt_releases_model_without_forging_global_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, backend, companion, ledger, _runtimes, _clients = actual_adapter_manager(monkeypatch)
    manager.start("r")
    backend._last_turn_request_id = "cancel-step"
    wrapped = manager.wrappers()["r"]
    model_token, tool_token = backend._shared_reservation, companion._reservation
    assert wrapped.request_state_released("cancel-step") is True
    proof = wrapped.release_evidence
    assert proof["exact_resource_token_released"] and proof["release_scope"] == "exact_model_lease"
    assert proof["tool_lease_retained"] is True and proof["host_ledger_empty"] is False
    assert ledger.was_released(model_token) and ledger.owns(tool_token)
    assert manager.finalize_close()["ledger"]["owners"] == []


def test_historical_close_receipt_survives_reset_but_is_not_current_release_authority() -> None:
    companion = br.WindowsBrowserResourceCompanion(resolved())
    ledger = ResourceLedger({"host_ram": 100, "windows_commit": 100})
    token = ledger.reserve("tool-1", companion.resource_spec.memory_demands)
    companion.bind_resource_lease(ledger, token, joint_generation="joint-1", guard=lambda: None)
    assert companion.close() is True and ledger.release(token, drained=True)
    companion.reset_after_verified_drain()
    assert companion.release_evidence is None and companion.last_close_evidence["joint_generation"] == "joint-1"
    new = ledger.reserve("tool-2", companion.resource_spec.memory_demands)
    companion.bind_resource_lease(ledger, new, joint_generation="joint-2", guard=lambda: None)
    assert companion.release_evidence is None
    assert companion.last_close_evidence["resource_owner"] == "tool-1"
    assert companion.close() and companion.last_close_evidence["resource_owner"] == "tool-2"


@pytest.mark.parametrize("browser_headless", [False, True])
def test_profile_managed_factory_preserves_read_only_tools_and_lazy_guard(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, browser_headless: bool) -> None:
    route = ProfileRoute("r", "model", "artifact", "revision", "a" * 64,
                         "Q4", "external.llamacpp.text.v1", "cpu")
    captured: dict[str, Any] = {}
    calls: list[str] = []
    companion = br.WindowsBrowserResourceCompanion(resolved(),
        registry_factory=lambda generation: SimpleNamespace())
    class Browser:
        def __init__(self, **kwargs: Any) -> None:
            calls.append("browser_constructed")
            captured["browser_kwargs"] = kwargs
        def describe_target(self, selector: str) -> dict[str, Any]:
            return {"href": "https://other.example/"}
        def current_url(self) -> str:
            return "http://127.0.0.1:1234/"
    monkeypatch.setattr(native_profile, "ManagedEdgeBrowser", Browser)
    monkeypatch.setattr(native_profile, "_FixtureForegroundScreen", lambda: object())
    backend = SimpleNamespace(execution_plan=None)
    def start() -> None:
        calls.append("model_start")
        assert isinstance(captured["controller"].tools, ReadOnlyFixtureTools)
        assert "browser_constructed" not in calls
        backend.execution_plan = {"requested_device": "cpu"}
    backend.start = start
    def build(path: Path, **kwargs: Any) -> tuple[Any, Any]:
        assert set(kwargs) == {"tool_boundary_factory", "browser_factory"}
        def owned_browser_factory(**owned_kwargs: Any) -> Any:
            captured["owned_browser_kwargs"] = owned_kwargs
            return kwargs["browser_factory"](**owned_kwargs)
        companion._browser_factory = owned_browser_factory
        boundary = companion.make_tool_boundary(boundary_factory=kwargs["tool_boundary_factory"])
        controller = SimpleNamespace(tools=boundary, browser_resource_companion=companion,
            routes=[SimpleNamespace(route_id="r")], backends={"r": backend},
            limits=SimpleNamespace(max_model_steps=2), add_listener=lambda callback: None,
            admit=lambda route: SimpleNamespace(admitted=True), resource_snapshot=lambda: {"fixture": True})
        captured["controller"] = controller
        return controller, {}
    monkeypatch.setattr(native_app, "build_controller", build)
    telemetry = SimpleNamespace(sample=lambda: {"ram_used_bytes": 1},
        attach_browser_companion=lambda value: calls.append("attached"))
    bridge = native_profile.NativeProfileBridge(native_config={"routes": [{"route_id": "r"}],
        "browser_resource_envelope": envelope_file(tmp_path)}, config_root=tmp_path / "configs",
        private_root=tmp_path, fixture_origin="http://127.0.0.1:1234", telemetry=telemetry,
        browser_headless=browser_headless)
    preparation = asyncio.run(bridge.prepare(route))
    assert preparation.cold_start_confirmed and calls == ["attached", "model_start"]
    assert bridge.controller.tools is captured["controller"].tools
    with pytest.raises(ResourceUnavailable):
        bridge.controller.tools.execute(ToolAction("browser_read"))  # No bound tool token yet in this fake builder.
    ledger = ResourceLedger({"host_ram": 100, "windows_commit": 100})
    token = ledger.reserve("tool", companion.resource_spec.memory_demands)
    companion.bind_resource_lease(ledger, token, joint_generation="joint-1", guard=lambda: calls.append("guard"))
    with pytest.raises(PermissionError, match="forbids"):
        bridge.controller.tools.execute(ToolAction("browser_post", {}))
    with pytest.raises(PermissionError, match="external link"):
        bridge.controller.tools.execute(ToolAction("browser_follow", {"selector": "a"}))
    assert "browser_constructed" in calls
    assert captured["browser_kwargs"]["resource_guard"] is not None
    assert captured["browser_kwargs"]["process_observer"] is not None
    assert captured["browser_kwargs"]["headless"] is browser_headless
    assert captured["browser_kwargs"]["resource_guard"] is captured["owned_browser_kwargs"]["resource_guard"]
    assert captured["browser_kwargs"]["process_observer"] is captured["owned_browser_kwargs"]["process_observer"]
    assert bridge._profile_browser is companion._browser


def test_telemetry_keeps_agent_model_and_browser_process_sets_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = object.__new__(native_profile.WindowsTelemetry)
    telemetry.expected_power = "AC"
    telemetry._psutil = SimpleNamespace(sensors_battery=lambda: SimpleNamespace(power_plugged=True),
        virtual_memory=lambda: SimpleNamespace(total=1000, available=500))
    telemetry._nvml = telemetry._gpu = None
    telemetry._native_gpu_enabled = False
    telemetry._process_registry = SimpleNamespace(sample=lambda: {"roles": ["agent", "model"]})
    telemetry._browser_companion = None
    companion = SimpleNamespace(sample_process_memory=lambda: {"roles": ["playwright_node", "edge_browser"]})
    monkeypatch.setattr(native_app, "_windows_commit_available", lambda: 600)
    telemetry.attach_browser_companion(companion)
    sampled = telemetry.sample()
    assert sampled["bound_process_cpu_memory"]["roles"] == ["agent", "model"]
    assert sampled["bound_browser_companion_memory"]["roles"] == ["playwright_node", "edge_browser"]
    assert "sum" not in sampled["browser_process_memory_scope"]
    telemetry.detach_browser_companion(companion)
    assert telemetry._browser_companion is None


def test_profile_failed_controller_close_retains_companion_for_quarantine_and_retry(tmp_path: Path) -> None:
    companion = SimpleNamespace(last_close_evidence={"owned_work_drained": False})
    def fail() -> None:
        raise RuntimeError("fixture owned browser unretired")
    calls: list[Any] = []
    bridge = native_profile.NativeProfileBridge(native_config={}, config_root=tmp_path,
        private_root=tmp_path, fixture_origin="http://127.0.0.1:1234",
        telemetry=SimpleNamespace(detach_browser_companion=lambda value: calls.append(value)))
    bridge.controller = SimpleNamespace(close=fail,
        resource_snapshot=lambda: {"synthetic_fixture": True, "quarantined": True})
    bridge._browser_companion = companion
    with pytest.raises(RuntimeError, match="unretired"):
        bridge.close()
    assert bridge.controller is not None and bridge._browser_companion is companion and calls == []


def test_profile_final_close_uses_verified_historical_companion_receipt_after_reset(tmp_path: Path) -> None:
    proof = {"schema": "omni-companion-release-v1", "owned_work_drained": True,
        "required_ownership_verified": True, "all_descendants_retired": False,
        "quarantine_required": False, "browser_close_receipt": {"worker_joined": True}}
    companion = SimpleNamespace(last_close_evidence=proof, release_evidence=None)
    calls: list[Any] = []
    telemetry = SimpleNamespace(
        end_controller_processes=lambda **kwargs: calls.append(kwargs) or {"attribution_close_verified": True},
        detach_browser_companion=lambda value: calls.append(value))
    bridge = native_profile.NativeProfileBridge(native_config={}, config_root=tmp_path,
        private_root=tmp_path, fixture_origin="http://127.0.0.1:1234", telemetry=telemetry,
        process_memory_attribution=True)
    # This inert controller owns no real ledger; its empty snapshot is a
    # synthetic fixture, not production cleanup evidence.
    bridge.controller = SimpleNamespace(close=lambda: None, resource_snapshot=lambda: {})
    bridge._browser_companion = companion
    bridge.close()
    assert calls[0]["browser_companion_release"] is proof
    assert calls[0]["browser_close_receipt"] == {"worker_joined": True}
    assert calls[1] is companion and bridge._browser_companion is None and bridge.controller is None


def test_failed_close_cannot_borrow_prior_generation_success_for_attribution(tmp_path: Path) -> None:
    prior = {"owned_work_drained": True, "required_ownership_verified": True}
    companion = SimpleNamespace(last_close_evidence=prior, release_evidence=None)
    observed: list[Any] = []
    def close() -> None:
        raise RuntimeError("current owner unretired")
    telemetry = SimpleNamespace(end_controller_processes=lambda **kwargs: observed.append(kwargs)
        or {"attribution_close_verified": False}, detach_browser_companion=lambda value: pytest.fail("retain quarantine"))
    bridge = native_profile.NativeProfileBridge(native_config={}, config_root=tmp_path,
        private_root=tmp_path, fixture_origin="http://127.0.0.1:1234", telemetry=telemetry,
        process_memory_attribution=True)
    bridge.controller = SimpleNamespace(close=close,
        resource_snapshot=lambda: {"synthetic_fixture": True, "quarantined": True})
    bridge._browser_companion = companion
    with pytest.raises(RuntimeError, match="current owner"):
        bridge.close()
    assert observed[0]["browser_companion_release"] == {}
    assert observed[0]["browser_close_receipt"] is None and bridge._browser_companion is companion


@pytest.mark.parametrize("tool_verified", [False, True])
def test_telemetry_separate_agent_model_drain_does_not_prove_browser_drain(tool_verified: bool) -> None:
    telemetry = object.__new__(native_profile.WindowsTelemetry)
    telemetry._process_memory_closures = []
    telemetry._process_registry = SimpleNamespace(close=lambda: {
        "agent_alive_observed": True, "model_retirement_verified": True,
        "observer_handles_closed": True, "children": {"bound_set_drain_verified": True}})
    proof = {"schema": "omni-companion-release-v1", "owned_work_drained": tool_verified,
             "required_ownership_verified": tool_verified, "all_descendants_retired": False,
             "quarantine_required": not tool_verified}
    receipt = telemetry.end_controller_processes(browser_companion_release=proof)
    assert receipt["attribution_close_verified"] is tool_verified
    assert receipt["browser_companion_release"] == proof
    assert receipt["browser_attribution_scope"] == "separate_companion_exact_bound_set_no_descendant_claim"
