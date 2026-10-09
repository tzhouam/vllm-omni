# SPDX-License-Identifier: Apache-2.0
"""Native-app opt-in contracts; synthetic units, no native/NN/process work.

These tests are authored source only in the private candidate. Amounts below
are unit fixtures, never browser allowance recommendations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.edge_agent import native_profile
from benchmarks.edge_agent.profile import ProfileRoute
from vllm_omni.edge import windows_bound_gpu_memory as gpu
from vllm_omni.edge import windows_process_memory as process
from vllm_omni.edge.agent import native_app
from vllm_omni.edge.agent import native_browser_gpu as ng
from vllm_omni.engine.local_plan import CompanionResourceSpec
from vllm_omni.engine.resource_ledger import Reservation, ResourceLedger, ResourceUnavailable

IDENTITY = {"uuid": "GPU-00000000-0000-0000-0000-000000000001",
            "pci_bus_id": "00000000:64:00.0", "name_sha256": "a" * 64}
CLAIMS = {"host_ram": 20, "windows_commit": 30, "vram": 10}
CAPACITIES = {"host_ram": 100, "windows_commit": 100, "vram": 100}


def pinned_json(path: Path, body: Any) -> dict[str, str]:
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def gpu_declaration(tmp_path: Path, **changes: Any) -> dict[str, str]:
    return pinned_json(tmp_path / "native-gpu.json", {
        "schema": "omni-browser-native-gpu-accounting-v1", "cuda_ordinal": 0, "gpu_pool": "vram",
        "gpu_identity": deepcopy(IDENTITY), "coverage": "exact_retained_handle_bound_set_only",
        "all_descendants_covered": False, "hard_process_cap": False, **changes})


def envelope_declaration(tmp_path: Path, *, gpu_claim: bool = True) -> dict[str, str]:
    evidence = pinned_json(tmp_path / "evidence.json", {"unit_fixture": True})
    return pinned_json(tmp_path / "envelope.json", {
        "schema": "omni-browser-resource-envelope-v1", "purpose_id": "unit-browser",
        "reviewed_for_joint_admission": True, "coverage": "exact_retained_handle_bound_set_only",
        "hard_process_cap": False, "all_descendants_covered": False,
        "memory_demands": dict(CLAIMS) if gpu_claim else {"host_ram": 20, "windows_commit": 30},
        "measurement_evidence": [evidence], "budget_basis": "synthetic_test_units",
        "scope": "test_only", "headroom": {"unit_fixture": True}, "estimate_not_measured": True})


def resolved_factory(tmp_path: Path) -> ng.NativeBrowserGpuRegistryFactory:
    declaration = gpu_declaration(tmp_path)
    resolved = ng.resolve_native_gpu_accounting(Path(declaration["path"]), expected_sha256=declaration["sha256"])
    spec = CompanionResourceSpec("unit-browser", CLAIMS, "b" * 64, "unit-envelope.json")
    return ng.NativeBrowserGpuRegistryFactory(resolved, spec, review_metadata={"budget_basis": "test_only"})


def source() -> dict[str, Any]:
    return {"gpu_identity": deepcopy(IDENTITY), "gpu_index": 0, "gpu_pool": "vram",
            "route_scope": "target_only_characterization", "route_identity": None}


def bind(factory: Any, *, generation: str = "generation-1", callback: Any = source) -> tuple[Any, Any]:
    ledger = ResourceLedger(CAPACITIES)
    token = ledger.reserve("tool-" + generation, CLAIMS)
    factory.bind_route_source(callback)
    factory.bind_resource_context(ledger, token, joint_generation=generation)
    return ledger, token


def terminal() -> dict[str, Any]:
    return {"schema": "omni-windows-bound-process-gpu-memory-v1", "drained": True,
            "adapter_handles_closed": True, "borrowed_process_handles_retained": False,
            "process_handles_closed_by_provider": 0}


def registry_terminal(generation: str, gpu_receipt: Any) -> dict[str, Any]:
    coverage = {"scope": "exact_retained_handle_bound_set_only", "all_descendants_covered": False,
                "binding_registry_overflow": False, "unadopted_handle_close_failures": 0,
                "unknown_count": 0, "unknown_details": []}
    return {"schema": "omni-windows-bound-process-memory-v1", "cohort_generation": generation,
        "observer_handles_closed": True, "all_descendants_retired": False, "coverage": deepcopy(coverage),
        "children": {"cohort_generation": generation, "bound_set_drain_verified": True,
            "known_bound_children_retired": True, "required_owned_bindings_verified": True,
            "all_descendants_retired": False, "coverage": coverage, "gpu_sampling_close": gpu_receipt}}


def fake_native(monkeypatch: pytest.MonkeyPatch, *, init_error: BaseException | None = None,
                join_error: BaseException | None = None, registry_error: BaseException | None = None,
                close_error: BaseException | None = None, close_receipt: Any = None,
                join_hook: Any = None, registry_hook: Any = None) -> dict[str, Any]:
    """Patch OS seams, keeping the actual provider class and factory lifecycle."""
    state: dict[str, Any] = {"events": [], "providers": [], "registries": []}
    def inventory(**kwargs: Any) -> list[Any]:
        assert kwargs == {"include_gpu_accounting": True}
        state["events"].append("inventory")
        return ["synthetic-native-inventory"]
    def initialize(self: Any, adapters: Any) -> None:
        assert adapters == ["synthetic-native-inventory"]
        state["events"].append("provider-init")
        state["providers"].append(self)
        self.unit_close_calls = 0
        if init_error is not None:
            raise init_error
    def close(self: Any) -> Any:
        self.unit_close_calls += 1
        state["events"].append("provider-close")
        if close_error is not None:
            raise close_error
        return deepcopy(terminal() if close_receipt is None else close_receipt)
    def join(**kwargs: Any) -> dict[str, Any]:
        state["events"].append("join")
        assert type(kwargs["gpu_provider"]) is gpu.WindowsRetainedProcessGpuProvider
        assert kwargs["gpu_identity"] == IDENTITY and kwargs["gpu_index"] == 0
        assert kwargs["gpu_pool"] == "vram" and kwargs["ledger_capacities"] == CAPACITIES
        if join_hook is not None:
            join_hook()
        if join_error is not None:
            raise join_error
        return {"unit_join": True, "ledger_pool": "vram", "model_operator_placement_verified": False}
    def registry(generation: str, **kwargs: Any) -> Any:
        state["events"].append("registry")
        if registry_hook is not None:
            registry_hook()
        if registry_error is not None:
            raise registry_error
        value = SimpleNamespace(generation=generation, unit_close_calls=0, **kwargs)
        def close_registry() -> Any:
            value.unit_close_calls += 1
            state["events"].append("registry-close")
            return registry_terminal(generation, value.gpu_provider.close())
        value.close = close_registry
        state["registries"].append(value)
        return value
    monkeypatch.setattr(native_app, "_dxgi_adapter_inventory", inventory)
    monkeypatch.setattr(native_app, "_resolve_native_gpu_pool_identity", join)
    monkeypatch.setattr(gpu.WindowsRetainedProcessGpuProvider, "__init__", initialize)
    monkeypatch.setattr(gpu.WindowsRetainedProcessGpuProvider, "close", close)
    monkeypatch.setattr(process, "WindowsProcessMemoryRegistry", registry)
    return state


@pytest.mark.parametrize("changes", [
    {"cuda_ordinal": True}, {"cuda_ordinal": -1}, {"cuda_ordinal": 1 << 31},
    {"gpu_pool": "host_ram"}, {"gpu_pool": "vram:1"}, {"hard_process_cap": True},
    {"all_descendants_covered": True}, {"coverage": "all_processes"}, {"unexpected": 1},
    {"gpu_identity": {**IDENTITY, "name_sha256": "not-a-hash"}},
])
def test_descriptor_rejects_ambiguous_target_and_scope(tmp_path: Path, changes: Any) -> None:
    declaration = gpu_declaration(tmp_path, **changes)
    with pytest.raises(ValueError):
        ng.resolve_native_gpu_accounting(Path(declaration["path"]), expected_sha256=declaration["sha256"])


def test_descriptor_opened_read_bound_and_hash_are_enforced(tmp_path: Path) -> None:
    declaration = gpu_declaration(tmp_path)
    path = Path(declaration["path"])
    with pytest.raises(ValueError, match="SHA256 differs"):
        ng.resolve_native_gpu_accounting(path, expected_sha256="f" * 64)
    raw = b" " * ((64 << 10) + 1)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="bound"):
        ng.resolve_native_gpu_accounting(path, expected_sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("raw", [
    b'{"schema":"first","schema":"second"}',
    b'{"gpu_identity":{"uuid":"first","uuid":"second"}}',
    b'{"cuda_ordinal":NaN}', b'{"cuda_ordinal":Infinity}',
])
def test_descriptor_rejects_duplicate_keys_and_nonfinite_json(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "ambiguous.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="duplicate|non-finite"):
        ng.resolve_native_gpu_accounting(path, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_default_config_opens_no_provider_and_gpu_allowance_cannot_silently_drop(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native_app, "_dxgi_adapter_inventory", lambda **kwargs: pytest.fail("no inventory"))
    assert native_app._browser_companion_from_config({}, config_dir=tmp_path) is None
    cpu = native_app._browser_companion_from_config(
        {"browser_resource_envelope": envelope_declaration(tmp_path, gpu_claim=False)}, config_dir=tmp_path)
    assert cpu.native_gpu_accounting_snapshot() is None
    with pytest.raises(ValueError, match="GPU allowance requires"):
        native_app._browser_companion_from_config(
            {"browser_resource_envelope": envelope_declaration(tmp_path)}, config_dir=tmp_path)
    with pytest.raises(ValueError, match="explicit reviewed"):
        native_app._browser_companion_from_config(
            {"browser_native_gpu_accounting": gpu_declaration(tmp_path)}, config_dir=tmp_path)


def test_native_opt_in_uses_normal_app_factory_and_preserves_review_scope(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    state = fake_native(monkeypatch)
    config = {"browser_resource_envelope": envelope_declaration(tmp_path),
              "browser_native_gpu_accounting": gpu_declaration(tmp_path)}
    companion = native_app._browser_companion_from_config(config, config_dir=tmp_path,
        browser_factory=lambda **kwargs: SimpleNamespace(**kwargs))
    assert companion.is_cold() and state["events"] == []
    companion.bind_native_gpu_accounting_source(source)
    ledger = ResourceLedger(CAPACITIES)
    token = ledger.reserve("tool", CLAIMS)
    companion.bind_resource_lease(ledger, token, joint_generation="generation-1", guard=lambda: None)
    assert state["events"] == []  # Binding stores identity; admission does not open adapters.
    companion.browser()
    report = companion.native_gpu_accounting_snapshot()
    assert report["current"]["status"] == "transferred"
    assert report["current"]["source"]["route_scope"] == "target_only_characterization"
    assert report["current"]["join"]["ledger_pool"] == "vram"
    assert report["review_metadata"]["budget_basis"] == "synthetic_test_units"
    assert report["review_metadata"]["estimate_not_measured"] is True
    assert report["memory_allowance_selected"] is report["model_operator_placement_verified"] is False
    assert state["providers"][0].unit_close_calls == 0 and ledger.owns(token)
    with pytest.raises(ValueError, match="custom registry"):
        native_app._browser_companion_from_config(config, config_dir=tmp_path,
            browser_registry_factory=lambda generation: object())


@pytest.mark.parametrize("kind", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_partial_provider_constructor_preserves_primary_and_cleans_once(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, kind: Any) -> None:
    primary = kind("partial native constructor")
    state = fake_native(monkeypatch, init_error=primary)
    factory = resolved_factory(tmp_path)
    bind(factory)
    with pytest.raises(kind) as caught:
        factory("generation-1")
    assert caught.value is primary
    assert state["events"] == ["inventory", "provider-init", "provider-close"]
    assert state["providers"][0].unit_close_calls == 1
    row = factory.snapshot()["current"]
    assert row["phase"] == "provider_constructor" and row["provider_close_verified"] is True
    assert factory.snapshot()["retained_failed_provider_count"] == 0


@pytest.mark.parametrize("phase", ["join", "registry"])
@pytest.mark.parametrize("kind", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_downstream_failure_cleans_owned_provider_and_never_masks_primary(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, phase: str, kind: Any) -> None:
    primary = kind("exact original failure")
    state = fake_native(monkeypatch, **{phase + "_error": primary})
    factory = resolved_factory(tmp_path)
    bind(factory)
    with pytest.raises(kind) as caught:
        factory("generation-1")
    assert caught.value is primary and state["providers"][0].unit_close_calls == 1
    assert factory.snapshot()["current"]["provider_close_verified"] is True
    assert factory.snapshot()["current"]["provider_transferred_to_registry"] is False


@pytest.mark.parametrize("failure", ["interrupt", "undrained", "missing", "bool_count", "borrowed"])
def test_unverified_cleanup_retains_reference_and_blocks_before_new_inventory(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    receipt = terminal()
    close_error = None
    if failure == "interrupt":
        close_error = KeyboardInterrupt("cleanup must not replace primary")
    elif failure == "undrained":
        receipt["drained"] = False
    elif failure == "missing":
        del receipt["adapter_handles_closed"]
    elif failure == "bool_count":
        receipt["process_handles_closed_by_provider"] = False
    else:
        receipt["borrowed_process_handles_retained"] = True
    primary = ValueError("artifact validation failed")
    state = fake_native(monkeypatch, registry_error=primary, close_error=close_error, close_receipt=receipt)
    factory = resolved_factory(tmp_path)
    ledger, token = bind(factory)
    with pytest.raises(ValueError) as caught:
        factory("generation-1")
    assert caught.value is primary and factory._retained_failed_providers[0] is state["providers"][0]
    before = list(state["events"])
    assert ledger.release(token, drained=True)
    fresh = ledger.reserve("tool-generation-2", CLAIMS)
    with pytest.raises(ResourceUnavailable, match="terminally quarantined"):
        factory.bind_resource_context(ledger, fresh, joint_generation="generation-2")
    with pytest.raises(ResourceUnavailable, match="terminally quarantined"):
        factory("generation-2")
    factory.snapshot()
    assert state["events"] == before and state["providers"][0].unit_close_calls == 1


def test_cleanup_bookkeeping_interrupt_keeps_constructing_reference(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    primary = ValueError("registry construction")
    state = fake_native(monkeypatch, registry_error=primary)
    factory = resolved_factory(tmp_path)
    bind(factory)
    def fail(*args: Any) -> None:
        raise SystemExit("bookkeeping interruption")
    monkeypatch.setattr(factory, "_close_owned_failure", fail)
    with pytest.raises(ValueError) as caught:
        factory("generation-1")
    assert caught.value is primary and factory._constructing_provider is state["providers"][0]
    assert factory.snapshot()["retained_failed_provider_count"] == 1
    before = list(state["events"])
    with pytest.raises(ResourceUnavailable, match="terminally quarantined"):
        factory("generation-2")
    assert state["events"] == before and state["providers"][0].unit_close_calls == 0


@pytest.mark.parametrize("mutation", ["equal_token", "release_during_source", "quarantine_during_source",
                                      "wrong_generation", "identity_change", "oversized_source"])
def test_actual_lease_and_source_rechecked_before_any_native_work(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    state = fake_native(monkeypatch)
    factory = resolved_factory(tmp_path)
    ledger = ResourceLedger(CAPACITIES)
    token = ledger.reserve("tool", CLAIMS)
    def callback() -> Any:
        value = source()
        if mutation == "release_during_source":
            ledger.release(token, drained=True)
        elif mutation == "quarantine_during_source":
            ledger.release(token, drained=False)
        elif mutation == "identity_change":
            value["gpu_identity"]["uuid"] = "GPU-00000000-0000-0000-0000-000000000002"
        elif mutation == "oversized_source":
            value.update(route_scope="current_loaded_route", route_identity={"oversized": "x" * (64 << 10)})
        return value
    factory.bind_route_source(callback)
    if mutation == "equal_token":
        with pytest.raises(ResourceUnavailable):
            factory.bind_resource_context(ledger, Reservation(token.owner, token.demands),
                                          joint_generation="generation-1")
    else:
        factory.bind_resource_context(ledger, token, joint_generation="generation-1")
        with pytest.raises((ValueError, ResourceUnavailable)):
            factory("generation-other" if mutation == "wrong_generation" else "generation-1")
    assert state["events"] == []


@pytest.mark.parametrize("phase", ["join", "registry"])
def test_token_retired_during_native_construction_prevents_transfer(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    factory = resolved_factory(tmp_path)
    ledger, token = bind(factory)
    state = fake_native(monkeypatch, **{phase + "_hook": lambda: ledger.release(token, drained=True)})
    with pytest.raises(ResourceUnavailable, match="token"):
        factory("generation-1")
    assert state["providers"][0].unit_close_calls == 1
    row = factory.snapshot()["current"]
    assert row["provider_transferred_to_registry"] is (phase == "registry")
    if phase == "registry":
        assert state["registries"][0].unit_close_calls == 1
        assert row["registry_close_verified"] is True and row["provider_close_attempted"] is False


@pytest.mark.parametrize("cleanup", ["interrupt", "unknown_coverage", "provider_undrained", "bookkeeping"])
def test_postconstructor_failure_retains_registry_without_direct_provider_close(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, cleanup: str) -> None:
    factory = resolved_factory(tmp_path)
    ledger, token = bind(factory)
    state = fake_native(monkeypatch, registry_hook=lambda: ledger.release(token, drained=True))
    original_constructor = process.WindowsProcessMemoryRegistry
    def construct(*args: Any, **kwargs: Any) -> Any:
        registry = original_constructor(*args, **kwargs)
        original_close = registry.close
        def close() -> Any:
            if cleanup == "interrupt":
                registry.unit_close_calls += 1
                raise KeyboardInterrupt("registry close must preserve context failure")
            result = original_close()
            if cleanup == "unknown_coverage":
                result["coverage"]["unknown_count"] = 1
            elif cleanup == "provider_undrained":
                result["children"]["gpu_sampling_close"]["drained"] = False
            return result
        registry.close = close
        return registry
    monkeypatch.setattr(process, "WindowsProcessMemoryRegistry", construct)
    if cleanup == "bookkeeping":
        def fail(*args: Any) -> None:
            raise SystemExit("cleanup bookkeeping")
        monkeypatch.setattr(factory, "_close_registry_failure", fail)
    with pytest.raises(ResourceUnavailable, match="token"):
        factory("generation-1")
    report = factory.snapshot()
    assert report["current"]["provider_transferred_to_registry"] is True
    assert report["current"]["provider_close_attempted"] is False
    assert report["retained_failed_registry_count"] == 1
    assert report["retained_failed_provider_count"] == 0
    provider = state["providers"][0]
    assert provider.unit_close_calls == (0 if cleanup in {"interrupt", "bookkeeping"} else 1)
    before = list(state["events"])
    with pytest.raises(ResourceUnavailable, match="terminally quarantined"):
        factory("generation-2")
    assert state["events"] == before


@pytest.mark.parametrize("generation", [None, True, "", "g" * 257])
def test_bad_generation_refuses_before_saving_metadata_or_native_work(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, generation: Any) -> None:
    state = fake_native(monkeypatch)
    factory = resolved_factory(tmp_path)
    bind(factory)
    with pytest.raises(ValueError, match="bounded"):
        factory(generation)
    assert state["events"] == [] and factory.snapshot()["attempt_count"] == 0


def test_descriptor_revalidated_each_generation_before_inventory(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    state = fake_native(monkeypatch)
    factory = resolved_factory(tmp_path)
    bind(factory)
    Path(factory.declaration.path).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 differs"):
        factory("generation-1")
    assert state["events"] == [] and factory.snapshot()["current"]["phase"] == "descriptor_revalidation"


def test_success_transfers_once_and_report_is_bounded_across_fresh_released_generations(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    state = fake_native(monkeypatch)
    factory = resolved_factory(tmp_path)
    ledger, token = bind(factory)
    for index in range(20):
        generation = f"generation-{index + 1}"
        if index:
            token = ledger.reserve("tool-" + generation, CLAIMS)
            factory.bind_resource_context(ledger, token, joint_generation=generation)
        registry = factory(generation)
        assert registry.gpu_provider is state["providers"][-1]
        before = list(state["events"])
        with pytest.raises(ResourceUnavailable, match="reconstruct"):
            factory(generation)
        assert state["events"] == before
        registry.gpu_provider.close()  # Owned here after transfer; production registry owns this step.
        assert ledger.release(token, drained=True)
    report = factory.snapshot()
    assert report["attempt_count"] == 20 and report["retained_report_count"] == report["report_limit"] == 16
    assert report["retained_failed_provider_count"] == 0
    report["current"]["join"]["ledger_pool"] = "modified"
    assert factory.snapshot()["current"]["join"]["ledger_pool"] == "vram"
    assert all(provider.unit_close_calls == 1 for provider in state["providers"])


def test_nvml_identity_opt_in_uses_same_handle_and_default_output_stays_unchanged(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    handle = object()
    class Nvml:
        def nvmlInit(self) -> None:
            calls.append("init")
        def nvmlShutdown(self) -> None:
            calls.append("shutdown")
        def nvmlDeviceGetHandleByIndex(self, index: int) -> Any:
            assert index == 0
            return handle
        def nvmlDeviceGetMemoryInfo(self, actual: Any) -> Any:
            assert actual is handle
            return SimpleNamespace(free=100, total=200)
        def nvmlDeviceGetName(self, actual: Any) -> str:
            assert actual is handle
            return "fixture NVIDIA"
        def nvmlSystemGetDriverVersion(self) -> str:
            return "fixture driver"
        def nvmlDeviceGetUUID(self, actual: Any) -> str:
            assert actual is handle
            calls.append("uuid")
            return IDENTITY["uuid"]
        def nvmlDeviceGetPciInfo(self, actual: Any) -> Any:
            assert actual is handle
            calls.append("pci")
            return SimpleNamespace(busId=IDENTITY["pci_bus_id"])
    monkeypatch.setitem(sys.modules, "pynvml", Nvml())
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=200, available=100)))
    monkeypatch.setattr(native_app, "_windows_commit_available", lambda: 100)
    monkeypatch.setattr(native_app, "_power_condition", lambda: "AC")
    default = native_app._hardware_snapshot(include_topology=False)
    assert "native_gpu_identity" not in default and calls == ["init", "shutdown"]
    current = native_app._fresh_native_browser_gpu_target()
    assert current["gpu_identity"]["uuid"] == IDENTITY["uuid"]
    assert current["gpu_identity"]["pci_bus_id"] == IDENTITY["pci_bus_id"]
    assert calls == ["init", "shutdown", "init", "uuid", "pci", "shutdown"]
    assert current["gpu_index"] == 0 and current["gpu_pool"] == "vram"


@pytest.mark.parametrize("mutation", ["no_route", "two_routes", "wrong_demands", "wrong_identity", "wrong_requested"])
def test_loaded_route_source_refuses_stale_or_ambiguous_application_data(monkeypatch: pytest.MonkeyPatch,
        mutation: str) -> None:
    monkeypatch.setattr(native_app, "_fresh_native_browser_gpu_target", lambda: {
        "gpu_identity": deepcopy(IDENTITY), "gpu_index": 0, "gpu_pool": "vram"})
    entry = {"placement": "cpu+cuda:0", "memory_demands": CLAIMS,
             "artifact_id": "artifact", "backend": "external.strata.text.v1"}
    plan = {"requested_device": "cpu+cuda:0", "reserved_bytes": dict(CLAIMS), "gpu_pool": "vram",
            "gpu_observer_identity": {"status": "verified", "gpu": deepcopy(IDENTITY), "worker_generation": "w1"}}
    backends = {"r": SimpleNamespace(execution_plan=plan)}
    if mutation == "no_route":
        backends["r"].execution_plan = None
    elif mutation == "two_routes":
        backends["other"] = SimpleNamespace(execution_plan=deepcopy(plan))
    elif mutation == "wrong_demands":
        plan["reserved_bytes"] = {"vram": 1}
    elif mutation == "wrong_identity":
        plan["gpu_observer_identity"]["gpu"]["name_sha256"] = "f" * 64
    else:
        plan["requested_device"] = "cpu"
    with pytest.raises(RuntimeError):
        native_app._current_browser_gpu_source(backends, {"r": entry})


def test_loaded_route_source_is_detached_and_never_calls_health_or_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native_app, "_fresh_native_browser_gpu_target", lambda: {
        "gpu_identity": deepcopy(IDENTITY), "gpu_index": 0, "gpu_pool": "vram"})
    class Backend:
        execution_plan = {"requested_device": "cuda:0", "reserved_bytes": CLAIMS}
        @property
        def resident(self) -> Any:
            pytest.fail("no health or manager callback while companion lock is held")
    entry = {"placement": "cuda:0", "memory_demands": CLAIMS, "artifact_id": "artifact"}
    value = native_app._current_browser_gpu_source({"r": Backend()}, {"r": entry})
    assert value["route_scope"] == "current_loaded_route"
    assert value["route_identity"]["model_operator_placement_verified"] is False
    assert value["route_identity"]["artifact_id"] == "artifact"


def test_cpu_model_route_can_independently_account_browser_gpu_without_placement_claim(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native_app, "_fresh_native_browser_gpu_target", lambda: {
        "gpu_identity": deepcopy(IDENTITY), "gpu_index": 0, "gpu_pool": "vram"})
    claims = {"host_ram": 50, "windows_commit": 40}
    backend = SimpleNamespace(execution_plan={"requested_device": "cpu", "reserved_bytes": claims,
                                             "gpu_pool": "host_ram"})
    entry = {"placement": "cpu", "memory_demands": claims, "artifact_id": "cpu-artifact"}
    value = native_app._current_browser_gpu_source({"cpu": backend}, {"cpu": entry})
    assert value["gpu_pool"] == "vram" and value["route_identity"]["model_gpu_pool"] == "host_ram"
    assert value["route_identity"]["requested_device"] == "cpu"
    assert value["route_identity"]["model_operator_placement_verified"] is False


def test_companion_samples_and_failed_close_keep_gpu_join_and_unknowns(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    fake_native(monkeypatch)
    companion = native_app._browser_companion_from_config({
        "browser_resource_envelope": envelope_declaration(tmp_path),
        "browser_native_gpu_accounting": gpu_declaration(tmp_path)}, config_dir=tmp_path,
        browser_factory=lambda **kwargs: SimpleNamespace(close=lambda: None))
    companion.bind_native_gpu_accounting_source(source)
    ledger = ResourceLedger(CAPACITIES)
    token = ledger.reserve("tool", CLAIMS)
    companion.bind_resource_lease(ledger, token, joint_generation="generation-1", guard=lambda: None)
    companion.browser()
    companion._registry.sample = lambda: {"cohort_generation": "generation-1",
        "coverage": {"unknown_count": 1}, "gpu_memory": {"available": False}}
    companion._registry.close = lambda: {"observer_handles_closed": False, "children": {}}
    sampled = companion.sample_process_memory()
    assert sampled["coverage"]["unknown_count"] == 1 and sampled["gpu_memory"]["available"] is False
    assert sampled["native_gpu_accounting"]["current"]["join"]["ledger_pool"] == "vram"
    assert companion.close() is False
    proof = companion.release_evidence
    assert proof["native_gpu_accounting"]["current"]["status"] == "transferred"
    assert proof["owned_work_drained"] is False and proof["quarantine_required"] is True
    assert ledger.owns(token)  # Only the existing manager may release it.


def test_sample_cannot_attach_new_generation_metadata_to_old_registry_counters(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    fake_native(monkeypatch)
    companion = native_app._browser_companion_from_config({
        "browser_resource_envelope": envelope_declaration(tmp_path),
        "browser_native_gpu_accounting": gpu_declaration(tmp_path)}, config_dir=tmp_path,
        browser_factory=lambda **kwargs: SimpleNamespace())
    companion.bind_native_gpu_accounting_source(source)
    ledger = ResourceLedger(CAPACITIES)
    token = ledger.reserve("tool-1", CLAIMS)
    companion.bind_resource_lease(ledger, token, joint_generation="generation-1", guard=lambda: None)
    companion.browser()
    factory = companion._registry_factory
    old_registry = companion._registry
    def sample_with_rebind() -> Any:
        # Simulate reset/re-admission after the old registry reference was
        # captured. No OS work: only the same exact lease/source machinery.
        ledger.release(token, drained=True)
        fresh = ledger.reserve("tool-2", CLAIMS)
        factory.bind_resource_context(ledger, fresh, joint_generation="generation-2")
        with companion._lock:
            companion._generation = "generation-2"
            companion._registry = SimpleNamespace()
        return {"cohort_generation": "generation-1", "coverage": {"unknown_count": 2}}
    old_registry.sample = sample_with_rebind
    sampled = companion.sample_process_memory()
    assert sampled["cohort_generation"] == "generation-1"
    assert sampled["native_gpu_accounting"]["generation"] == "generation-1"
    assert sampled["native_gpu_accounting"]["current"]["generation"] == "generation-1"
    assert factory.snapshot()["generation"] == "generation-2"
    assert factory.snapshot()["current"] is None  # A prior join is never current authority for an unstarted lease.
    assert sampled["coverage"]["unknown_count"] == 2


def test_normal_build_binds_app_source_before_admission_without_factory_injection(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.edge.agent import model_output

    state = fake_native(monkeypatch)
    model_claims = {"host_ram": 50, "windows_commit": 40}
    contract = model_output.AgentOutputContract("strict_raw_agent_json_v1")
    consumer = contract.consumer_identity("unit-base-engine")
    entry = {"route_id": "r", "artifact_id": "llamacpp-agent:" + consumer["identity_sha256"],
             "base_artifact_id": "unit-base-engine", "model": "unit-model",
             "model_output_contract": contract.to_dict(), "model_output_workspace_bytes": 2 << 20,
             "backend": "external.llamacpp.text.v1", "placement": "cpu", "memory_demands": model_claims}
    config = {"routes": [entry], "browser_resource_envelope": envelope_declaration(tmp_path),
              "browser_native_gpu_accounting": gpu_declaration(tmp_path)}
    config_path = tmp_path / "normal-app.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    hardware = {"host_ram_available_bytes": 1 << 30, "vram_available_bytes": 100,
                "windows_commit_available_bytes": 1 << 30, "gpu_name": "unit-gpu", "power_condition": "AC"}
    backend = SimpleNamespace(execution_plan=None, resident=False,
        bind_output_contract=lambda *args, **kwargs: None)
    captured: dict[str, Any] = {}
    class Controller:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
    monkeypatch.setattr(native_app.sys, "platform", "win32")
    monkeypatch.setattr(native_app, "_hardware_snapshot", lambda **kwargs: hardware)
    monkeypatch.setattr(native_app, "_fresh_native_browser_gpu_target", lambda: {
        "gpu_identity": deepcopy(IDENTITY), "gpu_index": 0, "gpu_pool": "vram"})
    monkeypatch.setattr(native_app, "_fingerprint", lambda value: "unit-runtime")
    monkeypatch.setattr(native_app, "llamacpp_route_binding", lambda value: None)
    # Preserve real typed Route identity and real effective memory calculation;
    # only runtime/artifact validation and backend work are fixture seams.
    monkeypatch.setattr(model_output, "validate_output_contract_entry", lambda value: contract)
    monkeypatch.setattr(native_app, "llama_config_from_entry", lambda value, **kwargs: object())
    monkeypatch.setattr(native_app, "OmniLlamaBackend", lambda value: backend)
    monkeypatch.setattr(native_app, "EncryptedMemoryStore", lambda value: object())
    monkeypatch.setattr(native_app, "AgentController", Controller)
    controller, actual_hardware = native_app.build_controller(config_path,
        browser_factory=lambda **kwargs: SimpleNamespace())
    companion = controller.browser_resource_companion
    assert type(companion._registry_factory) is ng.NativeBrowserGpuRegistryFactory
    assert companion._registry_factory._source is not None and state["events"] == []
    assert captured["admit"].__self__._companion is companion
    assert actual_hardware["browser_joint_admission"]["native_gpu_accounting"]["current"] is None
    effective_claims = {key: amount + (2 << 20) for key, amount in model_claims.items()}
    assert dict(captured["routes"][0].memory_demands) == effective_claims
    backend.execution_plan = {"requested_device": "cpu", "reserved_bytes": effective_claims}
    ledger = ResourceLedger(CAPACITIES)
    token = ledger.reserve("tool", CLAIMS)
    companion.bind_resource_lease(ledger, token, joint_generation="generation-1", guard=lambda: None)
    companion.browser()
    report = controller.resource_snapshot()["native_gpu_accounting"]
    assert report["current"]["source"]["route_scope"] == "current_loaded_route"
    assert report["current"]["source"]["route_identity"]["requested_device"] == "cpu"
    assert report["current"]["join"]["ledger_pool"] == "vram"


@pytest.mark.parametrize("browser_headless", [False, True])
def test_profiler_forwards_exact_normal_gpu_config_and_uses_companion_factory(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, browser_headless: bool) -> None:
    route = ProfileRoute("r", "model", "artifact", "revision", "a" * 64,
                         "Q4", "external.llamacpp.text.v1", "cpu")
    config = {"routes": [{"route_id": "r"}],
              "browser_resource_envelope": envelope_declaration(tmp_path),
              "browser_native_gpu_accounting": gpu_declaration(tmp_path)}
    calls: list[Any] = []
    browser_kwargs: dict[str, Any] = {}
    class Browser:
        def __init__(self, **kwargs: Any) -> None:
            browser_kwargs.update(kwargs)
    monkeypatch.setattr(native_profile, "ManagedEdgeBrowser", Browser)
    backend = SimpleNamespace(execution_plan=None)
    def start() -> None:
        backend.execution_plan = {"requested_device": "cpu"}
    backend.start = start
    def build(path: Path, **kwargs: Any) -> tuple[Any, Any]:
        prepared = json.loads(path.read_text(encoding="utf-8"))
        assert prepared["browser_native_gpu_accounting"] == config["browser_native_gpu_accounting"]
        assert set(kwargs) == {"tool_boundary_factory", "browser_factory"}
        companion = native_app._browser_companion_from_config(prepared, config_dir=path.parent,
            browser_factory=kwargs["browser_factory"])
        tools = companion.make_tool_boundary(boundary_factory=kwargs["tool_boundary_factory"])
        controller = SimpleNamespace(tools=tools, browser_resource_companion=companion,
            routes=[SimpleNamespace(route_id="r")], backends={"r": backend},
            limits=SimpleNamespace(max_model_steps=2), add_listener=lambda callback: None,
            admit=lambda value: SimpleNamespace(admitted=True), resource_snapshot=lambda: {"test_only": True})
        calls.append(companion)
        return controller, {}
    monkeypatch.setattr(native_app, "build_controller", build)
    monkeypatch.setattr(native_profile, "_FixtureForegroundScreen", lambda: object())
    bridge = native_profile.NativeProfileBridge(native_config=config, config_root=tmp_path / "configs",
        private_root=tmp_path, fixture_origin="http://127.0.0.1:1234",
        telemetry=SimpleNamespace(sample=lambda: {"ram_used_bytes": 1},
                                  attach_browser_companion=lambda value: None),
        browser_headless=browser_headless)
    prepared = asyncio.run(bridge.prepare(route))
    assert prepared.cold_start_confirmed and bridge._browser_companion is calls[0]
    assert type(calls[0]._registry_factory) is ng.NativeBrowserGpuRegistryFactory
    assert calls[0]._registry is None and calls[0]._browser is None
    # Exercise only the cold, non-launching browser constructor closure. The
    # real companion registry/provider remains lazy and owns no native handles.
    def observer(action: str, payload: Any) -> None:
        return None
    def guard() -> None:
        return None
    browser = calls[0]._browser_factory(process_observer=observer, resource_guard=guard)
    assert browser_kwargs["headless"] is browser_headless
    assert browser_kwargs["process_observer"] is observer and browser_kwargs["resource_guard"] is guard
    assert bridge._profile_browser is browser
    assert calls[0]._registry is None and calls[0]._browser is None
