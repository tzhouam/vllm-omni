# SPDX-License-Identifier: Apache-2.0
"""Fake-handle helper capability tests; no OS/process/model launches.

All process counters and ledger amounts below are synthetic accounting units.
They are neither a measured browser envelope nor RAM/GPU safety qualification.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import ntpath
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.edge_agent import native_profile
from benchmarks.edge_agent.paired_suite import ReadOnlyFixtureTools
from benchmarks.edge_agent.profile import ProfileRoute
from vllm_omni.edge import windows_cdp_helper as helper
from vllm_omni.edge import windows_process_memory as memory
from vllm_omni.edge.agent import native_app
from vllm_omni.edge.agent.browser_resources import (
    ResolvedBrowserEnvelope,
    WindowsBrowserResourceCompanion,
    _ownership_coverage_verified,
)
from vllm_omni.engine.local_plan import CompanionResourceSpec
from vllm_omni.engine.resource_ledger import ResourceLedger, ResourceUnavailable

ROOT = r"C:\Edge\msedge.exe"
HELPER = r"C:\Edge\154.0.4258.62\identity_helper.exe"
LABEL = "winrt_app_id.mojom.WinrtAppIdService"


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    # Map only the two exact declared Windows names to real temporary files;
    # all hash/size/open-file-stat verification executes against their bytes.
    browser, child = tmp_path / "browser.bin", tmp_path / "helper.bin"
    browser.write_bytes(b"fixture browser artifact")
    child.write_bytes(b"fixture helper artifact")
    paths = {ntpath.normcase(ROOT): browser, ntpath.normcase(HELPER): child}
    monkeypatch.setattr(helper, "Path", lambda value: paths[ntpath.normcase(value)])
    capability = helper.CdpObservedHelperCapability(
        ROOT, browser.stat().st_size, hashlib.sha256(browser.read_bytes()).hexdigest(),
        HELPER, child.stat().st_size, hashlib.sha256(child.read_bytes()).hexdigest(), LABEL,
    )
    return capability, browser, child


class FakeCounters:
    def __init__(self) -> None:
        self.processes = {
            10: {"pid": 10, "creation_filetime_100ns": 30, "image_path": r"C:\playwright\node.exe"},
            20: {"pid": 20, "creation_filetime_100ns": 40, "image_path": ROOT},
            30: {"pid": 30, "creation_filetime_100ns": 41, "image_path": HELPER},
            31: {"pid": 31, "creation_filetime_100ns": 42, "image_path": HELPER},
        }
        self.handles: dict[int, dict[str, Any]] = {}
        self.opened: list[int] = []
        self.closed: Counter[int] = Counter()
        self.retired: set[tuple[int, int]] = set()
        self.counter_denied: set[int] = set()
        self.identity_denied: set[int] = set()
        self.wait_denied: set[int] = set()
        self.close_denied: set[int] = set()
        self.waits: list[tuple[int, int]] = []

    def _handle(self, pid: int) -> int:
        handle = len(self.handles) + 1000
        self.handles[handle] = dict(self.processes[pid])
        return handle

    def open(self, pid: int) -> int:
        self.opened.append(pid)
        return self._handle(pid)

    def duplicate_spawn_handle(self, handle: int) -> int:
        assert handle == 777
        return self._handle(10)

    def filetime_now(self) -> int:
        return 100

    def identity(self, handle: int) -> dict[str, Any]:
        assert not self.closed[handle], "no query after observer handle close"
        if self.handles[handle]["pid"] in self.identity_denied:
            raise PermissionError("fixture identity unavailable")
        return dict(self.handles[handle])

    def wait(self, handle: int, timeout_ms: int) -> dict[str, Any]:
        assert not self.closed[handle], "no wait after observer handle close"
        row = self.handles[handle]
        self.waits.append((handle, timeout_ms))
        if row["pid"] in self.wait_denied:
            raise OSError("fixture wait unavailable")
        retired = (row["pid"], row["creation_filetime_100ns"]) in self.retired
        return {"signaled": retired, "exit_code": 0 if retired else None}

    def memory(self, handle: int) -> dict[str, int]:
        assert not self.closed[handle], "no counter query after observer handle close"
        pid = self.handles[handle]["pid"]
        if pid in self.counter_denied:
            raise PermissionError("fixture counters unavailable")
        return {"working_set_bytes": pid * 10, "private_commit_bytes": pid * 20,
                "peak_working_set_bytes": pid * 30}

    def close(self, handle: int) -> None:
        self.closed[handle] += 1
        assert self.closed[handle] == 1, "observer handle closes once"
        if self.handles[handle]["pid"] in self.close_denied:
            raise OSError("fixture CloseHandle unavailable")

    def retire(self, *pids: int) -> None:
        self.retired.update((pid, self.processes[pid]["creation_filetime_100ns"]) for pid in pids)


def cdp(reg: memory.WindowsProcessMemoryRegistry, rows: list[Any], *, cutoff: int = 100) -> None:
    reg.browser_checkpoint("cdp_membership", {
        "process_info": rows, "cutoff_filetime_100ns": cutoff, "checkpoint": "fixture",
    })


def start(reg: memory.WindowsProcessMemoryRegistry) -> None:
    reg.browser_checkpoint("node_start_attempted", {})
    reg.browser_checkpoint("node_spawn", {"pid": 10, "spawn_handle": 777,
        "expected_image": r"C:\playwright\node.exe", "playwright_version": "1.63.0",
        "python_version": "3.12.10"})
    reg.browser_checkpoint("browser_launch_started", {})
    cdp(reg, [{"type": "browser", "id": 20}, {"type": LABEL, "id": 30}])


def registry(api: FakeCounters, capability: Any = None, **kwargs: Any) -> memory.WindowsProcessMemoryRegistry:
    return memory.WindowsProcessMemoryRegistry(
        "fixture-generation-1", transport=api, cdp_helper_capability=capability, **kwargs,
    )


def test_descriptor_is_frozen_exact_and_explicitly_not_loaded_image_or_ownership(artifacts: tuple[Any, ...]) -> None:
    capability, _browser, _child = artifacts
    capability.verify_named_artifacts()
    assert capability.matches(browser_image_path=ROOT.upper(), helper_image_path=HELPER,
                              cdp_process_type=LABEL)
    assert not capability.matches(browser_image_path=ROOT, helper_image_path=HELPER,
                                  cdp_process_type=LABEL.lower())
    with pytest.raises(FrozenInstanceError):
        capability.helper_sha256 = "f" * 64
    value = capability.metadata()
    assert value["named_file_hash_is_not_loaded_image_attestation"]
    assert value["ancestry_verified"] is False and value["exclusive_ownership_verified"] is False
    assert value["termination_authority"] is False and value["retirement_required"]
    value["helper_sha256"] = "f" * 64
    assert capability.metadata()["helper_sha256"] != value["helper_sha256"]


@pytest.mark.parametrize("updates", [
    {"browser_image_path": r"C:\Other\browser.exe"},
    {"helper_image_path": r"D:\Edge\identity_helper.exe"},
    {"helper_image_path": r"C:\Edge\version\msedge.exe"},
    {"helper_image_path": r"C:\Edge\..\Other\identity_helper.exe"},
    {"helper_image_path": r"\\?\C:\Edge\identity_helper.exe"},
    {"helper_image_path": r"C:\Edge\identity_helper.exe:stream"},
    {"helper_size_bytes": True}, {"helper_size_bytes": 0},
    {"helper_size_bytes": 256 * 1024 * 1024 + 1},
    {"helper_sha256": "F" * 64}, {"helper_sha256": "x" * 64},
    {"cdp_process_type": "browser"}, {"cdp_process_type": "x" * 129},
    {"cdp_process_type": "bad\nlabel"}, {"cdp_process_type": "é" * 65},
])
def test_descriptor_rejects_aliases_broadening_and_unbounded_data(
        artifacts: tuple[Any, ...], updates: dict[str, Any]) -> None:
    capability, _browser, _child = artifacts
    body = {key: value for key, value in capability.metadata().items() if key in {
        "browser_image_path", "browser_size_bytes", "browser_sha256", "helper_image_path",
        "helper_size_bytes", "helper_sha256", "cdp_process_type",
    }}
    with pytest.raises(ValueError):
        helper.CdpObservedHelperCapability(**{**body, **updates})


@pytest.mark.parametrize("mutation", ["same_size_hash", "size", "missing"])
def test_file_mismatch_refuses_registry_before_any_process_open(artifacts: tuple[Any, ...], mutation: str) -> None:
    capability, _browser, child = artifacts
    if mutation == "same_size_hash":
        child.write_bytes(b"x" * child.stat().st_size)
    elif mutation == "size":
        child.write_bytes(b"larger fixture artifact than declared")
    else:
        child.unlink()
    api = FakeCounters()
    with pytest.raises((ValueError, FileNotFoundError)):
        registry(api, capability)
    assert api.opened == [] and api.handles == {}


def test_external_mapping_cannot_supply_capability(artifacts: tuple[Any, ...]) -> None:
    api = FakeCounters()
    with pytest.raises(ValueError, match="app-owned"):
        registry(api, artifacts[0].metadata())
    assert api.opened == []


def declaration_file(tmp_path: Path, capability: Any) -> tuple[Path, str]:
    body = {key: value for key, value in capability.metadata().items() if key in {
        "schema", "browser_image_path", "browser_size_bytes", "browser_sha256", "helper_image_path",
        "helper_size_bytes", "helper_sha256", "cdp_process_type",
    }}
    path = tmp_path / "helper-declaration.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_resolver_binds_exact_local_declaration_and_rejects_later_mutation(
        artifacts: tuple[Any, ...], tmp_path: Path) -> None:
    path, digest = declaration_file(tmp_path, artifacts[0])
    resolved = helper.resolve_cdp_observed_helper_capability(path, expected_sha256=digest)
    assert resolved.declaration_sha256 == digest
    assert resolved.metadata()["declaration_sha256"] == digest
    assert resolved.capability_sha256 != artifacts[0].capability_sha256
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash differs"):
        helper.resolve_cdp_observed_helper_capability(path, expected_sha256=digest)
    # Frozen descriptor already returned remains only the reviewed snapshot;
    # current named artifacts must still be checked when a new registry opens.
    artifacts[2].write_bytes(b"x" * artifacts[2].stat().st_size)
    with pytest.raises(ValueError, match="SHA256"):
        registry(FakeCounters(), resolved)


@pytest.mark.parametrize("mutation", ["extra_authority", "schema", "bound", "missing_field", "duplicate_field"])
def test_resolver_rejects_unreviewed_or_broadened_file_schema(
        artifacts: tuple[Any, ...], tmp_path: Path, mutation: str) -> None:
    path, _digest = declaration_file(tmp_path, artifacts[0])
    body = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "extra_authority":
        body["ignore_unknown_services"] = True
    elif mutation == "schema":
        body["schema"] = "diagnostic-retirement-receipt"
    elif mutation == "missing_field":
        del body["helper_sha256"]
    raw = json.dumps(body)
    if mutation == "bound":
        raw = " " * (16 * 1024 + 1)
    elif mutation == "duplicate_field":
        raw = raw[:-1] + ', "helper_sha256": "' + body["helper_sha256"] + '"}'
    path.write_text(raw, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        helper.resolve_cdp_observed_helper_capability(path, expected_sha256=digest)


def test_default_absent_capability_preserves_original_refusal_and_diagnostic() -> None:
    api = FakeCounters()
    reg = registry(api)
    start(reg)
    view = reg.sample()
    assert [row["pid"] for row in view["processes"]] == [10, 20]
    assert "observed_cdp_helper_capability" not in view["coverage"]
    unknown = view["coverage"]["unknown_details"][0]
    assert unknown["rejected_cdp_candidate"]["failed_check"] == "edge_runtime_image"
    assert unknown["rejected_cdp_candidate"]["query_handle_closed"] is True
    assert api.opened == [20, 30]
    api.retire(10, 20)
    reg.close()
    assert len(api.closed) == len(api.handles) and set(api.closed.values()) == {1}


def test_exact_helper_uses_same_handle_once_and_accounts_cpu_counters(artifacts: tuple[Any, ...]) -> None:
    api = FakeCounters()
    reg = registry(api, artifacts[0])
    start(reg)
    helper_handle = next(handle for handle, row in api.handles.items() if row["pid"] == 30)
    for _ in range(2):
        cdp(reg, [{"type": LABEL, "id": 30}])
    view = reg.sample()
    row = next(row for row in view["processes"] if row["pid"] == 30)
    assert api.opened == [20, 30] and not api.closed[helper_handle]
    assert row["role"] == "edge_other" and row["cdp_process_type"] == LABEL
    assert row["available"] and row["working_set_bytes"] == 300 and row["private_commit_bytes"] == 600
    assert row["observed_cdp_helper"]["capability_sha256"] == artifacts[0].capability_sha256
    assert row["observed_cdp_helper"]["exclusive_ownership_verified"] is False
    assert view["coverage"]["observed_cdp_helper_binding_count"] == 1
    assert view["coverage"]["unknown_count"] == 0
    assert view["coverage"]["all_descendants_covered"] is False
    api.retire(10, 20, 30)
    receipt = reg.close()
    assert receipt["children"]["bound_set_drain_verified"] and receipt["observer_handles_closed"]
    assert set(api.closed.values()) == {1}
    reg.sample()  # The fake transport asserts that closed handles cannot be queried.


@pytest.mark.parametrize("mutation", [
    "label", "known_label", "helper_path", "root_path", "root_absent", "root_retired",
    "helper_after_cutoff", "helper_retired", "browser_artifact", "helper_artifact",
    "helper_label_same_msedge_image", "helper_label_other_msedge_image",
])
def test_any_helper_evidence_mismatch_is_unknown_and_original_handle_is_closed(
        artifacts: tuple[Any, ...], mutation: str) -> None:
    api = FakeCounters()
    reg = registry(api, artifacts[0])
    if mutation == "root_path":
        api.processes[20]["image_path"] = r"C:\Other\msedge.exe"
    if mutation != "root_absent":
        cdp(reg, [{"type": "browser", "id": 20}])
    label = LABEL
    if mutation == "label":
        label += ".unexpected"
    elif mutation == "known_label":
        label = "renderer"
    elif mutation == "helper_path":
        api.processes[30]["image_path"] = r"C:\Other\identity_helper.exe"
    elif mutation == "root_retired":
        api.retire(20)
    elif mutation == "helper_after_cutoff":
        api.processes[30]["creation_filetime_100ns"] = 101
    elif mutation == "helper_retired":
        api.retire(30)
    elif mutation == "browser_artifact":
        artifacts[1].write_bytes(b"x" * artifacts[1].stat().st_size)
    elif mutation == "helper_artifact":
        artifacts[2].write_bytes(b"x" * artifacts[2].stat().st_size)
    elif mutation == "helper_label_same_msedge_image":
        api.processes[30]["image_path"] = ROOT
    elif mutation == "helper_label_other_msedge_image":
        api.processes[30]["image_path"] = r"C:\Other\msedge.exe"
    cdp(reg, [{"type": label, "id": 30}])
    view = reg.sample()
    assert all(row["pid"] != 30 for row in view["processes"])
    assert view["coverage"]["unknown_count"] == 1
    detail = view["coverage"]["unknown_details"][0]["rejected_cdp_candidate"]
    assert detail["query_handle_closed"] is True
    assert detail["failed_check"] in {
        "declared_cdp_helper_capability", "pre_cdp_birth_cutoff", "existing_pre_adoption_liveness",
    }
    api.retire(20)
    reg.close()
    assert len(api.closed) == len(api.handles) and set(api.closed.values()) == {1}


@pytest.mark.parametrize("mutation", ["file", "label", "retired_pid_reuse", "second_identity"])
def test_repeated_or_additional_helper_contradiction_never_retargets_binding(
        artifacts: tuple[Any, ...], mutation: str) -> None:
    api = FakeCounters()
    reg = registry(api, artifacts[0])
    start(reg)
    pid, label = 30, LABEL
    if mutation == "file":
        artifacts[2].write_bytes(b"x" * artifacts[2].stat().st_size)
    elif mutation == "label":
        label += ".changed"
    elif mutation == "retired_pid_reuse":
        api.retire(30)
        api.processes[30]["creation_filetime_100ns"] = 99
    else:
        pid = 31
    before = list(api.opened)
    cdp(reg, [{"type": label, "id": pid}])
    view = reg.sample()
    accepted = [row for row in view["processes"] if row.get("observed_cdp_helper")]
    assert len(accepted) == 1 and accepted[0]["pid"] == 30
    assert accepted[0]["creation_filetime_100ns"] == 41
    assert view["coverage"]["unknown_count"] == 1
    assert api.opened == before + ([31] if mutation == "second_identity" else [])
    api.retire(10, 20, 30)
    reg.close()
    assert set(api.closed.values()) == {1}


@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_interrupted_verification_preserves_primary_cancel_and_latches_unknown(
        artifacts: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch,
        repeated: bool, failure_type: type[BaseException]) -> None:
    api = FakeCounters()
    reg = registry(api, artifacts[0])
    cdp(reg, [{"type": "browser", "id": 20}])
    if repeated:
        cdp(reg, [{"type": LABEL, "id": 30}])
    original = failure_type("fixture interrupted artifact verification")
    def interrupt(_capability: Any) -> None:
        raise original
    monkeypatch.setattr(helper.CdpObservedHelperCapability, "verify_named_artifacts", interrupt)
    with pytest.raises(failure_type) as caught:
        cdp(reg, [{"type": LABEL, "id": 30}])
    assert caught.value is original
    view = reg.sample()
    assert view["coverage"]["unknown_count"] == 1
    assert view["coverage"]["unknown_details"][0]["reason"].startswith(
        "declared_cdp_validation_interrupted:")
    if not repeated:
        assert next(api.closed[handle] for handle, row in api.handles.items() if row["pid"] == 30) == 1
    # Omission from a later CDP response cannot erase this uncertainty.
    cdp(reg, [{"type": "browser", "id": 20}])
    api.retire(20, 30)
    receipt = reg.close()
    assert not _ownership_coverage_verified(receipt)
    assert receipt["observer_handles_closed"] and set(api.closed.values()) == {1}


@pytest.mark.parametrize("failure", ["live", "wait", "close", "counter_denied_after_retirement"])
def test_helper_retirement_and_handle_close_are_required_independently_of_counter_access(
        artifacts: tuple[Any, ...], failure: str) -> None:
    api = FakeCounters()
    reg = registry(api, artifacts[0])
    start(reg)
    api.retire(10, 20)
    if failure != "live":
        api.retire(30)
    if failure == "wait":
        api.wait_denied.add(30)
    elif failure == "close":
        api.close_denied.add(30)
    elif failure == "counter_denied_after_retirement":
        api.counter_denied.add(30)
        api.identity_denied.add(30)
    receipt = reg.close_children(timeout_seconds=0)
    row = next(row for row in receipt["processes"] if row["pid"] == 30)
    if failure == "counter_denied_after_retirement":
        assert "counter_error" in row["last_counters"]
        assert row["last_counters"]["available"] is False
        assert row["signaled"] is True and row["observer_handle_closed"] is True
        assert receipt["bound_set_drain_verified"]
    else:
        assert not receipt["bound_set_drain_verified"] and receipt["quarantine_required"]
    assert reg.close_children() == receipt  # A failed finite wait cannot become a later success.
    reg.close()
    assert set(api.closed.values()) == {1}


class FakeGpuProvider:
    def __init__(self, api: FakeCounters) -> None:
        self.api = api
        self.sampled: list[tuple[int, int]] = []
        self.closed = False

    def sample_process(self, handle: int, metadata: dict[str, Any]) -> dict[str, Any]:
        assert not self.closed and not self.api.closed[handle]
        self.sampled.append((handle, metadata["pid"]))
        return {"enabled": True, "available": True, "fixture_only": True}

    def unavailable(self, metadata: dict[str, Any], reason: str) -> dict[str, Any]:
        return {"enabled": True, "available": False, "reason": reason}

    def capabilities(self) -> dict[str, Any]:
        return {"enabled": True, "fixture_only": True}

    def close(self) -> dict[str, Any]:
        assert not self.api.closed  # GPU borrowed use drains before any bound handle close.
        self.closed = True
        return {"drained": True, "adapter_handles_closed": True}


def test_optional_gpu_provider_sees_same_live_helper_and_drains_before_handles(artifacts: tuple[Any, ...]) -> None:
    api = FakeCounters()
    provider = FakeGpuProvider(api)
    reg = registry(api, artifacts[0], gpu_provider=provider)
    start(reg)
    row = next(row for row in reg.sample()["processes"] if row["pid"] == 30)
    handle = next(handle for handle, value in api.handles.items() if value["pid"] == 30)
    assert (handle, 30) in provider.sampled and row["gpu_memory"]["fixture_only"]
    assert row["observed_cdp_helper"]["termination_authority"] is False
    api.retire(10, 20, 30)
    receipt = reg.close()
    assert provider.closed and receipt["children"]["bound_set_drain_verified"]
    assert receipt["children"]["gpu_sampling_close"]["drained"]


class FakeBrowser:
    def __init__(self, api: FakeCounters, *, process_observer: Any, resource_guard: Any,
                 retire_helper: bool, receipt_mutation: str | None = None) -> None:
        self.api, self.observer, self.guard = api, process_observer, resource_guard
        self.retire_helper = retire_helper
        self.receipt_mutation = receipt_mutation
        self.process_memory_close_receipt: dict[str, Any] | None = None
        self.close_count = 0

    def open(self) -> None:
        self.guard()
        self.observer("node_start_attempted", {})
        self.observer("node_spawn", {"pid": 10, "spawn_handle": 777,
            "expected_image": r"C:\playwright\node.exe", "playwright_version": "1.63.0",
            "python_version": "3.12.10"})
        self.observer("browser_launch_started", {})
        self.observer("cdp_membership", {"cutoff_filetime_100ns": 100,
            "process_info": [{"type": "browser", "id": 20}, {"type": LABEL, "id": 30}]})

    def close(self) -> None:
        self.close_count += 1
        self.api.retire(10, 20)
        if self.retire_helper:
            self.api.retire(30)
        receipt = self.observer("children_closed", {})
        if self.receipt_mutation == "capability":
            receipt["coverage"]["observed_cdp_helper_capability"]["capability_sha256"] = "f" * 64
        elif self.receipt_mutation == "count":
            receipt["coverage"]["observed_cdp_helper_binding_count"] = 99
        self.process_memory_close_receipt = {"worker_joined": True, "bound_set": receipt,
            "observer_callback_error_count": 0, "all_descendants_retired": False}


@pytest.mark.parametrize("retire_helper", [True, False])
@pytest.mark.parametrize("receipt_mutation", [None, "capability", "count"])
def test_companion_uses_exact_current_helper_retirement_and_keeps_engine_token_authority(
        artifacts: tuple[Any, ...], retire_helper: bool, receipt_mutation: str | None) -> None:
    api = FakeCounters()
    envelope = ResolvedBrowserEnvelope(
        CompanionResourceSpec("synthetic-browser", {"host_ram": 20, "windows_commit": 30},
                              "a" * 64, "fixture-only-review.json"),
        (("fixture-only-measurement.json", "b" * 64),),
    )
    companion = WindowsBrowserResourceCompanion(
        envelope, registry_factory=lambda generation: memory.WindowsProcessMemoryRegistry(
            generation, transport=api, cdp_helper_capability=artifacts[0]),
        browser_factory=lambda **kwargs: FakeBrowser(
            api, retire_helper=retire_helper, receipt_mutation=receipt_mutation, **kwargs),
    )
    ledger = ResourceLedger({"host_ram": 100, "windows_commit": 100})
    tokens = ledger.reserve_many({"model": {"host_ram": 50, "windows_commit": 40},
                                  "tool": companion.resource_spec.memory_demands})
    companion.bind_resource_lease(ledger, tokens["tool"], joint_generation="joint-1", guard=lambda: None)
    browser = companion.browser()
    browser.open()
    verified = retire_helper and receipt_mutation is None
    assert companion.close() is verified
    proof = companion.release_evidence
    assert proof["resource_owner"] == tokens["tool"].owner and proof["joint_generation"] == "joint-1"
    assert proof["all_descendants_retired"] is False and proof["hard_process_cap"] is False
    assert proof["required_ownership_verified_scope"] == "owned_browser_worker_spawn_and_root_identity_only"
    assert proof["observed_dependency_ownership_verified"] is False
    assert proof["observed_dependency_capability"]["capability_sha256"] == artifacts[0].capability_sha256
    assert proof["required_observed_dependency_policy_verified"] is verified
    assert ledger.owns(tokens["model"]) and ledger.owns(tokens["tool"])
    assert browser.close_count == 1 and set(api.closed.values()) == {1}
    if verified:
        assert proof["owned_work_drained"]
        assert ledger.release(tokens["tool"], drained=True)
        companion.reset_after_verified_drain()
        assert companion.is_cold() and ledger.owns(tokens["model"])
    else:
        assert proof["quarantine_required"]
        assert not ledger.release(tokens["tool"], drained=False)
        assert ledger.snapshot()["quarantined"] == ["tool"] and ledger.owns(tokens["model"])
        api.retire(30)
        assert companion.close() is False and browser.close_count == 1
        with pytest.raises(ResourceUnavailable):
            companion.reset_after_verified_drain()


def fixture_envelope() -> ResolvedBrowserEnvelope:
    return ResolvedBrowserEnvelope(
        CompanionResourceSpec("synthetic-browser", {"host_ram": 20, "windows_commit": 30},
                              "a" * 64, "fixture-only-review.json"),
        (("fixture-only-measurement.json", "b" * 64),),
    )


@pytest.mark.parametrize("custom_factory", [False, True])
def test_app_resolves_descriptor_before_admission_and_passes_it_to_exact_lazy_registry(
        artifacts: tuple[Any, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        custom_factory: bool) -> None:
    from vllm_omni.edge.agent import browser_resources
    path, digest = declaration_file(tmp_path, artifacts[0])
    monkeypatch.setattr(browser_resources, "resolve_reviewed_browser_envelope",
                        lambda *_args, **_kwargs: fixture_envelope())
    api = FakeCounters()
    calls: list[Any] = []
    factory = memory.WindowsProcessMemoryRegistry
    def explicit_registry(generation: str, **kwargs: Any) -> Any:
        calls.append((generation, kwargs))
        return factory(generation, transport=api, **kwargs)
    if not custom_factory:
        monkeypatch.setattr(memory, "WindowsProcessMemoryRegistry", explicit_registry)
    companion = native_app._browser_companion_from_config({
        "browser_resource_envelope": {"path": "unused-fixture-envelope.json", "sha256": "a" * 64},
        "browser_cdp_helper": {"path": path.name, "sha256": digest},
    }, config_dir=tmp_path,
        browser_registry_factory=explicit_registry if custom_factory else None,
        browser_factory=lambda **kwargs: FakeBrowser(api, retire_helper=True, **kwargs))
    assert companion.is_cold() and not calls and not api.handles
    ledger = ResourceLedger({"host_ram": 100, "windows_commit": 100})
    token = ledger.reserve("tool", companion.resource_spec.memory_demands)
    companion.bind_resource_lease(ledger, token, joint_generation="joint-1", guard=lambda: None)
    browser = companion.browser()
    assert len(calls) == 1 and calls[0][0] == "joint-1"
    resolved = calls[0][1]["cdp_helper_capability"]
    assert type(resolved) is helper.CdpObservedHelperCapability and resolved.declaration_sha256 == digest
    browser.open()
    assert companion.close() is True
    assert companion.release_evidence["observed_dependency_capability"]["declaration_sha256"] == digest


def test_app_helper_and_registry_require_explicit_companion_envelope(tmp_path: Path) -> None:
    assert native_app._browser_companion_from_config({}, config_dir=tmp_path) is None
    for options in ({"browser_cdp_helper": {"path": "helper.json", "sha256": "a" * 64}}, {}):
        with pytest.raises(ValueError, match="explicit reviewed resource envelope"):
            native_app._browser_companion_from_config(options, config_dir=tmp_path,
                **({"browser_registry_factory": lambda _generation: None} if not options else {}))


def test_profile_forwards_registry_factory_and_preserves_managed_read_only_tools(
        artifacts: tuple[Any, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, digest = declaration_file(tmp_path, artifacts[0])
    route = ProfileRoute("r", "model", "artifact", "revision", "a" * 64,
                         "Q4", "external.llamacpp.text.v1", "cpu")
    def registry_factory(generation: str, **kwargs: Any) -> tuple[Any, ...]:
        return generation, kwargs
    captured: dict[str, Any] = {}
    sentinel = RuntimeError("fixture stop before controller/model construction")
    monkeypatch.setattr(native_profile, "_FixtureForegroundScreen", lambda: object())
    monkeypatch.setattr(native_profile, "ManagedEdgeBrowser", lambda **kwargs: SimpleNamespace(**kwargs))
    def build(config_path: Path, **kwargs: Any) -> Any:
        assert kwargs["browser_registry_factory"] is registry_factory
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["browser_cdp_helper"] == {"path": str(path), "sha256": digest}
        captured["tools"] = kwargs["tool_boundary_factory"]()
        captured["browser"] = kwargs["browser_factory"](
            process_observer="fixture exact companion observer", resource_guard="fixture exact caller guard")
        raise sentinel
    monkeypatch.setattr(native_app, "build_controller", build)
    bridge = native_profile.NativeProfileBridge(
        native_config={"routes": [{"route_id": "r"}],
            "browser_resource_envelope": {"path": str(tmp_path / "envelope.json"), "sha256": "b" * 64},
            "browser_cdp_helper": {"path": str(path), "sha256": digest}},
        config_root=tmp_path / "config", private_root=tmp_path, fixture_origin="http://127.0.0.1:1234",
        telemetry=SimpleNamespace(), browser_registry_factory=registry_factory)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(bridge.prepare(route))
    assert caught.value is sentinel
    assert isinstance(captured["tools"], ReadOnlyFixtureTools)
    assert captured["browser"].resource_guard == "fixture exact caller guard"
    assert captured["browser"].process_observer == "fixture exact companion observer"
    assert bridge._profile_browser is captured["browser"]


@pytest.mark.parametrize("problem", ["without_companion", "relative_helper", "helper_without_companion"])
def test_profile_rejects_unbound_registry_or_ambiguous_helper_before_telemetry_owners(
        tmp_path: Path, problem: str) -> None:
    route = ProfileRoute("r", "model", "artifact", "revision", "a" * 64,
                         "Q4", "external.llamacpp.text.v1", "cpu")
    config: dict[str, Any] = {"routes": [{"route_id": "r"}]}
    if problem == "relative_helper":
        config.update(browser_resource_envelope={"path": str(tmp_path / "envelope.json"), "sha256": "a" * 64},
                      browser_cdp_helper={"path": "relative-helper.json", "sha256": "b" * 64})
    elif problem == "helper_without_companion":
        config["browser_cdp_helper"] = {"path": str(tmp_path / "helper.json"), "sha256": "b" * 64}
    bridge = native_profile.NativeProfileBridge(native_config=config,
        config_root=tmp_path / "config", private_root=tmp_path, fixture_origin="http://127.0.0.1:1234",
        telemetry=SimpleNamespace(begin_controller_processes=lambda _generation: pytest.fail("no owners")),
        process_memory_attribution=True,
        browser_registry_factory=(None if problem == "helper_without_companion" else lambda _generation: None))
    with pytest.raises(ValueError):
        asyncio.run(bridge.prepare(route))
