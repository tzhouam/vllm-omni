# SPDX-License-Identifier: Apache-2.0
"""Retained-handle attribution regressions; fake OS/CDP, no models or browser spawn."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.edge_agent.native_profile import NativeProfileBridge, WindowsTelemetry
from vllm_omni.edge import windows_process_memory as memory
from vllm_omni.edge.agent import tools


class FakeCounters:
    def __init__(self) -> None:
        self.processes = {
            1: {"pid": 1, "creation_filetime_100ns": 10, "image_path": r"C:\Python\python.exe"},
            2: {"pid": 2, "creation_filetime_100ns": 20, "image_path": r"C:\engine\strata.exe"},
            10: {"pid": 10, "creation_filetime_100ns": 30, "image_path": r"C:\playwright\node.exe"},
            20: {"pid": 20, "creation_filetime_100ns": 40, "image_path": r"C:\Edge\msedge.exe"},
            21: {"pid": 21, "creation_filetime_100ns": 41, "image_path": r"C:\Edge\msedge.exe"},
        }
        self.handles: dict[int, dict[str, Any]] = {}
        self.opened: list[int] = []
        self.duplicated: list[int] = []
        self.closed: Counter[int] = Counter()
        self.retired: set[tuple[int, int]] = set()
        self.denied: set[int] = set()
        self.counter_errors: set[int] = set()
        self.wait_errors: set[int] = set()
        self.close_errors: set[int] = set()
        self.after_wait: Any = None

    def current_pid(self) -> int:
        return 1

    def filetime_now(self) -> int:
        return 100

    def _handle(self, pid: int) -> int:
        handle = len(self.handles) + 1000
        self.handles[handle] = dict(self.processes[pid])
        return handle

    def open(self, pid: int) -> int:
        self.opened.append(pid)
        if pid in self.denied:
            raise PermissionError("denied")
        return self._handle(pid)

    def duplicate_spawn_handle(self, spawn_handle: int) -> int:
        self.duplicated.append(spawn_handle)
        if spawn_handle != 777:
            raise OSError("invalid owned spawn handle")
        return self._handle(10)

    def identity(self, handle: int) -> dict[str, Any]:
        return dict(self.handles[handle])

    def wait(self, handle: int, timeout_ms: int) -> dict[str, Any]:
        row = self.handles[handle]
        if row["pid"] in self.wait_errors:
            raise OSError("wait denied")
        if timeout_ms and self.after_wait is not None:
            self.after_wait()
        signaled = (row["pid"], row["creation_filetime_100ns"]) in self.retired
        return {"signaled": signaled, "exit_code": 259 if signaled else None}

    def memory(self, handle: int) -> dict[str, int]:
        pid = self.handles[handle]["pid"]
        if pid in self.counter_errors:
            raise PermissionError("counter denied")
        return {"working_set_bytes": pid * 100, "private_commit_bytes": pid * 200,
                "peak_working_set_bytes": pid * 300}

    def close(self, handle: int) -> None:
        self.closed[handle] += 1
        if self.handles[handle]["pid"] in self.close_errors:
            raise OSError("close failed")


def registry(api: FakeCounters, **kwargs: Any) -> memory.WindowsProcessMemoryRegistry:
    return memory.WindowsProcessMemoryRegistry("controller-generation-1", transport=api, **kwargs)


def model_identity(api: FakeCounters) -> dict[str, Any]:
    return {**api.processes[2], "worker_generation": "native-generation-1"}


def node(reg: memory.WindowsProcessMemoryRegistry, **updates: Any) -> None:
    payload = {"pid": 10, "spawn_handle": 777, "expected_image": r"C:\playwright\node.exe",
               "playwright_version": "1.63.0", "python_version": "3.12.10"}
    payload.update(updates)
    reg.browser_checkpoint("node_spawn", payload)


def cdp(reg: memory.WindowsProcessMemoryRegistry, rows: list[Any] | None = None, cutoff: int = 100) -> None:
    reg.browser_checkpoint("cdp_membership", {"cutoff_filetime_100ns": cutoff,
        "process_info": rows if rows is not None else [{"type": "browser", "id": 20}]})


def test_exact_agent_model_and_spawn_handle_counters_stay_separate() -> None:
    api = FakeCounters()
    reg = registry(api)
    assert reg.bind_current_agent()
    assert reg.bind_model(model_identity(api))
    node(reg)
    cdp(reg)
    observed = reg.sample()
    assert api.duplicated == [777] and 10 not in api.opened
    rows = {row["role"]: row for row in observed["processes"]}
    assert rows["agent"]["working_set_bytes"] == 100
    assert rows["model"]["private_commit_bytes"] == 400
    assert rows["playwright_node"]["creation_filetime_100ns"] == 30
    assert observed["working_set_includes_shared_pages"]
    assert observed["private_commit_is_not_physical_residency"]
    assert observed["sampled_maxima_are_lower_bounds"]
    assert observed["coverage"]["all_descendants_covered"] is False
    assert "total_physical_ram_bytes" not in observed
    reg.close()


def test_pid_reuse_never_retargets_retained_binding() -> None:
    api = FakeCounters()
    reg = registry(api)
    assert reg.bind_model(model_identity(api))
    api.retired.add((2, 20))
    api.processes[2]["creation_filetime_100ns"] = 90
    assert not reg.bind_model(model_identity(api))
    row = reg.sample()["processes"][0]
    assert row["creation_filetime_100ns"] == 20
    assert api.opened == [2]
    assert row["signaled"] is True  # An exited process may legitimately have exit code 259.
    reg.close()


def test_repeated_model_generation_and_node_spawn_contradictions_are_refused() -> None:
    api = FakeCounters()
    reg = registry(api)
    identity = model_identity(api)
    assert reg.bind_model(identity)
    assert not reg.bind_model({**identity, "worker_generation": "contradictory-generation"})
    node(reg)
    node(reg, expected_image=r"C:\wrong\node.exe")
    node(reg, spawn_handle=778)
    row = next(row for row in reg.sample()["processes"] if row["role"] == "model")
    assert row["worker_generation"] == "native-generation-1"
    assert reg.sample()["coverage"]["unknown_count"] == 3
    assert api.opened == [2] and 10 not in api.opened
    reg.close()
    assert set(api.closed.values()) == {1}


@pytest.mark.parametrize("mutation", ["birth", "denied", "early_exit", "image"])
def test_failed_adoption_is_unknown_and_closes_each_opened_handle_once(mutation: str) -> None:
    api = FakeCounters()
    reg = registry(api)
    identity = model_identity(api)
    if mutation == "birth":
        identity["creation_filetime_100ns"] += 1
        assert not reg.bind_model(identity)
    elif mutation == "denied":
        api.denied.add(2)
        assert not reg.bind_model(identity)
    elif mutation == "early_exit":
        api.retired.add((2, 20))
        assert not reg.bind_model(identity)
    else:
        node(reg, expected_image=r"C:\different\node.exe")
    observed = reg.sample()
    assert observed["processes"] == []
    assert observed["coverage"]["unknown_count"] == 1
    assert all(count == 1 for count in api.closed.values())
    assert len(api.closed) == len(api.handles)
    reg.close()
    assert all(count == 1 for count in api.closed.values())


@pytest.mark.parametrize("updates", [
    {"spawn_handle": None}, {"spawn_handle": False}, {"expected_image": None},
    {"playwright_version": "unreviewed"}, {"python_version": "3.13.0"},
])
def test_missing_or_unpinned_node_handle_never_reopens_pid(updates: dict[str, Any]) -> None:
    api = FakeCounters()
    reg = registry(api)
    node(reg, **updates)
    assert api.opened == [] and api.duplicated == []
    assert not reg.sample()["coverage"]["node_spawn_identity_verified"]
    reg.close()


def test_cdp_cutoff_type_image_and_registry_overflow_remain_unknown() -> None:
    api = FakeCounters()
    reg = registry(api, max_bindings=2, max_unknown_details=2)
    api.processes[21]["creation_filetime_100ns"] = 101
    cdp(reg, [{"type": "renderer", "id": 21}])
    api.processes[21]["creation_filetime_100ns"] = 41
    api.processes[21]["image_path"] = r"C:\unrelated\other.exe"
    cdp(reg, [{"type": "renderer", "id": 21}])
    cdp(reg, [{"type": "unknown", "id": 22}])
    assert reg.bind_current_agent()
    assert reg.bind_model(model_identity(api))
    cdp(reg)
    view = reg.sample()
    assert view["coverage"]["binding_registry_overflow"]
    assert view["coverage"]["unknown_count"] == 4
    assert len(view["coverage"]["unknown_details"]) == 2
    assert view["coverage"]["unknown_details_truncated"]
    assert [row["role"] for row in view["processes"]] == ["agent", "model"]
    reg.close()


def test_counter_denial_is_not_zero_memory_and_retirement_is_independent() -> None:
    api = FakeCounters()
    reg = registry(api)
    assert reg.bind_model(model_identity(api))
    api.counter_errors.add(2)
    row = reg.sample()["processes"][0]
    assert row["available"] is False and "counter_error" in row
    assert "working_set_bytes" not in row and "private_commit_bytes" not in row
    api.retired.add((2, 20))
    receipt = reg.close()
    assert receipt["model_retirement_verified"]
    assert receipt["observer_handles_closed"]


def test_partial_child_retirement_blocks_drain_and_handles_close_once() -> None:
    api = FakeCounters()
    reg = registry(api)
    reg.browser_checkpoint("node_start_attempted", {})
    reg.browser_checkpoint("browser_launch_started", {})
    node(reg)
    cdp(reg, [{"type": "browser", "id": 20}, {"type": "renderer", "id": 21}])
    api.retired.update({(10, 30), (20, 40)})
    receipt = reg.close_children(timeout_seconds=0)
    assert receipt["required_owned_bindings_verified"]
    assert receipt["bound_set_drain_verified"] is False
    assert receipt["quarantine_required"]
    assert receipt["all_descendants_retired"] is False
    assert len(receipt["processes"]) == 3
    assert reg.close_children() == receipt
    reg.close()
    reg.close()
    assert len(api.closed) == 3 and set(api.closed.values()) == {1}


def test_wait_failure_and_handle_close_failure_remain_in_partial_receipt() -> None:
    api = FakeCounters()
    reg = registry(api)
    node(reg)
    cdp(reg)
    api.wait_errors.add(10)
    api.close_errors.add(20)
    receipt = reg.close_children(timeout_seconds=0)
    rows = {row["pid"]: row for row in receipt["processes"]}
    assert "wait_error" in rows[10]
    assert rows[10]["signaled"] is None
    assert "handle_close_error" in rows[20]
    assert receipt["bound_set_drain_verified"] is False
    assert reg.close()["observer_handles_closed"] is False
    assert set(api.closed.values()) == {1}


def test_unknown_required_bindings_prevent_successful_owned_drain() -> None:
    api = FakeCounters()
    reg = registry(api)
    reg.browser_checkpoint("node_start_attempted", {})
    reg.browser_checkpoint("browser_launch_started", {})
    reg.browser_checkpoint("node_spawn_unavailable", {"reason": "private API missing"})
    reg.browser_checkpoint("cdp_unavailable", {"reason": "unsupported command"})
    receipt = reg.close_children(timeout_seconds=0)
    assert receipt["known_bound_children_retired"]
    assert not receipt["required_owned_bindings_verified"]
    assert not receipt["bound_set_drain_verified"]


def test_node_start_failure_before_browser_launch_is_not_empty_success() -> None:
    api = FakeCounters()
    reg = registry(api)
    reg.browser_checkpoint("node_start_attempted", {})
    reg.browser_checkpoint("node_start_failed", {"reason": "partial Playwright startup"})
    receipt = reg.close_children(timeout_seconds=0)
    assert receipt["known_bound_children_retired"]
    assert receipt["required_owned_bindings_verified"] is False
    assert receipt["bound_set_drain_verified"] is False
    assert receipt["coverage"]["browser_launch_attempted"] is False
    assert receipt["coverage"]["node_start_attempted"] is True


def test_published_snapshots_cannot_mutate_owned_receipts() -> None:
    api = FakeCounters()
    reg = registry(api)
    node(reg)
    observed = reg.sample()
    observed["processes"][0]["pid"] = 999
    assert reg.sample()["processes"][0]["pid"] == 10
    receipt = reg.close_children(timeout_seconds=0)
    receipt["processes"][0]["signaled"] = True
    assert reg.close_children()["processes"][0]["signaled"] is False
    reg.close()


def test_close_wait_releases_registry_lock_and_samples_do_not_use_closing_handle() -> None:
    api = FakeCounters()
    reg = registry(api)
    node(reg)
    completed: list[bool] = []

    def during_wait() -> None:
        def sample() -> None:
            rows = reg.sample()["processes"]
            completed.append(rows[0]["available"] is False)
        observer = threading.Thread(target=sample)
        observer.start()
        observer.join(timeout=1)
        assert not observer.is_alive(), "registry lock must not cover process waits"

    api.after_wait = during_wait
    reg.close_children(timeout_seconds=.01)
    assert completed == [True]
    assert set(api.closed.values()) == {1}


def test_agent_alive_and_model_retirement_have_distinct_outcomes() -> None:
    api = FakeCounters()
    reg = registry(api)
    assert reg.bind_current_agent() and reg.bind_model(model_identity(api))
    api.retired.add((2, 20))
    receipt = reg.close()
    assert receipt["agent_alive_expected"] and receipt["agent_alive_observed"]
    assert receipt["model_retirement_verified"]
    assert receipt["children"]["bound_set_drain_verified"]
    assert receipt["all_descendants_retired"] is False
    assert set(api.closed.values()) == {1}


def _owned_fake_playwright() -> Any:
    process = type("Process", (), {"__module__": "asyncio.subprocess"})()
    transport = type("Transport", (), {"__module__": "asyncio.windows_events"})()
    popen = type("Popen", (), {"__module__": "asyncio.windows_utils"})()
    popen.pid, popen._handle, popen.args = 10, 777, [r"C:\playwright\node.exe", "run-driver"]
    process.pid, process._transport = 10, transport
    transport._proc = popen
    return SimpleNamespace(_impl_obj=SimpleNamespace(_connection=SimpleNamespace(
        _transport=SimpleNamespace(_proc=process))))


def test_private_node_transport_shape_is_pinned_without_name_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "1.63.0")
    monkeypatch.setattr(tools.sys, "version_info", (3, 12, 10))
    playwright = _owned_fake_playwright()
    spawn = tools._playwright_owned_node_spawn(playwright)
    assert spawn["pid"] == 10 and spawn["spawn_handle"] == 777
    playwright._impl_obj._connection._transport._proc._transport._proc = None
    with pytest.raises((RuntimeError, AttributeError)):
        tools._playwright_owned_node_spawn(playwright)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "unreviewed")
    with pytest.raises(RuntimeError, match="requires reviewed"):
        tools._playwright_owned_node_spawn(playwright)


class FakeCDP:
    def __init__(self, owner: list[int]) -> None:
        self.owner = owner
        self.called: list[int] = []

    def send(self, command: str) -> dict[str, Any]:
        assert command == "SystemInfo.getProcessInfo"
        self.called.append(threading.get_ident())
        assert threading.get_ident() == self.owner[0]
        return {"processInfo": [{"type": "browser", "id": 20}]}

    def detach(self) -> None:
        assert threading.get_ident() == self.owner[0]


def _browser(monkeypatch: pytest.MonkeyPatch, *, close_failure: bool = False) -> tuple[Any, Any, list[int]]:
    monkeypatch.setattr(tools, "_require_windows", lambda: None)
    owner: list[int] = []
    cdp_session = FakeCDP(owner)
    events: list[tuple[str, int]] = []

    def observe(action: str, payload: Any) -> dict[str, Any] | None:
        events.append((action, threading.get_ident()))
        if action == "cdp_begin":
            return {"cutoff_filetime_100ns": 100}
        if action == "children_closed":
            return {"bound_set_drain_verified": True, "all_descendants_retired": False}
        return None

    browser = tools.ManagedEdgeBrowser(process_observer=observe)

    def start() -> None:
        owner.append(threading.get_ident())
        browser._owner_thread = owner[0]
        browser._browser = SimpleNamespace(new_browser_cdp_session=lambda: cdp_session, close=lambda: None)
        def close_context() -> None:
            if close_failure:
                raise ValueError("original context close failure")
        browser._context = SimpleNamespace(close=close_context)
        browser._playwright = SimpleNamespace(stop=lambda: None)
    browser._call(start)
    return browser, (cdp_session, events), owner


def test_cdp_is_owner_thread_only_and_worker_joins_after_partial_close(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, (session, events), owner = _browser(monkeypatch, close_failure=True)
    browser._call(lambda: None)
    with pytest.raises(ValueError, match="original context close failure"):
        browser.close()
    assert session.called and set(session.called) == {owner[0]}
    assert all(not thread.is_alive() for thread in browser._worker._threads)
    receipt = browser.process_memory_close_receipt
    assert receipt["worker_joined"]
    assert receipt["all_descendants_retired"] is False
    assert events[-1][0] == "children_closed" and events[-1][1] != owner[0]


def test_multiple_cleanup_failures_preserve_first_failure_and_join_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, (_session, _events), _owner = _browser(monkeypatch, close_failure=True)
    cleanup: list[str] = []

    def install_failures() -> None:
        def browser_close() -> None:
            cleanup.append("browser")
            raise OSError("secondary browser failure")
        def driver_stop() -> None:
            cleanup.append("playwright")
            raise RuntimeError("secondary driver failure")
        browser._browser.close = browser_close
        browser._playwright.stop = driver_stop
    browser._call(install_failures)
    with pytest.raises(ValueError, match="original context close failure") as error:
        browser.close()
    assert cleanup == ["browser", "playwright"]
    assert len(error.value.__notes__) == 2
    assert all(not thread.is_alive() for thread in browser._worker._threads)
    assert browser.process_memory_close_receipt["worker_joined"]


def test_startup_cleanup_preserves_original_failure_and_releases_all_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, (_session, _events), _owner = _browser(monkeypatch, close_failure=True)
    original = LookupError("original startup failure")
    assert browser._call(browser._cleanup_browser_objects, original) is original
    assert original.__notes__
    assert browser._playwright is None and browser._browser is None and browser._context is None
    browser.close()
    assert all(not thread.is_alive() for thread in browser._worker._threads)


def test_observer_is_default_off_and_telemetry_never_calls_cdp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_require_windows", lambda: None)
    browser = tools.ManagedEdgeBrowser()
    assert browser._process_observer is None
    browser.close()
    from vllm_omni.edge.agent import native_app
    monkeypatch.setattr(native_app, "_windows_commit_available", lambda: 1234)
    calls: list[str] = []
    sampler = object.__new__(WindowsTelemetry)
    sampler._psutil = SimpleNamespace(
        sensors_battery=lambda: SimpleNamespace(power_plugged=True),
        virtual_memory=lambda: SimpleNamespace(total=10000, available=5000))
    sampler.expected_power = "AC"
    sampler._nvml = sampler._gpu = None
    sampler._native_gpu_enabled = False
    sampler._process_registry = SimpleNamespace(sample=lambda: calls.append("sample") or {"processes": []})
    result = sampler.sample()
    assert calls == ["sample"]
    assert result["ram_available_bytes"] == 5000
    assert result["windows_commit_available_bytes"] == 1234
    assert result["bound_process_cpu_memory"] == {"processes": []}


def test_bridge_preserves_primary_controller_failure_and_partial_process_receipt(tmp_path: Path) -> None:
    calls: list[Any] = []
    original = ValueError("original controller close failure")
    def close_controller() -> None:
        raise original
    receipt = {"attribution_close_verified": False, "all_descendants_retired": False}
    def end_controller_processes(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return receipt
    telemetry = SimpleNamespace(end_controller_processes=end_controller_processes)
    bridge = NativeProfileBridge(native_config={}, config_root=tmp_path / "config",
        private_root=tmp_path / "private", fixture_origin="http://127.0.0.1:1",
        telemetry=telemetry, process_memory_attribution=True)
    bridge.controller = SimpleNamespace(close=close_controller)
    browser_receipt = {"worker_joined": False}
    bridge._profile_browser = SimpleNamespace(process_memory_close_receipt=browser_receipt)
    with pytest.raises(ValueError) as failure:
        bridge.close()
    assert failure.value is original
    assert calls == [{"browser_close_receipt": browser_receipt}]
    assert bridge.process_memory_close_receipt is receipt


def test_telemetry_close_receipt_keeps_worker_model_and_agent_outcomes_distinct() -> None:
    api = FakeCounters()
    reg = registry(api)
    reg.bind_current_agent()
    reg.bind_model(model_identity(api))
    api.retired.add((2, 20))
    sampler = object.__new__(WindowsTelemetry)
    sampler._process_registry = reg
    sampler._process_memory_closures = []
    tool = {"worker_joined": True, "observer_callback_error_count": 0}
    receipt = sampler.end_controller_processes(browser_close_receipt=tool)
    assert receipt["attribution_close_verified"]
    assert receipt["agent_alive_observed"] and receipt["model_retirement_verified"]
    assert receipt["browser_tool_close"] == tool
    assert receipt["all_descendants_retired"] is False
    assert sampler.end_controller_processes() == receipt
    assert len(sampler.process_memory_closures) == 1
    receipt["browser_tool_close"]["worker_joined"] = False
    published = sampler.process_memory_closures
    published[0]["children"]["bound_set_drain_verified"] = False
    assert sampler.process_memory_closures[0]["browser_tool_close"]["worker_joined"] is True
    assert sampler.process_memory_closures[0]["children"]["bound_set_drain_verified"] is True


def test_owner_counter_checkpoints_are_bounded_drained_by_telemetry_and_preserved_on_close() -> None:
    api = FakeCounters()
    reg = registry(api)
    node(reg)
    for _ in range(18):
        cdp(reg)
    observed = reg.sample()
    assert len(observed["owner_checkpoint_samples"]) == 16
    assert observed["coverage"]["owner_checkpoint_queue_overflow"]
    assert reg.sample()["owner_checkpoint_samples"] == []
    reg.browser_checkpoint("cdp_membership", {"cutoff_filetime_100ns": 100,
        "process_info": [{"type": "browser", "id": 20}], "checkpoint": "before_browser_close"})
    api.retired.update({(10, 30), (20, 40)})
    receipt = reg.close_children(timeout_seconds=0)
    checkpoints = receipt["owner_checkpoint_samples"]
    assert len(checkpoints) == 1 and checkpoints[0]["checkpoint"] == "before_browser_close"
    before = next(row for row in checkpoints[0]["processes"] if row["pid"] == 20)
    assert before["available"] and before["working_set_bytes"] == 2000
    assert receipt["processes"][0]["last_counters"]["available"] is False
    assert receipt["all_descendants_retired"] is False


def test_cold_start_has_explicit_before_node_and_browser_counter_checkpoints() -> None:
    api = FakeCounters()
    reg = registry(api)
    reg.bind_current_agent()
    reg.bind_model(model_identity(api))
    reg.browser_checkpoint("node_start_attempted", {})
    node(reg)
    reg.browser_checkpoint("browser_launch_started", {})
    checkpoints = reg.sample()["owner_checkpoint_samples"]
    assert [row["checkpoint"] for row in checkpoints] == ["before_node_start", "before_browser_launch"]
    assert [row["role"] for row in checkpoints[0]["processes"]] == ["agent", "model"]
    assert [row["role"] for row in checkpoints[1]["processes"]] == ["agent", "model", "playwright_node"]
    assert checkpoints[0]["coverage"]["cdp_checkpoint_count"] == 0
    assert checkpoints[1]["coverage"]["cdp_checkpoint_count"] == 0
    assert all(row["available"] for checkpoint in checkpoints for row in checkpoint["processes"])
    reg.close()


@pytest.mark.parametrize("label,role", [
    ("browser", "edge_browser"), ("renderer", "edge_renderer"), ("GPU", "edge_gpu"),
    ("utility", "edge_utility"), ("other", "edge_other"),
])
def test_cdp_known_roles_keep_exact_original_labels_and_hashes(label: str, role: str) -> None:
    api = FakeCounters()
    reg = registry(api)
    cdp(reg, [{"type": label, "id": 21}])
    row = reg.sample()["processes"][0]
    assert row["role"] == role and row["pid"] == 21
    assert row["cdp_process_type"] == label and row["cdp_process_type_known"] is True
    assert row["cdp_process_type_sha256"] == hashlib.sha256(label.encode("utf-8")).hexdigest()
    assert row["identity_source"] == "owned_browser_cdp"
    assert reg.sample()["coverage"]["unrecognized_cdp_type_binding_count"] == 0
    reg.close()


def test_cdp_new_type_is_generic_bound_identity_and_part_of_finite_close() -> None:
    api = FakeCounters()
    reg = registry(api)
    label = "Network Service / experimental-v2"
    cdp(reg, [{"type": label, "id": 21}])
    cdp(reg, [{"type": label, "id": 21}])
    observed = reg.sample()
    row = observed["processes"][0]
    assert api.opened == [21] and row["role"] == "edge_other"
    assert row["creation_filetime_100ns"] == 41
    assert row["cdp_process_type"] == label and row["cdp_process_type_known"] is False
    assert row["cdp_process_type_sha256"] == hashlib.sha256(label.encode("utf-8")).hexdigest()
    assert row["working_set_bytes"] == 2100 and row["private_commit_bytes"] == 4200
    assert observed["coverage"]["unknown_count"] == 0
    assert observed["coverage"]["unrecognized_cdp_type_binding_count"] == 1
    assert observed["coverage"]["all_descendants_covered"] is False
    api.retired.add((21, 41))
    close = reg.close_children(timeout_seconds=0)
    assert close["bound_set_drain_verified"]
    assert close["processes"][0]["cdp_process_type"] == label
    assert close["all_descendants_retired"] is False
    reg.close()
    assert set(api.closed.values()) == {1}


@pytest.mark.parametrize("label", [None, False, 7, [], {}, "", "   ", "bad\nlabel", "bad\x00label",
                                    "x" * 129, "界" * 43, "\ud800"])
def test_malformed_cdp_type_is_bounded_unknown_without_opening_pid(label: Any) -> None:
    api = FakeCounters()
    reg = registry(api)
    cdp(reg, [{"type": label, "id": 21}])
    sample = reg.sample()
    assert api.opened == [] and sample["processes"] == []
    detail = sample["coverage"]["unknown_details"][0]
    assert detail["reason"] == "invalid_cdp_process_type" and detail["pid"] == 21
    assert detail["cdp_process_type_valid"] is False
    assert len(detail.get("cdp_process_type_escaped_prefix", "")) <= 128
    # Publishing malformed CDP data cannot inject an invalid UTF-8 string.
    json.dumps(detail, ensure_ascii=False).encode("utf-8")
    reg.close()


@pytest.mark.parametrize("pid", [None, True, 0, -1, 1.0, "21", 0x100000000])
def test_invalid_cdp_pid_never_opens_handle_or_truncates_identity(pid: Any) -> None:
    api = FakeCounters()
    reg = registry(api)
    cdp(reg, [{"type": "New Process Type", "id": pid}])
    sample = reg.sample()
    assert api.opened == [] and sample["processes"] == []
    detail = sample["coverage"]["unknown_details"][0]
    assert detail["reason"] == "invalid_cdp_process_id" and detail["pid"] is None
    assert detail["cdp_process_type"] == "New Process Type"
    assert detail["cdp_process_type_known"] is False
    reg.close()


def test_malformed_cdp_row_does_not_suppress_later_valid_bound_member() -> None:
    api = FakeCounters()
    reg = registry(api)
    cdp(reg, [[], {"id": 21}, {"type": "browser", "id": 20}])
    sample = reg.sample()
    assert [row["pid"] for row in sample["processes"]] == [20]
    assert sample["coverage"]["unknown_count"] == 2
    assert sample["coverage"]["browser_root_identity_verified"]
    assert api.opened == [20]
    reg.close()


@pytest.mark.parametrize("mutation", ["wrong_image", "after_cutoff", "denied", "early_exit"])
def test_new_cdp_type_uses_the_same_identity_refusals_and_diagnostics(mutation: str) -> None:
    api = FakeCounters()
    reg = registry(api)
    label = "New Service Type"
    if mutation == "wrong_image":
        api.processes[21]["image_path"] = r"C:\unrelated\other.exe"
    elif mutation == "after_cutoff":
        api.processes[21]["creation_filetime_100ns"] = 101
    elif mutation == "denied":
        api.denied.add(21)
    else:
        api.retired.add((21, 41))
    cdp(reg, [{"type": label, "id": 21}])
    sample = reg.sample()
    assert sample["processes"] == [] and sample["coverage"]["unknown_count"] == 1
    detail = sample["coverage"]["unknown_details"][0]
    assert detail["pid"] == 21 and detail["cdp_process_type"] == label
    assert detail["cdp_process_type_known"] is False
    assert detail["cdp_process_type_sha256"] == hashlib.sha256(label.encode("utf-8")).hexdigest()
    assert len(api.closed) == len(api.handles) and all(n == 1 for n in api.closed.values())
    reg.close()


def test_new_cdp_type_pid_reuse_never_retargets_the_original_handle() -> None:
    api = FakeCounters()
    reg = registry(api)
    cdp(reg, [{"type": "New Service Type", "id": 21}])
    api.retired.add((21, 41))
    api.processes[21]["creation_filetime_100ns"] = 90
    cdp(reg, [{"type": "New Service Type", "id": 21}])
    sample = reg.sample()
    assert api.opened == [21] and len(sample["processes"]) == 1
    assert sample["processes"][0]["creation_filetime_100ns"] == 41
    assert sample["processes"][0]["signaled"] is True
    assert sample["coverage"]["unknown_count"] == 1
    assert sample["coverage"]["unknown_details"][0]["cdp_process_type"] == "New Service Type"
    reg.close()
    assert set(api.closed.values()) == {1}


def test_generic_cdp_type_cannot_silently_relabel_an_existing_bound_process() -> None:
    api = FakeCounters()
    reg = registry(api)
    cdp(reg, [{"type": "First Service Type", "id": 21}])
    cdp(reg, [{"type": "Second Service Type", "id": 21}])
    sample = reg.sample()
    assert sample["processes"][0]["cdp_process_type"] == "First Service Type"
    assert sample["coverage"]["unknown_count"] == 1 and api.opened == [21]
    assert sample["coverage"]["unknown_details"][0]["cdp_process_type"] == "Second Service Type"
    reg.close()


def test_large_filetime_and_owned_handle_are_not_subject_to_pid_dword_limit() -> None:
    class LargeHandleCounters(FakeCounters):
        def duplicate_spawn_handle(self, spawn_handle: int) -> int:
            self.duplicated.append(spawn_handle)
            assert spawn_handle == 1 << 60
            return self._handle(10)

    api = LargeHandleCounters()
    reg = registry(api)
    birth = 1 << 60
    api.processes[2]["creation_filetime_100ns"] = birth
    assert reg.bind_model(model_identity(api))
    api.processes[21]["creation_filetime_100ns"] = birth + 1
    cdp(reg, [{"type": "New Service Type", "id": 21}], cutoff=birth + 2)
    node(reg, spawn_handle=1 << 60)
    sample = reg.sample()
    assert [row["creation_filetime_100ns"] for row in sample["processes"]] == [birth, birth + 1, 30]
    assert api.duplicated == [1 << 60] and 10 not in api.opened
    reg.close()


@pytest.mark.parametrize("label", ["x" * 128, "界" * 42 + "ab"])
def test_cdp_type_at_utf8_bound_preserves_exact_label_without_truncation(label: str) -> None:
    api = FakeCounters()
    reg = registry(api)
    cdp(reg, [{"type": label, "id": 21}])
    sample = reg.sample()
    assert sample["processes"][0]["cdp_process_type"] == label
    assert sample["processes"][0]["cdp_process_type_sha256"] == hashlib.sha256(label.encode("utf-8")).hexdigest()
    assert sample["coverage"]["unknown_count"] == 0
    reg.close()
