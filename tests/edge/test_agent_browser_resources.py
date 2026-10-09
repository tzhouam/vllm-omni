# SPDX-License-Identifier: Apache-2.0
"""Authored companion regressions. Fake OS/browser; no model/process launches.

Fixture amounts are dimensionless accounting examples, not a chosen browser
allowance. These tests require the private joint engine candidate to be applied.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.edge.agent import browser_resources as resources
from vllm_omni.edge.agent import tools
from vllm_omni.edge.agent.controller import AgentController
from vllm_omni.edge.windows_process_memory import WindowsProcessMemoryRegistry
from vllm_omni.engine.local_plan import CompanionResourceSpec, LocalPlanManager
from vllm_omni.engine.resource_ledger import Reservation, ResourceLedger, ResourceUnavailable


def envelope() -> resources.ResolvedBrowserEnvelope:
    return resources.ResolvedBrowserEnvelope(
        CompanionResourceSpec("fixture-browser", {"host_ram": 20, "windows_commit": 30},
                              "a" * 64, "fixture-reviewed-envelope.json"),
        (("fixture-measurement.json", "b" * 64),),
    )


class FakeCounters:
    def __init__(self) -> None:
        self.rows = {
            10: {"pid": 10, "creation_filetime_100ns": 30, "image_path": r"C:\playwright\node.exe"},
            20: {"pid": 20, "creation_filetime_100ns": 40, "image_path": r"C:\Edge\msedge.exe"},
            21: {"pid": 21, "creation_filetime_100ns": 41, "image_path": r"C:\Edge\msedge.exe"},
        }
        self.handles: dict[int, dict[str, Any]] = {}
        self.closed: Counter[int] = Counter()
        self.retired: set[int] = set()
        self.denied: set[int] = set()

    def filetime_now(self) -> int:
        return 100

    def open(self, pid: int) -> int:
        if pid in self.denied:
            raise PermissionError("fixture access denied")
        handle = len(self.handles) + 100
        self.handles[handle] = dict(self.rows[pid])
        return handle

    def duplicate_spawn_handle(self, handle: int) -> int:
        assert handle == 777
        return self.open(10)

    def identity(self, handle: int) -> dict[str, Any]:
        return dict(self.handles[handle])

    def wait(self, handle: int, timeout_ms: int) -> dict[str, Any]:
        retired = self.handles[handle]["pid"] in self.retired
        return {"signaled": retired, "exit_code": 0 if retired else None}

    def memory(self, handle: int) -> dict[str, int]:
        return {"working_set_bytes": 1, "private_commit_bytes": 2, "peak_working_set_bytes": 3}

    def close(self, handle: int) -> None:
        self.closed[handle] += 1


class FakeBrowser:
    def __init__(self, *, process_observer: Any, resource_guard: Any,
                 api: FakeCounters, events: list[str], partial: bool = False,
                 retire_on_close: bool = True, close_error: BaseException | None = None,
                 extra_member: bool = False) -> None:
        self.observer = process_observer
        self.guard = resource_guard
        self.api, self.events = api, events
        self.partial, self.retire_on_close = partial, retire_on_close
        self.close_error = close_error
        self.extra_member = extra_member
        self.process_memory_close_receipt: Mapping[str, Any] | None = None
        self.close_count = 0
        self.started = False

    def open(self, url: str) -> dict[str, Any]:
        self.events.append("open")
        if not self.started:
            self.started = True
            self.observer("node_start_attempted", {})
            if self.partial:
                self.observer("node_start_failed", {"reason": "fixture partial startup"})
                raise OSError("fixture startup failure")
            self.observer("node_spawn", {"pid": 10, "spawn_handle": 777,
                "expected_image": r"C:\playwright\node.exe",
                "playwright_version": "1.63.0", "python_version": "3.12.10"})
            self.observer("browser_launch_started", {})
            members = [{"type": "browser", "id": 20}]
            if self.extra_member:
                members.append({"type": "renderer", "id": 21})
            self.observer("cdp_membership", {"cutoff_filetime_100ns": 100,
                "process_info": members})
        return {"url": url, "text": "fixture"}

    def read(self) -> dict[str, Any]:
        self.events.append("read")
        return {"url": "https://example.com/", "text": "fixture"}

    def current_url(self) -> str:
        self.events.append("current_url")
        return "https://example.com/"

    def describe_target(self, selector: str) -> dict[str, Any]:
        self.events.append("inspect")
        return {"tag": "A", "url": "https://example.com/", "href": "/next", "visible": True,
                "text": "next", "aria_label": "next"}

    def close(self) -> None:
        self.close_count += 1
        self.events.append("close")
        if self.retire_on_close:
            self.api.retired.update({10, 20, 21})
        receipt = self.observer("children_closed", {})
        self.process_memory_close_receipt = {"worker_joined": True, "bound_set": receipt,
            "observer_callback_error_count": 0, "all_descendants_retired": False}
        if self.close_error:
            raise self.close_error


def fixture(*, guard: Any = None, registry_max_bindings: int = 128, **browser_options: Any) -> tuple[Any, ...]:
    api, events, browsers, registries = FakeCounters(), [], [], []

    def registry(generation: str) -> WindowsProcessMemoryRegistry:
        value = WindowsProcessMemoryRegistry(generation, transport=api, max_bindings=registry_max_bindings)
        registries.append(value)
        return value

    def factory(**kwargs: Any) -> FakeBrowser:
        events.append("factory")
        value = FakeBrowser(**kwargs, api=api, events=events, **browser_options)
        browsers.append(value)
        return value

    companion = resources.WindowsBrowserResourceCompanion(
        envelope(), browser_factory=factory, registry_factory=registry,
    )
    ledger = ResourceLedger({"host_ram": 100, "windows_commit": 100})
    tokens = ledger.reserve_many({"model": {"host_ram": 50, "windows_commit": 40},
                                  "tool": companion.resource_spec.memory_demands})
    callback = guard if guard is not None else lambda: events.append("guard")
    companion.bind_resource_lease(ledger, tokens["tool"], joint_generation="joint-1", guard=callback)
    return companion, ledger, tokens, api, events, browsers, registries


def test_resolver_pins_envelope_and_evidence_and_freezes_demands(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement.json"
    measurement.write_text('{"fixture_only":true}', encoding="utf-8")
    demands = {"host_ram": 20, "windows_commit": 30}
    body = {"schema": "omni-browser-resource-envelope-v1", "purpose_id": "fixture-browser",
            "memory_demands": demands, "reviewed_for_joint_admission": True,
            "coverage": "exact_retained_handle_bound_set_only", "hard_process_cap": False,
            "all_descendants_covered": False, "measurement_evidence": [{"path": measurement.name,
                "sha256": hashlib.sha256(measurement.read_bytes()).hexdigest()}]}
    path = tmp_path / "envelope.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    resolved = resources.resolve_reviewed_browser_envelope(path, expected_sha256=digest)
    assert resolved.resource_spec.envelope_sha256 == digest
    assert resolved.measurement_references[0][0] == str(measurement.resolve())
    with pytest.raises(TypeError):
        resolved.resource_spec.memory_demands["host_ram"] = 1
    measurement.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence hash"):
        resources.resolve_reviewed_browser_envelope(path, expected_sha256=digest)


@pytest.mark.parametrize("field,value", [("reviewed_for_joint_admission", False),
    ("hard_process_cap", True), ("all_descendants_covered", True),
    ("coverage", "all_children"), ("memory_demands", {"host_ram": 20})])
def test_resolver_refuses_unreviewed_or_broadened_declaration(tmp_path: Path, field: str, value: Any) -> None:
    body = {"schema": "omni-browser-resource-envelope-v1", "purpose_id": "fixture",
            "reviewed_for_joint_admission": True, "hard_process_cap": False,
            "all_descendants_covered": False, "coverage": "exact_retained_handle_bound_set_only",
            "memory_demands": {"host_ram": 20, "windows_commit": 30}}
    body[field] = value
    path = tmp_path / "envelope.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError):
        resources.resolve_reviewed_browser_envelope(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_bind_is_nonlaunching_and_uses_only_exact_companion_claim() -> None:
    companion, ledger, tokens, _api, events, browsers, registries = fixture()
    assert companion.is_cold()
    assert events == [] and browsers == [] and registries == []
    assert ledger.owns(tokens["model"]) and ledger.owns(tokens["tool"])
    assert ledger.snapshot()["reserved"] == {"host_ram": 70, "windows_commit": 70}
    assert dict(tokens["model"].demands) == {"host_ram": 50, "windows_commit": 40}
    with pytest.raises(ResourceUnavailable, match="exact cold"):
        companion.bind_resource_lease(ledger, tokens["tool"], joint_generation="joint-2", guard=lambda: None)


def test_equal_forged_token_is_not_a_lease() -> None:
    companion = resources.WindowsBrowserResourceCompanion(envelope())
    ledger = ResourceLedger({"host_ram": 100, "windows_commit": 100})
    token = ledger.reserve("tool", companion.resource_spec.memory_demands)
    equal = Reservation(token.owner, token.demands)
    with pytest.raises(ResourceUnavailable):
        companion.bind_resource_lease(ledger, equal, joint_generation="joint-1", guard=lambda: None)
    assert companion.is_cold() and companion.release_evidence is None


def test_refused_guard_prevents_factory_registry_and_lazy_start() -> None:
    def refuse() -> None:
        raise ResourceUnavailable("fixture immutable floor refusal")
    companion, ledger, tokens, _api, events, browsers, registries = fixture(guard=refuse)
    boundary = companion.make_tool_boundary()
    with pytest.raises(ResourceUnavailable, match="immutable floor"):
        boundary.execute(tools.ToolAction("browser_read"))
    assert events == [] and browsers == [] and registries == []
    assert ledger.owns(tokens["tool"]) and ledger.owns(tokens["model"])


def test_guard_runs_before_inspection_before_approval_and_after_operation() -> None:
    companion, _ledger, _tokens, _api, events, _browsers, _registries = fixture()
    boundary = companion.make_tool_boundary()
    action = tools.ToolAction("browser_follow", {"selector": "a"}, request_id="request-1")
    boundary.register_user_task("request-1", "inspect next link")
    with pytest.raises(tools.ApprovalRequired) as proposed:
        boundary.execute(action)
    assert "inspect" in events and "open" not in events
    assert events[0] == "guard" and events[-1] == "guard"
    before = len(events)
    result = boundary.approve(proposed.value.challenge.challenge_id)
    tail = events[before:]
    assert result.operation == "browser_follow"
    assert tail[0] == "guard" and tail[-1] == "guard"
    assert tail.index("inspect") < tail.index("open")
    assert "guard" in tail[tail.index("inspect") + 1:tail.index("open")]
    with pytest.raises(ValueError, match="already used"):
        boundary.approve(proposed.value.challenge.challenge_id)


def test_approval_is_still_required_for_untrusted_navigation_and_high_impact_get() -> None:
    companion, _ledger, _tokens, _api, events, _browsers, _registries = fixture()
    boundary = companion.make_tool_boundary()
    boundary.register_user_task("request-1", "open https://example.com/delete")
    for url in ("https://other.example/", "https://example.com/delete"):
        with pytest.raises(tools.ApprovalRequired):
            boundary.execute(tools.ToolAction("browser_open", {"url": url}, "request-1"))
    assert events and set(events) == {"guard"}  # No inspection or browser construction for these proposals.


def test_post_operation_refusal_is_observable_and_does_not_release_any_token() -> None:
    refusing = False
    def guard() -> None:
        if refusing:
            raise ResourceUnavailable("fixture floor now exceeded")
    companion, ledger, tokens, _api, _events, _browsers, _registries = fixture(guard=guard)
    boundary = companion.make_tool_boundary()
    browser = companion.browser()
    def read() -> dict[str, Any]:
        nonlocal refusing
        refusing = True
        return {"url": "https://example.com/", "text": "completed before refusal"}
    browser.read = read
    with pytest.raises(tools.BrowserResourcePostconditionFailed):
        boundary.execute(tools.ToolAction("browser_read"))
    assert ledger.owns(tokens["model"]) and ledger.owns(tokens["tool"])


def test_approve_guard_refusal_consumes_grant_without_inspecting_or_acting() -> None:
    allowed = True
    def guard() -> None:
        if not allowed:
            raise ResourceUnavailable("fixture floor refusal")
    companion, _ledger, _tokens, _api, events, _browsers, _registries = fixture(guard=guard)
    boundary = companion.make_tool_boundary()
    with pytest.raises(tools.ApprovalRequired) as proposed:
        boundary.execute(tools.ToolAction("browser_open", {"url": "https://example.com/"}))
    allowed = False
    before = list(events)
    with pytest.raises(ResourceUnavailable):
        boundary.approve(proposed.value.challenge.challenge_id)
    assert events == before
    with pytest.raises(ValueError, match="already used"):
        boundary.approve(proposed.value.challenge.challenge_id)


def test_verified_close_is_scoped_idempotent_and_engine_owns_release() -> None:
    companion, ledger, tokens, api, _events, browsers, _registries = fixture()
    companion.browser().open("https://example.com/")
    assert companion.close() is True
    assert companion.close() is True and browsers[0].close_count == 1
    proof = companion.release_evidence
    assert proof["joint_generation"] == "joint-1" and proof["resource_owner"] == tokens["tool"].owner
    assert proof["owned_work_drained"] and proof["required_ownership_verified"]
    assert proof["all_descendants_retired"] is False and proof["hard_process_cap"] is False
    assert ledger.owns(tokens["tool"]) and ledger.owns(tokens["model"])
    assert set(api.closed.values()) == {1}
    proof["owned_work_drained"] = False
    assert companion.release_evidence["owned_work_drained"] is True


@pytest.mark.parametrize("partial,retire", [(True, True), (False, False)])
def test_partial_startup_or_live_bound_process_retains_failure_across_retry(partial: bool, retire: bool) -> None:
    companion, ledger, tokens, api, _events, browsers, _registries = fixture(
        partial=partial, retire_on_close=retire)
    browser = companion.browser()
    if partial:
        with pytest.raises(OSError, match="startup failure"):
            browser.open("https://example.com/")
    else:
        browser.open("https://example.com/")
    assert companion.close() is False
    assert companion.release_evidence["quarantine_required"] is True
    ledger.release(tokens["tool"], drained=False)  # Engine, not this adapter, quarantines the exact token.
    api.retired.update({10, 20})
    assert companion.close() is False and browsers[0].close_count == 1
    with pytest.raises(ResourceUnavailable):
        companion.reset_after_verified_drain()
    assert ledger.snapshot()["quarantined"] == ["tool"] and ledger.owns(tokens["model"])


def test_cleanup_error_is_preserved_even_when_bound_processes_retire() -> None:
    original = OSError("fixture browser cleanup failure")
    companion, ledger, tokens, _api, _events, browsers, _registries = fixture(close_error=original)
    companion.browser().open("https://example.com/")
    with pytest.raises(OSError) as closed:
        companion.close()
    assert closed.value is original
    assert companion.release_evidence["owned_work_drained"] is False
    assert companion.close() is False and browsers[0].close_count == 1
    assert ledger.owns(tokens["tool"])


def test_factory_failure_without_returned_owner_is_quarantined() -> None:
    api = FakeCounters()
    def registry(generation: str) -> WindowsProcessMemoryRegistry:
        return WindowsProcessMemoryRegistry(generation, transport=api)
    def fail_factory(**kwargs: Any) -> Any:
        kwargs["process_observer"]("node_start_attempted", {})
        raise RuntimeError("fixture partial constructor")
    companion = resources.WindowsBrowserResourceCompanion(envelope(), browser_factory=fail_factory,
                                                          registry_factory=registry)
    ledger = ResourceLedger({"host_ram": 100, "windows_commit": 100})
    token = ledger.reserve("tool", companion.resource_spec.memory_demands)
    companion.bind_resource_lease(ledger, token, joint_generation="joint-1", guard=lambda: None)
    with pytest.raises(RuntimeError, match="partial constructor"):
        companion.browser()
    assert not companion.is_cold() and companion.close() is False
    assert companion.release_evidence["factory_failed"] is True
    assert companion.release_evidence["quarantine_required"] is True
    assert ledger.owns(token)


def test_verified_reset_replaces_browser_factory_owner_and_revokes_only_browser_grants() -> None:
    companion, ledger, tokens, _api, _events, browsers, _registries = fixture()
    settings = SimpleNamespace(read=lambda name: {"value": 10}, set=lambda name, value: {"value": value})
    boundary = companion.make_tool_boundary(settings=settings)
    boundary.register_user_task("request-1", "open https://example.com/")
    old = companion.browser()
    old.open("https://example.com/")
    with pytest.raises(tools.ApprovalRequired) as browser_grant:
        boundary.execute(tools.ToolAction("browser_open", {"url": "https://other.example/"}, "request-1"))
    with pytest.raises(tools.ApprovalRequired) as setting_grant:
        boundary.execute(tools.ToolAction("settings_set", {"setting": "mouse_speed", "value": 11}, "request-1"))
    assert companion.close() is True
    with pytest.raises(ResourceUnavailable, match="exact engine token"):
        companion.reset_after_verified_drain()
    assert ledger.release(tokens["tool"], drained=True)
    companion.reset_after_verified_drain()
    assert companion.is_cold() and len(browsers) == 1 and old.close_count == 1
    assert boundary._browser is None  # Factory boundary never caches the old executor.
    with pytest.raises(ValueError, match="already used"):
        boundary.approve(browser_grant.value.challenge.challenge_id)
    assert boundary.approve(setting_grant.value.challenge.challenge_id).data["value"] == 11
    assert ledger.release(tokens["model"], drained=True)
    api2 = FakeCounters()  # New process identities/counters belong to the next cohort.
    companion._registry_factory = lambda generation: WindowsProcessMemoryRegistry(generation, transport=api2)
    new_tokens = ledger.reserve_many({"model-2": {"host_ram": 50, "windows_commit": 40},
                                     "tool-2": companion.resource_spec.memory_demands})
    companion.bind_resource_lease(ledger, new_tokens["tool-2"], joint_generation="joint-2", guard=lambda: None)
    result = boundary.execute(tools.ToolAction("browser_open", {"url": "https://example.com/"}, "request-1"))
    assert result.operation == "browser_open" and len(browsers) == 2
    assert browsers[1] is not old and old.close_count == 1
    with pytest.raises(ResourceUnavailable, match="retired joint generation"):
        old.guard()
    with pytest.raises(RuntimeError, match="exact registry generation"):
        old.observer("cdp_begin", {})


def test_never_admitted_boundary_close_and_bound_never_started_drain_do_not_construct() -> None:
    companion = resources.WindowsBrowserResourceCompanion(envelope(),
        browser_factory=lambda **kwargs: pytest.fail("must remain cold"),
        registry_factory=lambda generation: pytest.fail("must remain cold"))
    boundary = companion.make_tool_boundary()
    boundary.close()
    assert companion.is_cold() and companion.release_evidence is None
    ledger = ResourceLedger({"host_ram": 100, "windows_commit": 100})
    token = ledger.reserve("tool", companion.resource_spec.memory_demands)
    companion.bind_resource_lease(ledger, token, joint_generation="joint-1", guard=lambda: pytest.fail("close must not guard"))
    boundary.close()
    assert companion.release_evidence["scope"] == "never_started_browser_owner_no_processes_or_worker"
    assert ledger.owns(token)


def test_owner_worker_never_calls_manager_guard_and_close_with_manager_lock_does_not_deadlock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_require_windows", lambda: None)
    manager_lock = threading.RLock()
    caller = threading.get_ident()
    guards: list[int] = []
    owner: list[int] = []
    def guard() -> None:
        if threading.get_ident() != caller:
            raise AssertionError("owner worker must never call the manager guard")
        with manager_lock:
            guards.append(threading.get_ident())
    browser = tools.ManagedEdgeBrowser(resource_guard=guard)
    def operation() -> None:
        owner.append(threading.get_ident())
        browser._owner_thread = owner[-1]
    browser._call(operation)
    assert guards == [caller, caller] and owner[0] != caller
    with manager_lock:
        browser.close()  # A guard on this owner worker would deadlock here.
    assert guards == [caller, caller]
    assert all(not thread.is_alive() for thread in browser._worker._threads)


def test_guarded_nested_owner_operation_is_refused_without_manager_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_require_windows", lambda: None)
    callers: list[int] = []
    browser = tools.ManagedEdgeBrowser(resource_guard=lambda: callers.append(threading.get_ident()))
    def operation() -> None:
        browser._owner_thread = threading.get_ident()
        browser._call(lambda: None)
    with pytest.raises(RuntimeError, match="outside the owner worker"):
        browser._call(operation)
    assert callers == [threading.get_ident(), threading.get_ident()]
    browser.close()


def test_optional_seams_are_default_disabled_and_settings_do_not_acquire_browser_allowance() -> None:
    boundary = tools.WindowsToolBoundary(settings=SimpleNamespace(read=lambda name: {"value": 10}))
    assert boundary._browser_factory is boundary._browser_resource_guard is boundary._browser_close is None
    assert boundary.execute(tools.ToolAction("settings_read", {"setting": "mouse_speed"})).data["value"] == 10
    guarded = tools.WindowsToolBoundary(settings=SimpleNamespace(read=lambda name: {"value": 10}),
        browser_resource_guard=lambda: pytest.fail("settings are outside this browser allowance"))
    guarded.execute(tools.ToolAction("settings_read", {"setting": "mouse_speed"}))
    with pytest.raises(ValueError, match="requires its admission guard"):
        tools.WindowsToolBoundary(browser_factory=lambda: None)


def test_envelope_read_is_bounded_before_hash_or_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (1024 * 1024 + 128))
    original = Path.open
    requests: list[int] = []
    class Reader:
        def __enter__(self) -> Any:
            self.stream = original(path, "rb")
            return self
        def __exit__(self, *args: Any) -> None:
            self.stream.close()
        def read(self, amount: int = -1) -> bytes:
            requests.append(amount)
            assert amount == 1024 * 1024 + 1
            return self.stream.read(amount)
    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: Reader())
    with pytest.raises(ValueError, match="metadata bound"):
        resources.resolve_reviewed_browser_envelope(path, expected_sha256="a" * 64)
    assert requests == [1024 * 1024 + 1]


@pytest.mark.parametrize("failure", ["adoption", "overflow", "cdp_unavailable"])
def test_known_root_retirement_does_not_hide_unresolved_member_or_checkpoint(failure: str) -> None:
    companion, ledger, tokens, api, _events, _browsers, registries = fixture(
        extra_member=failure != "cdp_unavailable", registry_max_bindings=2 if failure == "overflow" else 128)
    if failure == "adoption":
        api.denied.add(21)
    companion.browser().open("https://example.com/")
    if failure == "cdp_unavailable":
        registries[0].browser_checkpoint("cdp_unavailable", {"reason": "fixture missing later membership"})
    assert companion.close() is False
    # The registry's narrow required-root proof alone is insufficient.
    proof = companion.release_evidence
    children = proof["registry_close_receipt"]["children"]
    assert children["required_owned_bindings_verified"] is True
    assert children["known_bound_children_retired"] is True
    assert children["coverage"]["unknown_count"] > 0
    assert proof["quarantine_required"] is True
    assert ledger.owns(tokens["tool"]) and ledger.owns(tokens["model"])


def test_only_fully_accounted_counter_queue_truncation_is_informational() -> None:
    companion, _ledger, _tokens, _api, _events, _browsers, registries = fixture()
    companion.browser().open("https://example.com/")
    registry = registries[0]
    for _ in range(18):
        registry.browser_checkpoint("cdp_membership", {"cutoff_filetime_100ns": 100,
            "process_info": [{"type": "browser", "id": 20}]})
    sample = companion.sample_process_memory()
    assert sample["coverage"]["owner_checkpoint_queue_overflow"] is True
    assert all(row["reason"] == "owner_checkpoint_queue_overflow"
               for row in sample["coverage"]["unknown_details"])
    assert companion.close() is True


@pytest.mark.parametrize("inner_guard", [False, True])
def test_approved_post_completed_before_refusal_preserves_result_and_never_reuses_grant(inner_guard: bool) -> None:
    refused = False
    def guard() -> None:
        if refused:
            raise ResourceUnavailable("fixture post-operation floor refusal")
    companion, ledger, tokens, _api, events, _browsers, _registries = fixture(guard=guard)
    boundary = companion.make_tool_boundary()
    browser = companion.browser()
    browser.post_context = lambda url: {"current_url": "https://example.com/",
        "cookie_fingerprint": "fixture-cookie", "cookie_count": 0}
    def complete(*args: Any) -> dict[str, Any]:
        nonlocal refused
        events.append("post_committed")
        refused = True
        return {"status": 200, "response_excerpt": "fixture committed"}
    browser.post_exact = (lambda *args: tools._guarded_browser_call(browser.guard, complete, *args)) if inner_guard else complete
    action = tools.ToolAction("browser_post", {"url": "https://example.com/submit",
        "body_b64": "eA==", "content_type": "text/plain; charset=utf-8"}, "request-1")
    with pytest.raises(tools.ApprovalRequired) as proposed:
        boundary.execute(action)
    with pytest.raises(tools.BrowserResourcePostconditionFailed) as stopped:
        boundary.approve(proposed.value.challenge.challenge_id)
    result = stopped.value.completed_result
    assert result is not None and result.operation == "browser_post" and result.data["status"] == 200
    assert result.source == "https://example.com/submit" and events.count("post_committed") == 1
    with pytest.raises(ValueError, match="already used"):
        boundary.approve(proposed.value.challenge.challenge_id)
    assert ledger.owns(tokens["model"]) and ledger.owns(tokens["tool"])


@pytest.mark.parametrize("cancel", [False, True])
def test_controller_records_completed_post_when_resource_postcondition_stops_turn(cancel: bool) -> None:
    result = tools.ToolResult("browser_post", {"status": 200}, "https://example.com/submit", 1.0)
    failure = tools.BrowserResourcePostconditionFailed(result)
    records: list[Any] = []
    events: list[Any] = []
    started, finish = threading.Event(), threading.Event()
    controller = object.__new__(AgentController)
    controller._record_tool_result = lambda value, **kwargs: records.append((value, kwargs))
    controller._emit = lambda kind, payload: events.append((kind, payload))
    def callback() -> None:
        started.set()
        if cancel:
            assert finish.wait(timeout=2)
        raise failure
    async def run() -> None:
        task = asyncio.create_task(controller._run_tool(callback, operation="browser_post"))
        if cancel:
            while not started.is_set():
                await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            finish.set()
        with pytest.raises(asyncio.CancelledError if cancel else tools.BrowserResourcePostconditionFailed):
            await task
    with ThreadPoolExecutor(max_workers=1) as executor:
        controller._tool_executor = executor
        asyncio.run(run())
    assert records == [(result, {"completed_during_cancel": cancel})]
    assert events[0][0] == "tool_resource_refused"
    assert events[0][1]["automatic_retry_forbidden"] is True
    assert events[0][1]["completed_result_preserved"] is True


def managed_fixture() -> tuple[Any, ...]:
    api, events, browsers = FakeCounters(), [], []
    route = SimpleNamespace(route_id="model", placement="cpu",
        memory_demands={"host_ram": 50, "windows_commit": 40})
    available = {"host_ram": 100, "windows_commit": 100}
    samples: list[Any] = []
    def sample() -> dict[str, int]:
        samples.append(dict(available))
        return dict(available)
    class Backend:
        resident = False
        execution_plan = None
        token = None
        def bind_resource_lease(self, ledger: ResourceLedger, token: Reservation) -> None:
            self.ledger, self.token = ledger, token
            assert dict(token.demands) == route.memory_demands
        def start(self) -> None:
            assert len(self.ledger.snapshot()["owners"]) == 2
            self.resident = True
            for pool, claim in route.memory_demands.items():
                available[pool] -= claim
            self.execution_plan = {"requested_device": "cpu", "observed_model_placement": "cpu",
                                   "reserved_bytes": dict(route.memory_demands)}
        def close(self) -> bool:
            if self.resident:
                for pool, claim in route.memory_demands.items():
                    available[pool] += claim
            self.resident = False
            self.execution_plan = None
            if self.token is not None:
                self.ledger.release(self.token, drained=True)
            return True
    def browser_factory(**kwargs: Any) -> FakeBrowser:
        value = FakeBrowser(**kwargs, api=api, events=events)
        browsers.append(value)
        return value
    companion = resources.WindowsBrowserResourceCompanion(envelope(), browser_factory=browser_factory,
        registry_factory=lambda generation: WindowsProcessMemoryRegistry(generation, transport=api))
    ledger = ResourceLedger({"host_ram": 100, "windows_commit": 100})
    backend = Backend()
    manager = LocalPlanManager(routes=[route], backends={"model": backend},
        capacities=ledger.capacities, free_bytes=sample, resource_ledger=ledger, companion=companion)
    return companion, manager, ledger, backend, api, browsers, available, samples


def test_manager_real_adapter_recovers_pre_attachment_bind_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    companion, manager, ledger, backend, _api, browsers, _available, samples = managed_fixture()
    bind = companion.bind_resource_lease
    def fail_before_attach(*args: Any, **kwargs: Any) -> None:
        raise OSError("fixture bind failure before attachment")
    monkeypatch.setattr(companion, "bind_resource_lease", fail_before_attach)
    with pytest.raises(OSError, match="before attachment"):
        manager.start("model")
    assert companion.is_cold() and not backend.resident and browsers == []
    assert ledger.snapshot()["owners"] == ledger.snapshot()["quarantined"] == []
    assert manager.snapshot()["joint_admission"]["needs_drain"] is False
    monkeypatch.setattr(companion, "bind_resource_lease", bind)
    manager.start("model")
    assert backend.resident and len(samples) >= 2
    assert manager.finalize_close()["ledger"]["owners"] == []


def test_manager_exact_model_release_retains_tool_then_fully_drains_before_readmission() -> None:
    companion, manager, ledger, backend, api, browsers, _available, samples = managed_fixture()
    boundary = companion.make_tool_boundary()
    manager.start("model")
    old_token, old_model = companion._reservation, backend.token
    old_generation = manager.snapshot()["joint_admission"]["generation"]
    old_browser = companion.browser()
    old_browser.open("https://example.com/")
    assert manager.close("model") is True
    assert ledger.was_released(old_model) and ledger.owns(old_token)
    assert ledger.snapshot()["reserved"] == {"host_ram": 20, "windows_commit": 30}
    with pytest.raises(ResourceUnavailable, match="full drain"):
        boundary.execute(tools.ToolAction("browser_read"))
    manager.start("model")
    assert old_browser.close_count == 1 and ledger.was_released(old_token)
    assert backend.token is not old_model and companion._reservation is not old_token
    assert companion._generation != old_generation and len(browsers) == 1
    assert manager.snapshot()["joint_admission"]["startup_free"] == {"host_ram": 100, "windows_commit": 100}
    assert manager.snapshot()["joint_admission"]["protected_free_floor"] == {"host_ram": 30, "windows_commit": 30}
    with pytest.raises(ResourceUnavailable, match="retired joint generation"):
        old_browser.guard()
    # Do not reuse old fake process identities for the new generation.
    api.retired.clear()
    boundary.execute(tools.ToolAction("browser_read"))
    assert len(browsers) == 2 and browsers[1] is not old_browser
    assert manager.finalize_close()["ledger"]["owners"] == []
    assert len(samples) >= 4
