"""Focused Agent routing, memory and approval boundary checks."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass

import pytest

from vllm_omni.edge.agent.controller import AgentController, _model_command, _permitted_tools
from vllm_omni.edge.agent.memory import AesGcmCipher, EncryptedMemoryStore
from vllm_omni.edge.agent.omni_backend import BackendChunk
from vllm_omni.edge.agent.router import (
    Admission, Qualification, Route, classify_task, select_route,
)
from vllm_omni.edge.agent.tools import WindowsToolBoundary


def _profile(route_id: str, *, successes: int, seconds: float) -> Qualification:
    samples = {name: (seconds,) * 20 for name in ("short", "medium", "long")}
    return Qualification(
        route_id=route_id, artifact_id="artifact", task_class="basic",
        suite_id="paired", environment_fingerprint="hardware", power_condition="AC",
        successes=successes, attempts=20, answer_latency_s=samples,
        ttft_s={name: (seconds / 2,) * 20 for name in samples},
        warmups_per_length={name: 2 for name in samples},
        sustained_seconds=1800, correctness_pass=True, tool_safety_pass=True,
        memory_pass=True, cancel_recovery_pass=True, stability_pass=True,
        actual_placement_verified=True, raw_evidence="samples.jsonl",
    )


def _route(route_id: str) -> Route:
    return Route(route_id, "artifact", route_id, "external.llamacpp.text.v1",
                 frozenset({"text"}), "cpu", {"host_ram": 1})


@pytest.mark.parametrize("reply", [
    '{"tool":"settings_read","args":',
    '{"tool":"settings_read"}',
    '[{"tool":"settings_read","args":{}}]',
])
def test_malformed_json_action_is_never_reported_as_final(reply: str) -> None:
    with pytest.raises(ValueError, match="JSON action"):
        _model_command(reply)


def test_quality_precedes_latency_and_profiles_must_share_suite() -> None:
    routes = [_route("fast"), _route("good")]
    profiles = [_profile("fast", successes=19, seconds=1),
                _profile("good", successes=20, seconds=10)]
    decision = select_route(
        "basic", routes, profiles, suite_id="paired",
        environment_fingerprint="hardware", power_condition="AC",
        admit=lambda route: Admission(True, "preflight"),
    )
    assert decision.route.route_id == "good"
    other = select_route(
        "basic", routes, profiles, suite_id="different",
        environment_fingerprint="hardware", power_condition="AC",
        admit=lambda route: Admission(True, "preflight"),
    )
    assert other.route is None


def test_visual_task_refuses_text_only_bootstrap_before_loading() -> None:
    route = _route("bootstrap")
    decision = select_route(
        "browser_vision", [route], [], suite_id="paired",
        environment_fingerprint="hardware", power_condition="AC",
        admit=lambda candidate: Admission(True, "preflight"),
        bootstrap_route_id=route.route_id,
    )
    assert decision.route is None
    assert "image" in decision.refusals[route.route_id]


def test_task_classifier_routes_mouse_speed_and_web_urls_to_tool_classes() -> None:
    assert classify_task("Set mouse speed to 10") == "windows_settings"
    assert classify_task("Read the title at example.com") == "browser_text"


def test_browser_image_task_does_not_grant_desktop_capture() -> None:
    assert "browser_screenshot" in _permitted_tools(
        "browser_vision", "Read the image at https://example.com/image.jpg",
    )
    assert "screen_capture" not in _permitted_tools(
        "browser_vision", "Read the image at https://example.com/image.jpg",
    )
    assert "screen_capture" not in _permitted_tools(
        "browser_vision", "Read https://example.com/screen_capture/image.jpg",
    )
    assert "screen_capture" in _permitted_tools(
        "browser_vision", "Use screen_capture to read the Windows desktop",
    )


def test_model_cannot_capture_desktop_for_web_image_task(tmp_path) -> None:
    backend = _Backend(['{"tool":"screen_capture","args":{}}'])
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=_Browser()))
    controller.routes = [Route(
        "bootstrap", "artifact", "bootstrap", "external.llamacpp.vision.v1",
        frozenset({"text", "image"}), "cpu", {"host_ram": 1},
    )]
    try:
        with pytest.raises(PermissionError, match="screen_capture is not admitted"):
            controller.submit("Read the image at https://example.com/image.jpg").result(timeout=10)
        assert "screen_capture" not in backend.prompts[0]
    finally:
        controller.close()


class _Backend:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.execution_plan = {"requested_device": "cpu"}
        self.prompts: list[str] = []

    def start(self) -> None:
        pass

    async def generate(self, prompt: str, *, request_id: str, max_tokens: int,
                       image_data_url=None):
        self.prompts.append(prompt)
        yield BackendChunk(self.replies.pop(0), terminal=True)

    async def cancel(self, request_id: str) -> None:
        pass

    def close(self) -> None:
        pass


class _Browser:
    def __init__(self) -> None:
        self.filled: list[tuple[str, str]] = []
        self.opened: list[str] = []
        self.closed = False

    def close(self):
        self.closed = True

    def open(self, url):
        self.opened.append(url)
        return {"url": url, "title": "Example Domain"}

    def read(self):
        return {"url": self.current_url(), "title": "Example Domain", "text": "Example Domain"}

    def current_url(self):
        return "https://example.com"

    def describe_target(self, selector):
        return {"selector": selector, "url": self.current_url(), "label": "Draft"}

    def fill(self, selector, value):
        self.filled.append((selector, value))
        return {"url": self.current_url(), "filled": selector}


def _controller(tmp_path, backend, tools):
    store = EncryptedMemoryStore(tmp_path / "memory.sqlite", AesGcmCipher(b"k" * 32))
    return AgentController(
        routes=[_route("bootstrap")], qualifications=[], backends={"bootstrap": backend},
        memory=store, tools=tools, admit=lambda route: Admission(True, "preflight"),
        environment_fingerprint="hardware", power_condition="AC",
        qualification_suite_id="paired", bootstrap_route_id="bootstrap",
    )


def test_bootstrap_is_explicit_and_observations_persist_encrypted(tmp_path) -> None:
    backend = _Backend(['{"final":"ready"}'])
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=_Browser()))
    events = []
    controller.add_listener(events.append)
    try:
        assert controller.submit("Say ready").result(timeout=10) == "ready"
        assert next(event for event in events if event["kind"] == "route")["payload"]["experimental"]
        assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
        assert controller.memory.search("ready")
    finally:
        controller.close()
    assert b"ready" not in (tmp_path / "memory.sqlite").read_bytes()


def test_deleting_a_source_observation_cascades_to_agent_outputs(tmp_path) -> None:
    controller = _controller(
        tmp_path, _Backend(['{"final":"ready"}']),
        WindowsToolBoundary(browser=_Browser()),
    )
    try:
        assert controller.submit("Say ready").result(timeout=10) == "ready"
        rows = controller.memory.iter_events(session_id=controller.session_id)
        user = next(row for row in rows if row.kind == "user_observation")
        final = next(row for row in rows if row.kind == "final")
        assert user.event_id in final.derived_from
        assert controller.memory.delete_event(user.event_id) == len(rows)
        assert controller.memory.iter_events(session_id=controller.session_id) == []
    finally:
        controller.close()


def test_recall_crosses_app_sessions_without_losing_source_provenance(tmp_path) -> None:
    first = _controller(tmp_path, _Backend(["noted"]), WindowsToolBoundary(browser=_Browser()))
    original_session = first.session_id
    try:
        assert first.submit("The private code word is orchid").result(timeout=10) == "noted"
    finally:
        first.close()
    second_backend = _Backend(["orchid"])
    second = _controller(tmp_path, second_backend, WindowsToolBoundary(browser=_Browser()))
    try:
        assert second.session_id != original_session
        assert second.submit("What is the code word?").result(timeout=10) == "orchid"
        recalled = json.loads(second_backend.prompts[0])["recalled_memory"]
        assert recalled
        assert any("orchid" in item["text"] for item in recalled)
        assert all(item["event_id"] for item in recalled)
    finally:
        second.close()


def test_recall_excludes_approval_tokens_and_image_base64(tmp_path) -> None:
    backend = _Backend(['{"final":"okay"}'])
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=_Browser()))
    store = controller.memory
    store.append_event(
        session_id="previous", request_id="prior", epoch=1, sequence=1,
        kind="approval_required",
        payload={"challenge_id": "UI-ONLY-SECRET", "text": "orchid approval"},
        source="agent",
    )
    store.append_event(
        session_id="previous", request_id="prior", epoch=1, sequence=2,
        kind="tool_result",
        payload={"operation": "screen_capture", "data": {
            "text": "orchid card", "base64": "IMAGE-BASE64-SECRET",
        }}, source="windows-screen",
    )
    store.append_event(
        session_id="previous", request_id="prior", epoch=1, sequence=3,
        kind="user_observation", payload={"text": "orchid note"}, source="user",
    )
    try:
        assert controller.submit("Recall orchid").result(timeout=10) == "okay"
        recalled = json.loads(backend.prompts[0])["recalled_memory"]
        assert {entry["kind"] for entry in recalled} == {"user_observation", "tool_result"}
        assert "UI-ONLY-SECRET" not in backend.prompts[0]
        assert "IMAGE-BASE64-SECRET" not in backend.prompts[0]
    finally:
        controller.close()


def test_close_waits_for_cancelled_native_loader_before_backend_shutdown(tmp_path) -> None:
    loading = threading.Event()
    release = threading.Event()

    class SlowBackend(_Backend):
        def __init__(self):
            super().__init__(["unused"])
            self.closed = False

        def start(self):
            loading.set()
            assert release.wait(5)

        def close(self):
            assert release.is_set()
            self.closed = True

    backend = SlowBackend()
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=_Browser()))
    controller.submit("Wait for load")
    assert loading.wait(5)
    closer = threading.Thread(target=controller.close)
    closer.start()
    time.sleep(.05)
    assert closer.is_alive()
    assert not backend.closed
    release.set()
    closer.join(5)
    assert not closer.is_alive()
    assert backend.closed


def test_shutdown_reports_failed_backend_drain_after_other_cleanup(tmp_path) -> None:
    class Undrained(_Backend):
        def close(self):
            return False

    browser = _Browser()
    controller = _controller(tmp_path, Undrained([]), WindowsToolBoundary(browser=browser))
    with pytest.raises(RuntimeError, match="state release"):
        controller.close()
    assert browser.closed


def test_explicit_memory_deletion_clears_observations_and_search_index(tmp_path) -> None:
    controller = _controller(tmp_path, _Backend(["stored"]), WindowsToolBoundary(browser=_Browser()))
    try:
        assert controller.submit("Remember the apricot note").result(timeout=10) == "stored"
        assert controller.memory.search("apricot")
        assert controller.delete_all_memory() > 0
        assert controller.memory.search("apricot") == []
        assert controller.memory.iter_events() == []
    finally:
        controller.close()


def test_tool_mutation_waits_for_ui_approval(tmp_path) -> None:
    backend = _Backend([
        json.dumps({"tool": "browser_fill", "args": {"selector": "#draft", "value": "hello"}}),
        '{"final":"filled"}',
    ])
    browser = _Browser()
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=browser))
    challenge = threading.Event()
    ids: list[str] = []

    def listener(event):
        if event["kind"] == "approval_required":
            ids.append(event["payload"]["challenge_id"])
            challenge.set()

    controller.add_listener(listener)
    try:
        result = controller.submit("Fill the browser draft")
        assert challenge.wait(5)
        assert browser.filled == []
        controller.approve(ids[0])
        assert result.result(timeout=10) == "filled"
        assert browser.filled == [("#draft", "hello")]
        assert "untrusted_data" in backend.prompts[1]
    finally:
        controller.close()


def test_direct_concurrent_turn_cannot_change_active_request_or_memory(tmp_path) -> None:
    class WaitingBackend(_Backend):
        def __init__(self):
            super().__init__([])
            self.entered = threading.Event()
            self.resume: asyncio.Event | None = None

        async def generate(self, prompt, *, request_id, max_tokens, image_data_url=None):
            self.entered.set()
            self.resume = asyncio.Event()
            await self.resume.wait()
            yield BackendChunk("first answer", terminal=True)

    backend = WaitingBackend()
    tools = WindowsToolBoundary(browser=_Browser())
    controller = _controller(tmp_path, backend, tools)
    events = []
    controller.add_listener(events.append)
    try:
        first = controller.submit("Remember first request")
        assert backend.entered.wait(5)
        active_id = controller._request_id
        active_epoch = controller._epoch
        active_seq = controller._seq
        active_sources = tuple(controller._memory_sources)
        with pytest.raises(RuntimeError, match="already active"):
            asyncio.run(controller.run_turn(
                "Remember forbidden second request", request_id="foreign", epoch=99,
            ))
        assert (controller._request_id, controller._epoch, controller._seq) == (
            active_id, active_epoch, active_seq,
        )
        assert tuple(controller._memory_sources) == active_sources
        assert "foreign" not in tools._task_urls
        assert backend.resume is not None
        controller._loop.call_soon_threadsafe(backend.resume.set)
        assert first.result(timeout=10) == "first answer"
        assert all(e["request_id"] == active_id and e["epoch"] == active_epoch for e in events)
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        assert all(row.request_id == active_id for row in controller.memory.iter_events())
    finally:
        if backend.resume is not None:
            controller._loop.call_soon_threadsafe(backend.resume.set)
        controller.close()


def test_reserved_turn_rejects_direct_entry_and_cancel_before_start_recovers(tmp_path) -> None:
    backend = _Backend(["recovered"])
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=_Browser()))
    events = []
    controller.add_listener(events.append)
    loop = controller._ensure_loop()
    entered = threading.Event()
    release = threading.Event()

    def block_loop() -> None:
        entered.set()
        assert release.wait(5)

    loop.call_soon_threadsafe(block_loop)
    assert entered.wait(5)
    try:
        first = controller.submit("Never start this turn")
        request_id, epoch = controller._request_id, controller._epoch
        with pytest.raises(RuntimeError, match="already active"):
            asyncio.run(controller.run_turn(
                "Spoof the reserved request", request_id=request_id, epoch=epoch,
            ))
        assert controller._request_id == request_id
        controller.cancel()
        assert controller._turn_done.wait(5)
        assert first.cancelled()
        assert controller._turn_owner is None
        release.set()
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=5)
        assert not any(e["kind"] == "user_observation" for e in events)
        assert controller.submit("Say recovered").result(timeout=10) == "recovered"
    finally:
        release.set()
        controller.close()


def test_stale_approval_raises_and_preserves_the_pending_action(tmp_path) -> None:
    backend = _Backend([
        json.dumps({"tool": "browser_fill", "args": {"selector": "#draft", "value": "hello"}}),
        '{"final":"filled"}',
    ])
    browser = _Browser()
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=browser))
    challenge = threading.Event()
    ids: list[str] = []

    def listener(event):
        if event["kind"] == "approval_required":
            ids.append(event["payload"]["challenge_id"])
            challenge.set()

    controller.add_listener(listener)
    try:
        turn = controller.submit("Fill the browser draft")
        assert challenge.wait(5)
        with pytest.raises(ValueError, match="not active"):
            controller.approve("stale-challenge")
        assert browser.filled == []
        assert not turn.done()
        controller.approve(ids[0])
        assert turn.result(timeout=10) == "filled"
        with pytest.raises(ValueError, match="not active"):
            controller.reject(ids[0])
    finally:
        controller.close()


def test_cancel_revokes_approved_write_queued_behind_another_tool(tmp_path) -> None:
    backend = _Backend([
        json.dumps({"tool": "browser_fill", "args": {"selector": "#draft", "value": "hello"}}),
    ])
    browser = _Browser()
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=browser))
    challenge = threading.Event()
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    events = []

    def listener(event):
        events.append(event)
        if event["kind"] == "approval_required":
            challenge.set()

    def blocker():
        blocker_started.set()
        assert release_blocker.wait(5)

    controller.add_listener(listener)
    try:
        result = controller.submit("Fill the browser draft")
        assert challenge.wait(5)
        controller._tool_executor.submit(blocker)
        assert blocker_started.wait(5)
        challenge_id = next(event["payload"]["challenge_id"] for event in events
                            if event["kind"] == "approval_required")
        controller.approve(challenge_id)
        deadline = time.monotonic() + 5
        while controller._tool_executor._work_queue.qsize() == 0 and time.monotonic() < deadline:
            time.sleep(.005)
        assert controller._tool_executor._work_queue.qsize() > 0
        controller.cancel()
        release_blocker.set()
        assert controller._turn_done.wait(5)
        with pytest.raises(CancelledError):
            result.result()
        assert browser.filled == []
        assert [event["kind"] for event in events if event["kind"] == "tool_result"] == []
        assert events[-1]["kind"] == "cancelled"
    finally:
        release_blocker.set()
        controller.close()


def test_cancel_waits_for_started_write_and_reports_its_result(tmp_path) -> None:
    class SlowBrowser(_Browser):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def fill(self, selector, value):
            self.entered.set()
            assert self.release.wait(5)
            return super().fill(selector, value)

    backend = _Backend([
        json.dumps({"tool": "browser_fill", "args": {"selector": "#draft", "value": "hello"}}),
    ])
    browser = SlowBrowser()
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=browser))
    challenge = threading.Event()
    events = []

    def listener(event):
        events.append(event)
        if event["kind"] == "approval_required":
            challenge.set()

    controller.add_listener(listener)
    try:
        result = controller.submit("Fill the browser draft")
        assert challenge.wait(5)
        challenge_id = next(event["payload"]["challenge_id"] for event in events
                            if event["kind"] == "approval_required")
        controller.approve(challenge_id)
        assert browser.entered.wait(5)
        controller.cancel()
        assert not controller._turn_done.wait(.05)
        assert not any(event["kind"] == "cancelled" for event in events)
        browser.release.set()
        assert controller._turn_done.wait(5)
        with pytest.raises(CancelledError):
            result.result()
        assert browser.filled == [("#draft", "hello")]
        kinds = [event["kind"] for event in events]
        assert kinds[-2:] == ["tool_result", "cancelled"]
        assert events[-2]["payload"]["completed_during_cancel"] is True
    finally:
        browser.release.set()
        controller.close()


def test_cancel_emits_verified_release_before_next_request(tmp_path) -> None:
    class ReleasableBackend(_Backend):
        def __init__(self):
            super().__init__(["recovered"])
            self.entered = threading.Event()
            self.calls = 0

        async def generate(self, prompt, *, request_id, max_tokens, image_data_url=None):
            self.calls += 1
            if self.calls == 1:
                self.entered.set()
                await asyncio.Event().wait()
            else:
                async for chunk in super().generate(
                    prompt, request_id=request_id, max_tokens=max_tokens,
                    image_data_url=image_data_url,
                ):
                    yield chunk

        def request_state_released(self, request_id):
            self.release_evidence = {
                "request_id": request_id, "release_mode": "worker_shutdown",
                "worker_pid_before": 1234, "worker_exit_code": 0,
                "worker_exit_confirmed": True, "stage_ledger_empty": True,
                "host_claim_released": True, "host_ledger_empty": True,
            }
            return bool(request_id and self.calls == 1)

    backend = ReleasableBackend()
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=_Browser()))
    events = []
    controller.add_listener(events.append)
    try:
        first = controller.submit("Wait for cancellation")
        assert backend.entered.wait(5)
        controller.cancel()
        assert controller._turn_done.wait(5)
        with pytest.raises(CancelledError):
            first.result()
        first_events = list(events)
        assert [e["kind"] for e in first_events][-2:] == ["cancelled", "state_released"]
        assert first_events[-1]["payload"]["backend_request_state_verified"]
        assert controller.submit("Say recovered").result(timeout=10) == "recovered"
        assert not any(e["kind"] in {"text_delta", "final"}
                       for e in first_events if e["seq"] > first_events[-2]["seq"])
    finally:
        controller.close()


def test_read_only_browser_chain_and_stream_output_have_one_final(tmp_path) -> None:
    backend = _Backend([
        json.dumps({"tool": "browser_open", "args": {"url": "https://example.com"}}),
        json.dumps({"tool": "browser_read", "args": {}}),
        "Example Domain",
    ])
    browser = _Browser()
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=browser))
    events = []
    controller.add_listener(events.append)
    try:
        assert controller.submit("Read the title at https://example.com").result(timeout=10) == "Example Domain"
        assert browser.opened == ["https://example.com"]
        assert [event["kind"] for event in events].count("final") == 1
        assert "".join(event["payload"]["text"] for event in events
                       if event["kind"] == "text_delta") == "Example Domain"
        assert all("\"tool\"" not in event["payload"].get("text", "") for event in events
                   if event["kind"] == "text_delta")
        assert any(event["kind"] == "tool_result" and event["payload"]["untrusted"]
                   for event in events)
        assert "Example Domain" in backend.prompts[2]
    finally:
        controller.close()


def test_navigation_includes_read_observation_before_next_model_step(tmp_path) -> None:
    backend = _Backend([
        json.dumps({"tool": "browser_open", "args": {"url": "https://example.com"}}),
        "Example Domain",
    ])
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=_Browser()))
    events = []
    controller.add_listener(events.append)
    try:
        assert controller.submit("Open https://example.com in the browser").result(timeout=10) == "Example Domain"
        observations = json.loads(backend.prompts[1])["observations"]
        assert [item["operation"] for item in observations] == ["browser_open", "browser_read"]
        assert "Example Domain" in observations[1]["untrusted_data"]
        assert any(event["kind"] == "tool_proposed" and
                   event["payload"].get("automatic_after_navigation") == "browser_open"
                   for event in events)
    finally:
        controller.close()


def test_model_cannot_open_url_mutated_from_user_task_without_approval(tmp_path) -> None:
    changed_url = "https://example.com/report?token=model"
    backend = _Backend([
        json.dumps({"tool": "browser_open", "args": {"url": changed_url}}),
    ])
    browser = _Browser()
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=browser))
    approvals = []

    def on_event(event):
        if event["kind"] == "approval_required":
            approvals.append(event)
            controller.reject(event["payload"]["challenge_id"])

    controller.add_listener(on_event)
    try:
        with pytest.raises(PermissionError, match="rejected"):
            controller.submit(
                "Read https://example.com/report?token=user"
            ).result(timeout=10)
        assert browser.opened == []
        assert approvals[0]["payload"]["target"] == {"url": changed_url}
    finally:
        controller.close()


def test_basic_task_cannot_escalate_to_settings_tool(tmp_path) -> None:
    backend = _Backend([
        json.dumps({"tool": "settings_set", "args": {"setting": "mouse_speed", "value": 11}}),
    ])
    controller = _controller(tmp_path, backend, WindowsToolBoundary(browser=_Browser()))
    events = []
    controller.add_listener(events.append)
    try:
        with pytest.raises(PermissionError, match="basic task class"):
            controller.submit("Say hello").result(timeout=10)
        assert not any(event["kind"] == "approval_required" for event in events)
    finally:
        controller.close()
