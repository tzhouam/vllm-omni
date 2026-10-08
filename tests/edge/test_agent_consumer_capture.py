# SPDX-License-Identifier: Apache-2.0
"""Bounded current-turn prompt metadata with synthetic backends, no inference."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from benchmarks.edge_agent import native_profile as n
from benchmarks.edge_agent.profile import AgentCase, ProfileRoute
from vllm_omni.edge.agent.consumer_trace import model_step_capture_policy
from vllm_omni.edge.agent.model_output import AgentOutputContract


def policy(steps=6):
    return model_step_capture_policy(AgentOutputContract("strict_outer_json_fence_agent_v1"), steps)


class Backend:
    def __init__(self):
        self.dispatched = []
        self.closed = []
        self.cancelled = []
        self.execution_plan = {}

    async def generate(self, prompt, *, request_id, max_tokens, image_data_url=None):
        self.dispatched.append(request_id)
        try:
            yield "chunk"
        finally:
            self.closed.append(request_id)

    async def cancel(self, request_id):
        self.cancelled.append(request_id)

    def close(self):
        return True


async def consume(wrapper, prompt="input", request_id="outer-step-0"):
    return [row async for row in wrapper.generate(prompt, request_id=request_id, max_tokens=4)]


def test_all_step_hashes_and_snapshot_are_exact_detached_and_prompt_free():
    async def exercise():
        wrapper = n._PromptIdentityBackend(Backend(), capture_policy=policy())
        prompts = ["first 中英", "next input"]
        for step, prompt in enumerate(prompts):
            await consume(wrapper, prompt, f"outer-step-{step}")
        rows, captured_policy = wrapper.snapshot_and_reset()
        assert rows == [
            {
                "step": step,
                "model_request_id": f"outer-step-{step}",
                "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "utf8_bytes": len(prompt.encode()),
                "chars": len(prompt),
            }
            for step, prompt in enumerate(prompts)
        ]
        assert wrapper.identities() == []
        assert wrapper.snapshot_and_reset() == ([], policy())
        captured_policy["max_model_steps"] = 99
        rows[0]["sha256"] = "changed"
        await consume(wrapper, "fresh input")
        assert wrapper.identities()[0]["step"] == 0
        assert wrapper.snapshot_and_reset()[1] == policy()
        assert all("prompt" not in key for key in vars(wrapper))

    asyncio.run(exercise())


def test_step_limit_refuses_before_encode_and_model_dispatch():
    class EncodeForbidden(str):
        def encode(self, *_args, **_kwargs):
            pytest.fail("overflowing step encoded its prompt")

    async def exercise():
        backend = Backend()
        wrapper = n._PromptIdentityBackend(backend, capture_policy=policy(2))
        await consume(wrapper)
        await consume(wrapper, request_id="outer-step-1")
        with pytest.raises(ValueError, match="step budget"):
            await consume(wrapper, EncodeForbidden("forbidden"), "outer-step-2")
        assert backend.dispatched == ["outer-step-0", "outer-step-1"]
        assert len(wrapper.identities()) == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("request_id", ["", "x" * 257, "中" * 257, "bad\ud800", None, True])
def test_invalid_request_id_refuses_before_prompt_encoding_and_dispatch(request_id):
    class EncodeForbidden(str):
        def encode(self, *_args, **_kwargs):
            pytest.fail("invalid identity encoded its prompt")

    async def exercise():
        backend = Backend()
        wrapper = n._PromptIdentityBackend(backend, capture_policy=policy())
        with pytest.raises(ValueError, match="record budget"):
            await consume(wrapper, EncodeForbidden("forbidden"), request_id)
        assert wrapper.identities() == []
        assert backend.dispatched == []

    asyncio.run(exercise())


def test_encoded_prompt_and_record_are_released_before_backend_dispatch_or_yield():
    async def exercise():
        holder = {}

        class InspectBackend(Backend):
            def generate(self, *args, **kwargs):
                frame = holder["stream"].ag_frame
                assert "encoded" not in frame.f_locals
                assert "record" not in frame.f_locals
                return super().generate(*args, **kwargs)

        wrapper = n._PromptIdentityBackend(InspectBackend(), capture_policy=policy())
        stream = holder["stream"] = wrapper.generate("actual borrowed input", request_id="outer-step-0", max_tokens=4)
        assert await anext(stream) == "chunk"
        assert "encoded" not in stream.ag_frame.f_locals
        assert "record" not in stream.ag_frame.f_locals
        await stream.aclose()

    asyncio.run(exercise())


def test_cancel_clears_active_metadata_before_delegate_preserves_failure_snapshot_and_refuses_reuse():
    async def exercise():
        class InspectBackend(Backend):
            async def cancel(self, request_id):
                assert wrapper.identities() == []
                raise RuntimeError("retirement failed")

        wrapper = n._PromptIdentityBackend(InspectBackend(), capture_policy=policy())
        await consume(wrapper)
        expected = wrapper.identities()
        with pytest.raises(RuntimeError, match="retirement failed"):
            await wrapper.cancel("outer-step-0")
        with pytest.raises(ValueError, match="must be reset"):
            await consume(wrapper, request_id="fresh-step-0")
        rows, captured_policy = wrapper.snapshot_and_reset()
        assert rows == expected and captured_policy == policy()
        await consume(wrapper, request_id="fresh-step-0")
        assert wrapper.identities()[0]["step"] == 0

    asyncio.run(exercise())


def test_close_clears_all_current_and_retired_metadata_before_failed_delegate():
    async def exercise():
        class InspectBackend(Backend):
            def close(self):
                assert wrapper.snapshot_and_reset() == ([], policy())
                raise RuntimeError("shutdown unverified")

        wrapper = n._PromptIdentityBackend(InspectBackend(), capture_policy=policy())
        await consume(wrapper)
        await wrapper.cancel("outer-step-0")
        with pytest.raises(RuntimeError, match="shutdown unverified"):
            wrapper.close()
        assert wrapper.identities() == []

    asyncio.run(exercise())


def test_failed_prompt_encoding_does_not_consume_step_budget():
    async def exercise():
        wrapper = n._PromptIdentityBackend(Backend(), capture_policy=policy(1))
        with pytest.raises(UnicodeError):
            await consume(wrapper, "bad\ud800")
        assert wrapper.identities() == []
        await consume(wrapper)
        assert wrapper.identities()[0]["step"] == 0

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_model_steps", True),
        ("max_model_steps", 0),
        ("max_record_bytes", 32768),
        ("transient_copies", True),
        ("declared_metadata_bytes", 1),
        ("schema", "unknown"),
    ],
)
def test_forged_policy_is_not_accepted(field, value):
    with pytest.raises(ValueError, match="capture policy"):
        n._PromptIdentityBackend(Backend(), capture_policy={**policy(), field: value})


def test_policy_uses_actual_controller_limit_and_rejects_workspace_overflow():
    contract = AgentOutputContract("strict_outer_json_fence_agent_v1")
    assert policy(6)["declared_metadata_bytes"] == 6 * 2 * 16384
    with pytest.raises(ValueError, match="workspace"):
        model_step_capture_policy(contract, 100)


@pytest.mark.parametrize("actual_limit", [8, 100])
def test_prepare_derives_policy_from_controller_limits_before_model_load(monkeypatch, tmp_path, actual_limit):
    import vllm_omni.edge.agent.native_app as app

    base = "strata:" + "a" * 64
    contract = AgentOutputContract("strict_outer_json_fence_agent_v1")
    consumer = contract.consumer_identity(base)
    route = ProfileRoute(
        "r",
        "model",
        "strata-agent:" + consumer["identity_sha256"],
        "rev",
        "b" * 64,
        "Q4",
        n.STRATA_BACKEND,
        "cpu+cuda:0",
        {
            "base_engine_artifact_id": base,
            "backend_config_sha256": "a" * 64,
            "model_output_consumer_identity": consumer,
        },
    )
    loaded = []

    class ColdBackend(Backend):
        def __init__(self):
            super().__init__()
            self.execution_plan = None

        def start(self):
            loaded.append(True)
            self.execution_plan = {"requested_device": "cpu+cuda:0"}

    backend = ColdBackend()
    controller = SimpleNamespace(
        limits=SimpleNamespace(max_model_steps=actual_limit),
        routes=[SimpleNamespace(route_id="r")],
        backends={"r": backend},
        tools=SimpleNamespace(close=lambda: None),
        add_listener=lambda _listener: None,
        admit=lambda _route: SimpleNamespace(admitted=True),
        close=lambda: None,
    )
    monkeypatch.setattr(app, "build_controller", lambda _path: (controller, {}))
    monkeypatch.setattr(n, "_FixtureForegroundScreen", lambda: object())
    monkeypatch.setattr(n, "ManagedEdgeBrowser", lambda **_kwargs: object())
    monkeypatch.setattr(n, "ReadOnlyFixtureTools", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(n, "validate_strata_profile_plan", lambda *_args: None)
    bridge = n.NativeProfileBridge(
        native_config={"routes": [{"route_id": "r"}]},
        config_root=tmp_path / "config",
        private_root=tmp_path / "private",
        fixture_origin="unused",
        telemetry=SimpleNamespace(sample=lambda: {"ram_used_bytes": 1, "vram_used_bytes": 2}),
    )
    try:
        if actual_limit == 100:
            with pytest.raises(ValueError, match="workspace"):
                asyncio.run(bridge.prepare(route))
            assert loaded == []
        else:
            asyncio.run(bridge.prepare(route))
            assert loaded == [True]
            assert bridge._prompt_backend.snapshot_and_reset()[1] == policy(actual_limit)
    finally:
        bridge.close()


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_bridge_finally_snapshots_before_emit_and_binds_original_trusted_case(monkeypatch, structured, explicit):
    async def exercise():
        route = ProfileRoute("r", "model", "artifact", "rev", "a" * 64, "Q4", "fixture", "cpu")
        backend = Backend()
        wrapper = n._PromptIdentityBackend(backend, capture_policy=policy(2) if explicit else None)
        bridge = object.__new__(n.NativeProfileBridge)
        bridge.route, bridge._prompt_backend = route, wrapper
        bridge._lock = threading.RLock()
        bridge.structured_read_url, bridge.fixture_origin = structured, "unused"
        url = "http://127.0.0.1:1234/task" if structured else None
        if structured:
            monkeypatch.setattr(n, "_structured_input", lambda *_args: (url, "trusted task"))
        received = []

        def submit(text):
            assert text == "trusted task"
            result = Future()

            async def turn():
                bridge._listen(
                    {
                        "kind": "route",
                        "payload": {
                            "model": route.model_id,
                            "artifact_id": route.artifact_id,
                            "backend": route.backend,
                            "actual_placement": "cpu",
                        },
                    }
                )
                await consume(wrapper, "first backend input", "outer-step-0")
                await consume(wrapper, "second backend input", "outer-step-1")
                result.set_result("answer")

            asyncio.create_task(turn())
            return result

        def submit_read_url(actual_url, text):
            assert actual_url == url
            return submit(text)

        bridge.controller = SimpleNamespace(
            backends={route.route_id: wrapper},
            submit=submit,
            submit_read_url=submit_read_url,
        )

        def trace_complete(events, answer, selected_route, evidence, *, trusted_task_binding):
            assert trusted_task_binding == {
                "text": "trusted task",
                "read_url": url,
                "mode": "read_url" if structured else "ordinary",
            }
            assert selected_route == route and answer == "answer"
            if explicit:
                assert len(evidence["model_step_identities"]) == 2
            else:
                assert "model_step_identities" not in evidence
                assert "model_step_identity_policy" not in evidence
            return False

        monkeypatch.setattr(n, "_trace_complete", trace_complete)

        def emit(kind, payload):
            if kind == "model_prompt_identity":
                assert wrapper.identities() == []
            received.append((kind, payload))

        result = await bridge.run(route, AgentCase("case", "basic", "en", "short", "trusted task"), emit)
        prompt = next(payload for kind, payload in received if kind == "model_prompt_identity")
        evidence = result.placement_evidence
        if explicit:
            assert set(prompt) == {"first", "model_steps", "steps", "policy"}
            assert prompt["steps"] == evidence["model_step_identities"]
            assert prompt["policy"] == evidence["model_step_identity_policy"] == policy(2)
            prompt["steps"][0]["sha256"] = "tampered outer copy"
            assert evidence["model_step_identities"][0]["sha256"] != "tampered outer copy"
        else:
            assert set(prompt) == {"first", "model_steps"}
        assert set(prompt["first"]) == {"step", "sha256", "utf8_bytes", "chars"}
        assert prompt["first"] == evidence["first_model_prompt_identity"]
        assert prompt["model_steps"] == evidence["model_prompt_step_count"] == 2
        assert bridge._events is None and bridge._emitter is None

    asyncio.run(exercise())


def test_bridge_finally_resets_metadata_even_when_emitter_fails():
    async def exercise():
        route = ProfileRoute("r", "model", "artifact", "rev", "a" * 64, "Q4", "fixture", "cpu")
        wrapper = n._PromptIdentityBackend(Backend(), capture_policy=policy())
        bridge = object.__new__(n.NativeProfileBridge)
        bridge.route, bridge._prompt_backend = route, wrapper
        bridge._lock, bridge.structured_read_url = threading.RLock(), False

        def submit(_text):
            result = Future()

            async def turn():
                await consume(wrapper)
                result.set_exception(RuntimeError("actual request failed"))

            asyncio.create_task(turn())
            return result

        bridge.controller = SimpleNamespace(submit=submit)

        def emit(kind, _payload):
            if kind == "model_prompt_identity":
                assert wrapper.identities() == []
                raise OSError("writer failed")

        with pytest.raises(OSError, match="writer failed"):
            await bridge.run(route, AgentCase("case", "basic", "en", "short", "trusted task"), emit)
        assert wrapper.snapshot_and_reset()[0] == []
        assert bridge._events is None and bridge._emitter is None

    asyncio.run(exercise())


def test_bridge_close_clears_capture_before_underlying_controller_release():
    wrapper = n._PromptIdentityBackend(Backend(), capture_policy=policy())
    asyncio.run(consume(wrapper))
    bridge = object.__new__(n.NativeProfileBridge)
    bridge._prompt_backend, bridge._lock = wrapper, threading.RLock()
    bridge._events, bridge._emitter = [{"stale": True}], object()

    def close():
        assert wrapper.identities() == []
        assert bridge._events is None and bridge._emitter is None
        raise RuntimeError("quarantined worker")

    controller = bridge.controller = SimpleNamespace(close=close)
    with pytest.raises(RuntimeError, match="quarantined worker"):
        bridge.close()
    assert bridge.controller is controller  # Failed release must not discard ownership.
