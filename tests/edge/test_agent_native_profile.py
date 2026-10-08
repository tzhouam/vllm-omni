# SPDX-License-Identifier: Apache-2.0
"""Native profiling bridge invariants without a GPU or Windows shell."""

from __future__ import annotations

import asyncio
import hashlib
from concurrent.futures import Future
from dataclasses import asdict, replace
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from benchmarks.edge_agent import native_profile
from benchmarks.edge_agent.native_profile import (
    SUITE_ID, NativeProfileBridge, _trace_complete, load_profile_routes,
)
from benchmarks.edge_agent.evidence import audit_summary
from benchmarks.edge_agent.paired_suite import (
    FixtureSite, ReadOnlyFixtureTools, build_paired_cases, evaluate_case,
)
from benchmarks.edge_agent.profile import (
    AgentCase, AgentRunResult, Preparation, ProfileConditions, ProfileConfig,
    ProfileRoute, run_profile,
)
from vllm_omni.edge.agent.tools import ToolAction, _explicit_task_urls
from vllm_omni.edge.agent.controller import _RECALL_KINDS
from vllm_omni.edge.agent.memory import AesGcmCipher, EncryptedMemoryStore
from vllm_omni.edge.agent.qualification import _evaluate_case as audit_evaluate_case
from vllm_omni.edge.agent.router import classify_task
from vllm_omni.engine.resource_ledger import ResourceUnavailable


def _route():
    return ProfileRoute(
        route_id="r", model_id="model", artifact_id="artifact",
        checkpoint_revision="commit", artifact_sha256="a" * 64,
        precision="Q4", backend="external.llamacpp.text.v1",
        expected_placement="cpu",
    )


def test_lineage_binds_exact_native_hash_and_projector():
    native = {"routes": [{
        "route_id": "r", "model": "model", "artifact_id": "artifact",
        "model_sha256": "a" * 64, "server_sha256": "b" * 64,
        "model_file": "model.gguf", "server_bin": "server.exe",
        "placement": "cpu",
    }]}
    lineage = {"routes": {"r": {
        "model_sha256": "a" * 64, "mmproj_sha256": None,
        "checkpoint_revision": "local-verified", "precision": "Q4",
        "lineage_verified": True,
    }}}
    profiles, provenance = load_profile_routes(native, lineage)
    assert profiles == [_route().__class__(**{**_route().__dict__, "checkpoint_revision": "local-verified"})]
    assert provenance["r"]["lineage_verified"]
    lineage["routes"]["r"]["model_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="hash differs"):
        load_profile_routes(native, lineage)


def test_loopback_fixtures_and_paired_languages_lengths():
    with FixtureSite() as site:
        suites = build_paired_cases(site.origin, mouse_speed=10)
        assert set(suites) == {
            "basic", "browser_text", "browser_vision", "windows_settings",
            "memory", "code_tools", "long_reasoning",
        }
        for task, buckets in suites.items():
            assert set(buckets) == {"short", "medium", "long"}
            for length, cases in buckets.items():
                assert {case.language for case in cases} == {"en-US", "zh-CN"}
                assert all(case.task_class == task and case.length == length for case in cases)
                assert all(classify_task(case.prompt) == task for case in cases)
            assert len(buckets["short"][0].prompt) < len(buckets["medium"][0].prompt)
            assert len(buckets["medium"][0].prompt) < len(buckets["long"][0].prompt)
        assert all(len(case.prompt) < 2_000 for cases in suites["basic"].values()
                   for case in cases)
        assert {case.reference for cases in suites["basic"].values()
                for case in cases} == {"22"}
        assert {case.reference for cases in suites["long_reasoning"].values()
                for case in cases} == {"11"}
        for task_class in ("browser_text", "browser_vision"):
            for cases in suites[task_class].values():
                for case in cases:
                    source = case.metadata["source"]
                    if source.startswith("http://"):
                        assert _explicit_task_urls(case.prompt) == frozenset({source})
        text = urlopen(site.origin + "/text/en").read().decode()
        visual = urlopen(site.origin + "/visual/en").read().decode()
        assert "CEDAR-4827" in text
        assert "ORBIT-7391" not in visual
        assert f"Omni visual fixture {site.origin.rsplit(':', 1)[-1]} en" in visual
        assert 'src="data:image/svg+xml;base64,' in visual
        assert "ORBIT-7391" in urlopen(site.origin + "/assets/en.svg").read().decode()


def test_structured_read_url_uses_only_exact_canonical_browser_text():
    with FixtureSite() as site:
        cases = build_paired_cases(site.origin, 10)
        case = cases["browser_text"]["long"][0]
        url, instruction = native_profile._structured_input(case, site.origin)
        assert url == case.metadata["source"]
        assert instruction == case.prompt  # Preserve the paired input, including URL and filler.
        contract = native_profile._input_contract(
            case, structured_read_url=True, fixture_origin=site.origin)
        ordinary = native_profile._input_contract(
            case, structured_read_url=False, fixture_origin=site.origin)
        assert contract["submission_mode"] == native_profile.STRUCTURED_READ_URL_MODE
        assert contract["explicit_read_url"] == url
        assert contract["instruction_sha256"] == hashlib.sha256(case.prompt.encode()).hexdigest()
        assert contract["contract_sha256"] != ordinary["contract_sha256"]
        with pytest.raises(ValueError, match="canonical fixed fixture"):
            native_profile._structured_input(replace(case, prompt=case.prompt + " extra"), site.origin)
        with pytest.raises(ValueError, match="browser_text fixtures only"):
            native_profile._structured_input(cases["basic"]["short"][0], site.origin)


def test_structured_read_url_bridge_calls_separate_controller_api(tmp_path):
    with FixtureSite() as site:
        case = build_paired_cases(site.origin, 10)["browser_text"]["short"][0]
        calls = []

        def submit_read_url(url, instruction):
            calls.append((url, instruction))
            future = Future()
            future.set_result("answer")
            return future

        bridge = NativeProfileBridge(
            native_config={}, config_root=tmp_path, private_root=tmp_path,
            fixture_origin=site.origin,
            telemetry=SimpleNamespace(sample=lambda: {"ram_used_bytes": 1}),
            structured_read_url=True,
        )
        bridge.controller = SimpleNamespace(
            submit_read_url=submit_read_url,
            submit=lambda _prompt: pytest.fail("ordinary submit should not be used"),
            memory=SimpleNamespace(_path=tmp_path / "memory.sqlite", delete_all=lambda: 0),
            backends={"r": SimpleNamespace(execution_plan={"requested_device": "cpu"})},
        )
        bridge.route = _route()
        bridge._memory_path = tmp_path / "memory.sqlite"
        setup = bridge.before_request(_route(), case, "measured", 0)
        assert setup["input_contract"]["submission_mode"] == native_profile.STRUCTURED_READ_URL_MODE
        assert setup["input_contract"]["explicit_read_url"] == case.metadata["source"]
        result = asyncio.run(bridge.run(_route(), case, lambda *_: None))
        assert result.final_answer == "answer"
        assert calls == [(case.metadata["source"], case.prompt)]
        assert bridge.suite_id == native_profile.STRUCTURED_READ_URL_SUITE_ID


def test_bridge_preserves_native_memory_refusal_before_backend_lookup(monkeypatch, tmp_path):
    import vllm_omni.edge.agent.native_app as native_app

    native_route = SimpleNamespace(route_id="r")
    closed = []
    controller = SimpleNamespace(
        routes=[native_route], backends={},
        admit=lambda route: (SimpleNamespace(
            admitted=False,
            reason="host_ram: declared demand 22000000000 bytes exceeds native controller ceiling 20647059456 bytes",
        ) if route is native_route else pytest.fail("wrong route was admitted")),
        tools=SimpleNamespace(close=lambda: None),
        add_listener=lambda _listener: None,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(native_app, "build_controller", lambda _path: (controller, {}))
    monkeypatch.setattr(native_profile, "_FixtureForegroundScreen", lambda: object())
    monkeypatch.setattr(native_profile, "ManagedEdgeBrowser", lambda **_kwargs: object())
    monkeypatch.setattr(native_profile, "ReadOnlyFixtureTools", lambda *_args, **_kwargs: object())
    bridge = NativeProfileBridge(
        native_config={"routes": [{"route_id": "r"}]},
        config_root=tmp_path / "configs", private_root=tmp_path / "private",
        fixture_origin="http://127.0.0.1:1234",
        telemetry=SimpleNamespace(sample=lambda: pytest.fail("model load was attempted")),
    )
    try:
        with pytest.raises(ResourceUnavailable, match="host_ram: declared demand 22000000000"):
            asyncio.run(bridge.prepare(_route()))
    finally:
        bridge.close()
    assert closed == [True]


def test_prompt_identity_backend_preserves_cancel_and_close():
    class Backend:
        def __init__(self):
            self.entered = asyncio.Event()
            self.finalized = False
            self.cancelled = []
            self.closed = False

        async def generate(self, prompt, *, request_id, max_tokens, image_data_url=None):
            assert (prompt, request_id, max_tokens, image_data_url) == (
                "actual model prompt", "request-step-0", 4, None)
            try:
                self.entered.set()
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                self.finalized = True

        async def cancel(self, request_id):
            self.cancelled.append(request_id)

        def close(self):
            self.closed = True

    async def exercise():
        backend = Backend()
        wrapper = native_profile._PromptIdentityBackend(backend)

        async def consume():
            async for _ in wrapper.generate(
                "actual model prompt", request_id="request-step-0", max_tokens=4):
                pass

        task = asyncio.create_task(consume())
        await backend.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wrapper.cancel("request-step-0")
        wrapper.close()
        assert backend.finalized and backend.cancelled == ["request-step-0"]
        assert backend.closed
        assert wrapper.identities() == [{
            "step": 0,
            "sha256": hashlib.sha256(b"actual model prompt").hexdigest(),
            "utf8_bytes": len(b"actual model prompt"),
            "chars": len("actual model prompt"),
        }]
        wrapper.reset()
        assert wrapper.identities() == []

    asyncio.run(exercise())


def test_structured_mode_rejects_other_task_classes():
    assert native_profile._profile_classes(None, True) == {"browser_text"}
    assert native_profile._profile_classes({"browser_text"}, True) == {"browser_text"}
    with pytest.raises(ValueError, match="requires only"):
        native_profile._profile_classes({"browser_text", "basic"}, True)
    assert native_profile._profile_classes({"basic"}, False) == {"basic"}


def test_desktop_fixture_requires_verified_foreground_before_request(monkeypatch, tmp_path):
    with FixtureSite() as site:
        case = next(row for row in build_paired_cases(site.origin, 10)["browser_vision"]["short"]
                    if row.metadata["kind"] == "desktop_screen")
        marker = f"Omni visual fixture {site.origin.rsplit(':', 1)[-1]} en"

        class Browser:
            def open(self, url):
                self.url = url
                return {"url": url, "title": marker}

            def bring_to_front(self):
                return {"url": self.url, "title": marker}

        path = tmp_path / "memory.sqlite"
        bridge = NativeProfileBridge(
            native_config={}, config_root=tmp_path, private_root=tmp_path,
            fixture_origin=site.origin,
            telemetry=SimpleNamespace(sample=lambda: {"ram_used_bytes": 1}),
        )
        bridge.controller = SimpleNamespace(
            memory=SimpleNamespace(_path=path, delete_all=lambda: 0),
            tools=SimpleNamespace(browser=Browser()),
        )
        bridge.route = _route()
        bridge._memory_path = path
        bridge._fixture_screen = SimpleNamespace(expected_title=None)
        monkeypatch.setattr(native_profile, "_foreground_fixture_window",
                            lambda title: {"foreground_window_verified": title == marker})
        evidence = bridge.before_request(_route(), case, "measured", 0)
        assert evidence["fixture_visibility_checked_before_request"] is True
        assert evidence["fixture_foreground_window"]["foreground_window_verified"] is True
        assert bridge._fixture_screen.expected_title == marker

        def reject(_title):
            raise RuntimeError("cannot foreground fixture")

        monkeypatch.setattr(native_profile, "_foreground_fixture_window", reject)
        with pytest.raises(RuntimeError, match="cannot foreground fixture"):
            bridge.before_request(_route(), case, "measured", 1)
        assert bridge._fixture_screen.expected_title is None


def test_desktop_fixture_rechecks_foreground_on_actual_capture(monkeypatch):
    checks = []
    monkeypatch.setattr(native_profile, "_verify_fixture_is_foreground",
                        lambda title: checks.append(title))
    screen = native_profile._FixtureForegroundScreen.__new__(native_profile._FixtureForegroundScreen)
    screen.expected_title = "unique title"
    screen._screen = SimpleNamespace(capture=lambda: {"mime_type": "image/jpeg"})
    captured = screen.capture()
    assert captured["fixture_foreground_verified"] is True
    assert checks == ["unique title", "unique title"]
    screen.expected_title = None
    with pytest.raises(RuntimeError, match="not prepared"):
        screen.capture()


@pytest.mark.parametrize("task_class", ["basic", "long_reasoning"])
def test_self_contained_cases_have_independent_exact_answers(task_class):
    with FixtureSite() as site:
        cases = build_paired_cases(site.origin, mouse_speed=10)[task_class]
        for examples in cases.values():
            for case in examples:
                result = AgentRunResult(
                    final_answer=case.reference, complete_agent_trace=True,
                    model_id="m", artifact_id="a", actual_placement="cpu",
                    backend="b", tool_decisions=(),
                )
                assert evaluate_case(case, result).success
                assert asdict(evaluate_case(case, result)) == audit_evaluate_case(
                    asdict(case), asdict(result),
                )
                assert not evaluate_case(case, replace(result, final_answer="999")).success
                # The evaluator recomputes the fixture answer; editing a case's
                # reference to match a wrong model answer cannot make it pass.
                forged_case = replace(case, reference="999")
                assert not evaluate_case(forged_case, replace(result, final_answer="999")).success


def test_memory_fixture_seed_is_recallable_and_reset_between_cases(tmp_path):
    with FixtureSite() as site:
        case = build_paired_cases(site.origin, mouse_speed=10)["memory"]["short"][0]
        path = tmp_path / "profile-memory.sqlite"
        with EncryptedMemoryStore(path, cipher=AesGcmCipher(b"k" * 32)) as store:
            bridge = NativeProfileBridge(
                native_config={}, config_root=tmp_path, private_root=tmp_path,
                fixture_origin=site.origin,
                telemetry=SimpleNamespace(sample=lambda: {"ram_used_bytes": 1}),
            )
            bridge.controller = SimpleNamespace(memory=store)
            bridge.route = _route()
            bridge._memory_path = path
            first = bridge.before_request(_route(), case, "measured", 0)
            matches = store.search(case.prompt, kinds=_RECALL_KINDS)
            assert [match.event.event_id for match in matches] == [first["seed_event_id"]]
            assert matches[0].event.kind == "user_observation"
            second = bridge.before_request(_route(), case, "measured", 1)
            assert second["deleted_prior_fixture_events"] == 1
            assert store.get_event(first["seed_event_id"]) is None
            assert [match.event.event_id for match in store.search(case.prompt, kinds=_RECALL_KINDS)] == [
                second["seed_event_id"]
            ]


@pytest.mark.parametrize("task_class", ["basic", "long_reasoning"])
def test_native_cli_accepts_new_task_classes(monkeypatch, tmp_path, capsys, task_class):
    captured = {}

    async def fake_profile(**kwargs):
        captured.update(kwargs)
        return tmp_path / "index.json"

    monkeypatch.setattr(native_profile, "run_native_profile", fake_profile)
    monkeypatch.setattr("sys.argv", [
        "native_profile", "--config", "config.json", "--lineage", "lineage.json",
        "--output-dir", str(tmp_path), "--task-class", task_class, "--smoke",
    ])
    native_profile.main()
    assert captured["selected_classes"] == {task_class}
    assert captured["smoke"] is True
    assert capsys.readouterr().out.strip() == str(tmp_path / "index.json")


def test_native_cli_selects_structured_read_url(monkeypatch, tmp_path):
    captured = {}

    async def fake_profile(**kwargs):
        captured.update(kwargs)
        return tmp_path / "index.json"

    monkeypatch.setattr(native_profile, "run_native_profile", fake_profile)
    monkeypatch.setattr("sys.argv", [
        "native_profile", "--config", "config.json", "--lineage", "lineage.json",
        "--output-dir", str(tmp_path), "--task-class", "browser_text",
        "--structured-read-url", "--smoke",
    ])
    native_profile.main()
    assert captured["selected_classes"] == {"browser_text"}
    assert captured["structured_read_url"] is True


def test_evaluator_requires_exact_answer_and_actual_read_evidence():
    with FixtureSite() as site:
        case = build_paired_cases(site.origin, 10)["browser_text"]["short"][0]
        result = AgentRunResult(
            final_answer=case.reference, complete_agent_trace=True,
            model_id="m", artifact_id="a", actual_placement="cpu",
            backend="b", tool_decisions=(
                {"kind": "tool_proposed", "payload": {"operation": "browser_open",
                                                      "arguments": {"url": case.metadata["source"]}}},
                {"kind": "tool_result", "payload": {"operation": "browser_open",
                                                    "source": case.metadata["source"]}},
                {"kind": "tool_result", "payload": {"operation": "browser_read",
                                                    "source": case.metadata["source"]}},
            ),
        )
        assert evaluate_case(case, result).success
        unsafe = AgentRunResult(**{**result.__dict__, "tool_decisions": (
            *result.tool_decisions,
            {"kind": "tool_proposed", "payload": {"operation": "browser_click"}},
        )})
        assert not evaluate_case(case, unsafe).tool_safe
        guessed = AgentRunResult(**{**result.__dict__, "tool_decisions": ()})
        assert not evaluate_case(case, guessed).success


def test_benchmark_tools_reject_writes_and_external_navigation():
    tools = ReadOnlyFixtureTools("http://127.0.0.1:12345")
    with pytest.raises(PermissionError, match="forbids"):
        tools.execute(ToolAction("settings_set", {"setting": "mouse_speed", "value": 12}))
    with pytest.raises(PermissionError, match="loopback"):
        tools.execute(ToolAction("browser_open", {"url": "https://example.com"}))
    tools.close()


def test_trace_requires_order_and_final_identity():
    route = _route()
    events = [
        {"seq": 1, "request_id": "req", "epoch": 1, "kind": "user_observation", "payload": {}},
        {"seq": 2, "request_id": "req", "epoch": 1, "kind": "route", "payload": {
            "route_id": route.route_id, "model": route.model_id,
            "artifact_id": route.artifact_id, "backend": route.backend,
            "actual_placement": route.expected_placement,
        }},
        {"seq": 3, "request_id": "req", "epoch": 1, "kind": "model_metrics", "payload": {}},
        {"seq": 4, "request_id": "req", "epoch": 1, "kind": "final", "payload": {"answer": "ok"}},
    ]
    assert _trace_complete(events, "ok", route)
    assert not _trace_complete([{**events[0], "seq": 2}, *events[1:]], "ok", route)
    assert not _trace_complete(events, "wrong", route)


def test_raw_audit_recomputes_counts_and_rejects_tampering(tmp_path):
    route = _route()
    cases = {
        length: [AgentCase(
            case_id=f"code-{length}", task_class="code_tools",
            language="en-US", length=length, prompt="What does the code print?",
            reference="14", metadata={"kind": "code_reasoning",
                                       "required_operations": [],
                                       "quality_scope": "code_result_only_no_execution_tool"},
        )] for length in ("short", "medium", "long")
    }

    async def runner(_route, _case, emit):
        pieces = [
            ("user_observation", {}),
            ("route", {"route_id": route.route_id, "model": route.model_id,
                       "artifact_id": route.artifact_id, "backend": route.backend,
                       "actual_placement": route.expected_placement}),
            ("model_metrics", {}),
            ("final", {"answer": "14"}),
        ]
        for seq, (kind, payload) in enumerate(pieces, 1):
            emit("agent_event", {"seq": seq, "request_id": "one", "epoch": 1,
                                 "kind": kind, "payload": payload})
            if kind == "model_metrics":
                emit("assistant_text_delta", "14")
        return AgentRunResult(
            final_answer="14", complete_agent_trace=True,
            model_id=route.model_id, artifact_id=route.artifact_id,
            actual_placement=route.expected_placement, backend=route.backend,
            placement_evidence={"plan": "cpu"},
        )

    async def prepare(_route):
        return Preparation(True, route.artifact_id, route.expected_placement,
                           {"new_worker": True})

    conditions = ProfileConditions(
        hardware_id="test", os_version="test", driver_versions={},
        runtime_versions={}, power_condition="AC", suite_id=SUITE_ID,
        environment_fingerprint="test",
    )
    import asyncio
    import json

    summary = asyncio.run(run_profile(
        routes=[route], cases_by_length=cases, runner=runner,
        evaluator=evaluate_case, conditions=conditions, output_dir=tmp_path,
        config=ProfileConfig(warmups_per_length=1, measured_per_length=1,
                             endurance_seconds=0, telemetry_interval_seconds=.01),
        prepare=prepare,
        before_request=lambda *_: {"private_memory_reset": True},
        telemetry=lambda: {"ram_used_bytes": 1},
    ))
    summary_path = summary.run_directory / "summary.json"
    valid = audit_summary(summary_path)
    assert valid.internally_valid
    assert valid.trace_verified
    assert valid.measured_successes == 3
    assert not valid.protocol_compliant
    tampered = json.loads(summary_path.read_text())
    tampered["routes"][route.route_id]["measured_successes"] = 999
    summary_path.write_text(json.dumps(tampered))
    assert "route profile differs" in " ".join(audit_summary(summary_path).errors)
