"""Cross-session quality harness tests without native inference or a GPU."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from benchmarks.edge_agent.experiments.native_memory_quality import (
    SUITE_ID,
    assess_case,
    build_cases,
    run_suite,
)
from benchmarks.edge_agent.native_profile import NativeProfileBridge, _PromptIdentityBackend
from benchmarks.edge_agent.paired_suite import build_paired_cases, evaluate_case
from benchmarks.edge_agent.profile import AgentCase, AgentRunResult, ProfileRoute, _one_request
from vllm_omni.edge.agent.consumer_trace import model_step_capture_policy
from vllm_omni.edge.agent.controller import AgentController, AgentLimits
from vllm_omni.edge.agent.memory import AesGcmCipher, EncryptedMemoryStore, MemoryEvent
from vllm_omni.edge.agent.model_output import AgentOutputContract
from vllm_omni.edge.agent.omni_backend import BackendChunk
from vllm_omni.edge.agent.router import Admission, Route, classify_task
from vllm_omni.edge.agent.tools import WindowsToolBoundary


class _NoBrowser:
    def close(self) -> None:
        pass


class _MemoryBackend:
    def __init__(self, cases, *, answer_mode="retrieve") -> None:
        self.cases = cases
        self.answer_mode = answer_mode
        self.execution_plan = {"requested_device": "cpu",
                               "observed_model_placement": "cpu"}
        self.prompts = []

    def start(self) -> None:
        pass

    async def generate(self, prompt, *, request_id, max_tokens, image_data_url=None):
        self.prompts.append(prompt)
        message = json.loads(prompt)
        task = message["task"]
        if any(task == seed for case in self.cases for seed in (
            case.source_prompt, case.distractor_prompt,
        )):
            answer = "noted"
        else:
            case = next(case for case in self.cases if task == case.recall_prompt)
            recalled = message["recalled_memory"]
            target_seen = any(
                case.source_prompt in item["text"] and item["kind"] == "user_observation"
                for item in recalled
            )
            if self.answer_mode == "distractor" and target_seen:
                answer = case.distractor_reference
            elif self.answer_mode == "always_guess":
                answer = case.reference
            else:
                answer = case.reference if target_seen else "UNKNOWN"
        yield BackendChunk(answer, terminal=True)

    async def cancel(self, request_id) -> None:
        pass

    def close(self) -> None:
        pass


def _factory(tmp_path, cases, *, answer_mode="retrieve", max_matches=5):
    controllers = []
    backends = []
    route = Route(
        "gemma-test", "artifact", "Gemma 4 31B test", "external.llamacpp.text.v1",
        frozenset({"text"}), "cpu", {"host_ram": 1},
    )

    def create(phase):
        backend = _MemoryBackend(cases, answer_mode=answer_mode)
        backends.append(backend)
        controller = AgentController(
            routes=[route], qualifications=[], backends={route.route_id: backend},
            memory=EncryptedMemoryStore(
                tmp_path / "encrypted-memory.sqlite", AesGcmCipher(b"k" * 32),
            ),
            tools=WindowsToolBoundary(browser=_NoBrowser()),
            admit=lambda _: Admission(True, "test preflight"),
            environment_fingerprint="test", power_condition="AC",
            qualification_suite_id=SUITE_ID,
            bootstrap_route_id=route.route_id,
            limits=AgentLimits(max_memory_matches=max_matches),
        )
        controllers.append((phase, controller))
        return controller

    return create, controllers, backends


def test_randomized_bilingual_cases_have_no_answer_in_recall_prompt() -> None:
    cases = build_cases(b"fixed-private-test-seed-0123456789")
    assert len(cases) == 4
    assert {case.language for case in cases} == {"en-US", "zh-CN"}
    assert len({code for case in cases for code in (
        case.reference, case.distractor_reference,
    )}) == 8
    for case in cases:
        assert case.reference in case.source_prompt
        assert case.distractor_reference in case.distractor_prompt
        assert case.reference not in case.recall_prompt
        assert case.distractor_reference not in case.recall_prompt
        assert case.reference != case.distractor_reference
    assert cases != build_cases(b"different-private-seed-0123456789")


def test_real_controller_store_reopens_and_source_deletion_cascades(tmp_path) -> None:
    cases = build_cases(b"fixed-private-test-seed-0123456789")
    records = []
    # This exact sequence reproduced a real 3/4 miss when generic Chinese
    # recall instructions outranked the older Amber compass source at top 5.
    create, controllers, backends = _factory(tmp_path, cases, max_matches=5)
    summary = run_suite(cases, controller_factory=create, timeout_s=5,
                        on_record=records.append)
    assert [phase for phase, _ in controllers] == ["seed", "replay"]
    assert controllers[0][1].session_id != controllers[1][1].session_id
    assert summary["cases"] == summary["passed"] == 4, summary["case_assessments"]
    assert summary["qualifies_default"] is False
    assert summary["review_status"] == "unreviewed_private_experiment"
    quality = [record for record in records if record["record_type"] == "quality_case"]
    assert len(quality) == 4
    for case, record in zip(cases, quality):
        assert record["checks"]["passed"]
        assert record["source_event_id"] in record["recall"]["final_derived_from"]
        assert record["distractor_event_id"] in record["recall"]["final_derived_from"]
        assert record["deletion"]["deleted_count"] >= 2
        assert record["after_delete"]["answer"] == "UNKNOWN"
        assert record["reference_sha256"] != record["recall"]["prompt_sha256"]
        assert record["recall"]["answer_sha256"] == record["reference_sha256"]
        assert record["after_delete"]["answer_sha256"] != record["reference_sha256"]
        assert case.reference not in record["recall"]["prompt"]
    # The second controller receives the first session's source events in its
    # actual model prompt, despite the user's recall prompt omitting the code.
    replay_prompts = [json.loads(prompt) for prompt in backends[1].prompts]
    assert any(cases[0].reference in json.dumps(prompt) for prompt in replay_prompts)
    assert b"fixed-private-test-seed" not in (tmp_path / "encrypted-memory.sqlite").read_bytes()
    with EncryptedMemoryStore(tmp_path / "encrypted-memory.sqlite",
                              AesGcmCipher(b"k" * 32)) as store:
        assert all(store.get_event(record["source_event_id"]) is None for record in quality)
        assert all(store.get_event(record["distractor_event_id"]) is not None
                   for record in quality)


def test_wrong_distractor_answer_cannot_pass_even_with_provenance(tmp_path) -> None:
    cases = build_cases(b"fixed-private-test-seed-0123456789")[:1]
    records = []
    create, _, _ = _factory(tmp_path, cases, answer_mode="distractor")
    summary = run_suite(cases, controller_factory=create, timeout_s=5,
                        on_record=records.append)
    assert summary["cases"] == 1 and summary["passed"] == 0
    check = next(record for record in records if record["record_type"] == "quality_case")
    assert check["checks"]["source_and_distractor_retrieved"]
    assert not check["checks"]["reference_answer_exact"]


def test_guess_after_deletion_fails_quality_gate(tmp_path) -> None:
    cases = build_cases(b"fixed-private-test-seed-0123456789")[:1]
    records = []
    create, _, _ = _factory(tmp_path, cases, answer_mode="always_guess")
    summary = run_suite(cases, controller_factory=create, timeout_s=5,
                        on_record=records.append)
    assert summary["passed"] == 0
    check = next(record for record in records if record["record_type"] == "quality_case")
    assert check["checks"]["deletion_cascaded"]
    assert not check["checks"]["after_delete_unknown"]


def test_recall_limit_cannot_claim_a_distractor_check_it_did_not_make(tmp_path) -> None:
    cases = build_cases(b"fixed-private-test-seed-0123456789")[:1]
    records = []
    create, _, _ = _factory(tmp_path, cases, max_matches=1)
    summary = run_suite(cases, controller_factory=create, timeout_s=5,
                        on_record=records.append)
    assert summary["passed"] == 0
    check = next(record for record in records if record["record_type"] == "quality_case")
    assert not check["checks"]["source_and_distractor_retrieved"]
    assert not check["checks"]["passed"]


def test_assessment_refuses_missing_source_lineage_even_with_exact_answer() -> None:
    case = build_cases(b"fixed-private-test-seed-0123456789")[0]
    source = MemoryEvent("source", "s1", "r1", 1, 1, "user_observation",
                         {"text": case.source_prompt}, "user", "2026-10-06T00:00:00+00:00")
    distractor = MemoryEvent("distractor", "s1", "r2", 1, 1, "user_observation",
                             {"text": case.distractor_prompt}, "user", "2026-10-06T00:00:01+00:00")
    recall = {"session_id": "s2", "trace_ok": True, "answer": case.reference,
              "final_derived_from": ["distractor"]}
    after = {"trace_ok": True, "answer": "UNKNOWN",
             "final_derived_from": ["distractor"]}
    checks = assess_case(
        case, source=source, distractor=distractor, recall=recall,
        after_delete=after, before_ids=["source", "distractor"],
        deleted_count=2, source_absent=True, distractor_retained=True,
        derived_final_absent=True, answer_index_absent=True,
        after_ids=["distractor"],
    )
    assert checks["reference_answer_exact"]
    assert not checks["source_and_distractor_retrieved"]
    assert not checks["passed"]


@pytest.fixture
def _paired_memory_profile(tmp_path):
    """Real encrypted seed/readback/render; model execution is a test double."""
    contract = AgentOutputContract("strict_outer_json_fence_agent_v1")
    base = "strata:" + "a" * 64
    consumer = contract.consumer_identity(base)
    artifact = "strata-agent:" + consumer["identity_sha256"]
    route = ProfileRoute(
        "memory-fixture-unit", "unit-model", artifact, "unit-revision", "b" * 64,
        "unit-precision", "external.strata.text.v1", "cpu+cuda:0",
        {"model_output_consumer_identity": consumer},
    )
    native_route = Route(
        route.route_id, artifact, route.model_id, route.backend, frozenset({"text"}),
        route.expected_placement, {"host_ram": 1}, model_output_contract=contract,
        base_artifact_id=base,
    )
    controller = AgentController(
        routes=[native_route], qualifications=[], backends={route.route_id: _MemoryBackend([])},
        memory=EncryptedMemoryStore(tmp_path / "fixture.sqlite", AesGcmCipher(b"m" * 32)),
        tools=WindowsToolBoundary(browser=_NoBrowser()), admit=lambda _: Admission(True, "unit"),
        environment_fingerprint="unit", power_condition="AC", qualification_suite_id="unit",
        bootstrap_route_id=route.route_id,
    )
    bridge = NativeProfileBridge(
        native_config={}, config_root=tmp_path, private_root=tmp_path,
        fixture_origin="http://127.0.0.1:1", telemetry=SimpleNamespace(sample=lambda: {}),
    )
    bridge.controller, bridge.route = controller, route
    bridge._memory_path = controller.memory._path
    case = build_paired_cases(bridge.fixture_origin, mouse_speed=10)["memory"]["short"][0]
    assert case.language == "en-US" and classify_task(case.prompt) == "memory"
    yield SimpleNamespace(controller=controller, bridge=bridge, case=case, route=route, contract=contract)
    controller.close()


def _actual_memory_prompt(fixture, *, mutation=None):
    from vllm_omni.edge.agent.controller import _RECALL_KINDS, _recall_payload

    current = fixture.controller.memory.append_event(
        session_id=fixture.controller.session_id, request_id="unit-request", epoch=1,
        sequence=1, kind="user_observation", payload={"text": fixture.case.prompt}, source="user",
    )
    matches = fixture.controller.memory.search(
        fixture.case.prompt, limit=5, kinds=_RECALL_KINDS,
        exclude_event_ids=frozenset({current.event_id}),
    )
    recalled = [{"source": match.event.source, "kind": match.event.kind,
                 "event_id": match.event.event_id,
                 "text": json.dumps(_recall_payload(match.event.payload), ensure_ascii=False)[:1200]}
                for match in matches]
    if mutation == "no_recall":
        recalled = []
    elif mutation == "current_user_only":
        recalled = [{"source": current.source, "kind": current.kind, "event_id": current.event_id,
                     "text": json.dumps(current.payload, ensure_ascii=False)}]
    elif mutation in {"event_id", "source", "kind", "text"}:
        recalled[0][mutation] = "tampered-unit-value"
    return fixture.controller._build_prompt(
        fixture.case.prompt, "memory", recalled, [], output_contract=fixture.contract,
    )


async def _captured_memory_result(fixture, setup, *, mutation=None):
    class InputOnlyModelDouble:
        async def generate(self, prompt, **kwargs):
            yield BackendChunk("unit-model-double", terminal=True)

    capture = _PromptIdentityBackend(
        InputOnlyModelDouble(), capture_policy=model_step_capture_policy(
            fixture.contract, fixture.controller.limits.max_model_steps,
        ),
    )
    prompt = _actual_memory_prompt(fixture, mutation=mutation)
    async for _ in capture.generate(prompt, request_id="unit-request-step-0", max_tokens=8):
        pass
    records, policy = capture.snapshot_and_reset()
    # Existing consumer-trace tests verify owned terminals. This test isolates
    # fixture provenance using real captured input, not a neural/terminal claim.
    context = copy.deepcopy(setup)
    context["profile_request"] = {
        "run_id": "unit", "route_id": fixture.route.route_id, "case_id": fixture.case.case_id,
        "phase": setup["phase"], "repetition": setup["repetition"],
    }
    return AgentRunResult(
        fixture.case.reference, True, fixture.route.model_id, fixture.route.artifact_id,
        None, fixture.route.backend,
        placement_evidence={
            "first_model_prompt_identity": {key: value for key, value in records[0].items()
                                             if key != "model_request_id"},
            "model_prompt_step_count": len(records), "model_step_identities": records,
            "model_step_identity_policy": policy,
        },
        fixture_setup=context,
    )


@pytest.mark.asyncio
async def test_paired_memory_gate_links_real_seed_to_captured_actual_input(_paired_memory_profile):
    fixture = _paired_memory_profile
    setup = fixture.bridge.before_request(fixture.route, fixture.case, "measured", 0)
    event = fixture.controller.memory.get_event(setup["seed_event_id"])
    assert event is not None and event.source == setup["seed_source"]
    result = await _captured_memory_result(fixture, setup)
    evaluated = evaluate_case(fixture.case, result)
    assert evaluated.success and evaluated.details["memory_provenance_verified"]
    assert fixture.case.reference not in fixture.case.prompt
    assert fixture.case.reference not in json.dumps(setup)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["no_recall", "current_user_only", "event_id", "source", "kind", "text"])
async def test_paired_memory_correct_answer_cannot_replace_actual_seed_recall(_paired_memory_profile, mutation):
    fixture = _paired_memory_profile
    setup = fixture.bridge.before_request(fixture.route, fixture.case, "measured", 0)
    result = await _captured_memory_result(fixture, setup, mutation=mutation)
    evaluated = evaluate_case(fixture.case, result)
    assert evaluated.details["answer_exact"]
    assert not evaluated.success and not evaluated.details["source_verified"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", [
    "missing_setup", "seed_event", "seed_source", "nested_event", "nested_source", "nested_kind",
    "input_contract", "case_id", "reference_identity", "missing_capture", "incomplete_trace", "tool",
    "phase", "repetition", "route_id",
])
async def test_paired_memory_gate_refuses_missing_or_tampered_provenance(_paired_memory_profile, mutation):
    fixture = _paired_memory_profile
    setup = fixture.bridge.before_request(fixture.route, fixture.case, "measured", 0)
    result = await _captured_memory_result(fixture, setup)
    changed = copy.deepcopy(result.fixture_setup)
    if mutation == "missing_setup":
        changed = {}
    elif mutation == "seed_event":
        changed["seed_event_id"] = "wrong-event"
    elif mutation == "seed_source":
        changed["seed_source"] = "user"
    elif mutation in {"nested_event", "nested_source", "nested_kind"}:
        key = {"nested_event": "seed_event_id", "nested_source": "seed_source", "nested_kind": "seed_kind"}[mutation]
        changed["memory_provenance"][key] = "wrong-unit-value"
    elif mutation == "input_contract":
        changed["input_contract"]["instruction_sha256"] = "0" * 64
    elif mutation == "case_id":
        changed["memory_provenance"]["case_id"] = "other-case"
    elif mutation == "reference_identity":
        changed["memory_provenance"]["reference_sha256"] = "0" * 64
    elif mutation == "phase":
        changed["phase"] = "warmup"
    elif mutation == "repetition":
        changed["repetition"] = 10
    elif mutation == "route_id":
        changed["profile_request"]["route_id"] = "other-route"
    elif mutation == "missing_capture":
        result = replace(result, placement_evidence={})
    elif mutation == "incomplete_trace":
        result = replace(result, complete_agent_trace=False)
    elif mutation == "tool":
        result = replace(result, tool_decisions=({"kind": "tool_result", "payload": {}},))
    evaluated = evaluate_case(fixture.case, replace(result, fixture_setup=changed))
    assert not evaluated.success and not evaluated.details["source_verified"]


@pytest.mark.asyncio
async def test_paired_memory_gate_rejects_previous_attempt_seed(_paired_memory_profile):
    fixture = _paired_memory_profile
    old = fixture.bridge.before_request(fixture.route, fixture.case, "measured", 0)
    new = fixture.bridge.before_request(fixture.route, fixture.case, "measured", 0)
    assert old["seed_event_id"] != new["seed_event_id"]
    assert fixture.controller.memory.get_event(old["seed_event_id"]) is None
    result = await _captured_memory_result(fixture, new)
    assert evaluate_case(fixture.case, result).success
    stale = copy.deepcopy(result.fixture_setup)
    stale.update(old)
    assert not evaluate_case(fixture.case, replace(result, fixture_setup=stale)).success


@pytest.mark.asyncio
async def test_paired_memory_gate_rejects_reference_in_current_task(_paired_memory_profile):
    fixture = _paired_memory_profile
    setup = fixture.bridge.before_request(fixture.route, fixture.case, "measured", 0)
    result = await _captured_memory_result(fixture, setup)
    leaked = replace(fixture.case, prompt=fixture.case.prompt + " " + fixture.case.reference)
    assert not evaluate_case(leaked, result).success


@pytest.mark.parametrize("field", ["event_id", "source", "kind", "payload"])
def test_paired_memory_setup_requires_actual_committed_seed_readback(_paired_memory_profile, monkeypatch, field):
    fixture = _paired_memory_profile
    read = fixture.controller.memory.get_event

    def corrupted(event_id):
        event = read(event_id)
        value = {"text": "wrong-unit-value"} if field == "payload" else "wrong-unit-value"
        return replace(event, **{field: value})

    monkeypatch.setattr(fixture.controller.memory, "get_event", corrupted)
    with pytest.raises(ValueError, match="committed memory seed"):
        fixture.bridge.before_request(fixture.route, fixture.case, "measured", 0)


@pytest.mark.asyncio
async def test_profiler_supplies_setup_before_evaluation_and_ignores_runner_claim(_paired_memory_profile):
    fixture = _paired_memory_profile
    setup = fixture.bridge.before_request(fixture.route, fixture.case, "measured", 0)
    actual = await _captured_memory_result(fixture, setup)

    async def runner(route, case, emit):
        return replace(actual, fixture_setup={"runner_claim": True})

    row = await _one_request(
        run_id="unit", route=fixture.route, case=fixture.case, phase="measured", repetition=0,
        runner=runner, evaluator=evaluate_case, telemetry=None, telemetry_interval_s=.1,
        fixture_setup=setup,
    )
    assert row["evaluation"]["success"]
    assert row["result"]["fixture_setup"] == actual.fixture_setup
    row_without_setup = await _one_request(
        run_id="unit", route=fixture.route, case=fixture.case, phase="measured", repetition=0,
        runner=runner, evaluator=evaluate_case, telemetry=None, telemetry_interval_s=.1,
    )
    assert not row_without_setup["evaluation"]["success"]


@pytest.mark.asyncio
async def test_profiler_fixture_snapshot_cannot_be_mutated_by_runner(_paired_memory_profile):
    fixture = _paired_memory_profile
    setup = fixture.bridge.before_request(fixture.route, fixture.case, "measured", 0)
    actual = await _captured_memory_result(fixture, setup)

    async def runner(route, case, emit):
        setup["memory_provenance"]["seed_source"] = "runner-mutated-source"
        return replace(actual, fixture_setup=setup)

    row = await _one_request(
        run_id="unit", route=fixture.route, case=fixture.case, phase="measured", repetition=0,
        runner=runner, evaluator=evaluate_case, telemetry=None, telemetry_interval_s=.1,
        fixture_setup=setup,
    )
    assert row["evaluation"]["success"]
    assert row["result"]["fixture_setup"] == actual.fixture_setup


@pytest.mark.parametrize("invalid", ["task_bound", "reference_leak", "workspace"])
def test_paired_memory_setup_refuses_unadmitted_or_out_of_scope_inputs(_paired_memory_profile, invalid):
    fixture = _paired_memory_profile
    case, route = fixture.case, fixture.route
    if invalid == "task_bound":
        case = replace(case, prompt=case.prompt + "x" * 8193)
    elif invalid == "reference_leak":
        case = replace(case, prompt=case.prompt + " " + case.reference)
    else:
        declared = model_step_capture_policy(fixture.contract, fixture.controller.limits.max_model_steps)
        contract = replace(
            fixture.contract,
            workspace_budget_bytes=fixture.contract.minimum_workspace_bytes
            + declared["declared_metadata_bytes"] + (128 << 10) - 1,
        )
        base = "strata:" + "a" * 64
        consumer = contract.consumer_identity(base)
        artifact = "strata-agent:" + consumer["identity_sha256"]
        route = replace(route, artifact_id=artifact,
                        backend_identity={"model_output_consumer_identity": consumer})
        fixture.controller.routes = [replace(fixture.controller.routes[0], artifact_id=artifact,
                                             model_output_contract=contract)]
        fixture.bridge.route = route
    with pytest.raises(ValueError):
        fixture.bridge.before_request(route, case, "measured", 0)


@pytest.mark.parametrize("task_class", ["basic", "browser_text"])
def test_memory_provenance_gate_does_not_change_other_evaluation(task_class):
    case = AgentCase("basic-unit", "basic", "en-US", "short", "What is six times seven?", "42")
    result = AgentRunResult("42", True, "model", "artifact", "cpu", "backend")
    if task_class == "browser_text":
        source = "http://127.0.0.1:1/unit"
        case = replace(case, task_class=task_class,
                       metadata={"source": source, "required_operations": ("browser_open", "browser_read")})
        result = replace(result, tool_decisions=(
            {"kind": "tool_proposed", "payload": {"operation": "browser_open", "arguments": {"url": source}}},
            {"kind": "tool_result", "payload": {"operation": "browser_open", "source": source}},
            {"kind": "tool_result", "payload": {"operation": "browser_read", "source": source}},
        ))
    before = evaluate_case(case, result)
    after = evaluate_case(case, replace(result, fixture_setup={"unrelated": True}))
    assert before == after and before.success


async def _offline_memory_row(fixture):
    """Real seed/capture and parser, synthetic native metadata; no model run."""
    from tests.edge.test_agent_consumer_qualification import _row
    from vllm_omni.edge.agent import consumer_trace as trace
    from vllm_omni.edge.agent.controller import _permitted_tools
    from vllm_omni.edge.agent.model_output import AgentOutputBuffer

    row, route, events, _ = _row(outer="unit-request")
    fixture.route = replace(
        fixture.route, route_id=route["route_id"], model_id=route["model_id"],
        artifact_id=route["artifact_id"], backend_identity=route["backend_identity"],
    )
    fixture.controller.routes = [replace(
        fixture.controller.routes[0], route_id=fixture.route.route_id, model=fixture.route.model_id,
    )]
    fixture.bridge.route = fixture.route
    setup = fixture.bridge.before_request(fixture.route, fixture.case, "measured", 0)
    actual = await _captured_memory_result(fixture, setup)
    actual.fixture_setup["profile_request"]["run_id"] = "fixture-run"
    evidence = row["result"]["placement_evidence"]
    evidence.update(actual.placement_evidence)
    terminal = evidence["terminal_model_metrics"][0]["metrics"]
    raw = json.dumps({"final": fixture.case.reference}, ensure_ascii=False)
    terminal["raw_model_output_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    stage = terminal["stage_event"]
    buffer = AgentOutputBuffer(
        fixture.contract, request_id=stage["request_id"],
        worker_generation=stage["worker_generation"], stage_id=stage["stage_id"],
        permitted_tools=_permitted_tools("memory", fixture.case.prompt), previous_epoch=0,
    )
    buffer.append(raw)
    buffer.terminal(terminal)
    _, _, proof = buffer.finish()
    proof.update(
        step=0, model_request_id=stage["request_id"],
        input_sha256=evidence["model_step_identities"][0]["sha256"],
        consumer_identity_sha256=route["backend_identity"]["model_output_consumer_identity"]["identity_sha256"],
    )
    for event in events:
        event["session_id"] = fixture.controller.session_id
        if event["kind"] == "user_observation":
            event["payload"] = {"text": fixture.case.prompt}
        elif event["kind"] == "model_output_contract":
            event["payload"] = proof
        elif event["kind"] == "final":
            event["payload"]["answer"] = fixture.case.reference
    visible = trace.validated_consumer_final(events[-1], route, events)
    assert visible is not None
    for record in row["events"]:
        if record["kind"] == "assistant_final":
            record["payload"] = visible
        elif record["kind"] == "model_prompt_identity":
            record["payload"] = {
                "first": copy.deepcopy(evidence["first_model_prompt_identity"]),
                "model_steps": evidence["model_prompt_step_count"],
                "steps": copy.deepcopy(evidence["model_step_identities"]),
                "policy": copy.deepcopy(evidence["model_step_identity_policy"]),
            }
    complete = replace(actual, placement_evidence=evidence)
    row.update(
        case=asdict(fixture.case), result=asdict(complete), fixture_setup=setup,
        evaluation=asdict(evaluate_case(fixture.case, complete)),
    )
    return row, asdict(fixture.route)


def _isolate_offline_native_placement(monkeypatch):
    # Same native/load isolation as the existing raw-consumer fixture. The
    # memory gate, parser, trace, timing, raw hash and reviewer are unchanged.
    from vllm_omni.edge.agent import consumer_trace as trace
    from vllm_omni.edge.agent import qualification as q

    monkeypatch.setattr(trace, "validate_strata_request_evidence", lambda *_a, **_k: None)
    monkeypatch.setattr(q, "preparation_placement_matches", lambda *_a, **_k: True)
    monkeypatch.setattr(q, "result_placement_matches", lambda *_a, **_k: True)


@pytest.mark.asyncio
async def test_offline_raw_audit_accepts_the_live_memory_provenance_evaluation(
    _paired_memory_profile, tmp_path, monkeypatch,
):
    from tests.edge.test_agent_consumer_qualification import _summary
    from vllm_omni.edge.agent import qualification as q

    _isolate_offline_native_placement(monkeypatch)
    row, route = await _offline_memory_row(_paired_memory_profile)
    assert row["evaluation"] == q._evaluate_case(row["case"], row["result"])
    assert row["evaluation"]["success"]
    audit = q.audit_summary(_summary(tmp_path, route, [row]))
    assert audit.internally_valid, audit.errors
    assert audit.trace_verified and audit.measured_successes == 1
    assert not audit.protocol_compliant  # one model-free row cannot qualify a route


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["missing_provenance", "seed_source", "input_identity"])
async def test_offline_raw_audit_refuses_memory_pass_without_the_bound_seed(
    _paired_memory_profile, tmp_path, monkeypatch, mutation,
):
    from tests.edge.test_agent_consumer_qualification import _summary
    from vllm_omni.edge.agent import qualification as q

    _isolate_offline_native_placement(monkeypatch)
    row, route = await _offline_memory_row(_paired_memory_profile)
    setup = row["result"]["fixture_setup"]
    if mutation == "missing_provenance":
        setup.pop("memory_provenance")
    elif mutation == "seed_source":
        setup["memory_provenance"]["seed_source"] = "untrusted-observation"
    else:
        setup["memory_provenance"]["expected_first_model_input"]["sha256"] = "0" * 64
    expected = q._evaluate_case(row["case"], row["result"])
    assert expected["details"]["answer_exact"]
    assert not expected["success"] and not expected["details"]["memory_provenance_verified"]
    # Recompute the raw/summary hashes and counts: rejection must come from
    # the claimed passing evaluation, not an unrelated stale file digest.
    audit = q.audit_summary(_summary(tmp_path, route, [row]))
    assert not audit.internally_valid
    assert any("evaluation differs" in error for error in audit.errors)
    row["evaluation"] = expected
    failure = q.audit_summary(_summary(tmp_path, route, [row]))
    assert failure.internally_valid, failure.errors
    assert failure.trace_verified and failure.measured_attempts == 1
    assert failure.measured_successes == 0  # truthful failure remains evidence
