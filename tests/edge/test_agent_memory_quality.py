"""Cross-session quality harness tests without native inference or a GPU."""

from __future__ import annotations

import json

from benchmarks.edge_agent.experiments.native_memory_quality import (
    SUITE_ID, assess_case, build_cases, run_suite,
)
from vllm_omni.edge.agent.controller import AgentController, AgentLimits
from vllm_omni.edge.agent.memory import AesGcmCipher, EncryptedMemoryStore, MemoryEvent
from vllm_omni.edge.agent.omni_backend import BackendChunk
from vllm_omni.edge.agent.router import Admission, Route
from vllm_omni.edge.agent.tools import WindowsToolBoundary


class _NoBrowser:
    def close(self) -> None:
        pass


class _MemoryBackend:
    def __init__(self, cases, *, answer_mode="retrieve") -> None:
        self.cases = cases
        self.answer_mode = answer_mode
        self.execution_plan = {"requested_device": "cpu"}
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
