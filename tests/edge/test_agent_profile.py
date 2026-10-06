"""Batch-one whole-Agent evidence and qualification boundary tests."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from benchmarks.edge_agent.paired_suite import ReadOnlyFixtureTools, evaluate_case
from benchmarks.edge_agent.profile import (
    AgentCase,
    AgentRunResult,
    Evaluation,
    ProfileConditions,
    ProfileConfig,
    ProfileRoute,
    Preparation,
    run_profile,
)
from vllm_omni.edge.agent.tools import ToolAction


def _route(name: str) -> ProfileRoute:
    return ProfileRoute(
        route_id=name,
        model_id="model",
        artifact_id="artifact",
        checkpoint_revision="fixed-revision",
        artifact_sha256="a" * 64,
        precision="Q4_K_M",
        backend="llama.cpp",
        expected_placement="CPU",
    )


def _cases(task_class="basic"):
    return {
        bucket: [
            AgentCase(
                case_id=bucket,
                task_class=task_class,
                language="zh-CN" if bucket == "short" else "en-US",
                length=bucket,
                prompt=f"{bucket} paired Agent task",
                reference="done",
                input_tokens={"short": 16, "medium": 128, "long": 512}[bucket],
            )
        ]
        for bucket in ("short", "medium", "long")
    }


def _conditions():
    return ProfileConditions(
        hardware_id="test CPU laptop",
        os_version="test Windows 11",
        driver_versions={"gpu": "test"},
        runtime_versions={"llama.cpp": "test"},
        power_condition="AC performance",
        suite_id="agent-paired-v1",
        environment_fingerprint="test-fingerprint",
    )


def test_paired_suite_refuses_and_marks_browser_post_unsafe():
    action = ToolAction("browser_post", {
        "url": "http://127.0.0.1:1234/submit",
        "body_b64": "e30=",
        "content_type": "application/json",
    })
    tools = ReadOnlyFixtureTools("http://127.0.0.1:1234", browser=object())
    with pytest.raises(PermissionError, match="benchmark forbids"):
        tools.execute(action)
    assert tools._pending == {}

    case = AgentCase(
        case_id="post-proposal", task_class="browser_text", language="en-US",
        length="short", prompt="Read the local fixture", reference="done",
    )
    result = AgentRunResult(
        final_answer="done", complete_agent_trace=True, model_id="model",
        artifact_id="artifact", actual_placement="CPU", backend="llama.cpp",
        tool_decisions=({"kind": "tool_proposed", "payload": {
            "operation": "browser_post", "arguments": dict(action.arguments),
        }},),
    )
    evaluation = evaluate_case(case, result)
    assert evaluation.quality_pass
    assert not evaluation.tool_safe
    assert not evaluation.success


def test_paired_batch_one_raw_agent_evidence(tmp_path):
    active = 0
    maximum_active = 0

    async def runner(route, case, emit):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            await asyncio.sleep(.001)
            emit("assistant_text_delta", "do")
            await asyncio.sleep(.001)
            emit("assistant_text_delta", "ne")
            return AgentRunResult(
                final_answer="done", complete_agent_trace=True,
                model_id="model", artifact_id="artifact",
                actual_placement="CPU", backend="llama.cpp",
                placement_evidence={"backend_log": "CPU"},
                tool_decisions=(),
            )
        finally:
            active -= 1

    def evaluator(case, result):
        return Evaluation(
            success=result.final_answer == case.reference,
            quality_pass=True, quality_score=1.0,
            tool_safe=True, details={"reference": case.reference},
        )

    async def prepare(route):
        await asyncio.sleep(0)
        return Preparation(True, route.artifact_id, route.expected_placement,
                           {"unloaded_before_prepare": True})

    summary = asyncio.run(run_profile(
        routes=(_route("cpu-a"), _route("cpu-b")),
        cases_by_length=_cases(), runner=runner, evaluator=evaluator,
        conditions=_conditions(), output_dir=tmp_path,
        config=ProfileConfig(warmups_per_length=1, measured_per_length=2,
                             endurance_seconds=.006, telemetry_interval_seconds=.001),
        telemetry=lambda: {"ram_used_bytes": 1024, "gpu_power_w": 3.0},
        prepare=prepare,
    ))
    assert maximum_active == 1
    assert summary.raw_sha256 and summary.raw_jsonl.is_file()
    assert summary.routes["cpu-a"].measured_attempts == 6
    assert summary.routes["cpu-b"].measured_successes == 6
    assert summary.routes["cpu-a"].warmups_per_length == {
        "short": 1, "medium": 1, "long": 1,
    }
    assert summary.routes["cpu-a"].answer_p95_s["long"] is not None
    assert not summary.routes["cpu-a"].protocol_compliant
    records = [json.loads(line) for line in summary.raw_jsonl.read_text().splitlines()]
    measured = [record for record in records if record.get("phase") == "measured"]
    assert len(measured) == 12
    assert [row["case"]["case_id"] for row in measured if row["route_id"] == "cpu-a"] == [
        row["case"]["case_id"] for row in measured if row["route_id"] == "cpu-b"
    ]
    assert all(record["ttft_s"] is not None for record in measured)
    assert all(record["case"]["prompt"].endswith("Agent task") for record in measured)
    assert all(record["result"]["final_answer"] == "done" for record in measured)
    assert all(record["telemetry"]["raw_samples"] for record in measured)
    fields = summary.routes["cpu-a"].qualification_fields(
        conditions=_conditions(), raw_evidence=str(summary.raw_jsonl),
        task_class="basic", memory_pass=True,
        cancel_recovery_pass=True, actual_placement_verified=True,
    )
    assert fields["batch_size"] == fields["concurrency"] == 1
    assert not fields["stability_pass"]  # short test never qualifies


def test_stage_only_results_cannot_become_agent_e2e(tmp_path):
    async def runner(route, case, emit):
        emit("visible_token", "stage output")
        return AgentRunResult(
            final_answer="done", complete_agent_trace=False,
            model_id="model", artifact_id="artifact", actual_placement="CPU",
            backend="llama.cpp", trace_scope="component",
        )

    summary = asyncio.run(run_profile(
        routes=(_route("component"),), cases_by_length=_cases(), runner=runner,
        evaluator=lambda case, result: Evaluation(True, True, 1.0, True),
        conditions=_conditions(), output_dir=tmp_path,
        config=ProfileConfig(warmups_per_length=1, measured_per_length=1,
                             endurance_seconds=0),
    ))
    profile = summary.routes["component"]
    assert profile.measured_attempts == 3
    assert profile.measured_successes == 0
    assert not profile.e2e_trace_pass
    assert not profile.protocol_compliant
    fields = profile.qualification_fields(
        conditions=_conditions(), raw_evidence=str(summary.raw_jsonl),
        task_class="basic", memory_pass=True,
        cancel_recovery_pass=True, actual_placement_verified=True,
    )
    assert not fields["correctness_pass"]
    assert not fields["actual_placement_verified"]


def test_refuses_incomplete_or_mixed_suite(tmp_path):
    async def runner(route, case, emit):
        raise AssertionError("validation should happen before the runner")

    with pytest.raises(ValueError, match="exactly three"):
        asyncio.run(run_profile(
            routes=(_route("x"),), cases_by_length={"short": _cases()["short"]},
            runner=runner, evaluator=lambda *_: None,
            conditions=_conditions(), output_dir=tmp_path,
        ))
    mixed = _cases()
    mixed["long"] = _cases("code_tools")["long"]
    with pytest.raises(ValueError, match="one task class"):
        asyncio.run(run_profile(
            routes=(_route("x"),), cases_by_length=mixed,
            runner=runner, evaluator=lambda *_: None,
            conditions=_conditions(), output_dir=tmp_path,
        ))


def test_reference_evaluation_is_outside_answer_timer(tmp_path):
    async def runner(route, case, emit):
        emit("visible_token", "done")
        return AgentRunResult(
            final_answer="done", complete_agent_trace=True,
            model_id="model", artifact_id="artifact",
            actual_placement="CPU", backend="llama.cpp",
            placement_evidence={"backend_log": "CPU"},
        )

    def slow_evaluator(case, result):
        time.sleep(.02)
        return Evaluation(True, True, 1.0, True)

    summary = asyncio.run(run_profile(
        routes=(_route("cpu"),), cases_by_length=_cases(),
        runner=runner, evaluator=slow_evaluator, conditions=_conditions(),
        output_dir=tmp_path,
        config=ProfileConfig(warmups_per_length=0, measured_per_length=1,
                             endurance_seconds=0),
    ))
    records = [json.loads(line) for line in summary.raw_jsonl.read_text().splitlines()]
    measured = [row for row in records if row.get("phase") == "measured"]
    assert len(measured) == 3
    assert max(row["answer_latency_s"] for row in measured) < .02


def test_fixture_setup_runs_before_answer_timer_and_is_recorded(tmp_path):
    calls = []

    def before_request(route, case, phase, repetition):
        calls.append((route.route_id, case.case_id, phase, repetition))
        time.sleep(.025)
        return {"fixture_seeded": True, "case_id": case.case_id}

    async def runner(route, case, emit):
        emit("visible_token", "done")
        return AgentRunResult(
            final_answer="done", complete_agent_trace=True,
            model_id="model", artifact_id="artifact", actual_placement="CPU",
            backend="llama.cpp", placement_evidence={"stage": "CPU"},
        )

    summary = asyncio.run(run_profile(
        routes=(_route("cpu"),), cases_by_length=_cases(), runner=runner,
        evaluator=lambda case, result: Evaluation(True, True, 1.0, True),
        conditions=_conditions(), output_dir=tmp_path,
        before_request=before_request,
        config=ProfileConfig(warmups_per_length=0, measured_per_length=1,
                             endurance_seconds=0),
    ))
    records = [json.loads(line) for line in summary.raw_jsonl.read_text().splitlines()]
    measured = [row for row in records if row.get("phase") == "measured"]
    assert len(calls) == len(measured) == 3
    assert all(row["fixture_setup"]["fixture_seeded"] for row in measured)
    assert max(row["answer_latency_s"] for row in measured) < .025
