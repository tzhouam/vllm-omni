# SPDX-License-Identifier: Apache-2.0
"""Qualification reconstructs explicit consumer traces, never release claims.

Only the existing model placement/load gate is isolated. The shared trace
state machine, actual output-parser proofs, visibility reconstruction and raw
profile reviewer execute normally; these fixtures do not run a model.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from tests.edge.test_agent_consumer_trace import scenario
from vllm_omni.edge.agent import consumer_trace as trace
from vllm_omni.edge.agent import qualification as q


def _input_contract(case):
    encoded = case["prompt"].encode("utf-8")
    value = {
        "submission_mode": "model_selected_tools_v1",
        "case_id": case["case_id"],
        "instruction_sha256": hashlib.sha256(encoded).hexdigest(),
        "instruction_utf8_bytes": len(encoded),
        "explicit_read_url": None,
    }
    value["contract_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def _row(*, outer="outer", outer_epoch=1, native_epoch=11, native_seq=7):
    events, route, evidence, trusted = scenario()
    for event in events:
        event["request_id"] = outer
        event["epoch"] = outer_epoch
    record = evidence["model_step_identities"][0]
    record["model_request_id"] = outer + "-step-0"
    terminal = evidence["terminal_model_metrics"][0]["metrics"]
    terminal["stage_event"].update(request_id=record["model_request_id"], epoch=native_epoch)
    io = terminal["backend_metrics"]["runtime_telemetry"]["native_io_observation"]
    io.update(request_id=record["model_request_id"], epoch=native_epoch, native_request_seq=native_seq)
    proof = next(event["payload"] for event in events if event["kind"] == "model_output_contract")
    proof["model_request_id"] = record["model_request_id"]
    proof["stage_event"].update(request_id=record["model_request_id"], epoch=native_epoch)
    evidence["execution_plan"]["observation_runtime"] = route["backend_identity"]["observation_runtime"]
    evidence["loaded_plan_sha256"] = "c" * 64
    case = {
        "case_id": "consumer-final-short",
        "task_class": "basic",
        "length": "short",
        "prompt": trusted["text"],
        "reference": "READY",
        "metadata": {"kind": "basic", "source": "self-contained", "required_operations": []},
    }
    captured = {
        "first": copy.deepcopy(evidence["first_model_prompt_identity"]),
        "model_steps": len(evidence["model_step_identities"]),
        "steps": copy.deepcopy(evidence["model_step_identities"]),
        "policy": copy.deepcopy(evidence["model_step_identity_policy"]),
    }
    records = [{"kind": "agent_event", "offset_s": index / 100, "payload": event} for index, event in enumerate(events)]
    visible = trace.validated_consumer_final(events[-1], route, events)
    assert visible is not None
    records.extend(
        [
            {"kind": "assistant_final", "offset_s": 0.1, "payload": visible},
            {"kind": "model_prompt_identity", "offset_s": 0.11, "payload": captured},
        ]
    )
    result = {
        "final_answer": "READY",
        "model_id": route["model_id"],
        "artifact_id": route["artifact_id"],
        "backend": route["backend"],
        "actual_placement": None,
        "trace_scope": "agent_e2e",
        "complete_agent_trace": True,
        "tool_decisions": [],
        "placement_evidence": evidence,
    }
    row = {
        "record_type": "request",
        "run_id": "fixture-run",
        "route_id": route["route_id"],
        "phase": "measured",
        "repetition": 0,
        "case": case,
        "batch_size": 1,
        "concurrency": 1,
        "events": records,
        "result": result,
        "evaluation": q._evaluate_case(case, result),
        "e2e_complete": True,
        "placement_matches": True,
        "error": None,
        "ttft_s": 0.1,
        "answer_latency_s": 0.2,
        "first_visible_event_kind": "assistant_final",
        "first_visible_output_scope": "validated_final_full_response_not_hidden_model_delta_or_Qt_render",
        "ttft_scope": "time_to_validated_final_visibility",
        "fixture_setup": {"private_memory_reset": True, "input_contract": _input_contract(case)},
        "telemetry": {"raw_samples": [], "errors": [], "summary": q._telemetry_summary([])},
    }
    return row, route, events, trusted


@pytest.fixture
def engine_gates(monkeypatch):
    monkeypatch.setattr(trace, "validate_strata_request_evidence", lambda *_a, **_k: None)
    monkeypatch.setattr(q, "preparation_placement_matches", lambda *_a, **_k: True)
    monkeypatch.setattr(q, "result_placement_matches", lambda *_a, **_k: True)


def _summary(tmp_path, route, requests):
    config = {"warmups_per_length": 1, "measured_per_length": 20, "endurance_seconds": 1800}
    conditions = {"suite_id": q.SUITE_ID}
    rows = [
        {
            "record_type": "manifest",
            "scope": "complete_agent_request",
            "batch_size": 1,
            "concurrency": 1,
            "conditions": conditions,
            "run_id": "fixture-run",
            "routes": [route],
            "config": config,
        },
        {
            "record_type": "route_prepare",
            "route_id": route["route_id"],
            "cold_start_s": 1.0,
            "preparation": {"details": {"loaded_plan_sha256": "c" * 64}},
        },
        *requests,
    ]
    raw = "".join(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n" for row in rows).encode()
    (tmp_path / "samples.jsonl").write_bytes(raw)
    profile = q._route_profile(
        route, requests, cold_start_s=1.0, sustained_seconds=0, endurance_wall_seconds=0, config=config
    )
    summary = {
        "raw_jsonl": "samples.jsonl",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "conditions": conditions,
        "run_id": "fixture-run",
        "routes": {route["route_id"]: profile},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return path


def test_raw_reviewer_uses_actual_shared_trace_and_final_visibility(tmp_path, engine_gates):
    row, route, _, _ = _row()
    audit = q.audit_summary(_summary(tmp_path, route, [row]))
    assert audit.internally_valid, audit.errors
    assert audit.trace_verified
    assert audit.measured_attempts == audit.measured_successes == 1
    assert not audit.protocol_compliant  # one synthetic row never meets release protocol


def test_raw_reviewer_accepts_serial_requests_with_fresh_outer_and_native_states(tmp_path, engine_gates):
    first, route, _, _ = _row()
    second, _, _, _ = _row(outer="next", outer_epoch=2, native_epoch=12, native_seq=8)
    audit = q.audit_summary(_summary(tmp_path, route, [first, second]))
    assert audit.internally_valid, audit.errors
    assert audit.trace_verified and audit.measured_attempts == 2


def test_consumer_dispatch_requires_external_trusted_task(engine_gates):
    row, route, events, trusted = _row()
    evidence = row["result"]["placement_evidence"]
    assert not q._trace_complete(events, "READY", route, evidence)
    assert q._trace_complete(events, "READY", route, evidence, trusted_task_binding=trusted)
    assert not q._trace_complete(
        events, "READY", route, evidence, trusted_task_binding={**trusted, "text": "untrusted alternate task"}
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["fixture_setup"]["input_contract"].update(instruction_sha256="f" * 64),
        lambda row: row["events"][0]["payload"]["payload"].update(text="page-issued instruction"),
        lambda row: row["events"][0]["payload"]["payload"].update(read_url="http://127.0.0.1:1/"),
        lambda row: row["events"].append(copy.deepcopy(row["events"][0])),
    ],
)
def test_trusted_case_input_cannot_be_replaced_by_observed_data(mutation):
    row, _, _, trusted = _row()
    assert q._consumer_task_binding(row, structured_origin=None) == trusted
    mutation(row)
    with pytest.raises(ValueError):
        q._consumer_task_binding(row, structured_origin=None)


def test_structured_task_reconstructed_from_exact_fixed_case():
    origin = "http://127.0.0.1:12345"
    case = q.canonical_fixed_cases(origin, 10)["browser_text"]["short"][0]
    observation = {"text": case["prompt"], "read_url": case["metadata"]["source"], "mode": "read_url"}
    row = {
        "case": case,
        "fixture_setup": {"input_contract": q._structured_contract(case)},
        "events": [{"kind": "agent_event", "payload": {"kind": "user_observation", "payload": observation}}],
    }
    assert q._consumer_task_binding(row, structured_origin=origin) == observation
    row["fixture_setup"]["input_contract"]["explicit_read_url"] += "elsewhere"
    with pytest.raises(ValueError):
        q._consumer_task_binding(row, structured_origin=origin)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["events"][-1]["payload"].pop("steps"),
        lambda row: row["events"][-1]["payload"]["steps"][0].update(model_request_id="different-step-0"),
        lambda row: row["events"][-1]["payload"]["policy"].update(declared_metadata_bytes=0),
        lambda row: row["events"][-1]["payload"]["first"].update(model_request_id="outer-step-0"),
        lambda row: row["events"][-1]["payload"]["first"].update(step=False),
        lambda row: row["events"][-1]["payload"]["steps"][0].update(step=False),
        lambda row: row["events"][-1]["payload"]["steps"][0].update(step=0.0),
        lambda row: row["events"][-1]["payload"]["policy"].update(max_record_bytes=16384.0),
        lambda row: row["result"]["placement_evidence"]["first_model_prompt_identity"].update(step=False),
        lambda row: row["result"]["placement_evidence"].update(model_prompt_step_count=True),
    ],
)
def test_separately_emitted_capture_must_equal_all_placement_copies(mutation):
    row, _, _, _ = _row()
    evidence = row["result"]["placement_evidence"]
    q._consumer_capture_valid(row, evidence)
    mutation(row)
    with pytest.raises(ValueError):
        q._consumer_capture_valid(row, evidence)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["events"].insert(-1, copy.deepcopy(row["events"][-2])),
        lambda row: row["events"][-2]["payload"].update(text="different"),
        lambda row: row["events"][-2].update(kind="assistant_text_delta"),
        lambda row: row["events"][-2].update(offset_s=float("nan")),
        lambda row: row["events"][-2].update(offset_s=True),
        lambda row: row["events"][-2].update(offset_s=-0.1),
        lambda row: row["events"][-2].update(offset_s=1.0),
        lambda row: row.update(answer_latency_s=float("inf")),
        lambda row: row.update(ttft_scope="hidden_model_sse"),
        lambda row: row["events"].pop(-2),
    ],
)
def test_visible_final_must_be_unique_owned_finite_and_ordered(mutation):
    row, route, events, _ = _row()
    assert q._consumer_visible_time(row, route, events) == 0.1
    mutation(row)
    with pytest.raises(ValueError):
        q._consumer_visible_time(row, route, events)


def test_visibility_cannot_precede_owned_final_even_with_equal_offsets():
    row, route, events, _ = _row()
    visible = row["events"].pop(-2)
    final_index = len(events) - 1
    visible["offset_s"] = row["events"][final_index]["offset_s"]
    row["events"].insert(final_index, visible)
    with pytest.raises(ValueError, match="precedes"):
        q._consumer_visible_time(row, route, events)


@pytest.mark.parametrize("epoch,native_seq", [(11, 8), (10, 8), (12, 7), (12, 6)])
def test_cross_request_native_epoch_and_sequence_cannot_replay_or_reverse(epoch, native_seq):
    first, _, _, _ = _row()
    second, _, _, _ = _row(outer="next", outer_epoch=2, native_epoch=epoch, native_seq=native_seq)
    state = {}
    q._consumer_state_order(first["result"]["placement_evidence"], state)
    with pytest.raises(ValueError, match="replayed or reversed"):
        q._consumer_state_order(second["result"]["placement_evidence"], state)


def test_state_order_can_reset_only_for_new_worker_generation():
    row, _, _, _ = _row()
    state = {}
    evidence = row["result"]["placement_evidence"]
    q._consumer_state_order(evidence, state)
    evidence = copy.deepcopy(evidence)
    metrics = evidence["terminal_model_metrics"][0]["metrics"]
    metrics["stage_event"].update(worker_generation="fresh", epoch=1)
    metrics["backend_metrics"]["runtime_telemetry"]["native_io_observation"].update(
        generation="fresh",
        epoch=1,
        native_request_seq=1,
    )
    q._consumer_state_order(evidence, state)
    assert state[(0, "fresh")] == (1, 1)


@pytest.mark.parametrize(
    "change", ["outer_replay", "outer_epoch_reverse", "native_epoch_reverse", "native_seq_reverse"]
)
def test_full_raw_audit_refuses_cross_request_identity_mutations(tmp_path, engine_gates, change):
    first, route, _, _ = _row()
    second, _, _, _ = _row(outer="next", outer_epoch=2, native_epoch=12, native_seq=8)
    if change == "outer_replay":
        second, _, _, _ = _row(outer="outer", outer_epoch=2, native_epoch=12, native_seq=8)
    elif change == "outer_epoch_reverse":
        second, _, _, _ = _row(outer="next", outer_epoch=1, native_epoch=12, native_seq=8)
    elif change == "native_epoch_reverse":
        second, _, _, _ = _row(outer="next", outer_epoch=2, native_epoch=10, native_seq=8)
    else:
        second, _, _, _ = _row(outer="next", outer_epoch=2, native_epoch=12, native_seq=6)
    audit = q.audit_summary(_summary(tmp_path, route, [first, second]))
    assert not audit.trace_verified
    assert any("cross-request state" in error for error in audit.errors)


def test_partial_consumer_marker_never_falls_through_to_legacy():
    events = [
        {"seq": 1, "request_id": "r", "epoch": 1, "kind": "user_observation", "payload": {"text": "x"}},
        {
            "seq": 2,
            "request_id": "r",
            "epoch": 1,
            "kind": "route",
            "payload": {
                "route_id": "legacy",
                "model": "model",
                "artifact_id": "artifact",
                "backend": "external.llamacpp.text.v1",
                "actual_placement": "cpu",
            },
        },
        {"seq": 3, "request_id": "r", "epoch": 1, "kind": "model_metrics", "payload": {}},
        {"seq": 4, "request_id": "r", "epoch": 1, "kind": "final", "payload": {"answer": "x"}},
    ]
    route = {
        "route_id": "legacy",
        "model_id": "model",
        "artifact_id": "artifact",
        "backend": "external.llamacpp.text.v1",
        "expected_placement": "cpu",
    }
    assert q._trace_complete(events, "x", route)
    route["backend_identity"] = {"base_engine_artifact_id": "strata:" + "a" * 64}
    assert not q._trace_complete(events, "x", route)
