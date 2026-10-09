# SPDX-License-Identifier: Apache-2.0
"""Bounded metadata validation for explicitly interpreted Agent output.

These checks establish trace completeness, not quality or release qualification.
Prompt/raw hashes bind source-owned observations; they are not saved preimages.
No inference backend, model, tool, or operating-system operation is started here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from vllm_omni.edge.agent.model_output import AgentOutputContract, validate_output_contract_entry
from vllm_omni.edge.agent.placement import STRATA_BACKEND, validate_strata_request_evidence

_TOP = {"model_output_contract", "model_output_consumer_identity", "base_artifact_id", "model_output_workspace_bytes"}
_STAGE = {"request_id", "worker_generation", "stage_id", "epoch", "seq", "kind", "terminal"}
_STEP = {"step", "model_request_id", "sha256", "utf8_bytes", "chars"}
_PROOF = {
    "step",
    "model_request_id",
    "input_sha256",
    "consumer_identity_sha256",
    "schema",
    "contract_sha256",
    "mode",
    "directive_sha256",
    "raw_output_sha256",
    "canonical_output_sha256",
    "normalization",
    "stage_event",
    "constrained_decoding",
    "extraction_used",
    "retry_count",
}
_POLICY_SCHEMA = "omni-agent-model-step-identities-v1"


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _consumer(route: Mapping[str, Any]) -> tuple[AgentOutputContract, Mapping[str, Any]] | None:
    _require(isinstance(route, Mapping), "consumer route is not a mapping")
    if route.get("backend") == "external.llamacpp.text.v1":
        return _llamacpp_consumer(route)
    nested = route.get("backend_identity")
    if nested is None:
        nested = {}
    _require(isinstance(nested, Mapping), "backend identity is not a mapping")
    _require(
        route.get("base_engine_artifact_id") is None
        and all(
            nested.get(key) is None
            for key in ("base_artifact_id", "model_output_contract", "model_output_workspace_bytes")
        )
        and not (isinstance(nested.get("artifact_id"), str) and nested["artifact_id"].startswith("strata-agent")),
        "consumer identity markers occur in an unsupported representation",
    )
    top_present = {key for key in _TOP if route.get(key) is not None}
    nested_present = {
        key for key in ("model_output_consumer_identity", "base_engine_artifact_id") if nested.get(key) is not None
    }
    marked = isinstance(route.get("artifact_id"), str) and route["artifact_id"].startswith("strata-agent")
    if not top_present and not nested_present and not marked:
        return None
    _require(route.get("backend") == STRATA_BACKEND, "consumer traces require the explicit Strata text backend")
    _require(not top_present or top_present == _TOP, "partial top-level consumer identity")
    _require(not nested_present or len(nested_present) == 2, "partial nested consumer identity")
    consumer = nested.get("model_output_consumer_identity", route.get("model_output_consumer_identity"))
    _require(
        isinstance(consumer, Mapping) and isinstance(consumer.get("contract"), Mapping),
        "consumer artifact has no complete identity",
    )
    contract = AgentOutputContract.from_dict(dict(consumer["contract"]))
    base = consumer.get("base_artifact_id")
    _require(isinstance(base, str) and base.startswith("strata:") and _digest(base[7:]), "invalid base engine identity")
    expected = contract.consumer_identity(base)
    _require(
        dict(consumer) == expected and route.get("artifact_id") == "strata-agent:" + expected["identity_sha256"],
        "consumer artifact/parser identity differs",
    )
    if nested_present:
        _require(
            nested["base_engine_artifact_id"] == base
            and _digest(nested.get("backend_config_sha256"))
            and base == "strata:" + nested["backend_config_sha256"],
            "nested base engine identity differs",
        )
    if top_present:
        _require(
            route["base_artifact_id"] == base
            and route["model_output_contract"] == contract.to_dict()
            and route["model_output_consumer_identity"] == expected
            and type(route["model_output_workspace_bytes"]) is int
            and route["model_output_workspace_bytes"] == contract.workspace_budget_bytes,
            "top-level and nested consumer identities conflict",
        )
        if "backend_config" in route:
            _require(validate_output_contract_entry(dict(route)) == contract, "native consumer binding differs")
    return contract, expected


def _llamacpp_consumer(route: Mapping[str, Any]):
    from vllm_omni.edge.agent.llamacpp_route import (
        llamacpp_route_binding,
        validate_llamacpp_consumer_binding,
    )

    nested = route.get("backend_identity", {})
    _require(isinstance(nested, Mapping), "llama.cpp backend identity is not a mapping")
    top = {key for key in _TOP if route.get(key) is not None}
    nested_marked = any(key in nested for key in (
        "base_engine_artifact_id", "model_output_consumer_identity",
        "consumer_memory_demands", "consumer_workspace_bytes", "consumer_memory_overhead_bytes",
        "base_artifact_id", "model_output_contract", "model_output_workspace_bytes",
    ))
    marked = str(route.get("artifact_id", "")).startswith(("llamacpp-agent", "strata-agent"))
    _require(route.get("base_engine_artifact_id") is None, "unsupported top-level engine identity")
    if not top and not nested_marked and not marked:
        return None
    _require(not top or top == _TOP, "partial llama.cpp consumer identity")
    if top:
        contract = validate_output_contract_entry(dict(route))
        expected_binding = llamacpp_route_binding(route)
        _require(not nested or _encoded(dict(nested)) == _encoded(expected_binding),
                 "native/profile llama.cpp consumer identities conflict")
        nested = expected_binding
    contract, expected = validate_llamacpp_consumer_binding(nested)
    _require(route.get("artifact_id") == "llamacpp-agent:" + expected["identity_sha256"],
             "llama.cpp consumer artifact/parser identity differs")
    return contract, expected


def consumer_trace_requested(route: Mapping[str, Any]) -> bool:
    """Detect explicit consumers, refusing orphan markers instead of falling through."""
    return _consumer(route) is not None


def model_step_capture_policy(
    contract: AgentOutputContract | Mapping[str, Any], max_model_steps: int
) -> dict[str, Any]:
    """Account for two bounded metadata copies within the admitted parser workspace."""
    if not isinstance(contract, AgentOutputContract):
        _require(isinstance(contract, Mapping), "missing model-step consumer contract")
        contract = AgentOutputContract.from_dict(dict(contract))
    _require(type(max_model_steps) is int and max_model_steps > 0, "invalid model-step limit")
    declared = 2 * 16384 * max_model_steps
    _require(
        contract.minimum_workspace_bytes + declared <= contract.workspace_budget_bytes,
        "model-step metadata exceeds admitted consumer workspace",
    )
    return {
        "schema": _POLICY_SCHEMA,
        "max_model_steps": max_model_steps,
        "max_record_bytes": 16384,
        "transient_copies": 2,
        "declared_metadata_bytes": declared,
    }


def _terminal(
    proof: Mapping[str, Any], metrics: Mapping[str, Any], consumer: Mapping[str, Any], outer: str, step: int
) -> Mapping[str, Any]:
    stage, full = proof.get("stage_event"), metrics.get("stage_event")
    contract = consumer["contract"]
    _require(
        proof.get("consumer_identity_sha256") == consumer["identity_sha256"]
        and proof.get("contract_sha256") == _hash(contract)
        and proof.get("schema") == "omni-agent-output-interpretation-v1"
        and proof.get("mode") == contract["mode"]
        and _digest(proof.get("raw_output_sha256"))
        and proof.get("model_request_id") == f"{outer}-step-{step}"
        and isinstance(stage, Mapping)
        and set(stage) == _STAGE
        and stage.get("request_id") == proof["model_request_id"]
        and isinstance(stage.get("worker_generation"), str)
        and bool(stage["worker_generation"])
        and type(stage.get("stage_id")) is int
        and stage["stage_id"] >= 0
        and type(stage.get("epoch")) is int
        and stage["epoch"] > 0
        and type(stage.get("seq")) is int
        and stage["seq"] > 0
        and stage.get("terminal") is True
        and stage.get("kind") == "text"
        and isinstance(full, Mapping)
        and all(type(full.get(key)) is type(stage[key]) and full[key] == stage[key] for key in _STAGE)
        and "error" in full
        and full["error"] is None
        and "state" in full
        and full["state"] is None
        and full.get("buffers") == []
        and full.get("release_token") == ""
        and metrics.get("finish_reason") == "stop"
        and metrics.get("raw_model_output_sha256") == proof["raw_output_sha256"]
        and proof.get("constrained_decoding") is False
        and proof.get("extraction_used") is False
        and type(proof.get("retry_count")) is int
        and proof["retry_count"] == 0,
        "consumer proof lacks an exact successful released terminal",
    )
    return stage


def validated_consumer_final(
    event: Mapping[str, Any], route: Mapping[str, Any], owned_events: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Identify final visibility only; this does not establish complete-trace qualification."""
    try:
        bound = _consumer(route)
        if bound is None or event.get("kind") != "final":
            return None
        _, consumer = bound
        payload, outer = event.get("payload"), event.get("request_id")
        _require(
            isinstance(payload, Mapping)
            and isinstance(payload.get("answer"), str)
            and bool(payload["answer"])
            and payload.get("streamed") is False
            and type(payload.get("model_step")) is int
            and payload["model_step"] >= 0
            and isinstance(outer, str)
            and bool(outer),
            "invalid final visibility",
        )
        step = payload["model_step"]
        owned = [
            item
            for item in owned_events
            if item.get("request_id") == outer
            and type(item.get("epoch")) is type(event.get("epoch"))
            and item.get("epoch") == event.get("epoch")
        ]
        routes = [item["payload"] for item in owned if item.get("kind") == "route"]
        proofs = [
            item["payload"]
            for item in owned
            if item.get("kind") == "model_output_contract"
            and type(item.get("payload", {}).get("step")) is int
            and item["payload"]["step"] == step
        ]
        terminals = [
            item["payload"]["metrics"]
            for item in owned
            if item.get("kind") == "model_metrics"
            and type(item.get("payload", {}).get("step")) is int
            and item["payload"]["step"] == step
        ]
        _require(len(routes) == len(proofs) == len(terminals) == 1, "missing final consumer evidence")
        identity, proof = routes[0], proofs[0]
        _require(
            identity.get("artifact_id") == route["artifact_id"]
            and identity.get("backend") == route["backend"]
            and identity.get("model_output_consumer_identity") == consumer
            and proof.get("canonical_output_sha256") == _hash({"final": payload["answer"]}),
            "final consumer identity/canonical output differs",
        )
        _terminal(proof, terminals[0], consumer, outer, step)
        return {
            "text": payload["answer"],
            "visibility": "validated_final_full_response",
            "model_request_id": proof["model_request_id"],
            "consumer_identity_sha256": consumer["identity_sha256"],
            "raw_output_sha256": proof["raw_output_sha256"],
            "scope": "Agent final event after validated terminal; not hidden SSE or Qt-render timing",
        }
    except (KeyError, TypeError, ValueError, AttributeError, UnicodeError):
        return None


def _identities(evidence: Mapping[str, Any], contract: AgentOutputContract, outer: str) -> list[Mapping[str, Any]]:
    policy, records = evidence.get("model_step_identity_policy"), evidence.get("model_step_identities")
    _require(
        isinstance(policy, Mapping)
        and _encoded(dict(policy)) == _encoded(model_step_capture_policy(contract, policy.get("max_model_steps"))),
        "model-step capture policy differs",
    )
    _require(
        isinstance(records, list)
        and 0 < len(records) <= policy["max_model_steps"]
        and type(evidence.get("model_prompt_step_count")) is int
        and evidence["model_prompt_step_count"] == len(records)
        and isinstance(records[0], Mapping),
        "missing bounded model-step identities",
    )
    for step, record in enumerate(records):
        _require(
            isinstance(record, Mapping)
            and set(record) == _STEP
            and type(record["step"]) is int
            and record["step"] == step
            and record["model_request_id"] == f"{outer}-step-{step}"
            and isinstance(record["model_request_id"], str)
            and len(record["model_request_id"]) <= 256
            and _digest(record["sha256"])
            and type(record["utf8_bytes"]) is int
            and type(record["chars"]) is int
            and 0 < record["chars"] <= record["utf8_bytes"] <= 4 * record["chars"]
            and len(_encoded(dict(record))) <= policy["max_record_bytes"],
            "invalid or overflowing model-step identity",
        )
    first = {key: value for key, value in records[0].items() if key != "model_request_id"}
    _require(
        isinstance(evidence.get("first_model_prompt_identity"), Mapping)
        and _encoded(dict(evidence["first_model_prompt_identity"])) == _encoded(first),
        "legacy first-model identity differs from independent capture",
    )
    _require(
        len(_encoded(records)) <= policy["declared_metadata_bytes"] // policy["transient_copies"],
        "encoded model-step identities exceed declared metadata",
    )
    return records


def _native_io(
    metrics: Mapping[str, Any],
    stage: Mapping[str, Any],
    plan: Mapping[str, Any],
    binding: Mapping[str, Any],
    previous_seq: int | None,
) -> int | None:
    observed = binding.get("observation_runtime")
    execution = binding.get("execution_observation")
    if observed is None and execution is None:
        return previous_seq
    _require(observed is None or execution is None, "legacy and combined runtime observations cannot be mixed")
    if execution is not None:
        _require(isinstance(execution, Mapping), "combined runtime observation binding is missing")
        identity = execution.get("static_identity_sha256")
        schema = "omni-strata-combined-request-io-observation-v1"
    else:
        _require(isinstance(observed, Mapping), "observed runtime binding is missing")
        identity = observed.get("identity_sha256")
        schema = "omni-strata-request-io-observation-v1"
    owner = plan.get("gpu_observer_identity")
    observation = metrics.get("backend_metrics", {}).get("runtime_telemetry", {}).get("native_io_observation")
    _require(
        isinstance(owner, Mapping)
        and owner.get("status") == "verified"
        and type(owner.get("pid")) is int
        and owner["pid"] > 0
        and type(owner.get("creation_filetime_100ns")) is int
        and owner["creation_filetime_100ns"] > 0
        and isinstance(observation, Mapping)
        and observation.get("schema") == schema
        and (
            execution is None
            or observation.get("runtime_identity_schema") == "omni-strata-combined-static-runtime-identity-v2"
        )
        and observation.get("status") == "complete"
        and observation.get("reasons") == []
        and observation.get("native_terminal") == "stop"
        and observation.get("request_id") == stage["request_id"]
        and type(observation.get("epoch")) is int
        and observation["epoch"] == stage["epoch"]
        and observation.get("generation")
        == stage["worker_generation"]
        == plan["worker_generation"]
        == owner.get("worker_generation")
        and type(observation.get("native_pid")) is int
        and observation["native_pid"] == owner["pid"]
        and type(observation.get("creation_filetime_100ns")) is int
        and observation["creation_filetime_100ns"] == owner["creation_filetime_100ns"]
        and observation.get("runtime_identity_sha256") == identity
        and observation.get("scope") == "native_FileExpertSource_and_PLE_counters_excludes_loading"
        and "physical_ssd_read_bytes" in observation
        and observation.get("physical_ssd_read_bytes") is None
        and observation.get("loading_covered") is False
        and observation.get("three_tier_memory_qualified") is False
        and type(observation.get("native_request_seq")) is int
        and observation["native_request_seq"] > 0
        and (previous_seq is None or observation["native_request_seq"] == previous_seq + 1),
        "observed runtime lacks complete owned sequential native I/O",
    )
    return observation["native_request_seq"]


def validate_consumer_trace(
    events: Sequence[Mapping[str, Any]],
    answer: str,
    route: Mapping[str, Any],
    placement_evidence: Mapping[str, Any],
    *,
    trusted_task_binding: Mapping[str, Any],
) -> None:
    """Require a complete finite model/tool/final chain, without promoting its route."""
    bound = _consumer(route)
    _require(bound is not None, "trace does not declare an explicit consumer")
    contract, consumer = bound
    _require(isinstance(placement_evidence, Mapping), "missing consumer placement evidence")
    _require(
        isinstance(events, Sequence)
        and not isinstance(events, (str, bytes))
        and bool(events)
        and all(isinstance(event, Mapping) and isinstance(event.get("payload"), Mapping) for event in events),
        "consumer trace has malformed events",
    )
    outer, epoch, session = events[0].get("request_id"), events[0].get("epoch"), events[0].get("session_id")
    _require(
        isinstance(outer, str)
        and bool(outer)
        and isinstance(session, str)
        and bool(session)
        and type(epoch) is int
        and epoch > 0
        and isinstance(answer, str)
        and bool(answer)
        and all(
            event.get("request_id") == outer
            and event.get("session_id") == session
            and type(event.get("epoch")) is int
            and event["epoch"] == epoch
            and type(event.get("seq")) is int
            and event["seq"] == index
            for index, event in enumerate(events, 1)
        ),
        "outer trace identities are reused or unordered",
    )
    _require(
        isinstance(trusted_task_binding, Mapping)
        and set(trusted_task_binding) == {"text", "read_url", "mode"}
        and isinstance(trusted_task_binding["text"], str)
        and bool(trusted_task_binding["text"].strip()),
        "missing exact trusted task binding",
    )
    # Deferred source-owned application semantics; never duplicate its permissions.
    from vllm_omni.edge.agent.controller import _permitted_tools, _validate_structured_read_url
    from vllm_omni.edge.agent.router import classify_task
    from vllm_omni.edge.agent.tools import _WRITE_OPERATIONS, ToolAction, WindowsToolBoundary

    text, url, mode = (trusted_task_binding[key] for key in ("text", "read_url", "mode"))
    if mode == "ordinary":
        _require(url is None, "ordinary task carries a structured URL")
        initial, task, task_class = {"text": text}, text, classify_task(text)
    elif mode == "read_url":
        _validate_structured_read_url(url, text)
        initial, task, task_class = (
            {"text": text, "read_url": url, "mode": mode},
            f"Read this URL: {url}\nInstruction: {text}",
            "browser_text",
        )
    else:
        raise ValueError("unknown trusted task mode")
    _require(task_class != "browser_vision", "text consumer traces cannot qualify an image task")
    permitted = _permitted_tools(task_class, task)
    boundary = WindowsToolBoundary()
    if mode == "read_url":
        boundary.register_explicit_url(outer, url)
    else:
        boundary.register_user_task(outer, text)
    records = _identities(placement_evidence, contract, outer)
    if route["backend"] == STRATA_BACKEND:
        validate_strata_request_evidence(placement_evidence, route, events=events)
    else:
        from vllm_omni.edge.agent.llamacpp_route import validate_llamacpp_consumer_request_evidence

        validate_llamacpp_consumer_request_evidence(placement_evidence, route)
    plan, binding = placement_evidence["execution_plan"], route["backend_identity"]
    _require(type(plan.get("stage_id")) is int and plan["stage_id"] >= 0, "invalid owned stage identity")
    cursor = 0

    def take(kind: str) -> Mapping[str, Any]:
        nonlocal cursor
        _require(
            cursor < len(events) and events[cursor].get("kind") == kind,
            "consumer event chain is incomplete or reordered",
        )
        payload = events[cursor]["payload"]
        cursor += 1
        return payload

    _require(take("user_observation") == initial, "trusted user input differs from initial observation")
    identity = take("route")
    _require(
        identity.get("route_id") == route["route_id"]
        and identity.get("model") == route["model_id"]
        and identity.get("artifact_id") == route["artifact_id"]
        and identity.get("backend") == route["backend"]
        and identity.get("model_output_consumer_identity") == consumer
        and identity.get("model_output_contract") == contract.to_dict(),
        "actual consumer route differs",
    )

    def tool(*, expected: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        proposal = take("tool_proposed")
        _require(
            set(proposal) == {"operation", "arguments"}
            and proposal["operation"] in permitted
            and isinstance(proposal["arguments"], Mapping),
            "unproven or automatic primary tool",
        )
        action = ToolAction(proposal["operation"], proposal["arguments"], request_id=outer)
        boundary._validate(action)
        _require(
            action.operation not in _WRITE_OPERATIONS | {"screen_capture", "browser_follow"},
            "tool approval or DOM authorization is not proven by this trace schema",
        )
        if action.operation == "browser_open":
            _, needs_approval = boundary._navigation_context(action)
            _require(needs_approval is False, "navigation is not authorized by the exact trusted user URL")
        envelope = {"tool": proposal["operation"], "args": proposal["arguments"]}
        if expected is not None:
            _require(envelope == expected, "structured tool prefix differs")

        def result(operation: str) -> Mapping[str, Any]:
            value = take("tool_result")
            _require(
                set(value) == {"operation", "data", "source", "untrusted", "completed_during_cancel"}
                and value["operation"] == operation
                and isinstance(value["data"], Mapping)
                and isinstance(value["source"], str)
                and bool(value["source"])
                and value["untrusted"] is True
                and value["completed_during_cancel"] is False,
                "tool result is missing, unsuccessful, or trusted as policy",
            )
            return value

        primary_result = result(proposal["operation"])
        if proposal["operation"] in {"browser_open", "browser_follow"}:
            automatic = take("tool_proposed")
            _require(
                automatic
                == {"operation": "browser_read", "arguments": {}, "automatic_after_navigation": proposal["operation"]},
                "unknown automatic navigation action",
            )
            _require(result("browser_read")["source"] == primary_result["source"], "automatic read changed source")
        return envelope

    if mode == "read_url":
        tool(expected={"tool": "browser_open", "args": {"url": url}})
    previous_epoch, native_seq = 0, None
    for step, record in enumerate(records):
        metric = take("model_metrics")
        _require(
            set(metric) == {"step", "metrics", "visibility"}
            and type(metric["step"]) is int
            and metric["step"] == step
            and metric["visibility"] == "withheld_until_validated_terminal"
            and isinstance(metric["metrics"], Mapping),
            "missing exact terminal model metric",
        )
        proof = take("model_output_contract")
        _require(
            set(proof) == _PROOF
            and type(proof["step"]) is int
            and proof["step"] == step
            and proof["input_sha256"] == record["sha256"]
            and proof["model_request_id"] == record["model_request_id"]
            and proof["directive_sha256"] == _hash(contract.directive(permitted)),
            "model input, step, or trusted directive differs",
        )
        _require(
            proof["normalization"]
            in (
                {"none", "single_outer_json_fence"} if contract.mode == "strict_outer_json_fence_agent_v1" else {"none"}
            ),
            "unsupported consumer normalization",
        )
        stage = _terminal(proof, metric["metrics"], consumer, outer, step)
        _require(
            stage["worker_generation"] == plan["worker_generation"]
            and stage["stage_id"] == plan["stage_id"]
            and stage["epoch"] > previous_epoch,
            "model state generation/epoch is reused",
        )
        previous_epoch = stage["epoch"]
        if route["backend"] == STRATA_BACKEND:
            native_seq = _native_io(metric["metrics"], stage, plan, binding, native_seq)
        else:
            backend_metrics = metric["metrics"].get("backend_metrics", {})
            _require(isinstance(backend_metrics, Mapping), "invalid llama.cpp backend metrics")
            telemetry = backend_metrics.get("runtime_telemetry", {})
            unsupported_provenance = (
                "native_io_observation", "expert_compute_verified", "expert_final_storage_verified",
                "cpu_compute_verified", "cpu_expert_compute_verified", "execution_observation",
                "observation_runtime", "gpu_observer_identity", "weight_tier_plan",
            )
            _require(
                isinstance(telemetry, Mapping)
                and all(key not in layer for layer in (backend_metrics, telemetry)
                        for key in unsupported_provenance),
                "llama.cpp consumer cannot carry Strata native I/O or unknown compute provenance",
            )
        if step < len(records) - 1:
            envelope = tool()
        else:
            final_event = events[cursor] if cursor < len(events) else {}
            final = take("final")
            _require(
                final.get("answer") == answer
                and type(final.get("model_step")) is int
                and final["model_step"] == step
                and final.get("streamed") is False
                and validated_consumer_final(final_event, route, events) is not None,
                "final answer lacks the last validated consumer proof",
            )
            envelope = {"final": answer}
        _require(
            proof["canonical_output_sha256"] == _hash(envelope), "model canonical command differs from executed action"
        )
    _require(cursor == len(events), "events occur after final or model steps are missing")
