# SPDX-License-Identifier: Apache-2.0
"""Packaged reviewer for fixed-suite native Agent qualification evidence.

This checks integrity and internal consistency, not independent RAM admission,
cancel/recovery, placement logs, model quality outside fixtures, or checkpoint
lineage. It deliberately does not produce a router Qualification file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from statistics import mean
from typing import Any
from urllib.parse import urlparse

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from vllm_omni.edge.agent.consumer_trace import (
    consumer_trace_requested,
    validate_consumer_trace,
    validated_consumer_final,
)
from vllm_omni.edge.agent.fixed_suite import canonical_fixed_cases
from vllm_omni.edge.agent.placement import (
    STRATA_BACKEND,
    native_gpu_sample_summary,
    preparation_placement_matches,
    result_placement_matches,
    strata_route_binding,
    validate_strata_profile_plan,
    validate_strata_request_evidence,
)
from vllm_omni.edge.agent.router import FIXED_SUITE_ID, Qualification

SUITE_ID = FIXED_SUITE_ID
STRUCTURED_READ_URL_SUITE_ID = SUITE_ID + "+explicit-read-url-v1"
LENGTH_BUCKETS = ("short", "medium", "long")
_WRITE_OPERATIONS = frozenset({"browser_click", "browser_fill", "browser_post", "settings_set"})
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _structured_contract(case: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the benchmark's explicit-URL input binding independently."""
    prompt = case["prompt"]
    encoded = prompt.encode("utf-8")
    contract = {
        "submission_mode": "explicit_read_url_v1",
        "case_id": case["case_id"],
        "instruction_sha256": hashlib.sha256(encoded).hexdigest(),
        "instruction_utf8_bytes": len(encoded),
        "explicit_read_url": case["metadata"]["source"],
    }
    contract["contract_sha256"] = hashlib.sha256(json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return contract


def _structured_request_valid(row: Mapping[str, Any], origin: str) -> bool:
    """A hosted fixture URL is trusted only through the separate user field."""
    try:
        case = row["case"]
        if case["task_class"] != "browser_text" or case["metadata"]["kind"] != "browser_text":
            return False
        expected = canonical_fixed_cases(origin, 10)["browser_text"][case["length"]]
        if case not in expected:
            return False
        if row["fixture_setup"]["input_contract"] != _structured_contract(case):
            return False
        observed = [event.get("payload", {}).get("payload", {})
                    for event in row.get("events", ())
                    if event.get("kind") == "agent_event" and
                    event.get("payload", {}).get("kind") == "user_observation"]
        if (len(observed) != 1 or observed[0].get("text") != case["prompt"] or
                observed[0].get("read_url") != case["metadata"]["source"] or
                observed[0].get("mode") != "read_url"):
            return False
        identities = [event.get("payload", {}) for event in row.get("events", ())
                      if event.get("kind") == "model_prompt_identity"]
        if identities:
            if len(identities) != 1:
                return False
            proof = identities[0]
            first = proof.get("first")
            count = proof.get("model_steps")
            if type(count) is not int or count < 0:
                return False
            if first is None:
                if count != 0:
                    return False
            elif (count < 1 or type(first.get("step")) is not int or first["step"] != 0 or
                  not isinstance(first.get("sha256"), str) or
                  not _SHA256_HEX.fullmatch(first["sha256"]) or
                  type(first.get("utf8_bytes")) is not int or first["utf8_bytes"] < 1 or
                  type(first.get("chars")) is not int or first["chars"] < 1):
                return False
            result = row.get("result")
            if result is not None:
                evidence = result["placement_evidence"]
                if (evidence.get("first_model_prompt_identity") != first or
                        evidence.get("model_prompt_step_count") != count):
                    return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _evaluate_case(case: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the fixed suite's exact-output, read-evidence and safety checks."""
    decisions = [dict(item) for item in result.get("tool_decisions", ())]
    proposed = [str(item.get("payload", {}).get("operation", "")) for item in decisions
                if item.get("kind") == "tool_proposed"]
    executed = [str(item.get("payload", {}).get("operation", "")) for item in decisions
                if item.get("kind") == "tool_result"]
    tool_safe = not any(op in _WRITE_OPERATIONS for op in (*proposed, *executed))
    tool_safe &= not any(item.get("kind") == "approval_required" for item in decisions)
    metadata = case.get("metadata", {})
    required = tuple(metadata.get("required_operations", ()))
    required_seen = all(op in executed for op in required)
    answer = (result.get("final_answer") or "").strip().strip("`* .。\n\t")
    answer_ok = answer.casefold() == str(case.get("reference")).casefold()
    source_ok = True
    expected_source = str(metadata.get("source", ""))
    if expected_source.startswith("http"):
        source_ok = any(
            item.get("kind") == "tool_result" and
            str(item.get("payload", {}).get("source", "")) == expected_source
            for item in decisions
        )
        tool_safe &= all(
            _origin(str(item.get("payload", {}).get("arguments", {}).get("url", "")))
            == _origin(expected_source)
            for item in decisions if item.get("kind") == "tool_proposed"
            and item.get("payload", {}).get("operation") == "browser_open"
        )
    elif expected_source == "windows-screen":
        source_ok = any(
            item.get("kind") == "tool_result" and
            item.get("payload", {}).get("source") == "windows-screen"
            for item in decisions
        )
    quality = bool(answer_ok and required_seen and source_ok and result.get("complete_agent_trace"))
    return {
        "success": quality and tool_safe,
        "quality_pass": quality,
        "quality_score": 1.0 if quality else 0.0,
        "tool_safe": tool_safe,
        "details": {
            "answer_exact": answer_ok,
            "required_tools_seen": required_seen,
            "source_verified": source_ok,
            "quality_scope": metadata.get("quality_scope"),
            "code_tool_available": False if metadata.get("kind") == "code_reasoning" else None,
        },
    }


def _consumer_task_binding(row: Mapping[str, Any], *, structured_origin: str | None) -> dict[str, Any]:
    """Bind permissions to the case and its independent trusted input field."""
    case = row["case"]
    text = case["prompt"]
    if not isinstance(text, str) or not text:
        raise ValueError("consumer task needs exact nonempty user text")
    encoded = text.encode("utf-8")
    read_url = None
    if structured_origin is not None:
        if not _structured_request_valid(row, structured_origin):
            raise ValueError("consumer Read URL differs from its fixed input contract")
        read_url = case["metadata"]["source"]
    else:
        expected = {
            "submission_mode": "model_selected_tools_v1",
            "case_id": case["case_id"],
            "instruction_sha256": hashlib.sha256(encoded).hexdigest(),
            "instruction_utf8_bytes": len(encoded),
            "explicit_read_url": None,
        }
        expected["contract_sha256"] = hashlib.sha256(json.dumps(
            expected, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        if row["fixture_setup"]["input_contract"] != expected:
            raise ValueError("consumer ordinary task differs from its fixed input contract")
    observations = [event.get("payload", {}).get("payload") for event in row.get("events", ())
                    if event.get("kind") == "agent_event"
                    and event.get("payload", {}).get("kind") == "user_observation"]
    expected_observation = ({"text": text, "read_url": read_url, "mode": "read_url"}
                            if read_url is not None else {"text": text})
    if observations != [expected_observation]:
        raise ValueError("consumer user observation differs from the trusted case")
    return {"text": text, "read_url": read_url, "mode": "read_url" if read_url is not None else "ordinary"}


def _consumer_capture_valid(row: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    """Cross-check backend-boundary metadata with the separately written event."""
    captured = [event.get("payload") for event in row.get("events", ())
                if event.get("kind") == "model_prompt_identity"]
    if len(captured) != 1 or not isinstance(captured[0], Mapping):
        raise ValueError("consumer needs one independent model-input capture")
    capture = captured[0]
    steps = evidence.get("model_step_identities")
    policy = evidence.get("model_step_identity_policy")
    if (set(capture) != {"first", "model_steps", "steps", "policy"}
            or not isinstance(steps, list) or not steps or not isinstance(policy, Mapping)
            or capture.get("steps") != steps or capture.get("policy") != policy
            or type(capture.get("model_steps")) is not int or capture["model_steps"] != len(steps)
            or type(evidence.get("model_prompt_step_count")) is not int
            or evidence.get("model_prompt_step_count") != len(steps)):
        raise ValueError("consumer model-input event and placement metadata differ")
    first = {key: steps[0][key] for key in ("step", "sha256", "utf8_bytes", "chars")}
    expected = {"first": first, "model_steps": len(steps), "steps": steps, "policy": policy}
    # JSON comparison also rejects bool/int substitutions that Python equality permits.
    if json.dumps(capture, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("consumer capture types differ from the complete step list")
    if (json.dumps(evidence.get("first_model_prompt_identity"), sort_keys=True, allow_nan=False)
            != json.dumps(first, sort_keys=True, allow_nan=False)):
        raise ValueError("consumer first-input aliases differ from the complete step list")


def _consumer_visible_time(row: Mapping[str, Any], route: Mapping[str, Any],
                           agent_events: list[Mapping[str, Any]]) -> float | None:
    """Reconstruct time to validated final visibility, independently of hidden SSE."""
    elapsed = row.get("answer_latency_s")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("consumer full-answer latency is invalid")
    records = row.get("events", ())
    previous = 0.0
    for record in records:
        offset = record.get("offset_s")
        if (type(offset) not in (int, float) or not math.isfinite(offset)
                or offset < previous or offset > elapsed):
            raise ValueError("consumer event offsets are invalid or reordered")
        previous = offset
    if any(record.get("kind") in {"assistant_text_delta", "visible_token"} for record in records):
        raise ValueError("consumer emitted text before complete validation")
    finals = [event for event in agent_events if event.get("kind") == "final"]
    visible = [(index, record) for index, record in enumerate(records)
               if record.get("kind") == "assistant_final"]
    if len(finals) > 1 or len(visible) > 1:
        raise ValueError("consumer final visibility is duplicated")
    expected = validated_consumer_final(finals[0], route, agent_events) if finals else None
    if expected is None:
        if visible:
            raise ValueError("consumer final visibility lacks a validated terminal")
        offset, kind = None, None
    else:
        if len(visible) != 1 or visible[0][1].get("payload") != expected:
            raise ValueError("consumer final visibility differs from its owned proof")
        final_records = [index for index, record in enumerate(records)
                         if record.get("kind") == "agent_event" and record.get("payload") == finals[0]]
        if len(final_records) != 1 or visible[0][0] <= final_records[0]:
            raise ValueError("consumer visibility precedes the owned Agent final")
        offset, kind = visible[0][1]["offset_s"], "assistant_final"
    scope = "validated_final_full_response_not_hidden_model_delta_or_Qt_render"
    if (row.get("first_visible_event_kind") != kind
            or row.get("first_visible_output_scope") != scope
            or row.get("ttft_scope") != "time_to_validated_final_visibility"):
        raise ValueError("consumer visibility timing has another scope")
    return offset


def _consumer_state_order(evidence: Mapping[str, Any],
                          previous: dict[tuple[int, str], tuple[int, int | None]]) -> None:
    """Keep single-worker state order across independently captured requests."""
    observed = evidence["execution_plan"].get("observation_runtime") is not None
    for terminal in evidence["terminal_model_metrics"]:
        metrics = terminal["metrics"]
        stage = metrics["stage_event"]
        stage_id, generation, epoch = (stage.get(key) for key in ("stage_id", "worker_generation", "epoch"))
        if (type(stage_id) is not int or stage_id < 0 or not isinstance(generation, str)
                or not generation or type(epoch) is not int or epoch < 1):
            raise ValueError("consumer native state identity is invalid")
        native_seq = None
        if observed:
            io = metrics["backend_metrics"]["runtime_telemetry"]["native_io_observation"]
            native_seq = io.get("native_request_seq")
            if (type(native_seq) is not int or native_seq < 1 or io.get("epoch") != epoch
                    or io.get("generation") != generation):
                raise ValueError("consumer native I/O state differs from its terminal")
        key = (stage_id, generation)
        old_epoch, old_seq = previous.get(key, (0, None))
        if epoch <= old_epoch or (old_seq is not None and (native_seq is None or native_seq <= old_seq)):
            raise ValueError("consumer native state is replayed or reversed across requests")
        previous[key] = epoch, native_seq


def _trace_complete(events: list[Mapping[str, Any]], answer: str | None,
                    route: Mapping[str, Any], placement_evidence: Mapping[str, Any] | None = None,
                    *, trusted_task_binding: Mapping[str, Any] | None = None) -> bool:
    try:
        if consumer_trace_requested(route):
            if trusted_task_binding is None:
                return False
            validate_consumer_trace(events, answer, route, placement_evidence,
                                    trusted_task_binding=trusted_task_binding)
            return True
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, StopIteration):
        return False
    if not events or [event.get("seq") for event in events] != list(range(1, len(events) + 1)):
        return False
    if len({event.get("request_id") for event in events}) != 1 or len({event.get("epoch") for event in events}) != 1:
        return False
    kinds = [event.get("kind") for event in events]
    if kinds[0] != "user_observation" or kinds[-1] != "final":
        return False
    if any(kind in kinds for kind in ("error", "refusal", "cancelled", "approval_required")):
        return False
    if kinds.count("route") != 1 or "model_metrics" not in kinds:
        return False
    identity = next(event for event in events if event["kind"] == "route").get("payload", {})
    if (identity.get("route_id") != route["route_id"] or
        identity.get("model") != route["model_id"] or
        identity.get("artifact_id") != route["artifact_id"] or
        identity.get("backend") != route["backend"]):
        return False
    if route["backend"] == "external.strata.multimodal.v1":
        return False  # requires a dedicated reviewed image task/profile suite
    if route["backend"] == STRATA_BACKEND:
        try:
            validate_strata_request_evidence(placement_evidence, route, events=events)
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, StopIteration):
            return False
    elif identity.get("actual_placement") != route["expected_placement"]:
        return False
    return events[-1].get("payload", {}).get("answer") == answer


def _telemetry_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_count": len(samples)}
    for name in ("ram_used_bytes", "vram_used_bytes", "gpu_power_w",
                 "system_power_w", "gpu_temp_c"):
        values = [sample[name] for sample in samples
                  if isinstance(sample.get(name), (int, float))]
        if values:
            result[name + "_peak"] = max(values)
            if name.endswith("power_w"):
                result[name + "_mean"] = mean(values)
    result["peak_is_sampled_lower_bound"] = True
    if any("native_process_gpu_memory" in sample for sample in samples):
        result["native_process_gpu_memory"] = native_gpu_sample_summary(samples)
    return result


def _nearest_rank(values: Sequence[float], fraction: float) -> float | None:
    return sorted(values)[math.ceil(len(values) * fraction) - 1] if values else None


def _route_profile(route: Mapping[str, Any], samples: Sequence[Mapping[str, Any]],
                   *, cold_start_s: float | None, sustained_seconds: float,
                   endurance_wall_seconds: float,
                   config: Mapping[str, Any]) -> dict[str, Any]:
    measured = [row for row in samples if row["phase"] == "measured"]
    warmups = [row for row in samples if row["phase"] == "warmup"]
    endurance = [row for row in samples if row["phase"] == "endurance"]
    by_length = {bucket: [row for row in measured if row["case"]["length"] == bucket]
                 for bucket in LENGTH_BUCKETS}
    answer = {bucket: [row["answer_latency_s"] for row in rows]
              for bucket, rows in by_length.items()}
    first = {bucket: [row["ttft_s"] for row in rows if row["ttft_s"] is not None]
             for bucket, rows in by_length.items()}
    warmup_counts = {bucket: sum(row["case"]["length"] == bucket for row in warmups)
                     for bucket in LENGTH_BUCKETS}
    success = sum(bool(row["e2e_complete"] and row["placement_matches"]
                       and row["evaluation"] and row["evaluation"]["success"])
                  for row in measured)
    all_rows = [*measured, *endurance]
    e2e = all(bool(row["e2e_complete"]) for row in all_rows)
    placement_matches = all(bool(row["placement_matches"]) for row in all_rows)
    placement_evidence = all(bool(row["result"] and row["result"]["placement_evidence"])
                             for row in all_rows)
    correctness = all(bool(row["evaluation"] and row["evaluation"]["quality_pass"]
                           and row["evaluation"]["success"])
                      for row in measured)
    tool_safety = all(bool(row["evaluation"] and row["evaluation"]["tool_safe"])
                      for row in all_rows)
    stable = bool(endurance) and sustained_seconds >= config["endurance_seconds"] and all(
        row["e2e_complete"] and row["error"] is None
        and row["evaluation"] and row["evaluation"]["success"]
        and row["evaluation"]["quality_pass"]
        and row["evaluation"]["tool_safe"] for row in endurance
    )
    telemetry_present = all(bool(row["telemetry"]["raw_samples"])
                            and not row["telemetry"]["errors"] for row in all_rows)
    sample_counts_ok = all(len(by_length[bucket]) >= 20 and
                           len(first[bucket]) == len(by_length[bucket]) and
                           warmup_counts[bucket] >= 1 for bucket in LENGTH_BUCKETS)
    protocol_minimums = (config["warmups_per_length"] >= 1 and
                         config["measured_per_length"] >= 20 and
                         config["endurance_seconds"] >= 1800)
    compliant = bool(protocol_minimums and sample_counts_ok and
                     sustained_seconds >= 1800 and e2e and placement_matches
                     and telemetry_present and cold_start_s is not None)
    return {
        "route": dict(route),
        "task_class": measured[0]["case"]["task_class"] if measured else "",
        "measured_attempts": len(measured),
        "measured_successes": success,
        "answer_latency_s": answer,
        "ttft_s": first,
        "answer_p50_s": {b: _nearest_rank(answer[b], .5) for b in LENGTH_BUCKETS},
        "answer_p95_s": {b: _nearest_rank(answer[b], .95) for b in LENGTH_BUCKETS},
        "ttft_p50_s": {b: _nearest_rank(first[b], .5) for b in LENGTH_BUCKETS},
        "ttft_p95_s": {b: _nearest_rank(first[b], .95) for b in LENGTH_BUCKETS},
        "warmups_per_length": warmup_counts,
        "sustained_seconds": sustained_seconds,
        "endurance_wall_seconds": endurance_wall_seconds,
        "endurance_requests": len(endurance),
        "correctness_pass": correctness,
        "tool_safety_pass": tool_safety,
        "stability_pass": stable,
        "e2e_trace_pass": e2e,
        "actual_placement_matches": placement_matches,
        "placement_evidence_present": placement_evidence,
        "telemetry_present": telemetry_present,
        "cold_start_s": cold_start_s,
        "protocol_compliant": compliant,
    }


@dataclass(frozen=True)
class EvidenceAudit:
    summary_path: Path
    raw_jsonl: Path
    raw_sha256: str
    route_id: str
    measured_attempts: int
    measured_successes: int
    trace_verified: bool
    protocol_compliant: bool
    errors: tuple[str, ...]

    @property
    def internally_valid(self) -> bool:
        return not self.errors


def audit_summary(summary_path: str | Path) -> EvidenceAudit:
    """Fail closed on forged counts, evaluations, timings, trace, or raw hash."""
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # The producer may be native Windows writing to a WSL checkout. Resolve
    # beside the summary instead of interpreting a Windows UNC string as a
    # relative POSIX filename when auditing from WSL.
    raw_path = (summary_path.parent / "samples.jsonl").resolve()
    errors: list[str] = []
    claimed_name = PureWindowsPath(str(summary.get("raw_jsonl", ""))).name
    if claimed_name != raw_path.name:
        errors.append("summary does not name the colocated raw JSONL")
    raw_bytes = raw_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest != summary.get("raw_sha256"):
        errors.append("raw JSONL SHA-256 differs from summary")
    rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines()]
    if not rows or rows[0].get("record_type") != "manifest":
        raise ValueError("raw profile manifest is missing")
    manifest = rows[0]
    if (manifest.get("scope") != "complete_agent_request" or
            manifest.get("batch_size") != 1 or manifest.get("concurrency") != 1):
        errors.append("raw manifest is not a batch-one complete Agent profile")
    if manifest.get("conditions") != summary.get("conditions"):
        errors.append("profile conditions differ from raw manifest")
    suite_id = manifest.get("conditions", {}).get("suite_id")
    if suite_id not in {SUITE_ID, STRUCTURED_READ_URL_SUITE_ID}:
        errors.append("profile uses a different evaluator suite")
    if manifest.get("run_id") != summary.get("run_id"):
        errors.append("run ID differs from raw manifest")
    route_rows = manifest.get("routes", [])
    if len(route_rows) != 1:
        raise ValueError("native fixed-suite audit expects exactly one route")
    route = route_rows[0]
    try:
        consumer_requested = consumer_trace_requested(route)
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError):
        # An orphan or conflicting consumer marker must never use legacy checks.
        consumer_requested = True
        errors.append("route has an invalid explicit consumer identity")
    if set(summary.get("routes", {})) != {route["route_id"]}:
        errors.append("route set differs from raw manifest")
    config = manifest["config"]
    requests = [row for row in rows if row.get("record_type") == "request"]
    structured_origin = None
    if suite_id == STRUCTURED_READ_URL_SUITE_ID:
        try:
            source = requests[0]["case"]["metadata"]["source"]
            parsed = urlparse(source)
            port = parsed.port
            if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or
                    parsed.username is not None or parsed.password is not None or
                    port is None or parsed.netloc != f"127.0.0.1:{port}"):
                raise ValueError("invalid structured fixture origin")
            structured_origin = f"http://127.0.0.1:{port}"
        except (IndexError, KeyError, TypeError, ValueError):
            errors.append("structured suite has no exact loopback fixture origin")
    prepare = [row for row in rows if row.get("record_type") == "route_prepare"]
    if len(prepare) != 1:
        errors.append("cold route preparation row is missing or duplicated")
        cold_start_s = None
    else:
        prepared = prepare[0]
        if prepared.get("route_id") != route["route_id"]:
            errors.append("cold preparation has another route ID")
        prep = prepared.get("preparation", {})
        if not preparation_placement_matches(prep, route):
            errors.append("cold preparation does not match the route")
        cold_start_s = prepared.get("cold_start_s")
    trace_verified = bool(requests)
    strata_states: set[tuple[str, int]] = set()
    consumer_states: dict[tuple[int, str], tuple[int, int | None]] = {}
    consumer_requests: set[str] = set()
    consumer_epochs: dict[str, int] = {}
    for row in requests:
        if (suite_id == STRUCTURED_READ_URL_SUITE_ID and
                (structured_origin is None or
                 not _structured_request_valid(row, structured_origin))):
            errors.append("structured Read URL request differs from fixed input contract")
        if (row.get("run_id") != summary.get("run_id") or
            row.get("route_id") != route["route_id"] or
            row.get("batch_size") != 1 or row.get("concurrency") != 1):
            errors.append("request identity or batch-one contract differs")
        if row.get("phase") not in {"warmup", "measured", "endurance"}:
            errors.append("request phase is invalid")
        case = row["case"]
        result_data = row.get("result")
        if result_data is not None:
            expected_eval = _evaluate_case(case, result_data)
            if row.get("evaluation") != expected_eval:
                errors.append(f"{case['case_id']}: evaluation differs from raw answer and tool decisions")
        agent_events = [event.get("payload") for event in row.get("events", [])
                        if event.get("kind") == "agent_event"]
        trusted_task_binding = None
        consumer_metadata_valid = not consumer_requested
        if consumer_requested:
            try:
                if suite_id == STRUCTURED_READ_URL_SUITE_ID and structured_origin is None:
                    raise ValueError("consumer structured input has no trusted fixture origin")
                trusted_task_binding = _consumer_task_binding(row, structured_origin=structured_origin)
                _consumer_capture_valid(row, result_data["placement_evidence"])
                first = agent_events[0]
                outer, session, epoch = (first.get(key) for key in ("request_id", "session_id", "epoch"))
                if (not isinstance(outer, str) or not outer or outer in consumer_requests
                        or not isinstance(session, str) or not session or type(epoch) is not int
                        or epoch <= consumer_epochs.get(session, 0)):
                    raise ValueError("consumer outer request identity is reused or reversed")
                consumer_requests.add(outer)
                consumer_epochs[session] = epoch
                _consumer_state_order(result_data["placement_evidence"], consumer_states)
                consumer_metadata_valid = True
            except (KeyError, IndexError, TypeError, ValueError, RuntimeError, AttributeError):
                errors.append(f"{case['case_id']}: consumer input, capture, or cross-request state is invalid")
        if result_data is not None and agent_events:
            decisions = [
                {"kind": event["kind"], "payload": event.get("payload", {})}
                for event in agent_events if event.get("kind") in
                {"tool_proposed", "approval_required", "tool_result"}
            ]
            if result_data.get("tool_decisions") != decisions:
                errors.append(f"{case['case_id']}: tool decisions differ from ordered Agent events")
            expected_complete = bool(
                result_data.get("trace_scope") == "agent_e2e"
                and result_data.get("complete_agent_trace")
                and result_data.get("final_answer") is not None
            )
            if row.get("e2e_complete") != expected_complete:
                errors.append(f"{case['case_id']}: E2E flag differs from raw result")
            expected_placement = result_placement_matches(result_data, route)
            if row.get("placement_matches") != expected_placement:
                errors.append(f"{case['case_id']}: placement flag differs from raw result")
            if route["backend"] == STRATA_BACKEND and len(prepare) == 1:
                if (result_data.get("placement_evidence", {}).get("loaded_plan_sha256") !=
                        prepare[0].get("preparation", {}).get("details", {}).get("loaded_plan_sha256")):
                    errors.append(f"{case['case_id']}: Strata request uses another loaded plan")
                for terminal in result_data.get("placement_evidence", {}).get("terminal_model_metrics", []):
                    stage = terminal.get("metrics", {}).get("stage_event", {})
                    state = (stage.get("worker_generation"), stage.get("epoch"))
                    if not isinstance(state[0], str) or type(state[1]) is not int:
                        errors.append(f"{case['case_id']}: Strata terminal state is invalid")
                        continue
                    if state in strata_states:
                        errors.append(f"{case['case_id']}: Strata native state is reused across requests")
                    strata_states.add(state)
        if result_data is None or not agent_events or not consumer_metadata_valid or not _trace_complete(
            agent_events, result_data["final_answer"], route, result_data.get("placement_evidence"),
            trusted_task_binding=trusted_task_binding,
        ):
            trace_verified = False
            # Errors/refusals legitimately have no complete trace, but cannot
            # contribute a passing request.
            if row.get("e2e_complete"):
                errors.append(f"{case['case_id']}: claimed complete Agent trace is not reconstructible")
        if consumer_requested:
            try:
                first_visible = _consumer_visible_time(row, route, agent_events)
                ttft = row.get("ttft_s")
                if (ttft is not None and (type(ttft) not in (int, float) or not math.isfinite(ttft))) \
                        or first_visible != ttft:
                    raise ValueError("TTFT differs from validated final visibility")
            except (KeyError, IndexError, TypeError, ValueError, RuntimeError, AttributeError):
                trace_verified = False
                errors.append(f"{case['case_id']}: consumer final visibility or timing is invalid")
        else:
            first_visible = next((event["offset_s"] for event in row.get("events", [])
                                  if event.get("kind") == "assistant_text_delta"), None)
            if first_visible != row.get("ttft_s"):
                errors.append(f"{case['case_id']}: TTFT differs from first visible delta")
            if row.get("answer_latency_s", -1) < 0 or (
                first_visible is not None and first_visible > row["answer_latency_s"]
            ):
                errors.append(f"{case['case_id']}: invalid full-answer latency")
        telem = row.get("telemetry", {})
        if telem.get("summary") != _telemetry_summary(telem.get("raw_samples", [])):
            errors.append(f"{case['case_id']}: telemetry summary differs from raw samples")
        if row.get("fixture_setup", {}).get("private_memory_reset") is not True:
            errors.append(f"{case['case_id']}: isolated fixture memory reset not recorded")
    stored_profile = summary.get("routes", {}).get(route["route_id"], {})
    active = sum(row["answer_latency_s"] for row in requests if row["phase"] == "endurance")
    wall = stored_profile.get("endurance_wall_seconds", 0)
    if wall < active:
        errors.append("endurance wall time is shorter than active request time")
    calculated = _route_profile(
        route, requests, cold_start_s=cold_start_s,
        sustained_seconds=active, endurance_wall_seconds=wall, config=config,
    )
    # The producer sums in-memory request durations before writing JSONL;
    # summing the round-tripped JSON floats can differ by a few ULPs.
    stored_other = {key: value for key, value in stored_profile.items()
                    if key != "sustained_seconds"}
    calculated_other = {key: value for key, value in calculated.items()
                        if key != "sustained_seconds"}
    if (stored_other != calculated_other or
        not math.isclose(stored_profile.get("sustained_seconds", -1),
                         calculated["sustained_seconds"], rel_tol=0,
                         abs_tol=1e-6)):
        errors.append("route profile differs from recomputed raw requests")
    if not trace_verified:
        errors.append("one or more full Agent event traces are unavailable or invalid")
    return EvidenceAudit(
        summary_path=summary_path, raw_jsonl=raw_path, raw_sha256=digest,
        route_id=route["route_id"],
        measured_attempts=int(calculated["measured_attempts"]),
        measured_successes=int(calculated["measured_successes"]),
        trace_verified=trace_verified,
        protocol_compliant=bool(calculated["protocol_compliant"]),
        errors=tuple(dict.fromkeys(errors)),
    )


def verify_fixed_suite_cases(audit: EvidenceAudit) -> None:
    """Bind a release candidate to the exact fixed prompts and task schedule.

    ``audit_summary`` also accepts synthetic diagnostic profiles. Release
    promotion and gate assembly call this stricter check. The fixture's port
    and the observed mouse setting vary per run; all other case fields are
    reconstructed from the package-local canonical suite definition.
    """
    rows = [json.loads(line) for line in audit.raw_jsonl.read_text(encoding="utf-8").splitlines()]
    manifest = rows[0]
    suite_id = manifest.get("conditions", {}).get("suite_id")
    if suite_id not in {SUITE_ID, STRUCTURED_READ_URL_SUITE_ID}:
        raise ValueError("fixed suite has an unknown submission mode")
    requests = [row for row in rows if row.get("record_type") == "request"]
    if not requests:
        raise ValueError("fixed suite has no request records")
    task_class = requests[0].get("case", {}).get("task_class")
    if suite_id == STRUCTURED_READ_URL_SUITE_ID and task_class != "browser_text":
        raise ValueError("fixed structured suite must contain browser_text only")
    origin = "http://127.0.0.1:1"
    if task_class in {"browser_text", "browser_vision"}:
        urls = [str(row.get("case", {}).get("metadata", {}).get("source", ""))
                for row in requests]
        urls = [url for url in urls if url.startswith("http")]
        if not urls:
            raise ValueError("fixed browser suite has no fixture URL")
        parsed = urlparse(urls[0])
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("fixed browser suite has an invalid loopback port") from exc
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or
            parsed.username is not None or parsed.password is not None or
            port is None or parsed.netloc != f"127.0.0.1:{port}"):
            raise ValueError("fixed browser suite must use the exact IPv4 loopback fixture")
        origin = f"http://127.0.0.1:{port}"
    mouse_speed = 1
    if task_class == "windows_settings":
        try:
            mouse_speed = int(requests[0]["case"]["reference"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("fixed Settings suite lacks an integer reference") from exc
    try:
        suites = canonical_fixed_cases(origin, mouse_speed)
        expected = suites[task_class]
        config = manifest["config"]
        warmups = config["warmups_per_length"]
        measured = config["measured_per_length"]
        if type(warmups) is not int or type(measured) is not int or warmups < 1 or measured < 1:
            raise ValueError("fixed suite has invalid request counts")
        schedule = []
        for length in LENGTH_BUCKETS:
            cases = expected[length]
            schedule.extend(("warmup", repetition, cases[repetition % len(cases)])
                            for repetition in range(warmups))
            schedule.extend(("measured", repetition, cases[repetition % len(cases)])
                            for repetition in range(measured))
        all_cases = [case for length in LENGTH_BUCKETS for case in expected[length]]
        endurance = len(requests) - len(schedule)
        if endurance < 1:
            raise ValueError("fixed suite has no sustained requests")
        schedule.extend(("endurance", repetition, all_cases[repetition % len(all_cases)])
                        for repetition in range(endurance))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("fixed suite case construction failed") from exc
    if len(requests) != len(schedule):
        raise ValueError("fixed suite request count differs from canonical schedule")
    for row, (phase, repetition, case) in zip(requests, schedule):
        if (row.get("phase") != phase or row.get("repetition") != repetition or
            row.get("case") != case):
            raise ValueError("fixed suite request differs from canonical prompt, reference, metadata, or order")
        observed = [event.get("payload", {}) for event in row.get("events", ())
                    if event.get("kind") == "agent_event" and
                    event.get("payload", {}).get("kind") == "user_observation"]
        if (len(observed) != 1 or
            observed[0].get("payload", {}).get("text") != case["prompt"]):
            raise ValueError("fixed suite model received a prompt different from the evaluated case")
        payload = observed[0]["payload"]
        if suite_id == STRUCTURED_READ_URL_SUITE_ID:
            if (payload.get("mode") != "read_url" or
                    payload.get("read_url") != case["metadata"]["source"] or
                    not _structured_request_valid(row, origin)):
                raise ValueError("fixed structured suite used another URL or input contract")
        elif payload.get("mode") == "read_url" or payload.get("read_url") is not None:
            raise ValueError("fixed ordinary suite contains a structured URL observation")
        if task_class == "windows_settings":
            readings = [event.get("payload", {}).get("payload", {})
                        for event in row.get("events", ())
                        if event.get("kind") == "agent_event" and
                        event.get("payload", {}).get("kind") == "tool_result"]
            if not any(item.get("operation") == "settings_read" and
                       item.get("source") == "windows-setting:mouse_speed" and
                       item.get("data", {}).get("value") == mouse_speed
                       for item in readings):
                raise ValueError("fixed Settings reference differs from observed tool reading")


PROMOTION_SCHEMA = "omni-agent-qualification-review-v1"
GATE_SCHEMA = "omni-agent-independent-gate-v1"
REQUIRED_GATES = frozenset({
    "memory_admission", "cancel_recovery", "runtime_placement",
    "checkpoint_lineage", "reference_quality",
})


def bundle_signing_bytes(bundle: Mapping[str, Any]) -> bytes:
    """Canonical bytes for a reviewer signature, excluding only the signature.

    The reviewing key is an explicit native-app trust anchor. A profile run
    never creates that key or signs its own result.
    """
    unsigned = dict(bundle)
    review = dict(unsigned.get("review", {}))
    review.pop("signature_ed25519", None)
    unsigned["review"] = review
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_file(base: Path, reference: Mapping[str, str]) -> Path:
    if set(reference) != {"path", "sha256"}:
        raise ValueError("evidence reference needs only path and SHA-256")
    path = Path(reference["path"])
    if not path.is_absolute():
        path = base / path
    path = path.resolve(strict=True)
    expected = reference["sha256"]
    if len(expected) != 64 or _sha256_file(path) != expected:
        raise ValueError(f"evidence SHA-256 mismatch: {path}")
    return path


def _identity(profile: Mapping[str, Any], conditions: Mapping[str, Any],
              raw_sha256: str) -> dict[str, str]:
    return {
        "route_id": profile["route_id"],
        "artifact_id": profile["artifact_id"],
        "artifact_sha256": profile["artifact_sha256"],
        "checkpoint_revision": profile["checkpoint_revision"],
        "backend": profile["backend"],
        "placement": profile["expected_placement"],
        "suite_id": conditions["suite_id"],
        "environment_fingerprint": conditions["environment_fingerprint"],
        "power_condition": conditions["power_condition"],
        "profile_raw_sha256": raw_sha256,
    }


def _verify_review_signature(bundle: Mapping[str, Any],
                             trusted_keys: Mapping[str, str]) -> None:
    review = bundle.get("review")
    if not isinstance(review, dict):
        raise ValueError("independent signed review is required")
    key_id = review.get("key_id")
    if not key_id or key_id not in trusted_keys or not review.get("reviewer") or not review.get("reviewed_at"):
        raise ValueError("review identity has no trusted key or reviewer")
    try:
        public_key = base64.b64decode(trusted_keys[key_id], validate=True)
        signature = base64.b64decode(review["signature_ed25519"], validate=True)
        if len(public_key) != 32 or len(signature) != 64:
            raise ValueError("review key or signature length is invalid")
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature, bundle_signing_bytes(bundle),
        )
    except (KeyError, ValueError, InvalidSignature) as exc:
        raise ValueError("qualification review signature is invalid") from exc


def _verify_gate_observations(name: str, observations: Mapping[str, Any],
                              identity: Mapping[str, str], base: Path,
                              native_route: Mapping[str, Any] | None = None) -> None:
    """Reject naked pass flags; require checkable data for each gate kind."""
    if name == "memory_admission":
        admitted = observations.get("admitted_demand_bytes")
        # Native telemetry reports whole-system used RAM/VRAM. A route's claim
        # must be checked against a baseline-adjusted *increment*, not that
        # system-wide total. Require all three readings so an old receipt with
        # an ambiguous `observed_peak_bytes` cannot be promoted.
        baseline = observations.get("pool_used_baseline_bytes")
        total_peaks = observations.get("pool_used_peak_bytes")
        route_peaks = observations.get("route_incremental_peak_bytes")
        ceilings = observations.get("available_ceiling_bytes")
        refused = observations.get("refused_demand_bytes")
        if not all(isinstance(x, dict) and x for x in (
            admitted, baseline, total_peaks, route_peaks, ceilings, refused,
        )):
            raise ValueError("memory gate lacks admitted, baseline, route peak, ceiling, or refusal evidence")
        if not set(admitted) == set(baseline) == set(total_peaks) == set(route_peaks) == set(ceilings):
            raise ValueError("memory gate pool names differ across demand and peak readings")
        if not isinstance(observations.get("refusal_reason"), str) or not observations["refusal_reason"]:
            raise ValueError("memory gate lacks explicit refusal reason")
        for pool, ceiling in ceilings.items():
            if (type(ceiling) is not int or ceiling <= 0 or
                type(admitted.get(pool)) is not int or
                type(baseline.get(pool)) is not int or
                type(total_peaks.get(pool)) is not int or
                type(route_peaks.get(pool)) is not int or
                not 0 <= admitted[pool] <= ceiling or
                not 0 <= baseline[pool] <= total_peaks[pool] or
                not 0 <= route_peaks[pool] <= ceiling):
                raise ValueError("memory gate has an invalid or overcommitted pool")
            if route_peaks[pool] != total_peaks[pool] - baseline[pool]:
                raise ValueError("memory gate route peak differs from baseline-adjusted pool observations")
            if route_peaks[pool] > admitted[pool]:
                raise ValueError("memory gate route peak exceeds admitted demand")
        if not any(type(refused.get(pool)) is int and refused[pool] > ceiling
                   for pool, ceiling in ceilings.items()):
            raise ValueError("memory gate has no measured over-budget refusal")
    elif name == "cancel_recovery":
        events = observations.get("events")
        if not isinstance(events, list) or len(events) < 3:
            raise ValueError("cancel gate lacks ordered request events")
        by_request: dict[str, list[tuple[int, int, str]]] = {}
        for event in events:
            request_id = event.get("request_id")
            seq = event.get("seq")
            epoch = event.get("epoch")
            kind = event.get("kind")
            if (not isinstance(request_id, str) or not request_id or
                type(seq) is not int or seq < 1 or
                type(epoch) is not int or epoch < 1 or not isinstance(kind, str)):
                raise ValueError("cancel gate has invalid event identity")
            by_request.setdefault(request_id, []).append((epoch, seq, kind))
        if any(len({epoch for epoch, _, _ in rows}) != 1 or
               [seq for _, seq, _ in rows] != sorted({seq for _, seq, _ in rows})
               for rows in by_request.values()):
            raise ValueError("cancel gate has duplicate or unordered sequence")
        cancelled = [(request_id, rows) for request_id, rows in by_request.items()
                     if any(kind == "cancelled" for _, _, kind in rows)]
        if len(cancelled) != 1:
            raise ValueError("cancel gate needs one cancelled request")
        cancel_request, cancel_rows = cancelled[0]
        cancel_epoch = cancel_rows[0][0]
        cancel_seq = next(seq for _, seq, kind in cancel_rows if kind == "cancelled")
        release_events = [event for event in events
                          if event["request_id"] == cancel_request and
                          event["kind"] == "state_released" and
                          event["seq"] > cancel_seq]
        if len(release_events) != 1:
            raise ValueError("cancelled state was not released")
        release = release_events[0].get("payload", {})
        if (release.get("request_id") != cancel_request or
            release.get("release_mode") != "worker_shutdown" or
            type(release.get("worker_pid_before")) is not int or
            release["worker_pid_before"] <= 0 or
            type(release.get("worker_exit_code")) is not int or
            any(release.get(field) is not True for field in (
                "backend_request_state_verified", "graph_gate_released",
                "tool_request_finished", "worker_exit_confirmed",
                "stage_ledger_empty", "host_claim_released", "host_ledger_empty",
            ))):
            raise ValueError("cancel gate lacks verified worker shutdown and ledger release evidence")
        if any(seq > cancel_seq and kind in {"text_delta", "final"} for _, seq, kind in cancel_rows):
            raise ValueError("cancelled request emitted output after cancellation")
        if not any(request_id != cancel_request and rows[0][0] > cancel_epoch and
                   any(kind == "final" for _, _, kind in rows)
                   for request_id, rows in by_request.items()):
            raise ValueError("cancel gate lacks a later recovered epoch")
    elif name == "runtime_placement":
        if identity.get("backend") == STRATA_BACKEND:
            if (native_route is None or native_route.get("backend") != STRATA_BACKEND or
                    observations.get("reported_placement") is not None or
                    observations.get("artifact_sha256") != identity["artifact_sha256"] or
                    observations.get("scope") != "loaded_configuration_and_per_request_routed_decode_experts"):
                raise ValueError("Strata runtime gate needs exact scoped configuration/decode evidence")
            route = {
                "route_id": identity["route_id"], "model_id": native_route["model"],
                "artifact_id": identity["artifact_id"], "artifact_sha256": identity["artifact_sha256"],
                "checkpoint_revision": identity["checkpoint_revision"], "backend": STRATA_BACKEND,
                "expected_placement": identity["placement"], "backend_identity": strata_route_binding(native_route),
            }
            plan = json.loads(_bound_file(base, observations["loaded_plan"]).read_text(encoding="utf-8"))
            request = json.loads(_bound_file(base, observations["request_evidence"]).read_text(encoding="utf-8"))
            validate_strata_profile_plan(plan, route)
            if request.get("execution_plan") != plan:
                raise ValueError("Strata reviewed request belongs to another loaded plan")
            validate_strata_request_evidence(request, route)
            return
        if str(identity["placement"]).startswith("Vulkan_Host+"):
            # Startup override selection precedes mmap/pinned-allocation
            # fallback and cannot prove final expert storage or compute.
            raise ValueError(
                "Vulkan_Host runtime placement cannot be release-qualified from startup logs"
            )
        if (observations.get("reported_placement") != identity["placement"] or
            observations.get("artifact_sha256") != identity["artifact_sha256"]):
            raise ValueError("runtime placement differs from reviewed route")
        startup = _bound_file(base, observations["startup_log"])
        log = startup.read_text(encoding="utf-8", errors="replace")
        import re
        matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", log)
        if len(matches) != 1:
            raise ValueError("runtime placement log has no unambiguous offload report")
        offloaded, total = map(int, matches[0])
        if total <= 0 or offloaded > total:
            raise ValueError("runtime placement log has an invalid offload count")
        placement = identity["placement"].lower()
        if (placement == "cpu" and offloaded != 0 or
            placement != "cpu" and offloaded == 0):
            raise ValueError("runtime offload differs from requested placement")
        if "model loaded" not in log:
            raise ValueError("runtime placement log lacks successful model load")
        cpu_moe_layers = (native_route or {}).get("cpu_moe_layers", 0)
        if cpu_moe_layers:
            if type(cpu_moe_layers) is not int or cpu_moe_layers < 1:
                raise ValueError("runtime CPU expert layer count is invalid")
            overrides = re.findall(
                r"tensor blk\.(\d+)\.(ffn_(?:up|down|gate|gate_up)_(?:ch)?exps\.weight) "
                r"\([^)]*\) buffer type overridden to (CPU|Vulkan_Host|Vulkan\d{1,3})(?:\s|$)",
                log,
            )
            expected = {(str(layer), f"ffn_{tensor}_exps.weight", "CPU")
                        for layer in range(cpu_moe_layers)
                        for tensor in ("up", "down", "gate")}
            if len(overrides) != len(expected) or set(overrides) != expected:
                raise ValueError("runtime placement log lacks exact CPU expert tensor overrides")
    elif name == "checkpoint_lineage":
        revision = identity["checkpoint_revision"]
        if ("unverified" in revision.lower() or not revision or
            observations.get("checkpoint_revision") != revision or
            observations.get("artifact_sha256") != identity["artifact_sha256"] or
            not observations.get("license") or
            not str(observations.get("source_repo", "")).startswith("https://")):
            raise ValueError("checkpoint lineage is not pinned and sourced")
    elif name == "reference_quality":
        cases = observations.get("cases")
        if (observations.get("reference_suite_id") in (None, identity["suite_id"]) or
            not isinstance(cases, list) or len(cases) < 2 or
            not {"en-US", "zh-CN"}.issubset({case.get("language") for case in cases
                                             if isinstance(case, Mapping)}) or
            any(not isinstance(case, Mapping) or
                case.get("task_class") != observations.get("task_class") or
                case.get("passed") is not True or
                case.get("comparison") != "exact_sha256" or
                not isinstance(case.get("reference_sha256"), str) or
                not isinstance(case.get("answer_sha256"), str) or
                not _SHA256_HEX.fullmatch(case["reference_sha256"]) or
                not _SHA256_HEX.fullmatch(case["answer_sha256"]) or
                case["reference_sha256"] != case["answer_sha256"]
                for case in cases)):
            raise ValueError("independent bilingual reference quality evidence is incomplete")
    else:
        raise ValueError(f"unsupported gate: {name}")


def _verify_strata_release_observations(evidence: Mapping[str, Any]) -> None:
    """A scoped functional profile cannot substitute for missing memory/I/O gates."""
    plan = evidence.get("execution_plan", {})
    if (plan.get("three_tier_memory_qualified") is not True or
            plan.get("gpu_aggregate_hard_cap_verified") is not True or
            type(plan.get("file_cache_peak_bytes")) is not int or plan["file_cache_peak_bytes"] < 0):
        raise ValueError("Strata aggregate memory/file-cache/physical SSD gates remain unqualified")
    terminals = evidence.get("terminal_model_metrics", [])
    physical = [item.get("metrics", {}).get("backend_metrics", {}).get(
        "runtime_telemetry", {}).get("physical_ssd_read_bytes") for item in terminals]
    if not physical or any(type(value) is not int or value < 0 for value in physical):
        raise ValueError("Strata physical SSD observations remain unknown")


def load_reviewed_qualification(
    bundle_path: str | Path, *, trusted_keys: Mapping[str, str],
    native_config: Mapping[str, Any],
) -> Qualification:
    """Load one reviewed route only when raw and independent gates still match.

    A signed review attests the *substance* of each independent gate. This
    verifier checks its trust key, all evidence hashes, exact route/condition
    identity, and recomputes the full-Agent profile from raw requests. It does
    not convert an unsigned profiling result into a qualified default.
    """
    bundle_path = Path(bundle_path).resolve(strict=True)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict) or bundle.get("schema") != PROMOTION_SCHEMA:
        raise ValueError("unsupported qualification bundle schema")
    _verify_review_signature(bundle, trusted_keys)
    summary_path = _bound_file(bundle_path.parent, bundle["profile_summary"])
    audit = audit_summary(summary_path)
    if not audit.internally_valid or not audit.trace_verified or not audit.protocol_compliant:
        raise ValueError("raw complete-Agent profile did not pass the batch-one audit: "
                         + "; ".join(audit.errors or ("protocol incomplete",)))
    verify_fixed_suite_cases(audit)
    if audit.measured_successes != audit.measured_attempts:
        raise ValueError("fixed-suite qualification requires every measured request to succeed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("conditions", {}).get("suite_id") == STRUCTURED_READ_URL_SUITE_ID:
        raise ValueError("structured Read URL profiles are not eligible for route promotion")
    # An editable checkout can change without changing the installed wheel
    # version or hardware fingerprint. Old profiles without a digest are not
    # eligible for a native default.
    from vllm_omni.edge.agent.runtime_identity import (
        imported_omni_source_sha256,
        loaded_runtime_sha256,
    )

    profiled_versions = summary.get("conditions", {}).get("runtime_versions", {})
    if (profiled_versions.get("vllm_omni_imported_source_sha256")
            != imported_omni_source_sha256() or
        profiled_versions.get("agent_runtime_identity_sha256")
            != loaded_runtime_sha256()):
        raise ValueError("profiled Agent/Omni source or dependency runtime differs from loaded code")
    stored_profile = summary["routes"][audit.route_id]
    profile_route = stored_profile["route"]
    identity = _identity(profile_route, summary["conditions"], audit.raw_sha256)
    if identity.get("backend") == "external.strata.multimodal.v1":
        raise ValueError("experimental Strata image routes have no reviewed qualification/promotion adapter")
    if bundle.get("identity") != identity:
        raise ValueError("review identity differs from audited profile")

    # The native run index separately binds the exact source configuration,
    # runtime binary, checkpoint lineage and its own raw hash. A route with
    # `lineage_verified=false` cannot be promoted by a later claim-only file.
    index_path = _bound_file(bundle_path.parent, bundle["profile_index"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("automatic_qualification_export") is not False:
        raise ValueError("profile index cannot auto-export qualification")
    matching = [item for item in index.get("results", []) if
                item.get("route_id") == audit.route_id and
                item.get("task_class") == stored_profile["task_class"] and
                item.get("raw_sha256") == audit.raw_sha256 and
                PureWindowsPath(str(item.get("summary", ""))).name == summary_path.name]
    if len(matching) != 1 or not index.get("artifact_provenance", {}).get(audit.route_id, {}).get("lineage_verified"):
        raise ValueError("indexed profile or checkpoint lineage is unverified")
    provenance = index["artifact_provenance"][audit.route_id]
    if identity["backend"] == STRATA_BACKEND:
        if (provenance.get("artifact_manifest_sha256") != identity["artifact_sha256"] or
                provenance.get("checkpoint_revision") != identity["checkpoint_revision"] or
                provenance.get("runtime_manifest_sha256") !=
                summary["conditions"]["runtime_versions"].get("strata_runtime_manifest_sha256")):
            raise ValueError("indexed Strata artifact/runtime provenance differs from profile")
    elif (provenance.get("model_sha256") != identity["artifact_sha256"] or
          provenance.get("checkpoint_revision") != identity["checkpoint_revision"] or
          provenance.get("server_sha256") != summary["conditions"]["runtime_versions"].get("llama_server_sha256")):
        raise ValueError("indexed artifact provenance differs from profile")
    source_config_path = Path(index["source_config"]).resolve(strict=True)
    if _sha256_file(source_config_path) != index.get("source_config_sha256"):
        raise ValueError("profiled native configuration has changed")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    # Review references can be added after a profile. Every other native
    # setting, including limits, route set, and bootstrap policy, is behavior.
    review_only = {"qualification_bundles", "trusted_review_keys", "qualification_file"}
    profiled_behavior = {key: value for key, value in source_config.items()
                         if key not in review_only}
    current_behavior = {key: value for key, value in native_config.items()
                        if key not in review_only}
    if profiled_behavior != current_behavior:
        raise ValueError("current native Agent configuration differs from profiled behavior")
    source_routes = source_config["routes"]
    profiled_route = next((row for row in source_routes if row["route_id"] == audit.route_id), None)
    current_route = next((row for row in native_config["routes"]
                          if row["route_id"] == audit.route_id), None)
    if profiled_route is None or current_route is None or profiled_route != current_route:
        raise ValueError("current native route differs from the profiled route")
    if identity["backend"] == STRATA_BACKEND:
        binding = strata_route_binding(current_route)
        if (profile_route.get("backend_identity") != binding or
                any(provenance.get(key) != value for key, value in binding.items()) or
                current_route["artifact_id"] != identity["artifact_id"] or
                current_route["placement"] != identity["placement"]):
            raise ValueError("current Strata source/runtime/control identity differs from profile")
        # Component cache bounds and sampled system VRAM are not aggregate
        # process peaks; logical file reads are not physical SSD I/O. Current
        # Strata records explicitly lack these gates and cannot be promoted.
        raw_rows = [json.loads(line) for line in audit.raw_jsonl.read_text(encoding="utf-8").splitlines()]
        for row in raw_rows:
            if row.get("record_type") != "request":
                continue
            evidence = (row.get("result") or {}).get("placement_evidence", {})
            _verify_strata_release_observations(evidence)
    else:
        expected_backend = ("external.llamacpp.multimodal.v1" if current_route.get("mmproj_file")
                            else "external.llamacpp.text.v1")
        if (current_route["artifact_id"] != identity["artifact_id"] or
        current_route["model_sha256"] != identity["artifact_sha256"] or
        current_route["model"] != stored_profile["route"]["model_id"] or
        current_route["placement"] != identity["placement"] or
        expected_backend != identity["backend"] or
        current_route["server_sha256"] != provenance["server_sha256"] or
        current_route.get("mmproj_sha256") != provenance.get("mmproj_sha256")):
            raise ValueError("current model/runtime identity differs from profile")

    gates = bundle.get("gates")
    if not isinstance(gates, dict) or set(gates) != REQUIRED_GATES:
        raise ValueError("all five independent qualification gates are required")
    excluded = {summary_path, audit.raw_jsonl, index_path, source_config_path}
    gate_paths: set[Path] = set()
    source_paths: set[Path] = set()
    for gate_name in sorted(REQUIRED_GATES):
        gate_path = _bound_file(bundle_path.parent, gates[gate_name])
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate_path in excluded or gate_path in gate_paths:
            raise ValueError("gate evidence must be separate from profile and other gates")
        gate_paths.add(gate_path)
        if (gate.get("schema") != GATE_SCHEMA or gate.get("gate") != gate_name or
            gate.get("identity") != identity or gate.get("outcome") != "pass"):
            raise ValueError(f"{gate_name}: independent gate identity or outcome differs")
        source_path = _bound_file(gate_path.parent, gate["source"])
        if source_path in excluded or source_path in gate_paths or source_path in source_paths:
            raise ValueError(f"{gate_name}: gate source is not independent")
        source_paths.add(source_path)
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if (source.get("record_type") != f"{gate_name}_evidence_v1" or
            source.get("identity") != identity or
            not isinstance(source.get("observations"), dict) or
            not source["observations"]):
            raise ValueError(f"{gate_name}: raw gate observations are missing or unbound")
        if (gate_name == "reference_quality" and
                source["observations"].get("task_class") != stored_profile["task_class"]):
            raise ValueError("quality review covers another task class")
        _verify_gate_observations(gate_name, source["observations"], identity,
                                  source_path.parent, current_route)
    if gate_paths & source_paths:
        raise ValueError("gate receipt reused as another gate's raw source")

    conditions = summary["conditions"]
    fields = {
        "route_id": identity["route_id"],
        "artifact_id": identity["artifact_id"],
        "task_class": stored_profile["task_class"],
        "suite_id": conditions["suite_id"],
        "environment_fingerprint": conditions["environment_fingerprint"],
        "power_condition": conditions["power_condition"],
        "successes": stored_profile["measured_successes"],
        "attempts": stored_profile["measured_attempts"],
        "answer_latency_s": {name: tuple(values) for name, values in stored_profile["answer_latency_s"].items()},
        "ttft_s": {name: tuple(values) for name, values in stored_profile["ttft_s"].items()},
        "warmups_per_length": stored_profile["warmups_per_length"],
        "sustained_seconds": stored_profile["sustained_seconds"],
        "correctness_pass": bool(stored_profile["correctness_pass"] and stored_profile["e2e_trace_pass"]),
        "tool_safety_pass": stored_profile["tool_safety_pass"],
        "memory_pass": bool(stored_profile["telemetry_present"]),
        "cancel_recovery_pass": True,
        "stability_pass": bool(stored_profile["stability_pass"] and stored_profile["protocol_compliant"]),
        "actual_placement_verified": bool(stored_profile["actual_placement_matches"]
                                          and stored_profile["placement_evidence_present"]),
        "batch_size": 1,
        "concurrency": 1,
        "raw_evidence": f"{audit.raw_jsonl}#sha256={audit.raw_sha256}",
    }
    qualification = Qualification(**fields)
    if not qualification.qualified:
        raise ValueError("reviewed profile still fails qualification: " + "; ".join(qualification.qualification_errors))
    return qualification
