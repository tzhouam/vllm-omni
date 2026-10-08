# SPDX-License-Identifier: Apache-2.0
"""Raw, batch-one whole-Agent profiling for local Omni routes.

This harness invokes a *complete* Agent request through the caller's runner.
An isolated model stage, a host reference answer, or summed stage timings are
recorded as incomplete evidence and cannot be promoted by the qualification
bridge.  The runner must emit visible text deltas and report actual placement.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

LENGTH_BUCKETS = ("short", "medium", "long")
VISIBLE_TEXT_EVENTS = frozenset({"assistant_text_delta", "visible_token"})
_STRATA_BACKEND = "external.strata.text.v1"


@dataclass(frozen=True)
class ProfileRoute:
    route_id: str
    model_id: str
    artifact_id: str
    checkpoint_revision: str
    artifact_sha256: str
    precision: str
    backend: str
    expected_placement: str
    backend_identity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(value for key, value in asdict(self).items() if key != "backend_identity"):
            raise ValueError("exact route, artifact, and placement identity required")


@dataclass(frozen=True)
class AgentCase:
    case_id: str
    task_class: str
    language: str
    length: str
    prompt: str
    reference: Any = None
    input_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id or not self.task_class or not self.prompt:
            raise ValueError("case identity, class, and prompt required")
        if self.length not in LENGTH_BUCKETS:
            raise ValueError("length must be short, medium, or long")
        if self.input_tokens is not None and self.input_tokens < 0:
            raise ValueError("input_tokens must be nonnegative")


@dataclass(frozen=True)
class ProfileConditions:
    hardware_id: str
    os_version: str
    driver_versions: Mapping[str, str]
    runtime_versions: Mapping[str, str]
    power_condition: str
    suite_id: str
    environment_fingerprint: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not all((self.hardware_id, self.os_version, self.power_condition,
                    self.suite_id, self.environment_fingerprint)):
            raise ValueError("hardware, OS, power, suite, and environment are required")


@dataclass(frozen=True)
class AgentRunResult:
    final_answer: str | None
    complete_agent_trace: bool
    model_id: str
    artifact_id: str
    actual_placement: str | None
    backend: str
    placement_evidence: Mapping[str, Any] = field(default_factory=dict)
    tool_decisions: Sequence[Mapping[str, Any]] = ()
    trace_scope: str = "agent_e2e"
    output_tokens: int | None = None


@dataclass(frozen=True)
class Evaluation:
    success: bool
    quality_pass: bool
    quality_score: float | None
    tool_safe: bool
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Preparation:
    """Evidence that a route was unloaded before the timed load."""

    cold_start_confirmed: bool
    artifact_id: str
    actual_placement: str | None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileConfig:
    warmups_per_length: int = 2
    measured_per_length: int = 20
    endurance_seconds: float = 1800.0
    telemetry_interval_seconds: float = 0.1

    def __post_init__(self) -> None:
        if self.warmups_per_length < 0 or self.measured_per_length < 1:
            raise ValueError("invalid warmup or measured request count")
        if self.endurance_seconds < 0 or self.telemetry_interval_seconds <= 0:
            raise ValueError("invalid endurance or telemetry interval")

    @property
    def protocol_minimums(self) -> bool:
        return (self.warmups_per_length >= 1
                and self.measured_per_length >= 20
                and self.endurance_seconds >= 1800)


@dataclass(frozen=True)
class RouteProfile:
    route: ProfileRoute
    task_class: str
    measured_attempts: int
    measured_successes: int
    answer_latency_s: Mapping[str, tuple[float, ...]]
    ttft_s: Mapping[str, tuple[float, ...]]
    answer_p50_s: Mapping[str, float | None]
    answer_p95_s: Mapping[str, float | None]
    ttft_p50_s: Mapping[str, float | None]
    ttft_p95_s: Mapping[str, float | None]
    warmups_per_length: Mapping[str, int]
    sustained_seconds: float
    endurance_wall_seconds: float
    endurance_requests: int
    correctness_pass: bool
    tool_safety_pass: bool
    stability_pass: bool
    e2e_trace_pass: bool
    actual_placement_matches: bool
    placement_evidence_present: bool
    telemetry_present: bool
    cold_start_s: float | None
    protocol_compliant: bool

    def qualification_fields(
        self,
        *,
        conditions: ProfileConditions,
        raw_evidence: str,
        task_class: str,
        memory_pass: bool = False,
        cancel_recovery_pass: bool = False,
        actual_placement_verified: bool = False,
    ) -> dict[str, Any]:
        """Bridge to ``router.Qualification`` without manufacturing gates.

        Memory admission, cancellation/recovery, and placement verification
        need separate evidence.  Their caller-supplied values default to false.
        """
        if task_class != self.task_class:
            raise ValueError("profile task class does not match qualification")
        return {
            "route_id": self.route.route_id,
            "artifact_id": self.route.artifact_id,
            "task_class": task_class,
            "suite_id": conditions.suite_id,
            "environment_fingerprint": conditions.environment_fingerprint,
            "power_condition": conditions.power_condition,
            "successes": self.measured_successes,
            "attempts": self.measured_attempts,
            "answer_latency_s": self.answer_latency_s,
            "ttft_s": self.ttft_s,
            "warmups_per_length": self.warmups_per_length,
            "sustained_seconds": self.sustained_seconds,
            "correctness_pass": self.correctness_pass and self.e2e_trace_pass,
            "tool_safety_pass": self.tool_safety_pass,
            "memory_pass": bool(memory_pass and self.telemetry_present),
            "cancel_recovery_pass": bool(cancel_recovery_pass),
            "stability_pass": self.stability_pass and self.protocol_compliant,
            "actual_placement_verified": bool(
                actual_placement_verified and self.actual_placement_matches
                and self.placement_evidence_present
                and self.route.backend != _STRATA_BACKEND
            ),
            "batch_size": 1,
            "concurrency": 1,
            "raw_evidence": raw_evidence,
        }


@dataclass(frozen=True)
class ProfileSummary:
    run_id: str
    run_directory: Path
    raw_jsonl: Path
    raw_sha256: str
    conditions: ProfileConditions
    routes: Mapping[str, RouteProfile]


EmitEvent = Callable[[str, Any], None]
AgentRunner = Callable[[ProfileRoute, AgentCase, EmitEvent], Awaitable[AgentRunResult]]
Evaluator = Callable[[AgentCase, AgentRunResult], Evaluation | Awaitable[Evaluation]]
Telemetry = Callable[[], Mapping[str, Any]]
Preparer = Callable[[ProfileRoute], Preparation | Awaitable[Preparation]]
BeforeRequest = Callable[[ProfileRoute, AgentCase, str, int], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"profile field is not JSON-compatible: {type(value).__name__}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    # Sampled peaks are lower bounds on true instantaneous peaks.
    result["peak_is_sampled_lower_bound"] = True
    if any("native_process_gpu_memory" in sample for sample in samples):
        from vllm_omni.edge.agent.placement import native_gpu_sample_summary

        result["native_process_gpu_memory"] = native_gpu_sample_summary(samples)
    return result


def _nearest_rank(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


async def _one_request(
    *,
    run_id: str,
    route: ProfileRoute,
    case: AgentCase,
    phase: str,
    repetition: int,
    runner: AgentRunner,
    evaluator: Evaluator,
    telemetry: Telemetry | None,
    telemetry_interval_s: float,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    telemetry_samples: list[dict[str, Any]] = []
    telemetry_errors: list[str] = []
    event_lock = threading.Lock()
    stop_polling = asyncio.Event()
    first_visible_ns: int | None = None
    closed = False
    started_ns = time.perf_counter_ns()
    started_at = _utc_now()

    def emit(kind: str, payload: Any = None) -> None:
        nonlocal first_visible_ns
        with event_lock:
            if closed:
                raise RuntimeError("Agent emitted output after request completion")
            now_ns = time.perf_counter_ns()
            if kind in VISIBLE_TEXT_EVENTS and first_visible_ns is None:
                first_visible_ns = now_ns
            events.append({
                "kind": kind,
                "offset_s": (now_ns - started_ns) / 1e9,
                "payload": _json_safe(payload),
            })

    async def poll_telemetry() -> None:
        if telemetry is None:
            return
        while not stop_polling.is_set():
            try:
                snapshot = await asyncio.to_thread(telemetry)
                telemetry_samples.append({
                    "offset_s": (time.perf_counter_ns() - started_ns) / 1e9,
                    **_json_safe(snapshot),
                })
            except Exception as exc:
                telemetry_errors.append(f"{type(exc).__name__}: {exc}")
            try:
                await asyncio.wait_for(stop_polling.wait(), telemetry_interval_s)
            except TimeoutError:
                pass

    polling_task = asyncio.create_task(poll_telemetry())
    await asyncio.sleep(0)
    result: AgentRunResult | None = None
    evaluation: Evaluation | None = None
    error: str | None = None
    try:
        result = await runner(route, case, emit)
        if not isinstance(result, AgentRunResult):
            raise TypeError("runner returned the wrong result type")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if not isinstance(result, AgentRunResult):
            result = None
    finally:
        with event_lock:
            closed = True
        ended_ns = time.perf_counter_ns()
        stop_polling.set()
        await polling_task
    if result is not None and error is None:
        try:
            evaluated = evaluator(case, result)
            evaluation = await evaluated if inspect.isawaitable(evaluated) else evaluated
            if not isinstance(evaluation, Evaluation):
                raise TypeError("evaluator returned the wrong result type")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    e2e_complete = bool(
        result is not None and result.trace_scope == "agent_e2e"
        and result.complete_agent_trace and result.final_answer is not None
    )
    # For Strata this checks configuration and scoped decode evidence; it
    # does not manufacture a whole-model actual placement string.
    if result is not None and route.backend == _STRATA_BACKEND:
        from vllm_omni.edge.agent.placement import result_placement_matches

        placement_matches = result_placement_matches(asdict(result), asdict(route))
    else:
        placement_matches = bool(
            result is not None and result.model_id == route.model_id and result.artifact_id == route.artifact_id
            and result.actual_placement == route.expected_placement and result.backend == route.backend
        )
    return {
        "record_type": "request",
        "run_id": run_id,
        "phase": phase,
        "repetition": repetition,
        "route_id": route.route_id,
        "case": asdict(case),
        "started_at": started_at,
        "batch_size": 1,
        "concurrency": 1,
        "ttft_s": ((first_visible_ns - started_ns) / 1e9
                   if first_visible_ns is not None else None),
        "answer_latency_s": (ended_ns - started_ns) / 1e9,
        "events": events,
        "result": asdict(result) if result is not None else None,
        "evaluation": asdict(evaluation) if evaluation is not None else None,
        "e2e_complete": e2e_complete,
        "placement_matches": placement_matches,
        "telemetry": {
            "raw_samples": telemetry_samples,
            "errors": telemetry_errors,
            "summary": _telemetry_summary(telemetry_samples),
        },
        "error": error,
    }


def _route_profile(
    route: ProfileRoute,
    samples: Sequence[Mapping[str, Any]],
    *,
    cold_start_s: float | None,
    sustained_seconds: float,
    endurance_wall_seconds: float,
    config: ProfileConfig,
) -> RouteProfile:
    measured = [row for row in samples if row["phase"] == "measured"]
    warmups = [row for row in samples if row["phase"] == "warmup"]
    endurance = [row for row in samples if row["phase"] == "endurance"]
    by_length = {bucket: [row for row in measured if row["case"]["length"] == bucket]
                 for bucket in LENGTH_BUCKETS}
    answer = {bucket: tuple(row["answer_latency_s"] for row in rows)
              for bucket, rows in by_length.items()}
    first = {bucket: tuple(row["ttft_s"] for row in rows if row["ttft_s"] is not None)
             for bucket, rows in by_length.items()}
    answer_p50 = {bucket: _nearest_rank(answer[bucket], .5) for bucket in LENGTH_BUCKETS}
    answer_p95 = {bucket: _nearest_rank(answer[bucket], .95) for bucket in LENGTH_BUCKETS}
    first_p50 = {bucket: _nearest_rank(first[bucket], .5) for bucket in LENGTH_BUCKETS}
    first_p95 = {bucket: _nearest_rank(first[bucket], .95) for bucket in LENGTH_BUCKETS}
    warmup_counts = {bucket: sum(row["case"]["length"] == bucket for row in warmups)
                     for bucket in LENGTH_BUCKETS}
    success = sum(bool(row["e2e_complete"] and row["placement_matches"]
                       and row["evaluation"] and row["evaluation"]["success"])
                  for row in measured)
    all_rows = [*measured, *endurance]
    e2e = all(bool(row["e2e_complete"]) for row in all_rows)
    placement_matches = all(bool(row["placement_matches"]) for row in all_rows)
    placement_evidence = all(bool(row["result"] and
                                  row["result"]["placement_evidence"])
                             for row in all_rows)
    correctness = all(bool(row["evaluation"] and row["evaluation"]["quality_pass"]
                           and row["evaluation"]["success"])
                      for row in measured)
    tool_safety = all(bool(row["evaluation"] and row["evaluation"]["tool_safe"])
                      for row in all_rows)
    stable = bool(endurance) and sustained_seconds >= config.endurance_seconds and all(
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
    compliant = bool(config.protocol_minimums and sample_counts_ok and
                     sustained_seconds >= 1800 and e2e and placement_matches
                     and telemetry_present and cold_start_s is not None)
    return RouteProfile(
        route=route,
        task_class=measured[0]["case"]["task_class"] if measured else "",
        measured_attempts=len(measured),
        measured_successes=success,
        answer_latency_s=answer,
        ttft_s=first,
        answer_p50_s=answer_p50,
        answer_p95_s=answer_p95,
        ttft_p50_s=first_p50,
        ttft_p95_s=first_p95,
        warmups_per_length=warmup_counts,
        sustained_seconds=sustained_seconds,
        endurance_wall_seconds=endurance_wall_seconds,
        endurance_requests=len(endurance),
        correctness_pass=correctness,
        tool_safety_pass=tool_safety,
        stability_pass=stable,
        e2e_trace_pass=e2e,
        actual_placement_matches=placement_matches,
        placement_evidence_present=placement_evidence,
        telemetry_present=telemetry_present,
        cold_start_s=cold_start_s,
        protocol_compliant=compliant,
    )


async def run_profile(
    *,
    routes: Sequence[ProfileRoute],
    cases_by_length: Mapping[str, Sequence[AgentCase]],
    runner: AgentRunner,
    evaluator: Evaluator,
    conditions: ProfileConditions,
    output_dir: str | Path,
    config: ProfileConfig = ProfileConfig(),
    telemetry: Telemetry | None = None,
    prepare: Preparer | None = None,
    before_request: BeforeRequest | None = None,
) -> ProfileSummary:
    """Run each complete route sequentially at batch=1, one active request.

    Identical cases and repetition order are used for every route.  A route's
    warmups, measured samples, and 30-minute sequential endurance occupy one
    contiguous block to avoid accidental cold reload on every paired sample.
    """
    if not routes or len({route.route_id for route in routes}) != len(routes):
        raise ValueError("at least one unique route is required")
    if set(cases_by_length) != set(LENGTH_BUCKETS):
        raise ValueError("exactly three input lengths are required")
    for bucket in LENGTH_BUCKETS:
        cases = cases_by_length[bucket]
        if not cases or any(case.length != bucket for case in cases):
            raise ValueError(f"{bucket} must contain matching cases")
    all_cases = [case for bucket in LENGTH_BUCKETS for case in cases_by_length[bucket]]
    if len({case.case_id for case in all_cases}) != len(all_cases):
        raise ValueError("case IDs must be unique across lengths")
    if len({case.task_class for case in all_cases}) != 1:
        raise ValueError("each profile run must cover one task class")
    if any(len(cases_by_length[bucket]) > config.measured_per_length
           for bucket in LENGTH_BUCKETS):
        raise ValueError("every case needs at least one measured repetition")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_directory = Path(output_dir).resolve() / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    raw_path = run_directory / "samples.jsonl"
    route_profiles: dict[str, RouteProfile] = {}

    def write_line(handle, row: Mapping[str, Any]) -> None:
        handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    async def run_one(route: ProfileRoute, case: AgentCase, phase: str,
                      repetition: int) -> dict[str, Any]:
        # Fixture setup (for example an isolated encrypted-memory seed) is
        # outside the answer timer. It is recorded separately, and must not
        # load or execute the model. The Agent request itself is still timed
        # from submission through its final event.
        setup: Mapping[str, Any] = {}
        if before_request is not None:
            prepared = before_request(route, case, phase, repetition)
            setup = await prepared if inspect.isawaitable(prepared) else prepared
            if not isinstance(setup, Mapping):
                raise TypeError("before_request must return setup evidence")
        row = await _one_request(
            run_id=run_id, route=route, case=case, phase=phase,
            repetition=repetition, runner=runner, evaluator=evaluator,
            telemetry=telemetry,
            telemetry_interval_s=config.telemetry_interval_seconds,
        )
        row["fixture_setup"] = _json_safe(setup)
        return row

    with raw_path.open("w", encoding="utf-8") as handle:
        write_line(handle, {
            "record_type": "manifest", "run_id": run_id,
            "started_at": _utc_now(), "conditions": asdict(conditions),
            "config": asdict(config), "routes": [asdict(route) for route in routes],
            "batch_size": 1, "concurrency": 1,
            "scope": "complete_agent_request",
        })
        for route in routes:
            route_samples: list[dict[str, Any]] = []
            cold_start_s: float | None = None
            if prepare is not None:
                began = time.perf_counter_ns()
                prepared = prepare(route)
                if inspect.isawaitable(prepared):
                    prepared = await prepared
                prepare_wall_s = (time.perf_counter_ns() - began) / 1e9
                if not isinstance(prepared, Preparation):
                    raise TypeError("prepare must return Preparation evidence")
                if route.backend == _STRATA_BACKEND:
                    from vllm_omni.edge.agent.placement import preparation_placement_matches

                    prepared_matches = preparation_placement_matches(asdict(prepared), asdict(route))
                else:
                    prepared_matches = bool(prepared.cold_start_confirmed and prepared.artifact_id == route.artifact_id
                                            and prepared.actual_placement == route.expected_placement)
                if prepared_matches:
                    cold_start_s = prepare_wall_s
                write_line(handle, {
                    "record_type": "route_prepare", "run_id": run_id,
                    "route_id": route.route_id, "cold_start_s": cold_start_s,
                    "prepare_wall_s": prepare_wall_s,
                    "preparation": asdict(prepared),
                    "finished_at": _utc_now(),
                })
            for bucket in LENGTH_BUCKETS:
                cases = cases_by_length[bucket]
                for repetition in range(config.warmups_per_length):
                    case = cases[repetition % len(cases)]
                    row = await run_one(route, case, "warmup", repetition)
                    write_line(handle, row)
                    route_samples.append(row)
                for repetition in range(config.measured_per_length):
                    case = cases[repetition % len(cases)]
                    row = await run_one(route, case, "measured", repetition)
                    write_line(handle, row)
                    route_samples.append(row)
            endurance_started = time.perf_counter()
            active_endurance_seconds = 0.0
            endurance_count = 0
            while active_endurance_seconds < config.endurance_seconds:
                case = all_cases[endurance_count % len(all_cases)]
                row = await run_one(route, case, "endurance", endurance_count)
                write_line(handle, row)
                route_samples.append(row)
                active_endurance_seconds += row["answer_latency_s"]
                endurance_count += 1
            endurance_wall_seconds = time.perf_counter() - endurance_started
            route_profiles[route.route_id] = _route_profile(
                route, route_samples, cold_start_s=cold_start_s,
                sustained_seconds=active_endurance_seconds,
                endurance_wall_seconds=endurance_wall_seconds, config=config,
            )
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    summary = ProfileSummary(
        run_id=run_id,
        run_directory=run_directory,
        raw_jsonl=raw_path,
        raw_sha256=raw_sha256,
        conditions=conditions,
        routes=route_profiles,
    )
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(asdict(summary)), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8",
    )
    return summary
