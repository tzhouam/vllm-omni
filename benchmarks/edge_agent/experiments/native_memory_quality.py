# SPDX-License-Identifier: Apache-2.0
"""Private, cross-session bilingual memory-quality experiment for native Gemma.

This is deliberately separate from the fixed performance profile.  It runs
sequential batch-one Agent turns through the ordinary controller, reopens the
same encrypted store in a second app session, and checks both answers and the
source-event lineage of the final events.  Its raw records are private and it
never creates or signs a router qualification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import secrets
import shutil
import sys
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from vllm_omni.edge.agent.controller import _RECALL_KINDS
from vllm_omni.edge.agent.memory import MemoryEvent


SUITE_ID = "edge-agent-bilingual-memory-provenance-v1"
PRIVATE_RESULTS = Path(__file__).resolve().parents[1] / "results"
_BAD_TURN_KINDS = frozenset({
    "error", "refusal", "cancelled", "approval_required", "tool_proposed", "tool_result",
})


def _sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


@dataclass(frozen=True)
class MemoryQualityCase:
    case_id: str
    language: str
    source_prompt: str
    distractor_prompt: str
    recall_prompt: str
    reference: str
    distractor_reference: str


def build_cases(seed: bytes) -> tuple[MemoryQualityCase, ...]:
    """Generate unpublished nonce-like answers; no public fixed code can pass.

    Both aliases are in each source so the English/Chinese query uses the same
    local observation.  Similar distractor names force disambiguation rather
    than a generic 'remember the identifier' success.
    """
    if len(seed) < 16:
        raise ValueError("fixture seed must contain at least 16 bytes")
    pairs = (
        ("en_en", "en-US", "Cedar atlas", "雪松图册", "Cedar calendar", "雪松日历", "en", "en"),
        ("zh_zh", "zh-CN", "Lotus ferry", "莲花渡轮", "Lotus station", "莲花车站", "zh", "zh"),
        ("en_zh", "zh-CN", "Amber compass", "琥珀罗盘", "Amber notebook", "琥珀笔记", "en", "zh"),
        ("zh_en", "en-US", "Heron parcel", "苍鹭包裹", "Heron ticket", "苍鹭车票", "zh", "en"),
    )
    cases: list[MemoryQualityCase] = []
    for case_id, language, en_name, zh_name, en_other, zh_other, source_lang, query_lang in pairs:
        reference = "ID" + _sha(seed + f"{case_id}:target".encode())[:12].upper()
        distractor = "ID" + _sha(seed + f"{case_id}:distractor".encode())[:12].upper()
        if source_lang == "en":
            source = (
                f"Remember this observed local-memory fact: {en_name} ({zh_name}) "
                f"has identifier {reference}. Reply only noted."
            )
            other = (
                f"Remember this separate observed local-memory fact: {en_other} "
                f"({zh_other}) has identifier {distractor}. Reply only noted."
            )
        else:
            source = (
                f"请记住这条本地记忆观察：{zh_name}（{en_name}）的标识符是 {reference}。只回答 noted。"
            )
            other = (
                f"请记住另一条本地记忆观察：{zh_other}（{en_other}）的标识符是 "
                f"{distractor}。只回答 noted。"
            )
        if query_lang == "en":
            recall = (
                f"Recall from local memory the identifier for {en_name} ({zh_name}). "
                "Use only the exact matching observation, not a similar item. "
                "Answer with only its identifier; if that observation is absent, answer UNKNOWN."
            )
        else:
            recall = (
                f"请从本地记忆中回忆 {zh_name}（{en_name}）的标识符。只依据完全匹配的观察，"
                "不要使用相似条目。只回答标识符；如果该观察不存在，只回答 UNKNOWN。"
            )
        cases.append(MemoryQualityCase(
            case_id, language, source, other, recall, reference, distractor,
        ))
    if len({code for case in cases for code in (case.reference, case.distractor_reference)}) != 8:
        raise AssertionError("fixture identifiers collided")
    return tuple(cases)


class TurnRecorder:
    """Capture exactly one complete controller turn at a time."""

    def __init__(self, controller: Any, *, route_id: str, placement: str,
                 timeout_s: float) -> None:
        self.controller = controller
        self.route_id = route_id
        self.placement = placement
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._active: list[dict[str, Any]] | None = None
        controller.add_listener(self._listen)

    def _listen(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            if self._active is not None:
                self._active.append(dict(event))
        if event.get("kind") == "approval_required":
            # An offline quality test cannot impersonate a human approver.
            self.controller.reject(str(event.get("payload", {}).get("challenge_id", "")))

    def run(self, prompt: str) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        with self._lock:
            if self._active is not None:
                raise RuntimeError("another quality request is active")
            self._active = events
        try:
            future = self.controller.submit(prompt)
            try:
                answer = future.result(timeout=self.timeout_s)
            except Exception:
                self.controller.cancel()
                raise
            if not self.controller._turn_done.wait(self.timeout_s):
                raise TimeoutError("Agent turn did not release its single-request gate")
        finally:
            with self._lock:
                self._active = None
        request_ids = {event.get("request_id") for event in events}
        request_id = next(iter(request_ids)) if len(request_ids) == 1 else None
        rows = (self.controller.memory.iter_events(
            session_id=self.controller.session_id, request_id=request_id,
        ) if request_id else [])
        observations = [row for row in rows if row.kind == "user_observation"]
        finals = [row for row in rows if row.kind == "final"]
        route_events = [event for event in events if event.get("kind") == "route"]
        trace_ok = (
            request_id is not None
            and [event.get("seq") for event in events] == list(range(1, len(events) + 1))
            and all(event.get("session_id") == self.controller.session_id for event in events)
            and len(observations) == len(finals) == len(route_events) == 1
            and events[0]["kind"] == "user_observation"
            and events[-1]["kind"] == "final"
            and not any(event.get("kind") in _BAD_TURN_KINDS for event in events)
            and route_events[0]["payload"].get("route_id") == self.route_id
            and route_events[0]["payload"].get("actual_placement") == self.placement
            and events[-1]["payload"].get("answer") == answer
            and observations[0].payload == {"text": prompt}
            and finals[0].payload.get("answer") == answer
        )
        return {
            "session_id": self.controller.session_id,
            "request_id": request_id,
            "answer": answer,
            "answer_sha256": _sha(answer or ""),
            "prompt": prompt,
            "prompt_sha256": _sha(prompt),
            "trace_ok": trace_ok,
            "observation_event_id": observations[0].event_id if observations else None,
            "final_event_id": finals[0].event_id if finals else None,
            "final_derived_from": list(finals[0].derived_from) if finals else [],
            "events": events,
        }


def assess_case(
    case: MemoryQualityCase, *, source: MemoryEvent, distractor: MemoryEvent,
    recall: Mapping[str, Any], after_delete: Mapping[str, Any],
    before_ids: list[str], deleted_count: int, source_absent: bool,
    distractor_retained: bool, derived_final_absent: bool,
    answer_index_absent: bool, after_ids: list[str],
) -> dict[str, bool]:
    """Task-quality checks independent of the model's own success claims."""
    source_id, distractor_id = source.event_id, distractor.event_id
    checks = {
        "cross_session": (
            source.session_id != recall.get("session_id")
            and distractor.session_id != recall.get("session_id")
        ),
        "source_is_prior_user_observation": (
            source.kind == "user_observation" and source.source == "user"
            and source.payload == {"text": case.source_prompt}
        ),
        "distractor_is_prior_user_observation": (
            distractor.kind == "user_observation" and distractor.source == "user"
            and distractor.payload == {"text": case.distractor_prompt}
        ),
        "source_and_distractor_retrieved": (
            source_id in before_ids and distractor_id in before_ids
            and source_id in recall.get("final_derived_from", ())
            and distractor_id in recall.get("final_derived_from", ())
        ),
        "ordered_complete_recall": recall.get("trace_ok") is True,
        "reference_answer_exact": recall.get("answer") == case.reference
            and recall.get("answer") != case.distractor_reference,
        "deletion_cascaded": (
            deleted_count >= 2 and source_absent and distractor_retained
            and derived_final_absent and answer_index_absent
            and source_id not in after_ids
        ),
        "ordered_complete_after_delete": after_delete.get("trace_ok") is True,
        "after_delete_unknown": (
            after_delete.get("answer") == "UNKNOWN"
            and source_id not in after_delete.get("final_derived_from", ())
            and distractor_id in after_delete.get("final_derived_from", ())
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def run_suite(
    cases: tuple[MemoryQualityCase, ...], *,
    controller_factory: Callable[[str], Any], timeout_s: float,
    on_record: Callable[[Mapping[str, Any]], None],
) -> dict[str, Any]:
    """Exercise real controller and encrypted store; factory may be native or test.

    ``controller_factory`` reopens the *same* memory file but creates an
    independent controller/session for each phase.  Each turn completes before
    the next begins; there is no throughput sweep or concurrent request.
    """
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("nonempty, unique quality cases required")
    sources: dict[str, tuple[MemoryEvent, MemoryEvent]] = {}
    seed_controller = controller_factory("seed")
    try:
        if not seed_controller.bootstrap_route_id or len(seed_controller.routes) != 1:
            raise RuntimeError("quality run requires one explicit experimental route")
        route = seed_controller.routes[0]
        recorder = TurnRecorder(seed_controller, route_id=route.route_id,
                                placement=route.placement, timeout_s=timeout_s)
        for case in cases:
            seeded: list[MemoryEvent] = []
            for role, prompt in (("source", case.source_prompt),
                                 ("distractor", case.distractor_prompt)):
                turn = recorder.run(prompt)
                if not turn["trace_ok"] or not turn["observation_event_id"]:
                    raise RuntimeError(f"{case.case_id}: {role} observation has an incomplete Agent trace")
                event = seed_controller.memory.get_event(turn["observation_event_id"])
                if event is None:
                    raise RuntimeError("seed observation disappeared")
                seeded.append(event)
                on_record({"record_type": "seed_turn", "case_id": case.case_id,
                           "role": role, "turn": turn, "source_event_id": event.event_id})
            sources[case.case_id] = (seeded[0], seeded[1])
        seed_session = seed_controller.session_id
    finally:
        seed_controller.close()

    replay_controller = controller_factory("replay")
    try:
        if replay_controller.session_id == seed_session:
            raise RuntimeError("controller factory reused the source session")
        route = replay_controller.routes[0]
        recorder = TurnRecorder(replay_controller, route_id=route.route_id,
                                placement=route.placement, timeout_s=timeout_s)
        assessments: list[dict[str, Any]] = []
        for case in cases:
            source, distractor = sources[case.case_id]
            if (replay_controller.memory.get_event(source.event_id) is None
                    or replay_controller.memory.get_event(distractor.event_id) is None):
                raise RuntimeError("encrypted source did not survive app-session restart")
            before_ids = [match.event.event_id for match in replay_controller.memory.search(
                case.recall_prompt, kinds=_RECALL_KINDS,
                limit=replay_controller.limits.max_memory_matches,
            )]
            recall = recorder.run(case.recall_prompt)
            deleted_count = replay_controller.memory.delete_event(source.event_id)
            source_absent = replay_controller.memory.get_event(source.event_id) is None
            distractor_retained = replay_controller.memory.get_event(distractor.event_id) is not None
            derived_final_absent = (
                bool(recall["final_event_id"])
                and replay_controller.memory.get_event(recall["final_event_id"]) is None
            )
            answer_index_absent = not replay_controller.memory.search(
                case.reference, kinds=_RECALL_KINDS,
            )
            after_ids = [match.event.event_id for match in replay_controller.memory.search(
                case.recall_prompt, kinds=_RECALL_KINDS,
                limit=replay_controller.limits.max_memory_matches,
            )]
            after_delete = recorder.run(case.recall_prompt)
            checks = assess_case(
                case, source=source, distractor=distractor, recall=recall,
                after_delete=after_delete, before_ids=before_ids,
                deleted_count=deleted_count, source_absent=source_absent,
                distractor_retained=distractor_retained,
                derived_final_absent=derived_final_absent,
                answer_index_absent=answer_index_absent, after_ids=after_ids,
            )
            record = {
                "record_type": "quality_case", "case_id": case.case_id,
                "language": case.language, "source_session_id": source.session_id,
                "source_event_id": source.event_id,
                "distractor_event_id": distractor.event_id,
                "reference": case.reference,
                "reference_sha256": _sha(case.reference),
                "distractor_reference": case.distractor_reference,
                "distractor_reference_sha256": _sha(case.distractor_reference),
                "before_retrieved_event_ids": before_ids,
                "recall": recall,
                "deletion": {
                    "deleted_count": deleted_count,
                    "source_absent": source_absent,
                    "distractor_retained": distractor_retained,
                    "derived_final_absent": derived_final_absent,
                    "answer_index_absent": answer_index_absent,
                    "after_retrieved_event_ids": after_ids,
                },
                "after_delete": after_delete,
                "checks": checks,
            }
            on_record(record)
            assessments.append({"case_id": case.case_id, "language": case.language,
                                "checks": checks})
        return {
            "suite_id": SUITE_ID, "batch_size": 1, "concurrency": 1,
            "scope": "cross_session_bilingual_memory_quality",
            "cases": len(assessments),
            "passed": sum(item["checks"]["passed"] for item in assessments),
            "case_assessments": assessments,
            "review_status": "unreviewed_private_experiment",
            "qualifies_default": False,
        }
    finally:
        replay_controller.close()


def run_native(config_path: Path, lineage_path: Path, *, timeout_s: float,
               seed: bytes | None = None,
               profile_index: Path | None = None) -> Path:
    if sys.platform != "win32":
        raise RuntimeError("native memory-quality execution requires Windows Python")
    from benchmarks.edge_agent.experiments.profile_binding import bind_profile_to_live
    from benchmarks.edge_agent.native_profile import load_profile_routes
    from vllm_omni.edge.agent.native_app import build_controller
    from vllm_omni.edge.agent.runtime_identity import loaded_runtime_sha256

    config_bytes = config_path.read_bytes()
    lineage_bytes = lineage_path.read_bytes()
    config = json.loads(config_bytes)
    lineage = json.loads(lineage_bytes)
    routes, provenance = load_profile_routes(config, lineage)
    if len(routes) != 1 or "gemma" not in routes[0].model_id.casefold():
        raise ValueError("this experiment requires one pinned Gemma route")
    route = routes[0]
    if not provenance[route.route_id]["lineage_verified"]:
        raise ValueError("Gemma checkpoint lineage has not been verified")
    fixture_seed = seed if seed is not None else secrets.token_bytes(32)
    cases = build_cases(fixture_seed)
    PRIVATE_RESULTS.mkdir(parents=True, exist_ok=True)
    run_dir = PRIVATE_RESULTS / ("memory_quality_" + uuid.uuid4().hex)
    run_dir.mkdir(mode=0o700)
    raw_path = run_dir / "raw.jsonl"
    hardware_by_phase: dict[str, Any] = {}
    summary: dict[str, Any] | None = None
    failure: Exception | None = None
    profile_binding: dict[str, Any] | None = None
    runtime_sha = loaded_runtime_sha256()
    with tempfile.TemporaryDirectory(prefix="omni-agent-memory-quality-") as temporary:
        local = Path(temporary)
        memory_path = local / "memory.sqlite"

        def factory(phase: str) -> Any:
            trial = copy.deepcopy(config)
            trial["routes"] = [dict(trial["routes"][0])]
            trial["routes"][0]["log_file"] = str(local / f"{phase}-server.log")
            trial["memory_file"] = str(memory_path)
            trial["qualification_file"] = None
            trial["qualification_bundles"] = []
            trial["trusted_review_keys"] = {}
            trial["qualification_suite_id"] = SUITE_ID
            trial["experimental_bootstrap_route_id"] = route.route_id
            trial["limits"] = {**trial.get("limits", {}), "max_answer_tokens": 128}
            path = local / f"{phase}-config.json"
            path.write_text(json.dumps(trial, ensure_ascii=False, indent=2), encoding="utf-8")
            controller, hardware = build_controller(path)
            hardware_by_phase[phase] = hardware
            return controller

        # A bound quality run must authenticate the completed profile against
        # this live controller before submitting its first neural request.
        # Keep the prebuilt seed controller for the suite, so there is no
        # unrecorded extra session or source observation.
        seed_controller: Any | None = None
        seed_handed_off = False
        preflight_error: Exception | None = None
        if profile_index is not None:
            try:
                seed_controller = factory("seed")
                profile_binding = bind_profile_to_live(
                    profile_index, source_config_bytes=config_bytes,
                    route_id=route.route_id, hardware=hardware_by_phase["seed"],
                )
            except Exception as exc:
                if seed_controller is not None:
                    seed_controller.close()
                    seed_controller = None
                preflight_error = exc

        def suite_factory(phase: str) -> Any:
            nonlocal seed_handed_off
            if phase == "seed" and seed_controller is not None:
                seed_handed_off = True
                return seed_controller
            controller = factory(phase)
            if profile_index is not None:
                try:
                    replay_binding = bind_profile_to_live(
                        profile_index, source_config_bytes=config_bytes,
                        route_id=route.route_id, hardware=hardware_by_phase[phase],
                    )
                    if replay_binding != profile_binding:
                        raise RuntimeError("profile binding changed between app sessions")
                except Exception:
                    controller.close()
                    raise
            return controller

        with raw_path.open("x", encoding="utf-8") as output:
            def record(item: Mapping[str, Any]) -> None:
                output.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()

            record({
                "record_type": "manifest", "suite_id": SUITE_ID,
                "review_status": "unreviewed_private_experiment",
                "qualifies_default": False, "batch_size": 1, "concurrency": 1,
                "route": asdict(route), "lineage": provenance[route.route_id],
                "native_config_sha256": _sha(config_bytes),
                "lineage_sha256": _sha(lineage_bytes),
                "loaded_runtime_sha256": runtime_sha,
                "profile_binding_requested": profile_index is not None,
                "profile_binding": profile_binding,
                "fixture_seed_hex": fixture_seed.hex(),
                "case_ids": [case.case_id for case in cases],
            })
            try:
                if preflight_error is not None:
                    raise preflight_error
                summary = run_suite(cases, controller_factory=suite_factory,
                                    timeout_s=timeout_s, on_record=record)
                record({"record_type": "summary", **summary,
                        "hardware_by_phase": hardware_by_phase})
            except Exception as exc:
                record({"record_type": "failure", "error_type": type(exc).__name__,
                        "message": str(exc), "hardware_by_phase": hardware_by_phase})
                failure = exc
            finally:
                if seed_controller is not None and not seed_handed_off:
                    seed_controller.close()
                if memory_path.exists():
                    shutil.copyfile(memory_path, run_dir / "memory-after-deletion.sqlite")
                for phase in ("seed", "replay"):
                    for suffix in ("config.json", "server.log"):
                        path = local / f"{phase}-{suffix}"
                        if path.exists():
                            shutil.copyfile(path, run_dir / path.name)
    index = {"suite_id": SUITE_ID, "raw_sha256": _sha(raw_path.read_bytes()),
             "status": "failed" if failure is not None else "completed",
             "summary": summary, "review_status": "unreviewed_private_experiment",
             "qualifies_default": False}
    index_path = run_dir / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    if failure is not None:
        raise RuntimeError(f"memory-quality experiment failed; private evidence: {index_path}") from failure
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=360)
    parser.add_argument("--profile-index", type=Path,
                        help="bind this new private run to a completed audited native profile")
    parser.add_argument("--fixture-seed-hex", help="private seed for paired reruns")
    args = parser.parse_args()
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    seed = bytes.fromhex(args.fixture_seed_hex) if args.fixture_seed_hex else None
    print(run_native(args.config, args.lineage, timeout_s=args.timeout_s,
                     seed=seed, profile_index=args.profile_index))


if __name__ == "__main__":
    main()
