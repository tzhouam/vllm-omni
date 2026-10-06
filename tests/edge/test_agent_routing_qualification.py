# SPDX-License-Identifier: Apache-2.0
"""A fixed-suite failure cannot become a default route through pass flags."""

from __future__ import annotations

from vllm_omni.edge.agent.router import (
    FIXED_SUITE_ID, Admission, Qualification, Route, select_route,
)


def _route(route_id: str) -> Route:
    return Route(route_id, "artifact", route_id,
                 "external.llamacpp.text.v1", frozenset({"text"}),
                 "cpu", {"host_ram": 1})


def _qualification(route_id: str, *, successes: int,
                   latency_s: float, attempts: int = 60) -> Qualification:
    samples = {length: (latency_s,) * 20
               for length in ("short", "medium", "long")}
    return Qualification(
        route_id=route_id, artifact_id="artifact", task_class="basic",
        suite_id=FIXED_SUITE_ID, environment_fingerprint="test-host",
        power_condition="AC", successes=successes, attempts=attempts,
        answer_latency_s=samples,
        ttft_s={length: (latency_s / 2,) * 20 for length in samples},
        warmups_per_length={length: 1 for length in samples},
        sustained_seconds=1800, correctness_pass=True,
        tool_safety_pass=True, memory_pass=True, cancel_recovery_pass=True,
        stability_pass=True, actual_placement_verified=True,
        raw_evidence="signed-evidence.jsonl#sha256=verified",
    )


def _choose(routes: list[Route], profiles: list[Qualification]):
    return select_route(
        "basic", routes, profiles, suite_id=FIXED_SUITE_ID,
        environment_fingerprint="test-host", power_condition="AC",
        admit=lambda route: Admission(True, "preflight", actual_placement="cpu"),
    )


def test_fixed_suite_rejects_zero_and_one_failed_measured_request() -> None:
    route = _route("fast-but-failed")
    for successes in (0, 59):
        profile = _qualification(route.route_id, successes=successes,
                                 latency_s=0.1)
        assert not profile.qualified
        assert any("every measured request" in error
                   for error in profile.qualification_errors)
        decision = _choose([route], [profile])
        assert decision.route is None
        assert decision.refusals[route.route_id] == (
            "no current whole-Agent batch-1 qualification"
        )


def test_fixed_suite_attempts_must_match_all_measured_latency_samples() -> None:
    profile = _qualification("inflated", successes=61, latency_s=0.1,
                             attempts=61)
    assert not profile.qualified
    assert "fixed-suite attempts differ from measured latency samples" in (
        profile.qualification_errors
    )


def test_fully_passing_fixed_suite_routes_use_whole_answer_p95() -> None:
    fast, slow, failed = (_route(name) for name in ("fast", "slow", "failed"))
    decision = _choose(
        [slow, failed, fast],
        [_qualification(slow.route_id, successes=60, latency_s=5.0),
         _qualification(failed.route_id, successes=59, latency_s=0.01),
         _qualification(fast.route_id, successes=60, latency_s=2.0)],
    )
    assert decision.route == fast
    assert decision.qualification is not None
    assert decision.qualification.whole_answer_p95_s == 2.0
    assert failed.route_id in decision.refusals
