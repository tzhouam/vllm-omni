# SPDX-License-Identifier: Apache-2.0
"""Evidence-gated, batch-one routing for the local Agent.

The router selects a *complete* model route. It never infers task quality from
an artifact size, a component benchmark, or an accelerator being present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite
import re
from typing import Callable, Mapping, Sequence

TASK_CLASSES = frozenset({"basic", "browser_text", "browser_vision", "windows_settings", "memory", "code_tools", "long_reasoning"})
LENGTH_BUCKETS = ("short", "medium", "long")


def nearest_rank(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("raw latency samples are required")
    if any(not isfinite(value) or value < 0 for value in values):
        raise ValueError("latency samples must be finite and nonnegative")
    return sorted(values)[ceil(len(values) * fraction) - 1]


@dataclass(frozen=True)
class Route:
    route_id: str
    artifact_id: str
    model: str
    backend: str
    modalities: frozenset[str]
    placement: str
    memory_demands: Mapping[str, int]
    requires_nvidia: bool = False

    def __post_init__(self) -> None:
        if not all((self.route_id, self.artifact_id, self.model, self.backend, self.placement)):
            raise ValueError("route identity and actual placement are required")
        if not self.modalities or not self.memory_demands or any(type(value) is not int or value < 0 for value in self.memory_demands.values()):
            raise ValueError("modalities and nonnegative memory demands are required")


@dataclass(frozen=True)
class Qualification:
    """Whole-Agent paired task evidence for a single exact runtime condition."""

    route_id: str
    artifact_id: str
    task_class: str
    suite_id: str
    environment_fingerprint: str
    power_condition: str
    successes: int
    attempts: int
    answer_latency_s: Mapping[str, tuple[float, ...]]
    ttft_s: Mapping[str, tuple[float, ...]]
    warmups_per_length: Mapping[str, int]
    sustained_seconds: float
    correctness_pass: bool
    tool_safety_pass: bool
    memory_pass: bool
    cancel_recovery_pass: bool
    stability_pass: bool
    actual_placement_verified: bool
    batch_size: int = 1
    concurrency: int = 1
    raw_evidence: str = ""

    @property
    def qualification_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.task_class not in TASK_CLASSES:
            errors.append("unknown task class")
        if not self.raw_evidence or not self.suite_id or not self.environment_fingerprint or not self.power_condition:
            errors.append("missing raw evidence or condition identity")
        if self.batch_size != 1 or self.concurrency != 1:
            errors.append("performance must use batch=1, one active request")
        if not 0 <= self.successes <= self.attempts or self.attempts == 0:
            errors.append("invalid paired task outcomes")
        for bucket in LENGTH_BUCKETS:
            latencies = self.answer_latency_s.get(bucket, ())
            firsts = self.ttft_s.get(bucket, ())
            if self.warmups_per_length.get(bucket, 0) < 1:
                errors.append(f"{bucket}: separate warmup absent")
            if len(latencies) < 20 or len(firsts) != len(latencies):
                errors.append(f"{bucket}: at least 20 paired raw answer/TTFT samples required")
            if any(not isfinite(value) or value < 0 for value in (*latencies, *firsts)):
                errors.append(f"{bucket}: invalid latency sample")
        if self.sustained_seconds < 1800:
            errors.append("30-minute sequential run absent")
        for name in (
            "correctness_pass", "tool_safety_pass", "memory_pass",
            "cancel_recovery_pass", "stability_pass", "actual_placement_verified",
        ):
            if not getattr(self, name):
                errors.append(name)
        return tuple(errors)

    @property
    def qualified(self) -> bool:
        return not self.qualification_errors

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def whole_answer_p95_s(self) -> float:
        return max(nearest_rank(self.answer_latency_s[bucket], .95) for bucket in LENGTH_BUCKETS)


@dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: str
    actual_placement: str | None = None


@dataclass(frozen=True)
class Decision:
    route: Route | None
    task_class: str
    qualification: Qualification | None
    admission: Admission | None
    experimental: bool
    refusals: Mapping[str, str] = field(default_factory=dict)


def select_route(
    task_class: str,
    routes: Sequence[Route],
    qualifications: Sequence[Qualification],
    *,
    suite_id: str,
    environment_fingerprint: str,
    power_condition: str,
    admit: Callable[[Route], Admission],
    bootstrap_route_id: str | None = None,
) -> Decision:
    """Select task success first, then full-answer p95 among equal success.

    A bootstrap route is available only by explicit caller configuration. It
    remains visibly experimental until its own Agent qualification passes.
    """
    if task_class not in TASK_CLASSES or not suite_id:
        raise ValueError(f"unknown task class: {task_class}")
    refusals: dict[str, str] = {}
    eligible: list[tuple[Route, Qualification, Admission]] = []
    by_route = {route.route_id: route for route in routes}
    required_modalities = frozenset({"text", "image"} if task_class == "browser_vision" else {"text"})
    for route in routes:
        if not required_modalities.issubset(route.modalities):
            refusals[route.route_id] = (
                f"route lacks required modalities: {sorted(required_modalities - route.modalities)}"
            )
            continue
        profile = next((p for p in qualifications if
                        p.route_id == route.route_id and p.artifact_id == route.artifact_id
                        and p.task_class == task_class
                        and p.suite_id == suite_id
                        and p.environment_fingerprint == environment_fingerprint
                        and p.power_condition == power_condition and p.qualified), None)
        if profile is None:
            refusals[route.route_id] = "no current whole-Agent batch-1 qualification"
            continue
        admission = admit(route)
        if not admission.admitted or (
            admission.actual_placement is not None
            and admission.actual_placement != route.placement
        ):
            refusals[route.route_id] = admission.reason or "actual placement differs from qualified route"
            continue
        eligible.append((route, profile, admission))
    if eligible:
        # Rates are compared exactly; latency breaks only a task-success tie.
        route, profile, admission = min(
            eligible,
            key=lambda row: (-row[1].success_rate, row[1].whole_answer_p95_s, row[0].route_id),
        )
        return Decision(route, task_class, profile, admission, False, refusals)
    if bootstrap_route_id is not None:
        route = by_route.get(bootstrap_route_id)
        if route is not None and not required_modalities.issubset(route.modalities):
            refusals[route.route_id] = (
                f"bootstrap route lacks required modalities: {sorted(required_modalities - route.modalities)}"
            )
        elif route is not None:
            admission = admit(route)
            if admission.admitted and admission.actual_placement in (None, route.placement):
                return Decision(route, task_class, None, admission, True, refusals)
            refusals[route.route_id] = admission.reason or "bootstrap route refused"
    return Decision(None, task_class, None, None, False, refusals)


def classify_task(text: str, *, has_image: bool = False) -> str:
    """A conservative local rule until a task classifier itself has evidence."""
    lowered = text.casefold()
    if has_image:
        return "browser_vision"
    if any(word in lowered for word in (
        "registry", "windows setting", "display setting", "mouse speed",
        "mouse_speed", "pointer speed", "settings_read", "settings_set",
        "系统设置", "注册表", "鼠标速度", "鼠标灵敏度",
    )):
        return "windows_settings"
    if any(word in lowered for word in (
        "screenshot", "screen_capture", "screen capture", "screen understanding",
        "image", "visual", "图像", "图片", "截图", "屏幕", "视觉",
    )):
        return "browser_vision"
    if any(word in lowered for word in ("code", "script", "debug", "代码", "编程")):
        return "code_tools"
    if any(word in lowered for word in ("remember", "memory", "recall", "记得", "回忆")):
        return "memory"
    if any(word in lowered for word in ("browser", "website", "web page", "url", "网页", "浏览器", "链接")) or re.search(
        r"https?://|\b(?:[a-z0-9-]+\.)+(?:com|org|net|io|edu|gov)\b", lowered,
    ):
        return "browser_text"
    if len(text) > 2000 or any(word in lowered for word in ("prove", "derive", "推导", "证明")):
        return "long_reasoning"
    return "basic"
