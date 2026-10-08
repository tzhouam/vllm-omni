# SPDX-License-Identifier: Apache-2.0
"""Evidence-gated, batch-one routing for the local Agent.

The router selects a *complete* model route. It never infers task quality from
an artifact size, a component benchmark, or an accelerator being present.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil, isfinite

from vllm_omni.edge.agent.model_output import AgentOutputContract

TASK_CLASSES = frozenset({
    "basic", "browser_text", "browser_vision", "windows_settings", "memory", "code_tools", "long_reasoning",
})
LENGTH_BUCKETS = ("short", "medium", "long")
FIXED_SUITE_ID = "edge-agent-fixed-local-fixtures-v1"
_DOTTED_TOKEN = re.compile(r"\w+\.\w+", re.UNICODE)
_DESKTOP_CAPTURE_REQUEST = re.compile(
    r"\b(?:use|run|invoke|call)\s+screen_capture\b|"
    r"\bcapture (?:my |the )?screen\b|"
    r"\b(?:show|read|view|inspect|describe|capture)\s+(?:my |the )?desktop (?:screen|screenshot)\b|"
    r"\b(?:show|read|view|inspect|describe|capture)\s+(?:my |the )?windows screen\b|"
    r"(?:请)?(?:使用|调用|用)\s*screen_capture\b|"
    r"截取屏幕|抓取屏幕|捕获屏幕|(?:请)?(?:给我|生成|发我)屏幕截图|"
    r"(?:查看|读取|显示|截取|抓取)\s*windows\s*屏幕",
    re.IGNORECASE,
)


def strip_task_links(text: str) -> str:
    """Mask URL- and path-shaped tokens before interpreting tool intent.

    A path such as ``/screen_capture`` or ``C:\\screen_capture`` is a resource
    name, not a request to capture the desktop. Relative queries/fragments
    such as ``?screen_capture`` are likewise data. Consume the entire token,
    including quotes and punctuation; explicit instructions next to a link
    need whitespace separation. This fail-closed rule is shared with the
    controller's permission check.
    """
    def mask(match: re.Match[str]) -> str:
        token = match.group()
        if ("/" in token or "\\" in token or _DOTTED_TOKEN.search(token)
                or any(marker in token[:-1] for marker in ("?", "#"))):
            return " "
        return token

    return re.sub(r"\S+", mask, text)


def desktop_capture_requested(text: str) -> bool:
    """Require a trusted, explicit desktop operation outside URL/path tokens."""
    return _DESKTOP_CAPTURE_REQUEST.search(strip_task_links(text)) is not None


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
    model_output_contract: AgentOutputContract | None = None
    base_artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not all((self.route_id, self.artifact_id, self.model, self.backend, self.placement)):
            raise ValueError("route identity and declared placement are required")
        if (not self.modalities or not self.memory_demands or
                any(type(value) is not int or value < 0 for value in self.memory_demands.values())):
            raise ValueError("modalities and nonnegative memory demands are required")
        if self.memory_demands.get("host_ram", 0) <= 0:
            raise ValueError("complete Agent route must reserve positive host RAM")
        if self.model_output_contract is not None:
            if not isinstance(self.model_output_contract, AgentOutputContract):
                raise ValueError("route requires a typed output contract")
            identity = self.model_output_contract.consumer_identity(self.base_artifact_id)
            if self.artifact_id != "strata-agent:" + identity["identity_sha256"]:
                raise ValueError("route artifact does not bind the actual Agent output consumer")


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
    output_contract_sha256: str | None = None

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
        if self.suite_id == FIXED_SUITE_ID:
            measured_count = sum(len(self.answer_latency_s.get(bucket, ()))
                                 for bucket in LENGTH_BUCKETS)
            if self.attempts != measured_count:
                errors.append("fixed-suite attempts differ from measured latency samples")
            if self.successes != self.attempts:
                errors.append("fixed-suite qualification requires every measured request to succeed")
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
                        and p.output_contract_sha256 == (route.model_output_contract.identity_sha256
                                                        if route.model_output_contract else None)
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
    prose = strip_task_links(lowered)
    if has_image:
        return "browser_vision"
    if any(word in prose for word in (
        "registry", "windows setting", "display setting", "mouse speed",
        "mouse_speed", "pointer speed", "settings_read", "settings_set",
        "系统设置", "注册表", "鼠标速度", "鼠标灵敏度",
    )):
        return "windows_settings"
    if any(word in prose for word in (
        "screenshot", "screen capture", "screen understanding",
        "desktop screen", "desktop screenshot", "windows screen",
        "image", "visual", "图像", "图片", "截图", "屏幕", "视觉",
    )) or desktop_capture_requested(prose):
        return "browser_vision"
    if any(word in prose for word in ("code", "script", "debug", "代码", "编程")):
        return "code_tools"
    if any(word in prose for word in ("remember", "memory", "recall", "记得", "回忆")):
        return "memory"
    if any(word in prose for word in ("browser", "website", "web page", "url", "网页", "浏览器", "链接")) or re.search(
        r"https?://|\b(?:[a-z0-9-]+\.)+(?:com|org|net|io|edu|gov)\b", lowered,
    ):
        return "browser_text"
    if len(text) > 2000 or any(word in prose for word in ("prove", "derive", "推导", "证明")):
        return "long_reasoning"
    return "basic"
