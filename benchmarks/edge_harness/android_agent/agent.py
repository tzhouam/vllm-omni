"""The observe-decide-act loop.

Three things in here are not generic agent scaffolding; they are direct
responses to how this model failed when it was measured one step at a time
(analysis/spark_edge_support.md section 5):

* **A step budget, not a bigger token cap.** On the one genuinely ambiguous
  screen the model reasoned past 1200 tokens without closing `<think>`, and
  vLLM's reasoning parser only emits content once the block closes -- so the
  reply came back empty, with no tool call at all. Raising the cap buys a
  longer silence. The loop instead retries that step with thinking disabled,
  which measured 0.61 s against 8.9 s, and takes the weaker-but-present answer.

* **Detecting the no-op.** The clearest wrong action was tapping a Wi-Fi
  switch that was already on -- undoing the task it had just completed. An
  action that leaves the screen signature unchanged is reported back as such,
  because "nothing happened" is information the model cannot otherwise have.

* **Repeat suppression.** Two identical actions on an identical screen means
  stuck, and a third will not help. The loop says so in the transcript and
  stops, rather than burning the budget.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from urllib import error as urlerror
from urllib import request as urlrequest

from action_space import SYSTEM_PROMPT, TOOLS
from device import Device
from executor import ActionResult, execute
from tool_parse import extract_tool_call
from ui_tree import Element, parse_ui_dump, render_tree, screen_signature


@dataclass
class AgentConfig:
    base_url: str = "http://127.0.0.1:8077/v1"
    model: str = "XHToken/Spark-X2.5-1.7B"
    max_steps: int = 12
    think: bool = True
    think_fallback: bool = True
    """On an empty reply, retry the step without thinking rather than fail."""
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout: float = 900.0
    history: int = 6
    retries: int = 3
    """Transient HTTP failures retried before a step is abandoned."""
    retry_backoff_s: float = 2.0
    """Recent actions shown back to the model."""


@dataclass
class Step:
    n: int
    screen: str
    tool: str | None
    args: dict
    result: ActionResult | None
    latency_s: float
    thinking: bool
    note: str = ""
    parsed_via: str = "tool_calls"
    """Which parser recovered the call; anything but tool_calls is worth seeing."""


@dataclass
class Episode:
    task: str
    steps: list[Step] = field(default_factory=list)
    done: bool = False
    summary: str = ""
    stop_reason: str = ""

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def transcript(self) -> str:
        lines = [f"task: {self.task}"]
        for s in self.steps:
            call = f"{s.tool}({json.dumps(s.args, ensure_ascii=False)})" if s.tool else "(no tool call)"
            outcome = s.result.message if s.result else s.note
            if s.parsed_via not in ("tool_calls", "none"):
                outcome += f"  [recovered via {s.parsed_via}]"
            lines.append(f"  {s.n:2d}. {call:48s} -> {outcome}  [{s.latency_s:.2f}s]")
        lines.append(f"stopped: {self.stop_reason}")
        return "\n".join(lines)


class ChatError(RuntimeError):
    pass


def chat(cfg: AgentConfig, messages: list[dict], *, think: bool) -> tuple[dict, float]:
    payload = {
        "model": cfg.model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if not think:
        # Spark's chat template opens <think> by default; closing it up front
        # is what makes the model answer directly.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urlrequest.Request(
        f"{cfg.base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    last: Exception | None = None
    for attempt in range(max(1, cfg.retries)):
        try:
            with urlrequest.urlopen(req, timeout=cfg.timeout) as r:
                return json.loads(r.read()), time.perf_counter() - t0
        except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
            # A phone task is a dozen round trips; one dropped connection
            # should cost a retry, not the episode.
            last = exc
            if attempt + 1 < max(1, cfg.retries):
                time.sleep(cfg.retry_backoff_s * (attempt + 1))
    raise ChatError(
        f"{cfg.base_url} failed after {cfg.retries} attempts: {last}") from last


def wait_for_server(cfg: AgentConfig, timeout_s: float = 300.0) -> bool:
    """Block until the endpoint serves a model list, or give up.

    Starting an episode against a server that is still loading produces a
    confusing transcript full of connection errors; this turns that into one
    clear wait.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urlrequest.urlopen(f"{cfg.base_url}/models", timeout=5) as r:
                if b'"id"' in r.read():
                    return True
        except Exception:
            time.sleep(2.0)
    return False


def parse_response(response: dict):
    """The one call to act on, from whichever shape the model used.

    Spark does not always stay in GLM-4.5 tool format -- see tool_parse -- and
    on a phone a dropped step is a stalled task, so the fallbacks matter.
    """
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError):
        message = {}
    return extract_tool_call(message)


def first_tool_call(response: dict) -> tuple[str | None, dict]:
    """Name and arguments only, for callers that do not care how it parsed."""
    parsed = parse_response(response)
    return parsed.name, parsed.args


def _user_message(task: str, screen: str, log: list[str], history: int) -> str:
    parts = [f"Task: {task}", "", "Current screen:", screen]
    if log:
        parts += ["", "Actions so far:"] + [f"  {line}" for line in log[-history:]]
    parts += ["", "Choose the next action. Call exactly one tool."]
    return "\n".join(parts)


def observe(device: Device) -> tuple[list[Element], str, str]:
    elements = parse_ui_dump(device.dump())
    return elements, render_tree(elements), screen_signature(elements)


def run_episode(device: Device, task: str, cfg: AgentConfig | None = None) -> Episode:
    cfg = cfg or AgentConfig()
    episode = Episode(task=task)
    log: list[str] = []
    recent: list[tuple[str, str]] = []  # (action, screen signature before it)
    signature_before_last: str | None = None
    last_action: str | None = None
    oscillations = 0

    for n in range(1, cfg.max_steps + 1):
        elements, screen, signature = observe(device)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": _user_message(task, screen, log, cfg.history)},
        ]

        thinking = cfg.think
        response, latency = chat(cfg, messages, think=thinking)
        parsed = parse_response(response)
        tool, args = parsed.name, parsed.args

        if tool is None and cfg.think_fallback and thinking:
            # The measured failure: reasoning never closed, so the parser
            # emitted nothing. A second, non-thinking pass is worth ~0.6 s.
            thinking = False
            retry, retry_latency = chat(cfg, messages, think=False)
            latency += retry_latency
            parsed = parse_response(retry)
            tool, args = parsed.name, parsed.args

        if tool is None:
            episode.steps.append(Step(n, screen, None, {}, None, latency, thinking,
                                      note="model returned no tool call",
                                      parsed_via=parsed.source))
            episode.stop_reason = "no tool call, even without thinking"
            return episode

        if (tool_key := f"{tool}:{json.dumps(args, sort_keys=True)}") and \
                recent.count((tool_key, signature)) >= 2:
            episode.steps.append(Step(n, screen, tool, args, None, latency, thinking,
                                      note="repeated a no-op action",
                                      parsed_via=parsed.source))
            episode.stop_reason = (
                f"stuck: {tool} repeated on an unchanged screen"
            )
            return episode
        recent.append((tool_key, signature))

        result = execute(tool, args, elements, device)
        if result.ok and not result.done:
            _, _, after = observe(device)
            if after == signature:
                # Say so out loud: the model has no other way to learn that
                # its action changed nothing.
                result = ActionResult(True, result.message + " -- the screen did not change",
                                      action=result.action)
            elif tool_key == last_action and after == signature_before_last:
                # Oscillation: the *same* action, undoing what it just did --
                # the measured failure, tapping a switch off and on forever.
                # It hides from the no-change check because each individual tap
                # really does change the screen. Deliberately narrow: revisiting
                # a screen by other means (open a submenu, press back) is normal
                # navigation and must not trip this.
                result = ActionResult(
                    True,
                    result.message + " -- that undid your previous action; the "
                    "setting is now back where it started",
                    action=result.action,
                )
                oscillations += 1
                if oscillations >= 2:
                    episode.steps.append(Step(n, screen, tool, args, result, latency,
                                              thinking, parsed_via=parsed.source))
                    episode.stop_reason = "stuck: repeatedly undoing its own action"
                    return episode
        signature_before_last, last_action = signature, tool_key
        episode.steps.append(Step(n, screen, tool, args, result, latency, thinking,
                                  parsed_via=parsed.source))
        log.append(f"{tool} -> {result.message}")

        if result.done:
            episode.done = True
            episode.summary = result.summary
            episode.stop_reason = "task_complete"
            return episode

    episode.stop_reason = f"step budget exhausted ({cfg.max_steps})"
    return episode
