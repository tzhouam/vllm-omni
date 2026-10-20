"""Recover a tool call from what the model actually emitted.

vLLM's stock `glm45` parser handles Spark's tool output most of the time --
4 of 5 screens in a spot check -- and then does not. On the Wi-Fi toggle
screen, the one where a switch's state decides the right action, the model
dropped out of GLM-4.5 format entirely and emitted

    [tap]
    {"tool": "tap", "parameters": {"index": 1}

with the JSON unterminated. `tool_calls` comes back empty, and an agent that
only reads `tool_calls` sees a step with no action at all.

For a phone agent that is the difference between finishing a task and
stopping in the middle of it, so this module tries, in order:

1. the parsed `tool_calls` (the normal path -- nothing here second-guesses it),
2. a GLM-4.5 `<tool_call>` block left in the content,
3. the bracketed `[name]` + JSON form above, repairing unbalanced braces,
4. a bare JSON object naming a tool.

Every recovery is reported, because an agent that quietly repairs its model's
output will hide a regression in that model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_BRACKET = re.compile(r"\[([a-z_][a-z0-9_]*)\]", re.I)
_GLM_CALL = re.compile(
    r"<tool_call>\s*([a-z_][a-z0-9_]*)(.*?)</tool_call>", re.I | re.S)
_GLM_ARG = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.I | re.S)

_ARG_CONTAINERS = ("parameters", "arguments", "args")
_NAME_KEYS = ("tool", "name", "tool_name", "function")


@dataclass
class ParsedCall:
    name: str | None
    args: dict
    source: str
    """Which path produced it: tool_calls | glm_block | bracketed | bare_json."""

    @property
    def recovered(self) -> bool:
        return self.name is not None and self.source != "tool_calls"


def _coerce(value: str):
    """Tool arguments arrive as strings; indices must become integers."""
    text = value.strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def _balance(text: str) -> str:
    """Close a JSON object the model stopped short of finishing."""
    depth_curly = text.count("{") - text.count("}")
    depth_square = text.count("[") - text.count("]")
    if depth_curly < 0 or depth_square < 0:
        return text
    return text + "]" * depth_square + "}" * depth_curly


def _load(text: str) -> dict | None:
    for candidate in (text, _balance(text)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _unwrap(payload: dict, fallback_name: str | None) -> tuple[str | None, dict]:
    """Pull a (name, args) pair out of assorted self-describing shapes."""
    name = fallback_name
    for key in _NAME_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            name = value
            break
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            name = value["name"]
            payload = {**payload, **value}
            break
    args: dict = {}
    for key in _ARG_CONTAINERS:
        inner = payload.get(key)
        if isinstance(inner, dict):
            args = dict(inner)
            break
        if isinstance(inner, str):
            parsed = _load(inner)
            if parsed is not None:
                args = parsed
                break
    if not args:
        args = {k: v for k, v in payload.items()
                if k not in _NAME_KEYS + _ARG_CONTAINERS}
    return name, args


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    return text[start:] if start >= 0 else None


def extract_tool_call(message: dict) -> ParsedCall:
    """The one call to act on, from whichever shape the model used."""
    calls = message.get("tool_calls") or []
    if calls:
        fn = calls[0].get("function", {}) or {}
        raw = fn.get("arguments")
        if isinstance(raw, dict):
            args = raw
        else:
            args = _load(raw or "{}") or {}
        if set(args) & set(_ARG_CONTAINERS):
            _, args = _unwrap(args, fn.get("name"))
        return ParsedCall(fn.get("name"), args, "tool_calls")

    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    for text in (content, reasoning):
        if not text.strip():
            continue

        if m := _GLM_CALL.search(text):
            args = {k.strip(): _coerce(v) for k, v in _GLM_ARG.findall(m.group(2))}
            return ParsedCall(m.group(1), args, "glm_block")

        bracket = _BRACKET.search(text)
        blob = _first_json_object(text)
        if blob is not None:
            payload = _load(blob)
            if payload is not None:
                name, args = _unwrap(payload, bracket.group(1) if bracket else None)
                if name:
                    return ParsedCall(name, args, "bracketed" if bracket else "bare_json")

        if bracket and not blob:
            # A naked "[press_key]" with no arguments is still an instruction.
            return ParsedCall(bracket.group(1), {}, "bracketed")

    return ParsedCall(None, {}, "none")
