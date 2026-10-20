"""Turn one tool call into one device action.

Every failure here is returned to the model as a sentence it can act on, not
raised. A 1.7B model asked for `index: 9` on a four-element screen has made a
recoverable mistake, and "index 9 is out of range, the screen has 4 elements
(0-3)" is the difference between the next step being right and the episode
ending. Exceptions are reserved for the device being gone.
"""

from __future__ import annotations

from dataclasses import dataclass

from device import KEYCODES, Device
from ui_tree import Element


@dataclass
class ActionResult:
    ok: bool
    message: str
    done: bool = False
    summary: str = ""
    action: str = ""

    def as_tool_reply(self) -> str:
        return self.message


def _need(args: dict, key: str, kind: type, what: str) -> tuple[object, str | None]:
    if key not in args:
        return None, f"missing required argument {key!r} for {what}"
    value = args[key]
    if kind is int:
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None, f"{key!r} must be a whole number, got {args[key]!r}"
    elif not isinstance(value, kind):
        return None, f"{key!r} must be {kind.__name__}, got {type(value).__name__}"
    return value, None


def _resolve(index: object, elements: list[Element]) -> tuple[Element | None, str | None]:
    i = int(index)
    if not elements:
        return None, "the screen has no addressable elements; try press_key(back)"
    if not 0 <= i < len(elements):
        return None, (
            f"index {i} is out of range; the screen has {len(elements)} "
            f"elements (0-{len(elements) - 1})"
        )
    return elements[i], None


def execute(name: str, args: dict, elements: list[Element],
            device: Device) -> ActionResult:
    """Run one tool call. Returns what to tell the model."""
    args = args or {}

    if name == "task_complete":
        summary = str(args.get("summary", "")).strip()
        return ActionResult(True, "task marked complete", done=True,
                            summary=summary, action="task_complete")

    if name == "tap":
        index, err = _need(args, "index", int, "tap")
        if err:
            return ActionResult(False, err, action="tap")
        el, err = _resolve(index, elements)
        if err:
            return ActionResult(False, err, action="tap")
        x, y = el.center
        device.tap(x, y)
        state = ""
        if "checked" in el.flags:
            state = f" (was {'on' if el.flags['checked'] else 'off'})"
        return ActionResult(True, f'tapped [{index}] "{el.text}"{state}',
                            action=f"tap:{index}")

    if name == "type_text":
        index, err = _need(args, "index", int, "type_text")
        if err:
            return ActionResult(False, err, action="type_text")
        text, err = _need(args, "text", str, "type_text")
        if err:
            return ActionResult(False, err, action="type_text")
        el, err = _resolve(index, elements)
        if err:
            return ActionResult(False, err, action="type_text")
        # Focus first: `input text` goes to whatever holds focus, which on a
        # freshly dumped screen is usually nothing.
        x, y = el.center
        device.tap(x, y)
        device.input_text(str(text))
        return ActionResult(True, f'typed "{text}" into [{index}] "{el.text}"',
                            action=f"type:{index}")

    if name == "swipe":
        direction = str(args.get("direction", "")).lower()
        if direction not in ("up", "down", "left", "right"):
            return ActionResult(
                False, "direction must be one of up, down, left, right",
                action="swipe")
        w, h = device.size()
        cx, cy = w // 2, h // 2
        dx, dy = int(w * 0.35), int(h * 0.30)
        # `direction` is where the *content* should move, i.e. what the user
        # means by "scroll down to see more", so the finger travels the other
        # way. Stated in the tool description too, because guessing this wrong
        # is invisible in the transcript.
        moves = {
            "down": (cx, cy + dy, cx, cy - dy),
            "up": (cx, cy - dy, cx, cy + dy),
            "right": (cx + dx, cy, cx - dx, cy),
            "left": (cx - dx, cy, cx + dx, cy),
        }
        device.swipe(*moves[direction])
        return ActionResult(True, f"swiped to scroll {direction}",
                            action=f"swipe:{direction}")

    if name == "press_key":
        key = str(args.get("key", "")).lower()
        if key not in KEYCODES:
            return ActionResult(
                False, f"unknown key {key!r}; use one of {', '.join(KEYCODES)}",
                action="press_key")
        device.keyevent(key)
        return ActionResult(True, f"pressed {key}", action=f"key:{key}")

    if name == "open_app":
        app = str(args.get("name", "")).strip()
        if not app:
            return ActionResult(False, "open_app needs a name", action="open_app")
        try:
            device.launch(app)
        except Exception as exc:  # device-specific launch failures are recoverable
            return ActionResult(False, f"could not launch {app!r}: {exc}",
                                action="open_app")
        return ActionResult(True, f"launched {app}", action=f"open:{app}")

    return ActionResult(
        False,
        f"unknown tool {name!r}; available: tap, type_text, swipe, press_key, "
        "open_app, task_complete",
        action="unknown",
    )
