"""Where the agent's actions actually land.

Two backends behind one interface. `AdbDevice` talks to a phone over `adb`.
`SimulatedDevice` runs a small scripted phone in-process, and exists for a
reason beyond convenience: no Android device, emulator or `/dev/kvm` is
reachable from the host this was built on, so without it the loop, the tool
executor and the stopping rules would all be untested code shipped on a
promise.

The simulator emits **real uiautomator XML**, not pre-rendered text, so the
parser, the index-to-coordinate mapping and the screen-change detection are
exercised on the same path a phone would drive. What it cannot test is
whether a real Settings app is laid out the way the fixtures assume; that is
stated in the report rather than hidden.
"""

from __future__ import annotations

import shutil
import subprocess
from xml.sax.saxutils import escape as _xml_escape
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

KEYCODES = {
    "back": "KEYCODE_BACK",
    "home": "KEYCODE_HOME",
    "enter": "KEYCODE_ENTER",
    "delete": "KEYCODE_DEL",
    "recent": "KEYCODE_APP_SWITCH",
}


class Device(Protocol):
    def dump(self) -> str: ...
    def tap(self, x: int, y: int) -> None: ...
    def input_text(self, text: str) -> None: ...
    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int = 300) -> None: ...
    def keyevent(self, key: str) -> None: ...
    def launch(self, app: str) -> None: ...
    def size(self) -> tuple[int, int]: ...


class AdbError(RuntimeError):
    pass


class AdbDevice:
    """A real phone over `adb`."""

    def __init__(self, serial: str | None = None, adb: str = "adb",
                 timeout: float = 30.0):
        if shutil.which(adb) is None:
            raise AdbError(
                f"{adb!r} is not on PATH. Install platform-tools and connect a "
                "device (`adb devices` should list it)."
            )
        self.adb, self.serial, self.timeout = adb, serial, timeout
        self._size: tuple[int, int] | None = None

    def _run(self, *args: str, binary: bool = False):
        cmd = [self.adb] + (["-s", self.serial] if self.serial else []) + list(args)
        proc = subprocess.run(cmd, capture_output=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise AdbError(f"{' '.join(cmd)} failed: {proc.stderr.decode(errors='replace')[:400]}")
        return proc.stdout if binary else proc.stdout.decode(errors="replace")

    def dump(self) -> str:
        # exec-out keeps the XML on stdout instead of round-tripping through
        # /sdcard, which is both faster and avoids a writable-storage
        # assumption that does not hold on every device.
        out = self._run("exec-out", "uiautomator", "dump", "/dev/tty")
        start = out.find("<?xml")
        if start < 0:
            start = out.find("<hierarchy")
        if start < 0:
            raise AdbError(f"no UI hierarchy in dump output: {out[:200]!r}")
        end = out.rfind("</hierarchy>")
        return out[start:end + len("</hierarchy>")] if end > 0 else out[start:]

    def tap(self, x: int, y: int) -> None:
        self._run("shell", "input", "tap", str(x), str(y))

    def input_text(self, text: str) -> None:
        # `input text` takes a single token: spaces become %s and the shell
        # metacharacters have to go, or half the string is eaten.
        escaped = text.replace(" ", "%s")
        for ch in "()<>|;&*\\~\"'`$":
            escaped = escaped.replace(ch, "\\" + ch)
        self._run("shell", "input", "text", escaped)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int = 300) -> None:
        self._run("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(ms))

    def keyevent(self, key: str) -> None:
        code = KEYCODES.get(key, key)
        self._run("shell", "input", "keyevent", code)

    def launch(self, app: str) -> None:
        """Launch by package name, or by visible label via the launcher."""
        if "." in app:
            self._run("shell", "monkey", "-p", app, "-c",
                      "android.intent.category.LAUNCHER", "1")
            return
        raise AdbError(
            f"open_app needs a package name, got {app!r}. Resolve the label to a "
            "package first (`adb shell pm list packages`) or tap its icon."
        )

    def size(self) -> tuple[int, int]:
        if self._size is None:
            out = self._run("shell", "wm", "size")
            part = out.strip().rsplit(":", 1)[-1].strip()
            w, h = part.split("x")
            self._size = (int(w), int(h))
        return self._size


# --------------------------------------------------------------------------
# Simulated phone
# --------------------------------------------------------------------------

@dataclass
class SimElement:
    label: str
    cls: str = "android.widget.TextView"
    clickable: bool = True
    checkable: bool = False
    checked: bool = False
    selected: bool = False
    editable: bool = False
    goto: str | None = None
    """Screen this element navigates to when tapped."""
    toggles: str | None = None
    """State key this element flips when tapped."""
    saves: str | None = None
    """Record key. Tapping commits the current field values into
    ``records[saves]`` and clears them -- the difference between typing a time
    and having an alarm, which the scored suite previously could not tell."""
    label_of: str | None = None
    """State key whose value picks the label from ``labels``."""
    labels: tuple[str, str] = ("", "")
    """(label when the state key is false, label when true)."""

    def rendered_label(self, state: dict) -> str:
        if self.label_of:
            return self.labels[bool(state.get(self.label_of))]
        return self.label


@dataclass
class SimScreen:
    screen_id: str
    elements: list[SimElement]
    back: str | None = None
    dynamic: "Callable[[SimulatedDevice], list[SimElement]] | None" = None
    """Rows derived from device state, appended after ``elements``. A saved
    alarm has to appear *on the screen*, or an oracle that reads the screen --
    which is the only kind a real phone can have -- cannot tell a committed
    alarm from a typed one."""


@dataclass
class SimulatedDevice:
    """A scripted phone that speaks uiautomator XML.

    Deliberately small. It is here to prove the loop terminates, recovers and
    stops -- not to model Android.
    """

    screens: dict[str, SimScreen]
    current: str
    state: dict[str, bool] = field(default_factory=dict)
    width: int = 1080
    height: int = 2400
    typed: list[str] = field(default_factory=list)
    actions: list[tuple[str, ...]] = field(default_factory=list)
    home: str = "home"
    focus: str | None = None
    """Label of the focused text field; typing lands here."""
    fields: dict[str, str] = field(default_factory=dict)
    records: dict[str, list[dict]] = field(default_factory=dict)
    toggled: list[tuple[str, bool, bool]] = field(default_factory=list)
    """(key, before, after) per switch tap. A task whose starting state is
    already correct is failed by a round trip, which final state cannot see."""

    _ROW_H = 160
    _TOP = 200

    def _visible(self) -> list[SimElement]:
        screen = self.screens[self.current]
        rows = list(screen.elements)
        if screen.dynamic:
            rows = [r for r in rows if not getattr(r, "_placeholder", False)]
            rows.extend(screen.dynamic(self))
        return rows

    def _bounds(self, row: int) -> tuple[int, int, int, int]:
        y1 = self._TOP + row * self._ROW_H
        return 0, y1, self.width, y1 + self._ROW_H

    def dump(self) -> str:
        rows = []
        for i, el in enumerate(self._visible()):
            x1, y1, x2, y2 = self._bounds(i)
            checked = el.checked
            if el.toggles:
                checked = self.state.get(el.toggles, el.checked)
            # uiautomator escapes attribute values, and labels really do
            # contain them -- "Network & internet" is on every Android
            # settings screen. Emitting raw breaks the parser on the second
            # screen of the first task.
            label = _xml_escape(el.rendered_label(self.state), {'"': "&quot;"})
            rows.append(
                f'  <node index="{i}" text="{label}" resource-id="" '
                f'class="{el.cls}" package="com.example.sim" content-desc="" '
                f'checkable="{str(el.checkable).lower()}" '
                f'checked="{str(checked).lower()}" '
                f'clickable="{str(el.clickable).lower()}" enabled="true" '
                f'focusable="{str(el.clickable).lower()}" focused="false" '
                f'scrollable="false" long-clickable="false" password="false" '
                f'selected="{str(el.selected).lower()}" '
                f'bounds="[{x1},{y1}][{x2},{y2}]" />'
            )
        body = "\n".join(rows)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<hierarchy rotation="0">\n{body}\n</hierarchy>'
        )

    def _row_at(self, y: int) -> int | None:
        row = (y - self._TOP) // self._ROW_H
        return row if 0 <= row < len(self._visible()) else None

    def tap(self, x: int, y: int) -> None:
        self.actions.append(("tap", str(x), str(y)))
        row = self._row_at(y)
        if row is None:
            return
        el = self._visible()[row]
        if el.editable:
            self.focus = el.label
        if el.toggles:
            before = bool(self.state.get(el.toggles, el.checked))
            self.state[el.toggles] = not before
            self.toggled.append((el.toggles, before, not before))
        if el.saves:
            self.records.setdefault(el.saves, []).append(
                {**self.fields, **{k: v for k, v in self.state.items()
                                   if k.startswith(el.saves + "_")}})
            self.fields = {}
            self.focus = None
        if el.goto and el.goto in self.screens:
            self.current = el.goto

    def input_text(self, text: str) -> None:
        self.actions.append(("text", text))
        self.typed.append(text)
        if self.focus is not None:
            self.fields[self.focus] = self.fields.get(self.focus, "") + text

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int = 300) -> None:
        self.actions.append(("swipe", str(x1), str(y1), str(x2), str(y2)))

    def keyevent(self, key: str) -> None:
        self.actions.append(("key", key))
        if key == "back":
            self.current = self.screens[self.current].back or self.home
        elif key == "home":
            self.current = self.home

    def launch(self, app: str) -> None:
        self.actions.append(("launch", app))
        for screen in self.screens.values():
            for el in screen.elements:
                if el.label.lower() == app.lower() and el.goto:
                    self.current = el.goto
                    return

    def size(self) -> tuple[int, int]:
        return self.width, self.height


def _saved_alarm_rows(device: "SimulatedDevice") -> list[SimElement]:
    """Render committed alarms the way a clock app lists them."""
    saved = device.records.get("alarm", [])
    if not saved:
        return [SimElement("No alarms", clickable=False)]
    rows = []
    for rec in saved:
        hour = (rec.get("Hour") or "?").lstrip("0") or "0"
        minute = (rec.get("Minute") or "?").rjust(2, "0")
        meridiem = "PM" if rec.get("alarm_pm") else "AM"
        rows.append(SimElement(f"{hour}:{minute} {meridiem}", clickable=False))
    return rows


def demo_phone() -> SimulatedDevice:
    """A two-app phone: Settings (Wi-Fi) and Clock (alarms).

    Enough to exercise a multi-screen task, a stateful toggle -- the case the
    earlier single-step run got wrong, tapping a switch that was already on --
    and a text field.
    """
    screens = {
        "home": SimScreen("home", [
            SimElement("Phone"),
            SimElement("Messages"),
            SimElement("Clock", goto="clock"),
            SimElement("Settings", goto="settings"),
            SimElement("Camera"),
        ]),
        "settings": SimScreen("settings", [
            SimElement("Settings", clickable=False),
            SimElement("Network & internet", goto="network"),
            SimElement("Connected devices"),
            SimElement("Apps"),
            SimElement("Battery"),
        ], back="home"),
        "network": SimScreen("network", [
            SimElement("Network & internet", clickable=False),
            SimElement("Wi-Fi", cls="android.widget.Switch", checkable=True,
                       toggles="wifi"),
            SimElement("Mobile network"),
            SimElement("Airplane mode", cls="android.widget.Switch",
                       checkable=True, toggles="airplane"),
        ], back="settings"),
        "clock": SimScreen("clock", [
            SimElement("Alarm", cls="android.widget.Button", goto="alarms"),
            SimElement("Clock", cls="android.widget.Button", selected=True),
            SimElement("Timer", cls="android.widget.Button"),
            SimElement("Stopwatch", cls="android.widget.Button"),
        ], back="home"),
        "alarms": SimScreen("alarms", [
            SimElement("Alarm", cls="android.widget.Button", selected=True),
            SimElement("Add alarm", cls="android.widget.ImageButton",
                       goto="new_alarm"),
        ], back="clock", dynamic=_saved_alarm_rows),
        "new_alarm": SimScreen("new_alarm", [
            SimElement("Set time", clickable=False),
            SimElement("Hour", cls="android.widget.EditText", editable=True),
            SimElement("Minute", cls="android.widget.EditText", editable=True),
            # Android shows one AM/PM control whose label is its value, so
            # "set an alarm for 7:30 AM" has a wrong answer as well as a right
            # one. Without it the task could be passed by typing any digits.
            SimElement("AM", cls="android.widget.Button",
                       label_of="alarm_pm", labels=("AM", "PM"),
                       toggles="alarm_pm"),
            SimElement("Save", cls="android.widget.Button", goto="alarms",
                       saves="alarm"),
        ], back="alarms"),
    }
    return SimulatedDevice(screens=screens, current="home", state={"wifi": False})
