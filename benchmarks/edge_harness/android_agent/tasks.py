"""A scored task suite, judged by what the phone shows rather than by what the
agent said -- or by what the simulator happens to remember.

Three things were wrong with the first version of this file, and all three
made it score more generously than it claimed to:

* **``add_alarm`` passed on any typed string containing "7".** Typing ``7``
  and stopping scored a success on a task whose prompt is "Set an alarm for
  7:30 AM": no minute, no meridiem, and no requirement that the alarm was ever
  saved. It now requires a *committed* alarm reading 7:30 AM, which the clock
  screen lists, so the check is one a real phone can also answer.
* **``wifi_on_already`` scored final state only.** Its note said "any tap on
  the switch fails the task"; the code could not see a tap, so turning Wi-Fi
  off and back on passed. Switch flips are now recorded and the task fails on
  any of them -- which is the exact failure W4A8 exhibited (it turned Wi-Fi on
  and then tapped the switch off again).
* **Predicates read ``SimulatedDevice`` attributes**, so every scored run was a
  simulator run by construction. They read :class:`PhoneState` now, which comes
  from the UI dump for both backends.

Repeats are only trials if something differs between them, so each task
carries **variants**: the same goal against a renamed control, a reordered
screen, or a different starting state. Five runs of one fixture at temperature
0 measure repeatability; five variants measure whether the agent can do the
task.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from device import SimElement, SimulatedDevice, demo_phone
from phone_state import PhoneState, read_state

Predicate = Callable[[PhoneState], bool]

CLOCK_PACKAGE = "com.google.android.deskclock"
"""Clock app to reset for the alarm tasks. AOSP builds and several OEMs ship a
different package (``com.android.deskclock``, ``com.sec.android.app.clockpackage``
on Samsung); override before a real-device run, because clearing the wrong
package leaves a stale alarm and the next run scores against it."""


# --------------------------------------------------------------- variations

def rename(screen: str, old: str, new: str) -> Callable[[SimulatedDevice], None]:
    """Same control, different word. Android OEMs really do ship "WLAN"."""
    def apply(d: SimulatedDevice) -> None:
        for el in d.screens[screen].elements:
            if el.label == old:
                el.label = new
    return apply


def reorder(screen: str, seed: int) -> Callable[[SimulatedDevice], None]:
    """Shuffle the tappable rows. Index-based tapping must not be luck."""
    def apply(d: SimulatedDevice) -> None:
        els = d.screens[screen].elements
        head = [e for e in els if not e.clickable]
        body = [e for e in els if e.clickable]
        random.Random(seed).shuffle(body)
        d.screens[screen].elements = head + body
    return apply


def decoy(screen: str, label: str) -> Callable[[SimulatedDevice], None]:
    """A plausible wrong row, e.g. "Wi-Fi calling" next to "Wi-Fi"."""
    def apply(d: SimulatedDevice) -> None:
        d.screens[screen].elements.append(SimElement(label))
    return apply


def chain(*fns: Callable[[SimulatedDevice], None]) -> Callable[[SimulatedDevice], None]:
    def apply(d: SimulatedDevice) -> None:
        for fn in fns:
            fn(d)
    return apply


@dataclass
class Variant:
    variant_id: str
    mutate: Callable[[SimulatedDevice], None] = lambda d: None


# --------------------------------------------------------------------- task

@dataclass
class Task:
    task_id: str
    prompt: str
    reaches: Predicate
    must_not: Predicate = lambda s: False
    setup: Callable[[SimulatedDevice], None] = lambda d: None
    max_steps: int = 10
    note: str = ""
    variants: list[Variant] = field(default_factory=lambda: [Variant("base")])
    needs: tuple[str, ...] = ()
    """PhoneState facts this task's predicates require that a screen cannot
    show (``toggled``, ``fields``, ``records``). A task listing any of these
    cannot be scored on a real phone, and says so instead of scoring wrong."""
    reset_adb: tuple[tuple[str, ...], ...] = ()
    """Commands that put a real phone into this task's starting state. Without
    them a real-device run is scored against an unknown start, so an empty
    tuple means "refuse to run me on a phone"."""

    def variant(self, n: int) -> Variant:
        return self.variants[n % len(self.variants)]

    def build(self, run: int = 0) -> SimulatedDevice:
        device = demo_phone()
        self.variant(run).mutate(device)
        self.setup(device)
        return device

    def scorable_on(self, state: PhoneState) -> str | None:
        """Why this task cannot be scored from ``state``, or None if it can."""
        if state.source == "device" and self.needs:
            return (f"needs {', '.join(self.needs)}, which a screen cannot "
                    f"show; simulator only")
        return None

    def score(self, device_or_state, declared_done: bool, run: int = 0) -> dict:
        state = (device_or_state if isinstance(device_or_state, PhoneState)
                 else read_state(device_or_state))
        blocked = self.scorable_on(state)
        if blocked:
            return {"task_id": self.task_id, "variant": self.variant(run).variant_id,
                    "scorable": False, "reason": blocked, "success": False,
                    "reached_goal": None, "collateral_damage": None,
                    "declared_done": declared_done}
        reached = bool(self.reaches(state))
        damaged = bool(self.must_not(state))
        return {
            "task_id": self.task_id,
            "variant": self.variant(run).variant_id,
            "scorable": True,
            "reached_goal": reached,
            "collateral_damage": damaged,
            "declared_done": declared_done,
            # The only row that counts: the phone is in the requested state,
            # nothing else was broken, and the agent knew to stop.
            "success": reached and not damaged and declared_done,
        }


# ----------------------------------------------------------------- oracles

# Both the on-screen labels and the setting key, because `switch()` matches
# either source: "Wi-Fi" normalises onto the `wifi` setting by luck, but
# "Airplane mode" does not normalise onto `airplane`, and a name the setting
# lookup misses silently falls through to "control not on screen".
WIFI = ("Wi-Fi", "WiFi", "WLAN", "wifi")
AIRPLANE = ("Airplane mode", "Flight mode", "airplane")


def _wifi_is(want: bool) -> Predicate:
    # `switch` returns None when the control is not on screen, which is not
    # the same as off: an agent that never navigated there has not turned
    # anything on.
    return lambda s: s.switch(*WIFI) is want


def _airplane_on(s: PhoneState) -> bool:
    return s.switch(*AIRPLANE) is True


def _alarm_set(hour: int, minute: int, pm: bool) -> Predicate:
    """A *saved* alarm reading the requested time.

    The clock screen lists committed alarms, so this reads the screen and works
    on a real phone. Typing the digits without saving leaves the list empty.
    """
    wanted = f"{hour}:{minute:02d} {'PM' if pm else 'AM'}"
    return lambda s: s.has_text(wanted)


SUITE: list[Task] = [
    Task(
        task_id="wifi_on",
        prompt="Turn on Wi-Fi.",
        reaches=_wifi_is(True),
        must_not=_airplane_on,
        note="Two screens deep, then a switch. The switch's state is visible, "
             "so stopping is decidable from the screen.",
        variants=[
            Variant("base"),
            Variant("wlan", rename("network", "Wi-Fi", "WLAN")),
            Variant("reordered", reorder("network", seed=7)),
            Variant("decoy", decoy("network", "Wi-Fi calling")),
            Variant("deep", chain(reorder("settings", seed=3),
                                  decoy("network", "Wi-Fi preferences"))),
        ],
        reset_adb=(("shell", "svc", "wifi", "disable"),),
    ),
    Task(
        task_id="wifi_on_already",
        prompt="Turn on Wi-Fi.",
        setup=lambda d: d.state.__setitem__("wifi", True),
        reaches=lambda s: s.switch(*WIFI) is True and s.times_toggled("wifi") == 0,
        must_not=_airplane_on,
        note="Already satisfied. The only correct action is task_complete. "
             "Any flip of the switch fails, including off-and-on again, which "
             "final state cannot see and which W4A8 actually did.",
        needs=("toggled",),
        variants=[
            Variant("base"),
            Variant("wlan", rename("network", "Wi-Fi", "WLAN")),
            Variant("reordered", reorder("network", seed=11)),
        ],
    ),
    Task(
        task_id="wifi_off",
        prompt="Turn off Wi-Fi.",
        setup=lambda d: d.state.__setitem__("wifi", True),
        reaches=_wifi_is(False),
        must_not=_airplane_on,
        note="The mirror of wifi_on: guards against a model that has simply "
             "learned to tap switches.",
        variants=[
            Variant("base"),
            Variant("wlan", rename("network", "Wi-Fi", "WLAN")),
            Variant("reordered", reorder("network", seed=5)),
        ],
        reset_adb=(("shell", "svc", "wifi", "enable"),),
    ),
    Task(
        task_id="airplane_on",
        prompt="Turn on airplane mode.",
        reaches=_airplane_on,
        note="Same screen as Wi-Fi, different row -- catches index confusion, "
             "which is why one variant reorders that screen.",
        variants=[
            Variant("base"),
            Variant("reordered", reorder("network", seed=2)),
            Variant("decoy", decoy("network", "Airplane mode settings")),
        ],
        # `settings put global` needs WRITE_SECURE_SETTINGS, which adb has on a
        # debuggable build but not on every retail phone; the run fails loudly
        # if it is refused rather than scoring against an unknown start.
        reset_adb=(("shell", "settings", "put", "global", "airplane_mode_on", "0"),),
    ),
    Task(
        task_id="open_clock",
        prompt="Open the Clock app and go to the Alarm tab.",
        reaches=lambda s: s.has_text("Add alarm") or s.has_text("Set time"),
        note="Navigation only, no state change. Isolates 'can it find an app' "
             "from 'can it finish'.",
        variants=[
            Variant("base"),
            Variant("reordered", reorder("home", seed=4)),
            Variant("decoy", decoy("home", "Clock Widget")),
        ],
        reset_adb=(("shell", "am", "force-stop", CLOCK_PACKAGE),
                   ("shell", "input", "keyevent", "KEYCODE_HOME")),
    ),
    Task(
        task_id="add_alarm",
        prompt="Set an alarm for 7:30 AM.",
        reaches=_alarm_set(7, 30, pm=False),
        must_not=lambda s: s.has_text("7:30 PM"),
        max_steps=12,
        note="The full flow: right app, right tab, add, type both fields, "
             "leave the meridiem alone, save. Previously this passed on any "
             "typed string containing '7'.",
        variants=[
            Variant("base"),
            Variant("reordered", reorder("clock", seed=9)),
            Variant("decoy", decoy("alarms", "Alarm settings")),
        ],
        # `pm clear` removes saved alarms, so run N does not see run N-1's.
        reset_adb=(("shell", "pm", "clear", CLOCK_PACKAGE),
                   ("shell", "input", "keyevent", "KEYCODE_HOME")),
    ),
]

SUITE_BY_ID = {t.task_id: t for t in SUITE}
