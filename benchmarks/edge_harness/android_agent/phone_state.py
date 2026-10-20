"""What the phone is actually in, read the same way for both backends.

The scored suite used to take its predicates over ``SimulatedDevice``'s Python
attributes, so every scored run was a simulator run by construction -- the
real-device path existed for *driving* a phone and not for judging one. That
made "the agent suite" a claim about fixtures.

Success is therefore read two ways, and the order matters:

1. **Out of band, from the setting itself** -- ``device.state`` in the
   simulator, ``settings get global wifi_on`` over adb. This is the ground
   truth, and it does not depend on where the agent happened to leave the
   phone. Scoring from the final screen alone fails an agent that turns Wi-Fi
   on correctly and then presses Home, which is not a failure.
2. **From the UI dump**, for anything with no setting behind it -- a saved
   alarm appears as a row in the clock app and nowhere else. The simulator
   emits real uiautomator XML, so this goes down the same parse, the same
   switch lookup and the same text matching a phone would.

Only the facts a phone genuinely cannot report after the fact -- which
switches were flipped on the way, and what was typed into which field -- come
from the simulator object, and any task needing them declares it, so a
real-device run refuses to score rather than scoring wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ui_tree import parse_ui_dump


def _norm(s: str) -> str:
    """Match labels across the punctuation Android is inconsistent about."""
    return "".join(c for c in s.lower() if c.isalnum())


@dataclass
class PhoneState:
    source: str
    """``"sim"`` or ``"device"`` -- which facts below are trustworthy."""
    settings: dict[str, bool] = field(default_factory=dict)
    """Authoritative setting values, read out of band. Preferred over
    :attr:`switches`, which only shows what is currently on screen."""
    screen: str | None = None
    """Simulator screen id. ``None`` on a real phone, where there is no such
    thing; use :meth:`has_text` against something the screen actually shows."""
    switches: dict[str, bool] = field(default_factory=dict)
    texts: list[str] = field(default_factory=list)
    typed: list[str] = field(default_factory=list)
    fields: dict[str, str] = field(default_factory=dict)
    records: dict[str, list[dict]] = field(default_factory=dict)
    toggled: list[tuple[str, bool, bool]] = field(default_factory=list)
    """(key, before, after) for every switch the agent actually flipped."""

    def switch(self, *names: str) -> bool | None:
        """State of the first control matching any of ``names``.

        The setting wins over the screen: an agent that turned Wi-Fi on and
        then navigated away has still turned Wi-Fi on. ``None`` means neither
        source knows the control, which is not the same as off -- an agent that
        never found the screen has not turned anything on, and predicates must
        not read that as "off".
        """
        wanted = {_norm(n) for n in names}
        for key, value in self.settings.items():
            if _norm(key) in wanted:
                return bool(value)
        for label, checked in self.switches.items():
            if _norm(label) in wanted:
                return checked
        return None

    def has_text(self, *needles: str) -> bool:
        joined = " ".join(self.texts).lower()
        return any(n.lower() in joined for n in needles)

    def times_toggled(self, key: str) -> int:
        return sum(1 for k, _, _ in self.toggled if k == key)


def read_state(device) -> PhoneState:
    """Read the phone through its screen, plus the simulator's own records."""
    elements = parse_ui_dump(device.dump())
    # `checkable` is deliberately not among the flags ui_tree parses, because
    # that list is also what the model is shown; adding to it would change the
    # agent's prompt. A switch is recognised by its class or by exposing a
    # `checked` flag instead.
    switches = {e.text: bool(e.flags.get("checked", False))
                for e in elements
                if "checked" in e.flags
                or e.short_class in ("Switch", "CheckBox", "ToggleButton")}
    texts = [e.text for e in elements if e.text]

    if not hasattr(device, "screens"):
        return PhoneState(source="device", switches=switches, texts=texts,
                          settings=adb_settings(device))

    return PhoneState(
        source="sim",
        screen=device.current,
        settings={k: bool(v) for k, v in device.state.items()},
        switches=switches,
        texts=texts,
        typed=list(device.typed),
        fields=dict(getattr(device, "fields", {})),
        records={k: [dict(r) for r in v]
                 for k, v in getattr(device, "records", {}).items()},
        toggled=list(getattr(device, "toggled", [])),
    )


ADB_SETTINGS = {
    "wifi": ("global", "wifi_on"),
    "airplane": ("global", "airplane_mode_on"),
    "bluetooth": ("global", "bluetooth_on"),
}
"""Settings readable out of band, so scoring does not depend on which screen
the agent left the phone on."""


def adb_settings(device) -> dict[str, bool]:
    """Read the authoritative values from a real phone. Missing keys are
    omitted rather than defaulted -- an unreadable setting must not be scored
    as "off"."""
    out: dict[str, bool] = {}
    for key, (namespace, name) in ADB_SETTINGS.items():
        try:
            raw = device._run("shell", "settings", "get", namespace, name)
        except Exception:
            continue
        raw = raw.strip()
        if raw in ("0", "1"):
            out[key] = raw == "1"
    return out
