"""Turn a uiautomator dump into what a text-only agent can act on.

Spark-X2.5 never sees a screenshot, so the whole agent hinges on this file:
a screen has to arrive as a short, stable, indexed list of elements, and an
index the model returns has to map back to a point on the glass.

Two decisions shape the format, both forced by the model rather than by taste:

* **Filter hard.** A real settings screen dumps 150+ nodes, most of them
  layout containers. At 56 KiB/token of KV cache and a 1.7B model that both
  costs context and buries the answer, so only nodes that are interactive or
  carry text survive.
* **Index, do not coordinate.** The model is asked for `index: 3`, never for
  pixels -- a text model has no way to know where 3 is on the glass, and
  coordinates in the transcript would be a source of silent nonsense. The
  mapping back to a centre point happens here, where the bounds are known.

The rendered form is the one the scenario fixtures were written in, so the
decision benchmark and the live agent observe identically.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

_BOUNDS = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")

# Attributes worth showing when they are true, in the order a reader wants
# them. `checked` earns its place: the one clearly wrong action in the earlier
# measurements was tapping a Wi-Fi switch that was already on, which is only
# avoidable if the state is on the screen the model sees.
_FLAGS = ("clickable", "checked", "selected", "scrollable", "focused")


@dataclass
class Element:
    index: int
    cls: str
    text: str
    resource_id: str = ""
    package: str = ""
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
    flags: dict[str, bool] = field(default_factory=dict)
    editable: bool = False

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bounds
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def short_class(self) -> str:
        return self.cls.rsplit(".", 1)[-1] if self.cls else "View"

    def render(self) -> str:
        parts = [f"[{self.index}]", self.short_class, f'"{self.text}"']
        for flag in _FLAGS:
            if flag in self.flags:
                parts.append(f"{flag}={str(self.flags[flag]).lower()}")
        return " ".join(parts)


def _truthy(node: ET.Element, name: str) -> bool:
    return node.get(name, "false") == "true"


def _label(node: ET.Element) -> str:
    """What to call this element, preferring what a person would read."""
    for attr in ("text", "content-desc"):
        value = (node.get(attr) or "").strip()
        if value:
            return value.replace("\n", " ")
    # A bare resource id is a poor label but far better than an empty string
    # when the element is the only way forward (unlabelled icon buttons).
    rid = node.get("resource-id") or ""
    return rid.rsplit("/", 1)[-1] if rid else ""


def _interesting(node: ET.Element) -> bool:
    if any(_truthy(node, a) for a in ("clickable", "checkable", "scrollable",
                                      "long-clickable", "focusable")):
        return True
    return bool(_label(node)) and not _truthy(node, "password")


def parse_bounds(raw: str | None) -> tuple[int, int, int, int]:
    m = _BOUNDS.search(raw or "")
    if not m:
        return (0, 0, 0, 0)
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def parse_ui_dump(xml_text: str, *, max_elements: int = 60) -> list[Element]:
    """uiautomator XML -> the elements worth showing, in reading order."""
    root = ET.fromstring(xml_text)
    elements: list[Element] = []
    for node in root.iter("node"):
        if not _interesting(node):
            continue
        label = _label(node)
        clickable = _truthy(node, "clickable")
        # An unlabelled, non-interactive node tells the model nothing it can
        # use and costs it context.
        if not label and not clickable and not _truthy(node, "checkable"):
            continue
        flags = {"clickable": clickable}
        if _truthy(node, "checkable"):
            flags["checked"] = _truthy(node, "checked")
        if _truthy(node, "selected"):
            flags["selected"] = True
        if _truthy(node, "scrollable"):
            flags["scrollable"] = True
        if _truthy(node, "focused"):
            flags["focused"] = True
        elements.append(
            Element(
                index=len(elements),
                cls=node.get("class", ""),
                text=label,
                resource_id=node.get("resource-id", ""),
                package=node.get("package", ""),
                bounds=parse_bounds(node.get("bounds")),
                flags=flags,
                editable="EditText" in (node.get("class") or ""),
            )
        )
        if len(elements) >= max_elements:
            break
    return elements


def render_tree(elements: list[Element]) -> str:
    """The exact text the model is shown."""
    return "\n".join(e.render() for e in elements)


def screen_signature(elements: list[Element]) -> str:
    """A cheap identity for a screen, to notice when an action changed nothing.

    Labels and flags only: coordinates shift with animations and scroll offsets,
    and treating those as a new screen would hide a stuck agent.
    """
    return "|".join(f"{e.short_class}:{e.text}:{e.flags.get('checked', '')}"
                    for e in elements)
