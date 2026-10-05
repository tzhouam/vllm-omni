"""Narrow, local Windows tool boundary for an Omni edge agent.

The model-facing side can request operations with :meth:`execute`.  Operations
that may change external or system state raise :class:`ApprovalRequired` and
can only be resumed through :meth:`approve`, which the desktop UI must keep
outside the model's tool schema.  Browser and UI text remain untrusted data.
"""

from __future__ import annotations

import ctypes
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import secrets
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from types import MappingProxyType
from urllib.parse import unquote, urljoin, urlparse


_READ_OPERATIONS = frozenset({
    "browser_open",
    "browser_read",
    "browser_follow",
    "browser_screenshot",
    "screen_capture",
    "settings_open",
    "settings_read",
    "settings_inspect",
})
_WRITE_OPERATIONS = frozenset({"browser_click", "browser_fill", "settings_set"})
_ALL_OPERATIONS = _READ_OPERATIONS | _WRITE_OPERATIONS
SUPPORTED_OPERATIONS = tuple(sorted(_ALL_OPERATIONS))
_SETTINGS_PAGES = {
    "display": "ms-settings:display",
    "sound": "ms-settings:sound",
    "mouse": "ms-settings:mousetouchpad",
    "privacy_camera": "ms-settings:privacy-webcam",
}
_APPROVAL_TTL_SECONDS = 300.0
_EXPLICIT_URL = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
_HIGH_IMPACT_CONTROL = re.compile(
    r"\b(send|submit|purchase|buy|checkout|pay|delete|remove|erase|reset|"
    r"logout|shutdown|revoke|transfer|withdraw|wire|unsubscribe|deactivate|"
    r"cancel subscription|close account|confirm order)\b"
    r"|发送|提交|购买|付款|结账|删除|移除|清空|重置|注销|提现|转账|退订|停用|关闭账户|确认订单",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolAction:
    operation: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def __post_init__(self) -> None:
        if self.operation not in _ALL_OPERATIONS:
            raise ValueError(f"unknown tool operation: {self.operation}")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")
        # Only JSON values cross the model/tool boundary.  Freeze a detached
        # copy so an approval cannot be changed by mutating the caller's dict.
        normalized = json.loads(json.dumps(dict(self.arguments), ensure_ascii=False, allow_nan=False))
        object.__setattr__(self, "arguments", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "arguments": dict(self.arguments), "request_id": self.request_id}


@dataclass(frozen=True)
class ToolResult:
    operation: str
    data: Mapping[str, Any]
    source: str
    observed_at_unix: float
    untrusted_output: bool = True


@dataclass(frozen=True)
class ApprovalChallenge:
    challenge_id: str
    action: ToolAction
    risk: str
    description: str
    target: Mapping[str, Any]
    expires_at_unix: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "action": self.action.to_dict(),
            "risk": self.risk,
            "description": self.description,
            "target": dict(self.target),
            "expires_at_unix": self.expires_at_unix,
        }


class ApprovalRequired(RuntimeError):
    def __init__(self, challenge: ApprovalChallenge) -> None:
        self.challenge = challenge
        super().__init__(f"approval required for {challenge.action.operation}")


class ToolRequestCancelled(RuntimeError):
    """The owning Agent request ended before this tool action could start."""


@dataclass
class _Pending:
    challenge: ApprovalChallenge
    context: Mapping[str, Any]
    deadline: float


class BrowserBackend(Protocol):
    def open(self, url: str) -> Mapping[str, Any]: ...
    def read(self) -> Mapping[str, Any]: ...
    def follow(self, selector: str) -> Mapping[str, Any]: ...
    def screenshot(self) -> Mapping[str, Any]: ...
    def click(self, selector: str) -> Mapping[str, Any]: ...
    def fill(self, selector: str, value: str) -> Mapping[str, Any]: ...
    def current_url(self) -> str: ...
    def describe_target(self, selector: str) -> Mapping[str, Any]: ...


class SettingsBackend(Protocol):
    def open(self, page: str) -> Mapping[str, Any]: ...
    def read(self, setting: str) -> Mapping[str, Any]: ...
    def inspect(self, page: str) -> Mapping[str, Any]: ...
    def set(self, setting: str, value: Any) -> Mapping[str, Any]: ...


class ScreenBackend(Protocol):
    def capture(self) -> Mapping[str, Any]: ...


def _require_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError("Windows desktop tools require a native Windows Python process")


def _http_url(url: str) -> str:
    if any(ord(char) < 32 or char == "\\" for char in url):
        raise ValueError("browser URL contains an unsafe character")
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("browser navigation requires an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentials in browser URL are not allowed")
    return url


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlparse(_http_url(url))
    if parsed.hostname is None:
        raise ValueError("browser URL has no host")
    return (parsed.scheme.lower(), parsed.hostname.lower(),
            parsed.port or (443 if parsed.scheme.lower() == "https" else 80))


def _explicit_task_urls(task: str) -> frozenset[str]:
    """Only literal HTTP(S) URLs in the current trusted task grant auto-open."""
    urls: set[str] = set()
    for match in _EXPLICIT_URL.finditer(task):
        candidate = match.group()
        try:
            _http_url(candidate)
        except ValueError:
            continue
        urls.add(candidate)
    return frozenset(urls)


def _high_impact_text(*values: Any) -> bool:
    # Decode URLs and split path/slug separators so /delete-account and
    # ?action=reset are checked like visible control labels. Split camelCase
    # as well: /api/deleteAll is still a state-changing GET candidate.
    text = unquote(" ".join(str(value or "") for value in values))
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"[_/\-]+", " ", text)
    return _HIGH_IMPACT_CONTROL.search(text) is not None


class ManagedEdgeBrowser:
    """One app-owned Edge profile; no arbitrary JavaScript or OS commands.

    Playwright is imported when the first browser action is executed.  Its
    synchronous context must be used from the same dedicated controller thread.
    """

    def __init__(self, profile_dir: Path | None = None, *, max_text_chars: int = 20_000) -> None:
        _require_windows()
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        self.profile_dir = profile_dir or local_app_data / "OmniEdgeAgent" / "browser-profile"
        self.max_text_chars = max_text_chars
        self._playwright: Any = None
        self._context: Any = None
        self._page: Any = None
        self._owner_thread: int | None = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omni-edge-browser")

    def _call(self, function: Any, *args: Any) -> Any:
        if self._owner_thread == threading.get_ident():
            return function(*args)
        return self._worker.submit(function, *args).result()

    def _ensure_page(self) -> Any:
        thread_id = threading.get_ident()
        if self._owner_thread is not None and thread_id != self._owner_thread:
            raise RuntimeError("browser actions must run on one controller thread")
        if self._page is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as exc:
                raise RuntimeError("install playwright and its Edge browser integration") from exc
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            self._owner_thread = thread_id
            # Omni's native-Windows vLLM compatibility shim installs a
            # Selector policy for its own IPC. Playwright starts a Node driver
            # subprocess, which requires a Proactor loop on Windows. Set the
            # policy only while Playwright creates its private loop, then
            # restore Omni's policy before continuing inference.
            old_policy = asyncio.get_event_loop_policy()
            try:
                if sys.platform == "win32":
                    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                self._playwright = sync_playwright().start()
            finally:
                asyncio.set_event_loop_policy(old_policy)
            try:
                self._context = self._playwright.chromium.launch_persistent_context(
                    str(self.profile_dir), channel="msedge", headless=False,
                    accept_downloads=False,
                )
                self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
            except Exception:
                self._playwright.stop()
                self._playwright = None
                self._owner_thread = None
                raise
        return self._page

    def current_url(self) -> str:
        return self._call(lambda: str(self._ensure_page().url))

    def open(self, url: str) -> Mapping[str, Any]:
        return self._call(self._open, url)

    def _open(self, url: str) -> Mapping[str, Any]:
        page = self._ensure_page()
        page.goto(_http_url(url), wait_until="domcontentloaded", timeout=30_000)
        return {"url": str(page.url), "title": page.title()}

    def read(self) -> Mapping[str, Any]:
        return self._call(self._read)

    def _read(self) -> Mapping[str, Any]:
        page = self._ensure_page()
        body = page.locator("body").inner_text(timeout=15_000)
        return {
            "url": str(page.url),
            "title": page.title(),
            "text": body[: self.max_text_chars],
            "truncated": len(body) > self.max_text_chars,
        }

    def screenshot(self) -> Mapping[str, Any]:
        return self._call(self._screenshot)

    def _screenshot(self) -> Mapping[str, Any]:
        page = self._ensure_page()
        image = page.screenshot(type="jpeg", quality=75, full_page=False)
        if len(image) > 2_000_000:
            image = page.screenshot(type="jpeg", quality=40, full_page=False)
        if len(image) > 2_000_000:
            raise RuntimeError("browser viewport screenshot exceeds the 2 MB tool limit")
        viewport = page.viewport_size or {}
        return {
            "url": str(page.url),
            "mime_type": "image/jpeg",
            "width": viewport.get("width"),
            "height": viewport.get("height"),
            "size_bytes": len(image),
            "sha256": hashlib.sha256(image).hexdigest(),
            "base64": base64.b64encode(image).decode("ascii"),
        }

    def follow(self, selector: str) -> Mapping[str, Any]:
        return self._call(self._follow, selector)

    def _follow(self, selector: str) -> Mapping[str, Any]:
        page = self._ensure_page()
        locator = page.locator(selector)
        if locator.count() != 1:
            raise ValueError("browser_follow requires exactly one matching link")
        if locator.evaluate("element => element.tagName") != "A":
            raise ValueError("browser_follow can only navigate a link")
        href = locator.get_attribute("href")
        if not href:
            raise ValueError("link has no URL; browser_click requires approval")
        return self._open(_http_url(urljoin(str(page.url), href)))

    def describe_target(self, selector: str) -> Mapping[str, Any]:
        return self._call(self._describe_target, selector)

    def _describe_target(self, selector: str) -> Mapping[str, Any]:
        page = self._ensure_page()
        locator = page.locator(selector)
        if locator.count() != 1:
            raise ValueError("action requires exactly one matching element")
        kind = str(locator.evaluate("element => element.tagName"))
        input_type = str(locator.get_attribute("type") or "")
        if input_type.lower() == "password":
            raise ValueError("password fields are not handled by the agent")
        return {
            "url": str(page.url),
            "tag": kind,
            "visible": bool(locator.is_visible()),
            "text": locator.inner_text(timeout=5_000)[:160] if kind not in {"INPUT", "TEXTAREA"} else "",
            "aria_label": locator.get_attribute("aria-label") or "",
            "href": locator.get_attribute("href") or "",
            "input_type": input_type,
        }

    def click(self, selector: str) -> Mapping[str, Any]:
        return self._call(self._click, selector)

    def _click(self, selector: str) -> Mapping[str, Any]:
        page = self._ensure_page()
        page.locator(selector).click(timeout=15_000)
        return {"url": str(page.url), "clicked": selector}

    def fill(self, selector: str, value: str) -> Mapping[str, Any]:
        return self._call(self._fill, selector, value)

    def _fill(self, selector: str, value: str) -> Mapping[str, Any]:
        page = self._ensure_page()
        page.locator(selector).fill(value, timeout=15_000)
        return {"url": str(page.url), "filled": selector}

    def close(self) -> None:
        self._call(self._close)
        self._worker.shutdown(wait=True)

    def _close(self) -> None:
        if self._owner_thread is not None and threading.get_ident() != self._owner_thread:
            raise RuntimeError("browser close must run on its controller thread")
        if self._context is not None:
            self._context.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._context = self._page = self._playwright = None
        self._owner_thread = None


class WindowsSettings:
    """Allowlisted Settings navigation and one documented Win32 setting.

    UI Automation is used for read-only inspection when pywinauto is present;
    mouse speed changes use SystemParametersInfoW instead of fragile UI clicks.
    """

    def __init__(self) -> None:
        _require_windows()

    def open(self, page: str) -> Mapping[str, Any]:
        uri = _SETTINGS_PAGES.get(page)
        if uri is None:
            raise ValueError(f"settings page is not allowlisted: {page}")
        os.startfile(uri)  # type: ignore[attr-defined]
        return {"page": page, "uri": uri}

    def read(self, setting: str) -> Mapping[str, Any]:
        if setting != "mouse_speed":
            raise ValueError(f"setting is not allowlisted: {setting}")
        value = ctypes.c_uint(0)
        if not ctypes.windll.user32.SystemParametersInfoW(0x0070, 0, ctypes.byref(value), 0):
            raise OSError(ctypes.get_last_error(), "SPI_GETMOUSESPEED failed")
        return {"setting": setting, "value": int(value.value), "unit": "Windows mouse speed (1–20)"}

    def inspect(self, page: str) -> Mapping[str, Any]:
        if page not in _SETTINGS_PAGES:
            raise ValueError(f"settings page is not allowlisted: {page}")
        try:
            from pywinauto import Desktop
        except ImportError as exc:
            raise RuntimeError("install pywinauto for read-only Settings UI Automation") from exc
        windows = [
            window for window in Desktop(backend="uia").windows()
            if "settings" in window.window_text().lower() or "设置" in window.window_text()
        ]
        if not windows:
            raise RuntimeError("Windows Settings is not open")
        window = windows[0]
        texts = [child.window_text() for child in window.descendants() if child.window_text()]
        return {"page": page, "window": window.window_text(), "text": "\n".join(texts)[:20_000]}

    def set(self, setting: str, value: Any) -> Mapping[str, Any]:
        if setting != "mouse_speed":
            raise ValueError(f"setting is not allowlisted: {setting}")
        if type(value) is not int or not 1 <= value <= 20:
            raise ValueError("mouse_speed must be an integer from 1 to 20")
        previous = self.read(setting)["value"]
        if not ctypes.windll.user32.SystemParametersInfoW(0x0071, 0, value, 0x03):
            raise OSError(ctypes.get_last_error(), "SPI_SETMOUSESPEED failed")
        return {"setting": setting, "previous": previous, "value": value}


class WindowsScreen:
    """Bounded native screen snapshot for a visual Agent route."""

    def __init__(self, *, max_bytes: int = 2_000_000) -> None:
        _require_windows()
        if type(max_bytes) is not int or max_bytes < 64_000:
            raise ValueError("max_bytes must be at least 64,000")
        self.max_bytes = max_bytes

    def capture(self) -> Mapping[str, Any]:
        try:
            from PIL import ImageGrab
        except ImportError as exc:
            raise RuntimeError("install Pillow for native screen capture") from exc
        image = ImageGrab.grab(all_screens=True).convert("RGB")
        original_size = image.size
        # The admitted multimodal StageClient accepts at most 1024² pixels.
        # Resize at the tool boundary and report the source dimensions so the
        # model and quality evaluator do not mistake this for a native-size
        # observation. A larger screenshot would otherwise fail only after
        # PNG conversion and model submission.
        if image.width * image.height > 1024 * 1024:
            image.thumbnail((1024, 1024))
        encoded = b""
        for quality in (85, 70, 55, 40):
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            encoded = output.getvalue()
            if len(encoded) <= self.max_bytes:
                break
        while len(encoded) > self.max_bytes and min(image.size) > 320:
            image.thumbnail((int(image.width * 0.75), int(image.height * 0.75)))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=40, optimize=True)
            encoded = output.getvalue()
        if len(encoded) > self.max_bytes:
            raise RuntimeError("screen capture exceeds the configured image byte limit")
        return {
            "mime_type": "image/jpeg",
            "width": image.width,
            "height": image.height,
            "original_width": original_size[0],
            "original_height": original_size[1],
            "resized_for_model_pixel_limit": image.size != original_size,
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "base64": base64.b64encode(encoded).decode("ascii"),
        }


class WindowsToolBoundary:
    """Model-facing local tools with exact-action, expiring UI approval.

    Expose ``execute`` to the agent.  ``approve`` and ``reject`` are intended
    only for trusted user-interface callbacks and must not be model tools.
    """

    def __init__(
        self,
        browser: BrowserBackend | None = None,
        settings: SettingsBackend | None = None,
        screen: ScreenBackend | None = None,
    ) -> None:
        self._browser = browser
        self._settings = settings
        self._screen = screen
        self._pending: dict[str, _Pending] = {}
        self._cancelled_requests: set[str] = set()
        self._task_urls: dict[str, frozenset[str]] = {}
        self._lock = threading.RLock()

    def register_user_task(self, request_id: str, task: str) -> None:
        """Record trusted user URLs before model output or recalled data is read."""
        if not request_id or not isinstance(task, str):
            raise ValueError("a request ID and trusted user task are required")
        urls = _explicit_task_urls(task)
        with self._lock:
            if request_id in self._task_urls:
                raise ValueError("user task is already registered for this request")
            self._task_urls[request_id] = urls

    def cancel_request(self, request_id: str) -> None:
        """Revoke queued actions and unused approvals for one request."""
        if not request_id:
            return
        with self._lock:
            self._cancelled_requests.add(request_id)

    def finish_request(self, request_id: str) -> None:
        """Forget a settled request after its tool worker has drained."""
        with self._lock:
            for challenge_id, pending in tuple(self._pending.items()):
                if pending.challenge.action.request_id == request_id:
                    del self._pending[challenge_id]
            self._cancelled_requests.discard(request_id)
            self._task_urls.pop(request_id, None)

    def _require_active(self, request_id: str) -> None:
        if request_id and request_id in self._cancelled_requests:
            raise ToolRequestCancelled("tool request was cancelled before execution")

    @property
    def browser(self) -> BrowserBackend:
        if self._browser is None:
            self._browser = ManagedEdgeBrowser()
        return self._browser

    @property
    def settings(self) -> SettingsBackend:
        if self._settings is None:
            self._settings = WindowsSettings()
        return self._settings

    @property
    def screen(self) -> ScreenBackend:
        if self._screen is None:
            self._screen = WindowsScreen()
        return self._screen

    def execute(self, action: ToolAction) -> ToolResult:
        self._validate(action)
        with self._lock:
            self._require_active(action.request_id)
        navigation_context: Mapping[str, Any] | None = None
        if action.operation in {"browser_open", "browser_follow"}:
            navigation_context, needs_approval = self._navigation_context(action)
            if needs_approval:
                challenge = self._propose(action, context=navigation_context)
                raise ApprovalRequired(challenge)
        if action.operation in _WRITE_OPERATIONS:
            challenge = self._propose(action)
            raise ApprovalRequired(challenge)
        with self._lock:
            self._require_active(action.request_id)
        return self._run(action, navigation_context=navigation_context)

    def _navigation_context(self, action: ToolAction) -> tuple[Mapping[str, Any], bool]:
        if action.operation == "browser_open":
            url = _http_url(str(action.arguments["url"]))
            with self._lock:
                explicitly_requested = url in self._task_urls.get(action.request_id, ())
            return {"url": url}, not explicitly_requested or _high_impact_text(url)
        context = dict(self.browser.describe_target(str(action.arguments["selector"])))
        if str(context.get("tag") or "").upper() != "A":
            raise ValueError("browser_follow requires one observed <a href> link")
        href = str(context.get("href") or "")
        if not href:
            raise ValueError("browser_follow requires one observed <a href> link")
        source_url = _http_url(str(context.get("url") or ""))
        target_url = _http_url(urljoin(source_url, href))
        context["target_url"] = target_url
        with self._lock:
            explicitly_requested = target_url in self._task_urls.get(action.request_id, ())
        # A page can hide arbitrary same-origin links, including destructive
        # GET endpoints. DOM presence and a benign label cannot grant authority.
        # Only the exact URL from the current trusted task can be automatic,
        # and even that must be visible, same-origin, and low impact.
        safe_to_auto_follow = (
            explicitly_requested and context.get("visible") is True and
            _origin(source_url) == _origin(target_url) and
            not _high_impact_text(context.get("text"), context.get("aria_label"),
                                  href, target_url)
        )
        return context, not safe_to_auto_follow

    def approve(self, challenge_id: str) -> ToolResult:
        with self._lock:
            pending = self._pending.pop(challenge_id, None)
        if pending is None:
            raise ValueError("approval challenge is unknown or already used")
        with self._lock:
            self._require_active(pending.challenge.action.request_id)
        if time.monotonic() > pending.deadline:
            raise TimeoutError("approval challenge expired")
        action = pending.challenge.action
        if action.operation == "browser_follow":
            current, _ = self._navigation_context(action)
            if current != pending.context:
                raise RuntimeError("browser target changed after approval was requested")
        elif action.operation in {"browser_click", "browser_fill"}:
            current = dict(self.browser.describe_target(str(action.arguments["selector"])))
            if current != pending.context:
                raise RuntimeError("browser target changed after approval was requested")
        elif action.operation == "settings_set":
            current = dict(self.settings.read(str(action.arguments["setting"])))
            if current != pending.context:
                raise RuntimeError("Windows setting changed after approval was requested")
        # Cancellation may have arrived while the target was being inspected.
        # This is the final admission point for a write; an action admitted
        # before cancellation must be drained and reported by the controller.
        with self._lock:
            self._require_active(action.request_id)
        return self._run(action, navigation_context=pending.context)

    def reject(self, challenge_id: str) -> None:
        with self._lock:
            if self._pending.pop(challenge_id, None) is None:
                raise ValueError("approval challenge is unknown or already used")

    def close(self) -> None:
        """Discard approval grants and release the app-owned browser profile."""
        with self._lock:
            self._pending.clear()
            self._task_urls.clear()
        browser = self._browser
        if browser is not None:
            closer = getattr(browser, "close", None)
            if callable(closer):
                closer()

    def _propose(self, action: ToolAction, *, context: Mapping[str, Any] | None = None) -> ApprovalChallenge:
        if context is None:
            if action.operation in {"browser_click", "browser_fill", "browser_follow"}:
                context = dict(self.browser.describe_target(str(action.arguments["selector"])))
            elif action.operation == "browser_open":
                context = {"url": str(action.arguments["url"])}
            elif action.operation == "settings_set":
                context = dict(self.settings.read(str(action.arguments["setting"])))
            else:
                context = {}
        # Keep the exact target used for the UI challenge immutable until
        # approval. The approved follow is resolved from this snapshot only
        # after the current DOM target has been checked against it again.
        context = MappingProxyType(json.loads(json.dumps(dict(context), ensure_ascii=False, allow_nan=False)))
        risk = "system_change" if action.operation == "settings_set" else "external_action"
        if action.operation in {"browser_open", "browser_follow"}:
            risk = "high_impact_external_action"
        if action.operation == "browser_click":
            if _high_impact_text(*(context.get(key) for key in
                                   ("text", "aria_label", "href", "url"))):
                risk = "high_impact_external_action"
        description = {
            "browser_open": "Open a URL that may send, delete, or change external data",
            "browser_follow": "Follow a link that may send, delete, or change external data",
            "browser_click": "Click a browser control; this may send, buy, delete, or change external data",
            "browser_fill": "Fill a browser control; this may save data on an external site",
            "settings_set": "Change an allowlisted Windows setting",
        }[action.operation]
        challenge = ApprovalChallenge(
            challenge_id=secrets.token_urlsafe(24),
            action=action,
            risk=risk,
            description=description,
            target=context,
            expires_at_unix=time.time() + _APPROVAL_TTL_SECONDS,
        )
        with self._lock:
            self._require_active(action.request_id)
            self._pending[challenge.challenge_id] = _Pending(
                challenge=challenge,
                context=context,
                deadline=time.monotonic() + _APPROVAL_TTL_SECONDS,
            )
        return challenge

    @staticmethod
    def _validate(action: ToolAction) -> None:
        arguments = action.arguments
        expected = {
            "browser_open": {"url"},
            "browser_read": set(),
            "browser_follow": {"selector"},
            "browser_screenshot": set(),
            "screen_capture": set(),
            "browser_click": {"selector"},
            "browser_fill": {"selector", "value"},
            "settings_open": {"page"},
            "settings_read": {"setting"},
            "settings_inspect": {"page"},
            "settings_set": {"setting", "value"},
        }[action.operation]
        if set(arguments) != expected:
            raise ValueError(f"{action.operation} requires exactly: {sorted(expected)}")
        for key in {"url", "selector", "page", "setting"} & expected:
            if not isinstance(arguments[key], str) or not arguments[key]:
                raise ValueError(f"{key} must be a nonempty string")
        if action.operation == "browser_open":
            _http_url(str(arguments["url"]))
        if action.operation.startswith("settings_"):
            if "page" in arguments and arguments["page"] not in _SETTINGS_PAGES:
                raise ValueError("settings page is not allowlisted")
            if "setting" in arguments and arguments["setting"] != "mouse_speed":
                raise ValueError("setting is not allowlisted")
        if action.operation == "browser_fill" and not isinstance(arguments["value"], str):
            raise ValueError("browser fill value must be a string")
        if action.operation == "settings_set":
            value = arguments["value"]
            if type(value) is not int or not 1 <= value <= 20:
                raise ValueError("mouse_speed must be an integer from 1 to 20")

    def _run(self, action: ToolAction, *,
             navigation_context: Mapping[str, Any] | None = None) -> ToolResult:
        op, args = action.operation, action.arguments
        if op == "browser_open":
            data = self.browser.open(str(args["url"]))
            source = str(data.get("url", args["url"]))
        elif op == "browser_read":
            data = self.browser.read()
            source = str(data.get("url", self.browser.current_url()))
        elif op == "browser_follow":
            if navigation_context is None:
                raise AssertionError("browser_follow requires an observed link target")
            # Open the observed, approved URL itself. Re-resolving the selector
            # here could navigate to a different href after the safety check.
            data = self.browser.open(str(navigation_context["target_url"]))
            source = str(data.get("url", self.browser.current_url()))
        elif op == "browser_screenshot":
            data = self.browser.screenshot()
            source = str(data.get("url", self.browser.current_url()))
        elif op == "screen_capture":
            data = self.screen.capture()
            source = "windows-screen"
        elif op == "browser_click":
            data = self.browser.click(str(args["selector"]))
            source = str(data.get("url", self.browser.current_url()))
        elif op == "browser_fill":
            data = self.browser.fill(str(args["selector"]), str(args["value"]))
            source = str(data.get("url", self.browser.current_url()))
        elif op == "settings_open":
            data = self.settings.open(str(args["page"]))
            source = str(data.get("uri", args["page"]))
        elif op == "settings_read":
            data = self.settings.read(str(args["setting"]))
            source = f"windows-setting:{args['setting']}"
        elif op == "settings_inspect":
            data = self.settings.inspect(str(args["page"]))
            source = f"windows-settings:{args['page']}"
        elif op == "settings_set":
            data = self.settings.set(str(args["setting"]), args["value"])
            source = f"windows-setting:{args['setting']}"
        else:
            raise AssertionError(op)
        return ToolResult(
            operation=op,
            data=data,
            source=source,
            observed_at_unix=time.time(),
            untrusted_output=True,
        )
