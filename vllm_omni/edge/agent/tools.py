"""Narrow, local Windows tool boundary for an Omni edge agent.

The model-facing side can request operations with :meth:`execute`.  Operations
that may change external or system state raise :class:`ApprovalRequired` and
can only be resumed through :meth:`approve`, which the desktop UI must keep
outside the model's tool schema.  Browser and UI text remain untrusted data.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import ctypes
import hashlib
import hmac
import io
import ipaddress
import json
import ntpath
import os
import re
import secrets
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import quote, unquote, urljoin, urlparse, urlsplit, urlunsplit

import psutil

from vllm_omni.engine.resource_ledger import ResourceUnavailable

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
_WRITE_OPERATIONS = frozenset({"browser_click", "browser_fill", "browser_post", "settings_set"})
_ALL_OPERATIONS = _READ_OPERATIONS | _WRITE_OPERATIONS
SUPPORTED_OPERATIONS = tuple(sorted(_ALL_OPERATIONS))
_SETTINGS_PAGES = {
    "display": "ms-settings:display",
    "sound": "ms-settings:sound",
    "mouse": "ms-settings:mousetouchpad",
    "privacy_camera": "ms-settings:privacy-webcam",
}
_SETTINGS_PAGE_HEADINGS = {
    "display": frozenset({"Display", "Screen", "显示", "屏幕"}),
    "sound": frozenset({"Sound", "声音"}),
    "mouse": frozenset({"Mouse", "鼠标"}),
    "privacy_camera": frozenset({"Camera", "相机"}),
}
_PAGE_HEADING_IDS = frozenset({"pagetitle", "pageheader", "settingspagetitle"})
_APPROVAL_TTL_SECONDS = 300.0
_MAX_POST_BODY_BYTES = 64 * 1024
_MAX_POST_URL_BYTES = 2048
_MAX_POST_RESPONSE_EXCERPT_BYTES = 4096
_POST_TOTAL_TIMEOUT_SECONDS = 30.0
_SCREEN_CAPTURE_SCOPE = "visible_screen_pixels_within_foreground_window_bounds"
_POST_CONTENT_TYPES = frozenset({
    "application/json", "application/x-www-form-urlencoded",
    "text/plain; charset=utf-8", "application/octet-stream",
})
_MANAGED_BROWSER_WRITE_BLOCK_REASON = (
    "Managed Edge browser clicks and fills are disabled until native interactions "
    "can be isolated and verified"
)
# Treat prose punctuation as a boundary, including punctuation used next to
# URLs in Chinese.  A bare URL containing literal Han characters is ambiguous
# with adjacent prose; require an exact UI approval for that uncommon case.
_EXPLICIT_URL = re.compile(
    r"https?://[^\s<>\"'`，。；：！？、（）【】《》“”‘’,;!]+", re.IGNORECASE,
)
_TRAILING_URL_PUNCTUATION = ".:?)]}"
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


class BrowserResourcePostconditionFailed(ResourceUnavailable):  # noqa: N818
    """Operation finished/failed before its postcondition refused more work.

    A completed ToolResult must be recorded by the controller before stopping
    the turn. An uncertain write must never be retried under the same approval.
    This exception does not release the model or browser resource reservation.
    """

    def __init__(self, completed_output: Any = None, *, operation_error: BaseException | None = None) -> None:
        self.completed_output = completed_output
        self.operation_error = operation_error
        super().__init__(
            "browser resource postcondition failed after an operation; "
            "preserve its outcome and do not retry automatically")

    @property
    def completed_result(self) -> ToolResult | None:
        return self.completed_output if isinstance(self.completed_output, ToolResult) else None


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


class ApprovalRequired(RuntimeError):  # noqa: N818 - public tool approval signal
    def __init__(self, challenge: ApprovalChallenge) -> None:
        self.challenge = challenge
        super().__init__(f"approval required for {challenge.action.operation}")


class ToolRequestCancelled(RuntimeError):  # noqa: N818 - public tool cancellation signal
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
    def post_context(self, url: str) -> Mapping[str, Any]: ...
    def post_exact(self, url: str, body: bytes, content_type: str,
                   cookie_fingerprint: str, page_url: str) -> Mapping[str, Any]: ...


class SettingsBackend(Protocol):
    def open(self, page: str) -> Mapping[str, Any]: ...
    def read(self, setting: str) -> Mapping[str, Any]: ...
    def inspect(self, page: str) -> Mapping[str, Any]: ...
    def set(self, setting: str, value: Any) -> Mapping[str, Any]: ...


class ScreenBackend(Protocol):
    def capture(self) -> Mapping[str, Any]: ...
    def capture_target(self) -> Mapping[str, Any]: ...
    def capture_approved(self, expected_target: Mapping[str, Any]) -> Mapping[str, Any]: ...


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


def _post_url(url: str) -> str:
    # Playwright's HTTP client uses the WHATWG URL parser, which removes dot
    # segments (including %2e variants) before sending. Admit only a narrow
    # spelling that is already canonical; approval must display the network
    # target, not a different URL that the client later normalizes.
    if not isinstance(url, str) or not url.isascii():
        raise ValueError("browser POST requires an exact ASCII URL")
    if len(url.encode("ascii")) > _MAX_POST_URL_BYTES:
        raise ValueError("browser POST URL exceeds 2048 bytes")
    target = _http_url(url)
    parsed = urlsplit(target)
    if (parsed.scheme not in {"http", "https"} or parsed.fragment or "#" in target
            or any(char.isspace() for char in target) or "%" in target):
        raise ValueError("browser POST URL has ambiguous encoding or fragment")
    host = parsed.hostname
    if host is None or not re.fullmatch(r"[a-z0-9.-]+", host) or host.endswith("."):
        raise ValueError("browser POST host is not canonical")
    if re.fullmatch(r"[0-9.]+", host):
        try:
            if str(ipaddress.IPv4Address(host)) != host:
                raise ValueError("browser POST host is not canonical")
        except ipaddress.AddressValueError as exc:
            raise ValueError("browser POST host is not canonical") from exc
    elif not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
                 for label in host.split(".")):
        raise ValueError("browser POST host is not canonical")
    port = parsed.port  # Raises on malformed or out-of-range ports.
    if port == 0:
        raise ValueError("browser POST port zero is not a destination")
    if port in {80 if parsed.scheme == "http" else 443}:
        raise ValueError("browser POST default port must be omitted")
    canonical_authority = host if port is None else f"{host}:{port}"
    if parsed.netloc != canonical_authority:
        raise ValueError("browser POST authority is not canonical")
    path = parsed.path
    if (not path.startswith("/") or "//" in path
            or any(segment in {".", ".."} for segment in path.split("/"))
            or not re.fullmatch(r"/[A-Za-z0-9._~/-]*", path)):
        raise ValueError("browser POST path is not canonical")
    if not re.fullmatch(r"[A-Za-z0-9._~=&+/-]*", parsed.query):
        raise ValueError("browser POST query is not canonical")
    if urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, "")) != target:
        raise ValueError("browser POST URL is not canonical")
    return target


def _post_body(body_b64: str) -> bytes:
    if not isinstance(body_b64, str):
        raise ValueError("browser POST body must be canonical base64")
    if len(body_b64) > ((_MAX_POST_BODY_BYTES + 2) // 3) * 4:
        raise ValueError("browser POST body exceeds the 64 KiB limit")
    try:
        body = base64.b64decode(body_b64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("browser POST body must be canonical base64") from exc
    if len(body) > _MAX_POST_BODY_BYTES or base64.b64encode(body).decode("ascii") != body_b64:
        raise ValueError("browser POST body must be canonical base64 under 64 KiB")
    return body


def _post_cookie_header(cookies: list[dict[str, Any]]) -> str:
    # Chromium has already selected only cookies applicable to the target.
    # Refuse values that cannot be represented in one unambiguous HTTP header.
    pairs: list[str] = []
    for cookie in cookies:
        name, value = cookie.get("name"), cookie.get("value")
        if (not isinstance(name, str) or not isinstance(value, str)
                or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name)
                or not re.fullmatch(r"[\x21\x23-\x2b\x2d-\x3a\x3c-\x5b\x5d-\x7e]*", value)):
            raise ValueError("browser POST cookie cannot be sent exactly")
        pairs.append(f"{name}={value}")
    return "; ".join(pairs)


async def _streamed_post(target: str, body: bytes, content_type: str,
                         cookie_header: str) -> Mapping[str, Any]:
    """Make one HTTP request and retain at most a bounded raw response excerpt."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("browser POST requires httpx") from exc

    headers = {"Content-Type": content_type, "Accept-Encoding": "identity"}
    if cookie_header:
        headers["Cookie"] = cookie_header
    # A fresh transport neither inherits proxy/certificate environment variables
    # nor shares a cookie jar with the managed browser page. asyncio.timeout is
    # the wall-clock deadline; httpx's own timeout is only per network operation.
    async with asyncio.timeout(_POST_TOTAL_TIMEOUT_SECONDS):
        transport = httpx.AsyncHTTPTransport(
            retries=0, trust_env=False, http2=False,
        )
        async with httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(5.0), http2=False,
        ) as client:
            request = client.build_request("POST", target, content=body, headers=headers)
            if str(request.url) != target or request.content != body:
                raise ValueError("browser POST client would rewrite the approved URL or body")
            response = await client.send(request, stream=True, follow_redirects=False)
            try:
                if str(response.url) != target:
                    raise RuntimeError("browser POST response target changed unexpectedly")
                excerpt = bytearray()
                complete = True
                async for chunk in response.aiter_raw(
                    chunk_size=_MAX_POST_RESPONSE_EXCERPT_BYTES + 1,
                ):
                    remaining = _MAX_POST_RESPONSE_EXCERPT_BYTES - len(excerpt)
                    if len(chunk) > remaining:
                        excerpt.extend(chunk[:remaining])
                        complete = False
                        break
                    excerpt.extend(chunk)
                status = int(response.status_code)
                return {
                    "url": target,
                    "status": status,
                    "redirect_followed": False,
                    "redirect_blocked": 300 <= status < 400,
                    "body_excerpt": bytes(excerpt).decode("utf-8", errors="replace"),
                    "body_excerpt_raw": True,
                    "body_excerpt_note": (
                        "Raw response bytes, possibly compressed, decoded as UTF-8 "
                        "with replacement; body_size is unknown when truncated"
                    ),
                    "content_encoding": response.headers.get("content-encoding", "identity"),
                    "body_truncated": not complete,
                    "body_size": len(excerpt) if complete else None,
                }
            finally:
                await response.aclose()


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlparse(_http_url(url))
    if parsed.hostname is None:
        raise ValueError("browser URL has no host")
    return (parsed.scheme.lower(), parsed.hostname.lower(),
            parsed.port if parsed.port is not None else
            (443 if parsed.scheme.lower() == "https" else 80))


def _network_target(url: str) -> tuple[str, str, int, str, str, str]:
    """Compare the actual HTTP request with an exact authorized navigation."""
    parsed = urlparse(_http_url(url))
    scheme, host, port = _origin(url)
    # Chromium percent-encodes raw Unicode in a path/query before Playwright
    # exposes request.url. Encode only non-ASCII code points on both sides;
    # decoding or normalizing ASCII escapes would collapse distinct network
    # targets such as /%2F and // or change an approved query byte sequence.
    def browser_unicode(value: str) -> str:
        return "".join(char if ord(char) < 128 else quote(char, safe="")
                       for char in value)

    return (scheme, host, port, browser_unicode(parsed.path or "/"),
            browser_unicode(parsed.params), browser_unicode(parsed.query))


def _explicit_task_urls(task: str) -> frozenset[str]:
    """Only literal HTTP(S) URLs in the current trusted task grant auto-open."""
    urls: set[str] = set()
    for match in _EXPLICIT_URL.finditer(task):
        # An unquoted comma/semicolon/exclamation followed immediately by
        # more URL-shaped text could itself be part of a URL.  Do not grant
        # authority to the shorter prefix in that ambiguous spelling.
        suffix = task[match.end():]
        if (len(suffix) > 1 and suffix[0] in ",;!" and
                not suffix[1].isspace() and suffix[1] not in ")]}>\"'`"):
            continue
        candidate = match.group().rstrip(_TRAILING_URL_PUNCTUATION)
        if any("\u3400" <= char <= "\u9fff" for char in candidate):
            continue
        try:
            _http_url(candidate)
        except ValueError:
            continue
        urls.add(candidate)
    return frozenset(urls)


def _exact_user_url(url: str) -> str:
    """Validate a URL supplied in a dedicated field, without parsing prose."""
    if not isinstance(url, str):
        raise ValueError("an exact HTTP(S) URL is required")
    _http_url(url)
    if any(char.isspace() for char in url):
        raise ValueError("an exact HTTP(S) URL cannot contain whitespace")
    parsed = urlsplit(url)
    # Commas and semicolons are legitimate in a path/query, but never in an
    # authority. Refuse pasted `host,https://other` rather than trusting the
    # apparent prefix; also reject malformed port spellings before admission.
    if (parsed.hostname is None or any(char in parsed.netloc for char in ",;!")
            or parsed.netloc.endswith(":")):
        raise ValueError("an exact HTTP(S) URL has an invalid authority")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("an exact HTTP(S) URL has an invalid port") from exc
    return url


def _high_impact_text(*values: Any) -> bool:
    # Decode URLs and split path/slug separators so /delete-account and
    # ?action=reset are checked like visible control labels. Split camelCase
    # as well: /api/deleteAll is still a state-changing GET candidate.
    text = unquote(" ".join(str(value or "") for value in values))
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"[_/\-]+", " ", text)
    return _HIGH_IMPACT_CONTROL.search(text) is not None


def _playwright_owned_node_spawn(playwright: Any) -> dict[str, Any]:
    """Pin a private ownership seam; never discover a Node PID by name."""
    import importlib.metadata

    if (importlib.metadata.version("playwright") != "1.63.0"
            or sys.implementation.name != "cpython" or tuple(sys.version_info[:3]) != (3, 12, 10)):
        raise RuntimeError("Node spawn ownership requires reviewed Playwright 1.63.0 / CPython 3.12.10")
    process = playwright._impl_obj._connection._transport._proc
    transport = process._transport
    popen = transport._proc
    if (type(process).__module__ != "asyncio.subprocess"
            or type(transport).__module__ != "asyncio.windows_events"
            or type(popen).__module__ != "asyncio.windows_utils"
            or type(process.pid) is not int or process.pid <= 0
            or popen.pid != process.pid):
        raise RuntimeError("reviewed Playwright subprocess ownership shape is unavailable")
    handle = int(popen._handle)
    args = popen.args
    if (handle <= 0 or not isinstance(args, (list, tuple)) or not args
            or not isinstance(args[0], str) or not ntpath.isabs(args[0])
            or ntpath.basename(args[0]).casefold() != "node.exe"):
        raise RuntimeError("reviewed owned Node spawn handle or image is unavailable")
    # The receiver duplicates this handle synchronously. The original remains
    # Playwright-owned and must never be closed by the observer.
    return {"pid": process.pid, "spawn_handle": handle, "expected_image": args[0],
            "playwright_version": "1.63.0", "python_version": "3.12.10"}


@contextmanager
def _browser_resource_scope(guard: Callable[[], None] | None) -> Iterator[None]:
    """Optional caller-thread admission before/after operations, never cleanup."""
    if guard is None:
        yield
        return
    guard()
    try:
        yield
    except BaseException:
        # A failed postcondition takes priority (with the original exception
        # as context), so an approval signal cannot mask memory refusal.
        guard()
        raise
    else:
        guard()


def _guarded_browser_call(guard: Callable[[], None] | None, function: Callable[..., Any], *args: Any) -> Any:
    if guard is None:
        return function(*args)
    guard()
    try:
        result = function(*args)
    except BaseException as original:
        try:
            guard()
        except BaseException as failure:
            if isinstance(original, BrowserResourcePostconditionFailed):
                original.add_note("outer browser resource postcondition also refused continued work")
                raise original from failure
            raise BrowserResourcePostconditionFailed(operation_error=original) from failure
        raise
    try:
        guard()
    except BaseException as failure:
        raise BrowserResourcePostconditionFailed(result) from failure
    return result


class ManagedEdgeBrowser:
    """One isolated app-owned Edge context; no arbitrary JavaScript or OS commands.

    Playwright is imported when the first browser action is executed.  Its
    synchronous context must be used from the same dedicated controller thread.
    ``profile_dir`` remains an app configuration path for compatibility but
    is never passed to Chromium; no stored tabs or cookies are restored.
    """

    def __init__(self, profile_dir: Path | None = None, *, max_text_chars: int = 20_000,
                 headless: bool = False,
                 process_observer: Callable[[str, Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
                 resource_guard: Callable[[], None] | None = None) -> None:
        _require_windows()
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        # Do not open any existing directory as a Chromium user-data-dir.
        self.profile_dir = profile_dir or local_app_data / "OmniEdgeAgent" / "browser-profile"
        self.max_text_chars = max_text_chars
        self._headless = headless
        self._process_observer = process_observer
        self._resource_guard = resource_guard
        self._process_observer_errors: list[str] = []
        self._process_observer_error_count = 0
        self._process_cdp: Any = None
        self.process_memory_close_receipt: Mapping[str, Any] | None = None
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._owner_thread: int | None = None
        self._navigation_target: str | None = None
        self._navigation_target_used = False
        self._blocked_navigation: str | None = None
        self._blocked_request_count = 0
        self._cookie_fingerprint_key = secrets.token_bytes(32)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omni-edge-browser")

    def _call(self, function: Any, *args: Any) -> Any:
        closing = function == self._close
        if self._owner_thread == threading.get_ident():
            if self._resource_guard is not None and not closing:
                raise RuntimeError("guarded browser operations must originate outside the owner worker")
            return self._observed_call(function, args)
        # Manager close may hold its lifecycle lock while waiting for this
        # worker. Guard only on the caller, and never on the close path.
        return _guarded_browser_call(
            None if closing else self._resource_guard,
            lambda: self._worker.submit(self._observed_call, function, args).result(),
        )

    def _process_notify(self, action: str, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self._process_observer is None:
            return None
        try:
            return self._process_observer(action, payload)
        except Exception as exc:
            self._process_observer_error_count += 1
            if len(self._process_observer_errors) < 16:
                self._process_observer_errors.append(action[:64] + ":" + type(exc).__name__)
            return None

    def _process_checkpoint(self, checkpoint: str = "tool_boundary") -> None:
        if self._process_observer is None or self._browser is None:
            return
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("browser CDP memory checkpoints require the owner worker")
        cutoff = self._process_notify("cdp_begin", {"checkpoint": checkpoint})
        if not isinstance(cutoff, Mapping) or type(cutoff.get("cutoff_filetime_100ns")) is not int:
            self._process_notify("cdp_unavailable", {"reason": "pre-CDP FILETIME cutoff unavailable"})
            return
        try:
            if self._process_cdp is None:
                self._process_cdp = self._browser.new_browser_cdp_session()
            info = self._process_cdp.send("SystemInfo.getProcessInfo")
            self._process_notify("cdp_membership", {
                "cutoff_filetime_100ns": cutoff["cutoff_filetime_100ns"],
                "process_info": info.get("processInfo") if isinstance(info, Mapping) else None,
                "checkpoint": checkpoint,
            })
        except Exception as exc:
            self._process_notify("cdp_unavailable", {"reason": "owned CDP failed:" + type(exc).__name__})

    def _observed_call(self, function: Any, args: tuple[Any, ...]) -> Any:
        if self._process_observer is None or function == self._close:
            return function(*args)
        self._process_checkpoint("before_tool")
        try:
            return function(*args)
        finally:
            self._process_checkpoint("after_tool")

    def _ensure_page(self) -> Any:
        thread_id = threading.get_ident()
        if self._owner_thread is not None and thread_id != self._owner_thread:
            raise RuntimeError("browser actions must run on one controller thread")
        if self._page is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as exc:
                raise RuntimeError("install playwright and its Edge browser integration") from exc
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
                self._process_notify("node_start_attempted", {})
                self._playwright = sync_playwright().start()
            except Exception as exc:
                self._process_notify("node_start_failed", {"reason": "Playwright start failed:" + type(exc).__name__})
                self._owner_thread = None
                raise
            finally:
                asyncio.set_event_loop_policy(old_policy)
            if self._process_observer is not None:
                try:
                    self._process_notify("node_spawn", _playwright_owned_node_spawn(self._playwright))
                except Exception as exc:
                    self._process_notify("node_spawn_unavailable", {
                        "reason": "owned Node spawn unavailable:" + type(exc).__name__,
                    })
            try:
                self._process_notify("browser_launch_started", {})
                # A persistent profile may restore a page before Playwright can
                # install its route. An isolated context begins with no pages;
                # install both guards before creating the first page.
                self._browser = self._playwright.chromium.launch(
                    channel="msedge", headless=self._headless,
                )
                self._process_checkpoint("after_browser_launch")
                self._context = self._browser.new_context(
                    accept_downloads=False, service_workers="block",
                )
                if self._context.pages:
                    raise RuntimeError("isolated Edge context unexpectedly contains a page")
                self._context.route("**/*", self._guard_navigation)
                # WebSocket traffic is not covered by HTTP routing.  A page
                # read must not be able to open a separate write channel.
                self._context.route_web_socket("**/*", lambda socket: socket.close())
                self._page = self._context.new_page()
                self._process_checkpoint("after_page_creation")
            except BaseException as failure:
                self._cleanup_browser_objects(failure)
                raise
        return self._page

    def current_url(self) -> str:
        return self._call(lambda: str(self._ensure_page().url))

    def _post_snapshot(self, url: str) -> tuple[dict[str, Any], str]:
        page = self._ensure_page()
        target = _post_url(url)
        page_url = str(page.url)
        if _origin(page_url) != _origin(target):
            raise PermissionError("browser POST must target the current page origin")
        # Only cookies applicable to the exact destination enter the request.
        # HMAC avoids disclosing even short cookie values through the UI hash.
        cookies = json.loads(json.dumps(self._context.cookies(target), allow_nan=False))
        cookie_header = _post_cookie_header(cookies)
        canonical = json.dumps(
            {"cookies_in_wire_order": cookies, "cookie_header": cookie_header},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        fingerprint = hmac.new(
            self._cookie_fingerprint_key, canonical, hashlib.sha256,
        ).hexdigest()
        return ({
            "current_url": page_url,
            "cookie_fingerprint": fingerprint,
            "cookie_count": len(cookies),
        }, cookie_header)

    def post_context(self, url: str) -> Mapping[str, Any]:
        return self._call(lambda: self._post_snapshot(url)[0])

    def post_exact(self, url: str, body: bytes, content_type: str,
                   cookie_fingerprint: str, page_url: str) -> Mapping[str, Any]:
        return self._call(self._post_exact, url, body, content_type,
                          cookie_fingerprint, page_url)

    def _post_exact(self, url: str, body: bytes, content_type: str,
                    cookie_fingerprint: str, page_url: str) -> Mapping[str, Any]:
        target = _post_url(url)
        if type(body) is not bytes or len(body) > _MAX_POST_BODY_BYTES:
            raise ValueError("browser POST body must be bytes under 64 KiB")
        if content_type not in _POST_CONTENT_TYPES:
            raise ValueError("browser POST Content-Type is not allowlisted")
        context, cookie_header = self._post_snapshot(target)
        if (context["current_url"] != page_url or
                not hmac.compare_digest(context["cookie_fingerprint"], cookie_fingerprint)):
            raise RuntimeError("browser page or cookie context changed after POST approval")
        # The snapshot is the only cookie state this one request can send.
        # Playwright's APIRequestContext buffers whole responses in memory,
        # including unbounded chunked bodies. Use an isolated streaming HTTP
        # client and close it after a bounded raw excerpt instead. Playwright's
        # sync driver runs an event loop on this browser worker thread, so the
        # async transport must run in its own short-lived thread.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="omni-edge-post") as worker:
            future = worker.submit(
                lambda: asyncio.run(_streamed_post(target, body, content_type, cookie_header)),
            )
            try:
                # Never abandon an in-flight write. Even if the transport's
                # deadline fails to interrupt an OS call, the owning turn must
                # retain its gate until this worker has finished.
                return future.result()
            except Exception as exc:
                # Once the attempt begins, timeout or disconnection cannot
                # reveal whether the server committed it. Never retry under
                # this approval.
                raise RuntimeError(
                    "browser POST outcome unknown; it may have reached the server; do not retry automatically"
                ) from exc

    def bring_to_front(self) -> Mapping[str, Any]:
        """Select the managed tab for a trusted desktop fixture setup."""
        def select() -> Mapping[str, Any]:
            page = self._ensure_page()
            page.bring_to_front()
            return {"url": str(page.url), "title": page.title()}

        return self._call(select)

    def open(self, url: str) -> Mapping[str, Any]:
        return self._call(self._open, url)

    def _open(self, url: str) -> Mapping[str, Any]:
        page = self._ensure_page()
        target = _http_url(url)
        self._navigation_target = target
        self._navigation_target_used = False
        self._blocked_navigation = None
        blocked_before = self._blocked_request_count
        try:
            page.goto(target, wait_until="domcontentloaded", timeout=30_000)
        except Exception as exc:
            if self._blocked_navigation is not None:
                # Chromium may still be loading its internal network-error
                # page after an aborted navigation.  Replace that tab so the
                # next approved navigation cannot race the error page.
                page.close()
                self._page = self._context.new_page()
                raise PermissionError(self._blocked_navigation) from exc
            raise
        finally:
            self._navigation_target = None
            self._navigation_target_used = False
        if self._blocked_navigation is not None:
            raise PermissionError(self._blocked_navigation)
        return {
            "url": str(page.url), "title": page.title(),
            "blocked_network_requests": self._blocked_request_count - blocked_before,
            "blocked_network_requests_total": self._blocked_request_count,
        }

    def _block_request(self, route: Any, reason: str, *, navigation: bool = False) -> None:
        self._blocked_request_count += 1
        if navigation:
            self._blocked_navigation = reason
        route.abort("blockedbyclient")

    def _guard_navigation(self, route: Any) -> None:
        """Keep page-initiated writes and unreviewed redirects off the wire.

        Playwright routes only the first URL in a redirect chain.  Fetching the
        first response with redirects disabled lets us withhold a 3xx before
        Chromium can send the second request, for documents and subresources.
        """
        request = route.request
        if self._page is None:
            self._block_request(route, "browser page is not ready")
            return
        navigation = request.is_navigation_request()
        if navigation:
            try:
                top_level = request.frame == self._page.main_frame
            except Exception:
                top_level = False
            if not top_level:
                # An iframe or popup has no separately approved destination.
                self._block_request(route, "unapproved document frame", navigation=False)
                return
            try:
                authorized_target = (
                    self._navigation_target is not None and
                    not self._navigation_target_used and
                    _network_target(request.url) == _network_target(self._navigation_target) and
                    request.method.upper() == "GET"
                )
            except ValueError:
                authorized_target = False
            if not authorized_target:
                self._block_request(
                    route, "browser navigation target changed before request", navigation=True,
                )
                return
            # A page cannot reload the approved URL repeatedly while
            # page.goto is still waiting for DOMContentLoaded.
            self._navigation_target_used = True
        else:
            # HTTP method and URL spelling do not prove a request is read-only.
            # A script, image, stylesheet or favicon can issue a state-changing
            # GET at an innocent-looking path or exfiltrate data in its query.
            # Only the exact top-level navigation may reach the network.
            # Inline/data assets remain available.
            self._block_request(route, "background HTTP request requires separate approval")
            return
        try:
            response = route.fetch(max_redirects=0, timeout=30_000)
        except Exception:
            route.abort("failed")
            return
        response_error: BaseException | None = None
        try:
            location = response.headers.get("location")
            if 300 <= response.status < 400 and location:
                self._block_request(
                    route, "HTTP redirect requires separate URL approval",
                    navigation=navigation,
                )
                return
            # The response body must remain available until synchronous fulfill returns.
            route.fulfill(response=response)
        except BaseException as exc:
            response_error = exc
            raise
        finally:
            try:
                response.dispose()
            except BaseException as cleanup_error:
                if response_error is not None:
                    # Keep navigation failure primary; expose cleanup failure
                    # through exception chaining.
                    raise response_error from cleanup_error
                raise

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
            # A page can schedule traffic after open() returns.  The running
            # total lets the next observation expose those blocked attempts.
            "blocked_network_requests_total": self._blocked_request_count,
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
            "blocked_network_requests_total": self._blocked_request_count,
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
            **self._describe_form_submit(locator),
        }

    @staticmethod
    def _describe_form_submit(locator: Any) -> Mapping[str, Any]:
        form = locator.evaluate("""element => {
            const owner = element.form;
            if (!owner || element.type !== 'submit') return null;
            const action = element.hasAttribute('formaction') ? element.formAction : owner.action;
            const method = (element.hasAttribute('formmethod') ?
                element.formMethod : owner.method).toUpperCase();
            const target = (element.hasAttribute('formtarget') ?
                element.formTarget : owner.target) || '_self';
            const enctype = element.hasAttribute('formenctype') ?
                element.formEnctype : owner.enctype;
            // Constructing FormData fires the page's `formdata` event and is
            // therefore not a read-only target description. Native form writes
            // are currently disabled; report metadata without invoking it.
            return {action, method, target, enctype, eligible: false};
        }""")
        if form is None:
            return {}
        action = str(form.get("action") or "")
        try:
            _http_url(action)
        except ValueError:
            form["eligible"] = False
        return {f"form_{key}": value for key, value in form.items()}

    def click(self, selector: str) -> Mapping[str, Any]:
        raise PermissionError(_MANAGED_BROWSER_WRITE_BLOCK_REASON)

    def click_authorized(self, selector: str, target: Mapping[str, Any]) -> Mapping[str, Any]:
        raise PermissionError(_MANAGED_BROWSER_WRITE_BLOCK_REASON)

    def _click(self, selector: str, target: Mapping[str, Any] | None) -> Mapping[str, Any]:
        raise PermissionError(_MANAGED_BROWSER_WRITE_BLOCK_REASON)

    def fill(self, selector: str, value: str) -> Mapping[str, Any]:
        raise PermissionError(_MANAGED_BROWSER_WRITE_BLOCK_REASON)

    def _fill(self, selector: str, value: str) -> Mapping[str, Any]:
        raise PermissionError(_MANAGED_BROWSER_WRITE_BLOCK_REASON)

    def close(self) -> None:
        failure: BaseException | None = None
        worker_joined = False
        try:
            self._call(self._close)
        except BaseException as exc:
            failure = exc
        finally:
            try:
                self._worker.shutdown(wait=True)
                worker_joined = True
            except BaseException as exc:
                if failure is None:
                    failure = exc
            if self._process_observer is not None:
                receipt = self._process_notify("children_closed", {})
                self.process_memory_close_receipt = {
                    "worker_joined": worker_joined,
                    "bound_set": dict(receipt) if isinstance(receipt, Mapping) else None,
                    "observer_callback_error_count": self._process_observer_error_count,
                    "observer_callback_errors": list(self._process_observer_errors),
                    "all_descendants_retired": False,
                }
                if (not isinstance(receipt, Mapping) or receipt.get("bound_set_drain_verified") is not True
                        or self._process_observer_error_count or not worker_joined):
                    if failure is None:
                        failure = RuntimeError("browser observed bound-set drain is unverified; quarantine required")
        if failure is not None:
            raise failure

    def _cleanup_browser_objects(self, failure: BaseException | None = None) -> BaseException | None:
        self._process_checkpoint("before_browser_close")
        if self._process_cdp is not None:
            try:
                self._process_cdp.detach()
            except BaseException as exc:
                self._process_notify("cdp_detach_failed", {"reason": "CDP detach failed:" + type(exc).__name__})
                if failure is None:
                    failure = exc
                else:
                    failure.add_note("secondary browser cleanup failure: CDP detach:" + type(exc).__name__)
            finally:
                self._process_cdp = None
        try:
            for label, owned, method in (
                ("context", self._context, "close"), ("browser", self._browser, "close"),
                ("playwright", self._playwright, "stop"),
            ):
                if owned is None:
                    continue
                try:
                    getattr(owned, method)()
                except BaseException as exc:
                    self._process_notify("browser_cleanup_failed", {
                        "reason": "owned cleanup failed:" + label + ":" + type(exc).__name__,
                    })
                    if failure is None:
                        failure = exc
                    else:
                        failure.add_note("secondary browser cleanup failure:" + label + ":" + type(exc).__name__)
        finally:
            self._context = self._page = self._browser = self._playwright = None
            self._owner_thread = None
        return failure

    def _close(self) -> None:
        if self._owner_thread is not None and threading.get_ident() != self._owner_thread:
            raise RuntimeError("browser close must run on its controller thread")
        failure = self._cleanup_browser_objects()
        if failure is not None:
            raise failure


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
        matched: list[tuple[Any, list[str]]] = []
        for window in Desktop(backend="uia").windows():
            title = window.window_text().strip()
            if title.casefold() not in {"settings", "设置"}:
                continue
            try:
                process_name = psutil.Process(window.process_id()).name().casefold()
            except (psutil.Error, OSError):
                continue
            if process_name not in {"applicationframehost.exe", "systemsettings.exe"}:
                continue
            descendants = window.descendants()
            # A navigation entry may also say "Display" or "Camera".  Only a
            # Settings breadcrumb or a page-title UIA control identifies the
            # active page; general descendant text cannot establish it.
            observed: set[str] = set()
            for child in descendants:
                automation_id = str(getattr(child.element_info, "automation_id", "")).casefold()
                if automation_id == "permanentnavigationviewbreadcrumbbar":
                    crumbs = [item.window_text().strip() for item in child.children()
                              if item.element_info.control_type == "Button" and
                              item.window_text().strip()]
                    if crumbs:
                        observed.add(crumbs[-1])
                elif automation_id in _PAGE_HEADING_IDS:
                    observed.add(child.window_text().strip())
            page_headings = {
                key for label in observed
                for key, labels in _SETTINGS_PAGE_HEADINGS.items()
                if label in labels
            }
            if page_headings == {page} and observed:
                matched.append((window, [child.window_text() for child in descendants
                                         if child.window_text()]))
        if len(matched) != 1:
            raise RuntimeError("the requested Windows Settings page cannot be verified")
        window, texts = matched[0]
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


def _foreground_window_bbox() -> tuple[int, int, int, int]:
    """Return the foreground window's screen rectangle in desktop coordinates."""
    class Rect(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
    user32.IsIconic.argtypes = [ctypes.c_void_p]
    user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(Rect)]
    hwnd = user32.GetForegroundWindow()
    if not hwnd or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        raise RuntimeError("no visible foreground window is available for capture")
    rect = Rect()
    # DWM frame bounds are physical pixels, unlike the DPI-virtualized
    # GetWindowRect returned to some processes.  Fall back for old Windows.
    dwmapi = ctypes.windll.dwmapi
    status = dwmapi.DwmGetWindowAttribute(
        ctypes.c_void_p(hwnd), 9, ctypes.byref(rect), ctypes.sizeof(rect),
    )
    if status != 0 and not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise OSError(ctypes.get_last_error(), "cannot locate foreground window")
    left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
    if right <= left or bottom <= top:
        raise RuntimeError("foreground window has no visible capture area")
    return (left, top, right, bottom)


def _foreground_window_target() -> dict[str, Any]:
    """Identify the foreground window rectangle without reading screen pixels."""
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        raise RuntimeError("no foreground window is available for capture")
    bbox = _foreground_window_bbox()
    if user32.GetForegroundWindow() != hwnd:
        raise RuntimeError("foreground window changed during capture inspection")
    user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    user32.GetWindowThreadProcessId.restype = ctypes.c_ulong
    pid = ctypes.c_ulong()
    if not user32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(pid)):
        raise OSError(ctypes.get_last_error(), "cannot identify foreground window process")
    user32.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    title_length = min(max(user32.GetWindowTextLengthW(ctypes.c_void_p(hwnd)), 0), 1024)
    title = ctypes.create_unicode_buffer(title_length + 1)
    user32.GetWindowTextW(ctypes.c_void_p(hwnd), title, len(title))
    if user32.GetForegroundWindow() != hwnd:
        raise RuntimeError("foreground window changed during capture inspection")
    return {
        "capture_scope": _SCREEN_CAPTURE_SCOPE,
        "window_handle": int(hwnd), "process_id": int(pid.value),
        "window_title": title.value,
        "source_bbox": {"left": bbox[0], "top": bbox[1],
                        "right": bbox[2], "bottom": bbox[3]},
    }


class WindowsScreen:
    """Bounded snapshot of screen pixels inside foreground-window bounds.

    Cropping before downscaling preserves legible text on multi-monitor
    desktops. This is a screen observation, not isolated window rendering:
    overlays and other windows can appear, and desktop background can show
    through transparent window corners.
    """

    def __init__(self, *, max_bytes: int = 2_000_000) -> None:
        _require_windows()
        if type(max_bytes) is not int or max_bytes < 64_000:
            raise ValueError("max_bytes must be at least 64,000")
        self.max_bytes = max_bytes

    def capture_target(self) -> Mapping[str, Any]:
        return _foreground_window_target()

    def capture_approved(self, expected_target: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.capture(expected_target=expected_target)

    def capture(self, *, expected_target: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        try:
            from PIL import ImageGrab
        except ImportError as exc:
            raise RuntimeError("install Pillow for native screen capture") from exc
        if expected_target is None:
            bbox = _foreground_window_bbox()
        else:
            observed = dict(self.capture_target())
            if observed != dict(expected_target):
                raise RuntimeError("foreground window changed after capture approval")
            bounds = observed["source_bbox"]
            bbox = (bounds["left"], bounds["top"], bounds["right"], bounds["bottom"])
        image = ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB")
        if expected_target is not None and dict(self.capture_target()) != dict(expected_target):
            raise RuntimeError("foreground window changed during approved capture")
        original_size = image.size
        # The admitted multimodal StageClient accepts at most 1024² pixels.
        # Resize at the tool boundary and report the source dimensions so the
        # model and quality evaluator do not mistake this for a native-size
        # observation. A larger screenshot would otherwise fail only after
        # PNG conversion and model submission.
        if image.width * image.height > 1024 * 1024:
            image.thumbnail((1024, 1024))
        model_resized = image.size != original_size
        encoded = b""
        for quality in (85, 70, 55, 40):
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            encoded = output.getvalue()
            if len(encoded) <= self.max_bytes:
                break
        byte_resized = False
        while len(encoded) > self.max_bytes and min(image.size) > 320:
            image.thumbnail((int(image.width * 0.75), int(image.height * 0.75)))
            byte_resized = True
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=40, optimize=True)
            encoded = output.getvalue()
        if len(encoded) > self.max_bytes:
            raise RuntimeError("screen capture exceeds the configured image byte limit")
        return {
            "mime_type": "image/jpeg",
            "capture_scope": _SCREEN_CAPTURE_SCOPE,
            "source_bbox": {"left": bbox[0], "top": bbox[1],
                            "right": bbox[2], "bottom": bbox[3]},
            "width": image.width,
            "height": image.height,
            "original_width": original_size[0],
            "original_height": original_size[1],
            "resized_for_model_pixel_limit": model_resized,
            "resized_for_byte_limit": byte_resized,
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
        *,
        browser_factory: Callable[[], BrowserBackend] | None = None,
        browser_resource_guard: Callable[[], None] | None = None,
        browser_close: Callable[[], None] | None = None,
    ) -> None:
        if browser is not None and browser_factory is not None:
            raise ValueError("choose one browser owner or a cold browser factory")
        if browser_factory is not None and (browser_resource_guard is None or browser_close is None):
            raise ValueError("a cold browser factory requires its admission guard and owner close callback")
        self._browser = browser
        self._browser_factory = browser_factory
        self._browser_resource_guard = browser_resource_guard
        self._browser_close = browser_close
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

    def register_explicit_url(self, request_id: str, url: str) -> None:
        """Trust only one exact URL from the dedicated Read URL input field.

        Unlike ``register_user_task``, this does not interpret prose, so a
        legitimate comma, semicolon, or Unicode path remains part of the URL.
        All other destinations and high-impact GETs still require approval.
        """
        if not request_id:
            raise ValueError("a request ID is required")
        exact_url = _exact_user_url(url)
        with self._lock:
            if request_id in self._task_urls:
                raise ValueError("user task is already registered for this request")
            self._task_urls[request_id] = frozenset({exact_url})

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

    def _guard_browser(self, operation: str) -> None:
        if operation.startswith("browser_") and self._browser_resource_guard is not None:
            self._browser_resource_guard()

    @property
    def browser(self) -> BrowserBackend:
        with _browser_resource_scope(self._browser_resource_guard):
            if self._browser_factory is not None:
                # Factory owner holds the current browser. Do not cache it
                # here: a verified joint drain replaces its closed executor.
                return self._browser_factory()
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

    def _require_supported_browser_write(self, operation: str) -> None:
        if (operation in {"browser_click", "browser_fill"} and
                (self._browser is None or isinstance(self._browser, ManagedEdgeBrowser))):
            # Reject before describe_target(), which evaluates page state, or
            # before creating an approval for a write the real backend cannot
            # safely execute. Injected test backends retain their own contract.
            raise PermissionError(_MANAGED_BROWSER_WRITE_BLOCK_REASON)

    def execute(self, action: ToolAction) -> ToolResult:
        self._validate(action)
        guard = self._browser_resource_guard if action.operation.startswith("browser_") else None
        return _guarded_browser_call(guard, self._execute_validated, action)

    def _execute_validated(self, action: ToolAction) -> ToolResult:
        self._require_supported_browser_write(action.operation)
        with self._lock:
            self._require_active(action.request_id)
        navigation_context: Mapping[str, Any] | None = None
        if action.operation in {"browser_open", "browser_follow"}:
            navigation_context, needs_approval = self._navigation_context(action)
            if needs_approval:
                challenge = self._propose(action, context=navigation_context)
                raise ApprovalRequired(challenge)
        if action.operation in _WRITE_OPERATIONS or action.operation == "screen_capture":
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
        guard = self._browser_resource_guard if action.operation.startswith("browser_") else None
        return _guarded_browser_call(guard, self._approve_pending, pending)

    def _approve_pending(self, pending: _Pending) -> ToolResult:
        action = pending.challenge.action
        self._require_supported_browser_write(action.operation)
        if action.operation == "browser_follow":
            current, _ = self._navigation_context(action)
            if current != pending.context:
                raise RuntimeError("browser target changed after approval was requested")
        elif action.operation == "browser_post":
            current = self._post_context(action)
            if current != pending.context:
                raise RuntimeError("browser POST target or cookie context changed after approval was requested")
        elif action.operation in {"browser_click", "browser_fill"}:
            current = dict(self.browser.describe_target(str(action.arguments["selector"])))
            if current != pending.context:
                raise RuntimeError("browser target changed after approval was requested")
        elif action.operation == "settings_set":
            current = dict(self.settings.read(str(action.arguments["setting"])))
            if current != pending.context:
                raise RuntimeError("Windows setting changed after approval was requested")
        elif action.operation == "screen_capture":
            current = self._screen_context()
            if current != pending.context:
                raise RuntimeError("foreground window changed after capture approval was requested")
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

    def reset_browser_owner_after_verified_drain(self) -> None:
        """Factory owner's reset hook; no guard, launch, inspection or close.

        Only the resource owner may call this after exact-token drain. Revoke
        browser grants from the previous generation; keep trusted task URLs
        and unrelated settings/screen grants in this boundary.
        """
        if self._browser_factory is None:
            raise RuntimeError("browser reset requires a cold factory owner")
        with self._lock:
            for challenge_id, pending in tuple(self._pending.items()):
                if pending.challenge.action.operation.startswith("browser_"):
                    del self._pending[challenge_id]

    def close(self) -> None:
        """Discard approval grants and release the app-owned browser profile."""
        with self._lock:
            self._pending.clear()
            self._task_urls.clear()
        if self._browser_close is not None:
            # Cleanup never calls the admission guard, even after model release.
            self._browser_close()
            return
        browser = self._browser
        if browser is not None:
            closer = getattr(browser, "close", None)
            if callable(closer):
                closer()

    def _propose(self, action: ToolAction, *, context: Mapping[str, Any] | None = None) -> ApprovalChallenge:
        if context is None:
            if action.operation in {"browser_click", "browser_fill", "browser_follow"}:
                context = dict(self.browser.describe_target(str(action.arguments["selector"])))
            elif action.operation == "browser_post":
                context = self._post_context(action)
            elif action.operation == "browser_open":
                context = {"url": str(action.arguments["url"])}
            elif action.operation == "settings_set":
                context = dict(self.settings.read(str(action.arguments["setting"])))
            elif action.operation == "screen_capture":
                context = self._screen_context()
            else:
                context = {}
        # Keep the exact target used for the UI challenge immutable until
        # approval. The approved follow is resolved from this snapshot only
        # after the current DOM target has been checked against it again.
        context = MappingProxyType(json.loads(json.dumps(dict(context), ensure_ascii=False, allow_nan=False)))
        risk = "system_change" if action.operation == "settings_set" else "external_action"
        if action.operation in {"browser_open", "browser_follow", "browser_post"}:
            risk = "high_impact_external_action"
        if action.operation == "screen_capture":
            risk = "sensitive_desktop_read"
        if action.operation == "browser_click":
            if _high_impact_text(*(context.get(key) for key in
                                   ("text", "aria_label", "href", "url", "form_action"))):
                risk = "high_impact_external_action"
        description = {
            "browser_open": "Open a URL that may send, delete, or change external data",
            "browser_follow": "Follow a link that may send, delete, or change external data",
            "browser_click": "Click a browser control; this may send, buy, delete, or change external data",
            "browser_fill": "Fill a browser control; this may save data on an external site",
            "browser_post": "Send this exact HTTP POST body to the approved same-origin URL once",
            "screen_capture": (
                "Capture screen pixels within the approved foreground window's bounds once; "
                "overlays, other windows, or background through transparent corners may appear. "
                "A changed foreground target is refused"
            ),
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

    def _post_context(self, action: ToolAction) -> dict[str, Any]:
        url = _post_url(str(action.arguments["url"]))
        body = _post_body(action.arguments["body_b64"])
        snapshot = getattr(self.browser, "post_context", None)
        if not callable(snapshot):
            raise PermissionError("browser backend has no isolated POST capability")
        observed = dict(snapshot(url))
        if (not isinstance(observed.get("cookie_fingerprint"), str)
                or not isinstance(observed.get("current_url"), str)):
            raise RuntimeError("browser POST cookie context could not be verified")
        return {
            "url": url, "current_url": observed["current_url"],
            "content_type": action.arguments["content_type"],
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "body_size": len(body),
            "cookie_fingerprint": observed["cookie_fingerprint"],
            "cookie_count": observed.get("cookie_count"),
        }

    def _screen_context(self) -> dict[str, Any]:
        screen = self.screen
        target_reader = getattr(screen, "capture_target", None)
        approved_capture = getattr(screen, "capture_approved", None)
        if not callable(target_reader) or not callable(approved_capture):
            raise PermissionError("screen backend cannot bind an approved foreground target")
        target = dict(target_reader())
        bbox = target.get("source_bbox")
        if (target.get("capture_scope") != _SCREEN_CAPTURE_SCOPE
                or type(target.get("window_handle")) is not int
                or target["window_handle"] <= 0
                or type(target.get("process_id")) is not int
                or target["process_id"] <= 0
                or not isinstance(target.get("window_title"), str)
                or not isinstance(bbox, Mapping)
                or set(bbox) != {"left", "top", "right", "bottom"}
                or any(type(value) is not int for value in bbox.values())
                or bbox["right"] <= bbox["left"] or bbox["bottom"] <= bbox["top"]):
            raise RuntimeError("screen backend did not report a bound foreground rectangle")
        return target

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
            "browser_post": {"url", "body_b64", "content_type"},
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
        if action.operation == "browser_post":
            _post_url(str(arguments["url"]))
            _post_body(arguments["body_b64"])
            if (not isinstance(arguments["content_type"], str)
                    or arguments["content_type"] not in _POST_CONTENT_TYPES):
                raise ValueError("browser POST Content-Type is not allowlisted")
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
        self._guard_browser(action.operation)
        try:
            return self._run_admitted(action, navigation_context=navigation_context)
        except BrowserResourcePostconditionFailed as failure:
            # ManagedEdgeBrowser can finish a real write before its caller
            # postcondition fails. Keep that result visible to the controller.
            if isinstance(failure.completed_output, Mapping):
                source = (str(action.arguments["url"]) if action.operation == "browser_post"
                          else str(failure.completed_output.get("url", "managed-browser")))
                failure.completed_output = ToolResult(
                    operation=action.operation, data=failure.completed_output,
                    source=source, observed_at_unix=time.time(), untrusted_output=True,
                )
            raise

    def _run_admitted(self, action: ToolAction, *,
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
            if navigation_context is None:
                raise AssertionError("screen capture requires frozen approval context")
            approved_capture = getattr(self.screen, "capture_approved", None)
            if not callable(approved_capture):
                raise PermissionError("screen backend cannot bind an approved foreground target")
            data = approved_capture(navigation_context)
            if data.get("capture_scope") != _SCREEN_CAPTURE_SCOPE:
                raise RuntimeError("screen backend did not report the approved capture scope")
            source = "windows-screen"
        elif op == "browser_click":
            authorized_click = getattr(self.browser, "click_authorized", None)
            if callable(authorized_click) and navigation_context is not None:
                data = authorized_click(str(args["selector"]), navigation_context)
            else:
                data = self.browser.click(str(args["selector"]))
            source = str(data.get("url", self.browser.current_url()))
        elif op == "browser_fill":
            data = self.browser.fill(str(args["selector"]), str(args["value"]))
            source = str(data.get("url", self.browser.current_url()))
        elif op == "browser_post":
            if navigation_context is None:
                raise AssertionError("browser POST requires frozen approval context")
            sender = getattr(self.browser, "post_exact", None)
            if not callable(sender):
                raise PermissionError("browser backend has no isolated POST capability")
            body = _post_body(args["body_b64"])
            if hashlib.sha256(body).hexdigest() != navigation_context["body_sha256"]:
                raise RuntimeError("browser POST body changed after approval")
            data = sender(
                str(navigation_context["url"]), body,
                str(navigation_context["content_type"]),
                str(navigation_context["cookie_fingerprint"]),
                str(navigation_context["current_url"]),
            )
            source = str(navigation_context["url"])
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
