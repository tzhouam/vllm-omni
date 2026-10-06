"""No-model regressions for trusted URL, navigation, and Settings boundaries."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest

from vllm_omni.edge.agent.tools import (
    ApprovalRequired,
    ManagedEdgeBrowser,
    ToolAction,
    WindowsSettings,
    WindowsToolBoundary,
    _explicit_task_urls,
    _network_target,
)
from benchmarks.edge_agent.paired_suite import FixtureSite


@pytest.mark.parametrize("task,expected", [
    ("Read https://example.test/report, then answer.", "https://example.test/report"),
    ("Read (https://example.test/report).", "https://example.test/report"),
    ("请查看https://example.test/report，然后总结。", "https://example.test/report"),
    ("请查看 https://example.test/report。", "https://example.test/report"),
    ("Read https://example.test/report?x=1&y=2!", "https://example.test/report?x=1&y=2"),
])
def test_trusted_url_ends_at_ordinary_prose_punctuation(task: str, expected: str) -> None:
    assert _explicit_task_urls(task) == frozenset({expected})


def test_ambiguous_unquoted_han_suffix_cannot_grant_auto_navigation() -> None:
    assert _explicit_task_urls("访问https://example.test/report请总结") == frozenset()
    assert _explicit_task_urls("Read https://example.test/report,section") == frozenset()


def test_browser_unicode_encoding_keeps_distinct_network_targets_distinct() -> None:
    raw = "https://example.test/报告?ids=1,2;lang=中文"
    encoded = ("https://example.test/%E6%8A%A5%E5%91%8A"
               "?ids=1,2;lang=%E4%B8%AD%E6%96%87")
    assert _network_target(raw) == _network_target(encoded)
    assert _network_target("https://example.test/a%2Fb") != _network_target(
        "https://example.test/a/b")
    assert _network_target("https://example.test/report?ids=1%2C2") != _network_target(
        "https://example.test/report?ids=1,2")


class _WindowChild:
    def __init__(self, title: str, automation_id: str, *, control_type: str = "Text",
                 children: list[_WindowChild] | None = None) -> None:
        self.title = title
        self.element_info = SimpleNamespace(automation_id=automation_id,
                                            control_type=control_type)
        self._children = children or []

    def window_text(self) -> str:
        return self.title

    def children(self) -> list[_WindowChild]:
        return self._children


class _Window:
    def __init__(self, title: str, children: list[_WindowChild]) -> None:
        self.title = title
        self.children = children

    def window_text(self) -> str:
        return self.title

    def descendants(self) -> list[_WindowChild]:
        return self.children

    def process_id(self) -> int:
        return 123


def _fake_settings_process(monkeypatch: pytest.MonkeyPatch, name: str = "SystemSettings.exe") -> None:
    monkeypatch.setattr("vllm_omni.edge.agent.tools.psutil.Process",
                        lambda _pid: SimpleNamespace(name=lambda: name))


def test_settings_inspect_requires_observed_page_header(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _Window("Settings", [
        _WindowChild("Display", "NavigationItem"),
        _WindowChild("Sound", "PageTitle"),
        _WindowChild("Volume", ""),
    ])
    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(
        Desktop=lambda **_: SimpleNamespace(windows=lambda: [window]),
    ))
    _fake_settings_process(monkeypatch)
    settings = object.__new__(WindowsSettings)
    with pytest.raises(RuntimeError, match="cannot be verified"):
        settings.inspect("display")
    result = settings.inspect("sound")
    assert result["page"] == "sound"
    assert "Volume" in result["text"]


def test_settings_inspect_rejects_ambiguous_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = [_Window("Settings", [_WindowChild("Display", "PageTitle")]) for _ in range(2)]
    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(
        Desktop=lambda **_: SimpleNamespace(windows=lambda: windows),
    ))
    _fake_settings_process(monkeypatch)
    with pytest.raises(RuntimeError, match="cannot be verified"):
        object.__new__(WindowsSettings).inspect("display")


def test_settings_inspect_uses_current_breadcrumb_not_navigation_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    breadcrumb = _WindowChild("", "PermanentNavigationViewBreadcrumbBar",
                              control_type="Group", children=[
                                  _WindowChild("System", "", control_type="Button"),
                                  _WindowChild("Screen", "", control_type="Button"),
                              ])
    window = _Window("Settings", [
        _WindowChild("Sound", "NavigationItem", control_type="ListItem"),
        breadcrumb,
        _WindowChild("Brightness", ""),
    ])
    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(
        Desktop=lambda **_: SimpleNamespace(windows=lambda: [window]),
    ))
    _fake_settings_process(monkeypatch)
    settings = object.__new__(WindowsSettings)
    assert settings.inspect("display")["page"] == "display"
    with pytest.raises(RuntimeError, match="cannot be verified"):
        settings.inspect("sound")


def test_settings_inspect_rejects_lookalike_process(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _Window("Settings", [_WindowChild("Display", "PageTitle")])
    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(
        Desktop=lambda **_: SimpleNamespace(windows=lambda: [window]),
    ))
    _fake_settings_process(monkeypatch, "notepad.exe")
    with pytest.raises(RuntimeError, match="cannot be verified"):
        object.__new__(WindowsSettings).inspect("display")


def test_form_action_is_part_of_click_risk_classification() -> None:
    class Browser:
        def describe_target(self, _selector: str) -> dict[str, object]:
            return {
                "url": "https://example.test/form", "tag": "BUTTON",
                "visible": True, "text": "Continue", "aria_label": "",
                "href": "", "input_type": "submit",
                "form_action": "https://example.test/delete-account",
                "form_method": "POST", "form_eligible": True,
            }

    boundary = WindowsToolBoundary(browser=Browser())
    with pytest.raises(ApprovalRequired) as required:
        boundary.execute(ToolAction("browser_click", {"selector": "button#continue"}))
    assert required.value.challenge.risk == "high_impact_external_action"


@pytest.mark.parametrize("method,url", [
    ("GET", "https://example.test/api/run?id=1"),
    ("HEAD", "https://example.test/quiet"),
    ("POST", "https://example.test/quiet"),
    ("GET", "https://elsewhere.test/collect?private=value"),
])
def test_page_background_requests_are_blocked_before_fetch(method: str, url: str) -> None:
    browser = object.__new__(ManagedEdgeBrowser)
    browser._page = SimpleNamespace(main_frame=object())
    browser._navigation_target = None
    browser._navigation_target_used = False
    browser._blocked_navigation = None
    browser._blocked_request_count = 0
    calls = []
    request = SimpleNamespace(
        is_navigation_request=lambda: False, method=method, url=url,
    )
    route = SimpleNamespace(
        request=request,
        abort=lambda reason: calls.append(("abort", reason)),
        fetch=lambda **kwargs: calls.append(("fetch", kwargs)),
    )
    browser._guard_navigation(route)
    assert calls == [("abort", "blockedbyclient")]
    assert browser._blocked_request_count == 1


def test_exact_top_level_get_reaches_network_but_form_post_does_not() -> None:
    browser = object.__new__(ManagedEdgeBrowser)
    frame = object()
    browser._page = SimpleNamespace(main_frame=frame)
    browser._navigation_target = "https://example.test/report"
    browser._navigation_target_used = False
    browser._blocked_navigation = None
    browser._blocked_request_count = 0
    calls = []

    def route_for(method: str, url: str, body: bytes = b""):
        request = SimpleNamespace(
            is_navigation_request=lambda: True, frame=frame,
            method=method, url=url, post_data_buffer=body,
        )
        return SimpleNamespace(
            request=request,
            abort=lambda reason: calls.append(("abort", reason)),
            fetch=lambda **kwargs: (calls.append(("fetch", kwargs)) or
                                    SimpleNamespace(status=200, headers={})),
            fulfill=lambda **kwargs: calls.append(("fulfill", kwargs)),
        )

    browser._guard_navigation(route_for("GET", "https://example.test/report"))
    assert [item[0] for item in calls] == ["fetch", "fulfill"]
    assert browser._navigation_target_used is True
    calls.clear()
    browser._guard_navigation(route_for("GET", "https://example.test/report"))
    assert calls == [("abort", "blockedbyclient")]
    calls.clear()
    browser._navigation_target = None
    browser._guard_navigation(route_for("POST", "https://example.test/submit", b"message=hello"))
    assert calls == [("abort", "blockedbyclient")]


@pytest.mark.parametrize("operation,args", [
    ("browser_click", {"selector": "a[href='ms-settings:display']"}),
    ("browser_fill", {"selector": "input#draft", "value": "private"}),
])
def test_managed_browser_rejects_writes_before_target_description_or_page_action(
    operation: str, args: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = object.__new__(ManagedEdgeBrowser)

    def unexpected(*_args: object) -> None:
        raise AssertionError("managed browser must not inspect or act on the page")

    monkeypatch.setattr(browser, "_call", unexpected)
    monkeypatch.setattr(browser, "describe_target", unexpected)
    boundary = WindowsToolBoundary(browser=browser)
    with pytest.raises(PermissionError, match="clicks and fills are disabled"):
        boundary.execute(ToolAction(operation, args))
    assert boundary._pending == {}
    with pytest.raises(PermissionError, match="clicks and fills are disabled"):
        if operation == "browser_click":
            browser.click(args["selector"])
        else:
            browser.fill(args["selector"], args["value"])


def test_browser_installs_guards_before_creating_isolated_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # A prior managed profile may exist. Never pass it to Chromium: an isolated
    # context starts empty, then both guards are installed before new_page.
    (tmp_path / "old-profile-state").write_text("unused", encoding="utf-8")
    sequence = []

    class Context:
        def __init__(self) -> None:
            self.pages = []
            self.routes = []

        def route(self, pattern, handler):
            sequence.append("http_guard")
            self.routes.append((pattern, handler))

        def route_web_socket(self, pattern, handler):
            sequence.append("websocket_guard")
            self.routes.append((pattern, handler))

        def new_page(self):
            sequence.append("first_page")
            assert len(self.routes) == 2
            return SimpleNamespace(url="about:blank")

        def close(self):
            pass

    class Browser:
        def new_context(self, **kwargs):
            sequence.append("new_context")
            assert kwargs["service_workers"] == "block"
            return Context()

        def close(self):
            pass

    class Driver:
        chromium = SimpleNamespace()

        def __init__(self) -> None:
            self.chromium.launch = self.launch

        def launch(self, **kwargs):
            sequence.append("launch")
            assert kwargs["channel"] == "msedge"
            return Browser()

        def stop(self):
            pass

    monkeypatch.setattr("vllm_omni.edge.agent.tools._require_windows", lambda: None)
    fake_sync_api = SimpleNamespace(
        sync_playwright=lambda: SimpleNamespace(start=lambda: Driver()),
    )
    monkeypatch.setitem(sys.modules, "playwright", SimpleNamespace(sync_api=fake_sync_api))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync_api)
    browser = ManagedEdgeBrowser(tmp_path, headless=True)
    try:
        assert browser._ensure_page().url == "about:blank"
        assert sequence == ["launch", "new_context", "http_guard",
                            "websocket_guard", "first_page"]
        assert (tmp_path / "old-profile-state").exists()
    finally:
        browser.close()


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Edge")
def test_native_edge_reads_text_and_inline_visual_fixture() -> None:
    pytest.importorskip("playwright.sync_api")
    edge_paths = [
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    if not any(path.is_file() for path in edge_paths):
        pytest.skip("Microsoft Edge is not installed")
    with FixtureSite() as fixture, tempfile.TemporaryDirectory(
        prefix="omni-edge-read-visual-"
    ) as profile:
        browser = ManagedEdgeBrowser(Path(profile), headless=True)
        try:
            browser.open(fixture.origin + "/text/en")
            text = browser.read()
            assert text["url"] == fixture.origin + "/text/en"
            assert "CEDAR-4827" in text["text"]

            browser.open(fixture.origin + "/visual/en")
            visual_text = browser.read()
            assert "ORBIT-7391" not in visual_text["text"]
            dimensions = browser._call(lambda: browser._page.locator("img").evaluate(
                "image => [image.complete, image.naturalWidth, image.naturalHeight]"
            ))
            assert dimensions == [True, 900, 300]
            screenshot = browser.screenshot()
            assert screenshot["mime_type"] == "image/jpeg"
            assert screenshot["size_bytes"] > 1_000
        finally:
            browser.close()


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Edge")
def test_native_edge_exact_read_url_handles_unicode_path_and_query_punctuation() -> None:
    pytest.importorskip("playwright.sync_api")
    edge_paths = [
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    if not any(path.is_file() for path in edge_paths):
        pytest.skip("Microsoft Edge is not installed")
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen.append(self.path)
            body = b"<html><title>Exact target</title><body>Observed page</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="omni-edge-exact-url-") as profile:
            browser = ManagedEdgeBrowser(Path(profile), headless=True)
            boundary = WindowsToolBoundary(browser=browser)
            try:
                origin = f"http://127.0.0.1:{server.server_port}"
                for number, url in enumerate((
                    origin + "/报告?ids=1,2;lang=中文",
                    origin + "/report?ids=1,2;edition=3",
                )):
                    request_id = f"exact-{number}"
                    boundary.register_explicit_url(request_id, url)
                    opened = boundary.execute(ToolAction(
                        "browser_open", {"url": url}, request_id=request_id,
                    ))
                    read = boundary.execute(ToolAction(
                        "browser_read", request_id=request_id,
                    ))
                    assert unquote(urlsplit(opened.source).path) == urlsplit(url).path
                    assert "Observed page" in read.data["text"]
                    boundary.finish_request(request_id)
                assert any("%E6%8A%A5%E5%91%8A" in path for path in seen)
                assert any("ids=1,2;edition=3" in path for path in seen)
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Edge")
def test_http_redirect_never_reaches_unapproved_destination() -> None:
    pytest.importorskip("playwright.sync_api")
    edge_paths = [
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    if not any(path.is_file() for path in edge_paths):
        pytest.skip("Microsoft Edge is not installed")

    source_hits: Counter[str] = Counter()
    destination_hits: Counter[str] = Counter()

    class Destination(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            destination_hits[self.path] += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"destination")

        def log_message(self, *_: object) -> None:
            pass

    destination = ThreadingHTTPServer(("127.0.0.1", 0), Destination)
    destination_url = f"http://127.0.0.1:{destination.server_port}/delete-account"

    class Source(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            source_hits[self.path] += 1
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", destination_url)
                self.end_headers()
            else:
                body = b'<html><title>Local</title><a id="go" href="/redirect">Go</a></html>'
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    source = ThreadingHTTPServer(("127.0.0.1", 0), Source)
    threads = [threading.Thread(target=server.serve_forever, daemon=True)
               for server in (source, destination)]
    for thread in threads:
        thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="omni-edge-redirect-") as profile:
            browser = ManagedEdgeBrowser(Path(profile), headless=True)
            try:
                root = f"http://127.0.0.1:{source.server_port}"
                with pytest.raises(PermissionError, match="HTTP redirect"):
                    browser.open(root + "/redirect")
                assert source_hits["/redirect"] == 1
                assert destination_hits["/delete-account"] == 0

                assert browser.open(root + "/links")["title"] == "Local"
                with pytest.raises(PermissionError, match="HTTP redirect"):
                    browser.follow("a#go")
                assert source_hits["/redirect"] == 2
                assert destination_hits["/delete-account"] == 0
            finally:
                browser.close()
            # Each browser instance uses a new session below the managed root.
            for _ in range(3):
                reopened = ManagedEdgeBrowser(Path(profile), headless=True)
                try:
                    assert reopened.open(root + "/links")["title"] == "Local"
                finally:
                    reopened.close()
    finally:
        for server in (source, destination):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Edge")
def test_page_network_writes_frames_and_redirects_are_blocked() -> None:
    pytest.importorskip("playwright.sync_api")
    edge_paths = [
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    if not any(path.is_file() for path in edge_paths):
        pytest.skip("Microsoft Edge is not installed")

    hits: Counter[str] = Counter()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits["GET " + self.path] += 1
            if self.path == "/resource-redirect":
                self.send_response(302)
                self.send_header("Location", "/delete-account")
                self.end_headers()
                return
            if self.path == "/active":
                body = (
                    b'<html><title>Active</title><script>'
                    b'fetch("/submit", {method:"POST", body:"change"});'
                    b'fetch("/api/run?id=1");'
                    b'fetch("/collect?private=value");'
                    b'</script><iframe src="/frame"></iframe>'
                    b'<img src="/delete-account">'
                    b'<img src="/resource-redirect"></html>'
                )
            elif self.path == "/click":
                body = b'<html><title>Click</title><a id="go" href="/next">Go</a></html>'
            elif self.path == "/fill":
                body = (
                    b'<html><title>Fill</title><input id="draft" '
                    b'oninput="fetch(\'/submit\', {method:\'POST\', body:this.value})"></html>'
                )
            elif self.path == "/form":
                body = (
                    b'<html><title>Form</title><form action="/submit" method="post">'
                    b'<input name="message" value="hello">'
                    b'<button id="send" type="submit" name="commit" value="yes">Send</button>'
                    b'</form><script>window.formdataCalls=0;'
                    b'document.querySelector("form").addEventListener("formdata",'
                    b'()=>{window.formdataCalls++})</script></html>'
                )
            elif self.path == "/external":
                body = (
                    b'<html><title>External</title><a id="external" '
                    b'href="ms-settings:display" '
                    b'onclick="document.body.dataset.clicked=\'yes\'">Open</a></html>'
                )
            elif self.path == "/js-button":
                body = (
                    b'<html><title>JS button</title><button id="send" '
                    b'onclick="fetch(\'/submit\', {method:\'POST\', body:\'change\'})">'
                    b'Send</button></html>'
                )
            else:
                body = b'<html><title>Other</title></html>'
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            hits["POST " + self.path] += 1
            self.send_response(303)
            self.send_header("Location", "/thanks")
            self.end_headers()

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="omni-edge-network-guard-") as profile:
            browser = ManagedEdgeBrowser(Path(profile), headless=True)
            try:
                root = f"http://127.0.0.1:{server.server_port}"
                result = browser.open(root + "/active")
                browser._call(lambda: browser._page.wait_for_timeout(300))
                assert result["title"] == "Active"
                assert hits["POST /submit"] == 0
                assert hits["GET /api/run?id=1"] == 0
                assert hits["GET /collect?private=value"] == 0
                assert hits["GET /frame"] == 0
                assert hits["GET /delete-account"] == 0
                assert hits["GET /resource-redirect"] == 0

                browser.open(root + "/click")
                with pytest.raises(PermissionError, match="clicks and fills are disabled"):
                    browser.click("a#go")
                assert hits["GET /next"] == 0
                assert browser.current_url() == root + "/click"

                browser.open(root + "/fill")
                with pytest.raises(PermissionError, match="clicks and fills are disabled"):
                    browser.fill("input#draft", "private draft")
                assert hits["POST /submit"] == 0
                assert browser._call(lambda: browser._page.locator("input#draft").input_value()) == ""

                browser.open(root + "/form")
                boundary = WindowsToolBoundary(browser=browser)
                with pytest.raises(PermissionError, match="clicks and fills are disabled"):
                    boundary.execute(ToolAction("browser_click", {"selector": "button#send"}))
                assert boundary._pending == {}
                assert browser._call(lambda: browser._page.evaluate("window.formdataCalls")) == 0
                # Even a direct read-only target description must not invoke
                # FormData and dispatch the page's `formdata` handler.
                description = browser.describe_target("button#send")
                assert description["form_action"] == root + "/submit"
                assert description["form_eligible"] is False
                assert browser._call(lambda: browser._page.evaluate("window.formdataCalls")) == 0
                assert hits["POST /submit"] == 0
                assert hits["GET /thanks"] == 0

                browser.open(root + "/external")
                with pytest.raises(PermissionError, match="clicks and fills are disabled"):
                    browser.click("a#external")
                with pytest.raises(PermissionError, match="clicks and fills are disabled"):
                    boundary.execute(ToolAction("browser_click", {"selector": "a#external"}))
                assert browser.current_url() == root + "/external"
                assert browser._call(lambda: browser._page.evaluate(
                    "document.body.dataset.clicked || ''")) == ""

                browser.open(root + "/js-button")
                with pytest.raises(PermissionError, match="clicks and fills are disabled"):
                    boundary.execute(ToolAction("browser_click", {"selector": "button#send"}))
                assert hits["POST /submit"] == 0
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
