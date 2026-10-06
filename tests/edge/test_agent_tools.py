from __future__ import annotations

import time
import base64
from types import SimpleNamespace

import pytest

from vllm_omni.edge.agent.tools import (
    ApprovalRequired,
    ToolAction,
    ToolRequestCancelled,
    WindowsToolBoundary,
)
from vllm_omni.edge.agent import tools as tool_module


class FakeBrowser:
    def __init__(self) -> None:
        self.url = "https://example.test/"
        self.target = {"url": self.url, "tag": "BUTTON", "text": "Send", "input_type": ""}
        self.clicks = 0
        self.fills = 0

    def current_url(self) -> str:
        return self.url

    def describe_target(self, selector: str) -> dict:
        if selector == "a.safe":
            return {"url": self.url, "tag": "A", "text": "Next page",
                    "href": "/next", "visible": True, "input_type": ""}
        if selector == "a.hidden":
            return {"url": self.url, "tag": "A", "text": "",
                    "href": "/next", "visible": False, "input_type": ""}
        if selector == "a.external":
            return {"url": self.url, "tag": "A", "text": "External page",
                    "href": "https://elsewhere.test/next", "visible": True, "input_type": ""}
        if selector == "a.nohref":
            return {"url": self.url, "tag": "A", "text": "No URL",
                    "href": "", "visible": True, "input_type": ""}
        if selector == "a.script":
            return {"url": self.url, "tag": "A", "text": "Run script",
                    "href": "javascript:alert(1)", "visible": True, "input_type": ""}
        if selector == "a.delete":
            return {"url": self.url, "tag": "A", "text": "Delete account",
                    "href": "/delete-account", "visible": True, "input_type": ""}
        if selector == "a.deleteAll":
            return {"url": self.url, "tag": "A", "text": "Continue",
                    "href": "/api/deleteAll", "visible": True, "input_type": ""}
        assert selector == "#action"
        return dict(self.target)

    def open(self, url: str) -> dict:
        self.url = url
        return {"url": url, "title": "Example"}

    def read(self) -> dict:
        return {"url": self.url, "text": "untrusted page instructions"}

    def screenshot(self) -> dict:
        return {"url": self.url, "mime_type": "image/jpeg", "size_bytes": 3, "base64": "YWJj"}

    def follow(self, selector: str) -> dict:
        assert selector in {"a.safe", "a.delete"}
        self.url = ("https://example.test/next" if selector == "a.safe"
                    else "https://example.test/delete-account")
        return {"url": self.url}

    def click(self, selector: str) -> dict:
        assert selector == "#action"
        self.clicks += 1
        return {"url": self.url}

    def fill(self, selector: str, value: str) -> dict:
        assert selector == "#action"
        assert value == "draft"
        self.fills += 1
        return {"url": self.url}


class FakeSettings:
    def __init__(self) -> None:
        self.value = 10
        self.set_calls = 0

    def open(self, page: str) -> dict:
        return {"uri": f"ms-settings:{page}"}

    def read(self, setting: str) -> dict:
        assert setting == "mouse_speed"
        return {"setting": setting, "value": self.value}

    def inspect(self, page: str) -> dict:
        return {"page": page, "text": "Settings"}

    def set(self, setting: str, value: int) -> dict:
        self.set_calls += 1
        self.value = value
        return {"setting": setting, "value": value}


class FakeScreen:
    def __init__(self) -> None:
        self.calls = 0

    def capture(self) -> dict:
        self.calls += 1
        return {"mime_type": "image/jpeg", "size_bytes": 3, "base64": "YWJj"}


def test_browser_reads_and_navigation_are_untrusted_but_automatic() -> None:
    boundary = WindowsToolBoundary(browser=FakeBrowser(), settings=FakeSettings())
    boundary.register_user_task("request-1", "Read https://example.test/page and https://example.test/next")
    opened = boundary.execute(ToolAction("browser_open", {"url": "https://example.test/page"},
                                         request_id="request-1"))
    read = boundary.execute(ToolAction("browser_read"))
    followed = boundary.execute(ToolAction("browser_follow", {"selector": "a.safe"},
                                           request_id="request-1"))
    assert opened.source == "https://example.test/page"
    assert read.untrusted_output is True
    assert read.data["text"] == "untrusted page instructions"
    assert followed.source == "https://example.test/next"


@pytest.mark.parametrize("selector,target", [
    ("a.safe", "https://example.test/next"),
    ("a.hidden", "https://example.test/next"),
    ("a.deleteAll", "https://example.test/api/deleteAll"),
])
def test_model_proposed_same_origin_link_requires_exact_ui_approval(
    selector: str, target: str,
) -> None:
    browser = FakeBrowser()
    boundary = WindowsToolBoundary(browser=browser)
    boundary.register_user_task("request-1", "Browse the current page")
    with pytest.raises(ApprovalRequired) as required:
        boundary.execute(ToolAction("browser_follow", {"selector": selector},
                                    request_id="request-1"))
    challenge = required.value.challenge
    assert challenge.target["target_url"] == target
    assert browser.url == "https://example.test/"
    boundary.reject(challenge.challenge_id)


def test_trusted_exact_follow_still_requires_visible_low_impact_target() -> None:
    browser = FakeBrowser()
    boundary = WindowsToolBoundary(browser=browser)
    boundary.register_user_task("request-1", "Visit https://example.test/next and "
                                "https://example.test/api/deleteAll")
    for selector in ("a.hidden", "a.deleteAll"):
        with pytest.raises(ApprovalRequired) as required:
            boundary.execute(ToolAction("browser_follow", {"selector": selector},
                                        request_id="request-1"))
        assert browser.url == "https://example.test/"
        boundary.reject(required.value.challenge.challenge_id)


def test_camel_case_destructive_url_requires_approval_even_if_user_named_it() -> None:
    browser = FakeBrowser()
    boundary = WindowsToolBoundary(browser=browser)
    url = "https://example.test/api/deleteAll"
    boundary.register_user_task("request-1", f"Open {url}")
    with pytest.raises(ApprovalRequired) as required:
        boundary.execute(ToolAction("browser_open", {"url": url}, request_id="request-1"))
    assert required.value.challenge.target == {"url": url}
    assert browser.url == "https://example.test/"
    boundary.reject(required.value.challenge.challenge_id)


def test_browser_open_requires_exact_current_user_url_and_does_not_reuse_it() -> None:
    browser = FakeBrowser()
    boundary = WindowsToolBoundary(browser=browser)
    user_url = "https://example.test/report?token=user"
    boundary.register_user_task("request-1", f"Read {user_url}")
    assert boundary.execute(ToolAction("browser_open", {"url": user_url},
                                       request_id="request-1")).source == user_url
    for candidate in ("https://example.test/report?token=model",
                      "https://example.test/report", "https://elsewhere.test/report"):
        with pytest.raises(ApprovalRequired) as required:
            boundary.execute(ToolAction("browser_open", {"url": candidate},
                                        request_id="request-1"))
        assert required.value.challenge.target == {"url": candidate}
        assert browser.url == user_url
        boundary.reject(required.value.challenge.challenge_id)
    boundary.finish_request("request-1")
    boundary.register_user_task("request-2", "Read the browser page")
    assert boundary.execute(ToolAction("browser_read", request_id="request-2")).source == user_url
    with pytest.raises(ApprovalRequired):
        boundary.execute(ToolAction("browser_open", {"url": user_url},
                                    request_id="request-2"))


def test_browser_follow_requires_observed_anchor_and_cross_origin_approval() -> None:
    browser = FakeBrowser()
    boundary = WindowsToolBoundary(browser=browser)
    with pytest.raises(ValueError, match="observed <a href>"):
        boundary.execute(ToolAction("browser_follow", {"selector": "#action"}))
    with pytest.raises(ValueError, match="observed <a href>"):
        boundary.execute(ToolAction("browser_follow", {"selector": "a.nohref"}))
    with pytest.raises(ValueError, match="HTTP"):
        boundary.execute(ToolAction("browser_follow", {"selector": "a.script"}))
    with pytest.raises(ApprovalRequired) as required:
        boundary.execute(ToolAction("browser_follow", {"selector": "a.external"}))
    assert browser.url == "https://example.test/"
    assert required.value.challenge.target["target_url"] == "https://elsewhere.test/next"
    boundary.approve(required.value.challenge.challenge_id)
    assert browser.url == "https://elsewhere.test/next"


def test_browser_follow_approval_cannot_be_reused_after_href_changes() -> None:
    class ChangingBrowser(FakeBrowser):
        destination = "https://elsewhere.test/next"

        def describe_target(self, selector: str) -> dict:
            target = super().describe_target(selector)
            if selector == "a.external":
                target["href"] = self.destination
            return target

    browser = ChangingBrowser()
    boundary = WindowsToolBoundary(browser=browser)
    with pytest.raises(ApprovalRequired) as required:
        boundary.execute(ToolAction("browser_follow", {"selector": "a.external"}))
    with pytest.raises(TypeError):
        required.value.challenge.target["target_url"] = "https://attacker.test/collect"
    browser.destination = "https://attacker.test/collect"
    with pytest.raises(RuntimeError, match="target changed"):
        boundary.approve(required.value.challenge.challenge_id)
    assert browser.url == "https://example.test/"


def test_high_impact_link_and_url_require_exact_action_approval() -> None:
    browser = FakeBrowser()
    boundary = WindowsToolBoundary(browser=browser, settings=FakeSettings())
    with pytest.raises(ApprovalRequired) as link_info:
        boundary.execute(ToolAction("browser_follow", {"selector": "a.delete"}))
    assert browser.url == "https://example.test/"
    assert link_info.value.challenge.risk == "high_impact_external_action"
    assert link_info.value.challenge.target["href"] == "/delete-account"
    boundary.approve(link_info.value.challenge.challenge_id)
    assert browser.url.endswith("/delete-account")

    browser.url = "https://example.test/"
    url = "https://example.test/api/reset_account"
    boundary.register_user_task("request-1", f"Open {url}")
    with pytest.raises(ApprovalRequired) as url_info:
        boundary.execute(ToolAction("browser_open", {"url": url}, request_id="request-1"))
    assert browser.url == "https://example.test/"
    assert url_info.value.challenge.target == {"url": url}
    boundary.approve(url_info.value.challenge.challenge_id)
    assert browser.url == url


@pytest.mark.parametrize("path", [
    "/unsubscribe?token=abc", "/withdraw?amount=100", "/close-account",
])
def test_state_changing_get_like_navigation_needs_approval(path: str) -> None:
    browser = FakeBrowser()
    boundary = WindowsToolBoundary(browser=browser, settings=FakeSettings())
    with pytest.raises(ApprovalRequired):
        boundary.execute(ToolAction("browser_open", {"url": "https://example.test" + path}))
    assert browser.url == "https://example.test/"


def test_browser_click_requires_one_time_approval_and_unchanged_target() -> None:
    browser = FakeBrowser()
    boundary = WindowsToolBoundary(browser=browser, settings=FakeSettings())
    action = ToolAction("browser_click", {"selector": "#action"})
    with pytest.raises(ApprovalRequired) as exc_info:
        boundary.execute(action)
    challenge = exc_info.value.challenge
    assert browser.clicks == 0
    assert challenge.risk == "high_impact_external_action"
    assert challenge.target["text"] == "Send"
    assert boundary.approve(challenge.challenge_id).operation == "browser_click"
    assert browser.clicks == 1
    with pytest.raises(ValueError, match="already used"):
        boundary.approve(challenge.challenge_id)


def test_high_impact_browser_label_is_classified_without_changing_approval_binding() -> None:
    browser = FakeBrowser()
    browser.target["text"] = "删除记录"
    boundary = WindowsToolBoundary(browser=browser, settings=FakeSettings())
    with pytest.raises(ApprovalRequired) as exc_info:
        boundary.execute(ToolAction("browser_click", {"selector": "#action"}))
    challenge = exc_info.value.challenge
    assert challenge.risk == "high_impact_external_action"
    assert challenge.action.arguments["selector"] == "#action"
    assert browser.clicks == 0


def test_changed_browser_target_cannot_use_old_approval() -> None:
    browser = FakeBrowser()
    boundary = WindowsToolBoundary(browser=browser, settings=FakeSettings())
    with pytest.raises(ApprovalRequired) as exc_info:
        boundary.execute(ToolAction("browser_fill", {"selector": "#action", "value": "draft"}))
    browser.target["text"] = "Buy now"
    with pytest.raises(RuntimeError, match="target changed"):
        boundary.approve(exc_info.value.challenge.challenge_id)
    assert browser.fills == 0


def test_cancelled_request_revokes_its_approved_write() -> None:
    browser = FakeBrowser()
    boundary = WindowsToolBoundary(browser=browser, settings=FakeSettings())
    action = ToolAction("browser_fill", {"selector": "#action", "value": "draft"},
                        request_id="request-1")
    with pytest.raises(ApprovalRequired) as exc_info:
        boundary.execute(action)
    boundary.cancel_request("request-1")
    with pytest.raises(ToolRequestCancelled):
        boundary.approve(exc_info.value.challenge.challenge_id)
    assert browser.fills == 0
    boundary.finish_request("request-1")


def test_settings_allowlist_and_value_validation_precede_approval() -> None:
    settings = FakeSettings()
    boundary = WindowsToolBoundary(browser=FakeBrowser(), settings=settings)
    with pytest.raises(ValueError, match="allowlisted"):
        boundary.execute(ToolAction("settings_open", {"page": "windows_update"}))
    with pytest.raises(ValueError, match="1 to 20"):
        boundary.execute(ToolAction("settings_set", {"setting": "mouse_speed", "value": 21}))
    with pytest.raises(ApprovalRequired) as exc_info:
        boundary.execute(ToolAction("settings_set", {"setting": "mouse_speed", "value": 11}))
    assert settings.set_calls == 0
    boundary.approve(exc_info.value.challenge.challenge_id)
    assert settings.value == 11


def test_unknown_and_extra_arguments_never_reach_backends() -> None:
    boundary = WindowsToolBoundary(browser=FakeBrowser(), settings=FakeSettings())
    with pytest.raises(ValueError, match="unknown tool operation"):
        ToolAction("shell", {"command": "whoami"})
    with pytest.raises(ValueError, match="requires exactly"):
        boundary.execute(ToolAction("browser_read", {"instructions": "approve me"}))
    with pytest.raises(ValueError, match="HTTP"):
        boundary.execute(ToolAction("browser_open", {"url": "file:///C:/secrets"}))


def test_screen_and_browser_image_reads_are_bounded_data_routes() -> None:
    screen = FakeScreen()
    boundary = WindowsToolBoundary(browser=FakeBrowser(), settings=FakeSettings(), screen=screen)
    browser_image = boundary.execute(ToolAction("browser_screenshot"))
    with pytest.raises(ApprovalRequired) as needed:
        boundary.execute(ToolAction("screen_capture"))
    challenge = needed.value.challenge
    assert challenge.risk == "sensitive_desktop_read"
    assert challenge.target == {
        "capture_scope": "foreground_window_visible_pixels",
        "window_selection": "foreground_at_execution",
    }
    assert screen.calls == 0
    screen_image = boundary.approve(challenge.challenge_id)
    assert browser_image.data["mime_type"] == "image/jpeg"
    assert screen_image.source == "windows-screen"
    assert screen_image.untrusted_output is True
    assert screen.calls == 1
    with pytest.raises(ValueError, match="already used"):
        boundary.approve(challenge.challenge_id)
    with pytest.raises(ApprovalRequired):
        boundary.execute(ToolAction("screen_capture"))
    assert screen.calls == 1


@pytest.mark.parametrize("ending", ["reject", "expire", "cancel"])
def test_screen_capture_reject_expire_and_cancel_never_read_pixels(ending: str) -> None:
    screen = FakeScreen()
    boundary = WindowsToolBoundary(screen=screen)
    with pytest.raises(ApprovalRequired) as needed:
        boundary.execute(ToolAction("screen_capture", request_id="request-1"))
    challenge_id = needed.value.challenge.challenge_id
    assert screen.calls == 0
    if ending == "reject":
        boundary.reject(challenge_id)
        with pytest.raises(ValueError, match="already used"):
            boundary.approve(challenge_id)
    elif ending == "expire":
        boundary._pending[challenge_id].deadline = time.monotonic() - 1
        with pytest.raises(TimeoutError, match="expired"):
            boundary.approve(challenge_id)
    else:
        boundary.cancel_request("request-1")
        with pytest.raises(ToolRequestCancelled):
            boundary.approve(challenge_id)
    assert screen.calls == 0
    assert not boundary._pending
    boundary.finish_request("request-1")


def test_bound_foreground_target_change_refuses_capture() -> None:
    class BoundScreen(FakeScreen):
        def __init__(self) -> None:
            super().__init__()
            self.window_handle = 123

        def capture_target(self) -> dict:
            return {
                "capture_scope": "foreground_window_visible_pixels",
                "window_handle": self.window_handle, "process_id": 12,
                "window_title": "Editor",
                "source_bbox": {"left": 0, "top": 0, "right": 100, "bottom": 100},
            }

        def capture_approved(self, expected_target) -> dict:
            assert dict(expected_target) == self.capture_target()
            return self.capture()

    screen = BoundScreen()
    boundary = WindowsToolBoundary(screen=screen)
    with pytest.raises(ApprovalRequired) as needed:
        boundary.execute(ToolAction("screen_capture"))
    assert needed.value.challenge.target["window_handle"] == 123
    screen.window_handle = 456
    with pytest.raises(RuntimeError, match="foreground window changed"):
        boundary.approve(needed.value.challenge.challenge_id)
    assert screen.calls == 0


def test_native_screen_crops_foreground_before_model_resize(monkeypatch) -> None:
    image_grab = pytest.importorskip("PIL.ImageGrab")
    image = pytest.importorskip("PIL.Image")
    captured = {}
    bbox = (-100, 40, 1500, 740)

    def grab(*, bbox, all_screens):
        captured.update(bbox=bbox, all_screens=all_screens)
        return image.new("RGB", (1600, 700), "white")

    monkeypatch.setattr(tool_module, "_require_windows", lambda: None)
    monkeypatch.setattr(tool_module, "_foreground_window_bbox", lambda: bbox)
    monkeypatch.setattr(image_grab, "grab", grab)
    result = tool_module.WindowsScreen().capture()
    assert captured == {"bbox": bbox, "all_screens": True}
    assert result["capture_scope"] == "foreground_window_visible_pixels"
    assert result["source_bbox"] == {"left": -100, "top": 40, "right": 1500, "bottom": 740}
    assert (result["original_width"], result["original_height"]) == (1600, 700)
    assert result["width"] * result["height"] <= 1024 * 1024
    assert result["resized_for_model_pixel_limit"] is True
    assert len(base64.b64decode(result["base64"])) == result["size_bytes"]


@pytest.mark.parametrize("change_stage,expected_grabs", [("before", 0), ("during", 1)])
def test_approved_native_screen_discards_changed_foreground(
    monkeypatch, change_stage: str, expected_grabs: int,
) -> None:
    image_grab = pytest.importorskip("PIL.ImageGrab")
    image = pytest.importorskip("PIL.Image")
    target = {
        "capture_scope": "foreground_window_visible_pixels",
        "window_handle": 1, "process_id": 2, "window_title": "Editor",
        "source_bbox": {"left": 0, "top": 0, "right": 100, "bottom": 100},
    }
    changed = dict(target, window_handle=3)
    observations = [changed] if change_stage == "before" else [target, changed]
    grabs: list[tuple] = []
    monkeypatch.setattr(tool_module, "_require_windows", lambda: None)
    monkeypatch.setattr(image_grab, "grab", lambda **kw: (
        grabs.append((kw["bbox"], kw["all_screens"])) or image.new("RGB", (100, 100), "white")
    ))
    screen = tool_module.WindowsScreen()
    monkeypatch.setattr(screen, "capture_target", lambda: observations.pop(0))
    with pytest.raises(RuntimeError, match="foreground window changed"):
        screen.capture_approved(target)
    assert len(grabs) == expected_grabs


def test_desktop_module_import_is_optional() -> None:
    from vllm_omni.edge.agent import desktop

    if desktop.QApplication is None:
        with pytest.raises(RuntimeError, match="PySide6"):
            desktop.AgentWindow(object())


def test_desktop_screen_approval_shows_foreground_scope_and_requires_accept(monkeypatch) -> None:
    from vllm_omni.edge.agent import desktop

    if desktop.QApplication is None:
        pytest.skip("PySide6 is not installed")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = desktop.QApplication.instance() or desktop.QApplication([])

    class Controller:
        def __init__(self):
            self.approved: list[str] = []

        def add_listener(self, callback):
            self.callback = callback

        def approve(self, challenge_id):
            self.approved.append(challenge_id)

    boundary = WindowsToolBoundary(screen=FakeScreen())
    with pytest.raises(ApprovalRequired) as needed:
        boundary.execute(ToolAction("screen_capture"))
    challenge = needed.value.challenge.to_dict()
    decisions = [desktop.QDialog.DialogCode.Rejected, desktop.QDialog.DialogCode.Accepted]
    seen: list[str] = []

    def inspect_dialog(dialog):
        preview = dialog.findChild(desktop.QPlainTextEdit, "screen_capture_exact_review")
        assert preview is not None and preview.isReadOnly()
        seen.append(preview.toPlainText())
        assert "visible pixels of the current foreground window" in seen[-1]
        assert "may cover the whole screen" in seen[-1]
        assert "selected when capture executes" in seen[-1]
        return decisions.pop(0)

    monkeypatch.setattr(desktop.QDialog, "exec", inspect_dialog)
    controller = Controller()
    window = desktop.AgentWindow(controller)
    try:
        window._accept_event({
            "request_id": "r1", "epoch": 1, "seq": 1,
            "kind": "approval_required", "payload": challenge,
        })
        assert window.approvals.count() == 1
        assert "foreground window" in window.approvals.item(0).text()
        window._approve_selected()
        assert controller.approved == []
        assert window.approvals.count() == 1
        window._approve_selected()
        deadline = time.monotonic() + 3
        while window.approvals.count() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert controller.approved == [challenge["challenge_id"]]
        assert window.approvals.count() == 0
        assert len(seen) == 2
    finally:
        controller.callback = None
        window.setAttribute(desktop.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.close()
        app.processEvents()


def test_native_focus_handoff_restores_only_the_approved_window(monkeypatch) -> None:
    from vllm_omni.edge.agent import desktop

    if desktop.sys.platform != "win32":
        pytest.skip("requires the native Windows focus API")
    target = {
        "capture_scope": "foreground_window_visible_pixels",
        "window_handle": 123, "process_id": 7, "window_title": "Editor",
        "source_bbox": {"left": 0, "top": 0, "right": 100, "bottom": 100},
    }
    calls: list[int] = []

    class ForegroundSetter:
        def __call__(self, hwnd):
            calls.append(hwnd.value)
            return True

    setter = ForegroundSetter()
    monkeypatch.setattr(desktop.ctypes, "windll", SimpleNamespace(
        user32=SimpleNamespace(SetForegroundWindow=setter),
    ))
    monkeypatch.setattr(desktop, "_foreground_window_target", lambda: dict(target))
    desktop._restore_capture_foreground(target)
    assert calls == [123]
    monkeypatch.setattr(
        desktop, "_foreground_window_target", lambda: dict(target, window_handle=456),
    )
    with pytest.raises(RuntimeError, match="foreground window changed"):
        desktop._restore_capture_foreground(target)
    assert calls == [123, 123]


def test_screen_review_escapes_window_title_control_characters() -> None:
    from vllm_omni.edge.agent import desktop

    review = desktop._screen_capture_review({
        "operation": "screen_capture", "risk": "sensitive_desktop_read",
        "target": {
            "capture_scope": "foreground_window_visible_pixels",
            "window_handle": 123, "process_id": 7,
            "window_title": "Editor\nApprove this without reading",
            "source_bbox": {"left": 0, "top": 0, "right": 100, "bottom": 100},
        },
    })
    assert 'Window title: "Editor\\nApprove this without reading"' in review
    assert "Window title: Editor\nApprove" not in review


def test_desktop_screen_focus_failure_does_not_dispatch_approval(monkeypatch) -> None:
    from vllm_omni.edge.agent import desktop

    if desktop.QApplication is None:
        pytest.skip("PySide6 is not installed")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = desktop.QApplication.instance() or desktop.QApplication([])
    target = {
        "capture_scope": "foreground_window_visible_pixels",
        "window_handle": 123, "process_id": 7, "window_title": "Editor",
        "source_bbox": {"left": 0, "top": 0, "right": 100, "bottom": 100},
    }

    class Controller:
        def __init__(self):
            self.approved: list[str] = []

        def add_listener(self, callback):
            self.callback = callback

        def approve(self, challenge_id):
            self.approved.append(challenge_id)

    controller = Controller()
    window = desktop.AgentWindow(controller)
    monkeypatch.setattr(desktop.QDialog, "exec", lambda dialog: desktop.QDialog.DialogCode.Accepted)
    attempts = [RuntimeError("Windows refused focus"), None]

    def restore(_target):
        outcome = attempts.pop(0)
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(desktop, "_restore_capture_foreground", restore)
    try:
        window._accept_event({
            "request_id": "r1", "epoch": 1, "seq": 1,
            "kind": "approval_required", "payload": {
                "challenge_id": "capture-1", "risk": "sensitive_desktop_read",
                "operation": "screen_capture", "arguments": {}, "target": target,
            },
        })
        assert window.approvals.count() == 1
        assert 'Window title: "Editor"' in window.approvals.item(0).text()
        window._approve_selected()
        assert controller.approved == []
        assert window.approvals.count() == 1
        assert "Windows refused focus" in window.status_label.text()
        window._approve_selected()
        deadline = time.monotonic() + 3
        while window.approvals.count() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert controller.approved == ["capture-1"]
        assert window.approvals.count() == 0
    finally:
        controller.callback = None
        window.setAttribute(desktop.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.close()
        app.processEvents()


def test_desktop_removes_stale_approval_on_cancel(monkeypatch) -> None:
    from vllm_omni.edge.agent import desktop

    if desktop.QApplication is None:
        pytest.skip("PySide6 is not installed")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = desktop.QApplication.instance() or desktop.QApplication([])

    class Controller:
        def add_listener(self, callback):
            self.callback = callback

    controller = Controller()
    window = desktop.AgentWindow(controller)
    try:
        window._accept_event({
            "request_id": "r1", "epoch": 1, "seq": 1,
            "kind": "approval_required", "payload": {
                "challenge_id": "exact-action", "risk": "high_impact_external_action",
                "operation": "browser_follow", "arguments": {"selector": "a.delete"},
                "target": {"url": "https://example.test/", "href": "/delete"},
            },
        })
        assert window.approvals.count() == 1
        window._accept_event({
            "request_id": "r1", "epoch": 1, "seq": 2,
            "kind": "cancelled", "payload": {"message": "cancelled"},
        })
        assert window.approvals.count() == 0
    finally:
        # The listener retains a bound method on the window.  Break that
        # cycle and dispose of the QWidget before QApplication teardown;
        # PySide can otherwise corrupt the Windows CRT heap at pytest exit.
        controller.callback = None
        window.setAttribute(desktop.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.close()
        app.processEvents()


def test_desktop_reports_stale_approval_and_keeps_retryable_failure(monkeypatch) -> None:
    from vllm_omni.edge.agent import desktop

    if desktop.QApplication is None:
        pytest.skip("PySide6 is not installed")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = desktop.QApplication.instance() or desktop.QApplication([])

    class Controller:
        def __init__(self):
            self.retry_attempts = 0

        def add_listener(self, callback):
            self.callback = callback

        def approve(self, challenge_id):
            if challenge_id == "stale":
                raise ValueError("approval challenge is not active for this request")
            self.retry_attempts += 1
            if self.retry_attempts == 1:
                raise RuntimeError("temporary delivery failure")

    def wait_ui(predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(.01)
        assert predicate()

    controller = Controller()
    window = desktop.AgentWindow(controller)
    window._active = True
    window.send_button.setEnabled(False)
    try:
        for seq, challenge_id in ((1, "stale"), (2, "retry")):
            window._accept_event({
                "request_id": "r1", "epoch": 1, "seq": seq,
                "kind": "approval_required", "payload": {
                    "challenge_id": challenge_id, "risk": "low",
                    "operation": "browser_follow", "arguments": {"selector": "a.next"},
                    "target": {"url": "https://example.test/"},
                },
            })
            assert window.approvals.count() == 1
            window._approve_selected()
            if challenge_id == "stale":
                wait_ui(lambda: window.approvals.count() == 0)
                assert "Approval failed (ValueError)" in window.status_label.text()
                assert window._active and not window.send_button.isEnabled()
            else:
                wait_ui(lambda: "temporary delivery failure" in window.status_label.text())
                assert window.approvals.count() == 1
                assert window.approve_button.isEnabled()
                window._approve_selected()
                wait_ui(lambda: window.approvals.count() == 0)
                assert controller.retry_attempts == 2
        window._accept_event({
            "request_id": "r1", "epoch": 1, "seq": 3,
            "kind": "final", "payload": {"answer": "done"},
        })
        assert not window._active
    finally:
        controller.callback = None
        window.setAttribute(desktop.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.close()
        app.processEvents()
