from __future__ import annotations

import pytest

from vllm_omni.edge.agent.tools import (
    ApprovalRequired,
    ToolAction,
    ToolRequestCancelled,
    WindowsToolBoundary,
)


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
    def capture(self) -> dict:
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
    boundary = WindowsToolBoundary(browser=FakeBrowser(), settings=FakeSettings(), screen=FakeScreen())
    browser_image = boundary.execute(ToolAction("browser_screenshot"))
    screen_image = boundary.execute(ToolAction("screen_capture"))
    assert browser_image.data["mime_type"] == "image/jpeg"
    assert screen_image.source == "windows-screen"
    assert screen_image.untrusted_output is True


def test_desktop_module_import_is_optional() -> None:
    from vllm_omni.edge.agent import desktop

    if desktop.QApplication is None:
        with pytest.raises(RuntimeError, match="PySide6"):
            desktop.AgentWindow(object())


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
