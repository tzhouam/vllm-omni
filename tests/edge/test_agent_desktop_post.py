"""Native approval review must expose an exact POST before a UI click."""

from __future__ import annotations

import base64
import hashlib
import time

import pytest

from vllm_omni.edge.agent import desktop


def test_read_url_button_uses_separate_structured_action_not_chat_send(monkeypatch) -> None:
    if desktop.QApplication is None:
        pytest.skip("PySide6 is not installed")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = desktop.QApplication.instance() or desktop.QApplication([])

    class Controller:
        def add_listener(self, callback):
            self.callback = callback

        def submit(self, prompt):
            raise AssertionError("structured Read URL must not submit ordinary chat")

        def submit_read_url(self, url, instruction):
            return (url, instruction)

    controller = Controller()
    window = desktop.AgentWindow(controller)
    calls = []
    window._invoke = lambda callback, *args: calls.append((callback, args))
    try:
        window.read_url_field.setText("https://example.test/page")
        window.prompt.setPlainText("Answer with the title")
        window._submit_read_url()
        assert calls == [(
            controller.submit_read_url,
            ("https://example.test/page", "Answer with the title"),
        )]
        assert not window.send_button.isEnabled()
        assert not window.read_url_button.isEnabled()
        assert window.read_url_field.text() == "https://example.test/page"
        assert window.prompt.toPlainText() == "Answer with the title"
        assert "Read URL: https://example.test/page" in window.transcript.toPlainText()

        window._accept_event({
            "request_id": "request-1", "epoch": 1, "seq": 1,
            "kind": "final", "payload": {"answer": "Example"},
        })
        assert window.send_button.isEnabled()
        assert window.read_url_button.isEnabled()

        # A URL left in the separate field cannot make ordinary Send use it.
        window.read_url_field.setText("https://example.test/other")
        window.prompt.setPlainText("Say hello")
        window._submit()
        assert calls[-1] == (controller.submit, ("Say hello",))
        assert window.read_url_field.text() == "https://example.test/other"
    finally:
        controller.callback = None
        window.setAttribute(desktop.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.close()
        app.processEvents()


def _challenge(body: bytes, *, content_type: str = "application/json") -> dict:
    url = "https://example.test/api/submit?source=agent"
    return {
        "challenge_id": "exact-post",
        "risk": "high_impact_external_action",
        "description": "Send this exact HTTP POST body once",
        "operation": "browser_post",
        "arguments": {
            "url": url,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "content_type": content_type,
        },
        "target": {
            "url": url,
            "current_url": "https://example.test/page",
            "content_type": content_type,
            "body_size": len(body),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "cookie_fingerprint": "a" * 64,
            "cookie_count": 1,
        },
    }


def test_review_shows_every_text_and_binary_byte_with_bound_metadata() -> None:
    json_body = b'{"message":"' + b"a" * 65_000 + b'"}'
    challenge = _challenge(json_body)
    review = desktop._browser_post_review(challenge)
    assert challenge["target"]["url"] in review
    assert "Content-Type: application/json" in review
    assert f"Body size: {len(json_body)} bytes" in review
    assert challenge["target"]["body_sha256"] in review
    assert json_body.decode("utf-8") in review
    assert json_body.hex(" ") in review
    assert challenge["arguments"]["body_b64"] not in review
    transcript = str(desktop._display_payload({"arguments": challenge["arguments"]}))
    assert challenge["arguments"]["body_b64"] not in transcript
    assert "inspect the exact approval review" in transcript

    form = b"name=Alice+Chen&name=Bob&note=%E4%BD%A0%E5%A5%BD"
    form_review = desktop._browser_post_review(
        _challenge(form, content_type="application/x-www-form-urlencoded"),
    )
    assert form.decode() in form_review
    assert '"name",\n    "Alice Chen"' in form_review
    assert '"name",\n    "Bob"' in form_review
    assert "你好" in form_review

    binary = b"\x00\xff\x10\x80"
    binary_review = desktop._browser_post_review(
        _challenge(binary, content_type="application/octet-stream"),
    )
    assert "Body is not UTF-8" in binary_review
    assert "00 ff 10 80" in binary_review


@pytest.mark.parametrize("field,bad", [
    ("url", "https://other.test/api/submit"),
    ("content_type", "text/plain"),
    ("body_size", 0),
    ("body_sha256", "0" * 64),
    ("cookie_fingerprint", "missing"),
])
def test_review_rejects_inconsistent_approval_metadata(field: str, bad: object) -> None:
    challenge = _challenge(b'{"message":"hello"}')
    challenge["target"][field] = bad
    with pytest.raises(ValueError):
        desktop._browser_post_review(challenge)


def test_review_rejects_noncanonical_body() -> None:
    challenge = _challenge(b"a")
    challenge["arguments"]["body_b64"] = "YQ="
    with pytest.raises(ValueError, match="canonical"):
        desktop._browser_post_review(challenge)


def test_post_ui_requires_full_review_and_explicit_accept(monkeypatch) -> None:
    if desktop.QApplication is None:
        pytest.skip("PySide6 is not installed")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = desktop.QApplication.instance() or desktop.QApplication([])
    body = b'{"message":"' + b"z" * 65_000 + b'"}'
    challenge = _challenge(body)

    class Controller:
        def __init__(self):
            self.approved: list[str] = []

        def add_listener(self, callback):
            self.callback = callback

        def approve(self, challenge_id):
            self.approved.append(challenge_id)

    controller = Controller()
    window = desktop.AgentWindow(controller)
    window._active = True
    window.send_button.setEnabled(False)
    decisions = [desktop.QDialog.DialogCode.Rejected, desktop.QDialog.DialogCode.Accepted]
    seen: list[str] = []

    def inspect_dialog(dialog):
        preview = dialog.findChild(desktop.QPlainTextEdit, "browser_post_exact_review")
        assert preview is not None and preview.isReadOnly()
        review = preview.toPlainText()
        assert body.decode() in review
        assert body.hex(" ") in review
        assert challenge["target"]["url"] in review
        assert challenge["target"]["body_sha256"] in review
        assert preview.lineWrapMode() == desktop.QPlainTextEdit.LineWrapMode.WidgetWidth
        seen.append(review)
        return decisions.pop(0)

    monkeypatch.setattr(desktop.QDialog, "exec", inspect_dialog)
    try:
        window._accept_event({
            "request_id": "request-1", "epoch": 1, "seq": 1,
            "kind": "approval_required", "payload": challenge,
        })
        assert window.approvals.count() == 1
        assert challenge["arguments"]["body_b64"] not in window.approvals.item(0).text()
        window._approve_selected()
        assert controller.approved == []
        assert window.approvals.count() == 1
        window._approve_selected()
        deadline = time.monotonic() + 3
        while window.approvals.count() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert controller.approved == ["exact-post"]
        assert window.approvals.count() == 0
        assert len(seen) == 2
    finally:
        controller.callback = None
        window.setAttribute(desktop.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.close()
        app.processEvents()


def test_post_review_wraps_full_near_limit_url_and_removes_expired_item(monkeypatch) -> None:
    if desktop.QApplication is None:
        pytest.skip("PySide6 is not installed")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = desktop.QApplication.instance() or desktop.QApplication([])
    challenge = _challenge(b"hello")
    url = "https://example.test/api/submit?x=" + "a" * 1900 + "TAIL"
    challenge["arguments"]["url"] = url
    challenge["target"]["url"] = url

    class Controller:
        def add_listener(self, callback):
            self.callback = callback

        def approve(self, challenge_id):
            raise TimeoutError("approval challenge expired")

    controller = Controller()
    window = desktop.AgentWindow(controller)
    window._active = True
    seen: list[str] = []

    def inspect_dialog(dialog):
        preview = dialog.findChild(desktop.QPlainTextEdit, "browser_post_exact_review")
        assert preview is not None
        assert preview.lineWrapMode() == desktop.QPlainTextEdit.LineWrapMode.WidgetWidth
        seen.append(preview.toPlainText())
        assert url in seen[-1]
        assert seen[-1].find("TAIL") > seen[-1].find("Target URL:")
        return desktop.QDialog.DialogCode.Accepted

    monkeypatch.setattr(desktop.QDialog, "exec", inspect_dialog)
    try:
        window._accept_event({
            "request_id": "request-1", "epoch": 1, "seq": 1,
            "kind": "approval_required", "payload": challenge,
        })
        assert window.approvals.count() == 1
        window._approve_selected()
        deadline = time.monotonic() + 3
        while window.approvals.count() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert len(seen) == 1
        assert window.approvals.count() == 0
        assert "TimeoutError" in window.status_label.text()
    finally:
        controller.callback = None
        window.setAttribute(desktop.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.close()
        app.processEvents()


def test_malformed_post_approval_is_not_presented_as_approvable(monkeypatch) -> None:
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
        challenge = _challenge(b"hello")
        challenge["target"]["body_size"] = 999
        window._accept_event({
            "request_id": "request-1", "epoch": 1, "seq": 1,
            "kind": "approval_required", "payload": challenge,
        })
        assert window.approvals.count() == 0
        assert "cannot be reviewed" in window.status_label.text()
    finally:
        controller.callback = None
        window.setAttribute(desktop.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.close()
        app.processEvents()
