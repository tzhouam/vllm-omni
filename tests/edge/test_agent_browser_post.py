"""Exact, one-shot browser POST boundary against loopback fixtures only."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.edge.agent.tools import (
    ApprovalRequired, ManagedEdgeBrowser, ToolAction, WindowsToolBoundary,
)


def _post_action(url: str, body: bytes = b'{"message":"hello"}') -> ToolAction:
    return ToolAction("browser_post", {
        "url": url,
        "body_b64": base64.b64encode(body).decode("ascii"),
        "content_type": "application/json",
    }, request_id="post-1")


@pytest.mark.parametrize("arguments,reason", [
    ({"url": "https://example.test/submit#fragment", "body_b64": "YQ==",
      "content_type": "application/json"}, "without whitespace or fragment"),
    ({"url": "https://example.test/submit", "body_b64": "YQ=",
      "content_type": "application/json"}, "canonical base64"),
    ({"url": "https://example.test/submit", "body_b64": "YQ==",
      "content_type": "application/json; charset=utf-8"}, "Content-Type"),
    ({"url": "https://example.test/submit", "body_b64": base64.b64encode(b"x" * 65537).decode(),
      "content_type": "application/json"}, "64 KiB"),
])
def test_post_validation_rejects_ambiguous_or_large_payloads(
    arguments: dict[str, str], reason: str,
) -> None:
    boundary = WindowsToolBoundary(browser=object())
    with pytest.raises(ValueError, match=reason):
        boundary.execute(ToolAction("browser_post", arguments))
    assert boundary._pending == {}


def test_post_timeout_is_ambiguous_and_never_retried() -> None:
    calls = []

    class RequestContext:
        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            raise TimeoutError("response timed out after send")

        def dispose(self):
            calls.append("dispose")

    browser = object.__new__(ManagedEdgeBrowser)
    browser._post_snapshot = lambda url: ({
        "current_url": "https://example.test/page",
        "cookie_fingerprint": "f" * 64,
    }, [])
    browser._playwright = SimpleNamespace(request=SimpleNamespace(
        new_context=lambda **kwargs: RequestContext(),
    ))
    with pytest.raises(RuntimeError, match="outcome unknown.*do not retry"):
        browser._post_exact(
            "https://example.test/submit", b"hello", "text/plain; charset=utf-8",
            "f" * 64, "https://example.test/page",
        )
    assert len([call for call in calls if isinstance(call, tuple)]) == 1
    assert calls[0][1]["data"] == b"hello"
    assert calls[0][1]["max_redirects"] == 0
    assert calls[0][1]["max_retries"] == 0
    assert calls[-1] == "dispose"


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Edge")
def test_native_post_is_exact_approved_one_shot_and_does_not_follow_redirect() -> None:
    pytest.importorskip("playwright.sync_api")
    edge_paths = [
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    if not any(path.is_file() for path in edge_paths):
        pytest.skip("Microsoft Edge is not installed")

    hits: Counter[str] = Counter()
    requests: list[tuple[str, bytes, str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits["GET " + self.path] += 1
            body = b"<html><title>POST fixture</title><body>Local page</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Set-Cookie", "session=original; Path=/; HttpOnly")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            hits["POST " + self.path] += 1
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append((self.path, body, self.headers.get("Cookie", ""),
                             self.headers.get("Content-Type", "")))
            self.send_response(303)
            self.send_header("Location", "/redirect-target")
            self.send_header("Set-Cookie", "session=mutated; Path=/; HttpOnly")
            self.end_headers()

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="omni-edge-post-") as profile:
            browser = ManagedEdgeBrowser(Path(profile), headless=True)
            try:
                root = f"http://127.0.0.1:{server.server_port}"
                browser.open(root + "/page")
                boundary = WindowsToolBoundary(browser=browser)
                boundary.register_user_task("post-1", "Send the exact local fixture body")
                body = b'{"message":"exact bytes"}'
                with pytest.raises(ApprovalRequired) as required:
                    boundary.execute(_post_action(root + "/submit", body))
                challenge = required.value.challenge
                assert challenge.risk == "high_impact_external_action"
                assert challenge.target["url"] == root + "/submit"
                assert challenge.target["body_sha256"] == hashlib.sha256(body).hexdigest()
                assert challenge.target["body_size"] == len(body)
                assert challenge.target["cookie_count"] == 1
                assert "original" not in json.dumps(dict(challenge.target))
                assert hits["POST /submit"] == 0
                with pytest.raises(ApprovalRequired) as rejected:
                    boundary.execute(_post_action(root + "/submit", body))
                boundary.reject(rejected.value.challenge.challenge_id)
                assert hits["POST /submit"] == 0
                result = boundary.approve(challenge.challenge_id)
                assert result.data["status"] == 303
                assert result.data["redirect_blocked"] is True
                assert result.data["redirect_followed"] is False
                assert requests == [("/submit", body, "session=original", "application/json")]
                assert hits["GET /redirect-target"] == 0
                assert browser.current_url() == root + "/page"
                assert browser._call(lambda: browser._context.cookies(root + "/submit"))[0]["value"] == "original"
                with pytest.raises(ValueError, match="already used"):
                    boundary.approve(challenge.challenge_id)

                with pytest.raises(ApprovalRequired) as changed:
                    boundary.execute(_post_action(root + "/submit", body))
                browser._call(lambda: browser._context.add_cookies([{
                    "name": "session", "value": "changed", "url": root,
                }]))
                with pytest.raises(RuntimeError, match="cookie context changed"):
                    boundary.approve(changed.value.challenge.challenge_id)
                assert hits["POST /submit"] == 1

                other_port = server.server_port + 1
                with pytest.raises(PermissionError, match="current page origin"):
                    boundary.execute(_post_action(
                        f"http://127.0.0.1:{other_port}/submit", body,
                    ))
                assert hits["POST /submit"] == 1
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
