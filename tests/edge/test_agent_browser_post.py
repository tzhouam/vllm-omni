"""Exact, one-shot browser POST boundary against loopback fixtures only."""

from __future__ import annotations

import base64
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import tracemalloc
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from vllm_omni.edge.agent.tools import (
    ApprovalRequired, ManagedEdgeBrowser, ToolAction, WindowsToolBoundary,
    _post_url, _streamed_post,
)


def _post_action(url: str, body: bytes = b'{"message":"hello"}') -> ToolAction:
    return ToolAction("browser_post", {
        "url": url,
        "body_b64": base64.b64encode(body).decode("ascii"),
        "content_type": "application/json",
    }, request_id="post-1")


@pytest.mark.parametrize("arguments,reason", [
    ({"url": "https://example.test/submit#fragment", "body_b64": "YQ==",
      "content_type": "application/json"}, "fragment"),
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


@pytest.mark.parametrize("path", [
    "/safe/../admin?x=1", "/safe/./admin", "/safe/%2e%2e/admin",
    "/safe/%2E./admin", "/safe/%2f/admin", "/safe\\..\\admin",
    "/submit\x7f?x=1", "/submit?x=1\x7f", "/submit#fragment",
    "/submit?x=%20", "/submit?x=hello world", "/submit//admin",
])
def test_post_url_rejects_client_canonicalization(path: str) -> None:
    with pytest.raises(ValueError):
        _post_url("http://127.0.0.1:19372" + path)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:0/submit", "http://127.0.0.1:80/submit",
    "http://127.000.000.001:19372/submit", "HTTP://127.0.0.1:19372/submit",
    "http://127.0.0.1:019372/submit", "http://127.0.0.1:19372/submit?x=" + "a" * 2048,
])
def test_post_url_rejects_ambiguous_authority_or_oversized_query(url: str) -> None:
    with pytest.raises(ValueError):
        _post_url(url)


def test_post_timeout_is_ambiguous_and_never_retried(monkeypatch) -> None:
    calls: list[tuple] = []

    async def fail_once(*args):
        calls.append(args)
        raise TimeoutError("response timed out after send")

    monkeypatch.setattr("vllm_omni.edge.agent.tools._streamed_post", fail_once)

    browser = object.__new__(ManagedEdgeBrowser)
    browser._post_snapshot = lambda url: ({
        "current_url": "https://example.test/page",
        "cookie_fingerprint": "f" * 64,
    }, "")
    with pytest.raises(RuntimeError, match="outcome unknown.*do not retry"):
        browser._post_exact(
            "https://example.test/submit", b"hello", "text/plain; charset=utf-8",
            "f" * 64, "https://example.test/page",
        )
    assert len(calls) == 1
    assert calls[0][:3] == (
        "https://example.test/submit", b"hello", "text/plain; charset=utf-8",
    )


def test_streamed_post_bounds_chunked_response_and_total_deadline(monkeypatch) -> None:
    import vllm_omni.edge.agent.tools as tools

    hits: Counter[str] = Counter()
    bodies: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            hits[self.path] += 1
            bodies.append(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/drip":
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for _ in range(100):
                    try:
                        self.wfile.write(b"1\r\na\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        break
                    threading.Event().wait(.05)
                return
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Set-Cookie", "session=mutated; Path=/")
            self.end_headers()
            chunk = b"z" * (64 * 1024)
            for _ in range(128):
                try:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    break

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root = f"http://127.0.0.1:{server.server_port}"
        __import__("httpx")  # Exclude one-time package import from the response-memory bound.
        tracemalloc.start()
        try:
            result = asyncio.run(_streamed_post(
                root + "/large", b"exact", "application/octet-stream", "session=original",
            ))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert result["status"] == 200
        assert len(result["body_excerpt"]) == 4096
        assert result["body_truncated"] is True
        assert result["body_size"] is None
        assert result["body_excerpt_raw"] is True
        assert peak < 6 * 1024 * 1024
        assert bodies == [b"exact"]
        assert hits["/large"] == 1

        monkeypatch.setattr(tools, "_POST_TOTAL_TIMEOUT_SECONDS", 1.2)
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            asyncio.run(_streamed_post(root + "/drip", b"exact", "application/json", ""))
        assert time.monotonic() - start < 2.5
        assert hits["/drip"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Edge")
def test_native_post_is_exact_approved_one_shot_and_does_not_follow_redirect(
    monkeypatch,
) -> None:
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
                malformed = [
                    root + "/safe/../admin?x=1", root + "/safe/./admin",
                    root + "/safe/%2e%2e/admin", root + "/safe/%2E./admin",
                    root + "/safe\\..\\admin", root + "/submit\x7f",
                    root + "/submit?x=" + "a" * 2048,
                ]
                for bad_url in malformed:
                    with pytest.raises(ValueError):
                        boundary.execute(_post_action(bad_url, body))
                assert not any(key.startswith("POST ") for key in hits)
                assert boundary._pending == {}
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

                browser._call(lambda: browser._context.add_cookies([
                    {"name": "route", "value": "root", "domain": "127.0.0.1", "path": "/"},
                    {"name": "route", "value": "endpoint", "domain": "127.0.0.1", "path": "/submit"},
                ]))
                original_cookies = browser._context.cookies
                observed = browser._call(lambda: original_cookies(root + "/submit"))
                assert sum(cookie["name"] == "route" for cookie in observed) == 2
                with pytest.raises(ApprovalRequired) as reordered:
                    boundary.execute(_post_action(root + "/submit", body))
                with monkeypatch.context() as patcher:
                    patcher.setattr(
                        browser._context, "cookies",
                        lambda url: list(reversed(original_cookies(url))),
                    )
                    with pytest.raises(RuntimeError, match="cookie context changed"):
                        boundary.approve(reordered.value.challenge.challenge_id)
                assert hits["POST /submit"] == 1
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
