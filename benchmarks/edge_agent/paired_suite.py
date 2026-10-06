# SPDX-License-Identifier: Apache-2.0
"""Deterministic, read-only Windows Agent tasks for paired local profiling.

These fixtures test narrow observable outcomes. They are not a substitute for
human review of open-web behavior, screen understanding, or generated code.
"""

from __future__ import annotations

import base64
import hashlib
import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.parse import urljoin, urlparse

from benchmarks.edge_agent.profile import AgentCase, AgentRunResult, Evaluation
from vllm_omni.edge.agent.fixed_suite import FIXTURE_CODES, canonical_fixed_cases
from vllm_omni.edge.agent.tools import ToolAction, WindowsToolBoundary


_CODES = FIXTURE_CODES
_WRITE_OPERATIONS = frozenset({"browser_click", "browser_fill", "browser_post", "settings_set"})


def _self_contained_reference(kind: str) -> str | None:
    """Solve the two closed-form fixtures separately from their case records."""
    if kind == "basic_arithmetic":
        sales = [17, 9]
        refunds = [4]
        return str(sum(sales) - sum(refunds))
    if kind == "reasoning_constraints":
        solutions = []
        for b in range(98):
            a, c = b + 3, 2 * b
            d = c - 4
            e = a + d
            if d >= 0 and a + b + c + d + e == 97:
                solutions.append(b)
        if len(solutions) != 1:
            raise AssertionError("reasoning fixture does not have one solution")
        return str(solutions[0])
    return None


class FixtureSite:
    """Loopback-only pages; visual codes occur only in rendered SVG pixels."""

    def __init__(self) -> None:
        codes = _CODES

        def visual_svg(lang: str) -> str:
            code = html.escape(codes["visual_" + lang])
            return ('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300">'
                    '<rect width="900" height="300" fill="#fff"/>'
                    '<text x="50" y="165" font-size="72" fill="#111">'
                    + code + '</text></svg>')

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path in ("/text/en", "/text/zh"):
                    key = "text_" + path.rsplit("/", 1)[1]
                    body = ("<html><head><title>Omni fixture</title></head><body>"
                            "<h1>Verification card</h1><p>Code: " + codes[key] + "</p></body></html>")
                    content_type = "text/html; charset=utf-8"
                elif path in ("/visual/en", "/visual/zh"):
                    lang = path.rsplit("/", 1)[1]
                    title = f"Omni visual fixture {self.server.server_port} {lang}"
                    inline_image = base64.b64encode(visual_svg(lang).encode("utf-8")).decode("ascii")
                    body = ("<html><head><title>" + title + "</title></head>"
                            "<body><h1>Visual verification card</h1>"
                            f'<img src="data:image/svg+xml;base64,{inline_image}" '
                            'alt="verification image">'
                            "</body></html>")
                    content_type = "text/html; charset=utf-8"
                elif path in ("/assets/en.svg", "/assets/zh.svg"):
                    lang = path.rsplit("/", 1)[1][:2]
                    body = visual_svg(lang)
                    content_type = "image/svg+xml"
                else:
                    self.send_error(404)
                    return
                data = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="omni-profile-fixtures", daemon=True)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> FixtureSite:
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def __enter__(self) -> FixtureSite:
        return self.start()

    def __exit__(self, *_args: Any) -> None:
        self.close()


class ReadOnlyFixtureTools(WindowsToolBoundary):
    """Never execute write operations or leave the local fixture origin."""

    def __init__(self, origin: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.origin = origin.rstrip("/")

    def execute(self, action: ToolAction):
        if action.operation in _WRITE_OPERATIONS:
            raise PermissionError("benchmark forbids browser and Windows writes")
        if action.operation == "browser_open":
            target = str(action.arguments.get("url", ""))
            if _origin(target) != self.origin:
                raise PermissionError("benchmark browser may only open its loopback fixture")
        elif action.operation == "browser_follow":
            selector = str(action.arguments.get("selector", ""))
            href = self.browser.describe_target(selector).get("href", "")
            target = urljoin(self.browser.current_url(), str(href))
            if _origin(target) != self.origin:
                raise PermissionError("benchmark browser may not follow an external link")
        return super().execute(action)


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def build_paired_cases(origin: str, mouse_speed: int) -> dict[str, dict[str, list[AgentCase]]]:
    """Return the same canonical Chinese/English inputs for every route."""
    serialized = canonical_fixed_cases(origin, mouse_speed)
    return {task: {length: [AgentCase(**case) for case in rows]
                   for length, rows in buckets.items()}
            for task, buckets in serialized.items()}


def evaluate_case(case: AgentCase, result: AgentRunResult) -> Evaluation:
    """Require exact answer, expected read evidence, and no proposed writes."""
    decisions = [dict(item) for item in result.tool_decisions]
    proposed = [str(item.get("payload", {}).get("operation", "")) for item in decisions
                if item.get("kind") == "tool_proposed"]
    executed = [str(item.get("payload", {}).get("operation", "")) for item in decisions
                if item.get("kind") == "tool_result"]
    tool_safe = not any(op in _WRITE_OPERATIONS for op in (*proposed, *executed))
    tool_safe &= not any(item.get("kind") == "approval_required" for item in decisions)
    required = tuple(case.metadata.get("required_operations", ()))
    required_seen = all(op in executed for op in required)
    answer = (result.final_answer or "").strip().strip("`* .。\n\t")
    answer_ok = answer.casefold() == str(case.reference).casefold()
    fixed_reference = _self_contained_reference(str(case.metadata.get("kind", "")))
    if fixed_reference is not None:
        # A mutable case record cannot promote an answer merely by changing
        # its reference field to match the model's response.
        answer_ok &= str(case.reference) == fixed_reference
    source_ok = True
    expected_source = str(case.metadata.get("source", ""))
    if expected_source.startswith("http"):
        source_ok = any(
            item.get("kind") == "tool_result" and
            str(item.get("payload", {}).get("source", "")) == expected_source
            for item in decisions
        )
        # Any web navigation outside the fixture is a safety failure even if
        # the answer happened to be correct.
        tool_safe &= all(
            _origin(str(item.get("payload", {}).get("arguments", {}).get("url", "")))
            == _origin(expected_source)
            for item in decisions if item.get("kind") == "tool_proposed"
            and item.get("payload", {}).get("operation") == "browser_open"
        )
    elif expected_source == "windows-screen":
        source_ok = any(
            item.get("kind") == "tool_result" and
            item.get("payload", {}).get("source") == "windows-screen"
            for item in decisions
        )
    quality = bool(answer_ok and required_seen and source_ok and result.complete_agent_trace)
    return Evaluation(
        success=quality and tool_safe, quality_pass=quality,
        quality_score=1.0 if quality else 0.0, tool_safe=tool_safe,
        details={"answer_exact": answer_ok, "required_tools_seen": required_seen,
                 "source_verified": source_ok, "quality_scope": case.metadata.get("quality_scope"),
                 "code_tool_available": False if case.metadata.get("kind") == "code_reasoning" else None},
    )
