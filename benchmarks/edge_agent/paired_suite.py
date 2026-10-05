# SPDX-License-Identifier: Apache-2.0
"""Deterministic, read-only Windows Agent tasks for paired local profiling.

These fixtures test narrow observable outcomes. They are not a substitute for
human review of open-web behavior, screen understanding, or generated code.
"""

from __future__ import annotations

import hashlib
import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.parse import urljoin, urlparse

from benchmarks.edge_agent.profile import AgentCase, AgentRunResult, Evaluation, LENGTH_BUCKETS
from vllm_omni.edge.agent.tools import ToolAction, WindowsToolBoundary


_CODES = {
    "text_en": "CEDAR-4827",
    "text_zh": "LANSHAN-6153",
    "visual_en": "ORBIT-7391",
    "visual_zh": "HAIYAN-2648",
    "memory_en": "MEMORY-5864",
    "memory_zh": "JII-9237",
}
_WRITE_OPERATIONS = frozenset({"browser_click", "browser_fill", "settings_set"})
_FILLER = {
    "en": "This neutral background fixes input length. Follow the primary task and its evidence. ",
    "zh": "这段测试背景仅用于固定输入长度，请以指定的实际观察为证据。",
}
_EXTRA = {"short": 0, "medium": 6, "long": 28}
# A basic request must stay below classify_task's 2,000-character
# long-reasoning boundary even in the long input bucket.
_BASIC_EXTRA = {"short": 0, "medium": 6, "long": 18}


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


def _prompt(base: str, language: str, length: str, *, basic: bool = False) -> str:
    repetitions = _BASIC_EXTRA[length] if basic else _EXTRA[length]
    return base + "\n" + _FILLER[language] * repetitions


class FixtureSite:
    """Loopback-only pages; visual codes occur only in rendered SVG pixels."""

    def __init__(self) -> None:
        codes = _CODES

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
                    body = ("<html><head><title>Omni visual fixture</title></head>"
                            "<body><h1>Visual verification card</h1>"
                            f'<img src="/assets/{lang}.svg" alt="verification image">'
                            "</body></html>")
                    content_type = "text/html; charset=utf-8"
                elif path in ("/assets/en.svg", "/assets/zh.svg"):
                    lang = path.rsplit("/", 1)[1][:2]
                    code = html.escape(codes["visual_" + lang])
                    body = ('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300">'
                            '<rect width="900" height="300" fill="#fff"/>'
                            '<text x="50" y="165" font-size="72" fill="#111">'
                            + code + '</text></svg>')
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
    """Return the same Chinese/English inputs for every route and length."""
    if _origin(origin) != origin.rstrip("/") or not 1 <= mouse_speed <= 20:
        raise ValueError("exact fixture origin and current mouse speed required")
    suites: dict[str, dict[str, list[AgentCase]]] = {
        name: {length: [] for length in LENGTH_BUCKETS}
        for name in ("basic", "browser_text", "browser_vision", "windows_settings",
                     "memory", "code_tools", "long_reasoning")
    }
    for length in LENGTH_BUCKETS:
        for lang in ("en", "zh"):
            language = "en-US" if lang == "en" else "zh-CN"
            text_url = origin + f"/text/{lang}"
            visual_url = origin + f"/visual/{lang}"
            text_base = (
                f"Use the browser to open {text_url}, read the page, and answer with only its verification identifier."
                if lang == "en" else
                f"请用浏览器打开 {text_url}，读取网页，只回答页面中的验证码。"
            )
            visual_base = (
                f"Use the browser to open {visual_url}, then use browser_screenshot to read the identifier in the image. "
                "Answer with only that identifier."
                if lang == "en" else
                f"请用浏览器打开 {visual_url}，然后使用 browser_screenshot 从图片像素读取验证码，只回答该码。"
            )
            desktop_base = (
                "A browser verification card is already visible on this Windows desktop. "
                "Use screen_capture to read its image; answer with only the identifier."
                if lang == "en" else
                "浏览器验证码卡片已显示在 Windows 屏幕上。请用 screen_capture 从屏幕图片读取，只回答该码。"
            )
            settings_base = (
                "Read the current Windows setting mouse_speed with settings_read and answer only its integer value."
                if lang == "en" else
                "请用 settings_read 读取 Windows setting mouse_speed，只回答当前整数值。"
            )
            memory_base = (
                "Recall from local memory the verification identifier for the Cedar memory card. Answer only the identifier."
                if lang == "en" else
                "请从本地记忆中回忆蓝杉记忆卡的验证码，只回答该码。"
            )
            code_base = (
                "For this Python code, what exact integer is printed? Answer only the integer: "
                "print(sum(i * i for i in range(4)))"
                if lang == "en" else
                "以下 Python 代码输出什么整数？只回答整数：print(sum(i * i for i in range(4)))"
            )
            basic_base = (
                "A cafe sold 17 cups in the morning and 9 in the afternoon, "
                "then refunded 4 cups. What is the net number sold? Answer only the integer."
                if lang == "en" else
                "一家店上午售出17杯，下午售出9杯，随后退回4杯。净售出多少杯？只回答整数。"
            )
            reasoning_base = (
                "Derive the answer from these constraints. Five teams A, B, C, D, E "
                "share tickets: A has 3 more than B; C has twice B; D has 4 fewer "
                "than C; E equals A plus D. Their total is 97. How many tickets "
                "does B have? Answer only the integer."
                if lang == "en" else
                "请推导以下约束。五组 A、B、C、D、E 分得票券：A 比 B 多3张；C 是 B 的两倍；"
                "D 比 C 少4张；E 等于 A 与 D 之和。五组共97张。B 分得几张？只回答整数。"
            )
            entries = (
                ("basic", "basic_arithmetic", basic_base, str(17 + 9 - 4),
                 (), "self-contained-arithmetic"),
                ("browser_text", "browser_text", text_base, _CODES["text_" + lang],
                 ("browser_open", "browser_read"), text_url),
                ("browser_vision", "screen_vision", visual_base, _CODES["visual_" + lang],
                 ("browser_open", "browser_screenshot"), visual_url),
                ("browser_vision", "desktop_screen", desktop_base, _CODES["visual_" + lang],
                 ("screen_capture",), "windows-screen"),
                ("windows_settings", "settings_read", settings_base, str(mouse_speed),
                 ("settings_read",), "windows-setting:mouse_speed"),
                ("memory", "memory_recall", memory_base, _CODES["memory_" + lang],
                 (), "benchmark-fixture:memory"),
                ("code_tools", "code_reasoning", code_base, "14", (), "static-code"),
                ("long_reasoning", "reasoning_constraints", reasoning_base,
                 str((97 + 2) // 9), (), "self-contained-constraints"),
            )
            for task_class, kind, base, answer, required, source in entries:
                prompt = _prompt(base, lang, length, basic=task_class == "basic")
                suites[task_class][length].append(AgentCase(
                    case_id=f"{kind}-{lang}-{length}", task_class=task_class,
                    language=language, length=length, prompt=prompt, reference=answer,
                    metadata={
                        "kind": kind, "required_operations": list(required),
                        "source": source, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "input_chars": len(prompt),
                        "input_utf8_bytes": len(prompt.encode("utf-8")),
                        "quality_scope": (
                            "code_result_only_no_execution_tool" if kind == "code_reasoning"
                            else "fixed_self_contained_task" if task_class in {"basic", "long_reasoning"}
                            else "fixed_local_fixture"
                        ),
                    },
                ))
    return suites


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
            str(item.get("payload", {}).get("source", "")).startswith(expected_source)
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
