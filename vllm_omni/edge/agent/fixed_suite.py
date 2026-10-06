# SPDX-License-Identifier: Apache-2.0
"""Canonical, package-local definition of the fixed native Agent fixtures.

The release verifier must reconstruct cases without importing the benchmark
package. The benchmark runner consumes this same definition so a changed
prompt, reference, or metadata cannot silently retain the suite identity.
"""

from __future__ import annotations

import hashlib
from urllib.parse import urlparse


FIXTURE_CODES = {
    "text_en": "CEDAR-4827", "text_zh": "LANSHAN-6153",
    "visual_en": "ORBIT-7391", "visual_zh": "HAIYAN-2648",
    "memory_en": "MEMORY-5864", "memory_zh": "JII-9237",
}
_FILLER = {
    "en": "This neutral background fixes input length. Follow the primary task and its evidence. ",
    "zh": "这段测试背景仅用于固定输入长度，请以指定的实际观察为证据。",
}
_EXTRA = {"short": 0, "medium": 6, "long": 28}
_BASIC_EXTRA = {"short": 0, "medium": 6, "long": 18}


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def canonical_fixed_cases(origin: str, mouse_speed: int) -> dict[str, dict[str, list[dict]]]:
    """Return exact serialized ``AgentCase`` payloads for the fixed suite."""
    if _origin(origin) != origin.rstrip("/") or not 1 <= mouse_speed <= 20:
        raise ValueError("exact fixture origin and current mouse speed required")
    classes = ("basic", "browser_text", "browser_vision", "windows_settings",
               "memory", "code_tools", "long_reasoning")
    lengths = ("short", "medium", "long")
    suites: dict[str, dict[str, list[dict]]] = {
        name: {length: [] for length in lengths} for name in classes
    }
    for length in lengths:
        for lang in ("en", "zh"):
            language = "en-US" if lang == "en" else "zh-CN"
            text_url = origin + f"/text/{lang}"
            visual_url = origin + f"/visual/{lang}"
            text_base = (
                f"Use the browser to open `{text_url}`, read the page, and answer with only its verification identifier."
                if lang == "en" else
                f"请用浏览器打开 `{text_url}`，读取网页，只回答页面中的验证码。"
            )
            visual_base = (
                f"Use the browser to open `{visual_url}`, then use browser_screenshot to read the identifier in the image. "
                "Answer with only that identifier."
                if lang == "en" else
                f"请用浏览器打开 `{visual_url}`，然后使用 browser_screenshot 从图片像素读取验证码，只回答该码。"
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
                ("basic", "basic_arithmetic", basic_base, "22", (), "self-contained-arithmetic"),
                ("browser_text", "browser_text", text_base, FIXTURE_CODES["text_" + lang],
                 ("browser_open", "browser_read"), text_url),
                ("browser_vision", "screen_vision", visual_base, FIXTURE_CODES["visual_" + lang],
                 ("browser_open", "browser_screenshot"), visual_url),
                ("browser_vision", "desktop_screen", desktop_base, FIXTURE_CODES["visual_" + lang],
                 ("screen_capture",), "windows-screen"),
                ("windows_settings", "settings_read", settings_base, str(mouse_speed),
                 ("settings_read",), "windows-setting:mouse_speed"),
                ("memory", "memory_recall", memory_base, FIXTURE_CODES["memory_" + lang],
                 (), "benchmark-fixture:memory"),
                ("code_tools", "code_reasoning", code_base, "14", (), "static-code"),
                ("long_reasoning", "reasoning_constraints", reasoning_base, "11", (),
                 "self-contained-constraints"),
            )
            for task_class, kind, base, answer, required, source in entries:
                repetitions = (_BASIC_EXTRA if task_class == "basic" else _EXTRA)[length]
                prompt = base + "\n" + _FILLER[lang] * repetitions
                suites[task_class][length].append({
                    "case_id": f"{kind}-{lang}-{length}",
                    "task_class": task_class, "language": language,
                    "length": length, "prompt": prompt,
                    "reference": answer, "input_tokens": None,
                    "metadata": {
                        "kind": kind, "required_operations": list(required),
                        "source": source,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "input_chars": len(prompt),
                        "input_utf8_bytes": len(prompt.encode("utf-8")),
                        "quality_scope": (
                            "code_result_only_no_execution_tool" if kind == "code_reasoning"
                            else "fixed_self_contained_task" if task_class in {"basic", "long_reasoning"}
                            else "fixed_local_fixture"
                        ),
                    },
                })
    return suites
