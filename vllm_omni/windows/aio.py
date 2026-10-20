# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""asyncio on Windows: the selector loop, and signal handlers that exist.

``loop.add_signal_handler`` is Unix-only; on Windows every loop raises
``NotImplementedError``. vLLM's ``serve_http`` registers SIGINT/SIGTERM through
it, so the API server died right after the engine came up (W8, measured). The
fallback registers a plain ``signal.signal`` handler that hops back onto the
loop with ``call_soon_threadsafe`` -- the same shutdown path, registered the
only way Windows allows. Signals Windows does not have raise the original
error, so nothing is silently ignored.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from collections.abc import Callable
from typing import Any

from vllm_omni.windows.shims import ensure_selector_policy

_MARK = "__vllm_omni_windows_signal_fallback__"

_WINDOWS_SIGNALS = {
    getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGABRT") if hasattr(signal, name)
}


def make_signal_fallback(original_add: Callable, original_remove: Callable):
    """Build ``add_signal_handler``/``remove_signal_handler`` replacements.

    Returned functions try the loop's own implementation first (a real selector
    loop on Unix keeps its native behaviour), and fall back to
    ``signal.signal`` when the loop says ``NotImplementedError``.
    """

    def add_signal_handler(self, sig: int, callback: Callable, *args: Any) -> None:
        try:
            return original_add(self, sig, callback, *args)
        except NotImplementedError:
            if sig not in _WINDOWS_SIGNALS:
                raise

            def _handler(signum, frame):  # runs on the main thread, outside the loop
                self.call_soon_threadsafe(callback, *args)

            signal.signal(sig, _handler)
            registry = self.__dict__.setdefault(_MARK, {})
            registry[sig] = _handler

    def remove_signal_handler(self, sig: int) -> bool:
        registry = self.__dict__.get(_MARK) or {}
        if sig in registry:
            signal.signal(sig, signal.SIG_DFL)
            del registry[sig]
            return True
        try:
            return original_remove(self, sig)
        except NotImplementedError:
            return False

    add_signal_handler.__vllm_omni_windows__ = True  # type: ignore[attr-defined]
    remove_signal_handler.__vllm_omni_windows__ = True  # type: ignore[attr-defined]
    return add_signal_handler, remove_signal_handler


def patch_signal_handlers() -> str:
    """Install the fallback on the loop classes Windows uses. Returns already|patched|n/a."""
    if sys.platform != "win32":
        return "n/a"
    targets = [cls for cls in (getattr(asyncio, "ProactorEventLoop", None), getattr(asyncio, "SelectorEventLoop", None)) if cls]
    if not targets:
        return "n/a"
    if all(getattr(cls.add_signal_handler, "__vllm_omni_windows__", False) for cls in targets):
        return "already"
    for cls in targets:
        if getattr(cls.add_signal_handler, "__vllm_omni_windows__", False):
            continue
        add, remove = make_signal_fallback(cls.add_signal_handler, cls.remove_signal_handler)
        cls.add_signal_handler = add  # type: ignore[method-assign]
        cls.remove_signal_handler = remove  # type: ignore[method-assign]
    return "patched"


def install_selector_policy() -> str:
    if sys.platform != "win32":
        return "n/a"
    return "set" if ensure_selector_policy() else "already"


def running_loop_class() -> str | None:
    try:
        return type(asyncio.get_running_loop()).__name__
    except RuntimeError:
        return None
