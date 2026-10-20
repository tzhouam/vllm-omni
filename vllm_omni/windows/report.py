# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The audit: what the Windows layer did in this process, and what it could not do.

Everything ``activate()`` does lands in ``LEDGER``; ``report()`` adds the
facts a Windows record needs next to a result (zmq transport, loop policy,
which vLLM sites came from the wheel, which native modules are present, which
optional dependencies import). Nothing here changes state.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from typing import Any

LEDGER: dict[str, Any] = {"active": False, "steps": {}, "activations": 0}

OPTIONAL_DEPS = ("torchaudio", "soundfile", "diffusers", "flashinfer", "pynvml", "uvloop", "fcntl")


def _importable(name: str) -> str:
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__vllm_omni_windows_shim__", False):
        return "shim"
    try:
        return "yes" if importlib.util.find_spec(name) is not None else "no"
    except (ImportError, ValueError):
        return "no"


def _native_modules() -> list[str]:
    try:
        import vllm
    except ImportError:
        return []
    from pathlib import Path

    root = Path(vllm.__file__).parent
    suffixes = (".pyd", ".so")
    return sorted(
        str(p.relative_to(root)).replace("\\", "/").split(".", 1)[0]
        for p in root.rglob("*")
        if p.suffix in suffixes and "third_party" not in p.parts
    )


def report() -> dict[str, Any]:
    from vllm_omni.windows import aio, paths, vllm_patches

    out: dict[str, Any] = {
        "platform": sys.platform,
        "active": LEDGER["active"],
        "activations": LEDGER["activations"],
        "steps": dict(LEDGER["steps"]),
    }
    out["asyncio"] = {
        "policy": type(asyncio.get_event_loop_policy()).__name__,
        "running_loop": aio.running_loop_class(),
    }
    try:
        import zmq

        out["zmq"] = {"version": zmq.zmq_version(), "has_ipc": bool(zmq.has("ipc"))}
    except ImportError:
        out["zmq"] = None
    out["shm_dir"] = paths.shm_dir() if sys.platform == "win32" else None
    try:
        import vllm

        out["vllm"] = {"version": vllm.__version__, "wheel_sites": vllm_patches.verify_wheel_sites()}
    except ImportError:
        out["vllm"] = None
    out["native_modules"] = _native_modules()
    out["optional_deps"] = {name: _importable(name) for name in OPTIONAL_DEPS}
    return out
