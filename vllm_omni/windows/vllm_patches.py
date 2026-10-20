# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runtime patches of vLLM control-plane functions, and verification of the rest.

The rule (design §4): a function may be replaced *whole* at runtime only when
its entire body is platform logic and its upstream body is one we recognise by
hash. Sites whose fix sits in the middle of a larger function (the engine
startup sentinel poll, the sampler seed dtype) must come from the wheel -- the
Windows branch carries them -- and are only *verified* here, so a run on an
unpatched wheel is reported as ``wheel_required`` at the exact site instead of
failing somewhere downstream.

Every outcome string used here: ``already`` (the wheel carries the branch
patch, marker found), ``patched`` (replaced now), ``unknown_upstream`` (no
marker and the upstream body is not one we know -- left alone, reported),
``wheel_required`` (verify-only site, marker absent), ``not_needed`` (another
layer covers it), ``n/a`` (off Windows or module absent).
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from typing import Any

BRANCH_MARK = "edge-infer"  # every branch patch carries "[edge-infer W? windows patch]"

# sha256[:16] of inspect.getsource() of the *upstream* function, measured on the
# PyPI wheels vllm==0.28.0 and vllm==0.29.0 (identical bodies at both tags).
KNOWN_UPSTREAM = {
    "get_open_zmq_ipc_path": {"1d9aeb608d4c5dbc"},
    "find_loaded_library": {"b7916f0a8ebae56c"},
}


def _src_hash(obj: Any) -> str | None:
    try:
        return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return None


def _has_mark(obj: Any) -> bool:
    try:
        return BRANCH_MARK in inspect.getsource(obj)
    except (OSError, TypeError):
        return False


# ---------------------------------------------------------------------------
# replaced whole
# ---------------------------------------------------------------------------


def _win_get_open_zmq_ipc_path() -> str:
    """The W0 body: libzmq on Windows has no ipc:// transport; loopback TCP is the local option.

    This drops the filesystem permissions a Unix socket carries -- a loopback
    port is reachable by any local user -- so it is a fallback conditional on
    ipc being truly unavailable, never a preference.
    """
    import zmq

    from vllm import envs
    from vllm.utils.network_utils import get_open_port, get_tcp_uri

    if not zmq.has("ipc"):
        return get_tcp_uri("127.0.0.1", get_open_port())
    from uuid import uuid4

    return f"ipc://{envs.VLLM_RPC_BASE_PATH}/{uuid4()}"


def _make_win_find_loaded_library(original):
    def find_loaded_library(lib_name: str) -> str | None:
        """On Windows an already-imported extension module's __file__ IS the loaded path (W0)."""
        if sys.platform != "win32":
            return original(lib_name)
        for mod in list(sys.modules.values()):
            f = getattr(mod, "__file__", None)
            if not f:
                continue
            base = f.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
            if base.startswith((f"{lib_name}.", f"{lib_name}-")):
                return f
        return None

    find_loaded_library.__vllm_omni_windows__ = True  # type: ignore[attr-defined]
    return find_loaded_library


def _replace(module, name: str, replacement) -> str:
    current = getattr(module, name, None)
    if current is None:
        return "n/a"
    if getattr(current, "__vllm_omni_windows__", False):
        return "already"
    if _has_mark(current):
        return "already"
    if _src_hash(current) not in KNOWN_UPSTREAM.get(name, set()):
        return "unknown_upstream"
    setattr(module, name, replacement)
    return "patched"


def patch_get_open_zmq_ipc_path() -> str:
    if sys.platform != "win32":
        return "n/a"
    try:
        from vllm.utils import network_utils
    except ImportError:
        return "n/a"
    _win_get_open_zmq_ipc_path.__vllm_omni_windows__ = True  # type: ignore[attr-defined]
    status = _replace(network_utils, "get_open_zmq_ipc_path", _win_get_open_zmq_ipc_path)
    if status == "patched":
        # Callers that imported the name directly (``from ... import get_open_zmq_ipc_path``)
        # before activation keep the old binding; the ones that matter import lazily
        # (Omni's stage_engine_startup) or after plugin load (vLLM's own engine code).
        pass
    return status


def patch_find_loaded_library() -> str:
    if sys.platform != "win32":
        return "n/a"
    try:
        from vllm.utils import system_utils
    except ImportError:
        return "n/a"
    current = getattr(system_utils, "find_loaded_library", None)
    if current is None:
        return "n/a"
    if getattr(current, "__vllm_omni_windows__", False) or _has_mark(current):
        return "already"
    if _src_hash(current) not in KNOWN_UPSTREAM["find_loaded_library"]:
        return "unknown_upstream"
    system_utils.find_loaded_library = _make_win_find_loaded_library(current)
    return "patched"


# ---------------------------------------------------------------------------
# verify only (the fix must be in the wheel)
# ---------------------------------------------------------------------------


def _verify(import_path: str, attr: str | None, needle: str) -> str:
    try:
        module = __import__(import_path, fromlist=["_"])
    except ImportError:
        return "n/a"
    obj = getattr(module, attr) if attr else module
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError):
        return "n/a"
    return "already" if needle in src else "wheel_required"


def verify_wheel_sites() -> dict[str, str]:
    """The sites a runtime plugin cannot fix; each must be `already` on Windows."""
    if sys.platform != "win32":
        return {}
    out = {
        # v1/engine/utils.py: zmq cannot poll a process HANDLE; 500 ms poll + exitcode checks.
        "wait_for_engine_startup_sentinel_poll": _verify("vllm.v1.engine.utils", "wait_for_engine_startup", "_win_sentinels"),
        # sample/states.py: numpy's default int is int32 on Windows; int64 seed bounds overflow.
        "sampler_seed_int64": _verify("vllm.v1.worker.gpu.sample.states", None, "dtype=np.int64"),
    }
    # v1/utils.py binds uvloop lazily on the branch; with the uvloop shim present the
    # module-level import would succeed anyway, so absence is "not_needed", not a failure.
    status = _verify("vllm.v1.utils", None, "__getattr__")
    if status == "wheel_required":
        status = "not_needed" if "uvloop" in sys.modules else "wheel_required"
    out["v1_utils_uvloop_lazy"] = status
    # launcher.serve_http: the branch has a signal.signal fallback; the loop-class patch
    # (aio.patch_signal_handlers) covers an unpatched wheel too.
    status = _verify("vllm.entrypoints.launchers.launcher", "serve_http", "signal-handler")
    out["serve_http_signal_fallback"] = "already" if status == "already" else ("not_needed" if status == "wheel_required" else status)
    return out
