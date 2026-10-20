# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``vllm_omni.windows`` -- host-OS compatibility so vLLM and vLLM-Omni run on native Windows.

Design: ``analysis/design_windows_compat_plugin_20260921.md`` (edge-infer). In
one sentence: the POSIX assumptions that killed Windows runs in W0–W8 are a
handful of classes repeated across vLLM, vLLM-Omni and model code; this
package provides the missing modules (``fcntl``, ``uvloop``), fixes the
process state Windows gets wrong (event loop policy, signal handlers), replaces
whole the two vLLM functions whose entire body is platform logic, and
*reports* the sites that must come from the Windows wheel.

Three doors, all idempotent, so import order does not matter:

* ``vllm_omni/patch.py`` calls :func:`activate` first thing;
* the ``vllm.general_plugins`` entry point ``vllm_omni_windows`` -- vLLM calls
  it in the API server, every engine-core child and every worker;
* the ``vllm_omni.general_plugins`` entry point of the same name.

Off Windows every function here is a no-op that reports ``n/a``.
"""

from __future__ import annotations

import logging
import sys

from vllm_omni.windows.report import LEDGER, report

logger = logging.getLogger(__name__)

__all__ = ["activate", "is_active", "is_windows", "report"]


def is_windows() -> bool:
    return sys.platform == "win32"


def is_active() -> bool:
    return bool(LEDGER["active"])


def activate() -> dict[str, str]:
    """Apply every layer once per process. Returns the per-step outcome."""
    LEDGER["activations"] += 1
    if not is_windows():
        LEDGER["steps"] = {"platform": "n/a"}
        return LEDGER["steps"]
    if LEDGER["active"]:
        return LEDGER["steps"]

    from vllm_omni.windows import aio, paths, shims, vllm_patches

    steps: dict[str, str] = {}
    # 1. modules that do not exist here -- before anything can import them
    steps["fcntl"] = shims.provide("fcntl", shims.make_msvcrt_fcntl)
    steps["uvloop"] = shims.provide("uvloop", shims.make_uvloop_shim)
    # 2. process state
    steps["event_loop_policy"] = aio.install_selector_policy()
    steps["signal_handlers"] = aio.patch_signal_handlers()
    steps["os_geteuid"] = paths.provide_geteuid()
    # 3. vLLM functions replaced whole (only when the wheel lacks the branch patch)
    steps["get_open_zmq_ipc_path"] = vllm_patches.patch_get_open_zmq_ipc_path()
    steps["find_loaded_library"] = vllm_patches.patch_find_loaded_library()
    LEDGER["steps"] = steps
    LEDGER["active"] = True
    logger.info("[vllm_omni.windows] active: %s", ", ".join(f"{k}={v}" for k, v in steps.items()))
    fc = sys.modules.get("fcntl")
    impl = getattr(fc, "_impl", None)
    if impl is not None and getattr(impl, "shared_served_as_exclusive", 0):
        logger.info("[vllm_omni.windows] LOCK_SH served as exclusive (msvcrt has no shared locks)")
    return steps
