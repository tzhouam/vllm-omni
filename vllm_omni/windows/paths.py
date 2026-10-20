# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Filesystem places POSIX code hard-codes, and the one ``os`` function Windows lacks."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_POSIX_SHM = "/dev/shm"


def shm_dir() -> str:
    """The directory for ``/dev/shm``-style lock and ack files.

    ``/dev/shm`` where it exists (Linux, WSL). On Windows a per-user directory
    under ``%LOCALAPPDATA%`` (falling back to the temp dir), created on first
    use. Callers that used to write ``f"/dev/shm/..."`` go through this so the
    path is the same on every platform that has one and a real one elsewhere.
    """
    if os.path.isdir(_POSIX_SHM):
        return _POSIX_SHM
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    path = Path(base) / "vllm_omni" / "shm"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def shm_path(name: str) -> str:
    return os.path.join(shm_dir(), name)


def provide_geteuid() -> str:
    """``os.geteuid``/``os.getuid`` do not exist on Windows.

    ``host_weight_runtime`` compares ``st_uid`` against ``os.geteuid()`` for a
    "did we create this" check; Windows ``os.stat`` reports ``st_uid == 0``, so
    returning 0 makes the comparison mean the same thing it can mean there.
    """
    if sys.platform != "win32":
        return "n/a"
    if hasattr(os, "geteuid"):
        return "already" if getattr(os.geteuid, "__vllm_omni_windows__", False) else "real"

    def geteuid() -> int:
        return 0

    geteuid.__vllm_omni_windows__ = True  # type: ignore[attr-defined]
    os.geteuid = geteuid  # type: ignore[attr-defined]
    if not hasattr(os, "getuid"):
        os.getuid = geteuid  # type: ignore[attr-defined]
    return "provided"
