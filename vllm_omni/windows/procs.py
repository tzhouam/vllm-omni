# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Process helpers that work on every OS (no ``os.killpg``, no ``kill -9``)."""

from __future__ import annotations

import subprocess


def kill_tree(pid: int, *, timeout: float = 30.0) -> dict[str, int]:
    """Terminate ``pid`` and every descendant; escalate to kill after ``timeout``.

    Returns counts: ``{"terminated": n, "killed": m}``. Missing processes are
    not an error (the tree may already be gone).
    """
    import psutil

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return {"terminated": 0, "killed": 0}
    procs = [*parent.children(recursive=True), parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    return {"terminated": len(procs) - len(alive), "killed": len(alive)}


def kill_popen_tree(proc: subprocess.Popen, *, timeout: float = 30.0) -> dict[str, int]:
    return kill_tree(proc.pid, timeout=timeout)
