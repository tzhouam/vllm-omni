# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Modules that do not exist on Windows, provided so ``import`` succeeds.

Two of them account for most Windows import deaths measured in W0–W8:

* ``fcntl`` — file locks. vLLM-Omni and model code call ``fcntl.flock`` for
  device-init locks, weight-store locks, media caches. Windows has
  ``msvcrt.locking`` instead: byte-range, mandatory, exclusive only.
* ``uvloop`` — no Windows wheels. Every ``uvloop.run(...)`` entry point can run
  on ``asyncio.run`` provided the loop is a *selector* loop (zmq.asyncio needs
  ``add_reader``; the Windows default Proactor loop has none).

Both shims are only installed when the real module is absent, and both are
built on injectable primitives so the logic is testable on Linux.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import types
from collections.abc import Callable

# ---------------------------------------------------------------------------
# fcntl
# ---------------------------------------------------------------------------

# POSIX values, so code that does arithmetic on them (``LOCK_EX | LOCK_NB``)
# behaves identically.
LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8

# The locked byte lives far past any content a caller writes into a lock file
# (PID lines and the like), so the lock never overlaps data. Same design as
# ``vllm_omni._filelock_compat`` (W1).
_LOCK_OFFSET = 1 << 30


class FcntlShim:
    """``flock``/``lockf`` on a byte-range primitive.

    ``locking(fd, mode, nbytes)`` follows ``msvcrt.locking``'s contract: mode
    ``NBLCK`` takes a non-blocking exclusive lock and raises ``OSError`` when
    held elsewhere; ``UNLCK`` releases. Shared locks (``LOCK_SH``) are served
    as exclusive because the primitive has none -- strictly more serialising
    than POSIX, never less safe; this is said once per process in the log.
    """

    def __init__(
        self,
        locking: Callable[[int, int, int], None],
        nblck: int,
        unlck: int,
        *,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval: float = 0.01,
    ) -> None:
        self._locking = locking
        self._nblck = nblck
        self._unlck = unlck
        self._sleep = sleep
        self._poll_interval = poll_interval
        self.shared_served_as_exclusive = 0

    def _at_lock_offset(self, fd: int, fn: Callable[[], None]) -> None:
        here = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            fn()
        finally:
            os.lseek(fd, here, os.SEEK_SET)

    def flock(self, fd: int, operation: int) -> None:
        fd = fd if isinstance(fd, int) else fd.fileno()
        if operation & LOCK_UN:
            self._at_lock_offset(fd, lambda: self._locking(fd, self._unlck, 1))
            return
        if operation & LOCK_SH and not operation & LOCK_EX:
            self.shared_served_as_exclusive += 1
        non_blocking = bool(operation & LOCK_NB)
        while True:
            try:
                self._at_lock_offset(fd, lambda: self._locking(fd, self._nblck, 1))
                return
            except OSError as exc:
                if non_blocking:
                    raise BlockingIOError(str(exc)) from exc
                self._sleep(self._poll_interval)

    def lockf(self, fd: int, cmd: int, len: int = 0, start: int = 0, whence: int = 0) -> None:  # noqa: A002
        # POSIX lockf's F_LOCK/F_TLOCK/F_ULOCK spell the same three states.
        self.flock(fd, cmd)

    def as_module(self) -> types.ModuleType:
        mod = types.ModuleType("fcntl")
        mod.__doc__ = "fcntl shim from vllm_omni.windows (msvcrt.locking underneath)"
        mod.LOCK_SH, mod.LOCK_EX, mod.LOCK_NB, mod.LOCK_UN = LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN
        # lockf's command constants, for callers that use them by name.
        mod.F_ULOCK, mod.F_LOCK, mod.F_TLOCK, mod.F_TEST = LOCK_UN, LOCK_EX, LOCK_EX | LOCK_NB, LOCK_NB
        mod.flock = self.flock
        mod.lockf = self.lockf
        mod.__vllm_omni_windows_shim__ = True
        mod._impl = self
        return mod


def make_msvcrt_fcntl() -> types.ModuleType:
    import msvcrt  # Windows only

    return FcntlShim(msvcrt.locking, msvcrt.LK_NBLCK, msvcrt.LK_UNLCK).as_module()


# ---------------------------------------------------------------------------
# uvloop
# ---------------------------------------------------------------------------


def selector_policy_class() -> type | None:
    return getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)


def ensure_selector_policy() -> bool:
    """Make new event loops selector loops. Returns True if it changed the policy."""
    policy = selector_policy_class()
    if policy is None:  # not Windows
        return False
    if isinstance(asyncio.get_event_loop_policy(), policy):
        return False
    asyncio.set_event_loop_policy(policy())
    return True


def make_uvloop_shim() -> types.ModuleType:
    """A ``uvloop`` whose ``run``/``install``/``new_event_loop`` are asyncio on a selector loop."""
    mod = types.ModuleType("uvloop")
    mod.__doc__ = "uvloop shim from vllm_omni.windows: asyncio on a selector event loop"
    mod.__version__ = "0.0.0+vllm_omni_windows_shim"

    def install() -> None:
        ensure_selector_policy()

    def new_event_loop() -> asyncio.AbstractEventLoop:
        ensure_selector_policy()
        return asyncio.new_event_loop()

    def run(main, **kwargs):
        ensure_selector_policy()
        return asyncio.run(main, **kwargs)

    mod.install = install
    mod.new_event_loop = new_event_loop
    mod.run = run
    mod.EventLoopPolicy = selector_policy_class() or asyncio.DefaultEventLoopPolicy
    mod.Loop = getattr(asyncio, "SelectorEventLoop", asyncio.AbstractEventLoop)
    mod.__vllm_omni_windows_shim__ = True
    return mod


# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------


def _real_module_importable(name: str) -> bool:
    if name in sys.modules:
        return not getattr(sys.modules[name], "__vllm_omni_windows_shim__", False)
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def provide(name: str, factory: Callable[[], types.ModuleType]) -> str:
    """Install ``factory()`` as ``sys.modules[name]`` unless a real module exists.

    Returns ``"real"`` (nothing done), ``"provided"`` (shim installed now) or
    ``"already"`` (our shim was installed earlier in this process).
    """
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__vllm_omni_windows_shim__", False):
        return "already"
    if _real_module_importable(name):
        return "real"
    sys.modules[name] = factory()
    return "provided"
