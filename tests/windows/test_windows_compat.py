# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``vllm_omni.windows`` on any OS: the shim logic through injected primitives,
the no-op contract off Windows, the report schema, and the marker/hash rules
that decide whether a vLLM function is replaced or left alone."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

import pytest

from vllm_omni.windows import activate, is_active, is_windows, report
from vllm_omni.windows import aio, shims, vllm_patches
from vllm_omni.windows.paths import shm_dir, shm_path


# ---- fcntl shim ------------------------------------------------------------


class FakeLocking:
    """Byte-range primitive with msvcrt.locking's contract, held state in memory."""

    NBLCK, UNLCK = 2, 0

    def __init__(self) -> None:
        self.held: set[int] = set()
        self.calls: list[tuple[int, int]] = []

    def __call__(self, fd: int, mode: int, nbytes: int) -> None:
        self.calls.append((fd, mode))
        if mode == self.NBLCK:
            if fd in self.held:
                raise OSError(36, "Resource deadlock avoided")
            self.held.add(fd)
        elif mode == self.UNLCK:
            self.held.discard(fd)


@pytest.fixture
def lock_fd():
    fd, path = tempfile.mkstemp()
    yield fd
    os.close(fd)
    os.unlink(path)


def _shim(locking: FakeLocking, sleeps: list[float]):
    return shims.FcntlShim(locking, FakeLocking.NBLCK, FakeLocking.UNLCK, sleep=sleeps.append, poll_interval=0.01)


def test_flock_exclusive_then_unlock(lock_fd):
    locking, sleeps = FakeLocking(), []
    s = _shim(locking, sleeps)
    s.flock(lock_fd, shims.LOCK_EX)
    assert lock_fd in locking.held
    s.flock(lock_fd, shims.LOCK_UN)
    assert lock_fd not in locking.held
    assert sleeps == []


def test_flock_non_blocking_raises_blockingioerror_when_held(lock_fd):
    locking, sleeps = FakeLocking(), []
    locking.held.add(lock_fd)
    s = _shim(locking, sleeps)
    with pytest.raises(BlockingIOError):
        s.flock(lock_fd, shims.LOCK_EX | shims.LOCK_NB)
    assert sleeps == []


def test_flock_blocking_waits_until_released(lock_fd):
    locking = FakeLocking()
    locking.held.add(lock_fd)
    sleeps: list[float] = []

    def sleep(dt: float) -> None:  # the other holder lets go after two polls
        sleeps.append(dt)
        if len(sleeps) == 2:
            locking.held.discard(lock_fd)

    s = shims.FcntlShim(locking, FakeLocking.NBLCK, FakeLocking.UNLCK, sleep=sleep, poll_interval=0.01)
    s.flock(lock_fd, shims.LOCK_EX)
    assert len(sleeps) == 2 and lock_fd in locking.held


def test_shared_lock_is_served_as_exclusive_and_counted(lock_fd):
    locking, sleeps = FakeLocking(), []
    s = _shim(locking, sleeps)
    s.flock(lock_fd, shims.LOCK_SH)
    assert s.shared_served_as_exclusive == 1
    assert lock_fd in locking.held


def test_lock_offset_leaves_file_position_alone(lock_fd):
    os.write(lock_fd, b"12345\n")
    os.lseek(lock_fd, 3, os.SEEK_SET)
    s = _shim(FakeLocking(), [])
    s.flock(lock_fd, shims.LOCK_EX)
    assert os.lseek(lock_fd, 0, os.SEEK_CUR) == 3


def test_fcntl_shim_module_shape():
    mod = _shim(FakeLocking(), []).as_module()
    assert mod.__name__ == "fcntl" and mod.__vllm_omni_windows_shim__
    assert (mod.LOCK_SH, mod.LOCK_EX, mod.LOCK_NB, mod.LOCK_UN) == (1, 2, 4, 8)
    assert callable(mod.flock) and callable(mod.lockf)


# ---- uvloop shim -----------------------------------------------------------


def test_uvloop_shim_runs_a_coroutine():
    mod = shims.make_uvloop_shim()
    assert mod.__vllm_omni_windows_shim__

    async def main():
        await asyncio.sleep(0)
        return 42

    assert mod.run(main()) == 42
    loop = mod.new_event_loop()
    try:
        assert isinstance(loop, asyncio.AbstractEventLoop)
    finally:
        loop.close()


def test_provide_leaves_a_real_module_alone():
    # ``os`` is always real; provide() must not touch it.
    assert shims.provide("os", lambda: None) == "real"
    assert sys.modules["os"] is os


def test_provide_installs_and_then_reports_already():
    name = "_vllm_omni_windows_test_missing_module_"
    sys.modules.pop(name, None)
    try:
        assert shims.provide(name, shims.make_uvloop_shim) == "provided"
        assert shims.provide(name, shims.make_uvloop_shim) == "already"
        assert sys.modules[name].__vllm_omni_windows_shim__
    finally:
        sys.modules.pop(name, None)


# ---- signal fallback -------------------------------------------------------


class FakeLoop:
    def __init__(self) -> None:
        self.scheduled = []
        self.__dict__["_x"] = None

    def call_soon_threadsafe(self, cb, *args):
        self.scheduled.append((cb, args))


def test_signal_fallback_registers_and_fires_via_call_soon_threadsafe(monkeypatch):
    import signal as _signal

    installed = {}

    def fake_signal(sig, handler):
        installed[sig] = handler

    monkeypatch.setattr(_signal, "signal", fake_signal)

    def original_add(self, sig, cb, *args):
        raise NotImplementedError

    def original_remove(self, sig):
        raise NotImplementedError

    add, remove = aio.make_signal_fallback(original_add, original_remove)
    loop = FakeLoop()
    hits = []
    add(loop, _signal.SIGINT, hits.append, "x")
    assert _signal.SIGINT in installed
    installed[_signal.SIGINT](_signal.SIGINT, None)  # the OS delivers the signal
    assert loop.scheduled == [(hits.append, ("x",))]
    assert remove(loop, _signal.SIGINT) is True
    assert remove(loop, _signal.SIGINT) is False  # nothing left, original says NotImplemented


def test_signal_fallback_prefers_the_native_implementation():
    calls = []

    def original_add(self, sig, cb, *args):
        calls.append(sig)

    add, _ = aio.make_signal_fallback(original_add, lambda self, sig: True)
    add(FakeLoop(), 2, lambda: None)
    assert calls == [2]


# ---- vLLM patch rules ------------------------------------------------------


def test_upstream_hashes_are_the_pinned_bodies_or_the_branch_marker():
    """On the installed vLLM each replaceable function is either the known upstream body or already patched."""
    vllm = pytest.importorskip("vllm")
    from vllm.utils import network_utils, system_utils

    for name, fn in (("get_open_zmq_ipc_path", network_utils.get_open_zmq_ipc_path),
                     ("find_loaded_library", system_utils.find_loaded_library)):
        marked = vllm_patches._has_mark(fn)
        known = vllm_patches._src_hash(fn) in vllm_patches.KNOWN_UPSTREAM[name]
        assert marked or known, f"{name} on vllm {vllm.__version__}: neither branch-patched nor a known upstream body"


def test_verify_reports_wheel_required_on_an_unpatched_source(monkeypatch):
    class Fake:
        pass

    import types

    m = types.ModuleType("fake_vllm_site")

    def wait_for_engine_startup():
        return "poller.register(sentinel)"

    m.wait_for_engine_startup = wait_for_engine_startup
    monkeypatch.setitem(sys.modules, "fake_vllm_site", m)
    import inspect

    monkeypatch.setattr(inspect, "getsource", lambda obj: "poller.register(sentinel)")
    assert vllm_patches._verify("fake_vllm_site", "wait_for_engine_startup", "_win_sentinels") == "wheel_required"
    assert vllm_patches._verify("fake_vllm_site", "wait_for_engine_startup", "sentinel") == "already"


# ---- paths -----------------------------------------------------------------


def test_shm_dir_exists_and_path_joins():
    d = shm_dir()
    assert os.path.isdir(d)
    assert shm_path("x.lock").startswith(d)
    if os.path.isdir("/dev/shm"):
        assert d == "/dev/shm"


# ---- activation contract ---------------------------------------------------


def test_activate_is_idempotent_and_a_noop_off_windows():
    first = activate()
    second = activate()
    assert first == second
    if not is_windows():
        assert first == {"platform": "n/a"}
        assert not is_active()
    else:
        assert is_active()
        assert set(first) >= {"fcntl", "uvloop", "event_loop_policy", "signal_handlers", "get_open_zmq_ipc_path"}


def test_report_schema():
    rep = report()
    for key in ("platform", "active", "activations", "steps", "asyncio", "zmq", "shm_dir", "vllm", "native_modules", "optional_deps"):
        assert key in rep, key
    assert rep["activations"] >= 1
    assert isinstance(rep["optional_deps"], dict) and "fcntl" in rep["optional_deps"]
    if rep["vllm"]:
        if is_windows():
            assert rep["vllm"]["wheel_sites"], "on Windows the wheel sites must be reported"
        else:
            assert rep["vllm"]["wheel_sites"] == {}
