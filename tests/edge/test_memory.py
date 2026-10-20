# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Returning freed weight-loading memory to the OS."""

import pytest

from vllm_omni.edge.memory import (
    anon_rss_bytes,
    release_after_load,
    release_free_memory,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_anon_rss_excludes_mapped_files():
    """RssAnon, not VmRSS: VmRSS counts the mmap'd checkpoint's pages and moved
    1.2-2.0 GB between identical runs on the host this was developed on."""
    anon = anon_rss_bytes()
    assert anon is not None and anon > 0
    with open("/proc/self/status") as f:
        status = f.read()
    total = int([l for l in status.splitlines() if l.startswith("VmRSS:")][0].split()[1])
    assert anon <= total * 1024


def test_release_reports_what_it_called_and_is_repeatable():
    first = release_free_memory()
    second = release_free_memory()
    assert isinstance(first, list)
    # Idempotent: calling it twice must not raise, and glibc is always there.
    assert first == second
    assert any("malloc_trim" in name for name in first)


def test_release_actually_returns_a_large_freed_block():
    """The real case is gigabytes of freed repack buffers; a smaller version of
    the same thing should still come back."""
    import ctypes

    before = anon_rss_bytes()
    blocks = [bytearray(64 * 1024 * 1024) for _ in range(4)]  # 256 MiB, touched
    for b in blocks:
        b[::4096] = b"\x01" * (len(b) // 4096)
    peak = anon_rss_bytes()
    assert peak - before > 200 * 2**20, "test did not actually allocate"
    del blocks
    release_free_memory()
    after = anon_rss_bytes()
    # Most of it should be back; allocators keep some, so this is deliberately
    # loose -- the point is that the call does something, not how much.
    assert after < peak - 100 * 2**20, (before, peak, after)


def test_release_after_load_is_safe_when_nothing_is_held():
    release_after_load("unit test")  # must not raise


def test_patch_installs_the_hook_on_cpu():
    from vllm.platforms import current_platform

    if not current_platform.is_cpu():
        pytest.skip("CPU-only patch")
    import vllm_omni.patch  # noqa: F401
    from vllm.v1.worker.cpu_worker import CPUWorker

    assert getattr(CPUWorker, "_omni_release_memory", False)
