# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Give freed weight-loading memory back to the OS.

Loading a quantized checkpoint churns far more memory than the weights
occupy. The GPTQ path is the extreme case: `_process_gptq_weights_w4a8`
unpacks 4-bit codes to **int32 -- four bytes per weight** -- permutes, repacks,
and drops the intermediates. For a 1.7 B model that is gigabytes of
short-lived allocation. It is correctly freed, but a caching allocator keeps
the arenas, so it stays resident for the life of the process.

Measured on Spark-X2.5-1.7B (Xeon 8480C, anonymous RSS, which repeats to
within 0.5 MB where VmRSS swings 1-2 GB):

    path                after load   after release   freed
    W4A8 (GPTQ)           7338 MB        3510 MB     3828 MB
    W4A16 (tinygemm)      5023 MB        3411 MB     1612 MB

Two things worth reading off that table. The release is worth **52%** of
resident memory on the path we actually want to deploy; and afterwards both
paths land in the same place, so W4A8's apparent 2.3 GB penalty over W4A16 was
never a real footprint difference -- it was the repack, held by the allocator.

This is only useful on CPU. On GPU the weights are in device memory and the
host-side churn is not what constrains anything.
"""

import ctypes

from vllm.logger import init_logger

logger = init_logger(__name__)

# glibc's trim, then tcmalloc's equivalent: vLLM's CPU docs tell you to
# LD_PRELOAD tcmalloc, so on a correctly configured host the second one is the
# one that does the work.
_RELEASERS = (
    ("libc.so.6", "malloc_trim", (0,)),
    ("libtcmalloc_minimal.so.4", "MallocExtension_ReleaseFreeMemory", ()),
    ("libtcmalloc.so.4", "MallocExtension_ReleaseFreeMemory", ()),
)


def anon_rss_bytes() -> int | None:
    """Resident anonymous memory: allocations, excluding mapped files.

    The right metric for a memory budget. ``VmRSS`` also counts the mmap'd
    checkpoint's pages, which come and go with page-cache pressure and moved
    1-2 GB between identical runs on this host.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def release_free_memory() -> list[str]:
    """Ask every allocator we can reach to return free arenas to the OS.

    Returns the names of the calls that succeeded. Safe to call repeatedly and
    safe when none are available -- it simply does nothing.
    """
    released: list[str] = []
    for lib, symbol, args in _RELEASERS:
        try:
            fn = getattr(ctypes.CDLL(lib), symbol)
        except (OSError, AttributeError):
            continue
        try:
            fn(*args)
            released.append(f"{lib}:{symbol}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("%s:%s failed: %s", lib, symbol, exc)
    return released


def release_after_load(context: str = "weight load") -> None:
    """Release, and say how much came back, at INFO."""
    before = anon_rss_bytes()
    released = release_free_memory()
    if not released:
        return
    after = anon_rss_bytes()
    if before is None or after is None:
        logger.info("Released allocator arenas after %s via %s", context, released)
        return
    logger.info(
        "Released %.0f MiB after %s (anonymous RSS %.0f -> %.0f MiB) via %s",
        (before - after) / 2**20,
        context,
        before / 2**20,
        after / 2**20,
        released,
    )
