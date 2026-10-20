# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Non-blocking exclusive file locks, on POSIX and on Windows.

[edge-infer W1] ``vllm_omni.engine.stage_init_utils`` imports ``fcntl`` at module
scope for the device-initialization lock. ``fcntl`` does not exist on Windows, and
because that module is in the import closure of ``AsyncOmni`` the import fails
before anything can be configured. The locking *logic* is sound and already
torch-free; only the primitive has to change.

Two behavioural differences are deliberate and worth knowing:

* ``flock`` is advisory and whole-file; ``msvcrt.locking`` is mandatory and
  byte-range. We therefore lock **one byte at a fixed high offset**, far past the
  PID text the caller writes, so the lock never overlaps the data. Locking byte 0
  instead would make the caller's own ``ftruncate``/``write`` fail on Windows.
* ``msvcrt.locking`` raises ``OSError`` when the region is already held, where
  ``flock`` raises ``BlockingIOError``. This module normalizes to
  ``BlockingIOError`` so the caller's existing ``except BlockingIOError`` branch
  -- which is what triggers stale-lock cleanup -- keeps working untouched.

Both flavours release on process death, which is what the stale-lock path relies
on: POSIX ``flock`` by the kernel dropping the fd, Windows by the file handle
closing when the process exits.
"""

from __future__ import annotations

import os

_WINDOWS = os.name == "nt"

# Far beyond any PID line the caller writes, so the locked byte and the file
# contents never overlap.
_LOCK_OFFSET = 1 << 30

if _WINDOWS:
    import msvcrt

    def flock_exclusive_nb(fd: int) -> None:
        """Take an exclusive, non-blocking lock. Raises BlockingIOError if held."""
        here = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError(str(exc)) from exc
        finally:
            os.lseek(fd, here, os.SEEK_SET)

    def funlock(fd: int) -> None:
        here = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.lseek(fd, here, os.SEEK_SET)

else:
    import fcntl

    def flock_exclusive_nb(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def funlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


__all__ = ["flock_exclusive_nb", "funlock"]
