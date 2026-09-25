#!/usr/bin/env python3
"""Run the MiniCPM-o profiler with a diagnostic vLLM shutdown grace period.

Only the head-side MPClient shutdown wait changes. This intentionally leaves
the model, transport and stage implementation untouched for comparison.
Remaining arguments are passed to profile_minicpmo_text_speech.py.
"""

from __future__ import annotations

import argparse
import sys

from vllm.v1.engine.core_client import AsyncMPClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--shutdown-grace-s", type=float, default=30.0)
    args, remaining = parser.parse_known_args()
    if args.shutdown_grace_s <= 0:
        parser.error("--shutdown-grace-s must be positive")

    original_shutdown = AsyncMPClient.shutdown

    def shutdown_with_grace(self: AsyncMPClient, timeout: float | None = None) -> None:
        original_shutdown(self, timeout=args.shutdown_grace_s if timeout is None else timeout)

    AsyncMPClient.shutdown = shutdown_with_grace
    sys.argv = ["profile_minicpmo_text_speech.py", *remaining]
    from benchmarks.edge_harness.profile_minicpmo_text_speech import main as profile_main

    profile_main()


if __name__ == "__main__":
    main()
