#!/usr/bin/env python3
"""Diagnostic MiniCPM-o CPU run with a longer per-stage shutdown grace.

This changes only lifecycle timeouts. The model, stage graph and input are
forwarded unchanged to profile_minicpmo_text_speech.py.
"""

from __future__ import annotations

import argparse
import sys

import vllm_omni.engine.async_omni_engine as async_engine_module
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--shutdown-grace-s", type=float, default=30.0)
    args, remaining = parser.parse_known_args()
    if args.shutdown_grace_s <= 0:
        parser.error("--shutdown-grace-s must be positive")

    def shutdown_with_grace(self: StageEngineCoreClientBase, timeout: float | None = None) -> None:
        super(StageEngineCoreClientBase, self).shutdown(
            timeout=args.shutdown_grace_s if timeout is None else timeout
        )

    StageEngineCoreClientBase.shutdown = shutdown_with_grace
    async_engine_module.orchestrator_shutdown_join_timeout = lambda pools: max(
        30.0, args.shutdown_grace_s * (sum(pool.live_num_replicas for pool in (pools or [])) + 1)
    )
    sys.argv = ["profile_minicpmo_text_speech.py", *remaining]
    from benchmarks.edge_harness.profile_minicpmo_text_speech import main as profile_main

    profile_main()


if __name__ == "__main__":
    main()
