"""Bounded stage shutdown grace shared by client and orchestrator budgets."""

from __future__ import annotations

from typing import Any

DEFAULT_STAGE_SHUTDOWN_GRACE_S = 15.0
CPU_STAGE_SHUTDOWN_GRACE_S = 30.0


def stage_shutdown_grace_s(vllm_config: Any) -> float:
    device_config = getattr(vllm_config, "device_config", None)
    if getattr(device_config, "device_type", None) == "cpu":
        return CPU_STAGE_SHUTDOWN_GRACE_S
    return DEFAULT_STAGE_SHUTDOWN_GRACE_S
