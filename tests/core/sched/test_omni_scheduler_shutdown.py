# SPDX-License-Identifier: Apache-2.0
"""The async-chunk transport must retire before the vLLM scheduler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_chunk_adapter_closes_before_base_scheduler() -> None:
    calls: list[str] = []

    class BaseScheduler:
        def shutdown(self) -> None:
            calls.append("base")

    class Scheduler(OmniSchedulerMixin, BaseScheduler):
        chunk_transfer_adapter = SimpleNamespace(shutdown=lambda: calls.append("adapter"))

    Scheduler().shutdown()
    assert calls == ["adapter", "base"]


def test_scheduler_without_chunk_adapter_still_shuts_down() -> None:
    calls: list[str] = []

    class BaseScheduler:
        def shutdown(self) -> None:
            calls.append("base")

    class Scheduler(OmniSchedulerMixin, BaseScheduler):
        pass

    Scheduler().shutdown()
    assert calls == ["base"]


def test_base_scheduler_still_shuts_down_when_adapter_raises() -> None:
    calls: list[str] = []

    class BaseScheduler:
        def shutdown(self) -> None:
            calls.append("base")

    def fail_adapter() -> None:
        calls.append("adapter")
        raise RuntimeError("connector close failed")

    class Scheduler(OmniSchedulerMixin, BaseScheduler):
        chunk_transfer_adapter = SimpleNamespace(shutdown=fail_adapter)

    with pytest.raises(RuntimeError, match="connector close failed"):
        Scheduler().shutdown()
    assert calls == ["adapter", "base"]
