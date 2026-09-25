# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StageEngineCoreClient.check_health()."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_client(*, engine_dead=False):
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 0
    client.resources = SimpleNamespace(engine_dead=engine_dead)
    return client


def test_check_health_passes_when_alive():
    client = _make_client(engine_dead=False)
    client.check_health()  # no exception


def test_check_health_raises_when_resources_engine_dead():
    client = _make_client(engine_dead=True)
    with pytest.raises(EngineDeadError, match="engine core is dead"):
        client.check_health()


@pytest.mark.parametrize(
    ("device_type", "requested_timeout", "expected_timeout"),
    [("cpu", None, 30.0), ("cuda", None, 15.0), ("cpu", 7.0, 7.0)],
)
def test_shutdown_grace_follows_device_and_preserves_explicit_timeout(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    requested_timeout: float | None,
    expected_timeout: float,
) -> None:
    client = _make_client()
    client.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device_type=device_type))
    observed: list[float | None] = []
    monkeypatch.setattr(AsyncMPClient, "shutdown", lambda self, timeout=None: observed.append(timeout))

    client.shutdown(timeout=requested_timeout)

    assert observed == [expected_timeout]
