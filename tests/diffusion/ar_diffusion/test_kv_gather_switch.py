# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Precedence of the contiguous-K/V gather switch: environment > deployment config > pipeline KV spec."""

from __future__ import annotations

import pytest

from vllm_omni.experimental.ar_diffusion.kv_cache import config as kv_config
from vllm_omni.experimental.ar_diffusion.runner import _prefer_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _restore_switch():
    yield
    kv_config.set_contiguous_kv_gather(False)


@pytest.mark.parametrize(
    ("configured", "pipeline_default", "expected"),
    [(None, True, True), (None, False, False), (False, True, False), (True, False, True)],
)
def test_deployment_value_wins_over_pipeline_default(configured, pipeline_default, expected):
    assert _prefer_config(configured, pipeline_default) is expected


def test_default_is_off_until_a_runner_resolves_it(monkeypatch):
    monkeypatch.delenv(kv_config.KV_GATHER_ENV, raising=False)
    assert kv_config.ARDiffusionKVConfig().contiguous_kv_gather is None
    assert kv_config.ARDiffusionKVConfig().reuse_history_staging is None
    assert not kv_config.contiguous_kv_gather_enabled()
    kv_config.set_contiguous_kv_gather(True)
    assert kv_config.contiguous_kv_gather_enabled()


@pytest.mark.parametrize(("forced", "resolved", "expected"), [("0", True, False), ("1", False, True)])
def test_environment_forces_either_way(monkeypatch, forced, resolved, expected):
    monkeypatch.setenv(kv_config.KV_GATHER_ENV, forced)
    kv_config.set_contiguous_kv_gather(resolved)
    assert kv_config.contiguous_kv_gather_enabled() is expected
