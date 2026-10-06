# SPDX-License-Identifier: Apache-2.0
"""A route cannot claim admission while omitting its physical memory pool."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_omni.edge.agent.omni_backend import OmniLlamaConfig
from vllm_omni.edge.agent.router import Route


def _route(demands: dict[str, int]) -> Route:
    return Route(
        "cpu", "artifact", "model", "external.llamacpp.text.v1",
        frozenset({"text"}), "cpu", demands,
    )


def _stage(placement: str, demands: dict[str, int], **changes) -> OmniLlamaConfig:
    values = dict(
        route_id="route", model_file="model.gguf", model_sha256="a" * 64,
        server_bin="llama-server", server_sha256="b" * 64,
        log_file="load.log", placement=placement,
        capacities={"host_ram": 100, "vram": 100}, demands=demands,
        memory_overhead_bytes=1,
    )
    values.update(changes)
    return OmniLlamaConfig(**values)


@pytest.mark.parametrize("demands", [{"vram": 10}, {"host_ram": 0}])
def test_complete_agent_route_requires_positive_host_claim(demands: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive host RAM"):
        _route(demands)


@pytest.mark.parametrize("demands", [{"vram": 10}, {"host_ram": 0}])
def test_cpu_stage_requires_positive_host_claim(demands: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive host RAM"):
        _stage("cpu", demands)


def test_cpu_stage_cannot_charge_unrelated_gpu_pool() -> None:
    with pytest.raises(ValueError, match="CPU-only stage"):
        _stage("cpu", {"host_ram": 10, "vram": 10})


@pytest.mark.parametrize("demands", [{"host_ram": 10}, {"host_ram": 10, "vram": 0}])
def test_discrete_gpu_stage_requires_positive_vram_claim(demands: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive physical GPU memory pool"):
        _stage("Vulkan0", demands)


def test_shared_memory_integrated_gpu_keeps_one_positive_host_claim() -> None:
    stage = _stage("Vulkan0", {"host_ram": 10}, gpu_memory_pool="host_ram")
    assert stage.demands == {"host_ram": 10}
    with pytest.raises(ValueError, match="positive host RAM"):
        replace(stage, demands={"host_ram": 0})
