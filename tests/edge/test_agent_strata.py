# SPDX-License-Identifier: Apache-2.0
"""Agent uses the complete Strata engine stage and keeps qualification strict."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm_omni.edge.agent.omni_backend import OmniStrataBackend, OmniStrataConfig
from vllm_omni.engine.resource_ledger import ResourceLedger


def config():
    return OmniStrataConfig(
        route_id="q4-ssd",
        backend_config={"name": "external.strata.text.v1"},
        placement="cpu+cuda:0",
        capacities={"host_ram": 4096, "vram": 2048},
        demands={"host_ram": 2048, "vram": 1024},
    )


def test_strata_adapter_preserves_identity_and_uses_one_engine_lease():
    cfg = config()
    backend = OmniStrataBackend(cfg)
    ledger = ResourceLedger(cfg.capacities)
    claim = ledger.reserve(cfg.route_id, cfg.demands)
    backend.bind_resource_lease(ledger, claim)
    assert backend._shared_ledger is ledger
    assert backend._shared_reservation is claim
    assert ledger.snapshot()["reserved"] == dict(cfg.demands)
    stage = backend._stage_backend_config()
    assert stage["name"] == "external.strata.text.v1"
    assert stage["gpu_pool"] == "vram"
    assert stage["context_tokens"] == 4096
    assert stage["max_new_tokens"] == 128


def test_unverified_strata_placement_cannot_enter_agent_route():
    backend = OmniStrataBackend(config())
    with pytest.raises(RuntimeError, match="verify model load placement"):
        backend._validate_loaded_plan(
            {
                "requested_device": "cpu+cuda:0",
                "observed_model_placement": None,
                "placement_evidence_level": "unverified",
            }
        )
    backend._validate_loaded_plan({"requested_device": "cpu+cuda:0", "observed_model_placement": "cpu+cuda:0"})


@pytest.mark.parametrize(
    "change",
    [
        {"placement": "Vulkan0"},
        {"placement": "cpu+cuda:1"},
        {"backend_config": {"name": "external.strata.text.v1", "gpu_index": 1}},
        {"mmproj_file": "unqualified-image.gguf"},
        {"backend_config": {"name": "external.llamacpp.text.v1"}},
        {"backend_config": {"name": "external.strata.text.v1", "context_tokens": 8192}},
        {"demands": {"host_ram": 9000, "vram": 1024}},
    ],
)
def test_strata_adapter_refuses_inconsistent_plan(change):
    with pytest.raises(ValueError):
        replace(config(), **change)


def test_shared_lease_release_proof_is_scoped_to_route():
    cfg = config()
    backend = OmniStrataBackend(cfg)
    ledger = ResourceLedger(cfg.capacities)
    claim = ledger.reserve(cfg.route_id, cfg.demands)
    unrelated = ledger.reserve("other-stage", {"host_ram": 100})
    backend.bind_resource_lease(ledger, claim)
    process = SimpleNamespace(pid=42, poll=lambda: 0)
    backend._pool = SimpleNamespace(stage_client=SimpleNamespace(_proc=process))
    backend._runtime = SimpleNamespace(
        resource_ledger=ledger,
        _resource_reservations={(0, 0): claim},
        shutdown=lambda: ledger.release(claim, drained=True),
    )
    backend._last_turn_request_id = "turn"
    assert backend.close()
    assert ledger.owns(unrelated)
    assert backend.release_evidence["shared_ledger"] is True
    assert backend.release_evidence["resource_owner"] == cfg.route_id
    assert backend.release_evidence["resource_claim_released"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_field", ["output", "event"])
async def test_complete_model_adapter_refuses_another_requests_terminal(stale_field):
    backend = OmniStrataBackend(config())
    released = []
    output = SimpleNamespace(
        request_id="stale" if stale_field == "output" else "current",
        error=None,
        outputs=[SimpleNamespace(text="")],
        custom_output={
            "stage_event": {"terminal": True, "request_id": "stale" if stale_field == "event" else "current"}
        },
        release_stage_buffers=lambda: released.append(True),
    )
    pool = SimpleNamespace(
        submit_initial=AsyncMock(),
        stage_client=SimpleNamespace(receive_agent_delta=AsyncMock(return_value=None)),
        poll_graph_output=lambda _: output,
        abort_requests=AsyncMock(),
    )
    backend._pool = pool
    backend.close = lambda: True
    with pytest.raises(RuntimeError, match="another request"):
        async for _ in backend.generate("hello", request_id="current", max_tokens=8):
            pass
    pool.abort_requests.assert_awaited_once_with(["current"])
    assert released == [True]
    assert backend._active is None
