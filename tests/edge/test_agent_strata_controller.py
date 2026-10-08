# SPDX-License-Identifier: Apache-2.0
"""Strata controller load proof is separate from compute and default qualification."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from vllm_omni.edge.agent.controller import AgentController
from vllm_omni.edge.agent.memory import AesGcmCipher, EncryptedMemoryStore
from vllm_omni.edge.agent.omni_backend import BackendChunk
from vllm_omni.edge.agent.router import Admission, Qualification, Route
from vllm_omni.edge.agent.tools import WindowsToolBoundary
from vllm_omni.engine.backends.strata import (
    BACKEND_NAME,
    PINNED_STRATA_REVISION,
    PINNED_STRATA_VERSION,
    _verify_load_configuration,
)
from vllm_omni.engine.resource_ledger import ResourceLedger


def _plan():
    memory = {"gpu_name_sha256": "a" * 64}
    info = {
        "engine": PINNED_STRATA_VERSION,
        "context": 4096,
        "kv": "fp16",
        "spec": 2,
        "lookup": 0,
        "conversation_cache_mib": 0,
        "conversation_cache_slots": 0,
        "pool_workers": 2,
        "expert_slots": 3,
        "expert_cache_mib": 4,
        "arena_mib": 5,
    }
    report = _verify_load_configuration(
        {
            "gpu": {"local_index": 0, "name_sha256": "a" * 64},
            "cpu_pool": {"tasks_per_phase": 2, "participating_threads": 3},
            "expert_workers": {"workers": 2, "host_thread": True},
            "native_pack": True,
            "native_starts": 1,
        },
        info,
        memory,
        gpu=0,
        context=4096,
        kv="fp16",
        verify_window=2,
    )
    assert report["status"] == "verified"
    return {
        "backend": BACKEND_NAME,
        "runtime_revision": PINNED_STRATA_REVISION,
        "requested_device": "cpu+cuda:0",
        "observed_model_placement": None,
        "verified_execution_configuration": "cpu+cuda:0",
        "placement_evidence_level": "native_loaded_configuration",
        "execution_configuration_evidence": report,
        "fresh_memory_admission": memory,
        "context_tokens": 4096,
        "kv_type": "fp16",
        "native_verify_window": 2,
    }


class LeaseBackend:
    """Protocol fixture; no model or accelerator result is claimed."""

    def __init__(self, plan):
        self.execution_plan = plan
        self.ledger = ResourceLedger({"host_ram": 10, "vram": 10})
        self.unrelated = self.ledger.reserve("other-stage", {"host_ram": 1})
        self.lease = None
        self.generated = False
        self.drained = True

    def start(self):
        self.lease = self.ledger.reserve("strata-route", {"host_ram": 2, "vram": 2})

    async def generate(self, prompt, *, request_id, max_tokens, image_data_url=None):
        self.generated = True
        yield BackendChunk("ready", terminal=True, metrics={"fixture_metrics": True})

    async def cancel(self, request_id):
        pass

    def close(self):
        if self.lease is not None and self.ledger.owns(self.lease):
            return self.ledger.release(self.lease, drained=self.drained)
        return True


def _controller(tmp_path, backend, *, backend_name=BACKEND_NAME, qualified=False, bootstrap=True):
    route = Route(
        "strata-route",
        "artifact",
        "fixture-model",
        backend_name,
        frozenset({"text"}),
        "cpu+cuda:0",
        {"host_ram": 2, "vram": 2},
        True,
    )
    samples = {band: (1.0,) * 20 for band in ("short", "medium", "long")}
    qualifications = (
        [
            Qualification(
                route.route_id,
                route.artifact_id,
                "basic",
                "paired",
                "fixture",
                "AC",
                20,
                20,
                samples,
                samples,
                {band: 1 for band in samples},
                1800,
                True,
                True,
                True,
                True,
                True,
                True,
                raw_evidence="fixture-only.jsonl",
            )
        ]
        if qualified
        else []
    )
    controller = AgentController(
        routes=[route],
        qualifications=qualifications,
        backends={route.route_id: backend},
        memory=EncryptedMemoryStore(tmp_path / "memory.sqlite", AesGcmCipher(b"k" * 32)),
        tools=WindowsToolBoundary(browser=SimpleNamespace(close=lambda: None)),
        admit=lambda _: Admission(True, "fixture admission"),
        environment_fingerprint="fixture",
        power_condition="AC",
        qualification_suite_id="paired",
        bootstrap_route_id=route.route_id if bootstrap else None,
    )
    return controller


def test_strata_experiment_reports_load_proof_without_claiming_actual_compute(tmp_path):
    backend = LeaseBackend(_plan())
    controller = _controller(tmp_path, backend)
    events = []
    controller.add_listener(events.append)
    try:
        assert controller.submit("Say ready").result(timeout=20) == "ready"
        event = next(event["payload"] for event in events if event["kind"] == "route")
        assert event["experimental"] is True and event["qualified_p95_s"] is None
        assert event["actual_placement"] is None
        assert event["verified_execution_configuration"] == "cpu+cuda:0"
        assert event["execution_configuration_evidence"]["status"] == "verified"
        assert any(event["kind"] == "model_metrics" for event in events)
    finally:
        controller.close()
    assert backend.ledger.was_released(backend.lease)
    assert backend.ledger.owns(backend.unrelated)


@pytest.mark.parametrize("drained", [True, False])
def test_bad_strata_proof_releases_exact_lease_or_preserves_quarantine(tmp_path, drained):
    plan = _plan()
    plan["execution_configuration_evidence"]["cpu_expert_pool"].pop("workers")
    backend = LeaseBackend(plan)
    backend.drained = drained
    controller = _controller(tmp_path, backend)
    events = []
    controller.add_listener(events.append)
    try:
        with pytest.raises(RuntimeError, match="execution configuration|could not be released"):
            controller.submit("Say ready").result(timeout=20)
        assert not backend.generated
        assert not any(event["kind"] == "route" for event in events)
        assert backend.ledger.owns(backend.unrelated)
        if drained:
            assert backend.ledger.was_released(backend.lease)
        else:
            assert backend.ledger.owns(backend.lease)
            assert "strata-route" in backend.ledger.snapshot()["quarantined"]
    finally:
        backend.drained = True
        controller.close()


@pytest.mark.parametrize("qualified", [True, False])
def test_strata_load_proof_does_not_weaken_generic_placement_gate(tmp_path, qualified):
    backend = LeaseBackend(_plan())
    controller = _controller(tmp_path, backend, backend_name="external.llamacpp.text.v1", qualified=qualified)
    try:
        with pytest.raises(RuntimeError, match="loaded route placement"):
            controller.submit("Say ready").result(timeout=20)
        assert not backend.generated
    finally:
        controller.close()


def test_independently_qualified_strata_route_keeps_compute_claim_separate(tmp_path):
    backend = LeaseBackend(_plan())
    controller = _controller(tmp_path, backend, qualified=True, bootstrap=False)
    events = []
    controller.add_listener(events.append)
    try:
        assert controller.submit("Say ready").result(timeout=20) == "ready"
        route = next(event["payload"] for event in events if event["kind"] == "route")
        assert route["experimental"] is False and route["qualified_p95_s"] == 1.0
        assert route["actual_placement"] is None
        assert route["verified_execution_configuration"] == "cpu+cuda:0"
        assert route["execution_configuration_evidence"]["status"] == "verified"
    finally:
        controller.close()
    assert backend.ledger.was_released(backend.lease)


def test_strata_configuration_does_not_enable_bootstrap_implicitly(tmp_path):
    backend = LeaseBackend(_plan())
    controller = _controller(tmp_path, backend, bootstrap=False)
    try:
        assert controller.submit("Say ready").result(timeout=20) is None
        assert backend.lease is None and not backend.generated
    finally:
        controller.close()


def test_cancellation_waits_for_failed_load_lease_cleanup_before_releasing_turn(tmp_path):
    plan = _plan()
    plan["execution_configuration_evidence"]["cpu_expert_pool"].pop("workers")
    backend = LeaseBackend(plan)
    closing, finish_close = threading.Event(), threading.Event()
    original_close = backend.close

    def close():
        closing.set()
        assert finish_close.wait(10)
        return original_close()

    backend.close = close
    controller = _controller(tmp_path, backend)
    try:
        controller.submit("Say ready")
        assert closing.wait(10)
        controller.cancel()
        assert not controller._turn_done.wait(0.1)
        assert backend.ledger.owns(backend.lease)
        finish_close.set()
        assert controller._turn_done.wait(10)
        assert backend.ledger.was_released(backend.lease)
        assert backend.ledger.owns(backend.unrelated)
        assert not backend.generated
    finally:
        finish_close.set()
        controller.close()
