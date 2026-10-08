# SPDX-License-Identifier: Apache-2.0
"""An engine route and its StageRuntime use one exact resource reservation."""

import gc
import weakref
from types import SimpleNamespace

import pytest

from vllm_omni.engine.local_plan import LocalPlanManager
from vllm_omni.engine.resource_ledger import Reservation, ResourceLedger, ResourceUnavailable
from vllm_omni.engine.stage_runtime import StageRuntime


def _native(tmp_path, *, demand=30, capacity=100, stage_id=0):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "model.safetensors").write_bytes(b"1234567890")
    replica = SimpleNamespace(
        metadata=SimpleNamespace(stage_id=stage_id, stage_type="llm", runtime_cfg={"devices": "cpu"}),
        replica_id=0, launch_mode="local",
        stage_cfg=SimpleNamespace(engine_args={"resource_budget": {
            "capacities": {"host_ram": capacity}, "demands": {"host_ram": demand},
        }}),
        stage_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model=str(model), enforce_eager=True),
            cache_config=SimpleNamespace(kv_cache_memory_bytes=20),
        ),
    )
    return SimpleNamespace(replicas=[replica])


def _runtime(ledger, tokens=None):
    return StageRuntime([], "model", "", stage_init_timeout=1, async_chunk=False,
                        resource_ledger=ledger, resource_reservations=tokens)


def test_runtime_adopts_manager_token_without_double_reservation(tmp_path, monkeypatch):
    import psutil

    ledger = ResourceLedger({"host_ram": 100})
    token = ledger.reserve("route", {"host_ram": 30})
    # A fixed controller ceiling is not compared to a new available reading;
    # otherwise already allocated resident models would be charged twice.
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=5))
    runtime = _runtime(ledger, {(0, 0): token})
    runtime._reserve_stage_resources([_native(tmp_path)])
    assert runtime.resource_ledger is ledger
    assert runtime._resource_reservations[(0, 0)] is token
    assert ledger.snapshot()["reserved"] == {"host_ram": 30}
    runtime._release_unstarted_resources({})
    assert ledger.was_released(token)


def test_shared_runtime_claim_owners_do_not_collide(tmp_path):
    ledger = ResourceLedger({"host_ram": 100})
    first, second = _runtime(ledger), _runtime(ledger)
    first._reserve_stage_resources([_native(tmp_path)])
    second._reserve_stage_resources([_native(tmp_path)])
    assert len(ledger.snapshot()["owners"]) == 2
    assert ledger.snapshot()["reserved"]["host_ram"] == 60
    assert first._resource_reservations[(0, 0)] is not second._resource_reservations[(0, 0)]


@pytest.mark.parametrize("fault", ["equal_token", "wrong_demand", "wrong_stage", "wrong_capacity", "quarantined"])
def test_runtime_refuses_invalid_inherited_lease_without_mutating_shared_claim(tmp_path, fault):
    ledger = ResourceLedger({"host_ram": 100})
    token = ledger.reserve("route", {"host_ram": 30})
    tokens = {(0, 0): token}
    stage = _native(tmp_path)
    if fault == "equal_token":
        tokens[(0, 0)] = Reservation(token.owner, token.demands)
    elif fault == "wrong_demand":
        stage = _native(tmp_path, demand=31)
    elif fault == "wrong_stage":
        tokens = {(1, 0): token}
    elif fault == "wrong_capacity":
        stage = _native(tmp_path, capacity=101)
    elif fault == "quarantined":
        ledger.release(token, drained=False)
    with pytest.raises(ValueError, match="lease|ledger"):
        _runtime(ledger, tokens)._reserve_stage_resources([stage])
    assert ledger.owns(token)
    assert ledger.snapshot()["reserved"]["host_ram"] == 30


def test_release_receipt_cannot_release_a_successor_claim():
    ledger = ResourceLedger({"host_ram": 100})
    old = ledger.reserve("route", {"host_ram": 30})
    assert ledger.release(old, drained=True)
    assert ledger.was_released(old)
    newer = ledger.reserve("route", {"host_ram": 30})
    assert not ledger.was_released(old)
    assert not ledger.release(old, drained=True)
    assert ledger.owns(newer)


def test_release_receipts_do_not_retain_completed_runtime_tokens():
    ledger = ResourceLedger({"host_ram": 100})
    token = ledger.reserve("runtime-unique-owner", {"host_ram": 30})
    observed = weakref.ref(token)
    assert ledger.release(token, drained=True)
    assert ledger.was_released(token)
    del token
    gc.collect()
    assert observed() is None
    assert not ledger._released


@pytest.mark.parametrize("factory_confirmed_drain", [False, True])
def test_graph_constructor_failure_cannot_release_an_unverified_worker_claim(monkeypatch, factory_confirmed_drain):
    import vllm_omni.engine.backends as backends

    ledger = ResourceLedger({"host_ram": 100})
    token = ledger.reserve("route", {"host_ram": 30})
    runtime = _runtime(ledger, {(0, 0): token})
    replica = SimpleNamespace(
        metadata=SimpleNamespace(stage_id=0, stage_type="graph", engine_input_source=[], final_output=True),
        replica_id=0, launch_mode="local",
        stage_cfg=SimpleNamespace(engine_args={"resource_budget": {
            "capacities": {"host_ram": 100}, "demands": {"host_ram": 30},
        }}),
    )
    runtime._reserve_stage_resources([SimpleNamespace(replicas=[replica])])

    def failed_constructor(metadata, config, shared_ledger, reservation):
        # This marker must be set before the factory could spawn a subprocess.
        assert (0, 0) in runtime._resource_started
        assert shared_ledger.owns(reservation)
        if factory_confirmed_drain:
            assert shared_ledger.release(reservation, drained=True)
        raise PermissionError("worker started; process-tree cleanup access denied")

    monkeypatch.setattr(backends, "create_graph_client", failed_constructor)
    with pytest.raises(PermissionError, match="cleanup access denied"):
        runtime._initialize_replica(replica, 1)
    runtime._release_unstarted_resources({})
    if factory_confirmed_drain:
        assert ledger.was_released(token)
        assert ledger.snapshot()["owners"] == []
    else:
        assert ledger.owns(token)
        assert ledger.snapshot()["quarantined"] == ["route"]


class _Stage:
    def __init__(self, route, *, replace_token=False):
        self.route = route
        self.execution_plan = None
        self.resident = False
        self.replace_token = replace_token
        self.free = 100

    def bind_resource_lease(self, ledger, token):
        assert ledger.owns(token)
        self.ledger, self.token = ledger, token

    def start(self):
        assert self.ledger.snapshot()["reserved"]["host_ram"] == 40
        self.resident = True
        self.free -= 30
        self.execution_plan = {
            "requested_device": "cpu", "observed_model_placement": "cpu",
            "reserved_bytes": {"host_ram": 30},
        }

    def close(self):
        if self.resident:
            self.free += 30
            self.resident = False
            self.ledger.release(self.token, drained=True)
            if self.replace_token:
                self.successor = self.ledger.reserve(self.token.owner, self.token.demands)
        return True


def _manager(*, replace_token=False):
    route = SimpleNamespace(route_id="route", placement="cpu", memory_demands={"host_ram": 30})
    stage = _Stage(route, replace_token=replace_token)
    ledger = ResourceLedger({"host_ram": 100})
    other = ledger.reserve("other", {"host_ram": 10})
    manager = LocalPlanManager(
        routes=[route], backends={"route": stage}, capacities={"host_ram": 100},
        free_bytes=lambda: {"host_ram": stage.free}, resource_ledger=ledger,
    )
    return manager, stage, ledger, other


def test_engine_manager_binds_single_shared_lease_and_accepts_stage_drain():
    manager, stage, ledger, other = _manager()
    wrapped = manager.wrappers()["route"]
    wrapped.start()
    assert manager._reservation is stage.token
    assert wrapped.close()
    assert ledger.was_released(stage.token)
    assert ledger.owns(other)
    assert ledger.snapshot()["reserved"] == {"host_ram": 10}


def test_engine_manager_refuses_to_reconcile_a_replaced_token():
    manager, stage, ledger, _ = _manager(replace_token=True)
    manager.wrappers()["route"].start()
    with pytest.raises(ResourceUnavailable, match="replaced"):
        manager.close("route")
    assert ledger.owns(stage.successor)


def test_scoped_cancel_proof_allows_unrelated_shared_owners():
    manager, stage, ledger, other = _manager()
    wrapper = manager.wrappers()["route"]
    wrapper.start()

    def release(request_id):
        stage.close()
        stage.release_evidence = {
            "schema": "omni-resource-release-v2", "request_id": request_id,
            "worker_exit_confirmed": True, "stage_ledger_empty": True,
            "resource_claim_released": True, "resource_owner": stage.token.owner,
            "shared_ledger": True,
        }
        return True

    stage.request_state_released = release
    assert wrapper.request_state_released("request")
    assert wrapper.release_evidence["resource_claim_released"] is True
    assert wrapper.release_evidence["host_ledger_empty"] is False
    assert ledger.owns(other)
