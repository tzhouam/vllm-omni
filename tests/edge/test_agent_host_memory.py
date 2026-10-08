# SPDX-License-Identifier: Apache-2.0
"""Host-wide resident-route accounting above individual Omni stage ledgers."""

from __future__ import annotations

import pytest

from vllm_omni.edge.agent.admission import HostMemoryCoordinator
from vllm_omni.edge.agent.router import Route
from vllm_omni.engine.resource_ledger import ResourceUnavailable


class _Host:
    def __init__(self, free: int) -> None:
        self.free = free

    def snapshot(self) -> dict[str, int]:
        return {"host_ram": self.free}


class _Stage:
    def __init__(self, route: Route, host: _Host, *, actual_bytes: int | None = None) -> None:
        self.route = route
        self.host = host
        self.actual_bytes = route.memory_demands["host_ram"] if actual_bytes is None else actual_bytes
        self.resident = False
        self.execution_plan = None
        self.loads = 0
        self.drains = 0
        self.drain_ok = True
        self.fail_load = False

    def start(self) -> None:
        self.loads += 1
        if self.fail_load:
            raise RuntimeError("stage load failed")
        assert not self.resident
        self.resident = True
        self.host.free -= self.actual_bytes
        self.execution_plan = {
            "requested_device": self.route.placement,
            "observed_model_placement": self.route.placement,
            "reserved_bytes": dict(self.route.memory_demands),
        }

    def close(self) -> bool:
        self.drains += 1
        if not self.drain_ok:
            return False
        if self.resident:
            self.host.free += self.actual_bytes
        self.resident = False
        self.execution_plan = None
        return True


def _route(name: str, demand: int) -> Route:
    return Route(
        name, name + "-artifact", name, "external.llamacpp.text.v1", frozenset({"text"}), "cpu", {"host_ram": demand}
    )


def _coordinator(host: _Host, *routes: Route, ceiling: int | None = None):
    stages = {route.route_id: _Stage(route, host) for route in routes}
    coordinator = HostMemoryCoordinator(
        routes=routes,
        backends=stages,
        capacities={"host_ram": ceiling or host.free},
        free_bytes=host.snapshot,
    )
    return coordinator, stages


def test_repeat_turn_reuses_resident_claim_without_double_charging_live_free() -> None:
    host = _Host(7)
    route = _route("model", 5)
    coordinator, stages = _coordinator(host, route)
    wrapped = coordinator.wrappers()[route.route_id]
    assert coordinator.admit(route).admitted
    wrapped.start()
    assert host.free == 2
    assert coordinator.admit(route).admitted
    wrapped.start()
    assert stages[route.route_id].loads == 1
    assert coordinator.snapshot()["ledger"]["reserved"] == {"host_ram": 5}
    assert wrapped.close()
    assert host.free == 7
    assert coordinator.snapshot()["ledger"]["owners"] == []


def test_resident_reuse_allows_lazy_growth_within_claim_and_refuses_overage() -> None:
    host = _Host(10)
    route = _route("model", 6)
    coordinator, stages = _coordinator(host, route)
    stages[route.route_id].actual_bytes = 4
    wrapped = coordinator.wrappers()[route.route_id]
    wrapped.start()
    assert coordinator.snapshot()["resident_free_floor"] == {"host_ram": 4}
    assert host.free == 6
    host.free -= 2  # admitted KV/workspace growth within the remaining claim
    assert coordinator.admit(route).admitted
    wrapped.start()
    assert stages[route.route_id].loads == 1
    host.free -= 1  # a different process now exceeds the full route claim
    decision = coordinator.admit(route)
    assert not decision.admitted
    assert "reservation floor" in decision.reason
    with pytest.raises(ResourceUnavailable, match="reservation floor"):
        wrapped.start()
    assert stages[route.route_id].loads == 1
    assert coordinator.snapshot()["ledger"]["owners"] == [route.route_id]
    wrapped.close()


def test_load_refuses_if_live_usage_exceeds_declared_claim() -> None:
    host = _Host(10)
    route = _route("model", 6)
    coordinator, stages = _coordinator(host, route)
    stages[route.route_id].actual_bytes = 7
    with pytest.raises(ResourceUnavailable, match="exceeded its declared"):
        coordinator.wrappers()[route.route_id].start()
    assert stages[route.route_id].drains == 1
    assert coordinator.snapshot()["ledger"]["owners"] == []


def test_cancel_release_drains_both_stage_and_host_claims() -> None:
    host = _Host(7)
    route = _route("model", 5)
    coordinator, stages = _coordinator(host, route)
    stage = stages[route.route_id]

    def release(request_id):
        stage.release_evidence = {
            "request_id": request_id,
            "release_mode": "worker_shutdown",
            "worker_pid_before": 1234,
            "worker_exit_code": 0,
            "worker_exit_confirmed": True,
            "stage_ledger_empty": True,
        }
        return bool(request_id) and stage.close()

    stage.request_state_released = release
    wrapped = coordinator.wrappers()[route.route_id]
    wrapped.start()
    assert wrapped.request_state_released("cancel-during-tool")
    assert wrapped.release_evidence["host_claim_released"] is True
    assert host.free == 7
    assert coordinator.snapshot()["ledger"]["owners"] == []


def test_unverified_cancel_quarantines_host_claim() -> None:
    host = _Host(8)
    route = _route("model", 4)
    coordinator, stages = _coordinator(host, route)
    stage = stages[route.route_id]
    stage.request_state_released = lambda request_id: False
    wrapped = coordinator.wrappers()[route.route_id]
    wrapped.start()
    assert not wrapped.request_state_released("cancel-during-tool")
    assert coordinator.snapshot()["ledger"]["quarantined"] == [route.route_id]


def test_cancel_without_worker_proof_quarantines_host_claim() -> None:
    host = _Host(8)
    route = _route("model", 4)
    coordinator, stages = _coordinator(host, route)
    stage = stages[route.route_id]
    stage.request_state_released = lambda request_id: True
    wrapped = coordinator.wrappers()[route.route_id]
    wrapped.start()
    assert not wrapped.request_state_released("cancel-during-tool")
    assert coordinator.snapshot()["ledger"]["quarantined"] == [route.route_id]


def test_switch_releases_old_worker_then_remeasures_and_claims_new_route() -> None:
    host = _Host(7)
    first, second = _route("first", 5), _route("second", 6)
    coordinator, stages = _coordinator(host, first, second)
    wrappers = coordinator.wrappers()
    wrappers[first.route_id].start()
    assert host.free == 2
    preview = coordinator.admit(second)
    assert preview.admitted and "provisional" in preview.reason
    assert stages[first.route_id].drains == 0  # ranking cannot evict a model
    wrappers[second.route_id].start()
    assert stages[first.route_id].drains == 1
    assert host.free == 1
    assert coordinator.snapshot()["ledger"]["owners"] == [second.route_id]
    wrappers[second.route_id].close()
    assert host.free == 7


def test_switch_refuses_if_reclaimed_memory_is_less_than_declared_claim() -> None:
    host = _Host(8)
    first, second = _route("first", 5), _route("second", 7)
    coordinator, stages = _coordinator(host, first, second)
    stages[first.route_id].actual_bytes = 3
    wrappers = coordinator.wrappers()
    wrappers[first.route_id].start()
    assert host.free == 5
    assert coordinator.admit(second).admitted  # potential reclaim only
    # A third-party process consumes two bytes before the actual transition.
    host.free -= 2
    with pytest.raises(ResourceUnavailable, match="live free 6"):
        wrappers[second.route_id].start()
    assert stages[second.route_id].loads == 0
    assert coordinator.snapshot()["ledger"]["owners"] == []


def test_unverified_drain_quarantines_outer_reservation_and_refuses_new_load() -> None:
    host = _Host(8)
    first, second = _route("first", 4), _route("second", 4)
    coordinator, stages = _coordinator(host, first, second)
    wrappers = coordinator.wrappers()
    wrappers[first.route_id].start()
    stages[first.route_id].drain_ok = False
    with pytest.raises(ResourceUnavailable, match="drain unverified"):
        wrappers[second.route_id].start()
    assert stages[second.route_id].loads == 0
    snapshot = coordinator.snapshot()["ledger"]
    assert snapshot["owners"] == [first.route_id]
    assert snapshot["quarantined"] == [first.route_id]
    assert not coordinator.admit(second).admitted
    stages[first.route_id].drain_ok = True
    assert coordinator.admit(second).admitted  # verified recovery releases claim
    wrappers[second.route_id].start()
    assert coordinator.snapshot()["ledger"]["owners"] == [second.route_id]


def test_failed_start_releases_claim_only_after_verified_cleanup() -> None:
    host = _Host(8)
    route = _route("failed", 4)
    coordinator, stages = _coordinator(host, route)
    stages[route.route_id].fail_load = True
    with pytest.raises(RuntimeError, match="stage load failed"):
        coordinator.wrappers()[route.route_id].start()
    assert coordinator.snapshot()["ledger"]["owners"] == []


def test_unknown_pool_reading_and_route_mutation_refuse_explicitly() -> None:
    host = _Host(8)
    route = _route("model", 4)
    coordinator, _ = _coordinator(host, route)
    assert not coordinator.admit(_route("model", 3)).admitted
    host.free = None  # type: ignore[assignment]
    decision = coordinator.admit(route)
    assert not decision.admitted
    assert "measurement unavailable" in decision.reason


def test_oversized_candidate_is_refused_without_disabling_smaller_route() -> None:
    host = _Host(8)
    small, large = _route("small", 4), _route("large", 12)
    stages = {small.route_id: _Stage(small, host)}
    coordinator = HostMemoryCoordinator(
        routes=[small, large],
        backends=stages,
        capacities={"host_ram": 8},
        free_bytes=host.snapshot,
        blocked_reasons={large.route_id: "host_ram: declared demand exceeds controller ceiling"},
    )
    assert not coordinator.admit(large).admitted
    assert "large" not in coordinator.wrappers()
    assert coordinator.admit(small).admitted
    coordinator.wrappers()[small.route_id].start()
    assert coordinator.snapshot()["ledger"]["owners"] == [small.route_id]
    coordinator.wrappers()[small.route_id].close()


def test_gpu_route_claims_host_ram_and_vram_and_switch_releases_both() -> None:
    free = {"host_ram": 12, "vram": 8}
    gpu = Route(
        "gpu",
        "gpu-artifact",
        "gpu-model",
        "external.llamacpp.text.v1",
        frozenset({"text"}),
        "Vulkan0",
        {"host_ram": 4, "vram": 6},
    )
    cpu = _route("cpu", 7)

    class DualStage:
        def __init__(self, route):
            self.route = route
            self.resident = False
            self.execution_plan = None

        def start(self):
            for pool, amount in self.route.memory_demands.items():
                free[pool] -= amount
            self.resident = True
            self.execution_plan = {
                "requested_device": self.route.placement,
                "observed_model_placement": self.route.placement,
                "reserved_bytes": dict(self.route.memory_demands),
            }

        def close(self):
            if self.resident:
                for pool, amount in self.route.memory_demands.items():
                    free[pool] += amount
            self.resident = False
            self.execution_plan = None
            return True

    coordinator = HostMemoryCoordinator(
        routes=[gpu, cpu],
        backends={"gpu": DualStage(gpu), "cpu": DualStage(cpu)},
        capacities=dict(free),
        free_bytes=lambda: dict(free),
    )
    wrappers = coordinator.wrappers()
    wrappers["gpu"].start()
    assert free == {"host_ram": 8, "vram": 2}
    assert coordinator.snapshot()["ledger"]["reserved"] == {"host_ram": 4, "vram": 6}
    wrappers["cpu"].start()
    assert free == {"host_ram": 5, "vram": 8}
    assert coordinator.snapshot()["ledger"]["reserved"] == {"host_ram": 7, "vram": 0}
    wrappers["cpu"].close()


def _strata_load_plan(route: Route) -> dict:
    from vllm_omni.engine.backends.strata import BACKEND_NAME, PINNED_STRATA_REVISION, PINNED_STRATA_VERSION

    return {
        "backend": BACKEND_NAME,
        "runtime_revision": PINNED_STRATA_REVISION,
        "requested_device": route.placement,
        "observed_model_placement": None,
        "observed_compute_units": None,
        "verified_execution_configuration": route.placement,
        "placement_evidence_level": "native_loaded_configuration",
        "context_tokens": 4096,
        "kv_type": "fp16",
        "native_verify_window": 2,
        "fresh_memory_admission": {"gpu_name_sha256": "a" * 64},
        "reserved_bytes": dict(route.memory_demands),
        "execution_configuration_evidence": {
            "status": "verified",
            "scope": "loaded_backend_execution_configuration_not_per_request_compute",
            "source_revision": PINNED_STRATA_REVISION,
            "physical_gpu_index": 0,
            "cuda_device": {"local_index": 0, "name_sha256": "a" * 64, "compute_capability": "12.0"},
            "cpu_expert_pool": {"tasks_per_phase": 27, "participating_threads": 9, "workers": 8, "host_thread": True},
            "native_pack": True,
            "native_starts": 1,
            "engine_info": {
                "engine": PINNED_STRATA_VERSION,
                "context": 4096,
                "kv": "fp16",
                "spec": 2,
                "lookup": 0,
                "conversation_cache_mib": 0,
                "conversation_cache_slots": 0,
                "pool_workers": 8,
                "expert_slots": 1,
                "expert_cache_mib": 1,
                "arena_mib": 1,
            },
        },
    }


def _strata_coordinator(*, backend_name="external.strata.text.v1", proof_change=None):
    route = Route(
        "strata", "artifact", "model", backend_name, frozenset({"text"}), "cpu+cuda:0", {"host_ram": 4, "vram": 2}
    )
    host = _Host(10)

    class Stage(_Stage):
        def start(self):
            super().start()
            self.execution_plan = _strata_load_plan(route)
            if proof_change:
                proof_change(self.execution_plan)

    stage = Stage(route, host)
    manager = HostMemoryCoordinator(
        routes=[route],
        backends={route.route_id: stage},
        capacities={"host_ram": 10, "vram": 10},
        free_bytes=lambda: host.snapshot() | {"vram": 10},
    )
    return manager, stage, host, route


def test_explicit_strata_load_uses_strict_configuration_proof_and_keeps_actual_unknown():
    manager, stage, host, route = _strata_coordinator()
    manager.start(route.route_id)
    admission = manager.admit(route)
    assert admission.admitted and admission.actual_placement is None
    assert stage.execution_plan["observed_compute_units"] is None
    assert stage.execution_plan["observed_model_placement"] is None
    assert manager.snapshot()["ledger"]["reserved"] == dict(route.memory_demands)
    manager.close(route.route_id)
    assert host.free == 10 and not manager.snapshot()["ledger"]["owners"]


@pytest.mark.parametrize(
    "proof_change",
    [
        lambda plan: plan["execution_configuration_evidence"].update(status="unverified"),
        lambda plan: plan["fresh_memory_admission"].update(gpu_name_sha256="b" * 64),
        lambda plan: plan.pop("execution_configuration_evidence"),
        lambda plan: plan.update(backend="external.llamacpp.text.v1", observed_model_placement="cpu+cuda:0"),
    ],
)
def test_strata_load_proof_failure_drains_before_releasing_claim(proof_change):
    manager, stage, host, route = _strata_coordinator(proof_change=proof_change)
    with pytest.raises(RuntimeError, match="loaded backend execution configuration"):
        manager.start(route.route_id)
    assert stage.drains == 1
    assert host.free == 10 and not manager.snapshot()["ledger"]["owners"]


def test_other_backends_cannot_bypass_actual_placement_with_strata_proof():
    manager, stage, host, route = _strata_coordinator(backend_name="external.llamacpp.text.v1")
    with pytest.raises(RuntimeError, match="no verified model placement"):
        manager.start(route.route_id)
    assert stage.drains == 1
    assert host.free == 10 and not manager.snapshot()["ledger"]["owners"]
