# SPDX-License-Identifier: Apache-2.0
"""Joint lifetime accounting; numbers are synthetic units, never allowances."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from vllm_omni.engine.local_plan import CompanionResourceSpec, LocalPlanManager
from vllm_omni.engine.resource_ledger import Reservation, ResourceLedger, ResourceUnavailable


def _route(name="model", demands=None):
    return SimpleNamespace(
        route_id=name, placement="cpu", backend="external.llamacpp.text.v1",
        memory_demands=dict(demands or {"host_ram": 60, "windows_commit": 80}),
    )


class Host:
    def __init__(self, **values):
        self.free = values or {"host_ram": 100, "windows_commit": 200}
        self.calls = []

    def sample(self):
        self.calls.append(dict(self.free))
        return dict(self.free)

    def change(self, demands, sign):
        for pool, value in demands.items():
            self.free[pool] += sign * value


class Stage:
    def __init__(self, host, route):
        self.host, self.route = host, route
        self.actual = dict(route.memory_demands)
        self.resident = False
        self.execution_plan = None
        self.loads = self.drains = 0
        self.drain_ok = True
        self.fail_load = False
        self.fail_bind = False
        self.bad_plan = False
        self.on_start = None
        self.token = self.ledger = None
        self.release_evidence = None
        self.generations = 0

    def bind_resource_lease(self, ledger, token):
        assert ledger.owns(token)
        assert dict(token.demands) == self.route.memory_demands
        self.token, self.ledger = token, ledger
        if self.fail_bind:
            raise RuntimeError("model bind failed")

    def start(self):
        self.loads += 1
        assert self.ledger.owns(self.token)
        assert len(self.ledger.snapshot()["owners"]) == 2
        if self.on_start is not None:
            self.on_start()
        self.resident = True
        self.host.change(self.actual, -1)
        if self.fail_load:
            raise RuntimeError("model startup failed after allocation")
        self.execution_plan = {
            "requested_device": "cpu", "observed_model_placement": "cpu",
            "reserved_bytes": dict(self.route.memory_demands),
        }
        if self.bad_plan:
            self.execution_plan["reserved_bytes"]["host_ram"] += 1

    def close(self):
        self.drains += 1
        if not self.drain_ok:
            return False
        if self.resident:
            self.host.change(self.actual, 1)
        self.resident = False
        self.execution_plan = None
        if self.token is not None:
            self.ledger.release(self.token, drained=True)
        return True

    def request_state_released(self, request_id):
        owner = self.token.owner
        if not self.close():
            return False
        self.release_evidence = {
            "schema": "omni-resource-release-v2", "request_id": request_id,
            "worker_exit_confirmed": True, "stage_ledger_empty": True,
            "resource_claim_released": True, "resource_owner": owner,
        }
        return True

    async def generate(self, prompt, **kwargs):
        self.generations += 1
        yield prompt


class Companion:
    def __init__(self, host, demands=None):
        self.host = host
        self.resource_spec = CompanionResourceSpec(
            "unit-tool", dict(demands or {"host_ram": 20, "windows_commit": 30}),
            "a" * 64, "unit-fixture:synthetic-accounting-envelope",
        )
        self.actual = dict(self.resource_spec.memory_demands)
        self.resident = False
        self.binds = self.loads = self.drains = self.resets = 0
        self.ledger = self.token = self.guard = self.generation = None
        self.release_evidence = None
        self.drain_ok = True
        self.ownership_verified = True
        self.fail_bind = False
        self.fail_launch = False
        self.fail_reset = False
        self.proof_override = {}
        self.bound_tokens, self.bound_guards = [], []

    def is_cold(self):
        return not self.resident

    def bind_resource_lease(self, ledger, token, *, joint_generation, guard):
        self.binds += 1
        assert self.is_cold()
        assert ledger.owns(token) and len(ledger.snapshot()["owners"]) == 2
        assert dict(token.demands) == dict(self.resource_spec.memory_demands)
        if self.fail_bind:
            raise RuntimeError("companion bind failed before launch")
        self.ledger, self.token, self.generation, self.guard = ledger, token, joint_generation, guard
        self.bound_tokens.append(token)
        self.bound_guards.append(guard)

    def launch(self):
        self.guard()
        assert self.is_cold()
        self.loads += 1
        self.host.change(self.actual, -1)
        self.resident = True
        if self.fail_launch:
            self.ledger.release(self.token, drained=False)
            raise RuntimeError("companion startup failed with an unretired owned resource")
        self.guard()

    def close(self):
        self.drains += 1
        if not self.drain_ok:
            return False
        if self.resident:
            self.host.change(self.actual, 1)
        self.resident = False
        self.release_evidence = {
            "schema": "omni-companion-release-v1", "joint_generation": self.generation,
            "resource_owner": self.token.owner, "owned_work_drained": True,
            "required_ownership_verified": self.ownership_verified,
            "scope": "unit_owned_contract_not_all_os_descendants",
            **self.proof_override,
        }
        return True

    def reset_after_verified_drain(self):
        self.resets += 1
        if self.fail_reset:
            raise RuntimeError("companion reset failed")
        assert self.is_cold()
        self.ledger = self.token = self.guard = self.generation = None


def _manager(*, host=None, companion=None, routes=None, capacities=None, ledger=None):
    host = host or Host()
    routes = routes or [_route()]
    companion = companion or Companion(host)
    stages = {route.route_id: Stage(host, route) for route in routes}
    manager = LocalPlanManager(
        routes=routes, backends=stages, capacities=capacities or dict(host.free),
        free_bytes=host.sample, resource_ledger=ledger, companion=companion,
    )
    return manager, stages, companion, host, routes


def test_one_prelaunch_snapshot_atomic_tokens_and_model_only_claim():
    manager, stages, companion, host, routes = _manager()
    stage = stages["model"]
    startup = dict(host.free)

    def observe_start():
        assert len(host.calls) == 1
        snapshot = manager.snapshot()
        assert snapshot["joint_admission"]["startup_free"] == startup
        assert snapshot["joint_admission"]["combined_demands"] == {"host_ram": 80, "windows_commit": 110}
        assert snapshot["resident_free_floor"] == {"host_ram": 20, "windows_commit": 90}
        assert companion.loads == 0
        assert manager._ledger.owns(companion.token)
        with pytest.raises(ResourceUnavailable, match="stale, quarantined, or needs full drain"):
            companion.guard()  # tools cannot launch during model loading

    stage.on_start = observe_start
    manager.start("model")
    assert len(host.calls) == 2  # one startup + independent post-load observation
    assert dict(stage.token.demands) == routes[0].memory_demands
    assert stage.execution_plan["reserved_bytes"] == routes[0].memory_demands
    assert companion.loads == 0
    companion.launch()
    assert host.free == {"host_ram": 20, "windows_commit": 90}
    assert manager.admit(routes[0]).admitted
    manager.start("model")
    assert stage.loads == 1 and companion.binds == 1
    assert manager.finalize_close()["ledger"]["owners"] == []


@pytest.mark.parametrize("pool", ["host_ram", "windows_commit"])
def test_combined_live_refusal_starts_neither_owner(pool):
    host = Host()
    manager, stages, companion, _, _ = _manager(host=host)
    host.free[pool] = 79 if pool == "host_ram" else 109
    with pytest.raises(ResourceUnavailable, match=pool):
        manager.start("model")
    assert stages["model"].loads == companion.binds == companion.loads == 0
    assert manager.snapshot()["ledger"]["owners"] == []


def test_combined_fixed_ceiling_refusal_is_atomic_despite_more_live_free():
    manager, stages, companion, _, _ = _manager(capacities={"host_ram": 70, "windows_commit": 200})
    with pytest.raises(ResourceUnavailable, match="80 requested > 70"):
        manager.start("model")
    assert stages["model"].loads == companion.binds == 0
    assert manager.snapshot()["ledger"]["reserved"] == {"host_ram": 0, "windows_commit": 0}


@pytest.mark.parametrize("value", [None, True, -1])
def test_missing_or_invalid_commit_measurement_refuses_before_reservation(value):
    manager, stages, companion, host, _ = _manager()
    host.free["windows_commit"] = value
    with pytest.raises(ResourceUnavailable, match="measurement unavailable"):
        manager.start("model")
    assert stages["model"].loads == companion.binds == 0
    assert manager.snapshot()["ledger"]["owners"] == []


@pytest.mark.parametrize("changes", [
    {"memory_demands": {"host_ram": True}}, {"memory_demands": {"host_ram": -1}},
    {"memory_demands": {"host_ram": 0}}, {"memory_demands": {}},
    {"envelope_sha256": ""}, {"envelope_sha256": "A" * 64}, {"evidence_reference": ""},
])
def test_companion_requires_frozen_explicit_evidence_bound_allowance(changes):
    values = dict(purpose_id="fixture", memory_demands={"host_ram": 1},
                  envelope_sha256="a" * 64, evidence_reference="unit:evidence")
    values.update(changes)
    with pytest.raises(ValueError):
        CompanionResourceSpec(**values)


def test_spec_copies_demands_and_replacement_refuses_without_lowering_floor():
    source = {"host_ram": 20, "windows_commit": 30}
    host = Host()
    companion = Companion(host, source)
    source["host_ram"] = 1
    assert companion.resource_spec.memory_demands["host_ram"] == 20
    with pytest.raises(TypeError):
        companion.resource_spec.memory_demands["host_ram"] = 1
    manager, stages, _, _, routes = _manager(host=host, companion=companion)
    manager.start("model")
    floor = manager.snapshot()["resident_free_floor"]
    companion.resource_spec = CompanionResourceSpec("other", {"host_ram": 1}, "b" * 64, "unit:other")
    assert not manager.admit(routes[0]).admitted
    with pytest.raises(ResourceUnavailable, match="identity or memory declaration changed"):
        manager.start("model")
    assert manager.snapshot()["resident_free_floor"] == floor and stages["model"].loads == 1
    # Restore the pinned declaration only to perform owned fixture cleanup.
    companion.resource_spec = manager._companion_spec
    manager.finalize_close()


def test_no_late_allowance_for_already_resident_or_unleased_tools():
    manager, stages, companion, _, routes = _manager()
    stages["model"].resident = True
    assert not manager.admit(routes[0]).admitted
    with pytest.raises(ResourceUnavailable, match="both be cold"):
        manager.start("model")
    assert companion.binds == 0 and not manager.snapshot()["ledger"]["owners"]
    stages["model"].resident = False
    companion.resident = True
    with pytest.raises(ResourceUnavailable, match="both be cold"):
        manager.start("model")
    with pytest.raises(ResourceUnavailable, match="unleased companion is not cold"):
        manager.finalize_close()


def test_final_check_refuses_an_unleased_resident_model():
    manager, stages, _, _, _ = _manager()
    stages["model"].resident = True
    with pytest.raises(ResourceUnavailable, match="still has a resident backend"):
        manager.finalize_close()
    stages["model"].resident = False
    manager.finalize_close()


def test_foreign_claim_is_not_reclaimed_by_joint_start_or_final_close():
    ledger = ResourceLedger({"host_ram": 100, "windows_commit": 200})
    foreign = ledger.reserve("foreign", {"host_ram": 1})
    manager, stages, companion, _, routes = _manager(ledger=ledger)
    assert not manager.admit(routes[0]).admitted
    with pytest.raises(ResourceUnavailable, match="reconciled controller ledger"):
        manager.start("model")
    with pytest.raises(ResourceUnavailable, match="not empty"):
        manager.finalize_close()
    assert ledger.owns(foreign) and stages["model"].loads == companion.loads == 0
    assert ledger.release(foreign, drained=True)


def test_lazy_growth_no_double_charge_and_floor_refusal_requires_full_drain():
    manager, stages, companion, host, routes = _manager()
    stages["model"].actual = {"host_ram": 50, "windows_commit": 70}
    companion.actual = {"host_ram": 10, "windows_commit": 20}
    manager.start("model")
    companion.launch()
    assert host.free == {"host_ram": 40, "windows_commit": 110}
    host.free.update(host_ram=20, windows_commit=90)
    assert manager.admit(routes[0]).admitted
    companion.guard()
    before = copy.deepcopy(manager.snapshot())
    host.free["host_ram"] = 19
    assert not manager.admit(routes[0]).admitted
    assert manager.snapshot()["resident_free_floor"] == before["resident_free_floor"]
    assert manager.snapshot()["ledger"] == before["ledger"]
    host.free["host_ram"] = 20
    with pytest.raises(ResourceUnavailable, match="needs full drain"):
        companion.guard()  # no reset of a refused joint generation
    manager.close_joint()
    # Synthetic external usage is restored before a new cold generation.
    host.free.update(host_ram=100, windows_commit=200)
    manager.start("model")
    assert manager.snapshot()["joint_admission"]["generation"] != before["joint_admission"]["generation"]
    manager.finalize_close()


@pytest.mark.asyncio
async def test_each_model_step_checks_the_same_joint_floor_before_generation():
    manager, stages, companion, host, _ = _manager()
    wrapped = manager.wrappers()["model"]
    wrapped.start()
    companion.launch()
    assert [chunk async for chunk in wrapped.generate("first", request_id="request-step-0", max_tokens=1)] == ["first"]
    host.free["host_ram"] -= 1
    with pytest.raises(ResourceUnavailable, match="reservation floor"):
        _ = [chunk async for chunk in wrapped.generate("second", request_id="request-step-1", max_tokens=1)]
    assert stages["model"].generations == 1
    manager.finalize_close()


def test_model_cancel_releases_only_exact_model_token_then_cold_recovery():
    manager, stages, companion, _, _ = _manager()
    wrapped = manager.wrappers()["model"]
    wrapped.start()
    companion.launch()
    model_token, tool_token, old_guard = stages["model"].token, companion.token, companion.guard
    old_generation = manager.snapshot()["joint_admission"]["generation"]
    assert wrapped.request_state_released("request-cancel")
    proof = wrapped.release_evidence
    assert proof["exact_resource_token_released"] is True
    assert proof["release_scope"] == "exact_model_lease"
    assert proof["host_ledger_empty"] is False and proof["tool_lease_retained"] is True
    assert proof["joint_generation"] == old_generation
    from vllm_omni.edge.agent.controller import _model_release_verified

    assert _model_release_verified(proof, request_id="request-cancel", route_id="model")
    assert manager._ledger.was_released(model_token) and manager._ledger.owns(tool_token)
    with pytest.raises(ResourceUnavailable, match="needs full drain"):
        old_guard()
    assert manager.admit(manager._routes["model"]).admitted  # provisional only
    wrapped.start()  # drains/recreates old tools before fresh atomic admission
    assert companion.drains == companion.resets == 1
    assert companion.loads == 1 and stages["model"].loads == 2
    assert companion.token is not tool_token
    assert not manager._ledger.release(model_token, drained=True)
    assert not manager._ledger.release(tool_token, drained=True)
    with pytest.raises(ResourceUnavailable, match="stale"):
        old_guard()
    manager.finalize_close()


def test_equal_token_cannot_release_an_exact_current_claim():
    manager, stages, companion, _, _ = _manager()
    manager.start("model")
    token = companion.token
    equal = Reservation(token.owner, dict(token.demands))
    assert equal == token and equal is not token
    assert not manager._ledger.release(equal, drained=True)
    assert manager._ledger.owns(token) and manager._ledger.owns(stages["model"].token)
    manager.finalize_close()


@pytest.mark.parametrize("kind", ["bind", "load", "plan"])
def test_model_start_failure_independently_drains_both_owned_tokens(kind):
    manager, stages, companion, host, _ = _manager()
    setattr(stages["model"], {"bind": "fail_bind", "load": "fail_load", "plan": "bad_plan"}[kind], True)
    with pytest.raises(RuntimeError):
        manager.start("model")
    assert manager.snapshot()["ledger"]["owners"] == []
    assert companion.drains == companion.resets == 1
    assert host.free == {"host_ram": 100, "windows_commit": 200}


def test_companion_failed_bind_while_independently_cold_retires_unstarted_claim():
    manager, stages, companion, _, _ = _manager()
    companion.fail_bind = True
    with pytest.raises(RuntimeError, match="bind failed before launch"):
        manager.start("model")
    assert stages["model"].loads == companion.loads == companion.drains == 0
    assert not manager.snapshot()["ledger"]["owners"]
    assert companion.resets == 1


@pytest.mark.parametrize("override", [
    {"required_ownership_verified": False}, {"owned_work_drained": False},
    {"resource_owner": "other"}, {"joint_generation": "old"}, {"schema": "wrong"}, {"scope": ""},
])
def test_tool_scope_failure_retains_only_tool_quarantine_and_blocks_recovery(override):
    manager, stages, companion, _, _ = _manager()
    manager.start("model")
    companion.launch()
    tool_token = companion.token
    companion.proof_override = override
    with pytest.raises(ResourceUnavailable, match="scoped release evidence"):
        manager.close_joint()
    assert manager.snapshot()["resident_route"] is None
    assert manager.snapshot()["ledger"]["owners"] == [tool_token.owner]
    assert manager.snapshot()["ledger"]["quarantined"] == [tool_token.owner]
    with pytest.raises(ResourceUnavailable):
        manager.start("model")
    assert stages["model"].loads == 1
    companion.proof_override = {}
    assert manager.finalize_close()["ledger"]["owners"] == []


def test_both_failed_drains_are_quarantined_and_primary_error_is_preserved():
    manager, stages, companion, _, _ = _manager()
    stages["model"].fail_load = True
    stages["model"].drain_ok = False
    companion.drain_ok = False
    with pytest.raises(RuntimeError, match="model startup failed after allocation") as failed:
        manager.start("model")
    snapshot = manager.snapshot()
    assert len(snapshot["ledger"]["owners"]) == len(snapshot["ledger"]["quarantined"]) == 2
    assert any("joint cleanup failed" in note for note in failed.value.__notes__)
    stages["model"].drain_ok = companion.drain_ok = True
    manager.finalize_close()


def test_model_failure_does_not_prevent_independent_tool_token_retirement():
    manager, stages, companion, _, _ = _manager()
    manager.start("model")
    stages["model"].drain_ok = False
    with pytest.raises(ResourceUnavailable, match="worker drain unverified"):
        manager.close_joint()
    assert companion.drains == 1
    assert manager.snapshot()["ledger"]["owners"] == ["model"]
    assert manager.snapshot()["ledger"]["quarantined"] == ["model"]
    stages["model"].drain_ok = True
    manager.finalize_close()


def test_partial_tool_startup_quarantines_its_exact_token_and_blocks_further_work():
    manager, stages, companion, _, routes = _manager()
    manager.start("model")
    companion.fail_launch = True
    companion.drain_ok = False
    with pytest.raises(RuntimeError, match="unretired owned resource"):
        companion.launch()
    assert manager.snapshot()["ledger"]["quarantined"] == [companion.token.owner]
    assert not manager.admit(routes[0]).admitted
    with pytest.raises(ResourceUnavailable):
        companion.guard()
    with pytest.raises(ResourceUnavailable):
        manager.close_joint()
    assert manager.snapshot()["ledger"]["owners"] == [companion.token.owner]
    assert stages["model"].loads == 1
    companion.drain_ok = True
    manager.finalize_close()


def test_reset_failure_cannot_reuse_old_joint_generation_or_closed_owner():
    manager, stages, companion, _, _ = _manager()
    manager.start("model")
    old_guard = companion.guard
    companion.fail_reset = True
    with pytest.raises(RuntimeError, match="reset failed"):
        manager.close_joint()
    assert not manager.snapshot()["ledger"]["owners"]
    assert manager.snapshot()["joint_admission"]["needs_drain"]
    with pytest.raises(RuntimeError, match="reset failed"):
        manager.start("model")
    assert stages["model"].loads == 1
    with pytest.raises(ResourceUnavailable):
        old_guard()
    companion.fail_reset = False
    manager.start("model")
    assert stages["model"].loads == 2
    manager.finalize_close()


def test_route_switch_preview_never_evicts_and_start_fully_drains_tools():
    routes = [_route("first"), _route("second", {"host_ram": 65, "windows_commit": 90})]
    manager, stages, companion, _, _ = _manager(routes=routes)
    manager.start("first")
    companion.launch()
    first_token = companion.token
    assert manager.admit(routes[1]).admitted
    assert stages["first"].drains == companion.drains == 0
    manager.start("second")
    assert stages["first"].drains == companion.drains == companion.resets == 1
    assert companion.token is not first_token and stages["second"].loads == 1
    manager.close("second")  # model scoped, not a global-close claim
    assert manager.snapshot()["ledger"]["owners"] == [companion.token.owner]
    manager.finalize_close()


def test_no_companion_retains_original_single_model_budget_and_floor():
    host = Host()
    route = _route()
    stage = Stage(host, route)
    # Stage's joint assertion is replaced by an explicit model-only fixture.
    def start():
        stage.loads += 1
        assert len(stage.ledger.snapshot()["owners"]) == 1
        stage.resident = True
        host.change(stage.actual, -1)
        stage.execution_plan = {"requested_device": "cpu", "observed_model_placement": "cpu",
                                "reserved_bytes": route.memory_demands}

    stage.start = start
    manager = LocalPlanManager(routes=[route], backends={"model": stage},
                               capacities=dict(host.free), free_bytes=host.sample)
    manager.start("model")
    assert manager.snapshot()["resident_free_floor"] == {"host_ram": 40, "windows_commit": 120}
    assert "joint_admission" not in manager.snapshot()
    assert manager.admit(route).admitted
    manager.start("model")
    assert stage.loads == 1
    assert manager.finalize_close()["ledger"]["owners"] == []


def test_finalized_controller_cannot_start_a_new_owner_or_raise_ceilings():
    manager, stages, companion, _, routes = _manager()
    before = manager.finalize_close()
    assert not manager.admit(routes[0]).admitted
    with pytest.raises(ResourceUnavailable, match="finalized"):
        manager.start("model")
    assert stages["model"].loads == companion.binds == 0
    assert manager.finalize_close() == before
