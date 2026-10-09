# SPDX-License-Identifier: Apache-2.0
"""Controller scope/finalization contracts, with no model/tool subprocesses."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.edge.agent.controller import AgentController, _model_release_verified


def _proof():
    return {
        "schema": "omni-resource-release-v2", "request_id": "request",
        "worker_exit_confirmed": True, "stage_ledger_empty": True,
        "host_claim_released": True, "host_ledger_empty": False,
        "resource_claim_released": True, "exact_resource_token_released": True,
        "resource_owner": "model", "release_scope": "exact_model_lease",
        "joint_generation": "unit-joint-generation", "tool_lease_retained": True,
    }


def test_scoped_model_release_keeps_global_nonempty_truthful():
    proof = _proof()
    assert _model_release_verified(proof, request_id="request", route_id="model")
    assert proof["host_ledger_empty"] is False and proof["tool_lease_retained"] is True


@pytest.mark.parametrize("change", [
    {"request_id": "other"}, {"worker_exit_confirmed": False}, {"stage_ledger_empty": False},
    {"host_claim_released": False}, {"host_ledger_empty": None}, {"schema": "legacy"},
    {"resource_claim_released": False}, {"exact_resource_token_released": False},
    {"resource_owner": "other"}, {"release_scope": "all_resources"},
    {"tool_lease_retained": False}, {"joint_generation": ""}, {"joint_generation": None},
])
def test_scoped_release_rejects_missing_or_mismatched_exact_model_evidence(change):
    assert not _model_release_verified(_proof() | change, request_id="request", route_id="model")


def test_legacy_global_empty_proof_retains_its_existing_contract():
    proof = {key: value for key, value in _proof().items()
             if key in {"request_id", "worker_exit_confirmed", "stage_ledger_empty", "host_claim_released"}}
    assert not _model_release_verified(proof, request_id="request", route_id="model")
    proof["host_ledger_empty"] = True
    assert _model_release_verified(proof, request_id="request", route_id="model")


def _controller(order, finalizer=None, *, tools_fail=False):
    def close_tools():
        order.append("tools")
        if tools_fail:
            raise RuntimeError("unit tools remain unverified")

    return AgentController(
        routes=[], qualifications=[],
        backends={"unit": SimpleNamespace(close=lambda: order.append("model"))},
        memory=SimpleNamespace(close=lambda: order.append("memory")),
        tools=SimpleNamespace(close=close_tools), admit=lambda route: None,
        environment_fingerprint="unit", power_condition="unit", qualification_suite_id="unit",
        resource_finalize=finalizer,
    )


def _empty():
    return {"ledger": {"owners": [], "quarantined": [], "reserved": {"host_ram": 0, "windows_commit": 0}}}


def test_final_resource_callback_runs_after_scoped_owners_and_memory_close():
    order = []

    def finalize():
        assert order == ["model", "tools", "memory"]
        order.append("resources")
        return _empty()

    controller = _controller(order, finalize)
    controller.close()
    assert order == ["model", "tools", "memory", "resources"]
    assert controller.resource_close_evidence == _empty()


@pytest.mark.parametrize("ledger", [
    {"owners": ["tool"], "quarantined": [], "reserved": {"host_ram": 1}},
    {"owners": [], "quarantined": ["tool"], "reserved": {"host_ram": 0}},
    {"owners": [], "quarantined": [], "reserved": {"host_ram": 1}},
    {"owners": [], "quarantined": [], "reserved": {"host_ram": False}},
    {"owners": [], "quarantined": [], "reserved": {}},
    {"owners": [], "quarantined": [], "reserved": None},
])
def test_final_close_refuses_nonempty_or_unverifiable_resource_evidence(ledger):
    controller = _controller([], lambda: {"ledger": ledger})
    with pytest.raises(RuntimeError, match="did not verify an empty shared ledger"):
        controller.close()
    assert controller.resource_close_evidence is None


def test_final_resource_check_still_runs_when_tools_fail_and_preserves_both_failures():
    order = []

    def finalize():
        order.append("resources")
        raise RuntimeError("unit tool token remains quarantined")

    controller = _controller(order, finalize, tools_fail=True)
    with pytest.raises(RuntimeError) as failed:
        controller.close()
    assert order == ["model", "tools", "memory", "resources"]
    assert "tools remain unverified" in str(failed.value)
    assert "token remains quarantined" in str(failed.value)


def test_no_resource_callback_preserves_existing_controller_close():
    order = []
    controller = _controller(order)
    controller.close()
    assert order == ["model", "tools", "memory"]
    assert controller.resource_close_evidence is None


def test_close_retry_clears_an_earlier_successful_resource_receipt():
    calls = []

    def finalize():
        calls.append(True)
        if len(calls) == 1:
            return _empty()
        raise RuntimeError("unit retry found an unresolved owner")

    controller = _controller([], finalize)
    controller.close()
    assert controller.resource_close_evidence == _empty()
    with pytest.raises(RuntimeError, match="retry found an unresolved owner"):
        controller.close()
    assert controller.resource_close_evidence is None
