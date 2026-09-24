# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Placing a stage on an external device, and refusing to.

The gate tested here is the one that keeps a CPU fallback from being reported
as NPU execution. It cannot be replaced by a cheaper check: a VitisAI session
lists the EP in ``get_providers()`` whether or not it claimed a node, and a
graph it declined runs on the CPU and returns bit-identical numbers. So these
tests assert on node assignments, never on output values.
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm_omni.edge.local.capabilities import FORMAT_ONNX_A16W8, FORMAT_ONNX_FP16
from vllm_omni.edge.local.external.stage import (
    ExternalStage,
    PlacementRefused,
    plan_external_stage,
    verify_placement,
)
from vllm_omni.edge.local.manifest import build_graph_artifact
from vllm_omni.edge.local.plan import (
    REFUSE_CAPACITY,
    REFUSE_EP_PLACEMENT,
    REFUSE_NO_ARTIFACT,
    REFUSE_NO_DEVICE,
    REFUSE_ROUTE,
)


def _artifact(path, fmt=FORMAT_ONNX_A16W8, component="stage"):
    return build_graph_artifact(
        path, fmt=fmt, opset=21, source_model="test", component=component,
        exporter="tests.edge.local.external",
    )


def test_a16w8_graph_is_placed_on_the_npu(a16w8_graph, npu_device, igpu_device):
    plan = plan_external_stage(_artifact(a16w8_graph), [igpu_device, npu_device])
    assert plan.admitted
    assert plan.device is not None and plan.device.device_id == "npu:amd"
    # The iGPU was considered and rejected on format; the record says so rather
    # than only naming the winner.
    assert [r.code for r in plan.refusals] == [REFUSE_NO_ARTIFACT]


def test_fp16_graph_is_refused_by_the_npu_not_quietly_downgraded(a16w8_graph, npu_device):
    """A16W8 is the only door on XDNA2, and the failure mode for anything else
    is silent: no error, every node on the CPU. So it is refused at plan time."""
    plan = plan_external_stage(_artifact(a16w8_graph, fmt=FORMAT_ONNX_FP16), [npu_device])
    assert not plan.admitted
    assert plan.refusals[0].code == REFUSE_NO_ARTIFACT
    assert "a16w8" in plan.refusals[0].remedy.lower()


def test_require_refuses_rather_than_substituting(a16w8_graph, npu_device, igpu_device):
    """"Does the NPU run this" must not be answered with the iGPU."""
    plan = plan_external_stage(
        _artifact(a16w8_graph, fmt=FORMAT_ONNX_FP16), [npu_device, igpu_device],
        require="npu:amd",
    )
    assert not plan.admitted
    assert {r.device_id for r in plan.refusals} == {"npu:amd"}


def test_prefer_falls_through(a16w8_graph, npu_device, igpu_device):
    plan = plan_external_stage(
        _artifact(a16w8_graph, fmt=FORMAT_ONNX_FP16), [npu_device, igpu_device],
        prefer="npu:amd",
    )
    assert plan.admitted and plan.device.device_id == "igpu:amd"


def test_unreachable_device_refuses_with_the_route_reason(a16w8_graph, unreachable_npu):
    plan = plan_external_stage(_artifact(a16w8_graph), [unreachable_npu])
    assert not plan.admitted
    assert plan.refusals[0].code == REFUSE_ROUTE
    assert "MCDM" in plan.refusals[0].message
    assert "VLLM_OMNI_EXTERNAL_PYTHON_ORT_VITISAI" in plan.refusals[0].remedy


def test_no_external_device_at_all(a16w8_graph):
    plan = plan_external_stage(_artifact(a16w8_graph), [])
    assert not plan.admitted and plan.refusals[0].code == REFUSE_NO_DEVICE


def test_required_device_absent_from_enumeration(a16w8_graph, igpu_device):
    plan = plan_external_stage(_artifact(a16w8_graph), [igpu_device], require="npu:amd")
    assert not plan.admitted and plan.refusals[0].code == REFUSE_NO_DEVICE


def test_budget_is_on_the_shared_pool(a16w8_graph, npu_device):
    plan = plan_external_stage(_artifact(a16w8_graph), [npu_device])
    assert {r.pool for r in plan.reservations} == {"host_ram"}
    assert plan.budget_bytes > 0


def test_open_verifies_placement_and_closes_the_ledger(a16w8_graph, npu_device, fake_route, tmp_path):
    fake_route()
    plan = plan_external_stage(_artifact(a16w8_graph), [npu_device])
    with ExternalStage(plan) as stage:
        report = stage.open({"x": np.ones(4, np.float32)}, profile_dir=tmp_path)
        assert report.fraction_on_target == 1.0
    runtime = next(r for r in plan.reservations if r.purpose == "worker runtime")
    assert runtime.actual_upper_bound_bytes == 122 * 2**20 - plan.artifact.bytes
    assert runtime.bytes >= runtime.actual_upper_bound_bytes, "budget must bound the measurement"


def test_load_peak_over_budget_refuses_before_any_measured_run(
    a16w8_graph, npu_device, fake_route, tmp_path
):
    fake_route(stats={"peak_rss_bytes": 4 * 2**30})
    plan = plan_external_stage(_artifact(a16w8_graph), [npu_device], require="npu:amd")
    with pytest.raises(PlacementRefused) as error:
        ExternalStage(plan).open({"x": np.ones(4, np.float32)}, profile_dir=tmp_path)
    assert error.value.refusal.code == REFUSE_CAPACITY
    assert plan.observed_worker_peak_rss_bytes == 4 * 2**30
    assert plan.budget_bytes < plan.observed_worker_peak_rss_bytes


def test_measured_peak_hint_is_reserved_and_checked_before_load(
    a16w8_graph, npu_device, fake_route, tmp_path
):
    peak = 4 * 2**30
    artifact = _artifact(a16w8_graph)
    refused = plan_external_stage(
        artifact, [npu_device], require="npu:amd",
        worker_peak_rss_hint_bytes=peak, pool_capacity_bytes=peak,
    )
    assert not refused.admitted and refused.refusals[0].code == REFUSE_CAPACITY

    fake_route(stats={"peak_rss_bytes": peak})
    plan = plan_external_stage(
        artifact, [npu_device], require="npu:amd",
        worker_peak_rss_hint_bytes=peak,
    )
    assert plan.admitted and plan.budget_bytes > peak
    with ExternalStage(plan) as stage:
        stage.open({"x": np.ones(4, np.float32)}, profile_dir=tmp_path)
        outputs, _ = stage.run({"x": np.ones(4, np.float32)})
        assert "x_out" in outputs
    assert plan.observed_worker_peak_rss_bytes == peak


def test_unreported_load_peak_is_refused(a16w8_graph, npu_device, fake_route, tmp_path):
    fake_route(stats={"peak_rss_bytes": 0})
    plan = plan_external_stage(_artifact(a16w8_graph), [npu_device], require="npu:amd")
    with pytest.raises(PlacementRefused, match="not reported") as error:
        ExternalStage(plan).open({"x": np.ones(4, np.float32)}, profile_dir=tmp_path)
    assert error.value.refusal.code == REFUSE_CAPACITY


def test_peak_hint_requires_a_specific_device(a16w8_graph, npu_device):
    with pytest.raises(ValueError, match="device-specific"):
        plan_external_stage(
            _artifact(a16w8_graph), [npu_device], worker_peak_rss_hint_bytes=2**30
        )


def test_declined_graph_is_refused_even_though_it_ran(a16w8_graph, npu_device, fake_route, tmp_path):
    """The whole point: it produced correct output, on the CPU."""
    fake_route(load={"node_counts": {"CPUExecutionProvider": 9, "vitisai": 1},
                     "total_nodes": 10, "target_nodes": 1, "fraction_on_target": 0.1})
    plan = plan_external_stage(_artifact(a16w8_graph), [npu_device])
    stage = ExternalStage(plan)
    with pytest.raises(PlacementRefused) as excinfo:
        stage.open({"x": np.ones(4, np.float32)}, profile_dir=tmp_path)
    assert excinfo.value.refusal.code == REFUSE_EP_PLACEMENT
    assert "1 of 10" in excinfo.value.refusal.message
    assert "Light" in excinfo.value.refusal.remedy


def test_unverified_placement_is_refused_not_assumed(a16w8_graph, npu_device, fake_route, tmp_path):
    """``None`` means nobody looked. It must not be rounded to "fine"."""
    fake_route(load={"fraction_on_target": None, "node_counts": {},
                     "placement_unknown_reason": "no profiled run"})
    plan = plan_external_stage(_artifact(a16w8_graph), [npu_device])
    stage = ExternalStage(plan)
    with pytest.raises(PlacementRefused) as excinfo:
        stage.open({"x": np.ones(4, np.float32)}, profile_dir=tmp_path)
    assert excinfo.value.refusal.code == REFUSE_EP_PLACEMENT
    assert "not measured" in excinfo.value.refusal.message


def test_open_without_example_inputs_is_a_usage_error(a16w8_graph, npu_device, fake_route):
    fake_route()
    plan = plan_external_stage(_artifact(a16w8_graph), [npu_device])
    stage = ExternalStage(plan)
    with pytest.raises(ValueError, match="example_inputs are required"):
        stage.open({})
    stage.close()


def test_verify_placement_accepts_a_real_split(a16w8_graph, npu_device):
    from vllm_omni.edge.local.external.client import LoadReport

    plan = plan_external_stage(_artifact(a16w8_graph), [npu_device])
    plan.min_fraction_on_target = 0.3
    report = LoadReport(ep="vitisai", graph_path="g.onnx", total_nodes=3, target_nodes=1,
                        fraction_on_target=1 / 3, node_counts={"vitisai": 1, "CPUExecutionProvider": 2})
    assert verify_placement(plan, report) is None


def test_load_report_preserves_selected_adapter_name():
    from vllm_omni.edge.local.external.client import LoadReport

    report = LoadReport.from_body(
        {"device_name": "AMD Radeon(TM) 890M Graphics\x00", "fraction_on_target": 1.0,
         "placement_granularity": "output_device", "target_nodes": 1, "total_nodes": 1},
        ep="dml", graph_path="cosmos.pt2",
    )
    assert report.device_name == "AMD Radeon(TM) 890M Graphics"
    assert report.placement_granularity == "output_device"
    assert "1/1 outputs" in report.summary()
    assert report.to_dict()["device_name"] == report.device_name


def test_refusing_a_plan_cannot_be_opened(a16w8_graph, npu_device):
    plan = plan_external_stage(_artifact(a16w8_graph, fmt=FORMAT_ONNX_FP16), [npu_device])
    with pytest.raises(PlacementRefused):
        ExternalStage(plan)


def test_a_masked_device_is_told_to_drop_the_mask(a16w8_graph, npu_device):
    """The remedy has to name the cause the caller actually has.

    A masked device and a device WSL cannot forward are both "unreachable", and
    explaining MCDM forwarding to somebody who simply passed ``--mask npu``
    sends them after a problem they do not have.
    """
    import dataclasses

    masked = dataclasses.replace(npu_device, runnable=False, masked=True,
                                 backend=None, reason="masked out by --mask (npu)")
    plan = plan_external_stage(_artifact(a16w8_graph), [masked])
    assert not plan.admitted
    assert plan.refusals[0].code == REFUSE_ROUTE
    assert "--mask" in plan.refusals[0].remedy
    assert "MCDM" not in plan.refusals[0].remedy


def test_a_stage_larger_than_the_shared_pool_is_refused(a16w8_graph, npu_device):
    plan = plan_external_stage(_artifact(a16w8_graph), [npu_device], pool_capacity_bytes=1)
    assert not plan.admitted
    from vllm_omni.edge.local.plan import REFUSE_CAPACITY

    assert plan.refusals[0].code == REFUSE_CAPACITY
    assert "same pool the CPU stage is budgeted against" in plan.refusals[0].message


def test_a_silently_swapped_provider_is_refused_by_name(a16w8_graph, npu_device, fake_route, tmp_path):
    """onnxruntime does not raise when a provider fails to initialise.

    It prints "Falling back to ['CPUExecutionProvider'] and retrying" and hands
    back a working CPU session — which is how a 27B vision tower ran end to end
    on the CPU here while the caller believed it was on the iGPU. The report
    carries the swap, and it outranks the node count because "the provider
    never initialised" says what to fix and "0 of 1171 nodes" does not.
    """
    fake_route(load={
        "session_providers": ["CPUExecutionProvider"],
        "requested_provider_missing": "DmlExecutionProvider",
        "fallback_note": "DmlExecutionProvider failed to initialise and onnxruntime silently fell back",
        "node_counts": {"CPUExecutionProvider": 113},
        "total_nodes": 113, "target_nodes": 0, "fraction_on_target": 0.0,
    })
    plan = plan_external_stage(_artifact(a16w8_graph), [npu_device])
    stage = ExternalStage(plan)
    with pytest.raises(PlacementRefused) as excinfo:
        stage.open({"x": np.ones(4, np.float32)}, profile_dir=tmp_path)
    assert excinfo.value.refusal.code == REFUSE_EP_PLACEMENT
    assert "silently fell back" in excinfo.value.refusal.message
