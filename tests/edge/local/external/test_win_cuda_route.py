# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""[W4] The native-Windows vLLM route, and why it is refused.

``win-cuda`` is the first route whose far side is torch rather than
onnxruntime, and the first that exists to leave WSL rather than to reach a
device WSL cannot see. Declaring it makes a Windows engine a row in the route
table instead of a parallel universe.

It is also the first route that **cannot pass the placement gate**, and these
tests pin that down so nobody later mistakes the refusal for a bug. M3 made a
measured device attribution a hard admission condition after a VitisAI session
listed its EP, claimed zero nodes, and returned output bit-identical to the
CPU's. vLLM emits no per-operation device map, so a ``win-cuda`` worker has
nothing of that kind to report -- and the rule is about evidence, not about
whose code it is, so our own engine is refused on it exactly as the NPU is.
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm_omni.edge.local.capabilities import FORMAT_ONNX_A16W8
from vllm_omni.edge.local.external import launch
from vllm_omni.edge.local.external.stage import (
    ExternalStage,
    PlacementRefused,
    plan_external_stage,
)
from vllm_omni.edge.local.manifest import build_graph_artifact
from vllm_omni.edge.local.plan import REFUSE_EP_PLACEMENT, REFUSE_NO_DEVICE

from .conftest import _device


def _artifact(path):
    return build_graph_artifact(
        path, fmt=FORMAT_ONNX_A16W8, opset=21, source_model="test",
        component="stage", exporter="tests.edge.local.external",
    )


def test_win_cuda_is_a_known_route():
    """It resolves through the same machinery as every other route."""
    route = launch.resolve(launch.ROUTE_WIN_CUDA)
    assert route.name == "win-cuda"
    assert route.ep == "cuda"
    # It needs a native-Windows interpreter, like the two AMD Windows routes.
    assert route.is_windows is True
    # Available or not is environment-dependent; a reason is not.
    assert route.available or route.reason


def test_win_cuda_reports_no_placement_evidence():
    """The constant is the contract: ``None`` means nobody can look yet.

    Asserted rather than assumed, because the tempting edit -- setting it to
    1.0 because the worker "obviously" runs on the GPU -- is exactly the
    reasoning M3 disproved.
    """
    assert launch.WIN_CUDA_PLACEMENT_EVIDENCE is None


def test_unverified_placement_is_refused_for_our_own_engine(a16w8_graph, fake_route, tmp_path):
    """A worker that cannot say where it ran is refused, not trusted.

    The fake worker stands in for a ``win-cuda`` one: it runs, it returns
    correct-looking numbers, and it reports ``fraction_on_target=None`` because
    vLLM has no per-op device map to derive one from. That is precisely the
    shape of a silent CPU fallback, so it must not be admitted.
    """
    fake_route(load={
        "fraction_on_target": None,
        "node_counts": {},
        "placement_unknown_reason": "vllm emits no per-operation device map",
    })
    device = _device("npu:amd", "npu", [FORMAT_ONNX_A16W8], launch.ROUTE_CPU)
    plan = plan_external_stage(_artifact(a16w8_graph), [device])
    stage = ExternalStage(plan)
    with pytest.raises(PlacementRefused) as excinfo:
        stage.open({"x": np.ones(4, np.float32)}, profile_dir=tmp_path)
    assert excinfo.value.refusal.code == REFUSE_EP_PLACEMENT
    assert "not measured" in excinfo.value.refusal.message


def test_external_planner_does_not_yet_admit_a_discrete_gpu(a16w8_graph, fake_route):
    """A documented gap, pinned so the next person does not rediscover it.

    ``plan_external_stage`` filters candidates to ``("gpu_integrated", "npu")``
    (``stage.py``). Every external route so far reaches a device WSL cannot see
    *and* that is integrated or an NPU, so the filter was exactly right. A
    ``win-cuda`` route breaks that coupling: its device is a **discrete** GPU
    reached out-of-process for an OS reason, not a visibility one.

    This test asserts the current refusal rather than a desired behaviour. When
    the planner learns about externally-routed discrete GPUs, this test should
    fail and be rewritten -- that failure is the signal, not a regression.
    """
    fake_route()
    device = _device("cuda:0", "gpu_discrete", [FORMAT_ONNX_A16W8], launch.ROUTE_CPU)
    plan = plan_external_stage(_artifact(a16w8_graph), [device])
    assert plan.device is None
    assert plan.refusals
    assert plan.refusals[0].code == REFUSE_NO_DEVICE
    assert "integrated GPU or NPU" in plan.refusals[0].message
