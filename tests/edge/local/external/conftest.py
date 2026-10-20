# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixtures that stand in for the two AMD devices.

Nothing here needs a Radeon, an NPU, onnxruntime or Windows: the worker is
replaced by ``fake_worker.py`` and the devices by hand-built capabilities. The
paths that matter -- framing, the placement gate, the refusal codes -- are
logic, and logic that only runs on one laptop is logic nobody can change
safely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from vllm_omni.edge.local.capabilities import (
    FORMAT_ONNX_A16W8,
    FORMAT_ONNX_FP16,
    FORMAT_ONNX_FP32,
    DeviceCapability,
)
from vllm_omni.edge.local.external import launch

FAKE_WORKER = str(Path(__file__).resolve().parent / "fake_worker.py")


@pytest.fixture
def fake_route(monkeypatch, tmp_path):
    """Point the ``ort-cpu`` route at the fake worker and this interpreter.

    Returns a callable that writes the worker's reply script and hands back the
    resolved route.
    """

    def configure(**config) -> launch.Route:
        config_path = tmp_path / "fake_worker.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setenv("VLLM_OMNI_FAKE_WORKER_JSON", str(config_path))
        monkeypatch.setenv("VLLM_OMNI_EXTERNAL_WORKER_ORT_CPU", FAKE_WORKER)
        monkeypatch.setenv("VLLM_OMNI_EXTERNAL_PYTHON_ORT_CPU", sys.executable)
        return launch.resolve(launch.ROUTE_CPU)

    return configure


def _device(device_id: str, kind: str, formats, route: str, *, runnable: bool = True,
            reason: str = "") -> DeviceCapability:
    return DeviceCapability(
        device_id=device_id,
        kind=kind,
        vendor="amd",
        name=f"test {kind}",
        backend="ort:cpu" if runnable else None,
        runnable=runnable,
        memory_pool="host_ram",
        memory_bytes=32 * 2**30,
        compute_capability=None,
        weight_formats=frozenset(formats),
        evidence="B" if runnable else "D",
        reason=reason,
        extra={"worker_route": route, "routes": [{"name": route, "available": runnable,
                                                  "reason": reason}]},
    )


@pytest.fixture
def npu_device() -> DeviceCapability:
    """A16W8 only -- the real constraint, not a simplification."""
    return _device("npu:amd", "npu", {FORMAT_ONNX_A16W8}, launch.ROUTE_CPU)


@pytest.fixture
def igpu_device() -> DeviceCapability:
    return _device("igpu:amd", "gpu_integrated", {FORMAT_ONNX_FP16, FORMAT_ONNX_FP32},
                   launch.ROUTE_CPU)


@pytest.fixture
def unreachable_npu() -> DeviceCapability:
    return _device(
        "npu:amd", "npu", {FORMAT_ONNX_A16W8}, launch.ROUTE_VITISAI,
        runnable=False,
        reason="an MCDM device; WSL2 forwards display adapters only, so there is no device node here",
    )


@pytest.fixture
def a16w8_graph(tmp_path) -> Path:
    """A file that stands in for an exported graph. Never loaded by ORT here."""
    path = tmp_path / "stage.onnx"
    path.write_bytes(b"not really onnx, and nothing in these tests parses it" * 64)
    return path
