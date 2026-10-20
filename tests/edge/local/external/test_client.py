# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Starting, driving and losing a worker process."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_omni.edge.local.external import launch
from vllm_omni.edge.local.external.client import ExternalWorker, WorkerStartError


def test_handshake_and_round_trip(fake_route):
    with ExternalWorker(fake_route()) as worker:
        assert worker.hello["onnxruntime"] == "fake"
        report = worker.load("/nonexistent/stage.onnx", example_inputs={"x": np.ones(4, np.float32)})
        assert report.fraction_on_target == 1.0
        assert report.rss_bytes == 111 * 2**20

        outputs, timing = worker.run({"x": np.ones(4, np.float32)})
        assert np.array_equal(outputs["x_out"], np.full(4, 2.0, np.float32))
        assert timing.round_trip_s >= timing.worker_s
        assert timing.transport_s == pytest.approx(timing.round_trip_s - timing.worker_s)

        assert worker.stats()["peak_rss_bytes"] == 122 * 2**20


def test_wrong_token_is_refused(fake_route):
    """The listener is on a host-visible address; a stray peer must not be served."""
    worker = ExternalWorker(fake_route(token="not-the-token"))
    with pytest.raises(WorkerStartError, match="wrong token"):
        worker.start()
    worker.close()


def test_worker_that_vanishes_mid_request_reports_rather_than_hangs(fake_route):
    worker = ExternalWorker(fake_route(die_on="run"))
    worker.start()
    worker.load("/nonexistent/stage.onnx", example_inputs={"x": np.ones(2, np.float32)})
    with pytest.raises(WorkerStartError, match="worker connection failed"):
        worker.run({"x": np.ones(2, np.float32)})
    worker.close()


def test_unavailable_route_refuses_at_construction():
    route = launch.Route(
        name="ort-vitisai", interpreter=None, worker="w.py", ep="vitisai",
        is_windows=True, available=False, reason="no interpreter installed",
    )
    with pytest.raises(WorkerStartError, match="no interpreter installed"):
        ExternalWorker(route)


def test_close_is_idempotent(fake_route):
    worker = ExternalWorker(fake_route())
    worker.start()
    worker.close()
    worker.close()


def test_bind_address_is_loopback_for_a_same_os_worker():
    from vllm_omni.edge.local.external import client

    assert client._bind_address(is_windows=False) == "127.0.0.1"


def test_bind_address_for_a_windows_worker_is_not_loopback_under_wsl():
    """The regression this exists for: probing the WSL nameserver instead of the
    gateway yields 10.255.255.254, an address only this distro can reach, and
    the Windows worker then dies with ECONNREFUSED after a silent start."""
    from vllm_omni.edge.local.external import client

    if not launch.is_wsl():
        pytest.skip("not running under WSL interop")
    assert not client._bind_address(is_windows=True).startswith(("127.", "10.255."))


def test_external_data_counts_toward_the_artifact_size(tmp_path):
    """An ONNX graph with external weights is a few MB of topology pointing at
    gigabytes beside it. Charging only the .onnx told the admission ledger that
    a 1.8 GB vision tower costs 2 MiB."""
    import onnx
    from onnx import helper, numpy_helper

    from vllm_omni.edge.local.manifest import build_graph_artifact

    weight = numpy_helper.from_array(np.zeros((256, 256), np.float32), name="w")
    node = helper.make_node("MatMul", ["x", "w"], ["y"])
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 256])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 256])],
        [weight],
    )
    path = tmp_path / "m.onnx"
    onnx.save(helper.make_model(graph), str(path), save_as_external_data=True,
              all_tensors_to_one_file=True, location="m.onnx.data")

    artifact = build_graph_artifact(path, fmt="onnx:fp32", opset=21, source_model="t",
                                    component="c", exporter="e")
    assert artifact.bytes > path.stat().st_size
    assert artifact.bytes >= 256 * 256 * 4
