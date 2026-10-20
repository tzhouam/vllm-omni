"""CudaIpcConnector: leaf walking (CPU) and a spawned producer/consumer round trip (CUDA)."""

from __future__ import annotations

import multiprocessing as mp
import os
import uuid

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.distributed.omni_connectors.connectors import cuda_ipc_connector as cic
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory


@pytest.mark.core_model
@pytest.mark.cpu
def test_walk_rewrites_tensor_leaves_only():
    seen = []

    def leaf(t):
        seen.append(t)
        return "L"

    data = {"a": torch.zeros(2), "b": [1, torch.ones(1), (torch.ones(1), "s")], "c": {"d": None}}
    out = cic._walk(data, leaf)
    assert out == {"a": "L", "b": [1, "L", ("L", "s")], "c": {"d": None}}
    assert len(seen) == 3
    # marker dicts are treated as leaves too
    marker = {cic.MARKER: 1, "args": b"", "event": b"", "dtype": "torch.float32", "shape": [1]}
    assert cic._walk({"m": marker}, lambda m: "M") == {"m": "M"}
    assert cic.is_marker(marker) and not cic.is_marker({"x": 1})


@pytest.mark.core_model
@pytest.mark.cpu
def test_cpu_leaves_pass_through_shared_memory(tmp_path):
    assert "CudaIpcConnector" in OmniConnectorFactory._registry
    conn = cic.CudaIpcConnector({"stage_id": 0, "extra": {"cuda_ipc_enabled": False}})
    key = f"cic_cpu_{uuid.uuid4().hex[:8]}"
    payload = {"codes": {"audio": torch.arange(6).reshape(3, 2)}, "meta": {"finished": True}}
    ok, size, meta = conn.put("stage0", "stage1", key, payload)
    assert ok and size > 0 and "cuda_ipc" not in (meta or {})
    got, _ = conn.get("stage0", "stage1", key, meta)
    assert torch.equal(got["codes"]["audio"], payload["codes"]["audio"]) and got["meta"]["finished"] is True
    conn.cleanup(key)
    assert "cuda_ipc" in conn.health()
    conn.close()


def _producer(key: str, q: mp.Queue):
    conn = cic.CudaIpcConnector({"stage_id": 0})
    t = torch.randint(0, 2048, (25, 16), device="cuda", dtype=torch.int64)
    hs = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    ok, size, meta = conn.put(
        "stage0", "stage1", key, {"codes": {"audio": t}, "hidden": hs, "meta": {"n": 25}, "cpu": torch.ones(2)}
    )
    q.put(("put", ok, size, meta, t.cpu(), hs.cpu()))
    # wait for consumer ack, then sweep on cleanup
    deadline = 60
    import time

    while deadline > 0 and not os.path.exists(cic._ack_path(key)):
        time.sleep(0.1)
        deadline -= 0.1
    conn.cleanup(key)
    q.put(("done", len(conn._exported)))


def _consumer(key: str, meta, q: mp.Queue):
    conn = cic.CudaIpcConnector({"stage_id": 1})
    got, size = conn.get("stage0", "stage1", key, meta)
    q.put(("got", got["codes"]["audio"].cpu(), got["hidden"].cpu(), got["meta"], got["cpu"], conn.health()["rebuilt"]))


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_cuda_ipc_round_trip_between_processes():
    ctx = mp.get_context("spawn")
    key = f"cic_gpu_{uuid.uuid4().hex[:8]}"
    q = ctx.Queue()
    prod = ctx.Process(target=_producer, args=(key, q))
    prod.start()
    tag, ok, size, meta, t_ref, hs_ref = q.get(timeout=120)
    assert tag == "put" and ok and meta.get("cuda_ipc") == 2
    cons = ctx.Process(target=_consumer, args=(key, meta, q))
    cons.start()
    tag, codes, hidden, m, cpu_t, rebuilt = q.get(timeout=120)
    assert tag == "got" and rebuilt == 2
    assert (
        torch.equal(codes, t_ref)
        and torch.equal(hidden, hs_ref)
        and m == {"n": 25}
        and torch.equal(cpu_t, torch.ones(2))
    )
    tag, live = q.get(timeout=120)
    assert tag == "done" and live == 0
    prod.join(30)
    cons.join(30)
    assert not os.path.exists(cic._ack_path(key))
