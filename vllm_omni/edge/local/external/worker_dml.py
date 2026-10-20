# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A torch-DirectML stage worker for the Radeon 890M.

Speaks the same protocol as :mod:`worker_ort`, in its own interpreter for the
usual reason: ``torch-directml`` pins torch 2.4.1 and Omni runs 2.13, so the
two cannot share an environment. Here that interpreter is a WSL venv
(``.venvs/dml``) rather than a Windows one -- the 890M is a WDDM display
adapter, which GPU-PV *does* forward, so DirectML reaches it over ``/dev/dxg``
without leaving Linux. The NPU is the one that cannot be reached this way.

**It consumes torch exports, not ONNX.** That is not a second way of doing the
same thing: ``vllm_omni.edge.decoder_export`` already emits ``.pt2`` via
``torch.export``, and the alternative iGPU route (onnxruntime-directml on
Windows, served by ``worker_ort``) needs its own venv because that package is
pinned at 1.24.4 and cannot load the VitisAI EP the NPU needs. Two routes to
one device, two artifact formats; the plan records which one ran.

**What placement evidence means here.** There is no per-node execution-provider
report, because there is no partitioner -- torch dispatches to the device the
tensors are on. So the evidence is the device the outputs actually came back
from: a module that silently fell back to the CPU returns CPU tensors, and
``fraction_on_target`` is 0. It is a coarser signal than ORT's node counts and
it catches the failure that matters, which is the same one: work reported on an
accelerator that ran on the CPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_edge_external_protocol", _HERE / "protocol.py")
proto = importlib.util.module_from_spec(spec)
sys.modules["_edge_external_protocol"] = proto
spec.loader.exec_module(proto)

DEVICE_TYPE = "privateuseone"
"""What torch calls a DirectML tensor's device. Checking for this string is how
a CPU fallback is caught."""


def _rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _peak_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class _Session:
    def __init__(self) -> None:
        self.module: Any = None
        self.device: Any = None
        self.dtype: Any = None
        self.input_order: list[str] = []
        self.output_names: list[str] = []
        self.runs = 0
        self.total_run_s = 0.0
        self.report: dict[str, Any] = {}


def _select_device(index: int | None) -> tuple[Any, str]:
    """The 890M, by name rather than by index.

    ``torch_directml`` enumerates every DirectML adapter, and on this laptop
    that includes the RTX 5090 at index 0 -- taking index 0 would quietly
    benchmark the discrete GPU and call it the iGPU.
    """
    import torch_directml

    count = torch_directml.device_count()
    if index is not None:
        return torch_directml.device(index), torch_directml.device_name(index)
    for i in range(count):
        name = torch_directml.device_name(i)
        if "890M" in name or "Radeon" in name:
            return torch_directml.device(i), name
    raise RuntimeError(
        "no integrated Radeon adapter among "
        f"{[torch_directml.device_name(i) for i in range(count)]}; pass device_index "
        "explicitly if the intended adapter is named differently"
    )


def _load(state: _Session, body: dict[str, Any], tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    import torch

    graph = str(body["graph_path"])
    device, device_name = _select_device(body.get("device_index"))
    state.device = device
    # No cast unless one is asked for. The artifact's precision is part of its
    # identity -- it is in the manifest and in the format string -- so quietly
    # casting an fp16 export to fp32 because that is the friendlier default
    # would make the record describe a graph that did not run.
    requested = body.get("dtype")
    state.dtype = getattr(torch, str(requested)) if requested else None

    started = time.perf_counter()
    if not graph.endswith(".pt2"):
        # Measured on this laptop, torch-directml 0.2.5 / torch 2.4.1: a
        # TorchScript module cannot be put on a DirectML device by either
        # route. ``torch.jit.load(map_location=dml)`` raises
        # "Cannot access storage of OpaqueTensorImpl", and loading to CPU then
        # ``.to(device)`` trips ``assert isinstance(param, Parameter)`` inside
        # ``Module._apply``, because a RecursiveScriptModule's parameters are
        # not ``Parameter`` instances. Refused with the reason rather than
        # left to surface as a bare AssertionError from inside torch.
        raise ValueError(
            f"{Path(graph).name}: torch-directml cannot load a TorchScript "
            "module onto the iGPU (both map_location and .to(device) fail "
            "inside torch). Export with torch.export to a .pt2 instead -- "
            "which is what vllm_omni.edge.decoder_export already produces."
        )
    module = torch.export.load(graph).module()
    if state.dtype is not None:
        module = module.to(state.dtype)
    # No ``.eval()``: an exported program's module raises
    # "Calling eval() is not supported yet" on torch 2.4.1, and it is already
    # captured in inference mode, so there is nothing to switch off.
    module = module.to(device)
    state.module = module
    create_s = time.perf_counter() - started

    state.input_order = list(body.get("input_order") or tensors.keys())
    state.output_names = list(body.get("output_names") or [])

    info: dict[str, Any] = {
        "ep": "dml",
        "device_name": device_name,
        "torch": torch.__version__,
        "session_create_s": create_s,
        "session_providers": ["torch-directml"],
        "available_providers": ["torch-directml"],
        "inputs": [{"name": n, "shape": list(tensors[n].shape)} for n in state.input_order
                   if n in tensors],
    }

    if tensors:
        started = time.perf_counter()
        outputs = _forward(state, tensors)
        info["warmup_s"] = time.perf_counter() - started
        on_device = sum(1 for t in outputs if getattr(t, "device", None) is not None
                        and t.device.type == DEVICE_TYPE)
        total = len(outputs)
        info["node_counts"] = {"dml": on_device, "cpu": total - on_device}
        info["total_nodes"] = total
        info["target_nodes"] = on_device
        info["fraction_on_target"] = (on_device / total) if total else 0.0
        info["outputs"] = [{"name": f"out_{i}", "shape": list(t.shape)}
                           for i, t in enumerate(outputs)]
        if not state.output_names:
            state.output_names = [f"out_{i}" for i in range(total)]
    else:
        info["node_counts"] = {}
        info["fraction_on_target"] = None
        info["placement_unknown_reason"] = (
            "no example inputs were sent, so nothing ran and no output device "
            "was observed; placement is unverified, not verified-empty"
        )

    info["rss_bytes"] = _rss_bytes()
    state.report = info
    return info


def _forward(state: _Session, tensors: dict[str, np.ndarray]) -> list[Any]:
    import torch

    args = []
    for name in state.input_order:
        if name not in tensors:
            continue
        tensor = torch.from_numpy(np.ascontiguousarray(tensors[name]))
        if state.dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(state.dtype)
        args.append(tensor.to(state.device))
    with torch.no_grad():
        result = state.module(*args)
    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, dict):
        state.output_names = list(result.keys())
        return list(result.values())
    return list(result)


def _run(state: _Session, tensors: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if state.module is None:
        raise RuntimeError("run before load")
    started = time.perf_counter()
    outputs = _forward(state, tensors)
    # ``.cpu()`` is the synchronisation point: DirectML queues work, so timing
    # without the copy back would measure submission, not execution.
    materialised = [t.detach().to("cpu").numpy() for t in outputs]
    elapsed = time.perf_counter() - started
    state.runs += 1
    state.total_run_s += elapsed
    names = state.output_names or [f"out_{i}" for i in range(len(materialised))]
    return {"run_s": elapsed, "runs": state.runs}, dict(zip(names, materialised))


def serve(host: str, port: int, token: str) -> int:
    sock = socket.create_connection((host, port), timeout=60.0)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    state = _Session()
    try:
        import torch
        import torch_directml

        versions = {
            "torch": torch.__version__,
            "torch_directml": getattr(torch_directml, "__version__", "unknown"),
            "available_providers": [
                torch_directml.device_name(i) for i in range(torch_directml.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - reported, not raised
        versions = {"torch": f"unavailable: {exc}", "available_providers": []}

    proto.send_message(
        sock, proto.OP_HELLO,
        {"token": token, "executable": sys.executable, "python": sys.version,
         "platform": sys.platform, "pid": os.getpid(), "onnxruntime": "n/a (torch-directml)",
         "numpy": np.__version__, "rss_bytes": _rss_bytes(), **versions},
    )

    while True:
        try:
            op, body, tensors = proto.recv_message(sock)
        except proto.ProtocolError:
            return 0
        try:
            if op == proto.OP_LOAD:
                proto.send_message(sock, proto.OP_OK, _load(state, body, tensors))
            elif op == proto.OP_RUN:
                reply, outputs = _run(state, tensors)
                proto.send_message(sock, proto.OP_OK, reply, outputs)
            elif op == proto.OP_STATS:
                proto.send_message(sock, proto.OP_OK, {
                    "runs": state.runs, "total_run_s": state.total_run_s,
                    "rss_bytes": _rss_bytes(), "peak_rss_bytes": _peak_rss_bytes(),
                    "report": state.report,
                })
            elif op == proto.OP_CLOSE:
                proto.send_message(sock, proto.OP_OK, {})
                return 0
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception as exc:
            proto.send_message(sock, proto.OP_ERR, {
                "message": f"{type(exc).__name__}: {exc}",
                "code": type(exc).__name__,
                "traceback": traceback.format_exc(),
            })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", default="")
    args = parser.parse_args(argv)
    return serve(args.host, args.port, args.token)


if __name__ == "__main__":
    raise SystemExit(main())
