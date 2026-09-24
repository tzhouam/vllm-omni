# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""An ONNX Runtime stage worker, run by an interpreter the engine cannot use.

Started by :mod:`vllm_omni.edge.local.external.client`, which binds a listener
first and passes its address in; this process connects *back*. That direction
is deliberate: a Windows process reaching into WSL needs no firewall rule,
while WSL reaching into Windows does.

Run with **stdlib + numpy + onnxruntime only**. It is executed by
``C:\\Users\\zhout\\npu-ep\\Scripts\\python.exe`` -- Python 3.12.10,
onnxruntime 1.30.0, onnx 1.22.0, numpy 2.5.3 -- which has no torch, no vllm and
no vllm_omni. The one thing it does borrow is
:mod:`vllm_omni.edge.npu_ryzenai`, loaded **by file path** rather than
imported: that module is plain stdlib (it only imports ``onnxruntime`` inside
its functions), so loading it as a standalone file reuses the EP recipe instead
of copying it, without dragging in the package ``__init__``.

**What "it ran on the NPU" means here.** A VitisAI session that lists the EP in
``get_providers()`` may still be executing every node on the CPU, and when it
does its outputs are bit-identical to the CPU's -- which is precisely what a
*correct* NPU run does not look like. So identical numbers cannot confirm
placement and small differences cannot confirm it either. The only evidence is
the per-node provider assignment, which ORT emits into its profile, so
:func:`_load` always does one profiled warm-up run and reports the counts. The
engine's planner refuses on that number; this process does not decide.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROTOCOL = _HERE / "protocol.py"
_NPU_RYZENAI = _HERE.parents[1] / "npu_ryzenai.py"


def _load_sibling(path: Path, name: str) -> Any:
    """Import a single .py file without importing its package.

    ``vllm_omni.edge.npu_ryzenai`` cannot be imported normally here: the
    package ``__init__`` chain reaches vllm, which is not installed on the
    Windows side. The module itself has no such dependency.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - path is ours
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


proto = _load_sibling(_PROTOCOL, "_edge_external_protocol")


def _memory_bytes() -> tuple[int, int]:
    """``(resident, peak resident)`` for this process, on either OS.

    The engine's :class:`~vllm_omni.edge.local.engine.MemorySampler` walks a
    psutil process tree, and that walk stops at the WSL boundary -- a Windows
    worker's memory is simply invisible to it. Since both AMD devices allocate
    out of the *same host RAM* the CPU stage is budgeted against, an unreported
    worker is a hole in the admission ledger rather than a rounding error. So
    the worker reports its own.

    The Windows path needs explicit ``argtypes``/``restype``: ``GetCurrentProcess``
    returns a pseudo-handle that ctypes truncates to a 32-bit int without them,
    and ``GetProcessMemoryInfo`` then fails and returns zero -- which reads as
    "the worker used no memory" instead of "the query failed".
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        # Since Windows 7 the psapi entry points are forwarders onto kernel32's
        # K32* symbols; prefer those and keep psapi as the fallback.
        query = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if query is None:
            query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.restype = wintypes.BOOL
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return 0, 0
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)

    resident = peak = 0
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                resident = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) * 1024
    except OSError:
        pass
    return resident, peak


def _rss_bytes() -> int:
    return _memory_bytes()[0]


class _Session:
    """One loaded graph, plus the placement evidence for it."""

    def __init__(self) -> None:
        self.session: Any = None
        self.report: dict[str, Any] = {}
        self.runs = 0
        self.total_run_s = 0.0
        self.output_names: list[str] = []


def _provider_counts(profile_path: str) -> dict[str, int]:
    """Per-EP node counts out of an ORT profile.

    Same rule as :func:`vllm_omni.edge.npu_ryzenai.nodes_on_npu`, generalised
    to every provider so the report can show the split rather than one number:
    a graph that is 3 nodes on the NPU and 6 on the CPU is a very different
    result from 9 and 0, and only the split says which one happened.
    """
    try:
        events = json.loads(Path(profile_path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    counts: dict[str, int] = {}
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = (event.get("args") or {}).get("provider")
        if provider:
            counts[str(provider)] = counts.get(str(provider), 0) + 1
    return counts


def _verify_artifact_members(graph: str, files: list[str]) -> None:
    """Reject ONNX external data omitted from the controller's verified bundle."""
    import onnx

    allowed = {Path(name).resolve() for name in files}
    graph_path = Path(graph).resolve()
    if graph_path not in allowed:
        raise ValueError("graph is not a verified artifact member")
    pending = [onnx.load_model(graph, load_external_data=False)]
    while pending:
        message = pending.pop()
        if message.DESCRIPTOR.full_name == "onnx.TensorProto":
            locations = [entry.value for entry in message.external_data if entry.key == "location"]
            if message.data_location == onnx.TensorProto.EXTERNAL and not locations:
                raise ValueError("external tensor has no payload location")
            for location in locations:
                if (graph_path.parent / location).resolve() not in allowed:
                    raise ValueError(f"external tensor payload is absent from verified manifest: {location}")
        for descriptor, value in message.ListFields():
            if descriptor.type == descriptor.TYPE_MESSAGE:
                pending.extend(value if descriptor.is_repeated else [value])


def _make_session(body: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Build the ORT session for the requested execution provider."""
    import onnxruntime as ort

    ep = str(body.get("ep", "cpu"))
    graph = str(body["graph_path"])
    if "artifact_files" in body:
        _verify_artifact_members(graph, body["artifact_files"])
    info: dict[str, Any] = {"ep": ep, "onnxruntime": ort.__version__}

    options = ort.SessionOptions()
    if body.get("profile"):
        options.enable_profiling = True
        if body.get("profile_prefix"):
            options.profile_file_prefix = str(body["profile_prefix"])
    if body.get("intra_op_num_threads"):
        options.intra_op_num_threads = int(body["intra_op_num_threads"])

    if ep == "vitisai":
        # The recipe, not a copy of it: the EP directory has to go on the DLL
        # search path before registration or the load fails with an unhelpful
        # "no dependency", and the package directory denies listing so it is
        # found through Get-AppxPackage.
        npu = _load_sibling(_NPU_RYZENAI, "_edge_npu_ryzenai")
        ep_dir = body.get("ep_dir")
        name = npu.register(Path(ep_dir) if ep_dir else None)
        merged = npu.session_options(name)
        merged.enable_profiling = options.enable_profiling
        if body.get("profile_prefix"):
            merged.profile_file_prefix = str(body["profile_prefix"])
        options = merged
        info["ep_directory"] = str(body.get("ep_dir") or npu.find_ep_directory())
        providers = [name]
    elif ep == "dml":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        device_id = body.get("device_id")
        if device_id is None:
            providers = ["DmlExecutionProvider"]
        else:
            if type(device_id) is not int or device_id < 0:
                raise ValueError("DirectML device_id must be a nonnegative integer")
            providers = [("DmlExecutionProvider", {"device_id": str(device_id)})]
            info["device_id_requested"] = device_id
    elif ep == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError(f"unknown execution provider {ep!r}; expected vitisai, dml or cpu")

    available = list(ort.get_available_providers())
    info["available_providers"] = available
    if ep == "dml" and "DmlExecutionProvider" not in available:
        raise RuntimeError(
            "DmlExecutionProvider is not in this interpreter's onnxruntime. "
            "The DirectML build is a separate package (onnxruntime-directml) "
            "and cannot share a venv with the plain onnxruntime that VitisAI "
            "needs, because that one is pinned at 1.24.4 and VitisAI wants "
            ">= 1.25. Two venvs, one per device."
        )

    started = time.perf_counter()
    session = ort.InferenceSession(graph, sess_options=options, providers=providers)
    info["session_create_s"] = time.perf_counter() - started
    info["session_providers"] = list(session.get_providers())

    # ORT's Python binding does not fail when a requested provider cannot
    # initialise: it prints "Falling back to ['CPUExecutionProvider'] and
    # retrying" and hands back a working CPU session. Observed here with
    # DirectML raising E_INVALIDARG on a Reshape carrying allowzero=1 -- the
    # session then ran the whole vision tower on the CPU and returned correct
    # features. The node-assignment gate downstream catches that, but only
    # after a profiled run; naming it at the source turns a 0%-placement number
    # into the actual reason.
    requested_provider = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
    if requested_provider not in info["session_providers"]:
        info["requested_provider_missing"] = requested_provider
        info["fallback_note"] = (
            f"{requested_provider} failed to initialise and onnxruntime silently fell "
            f"back to {info['session_providers']}. Anything this session computes "
            "runs there, not on the requested device."
        )
    return session, info


def _load(state: _Session, body: dict[str, Any], tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    session, info = _make_session(body)
    state.session = session
    state.output_names = [o.name for o in session.get_outputs()]

    info["inputs"] = [{"name": i.name, "type": i.type, "shape": list(i.shape)} for i in session.get_inputs()]
    info["outputs"] = [{"name": o.name, "type": o.type, "shape": list(o.shape)} for o in session.get_outputs()]

    # The placement evidence. One profiled run, because ORT only writes node
    # assignments once nodes have actually executed -- a session that has never
    # run has no placement to report, only an intention.
    if body.get("profile") and tensors:
        started = time.perf_counter()
        session.run(state.output_names, {k: v for k, v in tensors.items()})
        info["warmup_s"] = time.perf_counter() - started
        profile_path = session.end_profiling()
        info["profile_path"] = str(profile_path)
        counts = _provider_counts(str(profile_path))
        info["node_counts"] = counts
        total = sum(counts.values())
        target = str(body.get("ep", "cpu"))
        target_key = {"vitisai": "vitisai", "dml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider"}[target]
        info["total_nodes"] = total
        info["target_nodes"] = counts.get(target_key, 0)
        info["fraction_on_target"] = (counts.get(target_key, 0) / total) if total else 0.0
    elif body.get("profile"):
        info["node_counts"] = {}
        info["fraction_on_target"] = None
        info["placement_unknown_reason"] = (
            "no example inputs were sent, so no run happened and ORT wrote no "
            "node assignments; placement is unverified, not verified-empty"
        )

    info["rss_bytes"] = _rss_bytes()
    state.report = info
    return info


def _run(state: _Session, tensors: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if state.session is None:
        raise RuntimeError("run before load")
    started = time.perf_counter()
    outputs = state.session.run(state.output_names, dict(tensors))
    elapsed = time.perf_counter() - started
    state.runs += 1
    state.total_run_s += elapsed
    return {"run_s": elapsed, "runs": state.runs}, dict(zip(state.output_names, outputs))


def serve(host: str, port: int, token: str) -> int:
    sock = socket.create_connection((host, port), timeout=60.0)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(None)
    state = _Session()
    try:
        import onnxruntime as ort

        ort_version, providers = ort.__version__, list(ort.get_available_providers())
    except Exception as exc:  # pragma: no cover - reported, not raised
        ort_version, providers = f"unavailable: {exc}", []

    proto.send_message(
        sock,
        proto.OP_HELLO,
        {
            "token": token,
            "executable": sys.executable,
            "python": sys.version,
            "platform": sys.platform,
            "pid": os.getpid(),
            "onnxruntime": ort_version,
            "numpy": np.__version__,
            "available_providers": providers,
            "rss_bytes": _rss_bytes(),
        },
    )

    while True:
        try:
            op, body, tensors = proto.recv_message(sock)
        except proto.ProtocolError:
            return 0  # the engine went away; that is a normal shutdown here
        try:
            if op == proto.OP_LOAD:
                proto.send_message(sock, proto.OP_OK, _load(state, body, tensors))
            elif op == proto.OP_RUN:
                reply, outputs = _run(state, tensors)
                proto.send_message(sock, proto.OP_OK, reply, outputs)
            elif op == proto.OP_STATS:
                proto.send_message(
                    sock,
                    proto.OP_OK,
                    {
                        "runs": state.runs,
                        "total_run_s": state.total_run_s,
                        "rss_bytes": _memory_bytes()[0],
                        "peak_rss_bytes": _memory_bytes()[1],
                        "report": state.report,
                    },
                )
            elif op == proto.OP_CLOSE:
                proto.send_message(sock, proto.OP_OK, {})
                return 0
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception as exc:
            proto.send_message(
                sock,
                proto.OP_ERR,
                {
                    "message": f"{type(exc).__name__}: {exc}",
                    "code": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", default="")
    args = parser.parse_args(argv)
    return serve(args.host, args.port, args.token)


if __name__ == "__main__":
    raise SystemExit(main())
