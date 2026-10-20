# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-direct connector for co-located CUDA stages (single host, same GPU).

Tensor leaves that live on a CUDA device are exported through CUDA IPC
(``torch.multiprocessing.reductions.reduce_tensor``) instead of being copied
to host memory and msgpack-serialized; everything else (CPU tensors, ids,
metadata, msgspec structs) still travels through the wrapped
:class:`SharedMemoryConnector`, so the control-plane contract (``put`` /
``get`` / ``cleanup``, key-addressed lookup, metadata pass-through) is
unchanged.

Protocol per exported tensor:

* producer records an inter-process CUDA event on the current stream, packs
  ``(rebuild args, event handle, dtype, shape)`` into a marker dict and keeps
  the source tensor alive in ``_exported[put_key]``;
* consumer rebuilds the tensor, waits on the event, **clones** it into its own
  allocation and writes an ack file; the producer releases the source on its
  next ``put``/``cleanup`` sweep (or when the registry exceeds ``max_live``).

Requirements: producer and consumer are different processes on the same GPU
(``VLLM_WORKER_MULTIPROC_METHOD=spawn``). Tensors allocated inside CUDA-graph
private pools cannot be exported; they are cloned into a plain allocation first.
"""

from __future__ import annotations

import copy
import os
import pickle
import time
from collections import OrderedDict
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector

logger = init_logger(__name__)

MARKER = "__cuda_ipc__"
# [edge-infer] /dev/shm where it exists (Linux, WSL); a per-user directory on Windows.
from vllm_omni.windows.paths import shm_dir as _shm_dir

ACK_DIR = _shm_dir()


def _ack_path(put_key: str) -> str:
    return os.path.join(ACK_DIR, f"cuda_ipc_ack_{put_key}")


def is_marker(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get(MARKER) == 1


class _Exported:
    __slots__ = ("tensor", "event", "t")

    def __init__(self, tensor: torch.Tensor, event: Any):
        self.tensor, self.event, self.t = tensor, event, time.time()


def export_tensor(t: torch.Tensor) -> tuple[dict[str, Any], _Exported]:
    """Export a CUDA tensor; returns (marker dict, keep-alive record)."""
    from torch.multiprocessing.reductions import reduce_tensor

    src = t.detach()
    if not src.is_contiguous():
        src = src.contiguous()
    try:
        _, args = reduce_tensor(src)
    except RuntimeError:
        # e.g. CUDA-graph private pool allocation: move to a plain allocation
        src = src.clone()
        _, args = reduce_tensor(src)
    event = torch.cuda.Event(interprocess=True)
    event.record(torch.cuda.current_stream(src.device))
    marker = {
        MARKER: 1,
        "args": pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL),
        "event": event.ipc_handle(),
        "device": int(src.device.index or 0),
        "dtype": str(src.dtype),
        "shape": list(src.shape),
    }
    return marker, _Exported(src, event)


def rebuild_tensor(marker: dict[str, Any]) -> torch.Tensor:
    """Rebuild an exported tensor in the consumer process and own a copy of it."""
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    args = pickle.loads(marker["args"])
    device = torch.device("cuda", int(marker.get("device", 0)))
    with torch.cuda.device(device):
        tensor = rebuild_cuda_tensor(*args)
        event = torch.cuda.Event.from_ipc_handle(device, marker["event"])
        event.wait(torch.cuda.current_stream(device))
        owned = tensor.clone()
    return owned


def _walk(obj: Any, fn) -> Any:
    """Apply ``fn`` to tensor leaves / marker dicts inside dicts, lists, tuples and msgspec structs."""
    if isinstance(obj, torch.Tensor) or is_marker(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _walk(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_walk(v, fn) for v in obj)
    fields = getattr(obj, "__struct_fields__", None)
    if fields:
        out = copy.copy(obj)
        for name in fields:
            setattr(out, name, _walk(getattr(obj, name), fn))
        return out
    return obj


class CudaIpcConnector(OmniConnectorBase):
    """CUDA-IPC fast path for GPU tensor leaves, shared memory for the rest."""

    supports_raw_data = False

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        extra = config.get("extra", {}) if isinstance(config.get("extra"), dict) else {}
        self.max_live = int(extra.get("cuda_ipc_max_live", 256))
        self.enabled = bool(extra.get("cuda_ipc_enabled", True)) and torch.cuda.is_available()
        self._inner = SharedMemoryConnector(config)
        self._exported: OrderedDict[str, list[_Exported]] = OrderedDict()
        self._metrics = {"ipc_tensors": 0, "ipc_bytes": 0, "fallback_leaves": 0, "rebuilt": 0}

    # ------------------------------------------------------------- producer
    def _sweep(self) -> None:
        for key in list(self._exported.keys()):
            ack = _ack_path(key)
            if os.path.exists(ack):
                try:
                    os.remove(ack)
                except OSError:
                    pass
                self._exported.pop(key, None)
        while len(self._exported) > self.max_live:
            key, _ = self._exported.popitem(last=False)
            logger.warning("CudaIpcConnector: dropping unacked export %s (registry > %d)", key, self.max_live)

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        self._sweep()
        keep: list[_Exported] = []

        def leaf(t: Any) -> Any:
            if isinstance(t, torch.Tensor) and t.is_cuda and self.enabled and t.numel() > 0:
                try:
                    marker, rec = export_tensor(t)
                except Exception as exc:  # any IPC failure -> host path for this leaf
                    logger.debug("CudaIpcConnector: export failed (%s); falling back to shm", exc)
                    self._metrics["fallback_leaves"] += 1
                    return t
                keep.append(rec)
                self._metrics["ipc_tensors"] += 1
                self._metrics["ipc_bytes"] += t.numel() * t.element_size()
                return marker
            return t

        rewritten = _walk(data, leaf)
        ok, size, meta = self._inner.put(from_stage, to_stage, put_key, rewritten)
        if ok and keep:
            self._exported[put_key] = keep
            meta = dict(meta or {})
            meta["cuda_ipc"] = len(keep)
        return ok, size, meta

    # ------------------------------------------------------------- consumer
    def get(
        self, from_stage: str, to_stage: str, get_key: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        result = self._inner.get(from_stage, to_stage, get_key, metadata)
        if result is None:
            return None
        obj, size = result
        rebuilt = 0

        def leaf(m: Any) -> Any:
            nonlocal rebuilt
            if is_marker(m):
                rebuilt += 1
                return rebuild_tensor(m)
            return m

        obj = _walk(obj, leaf)
        if rebuilt:
            self._metrics["rebuilt"] += rebuilt
            try:
                with open(_ack_path(get_key), "w") as f:
                    f.write(str(time.time()))
            except OSError:
                pass
        return obj, size

    # ------------------------------------------------------------- lifecycle
    def cleanup(self, request_id: str) -> None:
        for key in list(self._exported.keys()):
            if (
                key == request_id
                or key.startswith(request_id + "_")
                or key.endswith("_" + request_id)
                or request_id in key
            ):
                self._exported.pop(key, None)
                try:
                    os.remove(_ack_path(key))
                except OSError:
                    pass
        self._inner.cleanup(request_id)

    def health(self) -> dict[str, Any]:
        h = self._inner.health()
        h.update({"cuda_ipc": self.enabled, "live_exports": len(self._exported), **self._metrics})
        return h

    def close(self) -> None:
        self._exported.clear()
        self._inner.close()
