# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The wire between the engine and a stage that runs in another process.

Deliberately tiny, and deliberately **stdlib + numpy only**. The worker on the
other end of this socket may be a native-Windows interpreter that has no torch,
no vllm and no vllm_omni on its path -- ``C:\\Users\\zhout\\npu-ep`` is exactly
that: Python 3.12, onnxruntime, onnx, numpy, nothing else. Anything imported
here has to be installable there, so this module imports neither.

**Why a socket at all.** Two devices on this laptop cannot be reached from the
engine's own interpreter, and for different reasons:

* the **XDNA2 NPU** is an MCDM device. WSL2's GPU-PV forwards WDDM display
  adapters only, so there is no ``/dev/accel`` inside WSL and no amount of
  installing changes that. VitisAI is a Windows DLL.
* the **Radeon 890M** is reachable from WSL through DirectML, but
  ``torch-directml`` pins torch 2.4.1 and Omni runs 2.13.

So out-of-process is forced, not preferred. One protocol covers both, plus the
native-Windows DirectML route, which needs its own venv anyway because
``onnxruntime-directml`` is pinned at 1.24.4 and cannot load the VitisAI EP
(that needs ORT >= 1.25) -- the two AMD devices cannot share an interpreter.

**Framing.** ``[4-byte big-endian header length][JSON header][raw tensor bytes]``.
Tensors are described in the header (name, dtype, shape, nbytes) and their
buffers are concatenated after it in the order listed, so a receive is two
reads and one ``frombuffer`` per tensor -- no per-element work and no pickle.
Measured on this machine, WSL <-> native Windows: 11 us half round trip,
330 MB/s. That is ample for a stage boundary (a 2 s Code2Wav chunk is ~192 KB,
about 0.6 ms) and nowhere near enough for per-layer traffic inside a decode
loop, which is the arithmetic behind "coarse stages only".

``pickle`` is not used anywhere in here on purpose: the header is JSON, the
payload is raw buffers, so a hostile or merely broken peer cannot execute code
by replying.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

import numpy as np

# Ops the client sends.
OP_LOAD = "load"
"""Open a session on a graph. Replies with the placement report."""
OP_RUN = "run"
"""One forward. Inputs in, outputs out."""
OP_STATS = "stats"
"""Counters and RSS. The worker's memory is invisible to the engine's sampler
when the worker is a Windows process, so it has to report its own."""
OP_CLOSE = "close"

# Ops the worker sends.
OP_HELLO = "hello"
"""First message after connect-back: interpreter, versions, available EPs."""
OP_OK = "ok"
OP_ERR = "err"

MAX_HEADER_BYTES = 1 << 20
"""A header is metadata; a megabyte of it is a bug or an attack, not a graph."""

MAX_PAYLOAD_BYTES = 2 << 30
"""Refuse rather than allocate on a corrupt length. Two gibibytes is far above
any legitimate stage payload here -- the largest is a vision tower's patch
tensor, single-digit MB."""


class ProtocolError(RuntimeError):
    """The peer sent something this protocol cannot represent."""


class WorkerError(RuntimeError):
    """The peer ran and failed. Carries the remote traceback when there is one."""

    def __init__(self, message: str, *, remote_traceback: str = "", code: str = "") -> None:
        # The remote traceback goes in the message, not only in an attribute:
        # a bare "AssertionError:" surfacing here with the actual stack one
        # attribute away is a debugging loop nobody should have to discover.
        detail = f"{message}\n--- worker traceback ---\n{remote_traceback}" if remote_traceback else message
        super().__init__(detail)
        self.message = message
        self.remote_traceback = remote_traceback
        self.code = code


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise.

    ``recv`` is allowed to return short reads on a stream socket and does, once
    payloads get past a page or two; a partial read here would be silently
    misparsed as the next tensor.
    """
    if n == 0:
        return b""
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ProtocolError(f"peer closed with {remaining} of {n} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks) if len(chunks) > 1 else chunks[0]


def send_message(
    sock: socket.socket,
    op: str,
    body: dict[str, Any] | None = None,
    tensors: dict[str, np.ndarray] | None = None,
) -> None:
    """Send one message. ``tensors`` are sent as raw buffers after the header."""
    tensors = tensors or {}
    described: list[dict[str, Any]] = []
    buffers: list[memoryview] = []
    for name, array in tensors.items():
        array = np.asarray(array)
        # Record the shape *before* making it contiguous: ``ascontiguousarray``
        # promotes a 0-d array to shape (1,), which would silently turn a
        # scalar graph input into a length-1 vector on the far side.
        shape = list(array.shape)
        contiguous = np.ascontiguousarray(array)
        described.append(
            {
                "name": name,
                # ``str(dtype)`` loses endianness for multibyte types; ``.str``
                # keeps the '<'/'>' so a big-endian peer would still decode.
                "dtype": contiguous.dtype.str,
                "shape": shape,
                "nbytes": int(contiguous.nbytes),
            }
        )
        # ``reshape(-1)`` rather than a direct cast: ``memoryview.cast`` refuses
        # a 0-dimensional source.
        buffers.append(memoryview(contiguous.reshape(-1)).cast("B"))

    header = json.dumps({"op": op, "body": body or {}, "tensors": described}).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ProtocolError(f"header is {len(header)} bytes, over the {MAX_HEADER_BYTES} limit")
    sock.sendall(struct.pack(">I", len(header)))
    sock.sendall(header)
    for buffer in buffers:
        sock.sendall(buffer)


def recv_message(sock: socket.socket) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
    """Receive one message as ``(op, body, tensors)``.

    Raises :class:`WorkerError` on an ``err`` reply so callers do not each have
    to remember to check; every other op is returned as-is.
    """
    (header_len,) = struct.unpack(">I", _recv_exactly(sock, 4))
    if header_len > MAX_HEADER_BYTES:
        raise ProtocolError(f"peer announced a {header_len}-byte header, over the limit")
    try:
        header = json.loads(_recv_exactly(sock, header_len).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"header is not JSON: {exc}") from exc

    total = sum(int(t["nbytes"]) for t in header.get("tensors", ()))
    if total > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"peer announced a {total}-byte payload, over the limit")

    tensors: dict[str, np.ndarray] = {}
    for spec in header.get("tensors", ()):
        raw = _recv_exactly(sock, int(spec["nbytes"]))
        array = np.frombuffer(raw, dtype=np.dtype(spec["dtype"]))
        # ``frombuffer`` is read-only and aliases ``raw``; a consumer that
        # writes in place (ORT's IOBinding does) needs its own buffer.
        tensors[spec["name"]] = array.reshape(spec["shape"]).copy()

    op = str(header.get("op", ""))
    body = header.get("body") or {}
    if op == OP_ERR:
        raise WorkerError(
            str(body.get("message", "worker failed with no message")),
            remote_traceback=str(body.get("traceback", "")),
            code=str(body.get("code", "")),
        )
    return op, body, tensors
