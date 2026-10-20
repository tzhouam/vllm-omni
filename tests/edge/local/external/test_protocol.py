# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The wire format, which has to survive a peer that is not us."""

from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

from vllm_omni.edge.local.external import protocol


def _round_trip(op, body=None, tensors=None):
    near, far = socket.socketpair()
    try:
        threading.Thread(
            target=lambda: protocol.send_message(near, op, body, tensors), daemon=True
        ).start()
        return protocol.recv_message(far)
    finally:
        near.close()
        far.close()


def test_tensors_survive_dtype_and_shape():
    sent = {
        "f32": np.arange(12, dtype=np.float32).reshape(3, 4),
        "i64": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "u16": np.array([7, 8, 9], dtype=np.uint16),
        "scalar": np.array(3.5, dtype=np.float64),
    }
    op, body, got = _round_trip(protocol.OP_RUN, {"n": 1}, sent)
    assert op == protocol.OP_RUN and body == {"n": 1}
    for name, array in sent.items():
        assert got[name].dtype == array.dtype
        assert got[name].shape == array.shape
        assert np.array_equal(got[name], array)


def test_received_tensors_are_writable():
    """``frombuffer`` hands back a read-only view aliasing the socket buffer.

    ORT's IOBinding writes into the arrays it is given, so a read-only result
    would fail at the far end of a working connection rather than here.
    """
    _, _, got = _round_trip(protocol.OP_RUN, {}, {"x": np.ones(4, dtype=np.float32)})
    got["x"][0] = 5.0  # must not raise
    assert got["x"][0] == 5.0


def test_payload_larger_than_one_recv():
    """Stream sockets short-read; a naive recv would misparse the next tensor."""
    big = np.arange(4 * 1024 * 1024, dtype=np.float32)
    _, _, got = _round_trip(protocol.OP_RUN, {}, {"big": big, "after": np.array([1.0])})
    assert np.array_equal(got["big"], big)
    assert np.array_equal(got["after"], np.array([1.0]))


def test_empty_message_and_no_tensors():
    op, body, got = _round_trip(protocol.OP_STATS)
    assert (op, body, got) == (protocol.OP_STATS, {}, {})


def test_error_reply_raises_with_remote_detail():
    with pytest.raises(protocol.WorkerError) as excinfo:
        _round_trip(protocol.OP_ERR, {"message": "boom", "code": "ValueError",
                                      "traceback": "far side traceback"})
    assert excinfo.value.code == "ValueError"
    assert "far side traceback" in excinfo.value.remote_traceback


def test_closed_peer_is_a_protocol_error_not_a_hang():
    near, far = socket.socketpair()
    near.close()
    with pytest.raises(protocol.ProtocolError):
        protocol.recv_message(far)
    far.close()


def test_oversized_header_is_refused_before_allocation():
    """A corrupt length must not become a multi-gigabyte allocation."""
    near, far = socket.socketpair()
    try:
        import struct

        near.sendall(struct.pack(">I", protocol.MAX_HEADER_BYTES + 1))
        with pytest.raises(protocol.ProtocolError, match="over the limit"):
            protocol.recv_message(far)
    finally:
        near.close()
        far.close()
