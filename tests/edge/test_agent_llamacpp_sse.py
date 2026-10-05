"""The Agent must see real ordered llama.cpp deltas, not a stage sum."""

from __future__ import annotations

import io
import json

import pytest

from vllm_omni.engine.backends.llamacpp import _stream_json_request


def _sse(*chunks):
    rows = []
    for chunk in chunks:
        rows.append(b"data: " + json.dumps(chunk).encode() + b"\n\n")
    rows.append(b"data: [DONE]\n\n")
    return io.BytesIO(b"".join(rows))


def test_sse_reassembles_and_emits_in_order(monkeypatch):
    stream = _sse(
        {"choices": [{"delta": {"content": "hel"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: stream)
    deltas = []
    result = _stream_json_request("http://127.0.0.1/test", {}, timeout=1, limit=4096,
                                  on_delta=deltas.append)
    assert deltas == ["hel", "lo"]
    assert result["content"] == "hello"
    assert result["finish_reason"] == "stop"


def test_sse_rejects_incomplete_or_oversized_output(monkeypatch):
    stream = io.BytesIO(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: stream)
    with pytest.raises(RuntimeError, match="incomplete"):
        _stream_json_request("http://127.0.0.1/test", {}, timeout=1, limit=4096,
                             on_delta=lambda _: None)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _sse(
        {"choices": [{"delta": {"content": "long"}, "finish_reason": "stop"}]}
    ))
    with pytest.raises(Exception, match="exceeds admitted"):
        _stream_json_request("http://127.0.0.1/test", {}, timeout=1, limit=8,
                             on_delta=lambda _: None)
