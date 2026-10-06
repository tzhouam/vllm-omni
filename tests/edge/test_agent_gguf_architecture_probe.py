"""Offline HTTP-range and parser tests for the pre-download GGUF probe."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmarks.edge_agent.experiments.probe_gguf_architecture import (
    ProbeError,
    parse_architecture,
    probe_architecture,
    scan_runtime_marker,
)


def _text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf() -> bytes:
    entries = [
        _text("general.name") + struct.pack("<I", 8) + _text("fixture"),
        _text("fixture.numbers") + struct.pack("<IIQII", 9, 4, 2, 7, 8),
        _text("general.architecture") + struct.pack("<I", 8) + _text("qwen3moe"),
        _text("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 1000000),
    ]
    return b"GGUF" + struct.pack("<IQQ", 3, 1, len(entries)) + b"".join(entries)


@contextmanager
def _range_server(body: bytes, *, mode: str = "range"):
    observed: dict[str, object] = {"requests": [], "body_bytes_sent": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested = self.headers.get("Range", "")
            observed["requests"].append((self.path, requested))
            match = re.fullmatch(r"bytes=0-(\d+)", requested)
            if not match:
                self.send_error(400)
                return
            count = int(match[1]) + 1
            if mode == "ignored":
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return
            self.send_response(206)
            total = len(body) + (1 if mode == "wrong_total" else 0)
            self.send_header("Content-Range", f"bytes 0-{count - 1}/{total}")
            self.send_header("Content-Length", str(count))
            self.end_headers()
            payload = body[:count]
            if mode == "truncated":
                payload = payload[:-1]
            self.wfile.write(payload)
            observed["body_bytes_sent"] += len(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, observed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _fixture(tmp_path):
    body = _gguf().ljust(2048, b"\0")
    manifest = tmp_path / "pinned.json"
    revision = "a" * 40
    manifest.write_text(json.dumps({
        "repo": "Qwen/Qwen3-30B-A3B-GGUF", "revision": revision,
        "files": [{"filename": "Qwen3-30B-A3B-Q4_K_M.gguf", "size": len(body),
                   "sha256": hashlib.sha256(body).hexdigest()}],
    }), encoding="utf-8")
    dll = tmp_path / "llama.dll"
    dll.write_bytes(b"MZ" + b"\0" * 21 + b"qwen3moe\0")
    return body, manifest, dll, revision


def _local_opener(server, revision):
    def open_range(request):
        assert request.full_url == (
            "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/"
            f"{revision}/Qwen3-30B-A3B-Q4_K_M.gguf"
        )
        url = f"http://127.0.0.1:{server.server_port}/fixture.gguf"
        forwarded = urllib.request.Request(url, headers=dict(request.header_items()))
        return urllib.request.urlopen(forwarded, timeout=2)
    return open_range


def test_probe_observes_pinned_range_and_local_marker_without_full_download(tmp_path):
    body, manifest, dll, revision = _fixture(tmp_path)
    with _range_server(body) as (server, observed):
        record = probe_architecture(manifest, dll, limit=256,
                                    open_range=_local_opener(server, revision))
    assert observed["requests"] == [("/fixture.gguf", "bytes=0-255")]
    assert observed["body_bytes_sent"] == 256 < len(body)
    assert record["header_range"]["range_sha256"] == hashlib.sha256(body[:256]).hexdigest()
    assert record["artifact"]["revision"] == revision
    assert record["artifact"]["whole_artifact_sha256_verified"] is False
    assert record["gguf_metadata"]["general_architecture"] == "qwen3moe"
    assert record["gguf_metadata"]["remaining_metadata_unchecked"] == 1
    assert record["runtime_marker"]["architecture_marker_found"] is True
    assert record["outcome"] == "architecture_marker_observed_diagnostic_only"
    assert record["qualification"] is False
    assert record["full_model_load_verified"] is False


@pytest.mark.parametrize("mode,expected", [
    ("ignored", "range was not honored"),
    ("wrong_total", "Content-Range differs"),
    ("truncated", "truncated header range"),
])
def test_probe_rejects_bad_range_response_without_claiming_success(tmp_path, mode, expected):
    body, manifest, dll, revision = _fixture(tmp_path)
    with _range_server(body, mode=mode) as (server, observed):
        with pytest.raises(ProbeError, match=expected):
            probe_architecture(manifest, dll, limit=256,
                               open_range=_local_opener(server, revision))
    assert observed["requests"] == [("/fixture.gguf", "bytes=0-255")]


def test_parser_rejects_truncation_bad_magic_and_nonstring_architecture():
    header = _gguf()
    assert parse_architecture(header)["general_architecture"] == "qwen3moe"
    with pytest.raises(ProbeError, match="magic"):
        parse_architecture(b"BAD!" + header[4:])
    with pytest.raises(ProbeError, match="exceeds bounded"):
        parse_architecture(header[:50])
    bad = header.replace(struct.pack("<I", 8) + _text("qwen3moe"),
                         struct.pack("<I", 4) + _text("qwen3moe"), 1)
    with pytest.raises(ProbeError, match="non-string"):
        parse_architecture(bad)


def test_marker_token_boundary_and_missing_marker_are_diagnostic(tmp_path):
    dll = tmp_path / "llama.dll"
    dll.write_bytes(b"MZ" + b"x" * (1024 * 1024 - 2 - len(b"qwen3moe")) +
                    b"qwen3moe" + b"x")
    result = scan_runtime_marker(dll, "qwen3moe")
    assert result["architecture_marker_found"] is False
    dll.write_bytes(b"MZ\0qwen3moe\0")
    assert scan_runtime_marker(dll, "qwen3moe")["architecture_marker_found"] is True
