"""Offline Range-server tests for pinned Agent model downloads."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmarks.edge_agent.download import (
    Artifact,
    ArtifactDownloader,
    DownloadError,
    MAX_RANGE_BYTES,
    load_manifest,
)


@contextmanager
def range_server(data: bytes, *, ignore_range: bool = False,
                 truncate_first: bool = False, max_range_bytes: int | None = None,
                 truncate_to_bytes: int | None = None,
                 wrong_length: bool = False, content_encoding: str | None = None,
                 wrong_content_range: bool = False):
    requests: list[tuple[int, int]] = []
    guard = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            if not match:
                self.send_error(400)
                return
            start, end = map(int, match.groups())
            with guard:
                requests.append((start, end))
                ordinal = len(requests)
            if ignore_range:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if max_range_bytes is not None and end - start + 1 > max_range_bytes:
                self.send_error(416)
                return
            body = data[start:end + 1]
            if truncate_first and ordinal == 1:
                body = body[:len(body) // 2]
            if truncate_to_bytes is not None:
                body = body[:truncate_to_bytes]
            self.send_response(206)
            total = len(data) + (1 if wrong_content_range else 0)
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Content-Length", str(end - start + 1 + int(wrong_length)))
            if content_encoding is not None:
                self.send_header("Content-Encoding", content_encoding)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def artifact(data: bytes) -> Artifact:
    return Artifact("test/model", "a" * 40, "sub/model.gguf", len(data),
                    hashlib.sha256(data).hexdigest())


def test_resume_rehashes_chunks_and_promotes_only_final_sha(tmp_path):
    data = bytes(range(256)) * 14
    item = artifact(data)
    target = tmp_path / "sub" / "model.gguf"
    target.parent.mkdir()
    partial = target.with_name("model.gguf.partial")
    progress = target.with_name("model.gguf.partial.json")
    partial.write_bytes(data[:1024] + b"x" * 1024)
    progress.write_text(json.dumps({**item.identity(1024), "chunks": {
        "0": hashlib.sha256(data[:1024]).hexdigest(),
        "1": hashlib.sha256(data[1024:2048]).hexdigest(),
    }}))
    with range_server(data) as (base_url, requests):
        downloader = ArtifactDownloader(tmp_path, base_url=base_url, chunk_bytes=1024,
                                        free_reserve_bytes=0, workers=3)
        outcome = downloader.download([item])[0]
        assert not outcome.cached
        assert outcome.sha256 == item.sha256
        assert target.read_bytes() == data
        assert (0, 1023) not in requests
        assert (1024, 2047) in requests  # corrupt cached chunk was detected
        assert len(requests) == 3
        assert not partial.exists()
        assert not progress.exists()
        assert downloader.download([item])[0].cached
        assert len(requests) == 3


def test_truncated_range_resumes_exact_tail_without_promoting_partial(tmp_path):
    data = b"a" * 1024 + b"b" * 10
    item = artifact(data)
    with range_server(data, truncate_first=True) as (base_url, requests):
        downloader = ArtifactDownloader(tmp_path, base_url=base_url, chunk_bytes=1024,
                                        free_reserve_bytes=0, workers=1, retries=2)
        assert downloader.download([item])[0].path.read_bytes() == data
        assert requests.count((0, 1023)) == 1
        assert (512, 1023) in requests


def test_large_verified_chunk_uses_bounded_ranges_and_only_missing_tails(tmp_path):
    data = bytes(range(256)) * (2 * MAX_RANGE_BYTES // 256) + b"tail"
    item = artifact(data)
    with range_server(data, truncate_first=True,
                      max_range_bytes=MAX_RANGE_BYTES) as (base_url, requests):
        downloader = ArtifactDownloader(tmp_path, base_url=base_url,
                                        chunk_bytes=16 * 1024 * 1024,
                                        free_reserve_bytes=0, workers=1, retries=2)
        assert downloader.download([item])[0].path.read_bytes() == data
        assert requests == [
            (0, MAX_RANGE_BYTES - 1),
            (MAX_RANGE_BYTES // 2, MAX_RANGE_BYTES - 1),
            (MAX_RANGE_BYTES, 2 * MAX_RANGE_BYTES - 1),
            (2 * MAX_RANGE_BYTES, len(data) - 1),
        ]


def test_truncated_subrange_stops_after_bounded_tail_retries(tmp_path):
    data = b"a" * 1024
    with range_server(data, truncate_to_bytes=1) as (base_url, requests):
        downloader = ArtifactDownloader(tmp_path, base_url=base_url,
                                        chunk_bytes=1024, free_reserve_bytes=0,
                                        workers=1, retries=3)
        with pytest.raises(DownloadError, match="after 3 attempts"):
            downloader.download([artifact(data)])
        assert requests == [(0, 1023), (1, 1023), (2, 1023)]
    assert not (tmp_path / "sub" / "model.gguf").exists()


@pytest.mark.parametrize("server_options, message", [
    ({"wrong_length": True}, "Content-Length"),
    ({"content_encoding": "gzip"}, "encoding"),
    ({"wrong_content_range": True}, "Content-Range"),
])
def test_malformed_subrange_metadata_is_refused(tmp_path, server_options, message):
    data = b"a" * 1024
    with range_server(data, **server_options) as (base_url, requests):
        downloader = ArtifactDownloader(tmp_path, base_url=base_url,
                                        chunk_bytes=1024, free_reserve_bytes=0,
                                        workers=1, retries=3)
        with pytest.raises(DownloadError, match=message):
            downloader.download([artifact(data)])
        assert requests == [(0, 1023)]
    assert not (tmp_path / "sub" / "model.gguf").exists()


def test_unbounded_retry_count_is_refused(tmp_path):
    with pytest.raises(ValueError, match="retries"):
        ArtifactDownloader(tmp_path, retries=9)


def test_ignored_range_and_wrong_global_hash_are_refused(tmp_path):
    data = b"a" * 1024
    item = artifact(data)
    with range_server(data, ignore_range=True) as (base_url, _):
        downloader = ArtifactDownloader(tmp_path, base_url=base_url, chunk_bytes=1024,
                                        free_reserve_bytes=0, workers=1, retries=1)
        with pytest.raises(DownloadError, match="did not honor Range"):
            downloader.download([item])
    assert not (tmp_path / "sub" / "model.gguf").exists()

    with range_server(b"b" * len(data)) as (base_url, _):
        downloader = ArtifactDownloader(tmp_path, base_url=base_url, chunk_bytes=1024,
                                        free_reserve_bytes=0, workers=1, retries=1)
        with pytest.raises(DownloadError, match="whole-file SHA-256 mismatch"):
            downloader.download([item])
    assert not (tmp_path / "sub" / "model.gguf").exists()
    with pytest.raises(DownloadError, match="reset-partial"):
        downloader.preflight([item])
    downloader.reset_partial(item)
    with range_server(data) as (base_url, _):
        recovered = ArtifactDownloader(tmp_path, base_url=base_url, chunk_bytes=1024,
                                       free_reserve_bytes=0, workers=1, retries=1)
        assert recovered.download([item])[0].path.read_bytes() == data


def test_manifest_requires_exact_revision_hash_and_safe_paths(tmp_path):
    data = b"abc"
    item = artifact(data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"repo": item.repo, "revision": item.revision,
                                    "files": [{"filename": item.filename, "size": item.size,
                                               "sha256": item.sha256}]}))
    assert load_manifest(manifest) == [item]
    with pytest.raises(ValueError, match="relative"):
        Artifact(item.repo, item.revision, "../escape", 1, item.sha256)
    with pytest.raises(ValueError, match="commit"):
        Artifact(item.repo, "main", item.filename, item.size, item.sha256)
    with pytest.raises(ValueError, match="duplicate"):
        manifest.write_text(json.dumps({"repo": item.repo, "revision": item.revision,
                                        "files": [{"filename": item.filename, "size": item.size,
                                                   "sha256": item.sha256}] * 2}))
        load_manifest(manifest)


def test_token_is_not_forwarded_to_redirect_host(tmp_path):
    data = b"g" * 1024
    seen_authorization: list[str | None] = []

    class Destination(BaseHTTPRequestHandler):
        def do_GET(self):
            seen_authorization.append(self.headers.get("Authorization"))
            self.send_response(206)
            self.send_header("Content-Range", "bytes 0-1023/1024")
            self.send_header("Content-Length", "1024")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, _format, *_args):
            return

    destination = ThreadingHTTPServer(("127.0.0.1", 0), Destination)

    class Source(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{destination.server_port}/file")
            self.end_headers()

        def log_message(self, _format, *_args):
            return

    source = ThreadingHTTPServer(("127.0.0.1", 0), Source)
    threads = [threading.Thread(target=server.serve_forever, daemon=True)
               for server in (destination, source)]
    for thread in threads:
        thread.start()
    try:
        downloader = ArtifactDownloader(tmp_path, base_url=f"http://localhost:{source.server_port}",
                                        chunk_bytes=1024, free_reserve_bytes=0,
                                        workers=1, retries=1, token="test-token")
        assert downloader.download([artifact(data)])[0].path.read_bytes() == data
        assert seen_authorization == [None]
    finally:
        for server in (destination, source):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
