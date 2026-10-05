"""Pinned, resumable downloads for edge Agent model artifacts.

The caller supplies an immutable Hugging Face commit, byte size, and LFS
SHA-256 for every file.  A completed download is never accepted by name or
size alone.  Partial files retain per-chunk hashes so a restart can verify
what is already on disk before requesting the missing ranges.

This is artifact transport only.  It does not imply that a model can load,
execute, fit the memory admission ledger, or pass task-quality gates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable


DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024
DEFAULT_FREE_RESERVE_BYTES = 2 * 1024 * 1024 * 1024
_HEX40 = re.compile(r"[0-9a-fA-F]{40}\Z")
_HEX64 = re.compile(r"[0-9a-fA-F]{64}\Z")
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)\Z")
_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


class DownloadError(RuntimeError):
    """Artifact is unavailable, unverified, or cannot safely be resumed."""


@dataclass(frozen=True)
class Artifact:
    repo: str
    revision: str
    filename: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.repo, str) or not _REPO.fullmatch(self.repo)
                or ".." in self.repo.split("/")):
            raise ValueError("repo must be a Hugging Face owner/name")
        if not isinstance(self.revision, str) or not _HEX40.fullmatch(self.revision):
            raise ValueError("revision must be an exact 40-character commit SHA")
        if not isinstance(self.filename, str):
            raise ValueError("filename must be a safe relative repository path")
        path = PurePosixPath(self.filename)
        if (not self.filename or path.is_absolute() or "\\" in self.filename
                or any(part in {"", ".", ".."} for part in self.filename.split("/"))
                or ":" in self.filename):
            raise ValueError("filename must be a safe relative repository path")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("size must be a positive byte count")
        if not isinstance(self.sha256, str) or not _HEX64.fullmatch(self.sha256):
            raise ValueError("sha256 must be a 64-character hex digest")

    def identity(self, chunk_bytes: int) -> dict[str, Any]:
        return {
            "schema": 1,
            "repo": self.repo,
            "revision": self.revision.lower(),
            "filename": self.filename,
            "size": self.size,
            "sha256": self.sha256.lower(),
            "chunk_bytes": chunk_bytes,
        }


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size: int
    sha256: str
    cached: bool


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward a Hub token to a signed CDN on another host."""

    def redirect_request(self, request: urllib.request.Request, fp: Any,
                         code: int, msg: str, headers: Any,
                         newurl: str) -> urllib.request.Request | None:
        old = urllib.parse.urlparse(request.full_url)
        new = urllib.parse.urlparse(newurl)
        if old.scheme == "https" and new.scheme != "https":
            raise DownloadError("HTTPS download redirected to an insecure URL")
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None and old.netloc != new.netloc:
            redirected.remove_header("Authorization")
            redirected.headers.pop("Authorization", None)
            redirected.unredirected_hdrs.pop("Authorization", None)
        return redirected


@contextlib.contextmanager
def _exclusive_file_lock(path: Path):
    """An OS-owned lock; process termination releases it on both platforms."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.read(1) == b"":
                handle.seek(0)
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise DownloadError(f"artifact is locked by another downloader: {path}") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise DownloadError(f"artifact is locked by another downloader: {path}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sha256_file(path: Path, *, start: int = 0, size: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = size
        while remaining is None or remaining > 0:
            block = stream.read(min(4 * 1024 * 1024, remaining) if remaining is not None
                                else 4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining is not None and remaining != 0:
        raise DownloadError(f"partial file is too short: {path}")
    return digest.hexdigest()


def _chunk_bounds(index: int, artifact: Artifact, chunk_bytes: int) -> tuple[int, int]:
    start = index * chunk_bytes
    return start, min(artifact.size, start + chunk_bytes) - 1


def _chunk_count(artifact: Artifact, chunk_bytes: int) -> int:
    return (artifact.size + chunk_bytes - 1) // chunk_bytes


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _load_progress(path: Path, artifact: Artifact, chunk_bytes: int,
                   partial: Path) -> dict[str, Any]:
    identity = artifact.identity(chunk_bytes)
    if not path.exists():
        if partial.exists():
            raise DownloadError(
                f"orphan partial file lacks its verification manifest: {partial}; "
                "move it away before retrying")
        return {**identity, "chunks": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DownloadError(f"invalid partial verification manifest: {path}") from exc
    if not isinstance(data, dict) or any(data.get(key) != value for key, value in identity.items()):
        raise DownloadError(f"partial file belongs to a different artifact or chunk size: {path}")
    chunks = data.get("chunks")
    if not isinstance(chunks, dict):
        raise DownloadError(f"invalid chunk metadata: {path}")
    if data.get("whole_sha256_mismatch"):
        raise DownloadError(
            f"partial failed the pinned whole-file SHA-256: {path}; "
            "inspect the recorded digest, then use --reset-partial to fetch it again")
    verified: dict[str, str] = {}
    for key, digest in chunks.items():
        if not str(key).isdigit() or not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise DownloadError(f"invalid chunk record in {path}")
        index = int(key)
        if index < 0 or index >= _chunk_count(artifact, chunk_bytes):
            raise DownloadError(f"out-of-range chunk record in {path}")
        if not partial.exists():
            continue
        start, end = _chunk_bounds(index, artifact, chunk_bytes)
        try:
            actual = _sha256_file(partial, start=start, size=end - start + 1)
        except DownloadError:
            continue
        if actual == digest.lower():
            verified[key] = digest.lower()
    return {**identity, "chunks": verified}


def load_manifest(path: Path) -> list[Artifact]:
    """Read a user-reviewed list of pinned Hugging Face LFS files."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    repo, revision = data.get("repo"), data.get("revision")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest needs a nonempty files array")
    if any(not isinstance(entry, dict) or not {"filename", "size", "sha256"} <= entry.keys()
           for entry in files):
        raise ValueError("every file needs filename, size and sha256")
    artifacts = [Artifact(repo, revision, entry["filename"], entry["size"],
                          entry["sha256"]) for entry in files]
    if len({item.filename for item in artifacts}) != len(artifacts):
        raise ValueError("manifest contains duplicate filenames")
    return artifacts


class ArtifactDownloader:
    def __init__(self, destination: Path, *, workers: int = 4,
                 chunk_bytes: int = DEFAULT_CHUNK_BYTES,
                 free_reserve_bytes: int = DEFAULT_FREE_RESERVE_BYTES,
                 timeout_seconds: float = 90, retries: int = 4,
                 token: str | None = None,
                 base_url: str = "https://huggingface.co") -> None:
        if workers < 1 or workers > 32:
            raise ValueError("workers must be 1..32")
        if chunk_bytes < 1024 or chunk_bytes > 256 * 1024 * 1024:
            raise ValueError("chunk_bytes must be 1 KiB..256 MiB")
        if workers * chunk_bytes > 512 * 1024 * 1024:
            raise ValueError("workers × chunk_bytes exceeds 512 MiB in-flight limit")
        if free_reserve_bytes < 0 or timeout_seconds <= 0 or retries < 1:
            raise ValueError("invalid reserve, timeout, or retries")
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and
                                                parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
            raise ValueError("base_url must be HTTPS or a local test server")
        self.destination = Path(destination)
        self.workers = workers
        self.chunk_bytes = chunk_bytes
        self.free_reserve_bytes = free_reserve_bytes
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.token = token
        self.base_url = base_url.rstrip("/")
        self._opener = urllib.request.build_opener(_SafeRedirect())

    def _paths(self, artifact: Artifact) -> tuple[Path, Path, Path, Path]:
        target = self.destination.joinpath(*artifact.filename.split("/"))
        return (target, target.with_name(target.name + ".partial"),
                target.with_name(target.name + ".partial.json"),
                target.with_name(target.name + ".download.lock"))

    def _url(self, artifact: Artifact) -> str:
        filename = urllib.parse.quote(artifact.filename, safe="/")
        return f"{self.base_url}/{artifact.repo}/resolve/{artifact.revision}/{filename}?download=true"

    def preflight(self, artifacts: Iterable[Artifact]) -> dict[str, int]:
        """Verify existing files/chunks and check remaining on-disk capacity."""
        items = list(artifacts)
        self.destination.mkdir(parents=True, exist_ok=True)
        remaining = 0
        already_verified = 0
        for artifact in items:
            target, partial, progress_path, _ = self._paths(artifact)
            if target.exists():
                if target.stat().st_size != artifact.size or _sha256_file(target) != artifact.sha256.lower():
                    raise DownloadError(f"existing file differs from pinned artifact: {target}")
                already_verified += artifact.size
                continue
            progress = _load_progress(progress_path, artifact, self.chunk_bytes, partial)
            verified = {int(key) for key in progress["chunks"]}
            for index in range(_chunk_count(artifact, self.chunk_bytes)):
                start, end = _chunk_bounds(index, artifact, self.chunk_bytes)
                if index in verified:
                    already_verified += end - start + 1
                else:
                    remaining += end - start + 1
        free = shutil.disk_usage(self.destination).free
        required = remaining + self.free_reserve_bytes if remaining else 0
        if free < required:
            raise DownloadError(
                f"insufficient disk space: need {required} free bytes "
                f"({remaining} download + {self.free_reserve_bytes} reserve), have {free}")
        return {"remaining_bytes": remaining, "verified_bytes": already_verified,
                "free_bytes": free, "required_free_bytes": required}

    def reset_partial(self, artifact: Artifact) -> None:
        """Explicitly discard untrusted partial bytes; never remove a final artifact."""
        target, partial, progress_path, lock_path = self._paths(artifact)
        with _exclusive_file_lock(lock_path):
            if target.exists():
                raise DownloadError(f"refusing to reset an existing final artifact: {target}")
            partial.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)

    def _fetch_chunk(self, artifact: Artifact, index: int) -> bytes:
        start, end = _chunk_bounds(index, artifact, self.chunk_bytes)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                position = start
                pieces: list[bytes] = []
                # CDNs and local proxies can close a valid 206 response a few
                # bytes early. Resume only that missing tail with another
                # exact Range; the whole pinned LFS SHA is still mandatory.
                for _segment in range(16):
                    headers = {"Range": f"bytes={position}-{end}",
                               "Accept-Encoding": "identity",
                               "User-Agent": "omni-edge-agent-artifact-downloader/1"}
                    if self.token:
                        headers["Authorization"] = f"Bearer {self.token}"
                    request = urllib.request.Request(self._url(artifact), headers=headers, method="GET")
                    with self._opener.open(request, timeout=self.timeout_seconds) as response:
                        if response.status != 206:
                            raise DownloadError(
                                f"server did not honor Range for {artifact.filename} chunk {index} "
                                f"(HTTP {response.status})")
                        match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                        if not match or tuple(map(int, match.groups())) != (position, end, artifact.size):
                            raise DownloadError(
                                f"unexpected Content-Range for {artifact.filename} chunk {index}")
                        expected = end - position + 1
                        try:
                            body = response.read(expected + 1)
                        except http.client.IncompleteRead as exc:
                            body = exc.partial
                        if not 0 < len(body) <= expected:
                            raise DownloadError(
                                f"invalid Range body for {artifact.filename} chunk {index}: "
                                f"{len(body)}/{expected} bytes")
                        pieces.append(body)
                        position += len(body)
                        if position == end + 1:
                            return b"".join(pieces)
                raise DownloadError(
                    f"incomplete Range for {artifact.filename} chunk {index}: "
                    f"{position - start}/{end - start + 1} bytes after 16 segments")
            except urllib.error.HTTPError as exc:
                if exc.code in {400, 401, 403, 404, 405, 410, 416}:
                    raise DownloadError(
                        f"HTTP {exc.code} for pinned artifact {artifact.filename} "
                        f"at commit {artifact.revision}") from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError,
                    DownloadError) as exc:
                last_error = exc
            if attempt + 1 < self.retries:
                time.sleep(min(2 ** attempt, 8))
        raise DownloadError(
            f"failed to fetch {artifact.filename} chunk {index} after {self.retries} attempts: "
            f"{last_error}") from last_error

    def download_one(self, artifact: Artifact,
                     progress_callback: Callable[[int, int], None] | None = None) -> DownloadResult:
        target, partial, progress_path, lock_path = self._paths(artifact)
        target.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_file_lock(lock_path):
            if target.exists():
                if target.stat().st_size != artifact.size or _sha256_file(target) != artifact.sha256.lower():
                    raise DownloadError(f"existing file differs from pinned artifact: {target}")
                return DownloadResult(target, artifact.size, artifact.sha256.lower(), True)
            progress = _load_progress(progress_path, artifact, self.chunk_bytes, partial)
            _atomic_json(progress_path, progress)
            completed: dict[str, str] = progress["chunks"]
            missing = [index for index in range(_chunk_count(artifact, self.chunk_bytes))
                       if str(index) not in completed]
            if not partial.exists():
                partial.touch()
            update_lock = threading.Lock()

            def receive(index: int) -> None:
                body = self._fetch_chunk(artifact, index)
                start, _ = _chunk_bounds(index, artifact, self.chunk_bytes)
                digest = hashlib.sha256(body).hexdigest()
                # Separate handles permit non-overlapping writes on Windows and Linux.
                with partial.open("r+b", buffering=0) as stream:
                    stream.seek(start)
                    stream.write(body)
                    stream.flush()
                    os.fsync(stream.fileno())
                with update_lock:
                    completed[str(index)] = digest
                    _atomic_json(progress_path, progress)

            if missing:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                    futures = {pool.submit(receive, index): index for index in missing}
                    try:
                        for future in concurrent.futures.as_completed(futures):
                            future.result()
                            if progress_callback is not None:
                                progress_callback(len(completed), _chunk_count(artifact,
                                                                             self.chunk_bytes))
                    except Exception:
                        for future in futures:
                            future.cancel()
                        raise
            if partial.stat().st_size != artifact.size:
                raise DownloadError(f"partial file has wrong final byte count: {partial}")
            actual = _sha256_file(partial)
            if actual != artifact.sha256.lower():
                # Do not promote a file assembled from individually transport-verified
                # chunks if its pinned LFS digest does not match.
                progress["whole_sha256_mismatch"] = actual
                _atomic_json(progress_path, progress)
                raise DownloadError(
                    f"whole-file SHA-256 mismatch for {artifact.filename}: "
                    f"expected {artifact.sha256.lower()}, got {actual}; "
                    "partial retained for audit (use --reset-partial to retry)")
            os.replace(partial, target)
            progress_path.unlink(missing_ok=True)
            return DownloadResult(target, artifact.size, actual, False)

    def download(self, artifacts: Iterable[Artifact],
                 progress_callback: Callable[[Artifact, int, int], None] | None = None,
                 ) -> list[DownloadResult]:
        items = list(artifacts)
        self.preflight(items)
        return [self.download_one(item, (lambda done, total, artifact=item:
                                         progress_callback(artifact, done, total))
                                  if progress_callback is not None else None)
                for item in items]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="JSON with repo, 40-hex revision and files (filename, size, sha256)")
    parser.add_argument("--dest", required=True, type=Path,
                        help="destination directory; verified files retain repository-relative names")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--check", action="store_true",
                        help="verify existing bytes and disk capacity without downloading")
    parser.add_argument("--reset-partial", action="store_true",
                        help="explicitly discard partial bytes and their verification records")
    args = parser.parse_args(argv)
    if args.check and args.reset_partial:
        parser.error("--check and --reset-partial cannot be combined")
    try:
        artifacts = load_manifest(args.manifest)
        downloader = ArtifactDownloader(
            args.dest, workers=args.workers, chunk_bytes=args.chunk_mib * 1024 * 1024,
            free_reserve_bytes=int(args.reserve_gib * 1024 ** 3),
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))
        if args.reset_partial:
            for artifact in artifacts:
                downloader.reset_partial(artifact)
        capacity = downloader.preflight(artifacts)
        print(json.dumps({"manifest": str(args.manifest), "destination": str(args.dest),
                          "capacity": capacity, "artifacts": [artifact.filename for artifact in artifacts]},
                         indent=2))
        if args.check:
            return 0
        last_log = 0.0

        def report(artifact: Artifact, done: int, total: int) -> None:
            nonlocal last_log
            now = time.monotonic()
            if now - last_log >= 5 or done == total:
                print(f"{artifact.filename}: {done}/{total} chunks verified", flush=True)
                last_log = now

        for result in downloader.download(artifacts, progress_callback=report):
            print(json.dumps({"path": str(result.path), "bytes": result.size,
                              "sha256": result.sha256, "cached": result.cached}))
        return 0
    except (DownloadError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"download refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
