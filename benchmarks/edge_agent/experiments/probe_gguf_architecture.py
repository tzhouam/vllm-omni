"""Bounded, diagnostic GGUF architecture check before downloading weights.

Only a header range at an immutable Hugging Face commit is requested. A marker
in a local llama DLL is weak evidence of compiled architecture awareness, not
evidence that the model can load, execute, fit, or pass any quality gate.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from benchmarks.edge_agent.download import Artifact, load_manifest


_HERE = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = _HERE / "configs/qwen3_30b_a3b_q4_k_m_download.json"
MAX_HEADER_BYTES = 1024 * 1024
# The architecture key is near the start of the pinned GGUF.  The Hub/CDN
# served this smaller exact range reliably where a 1 MiB range was truncated.
DEFAULT_HEADER_BYTES = 64 * 1024
MAX_DLL_BYTES = 256 * 1024 * 1024
_RANGE = re.compile(r"bytes 0-(\d+)/(\d+)\Z")
_ARCH = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
                 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class ProbeError(ValueError):
    """The bounded observation cannot be trusted."""


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ProbeError("model range redirected away from HTTPS")
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.add_header("Range", request.get_header("Range"))
            redirected.add_header("Accept-Encoding", "identity")
        return redirected


def _artifact(manifest: Path) -> tuple[Artifact, str]:
    manifest_bytes = manifest.read_bytes()
    artifacts = load_manifest(manifest)
    if len(artifacts) != 1 or not artifacts[0].filename.lower().endswith(".gguf"):
        raise ProbeError("probe needs a one-file pinned GGUF manifest")
    return artifacts[0], hashlib.sha256(manifest_bytes).hexdigest()


def _url(artifact: Artifact) -> str:
    repo = urllib.parse.quote(artifact.repo, safe="/")
    filename = urllib.parse.quote(artifact.filename, safe="/")
    return (f"https://huggingface.co/{repo}/resolve/"
            f"{artifact.revision.lower()}/{filename}")


def fetch_header(
    artifact: Artifact, *, limit: int = DEFAULT_HEADER_BYTES,
    open_range: Callable[[urllib.request.Request], Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Require an exact 206 response before reading at most ``limit`` bytes.

    ``open_range`` is only for offline fixture transport in tests. Production
    uses the fixed HTTPS Hub URL and an HTTPS-only redirect handler.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 64 <= limit <= MAX_HEADER_BYTES:
        raise ProbeError(f"header byte limit must be 64..{MAX_HEADER_BYTES}")
    requested = min(limit, artifact.size)
    request = urllib.request.Request(_url(artifact), headers={
        "Range": f"bytes=0-{requested - 1}",
        "Accept-Encoding": "identity",
        "User-Agent": "omni-agent-gguf-architecture-probe/1",
    })
    if open_range is None:
        opener = urllib.request.build_opener(_HttpsOnlyRedirect())
        open_range = lambda req: opener.open(req, timeout=30)
    with open_range(request) as response:
        if response.status != 206:
            raise ProbeError(f"range was not honored: HTTP {response.status}")
        match = _RANGE.fullmatch(response.headers.get("Content-Range", ""))
        if not match or int(match[1]) + 1 != requested or int(match[2]) != artifact.size:
            raise ProbeError("Content-Range differs from the pinned file and requested range")
        if response.headers.get("Content-Length") != str(requested):
            raise ProbeError("Content-Length differs from requested range")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ProbeError("compressed range response cannot be parsed safely")
        chunks = []
        remaining = requested
        while remaining:
            try:
                chunk = response.read(min(64 * 1024, remaining))
            except http.client.IncompleteRead as exc:
                raise ProbeError("truncated header range response") from exc
            if not chunk:
                raise ProbeError("truncated header range response")
            if len(chunk) > remaining:
                raise ProbeError("range response exceeded requested byte count")
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b"".join(chunks)
    if len(body) != requested:
        raise ProbeError("truncated header range response")
    return body, {
        "url": _url(artifact), "http_status": 206,
        "range_limit_bytes": limit, "range_requested_bytes": requested,
        "range_received_bytes": len(body),
        "range_sha256": hashlib.sha256(body).hexdigest(),
    }


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def take(self, size: int) -> bytes:
        if size < 0 or size > len(self.data) - self.offset:
            raise ProbeError("GGUF metadata exceeds bounded header range")
        start = self.offset
        self.offset += size
        return self.data[start:self.offset]

    def integer(self, width: int) -> int:
        return int.from_bytes(self.take(width), "little")

    def string(self) -> str:
        length = self.integer(8)
        if length > MAX_HEADER_BYTES:
            raise ProbeError("GGUF metadata string is too large")
        try:
            return self.take(length).decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise ProbeError("invalid UTF-8 in GGUF metadata") from exc


def _skip_value(reader: _Reader, kind: int) -> None:
    if kind in _SCALAR_SIZES:
        reader.take(_SCALAR_SIZES[kind])
    elif kind == 8:
        reader.string()
    elif kind == 9:
        element_kind, count = reader.integer(4), reader.integer(8)
        if count > MAX_HEADER_BYTES or element_kind == 9:
            raise ProbeError("unsupported or oversized GGUF metadata array")
        if element_kind in _SCALAR_SIZES:
            reader.take(count * _SCALAR_SIZES[element_kind])
        elif element_kind == 8:
            for _ in range(count):
                reader.string()
        else:
            raise ProbeError("unknown GGUF array element type")
    else:
        raise ProbeError("unknown GGUF metadata value type")


def parse_architecture(header: bytes) -> dict[str, Any]:
    """Parse bounded metadata through the first architecture field only."""
    reader = _Reader(header)
    if reader.take(4) != b"GGUF":
        raise ProbeError("GGUF magic missing")
    version = reader.integer(4)
    if version not in (2, 3):
        raise ProbeError(f"unsupported GGUF metadata version {version}")
    tensor_count, metadata_count = reader.integer(8), reader.integer(8)
    if metadata_count > MAX_HEADER_BYTES // 12:
        raise ProbeError("GGUF metadata count exceeds header bound")
    architecture: str | None = None
    entries_scanned = 0
    for _ in range(metadata_count):
        key = reader.string()
        if len(key) > 1024:
            raise ProbeError("GGUF metadata key is too large")
        kind = reader.integer(4)
        entries_scanned += 1
        if key == "general.architecture":
            if kind != 8:
                raise ProbeError("non-string GGUF architecture")
            architecture = reader.string()
            if not _ARCH.fullmatch(architecture):
                raise ProbeError("invalid GGUF architecture identifier")
            break
        else:
            _skip_value(reader, kind)
    if architecture is None:
        raise ProbeError("general.architecture is absent from bounded GGUF metadata")
    return {
        "gguf_version": version, "tensor_count_declared": tensor_count,
        "metadata_entries_declared": metadata_count,
        "metadata_entries_scanned": entries_scanned,
        "architecture_field_end_offset": reader.offset,
        "remaining_metadata_unchecked": metadata_count - entries_scanned,
        "general_architecture": architecture,
    }


def scan_runtime_marker(dll_path: Path, architecture: str) -> dict[str, Any]:
    """Hash a local PE DLL and find an exact ASCII architecture token."""
    if dll_path.suffix.lower() != ".dll" or not dll_path.is_file():
        raise ProbeError("local runtime path must be an existing DLL")
    size = dll_path.stat().st_size
    if not 2 <= size <= MAX_DLL_BYTES:
        raise ProbeError("local DLL size exceeds the diagnostic bound")
    pattern = re.compile(rb"(?<![A-Za-z0-9_])" + architecture.encode("ascii") +
                         rb"(?![A-Za-z0-9_])")
    digest = hashlib.sha256()
    offset: int | None = None
    tail = b""
    seen = 0
    with dll_path.open("rb") as handle:
        if handle.read(2) != b"MZ":
            raise ProbeError("local DLL does not have PE magic")
        handle.seek(0)
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            window = tail + chunk
            if offset is None:
                for match in pattern.finditer(window):
                    if match.end() < len(window):
                        offset = seen - len(tail) + match.start()
                        break
            seen += len(chunk)
            tail = window[-(len(architecture) + 1):]
    if offset is None:
        match = pattern.search(tail)
        if match:
            offset = seen - len(tail) + match.start()
    if seen != size:
        raise ProbeError("local DLL changed during marker scan")
    return {
        "dll_path": str(dll_path.resolve()), "dll_size_bytes": size,
        "dll_sha256": digest.hexdigest(),
        "architecture_marker_found": offset is not None,
        "architecture_marker_offset": offset,
    }


def probe_architecture(
    manifest_path: Path, dll_path: Path, *, limit: int = DEFAULT_HEADER_BYTES,
    open_range: Callable[[urllib.request.Request], Any] | None = None,
) -> dict[str, Any]:
    artifact, manifest_sha256 = _artifact(manifest_path)
    header, range_record = fetch_header(artifact, limit=limit, open_range=open_range)
    parsed = parse_architecture(header)
    runtime = scan_runtime_marker(dll_path, parsed["general_architecture"])
    record = {
        "schema": "omni-agent-gguf-architecture-probe-v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "artifact": {
            "repo": artifact.repo, "revision": artifact.revision.lower(),
            "filename": artifact.filename, "published_size_bytes": artifact.size,
            "published_lfs_sha256": artifact.sha256.lower(),
            "whole_artifact_sha256_verified": False,
        },
        "header_range": range_record,
        "gguf_metadata": parsed,
        "runtime_marker": runtime,
        "outcome": ("architecture_marker_observed_diagnostic_only"
                    if runtime["architecture_marker_found"]
                    else "architecture_marker_absent_diagnostic_only"),
        "full_model_load_verified": False,
        "inference_verified": False,
        "qualification": False,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--llama-dll", required=True, type=Path)
    parser.add_argument("--header-bytes", type=int, default=DEFAULT_HEADER_BYTES)
    parser.add_argument("--output", type=Path, help="optional new JSON evidence path")
    args = parser.parse_args(argv)
    try:
        record = probe_architecture(args.manifest, args.llama_dll,
                                    limit=args.header_bytes)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
