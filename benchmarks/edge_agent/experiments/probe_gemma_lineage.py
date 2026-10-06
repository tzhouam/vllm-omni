"""Read-only, pinned Gemma GGUF/projector lineage observation.

The probe reads a small Hugging Face metadata JSON and tree JSON at the exact
artifact commit, then hashes local files against the pinned download manifest
and the tree's LFS OIDs. It never requests model bytes, reads auth tokens, or
promotes a route. Only its private, hash-bound observation is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from benchmarks.edge_agent.download import Artifact, load_manifest


_HERE = Path(__file__).resolve().parents[1]
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_API_PATH = re.compile(
    r"/api/models/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/(?:revision|tree)/[0-9a-f]{40}\Z"
)
_MAX_METADATA_BYTES = 512 * 1024
_FETCH = Callable[[str], tuple[Any, str]]


class ProbeError(ValueError):
    """Pinned lineage inputs or remote metadata are inconsistent."""


class _HubOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
            raise ProbeError("metadata redirect left the pinned Hub origin")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def record_digest(payload: dict[str, Any]) -> str:
    """Hash the entire raw record except its self-check field."""
    return hashlib.sha256(_canonical({k: v for k, v in payload.items()
                                      if k != "record_sha256"})).hexdigest()


def verify_record(payload: dict[str, Any]) -> bool:
    digest = payload.get("record_sha256")
    return isinstance(digest, str) and bool(_HEX64.fullmatch(digest)) and digest == record_digest(payload)


def _fetch_json(url: str) -> tuple[Any, str]:
    """Fetch bounded public metadata only; never send Authorization."""
    parsed = urllib.parse.urlparse(url)
    query = "recursive=true&expand=true" if "/tree/" in parsed.path else ""
    if (parsed.scheme != "https" or parsed.netloc != "huggingface.co" or
        not _API_PATH.fullmatch(parsed.path) or parsed.query != query):
        raise ProbeError("metadata URL is outside the bounded pinned Hub API")
    request = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": "omni-agent-lineage-probe/1",
    })
    opener = urllib.request.build_opener(_HubOnlyRedirect())
    with opener.open(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > _MAX_METADATA_BYTES:
            raise ProbeError("metadata response exceeds the byte limit")
        body = response.read(_MAX_METADATA_BYTES + 1)
    if len(body) > _MAX_METADATA_BYTES:
        raise ProbeError("metadata response exceeds the byte limit")
    return json.loads(body), hashlib.sha256(body).hexdigest()


def _hash_local(path: Path) -> tuple[int, str]:
    """Hash a stable regular file without writing to it."""
    before = path.stat()
    if not path.is_file():
        raise ProbeError("local artifact is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ProbeError("local artifact changed while hashing")
    return before.st_size, digest.hexdigest()


def _atomic_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        if os.name != "nt":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link publishes the fully synced temporary file without
            # replacing an earlier observation, atomically where supported.
            os.link(name, path)
        except FileExistsError:
            raise
        except OSError:
            # Some Windows shares do not support hard links. Exclusive create
            # still prevents overwrites; a failed copy is removed so its
            # incomplete bytes cannot masquerade as a finished observation.
            created = False
            try:
                with path.open("xb") as output:
                    created = True
                    output.write(Path(name).read_bytes())
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException:
                if created:
                    path.unlink(missing_ok=True)
                raise
    finally:
        Path(name).unlink(missing_ok=True)


def _source_metadata(path: Path, artifacts: list[Artifact]) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("schema") != "omni-agent-lineage-source-v1":
        raise ProbeError("invalid source metadata schema")
    first = artifacts[0]
    if (source.get("repo") != first.repo or source.get("revision") != first.revision or
        source.get("source_repo") != f"https://huggingface.co/{first.repo}" or
        not isinstance(source.get("license"), str) or
        not source["license"] or source["license"].lower() in {"unknown", "unverified"}):
        raise ProbeError("source metadata does not pin the manifest repo, revision and license")
    return source


def _route(config_path: Path, route_id: str | None) -> tuple[dict[str, Any], str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    routes = config.get("routes")
    if not isinstance(routes, list):
        raise ProbeError("native config has no route list")
    selected = route_id or config.get("experimental_bootstrap_route_id")
    matching = [route for route in routes if isinstance(route, dict) and route.get("route_id") == selected]
    if len(matching) != 1:
        raise ProbeError("requested native route is not unique")
    route = matching[0]
    for field in ("model_file", "mmproj_file", "model_sha256", "mmproj_sha256"):
        if not isinstance(route.get(field), str) or not route[field]:
            raise ProbeError(f"native route lacks {field}")
    if route["model_sha256"] == route["mmproj_sha256"]:
        raise ProbeError("model and projector need distinct artifact digests")
    return route, selected


def _remote_observation(source: dict[str, Any], artifacts: list[Artifact],
                        fetch: _FETCH) -> dict[str, Any]:
    repo, revision = source["repo"], source["revision"]
    metadata_url = f"https://huggingface.co/api/models/{repo}/revision/{revision}"
    tree_url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
    metadata, metadata_sha = fetch(metadata_url)
    tree, tree_sha = fetch(tree_url)
    if (not isinstance(metadata, dict) or metadata.get("id") != repo or
        metadata.get("sha") != revision or
        not isinstance(metadata.get("cardData"), dict) or
        metadata["cardData"].get("license") != source["license"]):
        raise ProbeError("pinned model metadata disagrees with source claim")
    base = metadata["cardData"].get("base_model", [])
    if source.get("declared_base_model") and (not isinstance(base, list) or
                                                source["declared_base_model"] not in base):
        raise ProbeError("pinned model card disagrees with declared base model")
    if not isinstance(tree, list):
        raise ProbeError("pinned tree metadata is not a file list")
    entries = {item.get("path"): item for item in tree if isinstance(item, dict)
               and isinstance(item.get("path"), str)}
    observed = []
    for artifact in artifacts:
        item = entries.get(artifact.filename)
        lfs = item.get("lfs") if isinstance(item, dict) else None
        if (not isinstance(lfs, dict) or lfs.get("oid") != artifact.sha256.lower() or
            lfs.get("size") != artifact.size or item.get("size") != artifact.size):
            raise ProbeError("pinned Hub LFS identity disagrees with artifact manifest")
        observed.append({"manifest_filename": artifact.filename,
                         "lfs_oid_sha256": lfs["oid"], "lfs_size_bytes": lfs["size"]})
    return {
        "source_repo": source["source_repo"], "checkpoint_revision": revision,
        "license": metadata["cardData"]["license"],
        "license_link": metadata["cardData"].get("license_link"),
        "declared_base_model": source.get("declared_base_model"),
        "base_model_provenance_verified": False,
        "metadata_url": metadata_url, "metadata_response_sha256": metadata_sha,
        "tree_url": tree_url, "tree_response_sha256": tree_sha,
        "remote_artifacts": observed,
    }


def probe_lineage(*, manifest_path: Path, source_path: Path, config_path: Path,
                  output_path: Path, route_id: str | None = None,
                  fetch: _FETCH = _fetch_json) -> dict[str, Any]:
    """Record a candidate observation; human review remains mandatory."""
    artifacts = load_manifest(manifest_path)
    if len(artifacts) != 2:
        raise ProbeError("Gemma lineage probe requires model and projector artifacts")
    source = _source_metadata(source_path, artifacts)
    route, selected = _route(config_path, route_id)
    by_sha = {artifact.sha256.lower(): artifact for artifact in artifacts}
    if len(by_sha) != 2 or {route["model_sha256"].lower(), route["mmproj_sha256"].lower()} != set(by_sha):
        raise ProbeError("native model/projector hashes do not match the pinned manifest")
    record: dict[str, Any] = {
        "record_type": "checkpoint_lineage_probe_v1",
        "status": "blocked", "human_review_required": True,
        "route_id": selected,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_metadata_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "native_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "remote": None, "artifacts": [],
    }
    try:
        record["remote"] = _remote_observation(source, artifacts, fetch)
    except (OSError, ValueError, TypeError, urllib.error.URLError) as exc:
        record["failure_code"] = "remote_metadata_" + type(exc).__name__
    else:
        for role, file_field, sha_field in (
            ("model", "model_file", "model_sha256"),
            ("projector", "mmproj_file", "mmproj_sha256"),
        ):
            artifact = by_sha[route[sha_field].lower()]
            path = Path(route[file_field])
            item = {"role": role, "local_path": str(path),
                    "manifest_filename": artifact.filename,
                    "expected_size_bytes": artifact.size,
                    "expected_sha256": artifact.sha256.lower()}
            try:
                size, digest = _hash_local(path)
            except (OSError, ValueError) as exc:
                item.update({"verified": False,
                             "failure_code": "local_file_" + type(exc).__name__})
            else:
                item.update({"observed_size_bytes": size, "observed_sha256": digest,
                             "verified": size == artifact.size and digest == artifact.sha256.lower()})
            record["artifacts"].append(item)
        if all(item["verified"] for item in record["artifacts"]):
            record["status"] = "observed"
            record["observations_for_review"] = {
                "checkpoint_revision": source["revision"],
                "artifact_sha256": route["model_sha256"].lower(),
                "projector_sha256": route["mmproj_sha256"].lower(),
                "source_repo": source["source_repo"],
                "license": source["license"],
            }
        else:
            record["status"] = "mismatch"
    record["record_sha256"] = record_digest(record)
    _atomic_private_json(output_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=_HERE / "configs/gemma4_31b_qat_q4_download.json")
    parser.add_argument("--source-metadata", type=Path,
                        default=_HERE / "configs/gemma4_31b_qat_q4_lineage_source.metadata")
    parser.add_argument("--config", type=Path,
                        default=_HERE / "configs/windows_laptop_gemma4_31b_hybrid.experimental.json")
    parser.add_argument("--route-id")
    parser.add_argument("--output", type=Path,
                        default=_HERE / "results" / f"gemma_lineage_{uuid.uuid4().hex}.json")
    args = parser.parse_args()
    result = probe_lineage(manifest_path=args.manifest, source_path=args.source_metadata,
                           config_path=args.config, output_path=args.output,
                           route_id=args.route_id)
    print(json.dumps({"status": result["status"], "record_sha256": result["record_sha256"],
                      "output": str(args.output)}, sort_keys=True))
    return 0 if result["status"] == "observed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
