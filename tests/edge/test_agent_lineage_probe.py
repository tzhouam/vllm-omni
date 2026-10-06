"""Offline tests for the read-only Gemma artifact lineage probe."""

from __future__ import annotations

import hashlib
import json

import pytest

from benchmarks.edge_agent.experiments.probe_gemma_lineage import (
    ProbeError,
    _atomic_private_json,
    _fetch_json,
    probe_lineage,
    verify_record,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path, *, corrupt_model: bool = False):
    model, projector = b"GGUF-model-bytes", b"GGUF-projector-bytes"
    model_path, projector_path = tmp_path / "renamed-main.bin", tmp_path / "renamed-visual.bin"
    model_path.write_bytes(model + (b"!" if corrupt_model else b""))
    projector_path.write_bytes(projector)
    repo, revision = "test/pinned-gemma", "a" * 40
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "repo": repo, "revision": revision,
        "files": [
            {"filename": "original-model.gguf", "size": len(model), "sha256": _sha(model)},
            {"filename": "original-projector.gguf", "size": len(projector), "sha256": _sha(projector)},
        ],
    }))
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "schema": "omni-agent-lineage-source-v1", "repo": repo,
        "revision": revision, "source_repo": f"https://huggingface.co/{repo}",
        "license": "apache-2.0", "declared_base_model": "test/base-gemma",
    }))
    config = tmp_path / "native.json"
    config.write_text(json.dumps({
        "experimental_bootstrap_route_id": "gemma-route",
        "routes": [{"route_id": "gemma-route", "model_file": str(model_path),
                    "model_sha256": _sha(model), "mmproj_file": str(projector_path),
                    "mmproj_sha256": _sha(projector)}],
    }))
    remote = {
        "metadata": {"id": repo, "sha": revision,
                     "cardData": {"license": "apache-2.0", "base_model": ["test/base-gemma"],
                                  "license_link": "https://example.invalid/license"}},
        "tree": [
            {"path": "original-model.gguf", "size": len(model),
             "lfs": {"oid": _sha(model), "size": len(model)}},
            {"path": "original-projector.gguf", "size": len(projector),
             "lfs": {"oid": _sha(projector), "size": len(projector)}},
        ],
    }

    def fetch(url: str):
        assert url.startswith(f"https://huggingface.co/api/models/{repo}/")
        assert revision in url
        body = remote["tree" if "/tree/" in url else "metadata"]
        return body, _sha(json.dumps(body, sort_keys=True).encode())

    return manifest, source, config, remote, fetch


def test_probe_binds_pinned_remote_lfs_and_local_bytes_without_name_inference(tmp_path):
    manifest, source, config, _remote, fetch = _fixture(tmp_path)
    output = tmp_path / "private" / "lineage.json"
    result = probe_lineage(manifest_path=manifest, source_path=source,
                           config_path=config, output_path=output, fetch=fetch)
    assert result["status"] == "observed"
    assert result["human_review_required"] is True
    assert result["remote"]["base_model_provenance_verified"] is False
    assert {item["role"] for item in result["artifacts"]} == {"model", "projector"}
    assert all(item["verified"] for item in result["artifacts"])
    assert result["artifacts"][0]["local_path"].endswith("renamed-main.bin")
    assert result["artifacts"][0]["manifest_filename"] == "original-model.gguf"
    assert output.exists()
    assert verify_record(json.loads(output.read_text()))
    result["artifacts"][0]["verified"] = False
    assert not verify_record(result)


def test_probe_never_overwrites_an_existing_evidence_record(tmp_path):
    manifest, source, config, _remote, fetch = _fixture(tmp_path)
    output = tmp_path / "private" / "lineage.json"
    output.parent.mkdir()
    original = b'{"reviewed": "prior observation"}\n'
    output.write_bytes(original)
    with pytest.raises(FileExistsError):
        probe_lineage(manifest_path=manifest, source_path=source,
                      config_path=config, output_path=output, fetch=fetch)
    assert output.read_bytes() == original
    assert not list(output.parent.glob(".lineage.json.*.tmp"))


def test_no_hardlink_filesystem_still_creates_exclusively(tmp_path, monkeypatch):
    def no_hardlinks(*_args):
        raise OSError("hard links unavailable on this share")

    monkeypatch.setattr("benchmarks.edge_agent.experiments.probe_gemma_lineage.os.link",
                        no_hardlinks)
    output = tmp_path / "lineage.json"
    _atomic_private_json(output, {"status": "observed"})
    original = output.read_bytes()
    assert json.loads(original) == {"status": "observed"}
    with pytest.raises(FileExistsError):
        _atomic_private_json(output, {"status": "mismatch"})
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".lineage.json.*.tmp"))


def test_probe_records_local_mismatch_without_claiming_lineage(tmp_path):
    manifest, source, config, _remote, fetch = _fixture(tmp_path, corrupt_model=True)
    output = tmp_path / "private.json"
    result = probe_lineage(manifest_path=manifest, source_path=source,
                           config_path=config, output_path=output, fetch=fetch)
    assert result["status"] == "mismatch"
    assert result["artifacts"][0]["verified"] is False
    assert result["artifacts"][1]["verified"] is True
    assert "observations_for_review" not in result
    assert verify_record(json.loads(output.read_text()))


@pytest.mark.parametrize("field,value", [
    ("license", "unverified"),
    ("source_repo", "https://example.invalid/relabel"),
    ("revision", "b" * 40),
])
def test_source_claim_must_match_manifest(tmp_path, field, value):
    manifest, source, config, _remote, fetch = _fixture(tmp_path)
    data = json.loads(source.read_text())
    data[field] = value
    source.write_text(json.dumps(data))
    with pytest.raises(ProbeError):
        probe_lineage(manifest_path=manifest, source_path=source,
                       config_path=config, output_path=tmp_path / "out.json", fetch=fetch)


@pytest.mark.parametrize("change", ["oid", "license", "revision", "base_model"])
def test_probe_rejects_remote_metadata_or_tree_drift(tmp_path, change):
    manifest, source, config, remote, fetch = _fixture(tmp_path)
    if change == "oid":
        remote["tree"][0]["lfs"]["oid"] = "f" * 64
    elif change == "license":
        remote["metadata"]["cardData"]["license"] = "other"
    elif change == "revision":
        remote["metadata"]["sha"] = "b" * 40
    else:
        remote["metadata"]["cardData"]["base_model"] = ["test/other"]
    output = tmp_path / "blocked.json"
    result = probe_lineage(manifest_path=manifest, source_path=source,
                           config_path=config, output_path=output, fetch=fetch)
    assert result["status"] == "blocked"
    assert result["artifacts"] == []  # no large local reads after remote mismatch
    assert result["failure_code"] == "remote_metadata_ProbeError"
    assert verify_record(json.loads(output.read_text()))


def test_native_config_hash_must_match_manifest_even_if_path_is_renamed(tmp_path):
    manifest, source, config, _remote, fetch = _fixture(tmp_path)
    data = json.loads(config.read_text())
    data["routes"][0]["model_sha256"] = "f" * 64
    config.write_text(json.dumps(data))
    with pytest.raises(ProbeError, match="pinned manifest"):
        probe_lineage(manifest_path=manifest, source_path=source,
                       config_path=config, output_path=tmp_path / "out.json", fetch=fetch)


def test_fetch_failure_writes_hash_bound_blocker_without_exception_details(tmp_path):
    manifest, source, config, _remote, _fetch = _fixture(tmp_path)

    def fetch(_url):
        raise OSError("secret-user-token-should-not-appear")

    output = tmp_path / "blocked.json"
    result = probe_lineage(manifest_path=manifest, source_path=source,
                           config_path=config, output_path=output, fetch=fetch)
    assert result["status"] == "blocked"
    assert "secret-user-token" not in output.read_text()
    assert verify_record(json.loads(output.read_text()))


def test_public_metadata_fetch_is_bounded_and_sends_no_auth(monkeypatch):
    seen = []

    class Response:
        headers = {"Content-Length": "2"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, count):
            assert count == 512 * 1024 + 1
            return b"{}"

    class Opener:
        def open(self, request, timeout):
            seen.append((request, timeout))
            return Response()

    monkeypatch.setattr("urllib.request.build_opener", lambda *_args: Opener())
    payload, digest = _fetch_json("https://huggingface.co/api/models/test/model/revision/" + "a" * 40)
    assert payload == {}
    assert digest == _sha(b"{}")
    request, timeout = seen[0]
    assert request.get_header("Authorization") is None
    assert timeout == 30
    assert "/resolve/" not in request.full_url


def test_metadata_fetch_rejects_model_byte_url():
    with pytest.raises(ProbeError):
        _fetch_json("https://huggingface.co/test/model/resolve/" + "a" * 40 + "/model.gguf")
