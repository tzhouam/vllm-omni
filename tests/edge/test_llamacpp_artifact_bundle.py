# SPDX-License-Identifier: Apache-2.0
"""Every GGUF shard participates in existing llama.cpp admission and identity."""

import hashlib
from types import SimpleNamespace

import pytest

from vllm_omni.engine.backends.llamacpp import LlamaCppTextStageClient
from vllm_omni.engine.resource_ledger import ResourceLedger, ResourceUnavailable
from vllm_omni.engine.weight_tiers import manifest_from_gguf


def setup(tmp_path, *, manifest=True, host_bytes=1000):
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    binary = tmp_path / "server"
    first.write_bytes(b"small-header")
    second.write_bytes(b"large-weight" * 100)
    binary.write_bytes(b"no-server-must-launch")
    pin = manifest_from_gguf(first, checkpoint="test/model", revision="abc123", license="test")
    cfg = {
        "model_file": str(first),
        "model_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        "server_bin": str(binary),
        "server_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "log_file": str(tmp_path / "log"),
        "device": "cpu",
        "memory_pool": "host_ram",
        "memory_overhead_bytes": 100,
        "max_io_bytes": 100,
        "max_new_tokens": 8,
        "context_tokens": 128,
    }
    if manifest:
        cfg.update(artifact_root=str(tmp_path), artifact_manifest=pin.to_dict())
    ledger = ResourceLedger({"host_ram": host_bytes})
    claim = ledger.reserve("llama", {"host_ram": host_bytes})
    return cfg, ledger, claim, second


def test_split_gguf_requires_complete_manifest_before_worker_launch(tmp_path):
    cfg, ledger, claim, _ = setup(tmp_path, manifest=False)
    with pytest.raises(ValueError, match="multi-shard"):
        LlamaCppTextStageClient(SimpleNamespace(), cfg, ledger, claim)
    assert ledger.snapshot()["owners"] == []


def test_later_shard_corruption_is_rejected_before_worker_launch(tmp_path):
    cfg, ledger, claim, second = setup(tmp_path)
    second.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        LlamaCppTextStageClient(SimpleNamespace(), cfg, ledger, claim)
    assert ledger.snapshot()["owners"] == []


def test_whole_split_set_counts_towards_memory_admission(tmp_path):
    cfg, ledger, claim, _ = setup(tmp_path, host_bytes=200)
    # The first shard fits, but the complete 1,212-byte bundle does not.
    with pytest.raises(ResourceUnavailable, match="declared KV/workspace"):
        LlamaCppTextStageClient(SimpleNamespace(), cfg, ledger, claim)
    assert ledger.snapshot()["owners"] == []


def test_single_model_cannot_hide_weight_bytes_under_auxiliary_role(tmp_path):
    from vllm_omni.engine.weight_tiers import ArtifactFile, ArtifactManifest

    cfg, ledger, claim, _ = setup(tmp_path)
    single = tmp_path / "single.gguf"
    single.write_bytes(b"w" * 2000)
    cfg["model_file"] = str(single)
    cfg["model_sha256"] = hashlib.sha256(single.read_bytes()).hexdigest()
    cfg["artifact_manifest"] = ArtifactManifest(
        "test/single",
        "abc123",
        "test",
        (ArtifactFile(single.name, 2000, cfg["model_sha256"], role="auxiliary"),),
    ).to_dict()
    with pytest.raises(ValueError, match="weights role"):
        LlamaCppTextStageClient(SimpleNamespace(), cfg, ledger, claim)
    assert ledger.snapshot()["owners"] == []
