"""Offline tests for the catalog-to-downloader admission boundary."""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from benchmarks.edge_agent import download, download_gate
from benchmarks.edge_agent.download import Artifact, ArtifactDownloader, load_manifest
from vllm_omni.edge.agent.catalog import CapacitySnapshot, GiB


CONFIGS = Path(__file__).resolve().parents[2] / "benchmarks" / "edge_agent" / "configs"
QWEN = CONFIGS / "qwen3_30b_a3b_q4_k_m_download.json"
GEMMA = CONFIGS / "gemma4_31b_qat_q4_download.json"


def _binding(path: Path, key: str):
    return download_gate.bind_reviewed_manifest(key, path, load_manifest(path))


def _capacity(*, ram: int = 50, vram: int = 20, disk: int = 120):
    return CapacitySnapshot(ram * GiB, vram * GiB, disk * GiB, native_windows=True)


def _digest_json(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _research_probe(tmp_path: Path, binding):
    from benchmarks.edge_agent.experiments.probe_gguf_architecture import scan_runtime_marker

    dll = tmp_path / "llama.dll"
    dll.write_bytes(b"MZ" + b"dummy qwen3moe runtime marker\0")
    marker = scan_runtime_marker(dll, "qwen3moe")
    item = binding.model
    record = {
        "schema": "omni-agent-gguf-architecture-probe-v1",
        "manifest_sha256": binding.manifest_sha256,
        "artifact": {
            "repo": item.repo, "revision": item.revision.lower(),
            "filename": item.filename, "published_size_bytes": item.size,
            "published_lfs_sha256": item.sha256.lower(),
            "whole_artifact_sha256_verified": False,
        },
        "header_range": {"http_status": 206, "range_requested_bytes": 65536,
                         "range_received_bytes": 65536,
                         "range_sha256": "a" * 64},
        "gguf_metadata": {"general_architecture": "qwen3moe"},
        "runtime_marker": marker,
        "outcome": "architecture_marker_observed_diagnostic_only",
        "full_model_load_verified": False,
        "inference_verified": False,
        "qualification": False,
    }
    record["record_sha256"] = _digest_json(record)
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_reviewed_candidate_binding_refuses_modified_or_cross_candidate_manifest(tmp_path):
    binding = _binding(GEMMA, "gemma4-31b-qat-q4-0")
    assert binding.projector is not None
    with pytest.raises(download_gate.GateError, match="reviewed download manifest"):
        _binding(GEMMA, "glm4.7-flash-q4-k-m")
    changed = tmp_path / "changed.json"
    changed.write_bytes(GEMMA.read_bytes() + b"\n")
    with pytest.raises(download_gate.GateError, match="manifest bytes differ"):
        _binding(changed, "gemma4-31b-qat-q4-0")


def test_research_tier_is_quarantined_and_capacity_limited(tmp_path):
    binding = _binding(QWEN, "qwen3-30b-a3b-q4-k-m")
    assert binding.model.size == 18_556_685_824
    probe = _research_probe(tmp_path, binding)
    gate = download_gate.gate_download(binding, _capacity(), research_probe=probe)
    assert gate["tier"] == "quarantined_research_download_architecture_unverified"
    assert gate["release_qualified"] is False
    assert gate["exact_download_bytes"] == binding.model.size
    with pytest.raises(download_gate.GateError, match="lower bound"):
        download_gate.gate_download(binding, _capacity(ram=8, vram=0), research_probe=probe)
    with pytest.raises(download_gate.GateError, match="staging"):
        download_gate.gate_download(binding, _capacity(disk=35), research_probe=probe)


def test_research_probe_cannot_approve_changed_dll_or_claim_full_execution(tmp_path):
    binding = _binding(QWEN, "qwen3-30b-a3b-q4-k-m")
    probe = _research_probe(tmp_path, binding)
    record = json.loads(probe.read_text(encoding="utf-8"))
    record["full_model_load_verified"] = True
    record["record_sha256"] = _digest_json({k: v for k, v in record.items()
                                             if k != "record_sha256"})
    probe.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(download_gate.GateError, match="weak tier"):
        download_gate.verify_research_probe(probe, binding)
    probe = _research_probe(tmp_path, binding)
    (tmp_path / "llama.dll").write_bytes(b"MZ changed")
    with pytest.raises(download_gate.GateError, match="absent or changed"):
        download_gate.verify_research_probe(probe, binding)


def test_cli_refuses_network_before_downloader_or_token_lookup(monkeypatch, tmp_path):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("downloader or credential lookup occurred before gate")

    monkeypatch.setattr(download, "ArtifactDownloader", unexpected)
    monkeypatch.setattr(download.os, "getenv", unexpected)
    monkeypatch.setattr(download_gate, "live_capacity_snapshot", lambda _: _capacity())
    code = download.main(["--manifest", str(QWEN), "--dest", str(tmp_path),
                          "--candidate", "qwen3-30b-a3b-q4-k-m"])
    assert code == 2  # No exact runtime index or research diagnostic.


def test_check_keeps_offline_diagnostics_without_candidate(monkeypatch, tmp_path, capsys):
    class FakeDownloader:
        def __init__(self, *_args, **_kwargs):
            pass

        def preflight(self, _items):
            return {"remaining_bytes": 7, "verified_bytes": 0, "free_bytes": 100,
                    "required_free_bytes": 7}

        def download(self, *_args, **_kwargs):
            raise AssertionError("--check must not use the network")

    monkeypatch.setattr(download, "ArtifactDownloader", FakeDownloader)
    assert download.main(["--manifest", str(QWEN), "--dest", str(tmp_path), "--check"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["catalog_gate"] is None
    assert output["capacity"]["remaining_bytes"] == 7
    assert output["catalog_download_eligible"] is False
    assert download.main(["--manifest", str(QWEN), "--dest", str(tmp_path),
                          "--candidate", "qwen3-30b-a3b-q4-k-m", "--check"]) == 0
    bound = json.loads(capsys.readouterr().out)
    assert bound["candidate_binding_sha256"] == download_gate.APPROVED_MANIFEST_SHA256[
        "qwen3-30b-a3b-q4-k-m"]
    assert bound["catalog_status"] == "reviewed_binding_only_download_not_eligible"
    assert bound["catalog_download_eligible"] is False


def test_preflight_does_not_create_missing_destination(tmp_path):
    destination = tmp_path / "new" / "models"
    item = Artifact("test/model", "a" * 40, "model.gguf", 100, "b" * 64)
    capacity = ArtifactDownloader(destination, free_reserve_bytes=0).preflight([item])
    assert capacity["remaining_bytes"] == 100
    assert not destination.exists()


def test_wsl_snapshot_caps_host_ram_and_does_not_double_count_shared_gpu(monkeypatch, tmp_path):
    fake_psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(
        available=48 * GiB, total=60 * GiB))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(download_gate.sys, "platform", "linux")
    monkeypatch.setattr(download_gate.platform, "release", lambda: "microsoft-wsl2")
    monkeypatch.setattr(download_gate, "_wsl_host_free_ram", lambda: 32 * GiB)
    monkeypatch.setattr(download_gate, "_wsl_memory_limit", lambda _total: 24 * GiB)
    monkeypatch.setattr(download_gate, "_dedicated_vram", lambda: (0, None))
    snapshot = download_gate.live_capacity_snapshot(tmp_path / "new")
    assert snapshot.available_ram_bytes == 32 * GiB
    assert snapshot.wsl_ram_limit_bytes == 24 * GiB
    assert snapshot.available_vram_bytes == 0
    assert not snapshot.native_windows


def test_wsl_snapshot_refuses_unknown_host_ram(monkeypatch, tmp_path):
    fake_psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(
        available=48 * GiB, total=60 * GiB))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(download_gate.sys, "platform", "linux")
    monkeypatch.setattr(download_gate.platform, "release", lambda: "microsoft-wsl2")
    monkeypatch.setattr(download_gate, "_dedicated_vram", lambda: (0, None))
    monkeypatch.setattr(download_gate, "_wsl_host_free_ram", lambda: (_ for _ in ()).throw(
        download_gate.GateError("host unavailable")))
    with pytest.raises(download_gate.GateError, match="host unavailable"):
        download_gate.live_capacity_snapshot(tmp_path)


def test_native_runtime_requires_pinned_whole_agent_trace(monkeypatch, tmp_path):
    from vllm_omni.edge.agent import runtime_identity

    binding = _binding(GEMMA, "gemma4-31b-qat-q4-0")
    monkeypatch.setattr(download_gate.sys, "platform", "win32")
    source_hash, runtime_hash = "c" * 64, "d" * 64
    monkeypatch.setattr(runtime_identity, "imported_omni_source_sha256", lambda: source_hash)
    monkeypatch.setattr(runtime_identity, "loaded_runtime_sha256", lambda: runtime_hash)
    versions = {"vllm_omni_imported_source_sha256": source_hash,
                "agent_runtime_identity_sha256": runtime_hash}
    source, lineage, server, model_file, projector_file = (tmp_path / name for name in
        ("config.json", "lineage.json", "llama-server.exe", "model.gguf", "mmproj.gguf"))
    source.write_bytes(b"{}")
    lineage.write_bytes(b"{}")
    server.write_bytes(b"runtime executable")
    model_file.write_bytes(b"small stand-in for pinned model")
    projector_file.write_bytes(b"small stand-in for pinned projector")
    real_sha = download_gate._sha256

    def fixture_sha(path):
        if path == model_file:
            return binding.model.sha256.lower()
        if path == projector_file:
            return binding.projector.sha256.lower()
        return real_sha(path)

    monkeypatch.setattr(download_gate, "_sha256", fixture_sha)
    server_hash = download_gate._sha256(server)
    route_id = "gemma-exact-test-route"
    run_id = "test-run"
    artifact_id = binding.model.sha256[:16]
    raw = tmp_path / "samples.jsonl"
    rows = [
        {"record_type": "manifest", "batch_size": 1, "concurrency": 1, "run_id": run_id},
        {"record_type": "route_prepare", "route_id": route_id, "run_id": run_id,
         "preparation": {"cold_start_confirmed": True, "details": {
             "placement_independently_verified": True, "execution_plan": {
             "model_sha256": binding.model.sha256,
             "mmproj_sha256": binding.projector.sha256,
             "server_sha256": server_hash,
             "hybrid_placement_evidence": {"model_buffer_bytes_by_pool": {
                 "host_ram": 2, "vram": 2}},
         }}}},
        {"record_type": "request", "route_id": route_id, "run_id": run_id,
         "phase": "measured", "batch_size": 1, "concurrency": 1,
         "e2e_complete": True, "evaluation": {"success": True},
         "case": {"task_class": "browser_vision"},
         "result": {"complete_agent_trace": True, "artifact_id": artifact_id}},
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "run_id": run_id, "raw_sha256": download_gate._sha256(raw),
        "conditions": {"runtime_versions": versions},
        "routes": {route_id: {"correctness_pass": True, "e2e_trace_pass": True,
                              "measured_successes": 1,
                              "route": {"artifact_sha256": binding.model.sha256,
                                        "artifact_id": artifact_id}}},
    }), encoding="utf-8")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({
        "protocol": "smoke_incomplete",
        "conditions": {"os_version": "Windows-test", "runtime_versions": versions},
        "source_config": str(source), "source_config_sha256": download_gate._sha256(source),
        "lineage_manifest": str(lineage), "lineage_sha256": download_gate._sha256(lineage),
        "artifact_provenance": {route_id: {
            "model_sha256": binding.model.sha256,
            "mmproj_sha256": binding.projector.sha256,
            "precision": "official QAT Q4_0 GGUF + F16 projector",
            "model_file": str(model_file), "mmproj_file": str(projector_file),
            "server_bin": str(server), "server_sha256": server_hash}},
        "results": [{"route_id": route_id, "summary": str(summary),
                     "raw_jsonl": str(raw), "raw_sha256": download_gate._sha256(raw)}],
    }), encoding="utf-8")
    evidence = download_gate.native_runtime_evidence(index, binding)
    assert binding.candidate_key in evidence.verified_candidate_keys
    assert evidence.vision_projector and evidence.cpu_gpu_offload
    admitted = download_gate.gate_download(binding, _capacity(), runtime_index=index)
    assert admitted["tier"] == "measurement_download_eligible_not_route_qualified"
    assert admitted["release_qualified"] is False
    monkeypatch.setattr(runtime_identity, "loaded_runtime_sha256", lambda: "e" * 64)
    with pytest.raises(download_gate.GateError, match="different Omni runtime source"):
        download_gate.native_runtime_evidence(index, binding)
    monkeypatch.setattr(runtime_identity, "loaded_runtime_sha256", lambda: runtime_hash)
    rows[2]["evaluation"]["success"] = False
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(download_gate.GateError, match="whole-Agent load"):
        download_gate.native_runtime_evidence(index, binding)
