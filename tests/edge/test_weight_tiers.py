# SPDX-License-Identifier: Apache-2.0
"""Artifact corruption, shared admission and truthful observation boundaries."""

from dataclasses import FrozenInstanceError, replace

import pytest

from vllm_omni.engine.resource_ledger import ResourceLedger, ResourceUnavailable
from vllm_omni.engine.weight_tiers import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactTransformation,
    BackendCapabilities,
    ComputePlacement,
    PlacementReport,
    StoragePlacement,
    WeightTierBudget,
    WeightTierPlan,
    manifest_from_gguf,
    reserve_tier_plan,
)


def _artifacts(tmp_path):
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    first.write_bytes(b"small metadata shard")
    second.write_bytes(b"actual expert weights")
    (tmp_path / "mmproj.gguf").write_bytes(b"projector")
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    # An unrelated precision must not be inventoried into the selected route.
    (tmp_path / "other-q8.gguf").write_bytes(b"unrelated")
    return first, second


def _manifest(tmp_path):
    first, _ = _artifacts(tmp_path)
    return manifest_from_gguf(
        first, checkpoint="community/model", revision="a1b2c3", license="Apache-2.0",
        auxiliary_files={"mmproj": "mmproj.gguf", "tokenizer": ["tokenizer.json"]},
        lineage=[ArtifactTransformation("ptq", "upstream/model", "b2c3d4", "calibration.json")],
    )


def _plan(budget, **kwargs):
    return WeightTierPlan("q4-ssd", "a" * 64, "strata", "d5ea713", budget, **kwargs)


def test_inventory_binds_every_shard_and_auxiliary_file(tmp_path):
    manifest = _manifest(tmp_path)
    assert len(manifest.verify(tmp_path)) == 4
    assert manifest.weight_size_bytes == len(b"small metadata shardactual expert weights")
    assert manifest.total_size_bytes == sum(item.size_bytes for item in manifest.files)
    assert manifest.hash_origin == "local_observation"
    serialized = manifest.to_dict()
    restored = ArtifactManifest.from_dict(serialized)
    assert restored.manifest_sha256 == manifest.manifest_sha256
    assert replace(manifest, files=tuple(reversed(manifest.files))).manifest_sha256 == manifest.manifest_sha256
    serialized["files"][0]["sha256"] = "b" * 64
    assert restored.manifest_sha256 == manifest.manifest_sha256
    with pytest.raises(FrozenInstanceError):
        manifest.revision = "changed"


@pytest.mark.parametrize("filename", ["model-00002-of-00002.gguf", "mmproj.gguf", "tokenizer.json"])
def test_corruption_in_later_shard_or_auxiliary_file_is_rejected(tmp_path, filename):
    manifest = _manifest(tmp_path)
    target = tmp_path / filename
    target.write_bytes(b"x" * target.stat().st_size)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        manifest.verify(tmp_path)


def test_missing_split_or_size_change_refuses_before_load(tmp_path):
    first, second = _artifacts(tmp_path)
    second.unlink()
    with pytest.raises(FileNotFoundError, match="00002"):
        manifest_from_gguf(first, checkpoint="model", revision="abc123", license="test")
    second.write_bytes(b"experts")
    manifest = manifest_from_gguf(first, checkpoint="model", revision="abc123", license="test")
    with pytest.raises(ValueError, match="incomplete GGUF"):
        replace(manifest, files=manifest.files[:1])
    second.write_bytes(b"longer experts")
    with pytest.raises(ValueError, match="size mismatch"):
        manifest.verify(tmp_path)


@pytest.mark.parametrize("path", [
    "../outside", "/absolute", "C:/weights", "sub\\weights", "a:stream", "a//b", "con.gguf", "weights. ",
])
def test_manifest_paths_cannot_escape_on_windows_or_posix(path):
    with pytest.raises(ValueError, match="path"):
        ArtifactFile(path, 1, "a" * 64)


def test_manifest_rejects_symlink_escape_and_windows_aliases(tmp_path):
    model_root = tmp_path / "root"
    model_root.mkdir()
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"weights")
    (model_root / "model.gguf").symlink_to(outside)
    file = ArtifactFile("model.gguf", 7, "a" * 64)
    manifest = ArtifactManifest("model", "abc123", "test", (file,))
    with pytest.raises(ValueError, match="outside"):
        manifest.verify(model_root)
    with pytest.raises(ValueError, match="duplicate"):
        replace(manifest, files=(file, replace(file, path="MODEL.GGUF")))


def test_manifest_identity_includes_conversion_lineage_and_license(tmp_path):
    manifest = _manifest(tmp_path)
    repack = ArtifactTransformation("repack", "community/model", "a1b2c3", "pack-v1.json")
    assert replace(manifest, lineage=(*manifest.lineage, repack)).manifest_sha256 != manifest.manifest_sha256
    assert replace(manifest, license="community-license").manifest_sha256 != manifest.manifest_sha256
    with pytest.raises(ValueError, match="floating"):
        replace(manifest, revision="main")
    with pytest.raises(ValueError, match="together"):
        ArtifactTransformation("distillation", "student", "abc", "recipe", teacher_checkpoint="teacher")


def test_budget_counts_resident_file_pages_but_not_all_disk_weights_as_ram(tmp_path):
    manifest = _manifest(tmp_path)
    budget = WeightTierBudget(
        gpu_weights_bytes=10, gpu_expert_cache_bytes=15, gpu_kv_bytes=5,
        cpu_resident_weights_bytes=10, cpu_mapped_weights_bytes=20, cpu_expert_cache_bytes=30,
        pinned_bytes=4, host_kv_bytes=3, host_workspace_bytes=2, host_transfer_bytes=1,
        host_headroom_bytes=5, host_loading_peak_bytes=100, gpu_loading_peak_bytes=40,
        windows_commit_peak_bytes=90, ssd_artifact_bytes=manifest.total_size_bytes, ssd_temporary_bytes=5,
    )
    budget.check_manifest(manifest)
    assert budget.host_steady_bytes == 75
    assert budget.gpu_steady_bytes == 30
    assert budget.resource_demands(include_wsl=True, include_windows_commit=True) == {
        "host_ram": 100, "wsl_ram": 100, "vram:0": 40, "windows_commit": 90,
        "ssd": manifest.total_size_bytes + 5,
    }
    assert replace(budget, windows_commit_peak_bytes=0).resource_demands(
        include_windows_commit=True,
    )["windows_commit"] == 55
    with pytest.raises(ValueError, match="complete manifest"):
        replace(budget, ssd_artifact_bytes=manifest.total_size_bytes - 1).check_manifest(manifest)


def test_existing_shared_ledger_sees_native_and_tiered_claims_and_quarantine():
    ledger = ResourceLedger({"host_ram": 100, "vram:0": 40, "ssd": 1000, "windows_commit": 100})
    native = ledger.reserve("native", {"host_ram": 70})
    plan = _plan(WeightTierBudget(cpu_expert_cache_bytes=40, gpu_weights_bytes=20, ssd_artifact_bytes=200))
    with pytest.raises(ResourceUnavailable, match="host_ram"):
        reserve_tier_plan(ledger, "strata", plan, include_windows_commit=True)
    assert ledger.snapshot()["owners"] == ["native"]
    assert ledger.release(native, drained=True)
    claim = reserve_tier_plan(ledger, "strata", plan, include_windows_commit=True)
    assert ledger.snapshot()["reserved"]["host_ram"] == 40
    assert not ledger.release(claim, drained=False)
    assert ledger.snapshot()["quarantined"] == ["strata"]
    assert ledger.release(claim, drained=True)


def test_mobile_gpu_allocations_are_charged_to_the_same_physical_ram():
    budget = WeightTierBudget(cpu_expert_cache_bytes=20, gpu_weights_bytes=30, gpu_kv_bytes=10)
    demands = budget.resource_demands(shared_gpu_memory=True, include_windows_commit=True)
    assert demands == {"host_ram": 60, "windows_commit": 60}
    ledger = ResourceLedger({"host_ram": 50})
    with pytest.raises(ResourceUnavailable, match="host_ram"):
        reserve_tier_plan(ledger, "mobile", _plan(budget), shared_gpu_memory=True)


def test_tier_plan_does_not_turn_mmap_into_cpu_compute_capability():
    plan = _plan(WeightTierBudget(cpu_mapped_weights_bytes=20), cpu_expert_layers=2, ssd_experts=True)
    with pytest.raises(ValueError, match="cpu_expert_offload"):
        plan.check_capabilities(BackendCapabilities(bounded_ssd_expert_cache=True))
    plan.check_capabilities(BackendCapabilities(cpu_expert_offload=True, bounded_ssd_expert_cache=True))
    assert WeightTierPlan.from_dict(plan.to_dict()) == plan
    with pytest.raises(ValueError, match="batch size 1"):
        replace(plan, batch_size=2)
    with pytest.raises(ValueError, match="one active request"):
        replace(plan, max_active_requests=2)


def test_observations_separate_logical_physical_io_and_storage_compute():
    unknown = PlacementReport().to_dict()
    assert unknown["physical_ssd_read_bytes"] is None
    assert unknown["expert_cache_hit_ratio"] is None
    report = PlacementReport(
        compute=[ComputePlacement("experts", ("cuda:0",), "backend kernel trace")],
        storage=[StoragePlacement("experts", "ram", 40)], logical_read_bytes=100,
        physical_ssd_read_bytes=0, expert_cache_hit_ratio=.5, io_mode="buffered",
    )
    assert report.compute[0].execution_units == ("cuda:0",)
    assert report.storage[0].tier == "ram"
    assert report.physical_ssd_read_bytes == 0
    with pytest.raises(ValueError, match="finite"):
        replace(report, ssd_wait_seconds=float("nan"))
    with pytest.raises(ValueError, match="within"):
        replace(report, expert_cache_hit_ratio=1.1)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_byte_demands_cannot_enter_the_ledger(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        WeightTierBudget(cpu_expert_cache_bytes=value)
