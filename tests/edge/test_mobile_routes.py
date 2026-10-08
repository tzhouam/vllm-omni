# SPDX-License-Identifier: Apache-2.0
"""Publisher candidates and hosted replay do not qualify a resident phone."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vllm_omni.edge.mobile_routes import (
    MobileEvidence,
    MobileEvidenceKind,
    load_mobile_candidates,
    mobile_devices,
    mobile_resource_demands,
)
from vllm_omni.engine.resource_ledger import ResourceLedger, ResourceUnavailable
from vllm_omni.engine.weight_tiers import ArtifactManifest, WeightTierBudget

CONFIGS = Path(__file__).resolve().parents[2] / "benchmarks/edge_harness/configs"


def test_pinned_mobile_candidates_have_all_required_files_without_claiming_local_support():
    candidates = load_mobile_candidates(CONFIGS / "mobile/candidates.json")
    assert set(candidates) == {"gemma4-e2b-litert", "gemma4-e4b-litert", "qwen38-27b-ud-iq1-s"}
    qwen = candidates["qwen38-27b-ud-iq1-s"]
    assert {item.role for item in qwen.artifact_manifest.files} == {"weights", "mmproj"}
    assert qwen.artifact_manifest.total_size_bytes == 6192222208 + 927607488
    for candidate in candidates.values():
        assert candidate.runtime_revision is None
        assert not candidate.admission_ready
        assert "npu" not in candidate.candidate_compute_units
        assert len(candidate.artifact_manifest.revision) == 40
        assert candidate.artifact_manifest.hash_origin == "declared"
    with pytest.raises(TypeError):
        candidates["invented"] = qwen


def test_phone_cpu_gpu_and_npu_do_not_create_additive_ram_pools():
    devices = mobile_devices("s25-test")
    assert {device.kind for device in devices} == {"cpu", "gpu", "npu"}
    assert {device.memory_pool_ids for device in devices} == {("host_ram",)}
    budget = WeightTierBudget(cpu_resident_weights_bytes=20, gpu_weights_bytes=30, gpu_kv_bytes=10)
    demands = mobile_resource_demands(budget)
    assert demands == {"host_ram": 60}
    with pytest.raises(ResourceUnavailable, match="host_ram"):
        ResourceLedger({"host_ram": 50}).reserve("mobile", demands)


def test_aihub_complete_chain_keeps_resident_memory_latency_and_thermal_unknown():
    evidence = MobileEvidence(
        "gemma", MobileEvidenceKind.AIHUB_CHAIN, ("real-jobs.json",),
        complete_request=True, task_quality_passed=True, observed_execution_units=("qnn:npu",),
    )
    assert evidence.hosted_functional_pass
    assert not evidence.ready_for_device_qualification
    assert evidence.shared_ram_peak_bytes is None
    with pytest.raises(ValueError, match="device-resident"):
        replace(evidence, whole_request_seconds=1.0)
    with pytest.raises(ValueError, match="device-resident"):
        replace(evidence, shared_ram_peak_bytes=1)
    assert not replace(evidence, kind=MobileEvidenceKind.AIHUB_COMPONENT).hosted_functional_pass
    assert not replace(evidence, raw_evidence=()).hosted_functional_pass


def test_device_record_is_only_ready_for_later_qualification_when_measurements_exist():
    evidence = MobileEvidence("gemma", MobileEvidenceKind.DEVICE_LOCAL, ("local.json",),
                             complete_request=True, task_quality_passed=True)
    assert not evidence.ready_for_device_qualification
    measured = replace(evidence, observed_execution_units=("cpu", "gpu"), shared_ram_peak_bytes=100,
                       whole_request_seconds=2.0, cancel_release_passed=True, sustained_seconds=1800)
    assert measured.ready_for_device_qualification
    assert not measured.hosted_functional_pass
    assert not replace(measured, sustained_seconds=1799).ready_for_device_qualification
    assert not replace(measured, cancel_release_passed=False).ready_for_device_qualification


def test_deepseek_manifest_binds_all_seven_shards_and_exact_runtime_without_claiming_run():
    value = json.loads((CONFIGS / "deepseek/v41_q2_candidate.json").read_text(encoding="utf-8"))
    manifest = ArtifactManifest.from_dict(value["artifact_manifest"])
    assert len(manifest.files) == 7
    assert manifest.total_size_bytes == 264515279456
    assert value["runtime"]["revision"] == "5210c7c5ed61dddaee6ed476623abf4b63093d16"
    assert value["status"] == "not_executed"
    assert value["local_baseline"]["batch_size"] == 1
    with pytest.raises(ValueError, match="incomplete GGUF"):
        replace(manifest, files=manifest.files[:-1])
