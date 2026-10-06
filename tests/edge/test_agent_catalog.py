# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The candidate queue must not turn a file-size estimate into admission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from vllm_omni.edge.agent.catalog import (
    CANDIDATES,
    GB,
    GiB,
    CapacitySnapshot,
    RuntimeEvidence,
    bind_artifact_identity,
    inspect_local_artifact,
    preflight_artifact,
)


def _runtime(
    *formats: str,
    candidate: str | None = None,
    family: str = "llama.cpp/GGUF",
    vision: bool = True,
    offload: bool = True,
    lazy: bool = False,
):
    return RuntimeEvidence(
        executable="C:/llama.cpp/llama-cli.exe",
        version="probed-build",
        native_windows=True,
        formats=frozenset(formats),
        vision_projector=vision,
        cpu_gpu_offload=offload,
        runtime_family=family,
        verified_candidate_keys=frozenset({candidate}) if candidate else frozenset(),
        lazy_mmap=lazy,
    )


def _snapshot(*, ram_gib: int = 45, vram_gib: int = 22, disk_gib: int = 300):
    return CapacitySnapshot(
        available_ram_bytes=ram_gib * GiB,
        available_vram_bytes=vram_gib * GiB,
        disk_free_bytes=disk_gib * GiB,
        native_windows=True,
    )


def test_full_candidate_queue_keeps_pruned_and_capacity_only_separate():
    assert len(CANDIDATES) == 22
    assert len(set(CANDIDATES)) == len(CANDIDATES)
    assert CANDIDATES["qwen3.5-122b-reap-76b-q4-k-m"].lineage.startswith("pruned")
    assert CANDIDATES["deepseek-v4-flash-0731-iq1-s"].status == "capacity_only"
    assert CANDIDATES["glm5.3-flash-min-gguf"].status == "capacity_only"
    assert CANDIDATES["glm5.3-flash-min-gguf"].precision == "UD-IQ1_S"
    assert CANDIDATES["minimax-m3-min-gguf"].status == "capacity_only"
    assert CANDIDATES["minimax-m3-min-gguf"].precision == "IQ1_S"
    assert CANDIDATES["gpt-oss-120b-mxfp4"].size_bytes_estimate == 59 * GiB
    assert CANDIDATES["qwen3.8-flash-next-coder-gsq-rco"].precision == "GSQ-RCO_IQ1_M"
    assert CANDIDATES["qwen3.8-flash-next-gsq-rco-q2-0"].precision == "GSQ-RCO_Q2_0"
    reap = CANDIDATES["qwen3.5-122b-reap-76b-q4-k-m"]
    assert reap.modalities == ("text",)
    assert "screen" not in reap.task_routes
    assert reap.checkpoint_url.endswith("/0xSero/Qwen3.5-76B")
    assert reap.base_checkpoint_url.endswith("/Qwen/Qwen3.5-122B-A10B")
    assert CANDIDATES["deepseek-v4-flash-0731-reap25-mixed"].runtime_family == "ds4/DwarfStar GGUF"


def test_large_capacity_only_models_are_never_download_eligible():
    for key in (
        "deepseek-v4-flash-0731-iq1-s",
        "glm5.3-flash-min-gguf",
        "minimax-m3-min-gguf",
    ):
        candidate = CANDIDATES[key]
        result = preflight_artifact(
            candidate, _snapshot(ram_gib=63, vram_gib=23),
            _runtime(candidate.precision, candidate=key),
        )
        assert not result.download_eligible
        assert result.status == "capacity_only_refusal"
        assert any("do not download" in reason for reason in result.reasons)


def test_runtime_evidence_is_required_before_download():
    candidate = CANDIDATES["qwen3.6-35b-a3b-iq4-xs"]
    result = preflight_artifact(candidate, _snapshot(), None)
    assert not result.download_eligible
    assert result.status == "pending_preflight_evidence"
    assert any("runtime" in reason for reason in result.reasons)


def test_qwen36_pinned_bundle_includes_the_required_vision_projector():
    candidate = CANDIDATES["qwen3.6-35b-a3b-iq4-xs"]
    assert candidate.declared_plan_size_gb_decimal == 17.7
    manifest = json.loads((Path(__file__).resolve().parents[2] / "benchmarks"
                           / "edge_agent" / "configs"
                           / "qwen3_6_35b_a3b_ud_iq4_xs_download.json")
                          .read_text(encoding="utf-8"))
    files = {item["filename"]: item for item in manifest["files"]}
    assert candidate.artifact_revision == manifest["revision"]
    assert candidate.checkpoint_revision is None
    assert manifest["source_checkpoint_revision"] is None
    assert candidate.projector_url.endswith("/mmproj-F16.gguf")
    assert candidate.complete_download_bytes_estimate == sum(
        item["size"] for item in files.values()
    )
    assert files["Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"]["size"] == candidate.size_bytes_estimate
    assert files["mmproj-F16.gguf"]["size"] == int(0.89928368 * GB)
    assert all(len(item["sha256"]) == 64 for item in files.values())
    gemma = CANDIDATES["gemma4-31b-qat-q4-0"]
    assert gemma.declared_plan_size_gb_decimal == 17.7
    assert gemma.size_gb_decimal != gemma.declared_plan_size_gb_decimal


def test_pinned_artifact_can_enter_measurement_queue_but_not_claim_admission():
    candidate = CANDIDATES["qwen3.6-35b-a3b-iq4-xs"]
    result = preflight_artifact(
        candidate, _snapshot(), _runtime("IQ4_XS", candidate=candidate.key)
    )
    assert result.download_eligible
    assert result.status == "download_eligible_for_measurement"
    assert result.capacity_evidence == "E"
    assert result.minimum_resident_bytes == int((17.730509792 + 0.89928368) * GB)


def test_mutable_artifact_and_generic_format_evidence_do_not_allow_download():
    candidate = CANDIDATES["qwen3-30b-a3b-q4-k-m"]
    result = preflight_artifact(candidate, _snapshot(), _runtime("Q4_K_M"))
    assert not result.download_eligible
    assert any("exact candidate architecture" in reason for reason in result.reasons)
    assert not any("not pinned" in reason for reason in result.reasons)
    assert candidate.size_bytes_estimate == 18_556_685_824
    assert candidate.declared_plan_size_gb_decimal == 18.6
    assert candidate.checkpoint_revision is None
    assert candidate.artifact_revision == "e4d4bafdfb96a411a163846265362aceb0b9c63a"
    mutable = replace(candidate, artifact_revision="main")
    result = preflight_artifact(
        mutable, _snapshot(), _runtime("Q4_K_M", candidate=mutable.key)
    )
    assert not result.download_eligible
    assert any("not pinned" in reason for reason in result.reasons)


def test_lazy_shard_and_special_runtime_require_explicit_evidence():
    gsq = replace(CANDIDATES["qwen3.8-flash-next-gsq-rco-q2-0"], artifact_revision="a" * 40)
    assert gsq.complete_download_bytes_estimate == int((66.4 + 0.907543008) * GB)
    assert gsq.resident_bytes_lower_bound == int((37.6 + 0.907543008) * GB)
    result = preflight_artifact(
        gsq, _snapshot(), _runtime(gsq.precision, candidate=gsq.key)
    )
    assert not result.download_eligible
    assert any("lazy mapping" in reason for reason in result.reasons)
    with_lazy = preflight_artifact(
        gsq, _snapshot(), _runtime(gsq.precision, candidate=gsq.key, lazy=True)
    )
    assert with_lazy.download_eligible

    ds4 = replace(CANDIDATES["deepseek-v4-flash-0731-reap25-mixed"],
                  artifact_revision="b" * 40)
    result = preflight_artifact(
        ds4, _snapshot(ram_gib=80),
        _runtime(ds4.precision, candidate=ds4.key),
    )
    assert not result.download_eligible
    assert any("runtime family" in reason for reason in result.reasons)


def test_unverified_quantization_and_windows_backend_block_download():
    candidate = CANDIDATES["qwen3-coder-next-reap-40b-iq4-xs"]
    runtime = RuntimeEvidence(
        executable="/usr/bin/llama-cli", version="probed", native_windows=False,
        formats=frozenset({"Q4_K_M"}), vision_projector=False,
        cpu_gpu_offload=True,
        runtime_family="llama.cpp/GGUF",
        verified_candidate_keys=frozenset({candidate.key}),
    )
    result = preflight_artifact(candidate, _snapshot(), runtime)
    assert not result.download_eligible
    assert any("native Windows" in reason for reason in result.reasons)
    assert any("IQ4_XS" in reason for reason in result.reasons)


def test_shared_ram_is_not_added_twice_and_wsl_limit_is_respected():
    candidate = CANDIDATES["qwen3-30b-a3b-q4-k-m"]
    snapshot = CapacitySnapshot(
        available_ram_bytes=30 * GiB,
        available_vram_bytes=0,
        disk_free_bytes=100 * GiB,
        wsl_ram_limit_bytes=15 * GiB,
        native_windows=False,
    )
    result = preflight_artifact(
        candidate, snapshot,
        RuntimeEvidence("/usr/bin/llama-cli", "probed", False,
                        frozenset({"Q4_K_M"}), False, False,
                        "llama.cpp/GGUF", frozenset({candidate.key})),
    )
    assert not result.download_eligible
    assert result.status == "capacity_lower_bound_refusal"


def test_hash_manifest_records_exact_bytes_and_pinned_revisions(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"exact model bytes")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b"{}")
    revision = "a" * 40
    result = inspect_local_artifact(
        "qwen3-30b-a3b-q4-k-m", (model, tokenizer),
        artifact_revision=revision, checkpoint_revision="b" * 40,
        license_reviewed=True,
    )
    assert result.fully_traced
    assert result.total_bytes == len(b"exact model bytes") + 2
    assert next(f for f in result.files if f.path.endswith("model.gguf")).sha256 == (
        hashlib.sha256(b"exact model bytes").hexdigest()
    )
    assert len(result.bundle_sha256) == 64
    moved = tmp_path / "copy"
    moved.mkdir()
    (moved / "model.gguf").write_bytes(model.read_bytes())
    (moved / "tokenizer.json").write_bytes(tokenizer.read_bytes())
    moved_result = inspect_local_artifact(
        "qwen3-30b-a3b-q4-k-m",
        (moved / "tokenizer.json", moved / "model.gguf"),
        artifact_revision=revision,
    )
    assert moved_result.bundle_sha256 == result.bundle_sha256
    bound = bind_artifact_identity(CANDIDATES["qwen3-30b-a3b-q4-k-m"], result)
    assert bound.artifact_revision == revision
    assert bound.checkpoint_revision == "b" * 40
    assert bound.artifact_bundle_sha256 == result.bundle_sha256
    with pytest.raises(ValueError, match="pinned"):
        inspect_local_artifact(
            "qwen3-30b-a3b-q4-k-m", (model,), artifact_revision="main"
        )


def test_preflight_rejects_disk_pressure_before_weight_download():
    candidate = CANDIDATES["qwen3-32b-q4-k-m"]
    result = preflight_artifact(
        candidate, _snapshot(disk_gib=30),
        _runtime("Q4_K_M", candidate=candidate.key)
    )
    assert not result.download_eligible
    assert any("disk" in reason for reason in result.reasons)
