# SPDX-License-Identifier: Apache-2.0
"""No downloads or inference: verify source, pack conversion and launch bindings."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from benchmarks.edge_harness.strata_prepare import (
    PACK_REQUIRED,
    inventory,
    manifest_digest,
    prepare,
)
from benchmarks.edge_harness.strata_profile import RUNTIME_REVISION


def fixture(tmp_path: Path):
    runtime = tmp_path / "runtime"
    for name in (
        "serve/server.py",
        "tools/iq_pack.py",
        "tools/gguf_reader.py",
        "tools/strata_tokenizer.py",
        "engine/strata.exe",
        "engine/helper.dll",
        "data/expert-profile.bin",
    ):
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-runtime")
    artifacts = tmp_path / "source"
    artifacts.mkdir()
    files = []
    for index in (1, 2):
        name = f"test-0000{index}-of-00002.gguf"
        data = bytes([index])
        (artifacts / name).write_bytes(data)
        files.append(
            {
                "path": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "role": "weights",
                "quantization": "test",
                "layout": "GGUF",
            }
        )
    target = {
        "target_id": "test-q2",
        "storage_route": "ram-experts-ssd-ple",
        "runtime": {"revision": RUNTIME_REVISION},
        "artifact_manifest": {
            "schema": "omni-weight-artifacts-v1",
            "checkpoint": "test/model",
            "revision": "a" * 40,
            "license": "test",
            "hash_origin": "declared",
            "files": files,
            "lineage": [],
        },
    }
    return target, {
        "runtime_root": runtime,
        "python_bin": Path(sys.executable),
        "engine_file": "engine/strata.exe",
        "artifact_root": artifacts,
        "out_pack": tmp_path / "pack",
        "launch_out": tmp_path / "launch.json",
        "expert_ram_bytes": 100,
        "gpu_total_bytes": 2000,
        "gpu_budget_bytes": 1000,
        "host_overhead_bytes": 400,
        "host_capacity_bytes": 2000,
        "max_io_bytes": 100,
        "hardware_snapshot": {"power_condition": "test AC"},
        "component_budget": {
            "cpu_expert_cache_bytes": 100,
            "host_workspace_bytes": 400,
            "host_transfer_bytes": 100,
            "gpu_workspace_bytes": 1000,
            "windows_commit_peak_bytes": 600,
        },
        "windows_commit_capacity_bytes": 2000,
        "pack_temporary_bytes": 1,
        "ple_file": files[1]["path"],
    }


def fake_packer(command, runtime, log):
    pack = Path(command[command.index("--out") + 1])
    native = Path(command[command.index("--gguf") + 1])
    for name in PACK_REQUIRED:
        path = pack / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test", encoding="utf-8")
    (pack / "conversions.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "tool": "tools/iq_pack.py",
                "compat_bf16": "--compat-bf16" in command,
                "source_shards": [
                    {"name": path.name, "size": path.stat().st_size} for path in sorted(native.parent.glob("*.gguf"))
                ],
                "tensors": [],
            }
        ),
        encoding="utf-8",
    )
    if "--compat-bf16" in command:
        (pack / "compat-bf16.json").write_text(
            json.dumps({"rounding": "nearest-even", "tensors": []}), encoding="utf-8"
        )
    log.write_text("test packer, not real inference", encoding="utf-8")


def test_preparation_binds_complete_runtime_pack_and_weight_tier_plan(tmp_path):
    target, args = fixture(tmp_path)
    receipt = prepare(target, **args, pack_runner=fake_packer)
    assert receipt["status"] == "completed"
    launch = json.loads(args["launch_out"].read_text(encoding="utf-8"))
    backend = launch["backend"]
    assert backend["runtime_revision"] == RUNTIME_REVISION
    assert backend["python_environment"]["executable_sha256"] == backend["python_sha256"]
    assert "numpy" in backend["python_environment"]["dependencies"]
    assert backend["spec_tokens"] == 0
    assert backend["kv_type"] == "fp16"
    assert backend["conversion_manifest"]["complete"] is True
    assert "engine/helper.dll" in {item["path"] for item in backend["runtime_manifest"]["files"]}
    assert "data/expert-profile.bin" in {item["path"] for item in backend["runtime_manifest"]["files"]}
    assert backend["weight_tier_plan"]["budget"]["cpu_expert_cache_bytes"] == 100
    from vllm_omni.engine.weight_tiers import ArtifactManifest, WeightTierPlan

    assert (
        manifest_digest(backend["artifact_manifest"])
        == ArtifactManifest.from_dict(backend["artifact_manifest"]).manifest_sha256
    )
    plan = WeightTierPlan.from_dict(backend["weight_tier_plan"])
    assert plan.budget.resource_demands(include_windows_commit=True) == launch["resource_budget"]["demands"]
    assert backend["host_overhead_bytes"] == 400
    assert not (args["out_pack"] / "experts.bin").exists()


def test_corrupt_source_is_rejected_before_pack_tool_runs(tmp_path):
    target, args = fixture(tmp_path)
    (args["artifact_root"] / target["artifact_manifest"]["files"][1]["path"]).write_bytes(b"corrupt")
    calls = []
    with pytest.raises(ValueError, match="bytes changed"):
        prepare(target, **args, pack_runner=lambda *values: calls.append(values))
    assert calls == []
    receipt = json.loads(args["launch_out"].with_suffix(".prepare.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert not args["launch_out"].exists()


def test_existing_unbound_pack_is_never_adopted(tmp_path):
    target, args = fixture(tmp_path)
    args["out_pack"].mkdir()
    with pytest.raises(FileExistsError, match="unbound packs"):
        prepare(target, **args, pack_runner=fake_packer, reuse_bound_pack=True)


def test_reuse_bound_pack_verifies_files_and_never_reexecutes_packing(tmp_path):
    target, args = fixture(tmp_path)
    prepare(target, **args, pack_runner=fake_packer)
    args["launch_out"] = tmp_path / "second.json"
    calls = []
    receipt = prepare(target, **args, pack_runner=lambda *values: calls.append(values), reuse_bound_pack=True)
    assert receipt["reused_bound_pack"] is True
    assert calls == []
    (args["out_pack"] / "dense.bin").write_bytes(b"bad")
    args["launch_out"] = tmp_path / "third.json"
    with pytest.raises(ValueError, match="bytes changed"):
        prepare(target, **args, pack_runner=fake_packer, reuse_bound_pack=True)


def test_missing_tokenizer_and_extra_expert_copy_cannot_make_a_complete_receipt(tmp_path):
    target, args = fixture(tmp_path)

    def incomplete(command, runtime, log):
        fake_packer(command, runtime, log)
        (args["out_pack"] / "tokenizer/token_type.json").unlink()

    with pytest.raises(ValueError, match="incomplete prepared pack"):
        prepare(target, **args, pack_runner=incomplete)
    assert not args["launch_out"].exists()


def test_q4_compatibility_is_explicit_and_bound(tmp_path):
    target, args = fixture(tmp_path)
    target["target_id"] = "unsloth-q4-ssd"
    target["artifact_manifest"]["checkpoint"] = "unsloth/test"
    receipt = prepare(target, **args, pack_runner=fake_packer)
    assert "--compat-bf16" in receipt["pack_command"]
    assert receipt["conversion_manifest"]["upstream_compatibility_sha256"]


def test_component_budget_mismatch_is_rejected_before_source_packing(tmp_path):
    target, args = fixture(tmp_path)
    args["component_budget"]["cpu_expert_cache_bytes"] = 99
    with pytest.raises(ValueError, match="expert cache"):
        prepare(target, **args, pack_runner=fake_packer)
    assert not args["out_pack"].exists()


def test_runtime_symlink_escape_is_refused(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("unbound", encoding="utf-8")
    try:
        (root / "helper.py").symlink_to(outside)
    except OSError:
        pytest.skip("host does not permit symlink creation")
    with pytest.raises(ValueError, match="escapes root"):
        inventory(root, "test", "pinned", "test", runtime=True)


@pytest.mark.parametrize("change", [{"pack_temporary_bytes": -1}, {"gpu_index": -1}, {"kv_type": "automatic"}])
def test_invalid_controls_are_refused_before_packing(tmp_path, change):
    target, args = fixture(tmp_path)
    args.update(change)
    with pytest.raises(ValueError):
        prepare(target, **args, pack_runner=fake_packer)
    assert not args["out_pack"].exists()
