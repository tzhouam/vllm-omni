# SPDX-License-Identifier: Apache-2.0
"""Small real-file registration tests; synthetic build proof is explicit.

The production observation verifier has its own source/build contracts. These
tests exercise parent preparation, pack reuse, byte identity and admission;
they do not compile a native binary or claim model/device execution.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from benchmarks.edge_harness.strata_prepare import inventory, prepare
from benchmarks.edge_harness.strata_profile import RUNTIME_REVISION, canonical_hash, save_json
from benchmarks.edge_harness.strata_runtime import register_runtime_variant
from benchmarks.edge_harness.test_strata_prepare import fake_packer
from benchmarks.edge_harness.test_strata_prepare import fixture as preparation_fixture
from vllm_omni.engine.backends import strata_io
from vllm_omni.engine.weight_tiers import ArtifactManifest


@pytest.fixture
def variant(tmp_path, monkeypatch):
    target, args = preparation_fixture(tmp_path)
    prepare(target, **args, pack_runner=fake_packer)
    runtime = tmp_path / "variant-runtime"
    shutil.copytree(args["runtime_root"], runtime)
    (runtime / "engine/strata.exe").write_bytes(b"synthetic patched native executable")
    manifest = inventory(runtime, "fixture/Strata", RUNTIME_REVISION, "test", runtime=True)
    manifest_file = tmp_path / "variant-manifest.json"
    save_json(manifest_file, manifest)
    descriptor = {"engine_file": "engine/strata.exe", "fixture": "synthetic verifier boundary only"}
    identity = {"schema": "fixture-observed-runtime", "identity_sha256": "f" * 64, "three_tier_memory_qualified": False}
    calls = []

    def verify(value, root, files, engine):
        assert value == descriptor and root == runtime.resolve()
        assert engine == (runtime / "engine/strata.exe").resolve() and engine in files
        assert files == {path.resolve() for path in runtime.rglob("*") if path.is_file()}
        calls.append(True)
        return copy.deepcopy(identity)

    original_verifier = strata_io.verify_observation_runtime
    monkeypatch.setattr(strata_io, "verify_observation_runtime", verify)
    return {
        "target": target,
        "args": args,
        "runtime": runtime,
        "manifest": manifest_file,
        "descriptor": descriptor,
        "identity": identity,
        "calls": calls,
        "original_verifier": original_verifier,
        "out": tmp_path / "new-run/launch.json",
    }


def run(case, **kwargs):
    return register_runtime_variant(
        case["args"]["launch_out"],
        runtime_root=case["runtime"],
        runtime_manifest_path=case["manifest"],
        observation_runtime=case["descriptor"],
        launch_out=case["out"],
        **kwargs,
    )


def refresh_parent(case, mutate):
    path = case["args"]["launch_out"]
    launch = json.loads(path.read_text(encoding="utf-8"))
    mutate(launch)
    save_json(path, launch)
    receipt_path = Path(launch["preparation_receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["launch_sha256"] = canonical_hash(launch)
    save_json(receipt_path, receipt)


def test_registration_preserves_parent_pack_and_budgets_with_distinct_runtime(variant):
    parent_path = variant["args"]["launch_out"]
    pack = variant["args"]["out_pack"]
    binding = pack.with_name(pack.name + ".omni-binding.json")
    before = {path: path.read_bytes() for path in (parent_path, binding, *pack.rglob("*")) if path.is_file()}
    receipt = run(variant)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    derived = json.loads(variant["out"].read_text(encoding="utf-8"))
    assert variant["calls"] == [True]
    assert all(path.read_bytes() == data for path, data in before.items())
    assert receipt["status"] == "registered_not_executed" and receipt["qualification_created"] is False
    assert receipt["parent_binding_rewritten"] is False
    assert receipt["route_id"] != parent["backend"]["route_id"]
    assert derived["backend"]["route_id"] == derived["backend"]["weight_tier_plan"]["route_id"]
    assert derived["backend"]["observation_runtime_identity_sha256"] == "f" * 64
    assert derived["backend"]["prepared_pack_manifest"] == parent["backend"]["prepared_pack_manifest"]
    assert derived["backend"]["artifact_manifest"] == parent["backend"]["artifact_manifest"]
    assert derived["backend"]["conversion_manifest"] == parent["backend"]["conversion_manifest"]
    assert derived["resource_budget"]["capacities"] == parent["resource_budget"]["capacities"]
    runtime_bytes = ArtifactManifest.from_dict(derived["backend"]["runtime_manifest"]).total_size_bytes
    for pool, amount in parent["resource_budget"]["demands"].items():
        assert derived["resource_budget"]["demands"][pool] == amount + (runtime_bytes if pool == "ssd" else 0)
    assert canonical_hash(derived) == receipt["launch_sha256"]


@pytest.mark.parametrize("target", ["source", "pack", "runtime", "variant-runtime"])
def test_changed_file_bytes_are_refused_before_outputs(variant, target):
    args = variant["args"]
    path = {
        "source": args["artifact_root"] / "test-00001-of-00002.gguf",
        "pack": args["out_pack"] / "dense.bin",
        "runtime": args["runtime_root"] / "serve/server.py",
        "variant-runtime": variant["runtime"] / "engine/strata.exe",
    }[target]
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        run(variant)
    assert not variant["out"].exists()


@pytest.mark.parametrize("target", ["pack", "variant-runtime"])
def test_unbound_code_or_pack_files_are_rejected(variant, target):
    root = variant["args"]["out_pack"] if target == "pack" else variant["runtime"]
    (root / "unbound.pyc").write_bytes(b"unexpected executable code")
    with pytest.raises(ValueError, match="unbound"):
        run(variant)


def test_changed_packer_code_is_refused_even_with_a_new_valid_manifest(variant):
    (variant["runtime"] / "tools/iq_pack.py").write_text("changed", encoding="utf-8")
    save_json(variant["manifest"], inventory(variant["runtime"], "fixture/Strata", RUNTIME_REVISION, "test"))
    with pytest.raises(ValueError, match="packing-tool"):
        run(variant)
    assert not variant["out"].exists()


def test_parent_preparation_launch_identity_cannot_be_reassigned(variant):
    path = variant["args"]["launch_out"]
    parent = json.loads(path.read_text(encoding="utf-8"))
    parent["backend"]["kv_type"] = "int8"
    save_json(path, parent)
    with pytest.raises(ValueError, match="preparation receipt"):
        run(variant)


def test_parent_pack_conversion_identity_cannot_be_reassigned(variant):
    pack = variant["args"]["out_pack"]
    path = pack.with_name(pack.name + ".omni-binding.json")
    binding = json.loads(path.read_text(encoding="utf-8"))
    binding["conversion_manifest"]["conversions"].append({"invented": True})
    save_json(path, binding)
    with pytest.raises(ValueError, match="conversion/format"):
        run(variant)


def test_additional_runtime_storage_is_refused_without_capacity_increase(variant):
    refresh_parent(
        variant,
        lambda launch: launch["resource_budget"]["capacities"].update(
            ssd=launch["resource_budget"]["demands"]["ssd"],
        ),
    )
    with pytest.raises(ValueError, match="resource capacity"):
        run(variant)


def test_expected_identity_source_verifier_failure_propagates(variant, monkeypatch):
    monkeypatch.setattr(strata_io, "verify_observation_runtime", variant["original_verifier"])
    with pytest.raises(ValueError, match="descriptor"):
        run(variant)
    assert not variant["out"].exists()


def test_existing_outputs_or_reused_route_ids_are_never_overwritten(variant):
    parent = json.loads(variant["args"]["launch_out"].read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="distinct bounded route"):
        run(variant, route_id=parent["backend"]["route_id"])
    variant["out"].parent.mkdir()
    variant["out"].write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        run(variant)
    assert variant["out"].read_bytes() == b"existing"


def test_registration_cannot_write_inside_model_or_runtime_bundle(variant):
    variant["out"] = variant["args"]["out_pack"] / "new-launch.json"
    with pytest.raises(ValueError, match="outside immutable"):
        run(variant)
