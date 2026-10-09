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
import types
from pathlib import Path

import pytest

from benchmarks.edge_harness.strata_prepare import inventory, prepare
from benchmarks.edge_harness.strata_profile import RUNTIME_REVISION, canonical_hash, file_hash, save_json
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


@pytest.fixture
def execution_variant(variant, monkeypatch):
    from vllm_omni.engine.backends import strata

    runtime = variant["runtime"].resolve()
    descriptor = {"engine_file": "engine/strata.exe", "manifest_file": "members.json"}
    save_json(runtime / "execution.json", descriptor)
    save_json(runtime / "members.json", {"schema": "omni-strata-combined-members-v2", "files": []})
    save_json(runtime / "context.json", {"fixture": "synthetic context-verifier boundary"})
    save_json(variant["manifest"], inventory(runtime, "fixture/Strata", RUNTIME_REVISION, "test", runtime=True))
    receipts = variant["out"].parent.parent / "observer-receipts"
    receipts.mkdir()
    config = {
        "schema": strata._EXECUTION_CONFIG_SCHEMA,
        "descriptor_file": "execution.json",
        "static_identity_sha256": "e" * 64,
        "source_context_file": "context.json",
        "receipt_root": str(receipts),
        "workspace_bytes": 512 << 20,
        "receipt_storage_bytes": 16 << 20,
    }
    identity = {
        "schema": "omni-strata-combined-static-runtime-identity-v2",
        "identity_sha256": "e" * 64,
        "native_executable_sha256": file_hash(runtime / "engine/strata.exe"),
        "runtime_manifest_sha256": file_hash(runtime / "members.json"),
        "runtime_binding": None,
        "compiled_engine_ABI_verified": False,
        "observer_layout_scope": "compiled_standalone_fixture_reference_only",
    }
    calls = []

    def verify(name, root):
        assert name == "execution.json" and root == runtime
        assert json.loads((root / name).read_text(encoding="utf-8")) == descriptor
        calls.append("static")
        return copy.deepcopy(identity)

    def loader(root, files):
        assert root == runtime
        assert files == {path.resolve() for path in runtime.rglob("*") if path.is_file()}
        calls.append("source_pinned_loader")
        return {"strata_exec_runtime": types.SimpleNamespace(verify_combined_runtime=verify)}

    def context(root, actual, name, *, manifest_file):
        assert root == runtime and actual == identity and name == "context.json" and manifest_file == "members.json"
        calls.append("loaded_io_context")
        return {"schema": "synthetic-actual-IO-boundary", "io_adapter_sha256": "d" * 64}

    monkeypatch.setattr(strata, "_load_execution_adapters", loader)
    monkeypatch.setattr(strata_io, "combined_io_adapter_identity", context, raising=False)
    refresh_parent(
        variant,
        lambda launch: launch["resource_budget"]["capacities"].update(
            host_ram=2 << 30, windows_commit=2 << 30, ssd=3 << 30
        ),
    )
    return variant | {"execution_config": config, "execution_identity": identity, "execution_calls": calls}


def run_execution(case, **kwargs):
    return register_runtime_variant(
        case["args"]["launch_out"],
        runtime_root=case["runtime"],
        runtime_manifest_path=case["manifest"],
        execution_observation=case["execution_config"],
        launch_out=case["out"],
        **kwargs,
    )


def test_combined_registration_binds_actual_boundaries_and_charges_workspace_once(execution_variant):
    case = execution_variant
    parent_file = case["args"]["launch_out"]
    binding = case["args"]["out_pack"].with_name("pack.omni-binding.json")
    before = {path: path.read_bytes() for path in (parent_file, binding)}
    parent = json.loads(parent_file.read_text(encoding="utf-8"))
    receipt = run_execution(case)
    derived = json.loads(case["out"].read_text(encoding="utf-8"))
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert case["calls"] == []  # Never falls back to the legacy verifier.
    assert case["execution_calls"] == ["source_pinned_loader", "static", "loaded_io_context"]
    assert receipt["qualification_created"] is False and receipt["status"] == "registered_not_executed"
    assert receipt["observation_kind"] == "combined_execution"
    assert receipt["runtime_identity_schema"] == case["execution_identity"]["schema"]
    assert receipt["observation_runtime"] == case["execution_identity"]
    member_reference = {"file": "members.json", "sha256": case["execution_identity"]["runtime_manifest_sha256"]}
    assert receipt["combined_member_manifest"] == member_reference
    assert derived["backend"]["runtime_provenance"]["combined_member_manifest"] == member_reference
    assert receipt["runtime_manifest_sha256"] != member_reference["sha256"]
    assert derived["backend"]["execution_observation"] == case["execution_config"]
    assert "observation_runtime" not in derived["backend"]
    assert "observation_runtime_identity_sha256" not in derived["backend"]
    assert derived["backend"]["host_overhead_bytes"] == parent["backend"]["host_overhead_bytes"]
    before_budget = parent["backend"]["weight_tier_plan"]["budget"]
    after_budget = derived["backend"]["weight_tier_plan"]["budget"]
    workspace, receipts = case["execution_config"]["workspace_bytes"], case["execution_config"]["receipt_storage_bytes"]
    runtime_bytes = ArtifactManifest.from_dict(derived["backend"]["runtime_manifest"]).total_size_bytes
    for pool, amount in parent["resource_budget"]["demands"].items():
        extra = (
            workspace if pool in {"host_ram", "windows_commit"} else runtime_bytes + receipts if pool == "ssd" else 0
        )
        assert derived["resource_budget"]["demands"][pool] == amount + extra
    for field in ("host_workspace_bytes", "host_loading_peak_bytes", "windows_commit_peak_bytes"):
        assert after_budget[field] == before_budget[field] + workspace
    assert after_budget["ssd_temporary_bytes"] == before_budget["ssd_temporary_bytes"] + receipts
    # The exact Stage's separate-overhead formula fits the single workspace charge.
    backend = derived["backend"]
    assert (
        backend["expert_ram_budget_bytes"] + backend["host_overhead_bytes"] + backend["max_io_bytes"] + workspace
        == derived["resource_budget"]["demands"]["host_ram"]
    )
    changed = {
        "runtime_root",
        "runtime_manifest",
        "route_id",
        "weight_tier_plan",
        "runtime_provenance",
        "execution_observation",
    }
    assert {key: value for key, value in backend.items() if key not in changed} == {
        key: value for key, value in parent["backend"].items() if key not in changed
    }


@pytest.mark.parametrize("pool", ["host_ram", "windows_commit", "ssd"])
def test_combined_workspace_and_receipts_cannot_bypass_inherited_ceiling(execution_variant, pool):
    refresh_parent(
        execution_variant,
        lambda launch: launch["resource_budget"]["capacities"].update(
            {pool: launch["resource_budget"]["demands"][pool]}
        ),
    )
    with pytest.raises(ValueError, match="resource capacity"):
        run_execution(execution_variant)
    assert not execution_variant["out"].exists()


@pytest.mark.parametrize(
    "field,value", [("workspace_bytes", True), ("workspace_bytes", 2 << 20), ("receipt_storage_bytes", 0)]
)
def test_combined_uses_actual_stage_config_schema_and_minimums(execution_variant, field, value):
    execution_variant["execution_config"][field] = value
    with pytest.raises(ValueError, match="bounded budget"):
        run_execution(execution_variant)
    assert execution_variant["execution_calls"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity_sha256", "a" * 64),
        ("schema", "legacy-runtime"),
        ("native_executable_sha256", "b" * 64),
        ("compiled_engine_ABI_verified", True),
        ("runtime_binding", {"declared": True}),
    ],
)
def test_combined_static_identity_cannot_be_substituted_or_promoted(execution_variant, field, value):
    execution_variant["execution_identity"][field] = value
    with pytest.raises(ValueError, match="combined execution runtime identity"):
        run_execution(execution_variant)
    assert "loaded_io_context" not in execution_variant["execution_calls"]
    assert not execution_variant["out"].exists()


def test_combined_pinned_loader_failure_never_uses_legacy_verifier(execution_variant, monkeypatch):
    from vllm_omni.engine.backends import strata

    def refuse(*args):
        raise ValueError("execution adapter is not a reviewed source preimage")

    monkeypatch.setattr(strata, "_load_execution_adapters", refuse)
    with pytest.raises(ValueError, match="reviewed source"):
        run_execution(execution_variant)
    assert execution_variant["calls"] == [] and not execution_variant["out"].exists()


def test_combined_member_manifest_raw_hash_cannot_be_replaced_by_outer_digest(execution_variant):
    case = execution_variant
    case["execution_identity"]["runtime_manifest_sha256"] = ArtifactManifest.from_dict(
        json.loads(case["manifest"].read_text(encoding="utf-8"))
    ).manifest_sha256
    with pytest.raises(ValueError, match="member manifest raw hash"):
        run_execution(case)
    assert "loaded_io_context" not in case["execution_calls"] and not case["out"].exists()


def test_combined_member_manifest_must_be_in_complete_outer_manifest(execution_variant, monkeypatch):
    case = execution_variant
    descriptor = case["runtime"] / "execution.json"
    save_json(descriptor, {"engine_file": "engine/strata.exe", "manifest_file": "absent.json"})
    save_json(case["manifest"], inventory(case["runtime"], "fixture/Strata", RUNTIME_REVISION, "test", runtime=True))
    # This deliberately bypasses only the independent static verifier boundary;
    # the registrar still refuses a member omitted from the full outer inventory.
    from vllm_omni.engine.backends import strata

    def loader(root, files):
        return {
            "strata_exec_runtime": types.SimpleNamespace(
                verify_combined_runtime=lambda *args: copy.deepcopy(case["execution_identity"])
            )
        }

    # This test exercises the registrar's independent member binding, while the
    # actual static verifier's descriptor checks have their own focused tests.
    monkeypatch.setattr(strata, "_load_execution_adapters", loader)
    with pytest.raises(ValueError, match="member manifest raw hash"):
        run_execution(case)
    assert not case["out"].exists()


def test_combined_loaded_io_source_context_failure_is_not_registration(execution_variant, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("combined IO loaded source preimage differs")

    monkeypatch.setattr(strata_io, "combined_io_adapter_identity", refuse)
    with pytest.raises(ValueError, match="source preimage"):
        run_execution(execution_variant)
    assert not execution_variant["out"].exists()


@pytest.mark.parametrize(
    "observer", ["observation_runtime", "observation_runtime_identity_sha256", "execution_observation"]
)
def test_observed_parents_cannot_be_registered_again(execution_variant, observer):
    refresh_parent(execution_variant, lambda launch: launch["backend"].update({observer: {}}))
    with pytest.raises(ValueError, match="pinned original"):
        run_execution(execution_variant)
    assert execution_variant["execution_calls"] == []


def test_exactly_one_observer_api_is_required(execution_variant):
    with pytest.raises(ValueError, match="exactly one"):
        run_execution(execution_variant, observation_runtime=execution_variant["descriptor"])
    with pytest.raises(ValueError, match="exactly one"):
        register_runtime_variant(
            execution_variant["args"]["launch_out"],
            runtime_root=execution_variant["runtime"],
            runtime_manifest_path=execution_variant["manifest"],
            launch_out=execution_variant["out"],
        )


def test_combined_receipts_cannot_write_inside_original_pack(execution_variant):
    execution_variant["execution_config"]["receipt_root"] = str(execution_variant["args"]["out_pack"])
    with pytest.raises(ValueError, match="receipts must stay outside"):
        run_execution(execution_variant)
    assert not execution_variant["out"].exists()
