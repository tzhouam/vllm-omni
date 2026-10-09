# SPDX-License-Identifier: Apache-2.0
"""Register a separately verified Strata runtime against an existing bound pack.

This tool does not pack weights, execute inference, create qualification, or
rewrite the original preparation binding. A locally verified build receipt
records provenance; it is not an independent reproducible rebuild.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.edge_harness.strata_prepare import (
    SCHEMA as PREPARATION_SCHEMA,
)
from benchmarks.edge_harness.strata_prepare import (
    manifest_digest,
    read_conversion_receipt,
)
from benchmarks.edge_harness.strata_profile import RUNTIME_REVISION, canonical_hash, file_hash

SCHEMA = "omni-strata-runtime-registration-v1"
EXECUTION_IDENTITY_SCHEMA = "omni-strata-combined-static-runtime-identity-v2"


def _object(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate registration JSON key")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite registration JSON")),
    )
    if not isinstance(value, dict):
        raise ValueError("registration input must be a JSON object")
    return value


def _complete_files(manifest, root: Path, *, runtime: bool = False) -> set[Path]:
    verified = set(manifest.verify(root))
    actual = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and not (runtime and path.relative_to(root).parts[0] == ".git")
    }
    if actual != verified:
        raise ValueError("runtime/pack contains an unbound file")
    return verified


def _packing_code(manifest) -> dict[str, tuple[int, str]]:
    # Bind the whole local tools Python tree, rather than guess iq_pack's
    # transitive imports. Third-party Python identity stays unchanged below.
    result = {
        item.path: (item.size_bytes, item.sha256)
        for item in manifest.files
        if item.path.startswith("tools/") and item.path.endswith((".py", ".pyc"))
    }
    if not {"tools/iq_pack.py", "tools/gguf_reader.py", "tools/strata_tokenizer.py"} <= result.keys():
        raise ValueError("packing-tool code identity is incomplete")
    return result


def _execution_variant(config, runtime: Path, verified_files: set[Path], engine_file: str, runtime_manifest):
    """Use the Stage's reviewed source loader, never caller-provided verifiers.

    This verifies archived bytes and the loaded IO adapter only. Native owner,
    engine ABI, module paths and request completion still require a real load.
    """
    from vllm_omni.engine.backends import strata, strata_io

    settings = strata._execution_settings(copy.deepcopy(dict(config)), runtime)
    modules = strata._load_execution_adapters(runtime, verified_files)
    descriptor = _object(runtime / settings["descriptor_file"])
    if descriptor.get("engine_file") != engine_file:
        raise ValueError("runtime variant must retain the native engine entry point")
    identity = modules["strata_exec_runtime"].verify_combined_runtime(settings["descriptor_file"], runtime)
    engine = (runtime / engine_file).resolve(strict=True)
    if (
        identity.get("schema") != EXECUTION_IDENTITY_SCHEMA
        or identity.get("identity_sha256") != settings["static_identity_sha256"]
        or engine not in verified_files
        or identity.get("native_executable_sha256") != file_hash(engine)
        or identity.get("runtime_binding") is not None
        or identity.get("compiled_engine_ABI_verified") is not False
        or identity.get("observer_layout_scope") != "compiled_standalone_fixture_reference_only"
    ):
        raise ValueError("combined execution runtime identity differs or claims live eligibility")
    member_file = descriptor["manifest_file"]
    member = next((item for item in runtime_manifest.files if item.path == member_file), None)
    if (
        member is None
        or (runtime / member_file).resolve(strict=True) not in verified_files
        or member.sha256 != identity.get("runtime_manifest_sha256")
    ):
        raise ValueError("combined member manifest raw hash differs from its complete runtime record")
    member_reference = {"file": member_file, "sha256": member.sha256}
    context_identity = strata_io.combined_io_adapter_identity(
        runtime, identity, settings["source_context_file"], manifest_file=descriptor["manifest_file"]
    )
    return settings, identity, context_identity, member_reference


def register_runtime_variant(
    parent_launch_path: Path,
    *,
    runtime_root: Path,
    runtime_manifest_path: Path,
    observation_runtime: Mapping[str, Any] | None = None,
    execution_observation: Mapping[str, Any] | None = None,
    launch_out: Path,
    route_id: str | None = None,
) -> dict[str, Any]:
    """Verify a base preparation and create a new experimental launch identity.

    Memory ceilings are inherited declarations and require fresh startup
    checks. The additional runtime's full disk size is explicitly charged.
    Only a native-runtime variant is accepted; packing tools and pack format
    must remain byte-identical to the original preparation.
    """
    from vllm_omni.engine.weight_tiers import ArtifactManifest, WeightTierPlan

    if (observation_runtime is None) == (execution_observation is None):
        raise ValueError("select exactly one legacy IO or combined execution observation route")
    parent_path = parent_launch_path.resolve(strict=True)
    launch = _object(parent_path)
    if launch.get("schema") != "omni-strata-launch-v1":
        raise ValueError("parent must be a prepared Strata launch")
    backend = launch["backend"]
    if (
        backend.get("name") != "external.strata.text.v1"
        or backend.get("runtime_revision") != RUNTIME_REVISION
        or backend.get("observation_runtime") is not None
        or backend.get("observation_runtime_identity_sha256") is not None
        or backend.get("execution_observation") is not None
        or launch.get("execution_observation") is not None
        or launch.get("runtime_registration") is not None
    ):
        raise ValueError("parent must be the pinned original Strata preparation")
    parent_runtime = Path(backend["runtime_root"]).resolve(strict=True)
    new_root = runtime_root.resolve(strict=True)
    source_root = Path(backend["artifact_root"]).resolve(strict=True)
    pack_root = Path(backend["prepared_model_dir"]).resolve(strict=True)
    destination = launch_out.resolve()
    receipt_out = destination.with_suffix(".registration.json")
    if new_root == parent_runtime:
        raise ValueError("runtime variant needs a separate root")
    if destination == parent_path or any(
        path.is_relative_to(root)
        for path in (destination, receipt_out)
        for root in (parent_runtime, new_root, source_root, pack_root)
    ):
        raise ValueError("registration outputs must stay outside immutable runtime/model bundles")
    if destination.exists() or receipt_out.exists():
        raise FileExistsError("registration output already exists")

    source = ArtifactManifest.from_dict(backend["artifact_manifest"])
    original = ArtifactManifest.from_dict(backend["runtime_manifest"])
    packed = ArtifactManifest.from_dict(backend["prepared_pack_manifest"])
    new_manifest_path = runtime_manifest_path.resolve(strict=True)
    variant = ArtifactManifest.from_dict(_object(new_manifest_path))
    if (
        not re.fullmatch(r"[a-f0-9]{40}", source.revision)
        or original.revision != RUNTIME_REVISION
        or variant.revision != RUNTIME_REVISION
    ):
        raise ValueError("source and runtime revisions must be pinned")
    if variant.manifest_sha256 == original.manifest_sha256:
        raise ValueError("runtime variant is identical to its parent")
    _complete_files(original, parent_runtime, runtime=True)
    source.verify(source_root)
    _complete_files(packed, pack_root)
    verified_runtime = _complete_files(variant, new_root, runtime=True)
    packing_code = _packing_code(original)
    if packing_code != _packing_code(variant):
        raise ValueError("runtime variant changed packing-tool code or format identity")

    preparation_path = Path(launch["preparation_receipt"]).resolve(strict=True)
    preparation = _object(preparation_path)
    binding_path = pack_root.with_name(pack_root.name + ".omni-binding.json")
    binding = _object(binding_path)
    expected_hashes = {
        "source_manifest_sha256": source.manifest_sha256,
        "runtime_manifest_sha256": original.manifest_sha256,
        "prepared_manifest_sha256": packed.manifest_sha256,
    }
    if (
        preparation.get("schema") != PREPARATION_SCHEMA
        or preparation.get("status") != "completed"
        or preparation.get("launch_sha256") != canonical_hash(launch)
        or any(preparation.get(key) != value for key, value in expected_hashes.items())
        or Path(preparation["binding_path"]).resolve() != binding_path
        or Path(preparation["artifact_root"]).resolve() != source_root
        or Path(preparation["pack"]).resolve() != pack_root
        or preparation.get("python_sha256") != backend.get("python_sha256")
        or preparation.get("python_environment") != backend.get("python_environment")
    ):
        raise ValueError("parent preparation receipt does not bind this launch/source/runtime/pack")
    if (
        binding.get("schema") != PREPARATION_SCHEMA
        or binding.get("target_sha256") != preparation.get("target_sha256")
        or binding.get("source_manifest_sha256") != source.manifest_sha256
        or binding.get("runtime_manifest_sha256") != original.manifest_sha256
        or manifest_digest(binding["prepared_pack_manifest"]) != packed.manifest_sha256
        or Path(binding["artifact_root"]).resolve() != source_root
    ):
        raise ValueError("parent pack binding differs from the original preparation")
    conversion = read_conversion_receipt(
        pack_root,
        source.to_dict(),
        packed.to_dict(),
        preparation["compatibility_mode"],
    )
    if any(
        canonical_hash(value) != canonical_hash(conversion)
        for value in (
            backend["conversion_manifest"],
            preparation["conversion_manifest"],
            binding["conversion_manifest"],
        )
    ):
        raise ValueError("parent conversion/format receipt differs")
    python = Path(backend["python_bin"]).resolve(strict=True)
    if file_hash(python) != backend["python_sha256"]:
        raise ValueError("parent Python executable identity changed")
    settings = context_identity = member_reference = None
    if execution_observation is not None:
        settings, identity, context_identity, member_reference = _execution_variant(
            execution_observation, new_root, verified_runtime, backend["engine_file"], variant
        )
        if any(
            Path(settings["receipt_root"]).is_relative_to(root) for root in (parent_runtime, source_root, pack_root)
        ):
            raise ValueError("execution receipts must stay outside immutable runtime/model bundles")
    else:
        from vllm_omni.engine.backends.strata_io import verify_observation_runtime

        descriptor = copy.deepcopy(dict(observation_runtime))
        if descriptor.get("engine_file") != backend["engine_file"]:
            raise ValueError("runtime variant must retain the native engine entry point")
        engine = (new_root / descriptor["engine_file"]).resolve(strict=True)
        identity = verify_observation_runtime(descriptor, new_root, verified_runtime, engine)

    tier = WeightTierPlan.from_dict(backend["weight_tier_plan"])
    if (
        tier.route_id != backend["route_id"]
        or tier.backend != backend["name"]
        or tier.backend_revision != RUNTIME_REVISION
        or tier.artifact_manifest_sha256 != source.manifest_sha256
    ):
        raise ValueError("parent tier plan does not bind the source/runtime/route")
    resources = launch["resource_budget"]
    demands = tier.budget.resource_demands(
        gpu_pool=backend["gpu_pool"],
        include_wsl="wsl_ram" in resources["demands"],
        include_windows_commit="windows_commit" in resources["demands"],
    )
    if resources["demands"] != demands:
        raise ValueError("parent resource demands differ from the typed tier plan")
    new_id = route_id or tier.route_id + "-observed-" + identity["identity_sha256"][:12]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", new_id) or new_id == tier.route_id:
        raise ValueError("runtime variant needs a distinct bounded route ID")
    tier_data = tier.to_dict()
    tier_data["route_id"] = new_id
    tier_data["budget"]["ssd_artifact_bytes"] += variant.total_size_bytes
    if settings is not None:
        # The Stage adds this workspace separately to its host-overhead formula;
        # leave the original model/cache/I/O controls unchanged and charge once.
        workspace = settings["workspace_bytes"]
        for field in ("host_workspace_bytes", "host_loading_peak_bytes", "windows_commit_peak_bytes"):
            tier_data["budget"][field] += workspace
        tier_data["budget"]["ssd_temporary_bytes"] += settings["receipt_storage_bytes"]
    new_tier = WeightTierPlan.from_dict(tier_data)
    new_demands = new_tier.budget.resource_demands(
        gpu_pool=backend["gpu_pool"],
        include_wsl="wsl_ram" in resources["demands"],
        include_windows_commit="windows_commit" in resources["demands"],
    )
    if any(
        type(resources["capacities"].get(pool)) is not int or amount > resources["capacities"][pool]
        for pool, amount in new_demands.items()
    ):
        raise ValueError("runtime variant exceeds an inherited resource capacity")

    derived = copy.deepcopy(launch)
    target = derived["backend"]
    target.update(
        runtime_root=str(new_root),
        runtime_manifest=variant.to_dict(),
        route_id=new_id,
        weight_tier_plan=new_tier.to_dict(),
    )
    if settings is not None:
        target["execution_observation"] = settings
    else:
        target["observation_runtime"] = descriptor
        target["observation_runtime_identity_sha256"] = identity["identity_sha256"]
    provenance = {
        "schema": SCHEMA,
        "parent_launch_sha256": canonical_hash(launch),
        "parent_launch_file_sha256": file_hash(parent_path),
        "parent_preparation_receipt_sha256": file_hash(preparation_path),
        "parent_pack_binding_sha256": file_hash(binding_path),
        "parent_runtime_manifest_sha256": original.manifest_sha256,
        "runtime_manifest_sha256": variant.manifest_sha256,
        "observation_runtime_identity_sha256": identity["identity_sha256"],
        "packing_code_sha256": canonical_hash(packing_code),
        "source_manifest_sha256": source.manifest_sha256,
        "prepared_manifest_sha256": packed.manifest_sha256,
        "conversion_manifest_sha256": canonical_hash(conversion),
        "runtime_disk_bytes_added": variant.total_size_bytes,
        "scope": "locally verified build provenance; no independent rebuild, inference or qualification",
    }
    if settings is not None:
        provenance.update(
            runtime_identity_schema=identity["schema"],
            observation_kind="combined_execution",
            execution_observation_schema=settings["schema"],
            observer_workspace_bytes_added=settings["workspace_bytes"],
            observer_receipt_storage_bytes_added=settings["receipt_storage_bytes"],
            combined_io_adapter_identity=context_identity,
            combined_member_manifest=member_reference,
            scope="verified combined archived build/source bytes; live owner/modules/engine ABI and inference pending",
        )
    target["runtime_provenance"] = provenance
    derived["runtime_provenance"] = provenance
    derived["runtime_registration"] = provenance
    derived["resource_budget"]["demands"] = new_demands
    derived["preparation_receipt"] = str(preparation_path)
    derived["admission_preflight"] = {
        "admitted": True,
        "over_budget_pools": [],
        "scope": "inherited preparation ceilings; fresh startup admission remains required",
    }
    receipt = {
        "schema": SCHEMA,
        "status": "registered_not_executed",
        **provenance,
        "parent_launch": str(parent_path),
        "launch_out": str(destination),
        "launch_sha256": canonical_hash(derived),
        "route_id": new_id,
        "observation_runtime": identity,
        "qualification_created": False,
        "parent_binding_rewritten": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    for path, value in ((receipt_out, receipt), (destination, derived)):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-launch", "runtime-root", "runtime-manifest", "launch-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    observation = parser.add_mutually_exclusive_group(required=True)
    observation.add_argument("--observation-runtime", type=Path)
    observation.add_argument("--execution-observation", type=Path)
    parser.add_argument("--route-id")
    args = parser.parse_args()
    receipt = register_runtime_variant(
        args.parent_launch,
        runtime_root=args.runtime_root,
        runtime_manifest_path=args.runtime_manifest,
        observation_runtime=_object(args.observation_runtime) if args.observation_runtime else None,
        execution_observation=_object(args.execution_observation) if args.execution_observation else None,
        launch_out=args.launch_out,
        route_id=args.route_id,
    )
    print(json.dumps({key: receipt[key] for key in ("status", "route_id", "launch_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
