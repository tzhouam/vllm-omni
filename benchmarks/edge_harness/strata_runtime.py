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


def register_runtime_variant(
    parent_launch_path: Path,
    *,
    runtime_root: Path,
    runtime_manifest_path: Path,
    observation_runtime: Mapping[str, Any],
    launch_out: Path,
    route_id: str | None = None,
) -> dict[str, Any]:
    """Verify a base preparation and create a new experimental launch identity.

    Memory ceilings are inherited declarations and require fresh startup
    checks. The additional runtime's full disk size is explicitly charged.
    Only a native-runtime variant is accepted; packing tools and pack format
    must remain byte-identical to the original preparation.
    """
    from vllm_omni.engine.backends.strata_io import verify_observation_runtime
    from vllm_omni.engine.weight_tiers import ArtifactManifest, WeightTierPlan

    parent_path = parent_launch_path.resolve(strict=True)
    launch = _object(parent_path)
    if launch.get("schema") != "omni-strata-launch-v1":
        raise ValueError("parent must be a prepared Strata launch")
    backend = launch["backend"]
    if (
        backend.get("name") != "external.strata.text.v1"
        or backend.get("runtime_revision") != RUNTIME_REVISION
        or backend.get("observation_runtime") is not None
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
    new_tier = WeightTierPlan.from_dict(tier_data)
    new_demands = dict(demands, ssd=demands["ssd"] + variant.total_size_bytes)
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
        observation_runtime=descriptor,
        observation_runtime_identity_sha256=identity["identity_sha256"],
        weight_tier_plan=new_tier.to_dict(),
    )
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
    for name in ("parent-launch", "runtime-root", "runtime-manifest", "observation-runtime", "launch-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--route-id")
    args = parser.parse_args()
    receipt = register_runtime_variant(
        args.parent_launch,
        runtime_root=args.runtime_root,
        runtime_manifest_path=args.runtime_manifest,
        observation_runtime=_object(args.observation_runtime),
        launch_out=args.launch_out,
        route_id=args.route_id,
    )
    print(json.dumps({key: receipt[key] for key in ("status", "route_id", "launch_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
