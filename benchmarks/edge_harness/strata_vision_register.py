# SPDX-License-Identifier: Apache-2.0
"""Derive an explicitly experimental CPU-image route from verified text preparation.

Registration verifies files and probes current capacity. It never instantiates a
StageClient, loads neural weights, rewrites a preparation, or grants qualification.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from benchmarks.edge_harness import strata_runtime
from benchmarks.edge_harness.strata_profile import RUNTIME_REVISION, canonical_hash, file_hash

SCHEMA = "omni-strata-cpu-image-registration-v1"
BACKEND = "external.strata.multimodal.v1"
TEXT_BACKEND = "external.strata.text.v1"


def _helpers():
    from vllm_omni.engine.backends import strata, strata_io, strata_vision
    from vllm_omni.engine.weight_tiers import ArtifactManifest, WeightTierPlan

    return strata, strata_io, strata_vision, ArtifactManifest, WeightTierPlan


def _object(path):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 8 << 20:
        raise ValueError("image registration JSON exceeds bounded metadata size")
    return strata_runtime._object(path)


def _positive(value, name):
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError(name + " must be a positive bounded integer")
    return value


def _outside(paths, roots):
    if any(path.is_relative_to(root) for path in paths for root in roots):
        raise ValueError("image outputs must stay outside immutable source/pack/runtime/projector bundles")


def _disk_snapshot(roots):
    """Capacity observation only; does not measure physical SSD traffic/type."""
    result = []
    for root in roots:
        root = Path(root).resolve(strict=True)
        usage = shutil.disk_usage(root)
        result.append(
            {
                "path": str(root),
                "filesystem_device": root.stat().st_dev,
                "total_bytes": usage.total,
                "free_bytes": usage.free,
                "used_bytes": usage.used,
            }
        )
    return result


def _check_disk(rows, demands):
    if (
        not rows
        or any(type(row.get("filesystem_device")) is not int or row["filesystem_device"] < 0 for row in rows)
        or len({row["filesystem_device"] for row in rows}) != 1
    ):
        raise ValueError("one SSD ledger pool requires observed shared filesystem; multiple/unverified volumes refused")
    if any(type(row.get(key)) is not int or row[key] <= 0 for row in rows for key in ("total_bytes", "free_bytes")):
        raise ValueError("SSD volume capacity observation unavailable")
    if demands["ssd"] > min(row["total_bytes"] for row in rows):
        raise ValueError("declared managed SSD artifact/temporary claim exceeds observed volume capacity")


def _trusted_source_identity(strata, strata_io, vision):
    # Caller hashes are never accepted as a replacement for observed code bytes.
    modules = {
        "image_registrar": Path(__file__),
        "runtime_registrar": Path(strata_runtime.__file__),
        "strata_backend": Path(strata.__file__),
        "native_io_adapter": Path(strata_io.__file__),
        "vision_adapter": Path(vision.__file__),
        "image_stage_backend": Path(vision.__file__).with_name("strata_multimodal.py"),
    }
    return {role: file_hash(path.resolve(strict=True)) for role, path in modules.items()}


def _verified_text_intermediate(text_path, receipt_path, *, manifest_type):
    """Replay the existing registrar against real bytes, then compare metadata.

    The metadata-only temporary output avoids modifying the existing intermediate
    and uses the exact existing route ID. Only its destination path may differ.
    """
    launch, receipt = _object(text_path), _object(receipt_path)
    backend = launch.get("backend", {})
    if (
        launch.get("schema") != "omni-strata-launch-v1"
        or backend.get("name") != TEXT_BACKEND
        or backend.get("runtime_revision") != RUNTIME_REVISION
        or receipt.get("schema") != strata_runtime.SCHEMA
        or receipt.get("status") != "registered_not_executed"
        or receipt.get("qualification_created") is not False
        or receipt.get("parent_binding_rewritten") is not False
        or receipt.get("launch_sha256") != canonical_hash(launch)
        or Path(receipt.get("launch_out", "")).resolve() != text_path
        or receipt.get("route_id") != backend.get("route_id")
        or not isinstance(backend.get("observation_runtime"), dict)
        or not backend["observation_runtime"]
    ):
        raise ValueError("text intermediate/registration receipt is incomplete or mismatched")
    parent = Path(receipt["parent_launch"]).resolve(strict=True)
    if parent == text_path:
        raise ValueError("text intermediate cannot be its original prepared parent")
    with tempfile.TemporaryDirectory(prefix="omni-image-registration-") as temporary:
        temp = Path(temporary)
        manifest_path = temp / "runtime-manifest.json"
        manifest_path.write_text(json.dumps(backend["runtime_manifest"]), encoding="utf-8")
        output = temp / "verified-text.json"
        fresh = strata_runtime.register_runtime_variant(
            parent,
            runtime_root=Path(backend["runtime_root"]),
            runtime_manifest_path=manifest_path,
            observation_runtime=backend["observation_runtime"],
            launch_out=output,
            route_id=backend["route_id"],
        )
        if canonical_hash(_object(output)) != canonical_hash(launch):
            raise ValueError("text intermediate differs from freshly verified original preparation/runtime")
        normalized = copy.deepcopy(receipt)
        normalized["launch_out"] = fresh["launch_out"]
        if normalized != fresh:
            raise ValueError("text registration receipt differs from fresh source/pack/build/runtime verification")
    # Distinct explicit fields prevent raw JSON identity being confused with
    # ArtifactManifest's normalized/defaulted canonical schema identity.
    for key, field in (
        ("source_manifest_sha256", "artifact_manifest"),
        ("prepared_manifest_sha256", "prepared_pack_manifest"),
        ("runtime_manifest_sha256", "runtime_manifest"),
    ):
        if receipt[key] != manifest_type.from_dict(backend[field]).manifest_sha256:
            raise ValueError("text intermediate typed manifest identity differs")
    return launch, receipt, parent


def register_cpu_image_route(
    text_launch_path: Path,
    *,
    text_registration_path: Path,
    image_route: Mapping,
    launch_out: Path,
    route_id: str,
    max_io_bytes: int | None = None,
):
    """Verify and derive an exclusive CPU-encoder/CUDA-language launch.

    Existing text expert/cache/context/precision controls are retained. Encoder
    and scratch budgets are additional coexistence claims, never substitutions
    for the text stage's overhead. Fresh snapshots do not reserve resources.
    """
    strata, strata_io, vision, ArtifactManifest, WeightTierPlan = _helpers()
    text_path = Path(text_launch_path).resolve(strict=True)
    text_receipt_path = Path(text_registration_path).resolve(strict=True)
    text_input_hashes = {path: file_hash(path) for path in (text_path, text_receipt_path)}
    destination = Path(launch_out).resolve()
    receipt_out = destination.with_suffix(".image-registration.json")
    if destination.exists() or receipt_out.exists():
        raise FileExistsError("image launch/receipt already exists")
    if destination in (text_path, text_receipt_path) or receipt_out in (text_path, text_receipt_path):
        raise ValueError("image outputs must not replace the text intermediate")
    if not isinstance(route_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", route_id):
        raise ValueError("image route needs an explicit bounded ID")
    config = copy.deepcopy(dict(image_route))
    if config.get("encoder_device") != "cpu" or config.get("allow_cpu_fallback") is not False:
        raise ValueError("only an explicit CPU image encoder can be registered")
    initial = _object(text_path)
    backend = initial["backend"]
    roots = [Path(backend[key]).resolve(strict=True) for key in ("runtime_root", "artifact_root", "prepared_model_dir")]
    projector_root = Path(config["projector_root"]).resolve(strict=True)
    _outside((destination, receipt_out), [*roots, projector_root])
    sources_before = _trusted_source_identity(strata, strata_io, vision)
    launch, text_receipt, original_parent = _verified_text_intermediate(
        text_path,
        text_receipt_path,
        manifest_type=ArtifactManifest,
    )
    backend = launch["backend"]
    if route_id in {backend["route_id"], _object(original_parent)["backend"]["route_id"]}:
        raise ValueError("CPU image route ID must differ from original/text route IDs")
    runtime = ArtifactManifest.from_dict(backend["runtime_manifest"])
    source = ArtifactManifest.from_dict(backend["artifact_manifest"])
    prepared = ArtifactManifest.from_dict(backend["prepared_pack_manifest"])
    projector = ArtifactManifest.from_dict(config["projector_manifest"])
    runtime_files = strata_runtime._complete_files(runtime, roots[0], runtime=True)
    verified_vision = vision.verify_vision_route(
        config,
        roots[0],
        runtime,
        runtime_files,
        source,
        manifest_type=ArtifactManifest,
    )
    if (
        backend.get("spec_tokens") != 0
        or config["max_image_tokens"] + backend["max_new_tokens"] + 8 >= backend["context_tokens"]
    ):
        raise ValueError("image registration requires MTP off and explicit untruncated image/output context")
    maximum_encoded_image = len("data:image/png;base64,") + 4 * ((config["max_image_bytes"] + 2) // 3)
    transport = backend["max_io_bytes"] if max_io_bytes is None else _positive(max_io_bytes, "max_io_bytes")
    if transport < backend["max_io_bytes"] or transport > 64 << 20:
        raise ValueError("image transport override must preserve or expand the original bound within 64 MiB")
    transport_delta = transport - backend["max_io_bytes"]
    if maximum_encoded_image + 1024 > transport:
        raise ValueError("maximum inline PNG plus bounded text allowance exceeds existing admitted transport bytes")
    environment = backend.get("python_environment")
    if not isinstance(environment, dict) or not isinstance(environment.get("dependencies", {}).get("Pillow"), str):
        raise ValueError("image route requires exact prepared Python/Pillow environment")
    python = Path(backend["python_bin"]).resolve(strict=True)
    verified_python = strata._verify_python_environment(python, environment)
    if verified_python is None or file_hash(python) != backend["python_sha256"]:
        raise ValueError("prepared Python executable/environment is unverified")
    tier = WeightTierPlan.from_dict(backend["weight_tier_plan"])
    if tier.backend != TEXT_BACKEND or tier.route_id != backend["route_id"]:
        raise ValueError("text intermediate typed tier route is inconsistent")
    host = _positive(config["vision_host_bytes"], "vision_host_bytes")
    scratch = _positive(config["vision_scratch_bytes"], "vision_scratch_bytes")
    extra_host = host + scratch
    tier_data = tier.to_dict()
    tier_data.update(route_id=route_id, backend=BACKEND)
    budget = tier_data["budget"]
    budget["host_workspace_bytes"] += extra_host
    budget["host_transfer_bytes"] += transport_delta
    for field in ("host_loading_peak_bytes", "windows_commit_peak_bytes"):
        budget[field] += extra_host + transport_delta
    budget["ssd_artifact_bytes"] += projector.total_size_bytes
    budget["ssd_temporary_bytes"] += scratch
    new_tier = WeightTierPlan.from_dict(tier_data)
    if budget["ssd_artifact_bytes"] < sum(item.total_size_bytes for item in (source, prepared, runtime, projector)):
        raise ValueError("image SSD artifact budget omits source/pack/complete runtime/projector")
    resources = launch["resource_budget"]
    if "windows_commit" not in resources["demands"]:
        raise ValueError("CPU image launch requires an explicit Windows commit admission pool")
    demands = new_tier.budget.resource_demands(
        gpu_pool=backend["gpu_pool"],
        include_wsl="wsl_ram" in resources["demands"],
        include_windows_commit="windows_commit" in resources["demands"],
    )
    capacities = copy.deepcopy(resources["capacities"])
    if any(type(capacities.get(pool)) is not int or amount > capacities[pool] for pool, amount in demands.items()):
        raise ValueError("additional CPU image coexistence claim exceeds inherited resource ceilings")
    image_identity = copy.deepcopy(verified_vision["identity"])
    image_identity.pop("identity_sha256")
    image_identity.update(
        vision_bootstrap_sha256=hashlib.sha256(vision.bootstrap_source(strata._BOOTSTRAP).encode()).hexdigest(),
        vision_adapter_sha256=file_hash(Path(vision.__file__)),
        base_text_bootstrap_sha256=hashlib.sha256(strata._BOOTSTRAP.encode()).hexdigest(),
        prepared_artifact_size_bytes=prepared.total_size_bytes,
    )
    image_identity["identity_sha256"] = canonical_hash(image_identity)
    immutable_metadata = {
        original_parent: text_receipt["parent_launch_file_sha256"],
        Path(launch["preparation_receipt"]).resolve(strict=True): text_receipt["parent_preparation_receipt_sha256"],
        roots[2].with_name(roots[2].name + ".omni-binding.json"): text_receipt["parent_pack_binding_sha256"],
        **text_input_hashes,
    }
    # The replay already verified every source/pack byte and verify_vision_route
    # verified the projector. Do not rehash hundreds of GB for a stat guard.
    bound_files = set(runtime_files)
    for manifest, root in ((source, roots[1]), (prepared, roots[2]), (projector, projector_root)):
        bound_files.update((root / item.path).resolve(strict=True) for item in manifest.files)
    file_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in bound_files}
    # Last, obtain live read-only capacity snapshots. A later process launch
    # must re-probe and acquire its shared lease; this is not a reservation.
    probe_started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    memory = strata._probe_memory(backend["gpu_index"])
    strata._check_live_memory(memory, demands, backend["gpu_pool"], backend["gpu_total_bytes"])
    disk = _disk_snapshot([*roots, projector_root, Path(tempfile.gettempdir())])
    _check_disk(disk, demands)
    if min(row["free_bytes"] for row in disk) < new_tier.budget.ssd_temporary_bytes:
        raise ValueError("free filesystem bytes do not cover admitted scratch/temporary claim")
    sources_after = _trusted_source_identity(strata, strata_io, vision)
    if sources_after != sources_before:
        raise ValueError("image registrar/engine/adapter code changed during registration")
    if any(file_hash(path) != expected for path, expected in immutable_metadata.items()):
        raise ValueError("linked input/preparation/pack binding metadata changed during image registration")
    if any((path.stat().st_size, path.stat().st_mtime_ns) != expected for path, expected in file_stats.items()):
        raise ValueError("verified runtime/source/pack/projector file changed after verification")
    provenance = {
        "schema": SCHEMA,
        "text_launch_file_sha256": file_hash(text_path),
        "text_launch_canonical_sha256": canonical_hash(launch),
        "text_registration_file_sha256": file_hash(text_receipt_path),
        "text_registration_canonical_sha256": canonical_hash(text_receipt),
        "original_prepared_launch_file_sha256": file_hash(original_parent),
        "source_manifest_sha256": source.manifest_sha256,
        "source_manifest_input_canonical_sha256": canonical_hash(backend["artifact_manifest"]),
        "prepared_manifest_sha256": prepared.manifest_sha256,
        "runtime_manifest_sha256": runtime.manifest_sha256,
        "projector_manifest_sha256": projector.manifest_sha256,
        "projector_manifest_input_canonical_sha256": canonical_hash(config["projector_manifest"]),
        "observation_runtime_identity_sha256": backend["observation_runtime_identity_sha256"],
        "image_identity_sha256": image_identity["identity_sha256"],
        "source_code_sha256": sources_before,
        "additional_encoder_and_scratch_host_bytes": extra_host,
        "explicit_transport_max_io_bytes": transport,
        "additional_host_transport_buffer_bytes": transport_delta,
        "additional_projector_ssd_bytes": projector.total_size_bytes,
        "runtime_disk_bytes_already_added_by_text_registration": runtime.total_size_bytes,
        "additional_scratch_ssd_temporary_bytes": scratch,
        "qualification_created": False,
        "scope": (
            "verified experimental CPU image launch and fresh read-only admission snapshot; "
            "no model execution or resource lease"
        ),
    }
    derived = copy.deepcopy(launch)
    target = derived["backend"]
    target.update(
        name=BACKEND,
        route_id=route_id,
        image_route=config,
        weight_tier_plan=new_tier.to_dict(),
        # The backend adds max_io_bytes separately to expert+overhead. The
        # transport delta is already charged by host_transfer/loading/commit;
        # including it in overhead would count that same buffer twice.
        host_overhead_bytes=backend["host_overhead_bytes"] + extra_host,
        max_io_bytes=transport,
    )
    derived["resource_budget"] = {"capacities": capacities, "demands": demands}
    derived["image_registration"] = provenance
    derived["admission_preflight"] = {
        "admitted": True,
        "over_budget_pools": [],
        "probe_started_utc": probe_started,
        "memory": memory,
        "disk_capacity": disk,
        "scope": "fresh registration snapshot only; startup requires new probes and shared resource lease",
    }
    receipt = {
        **provenance,
        "status": "registered_experimental_image_not_executed",
        "text_launch": str(text_path),
        "text_registration": str(text_receipt_path),
        "original_prepared_launch": str(original_parent),
        "launch_out": str(destination),
        "launch_canonical_sha256": canonical_hash(derived),
        "route_id": route_id,
        "image_identity": image_identity,
        "fresh_admission": copy.deepcopy(derived["admission_preflight"]),
        "native_build_proof_scope": "hash_bound_recorded_build_evidence_not_independent_rebuild",
        "parent_binding_rewritten": False,
        "default_route_created": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    for path, value in ((receipt_out, receipt), (destination, derived)):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("text-launch", "text-registration", "image-route", "launch-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--max-io-bytes", type=int)
    args = parser.parse_args()
    receipt = register_cpu_image_route(
        args.text_launch,
        text_registration_path=args.text_registration,
        image_route=_object(args.image_route),
        launch_out=args.launch_out,
        route_id=args.route_id,
        max_io_bytes=args.max_io_bytes,
    )
    print(json.dumps({key: receipt[key] for key in ("status", "route_id", "launch_canonical_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
