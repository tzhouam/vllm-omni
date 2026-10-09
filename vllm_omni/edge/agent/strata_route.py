# SPDX-License-Identifier: Apache-2.0
"""Translate a prepared Strata launch into an explicitly experimental Agent route.

This is configuration translation, not qualification. The existing Agent owns
memory, tools, routing and the loop. StageRuntime re-verifies every file and
the loaded execution configuration; per-request compute evidence stays
separate. The engine manager lends the same resource lease. No stale
preparation capacity is installed as a live Agent ceiling.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vllm_omni.engine.weight_tiers import ArtifactManifest, WeightTierPlan

BACKEND = "external.strata.text.v1"
RUNTIME_REVISION = "d5ea7133741e67743c0e886bb426c0ce8d69cf6c"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def agent_entry_from_launch(launch: Mapping[str, Any], *, expected_device_name: str) -> dict[str, Any]:
    """Preserve exact artifacts/budgets while naming GPU 0's Agent pool ``vram``.

    Only a native Windows launch with explicit commit accounting is accepted.
    WSL quota/host claims cannot be reinterpreted as Windows qualification. The
    returned entry cannot enter default routing without reviewed Agent evidence.
    """
    if launch.get("schema") != "omni-strata-launch-v1":
        raise ValueError("expected a prepared omni-strata-launch-v1 launch")
    if not isinstance(expected_device_name, str) or not expected_device_name.strip():
        raise ValueError("the exact native GPU 0 device name is required")
    backend = copy.deepcopy(dict(launch["backend"]))
    from vllm_omni.edge.agent.strata_image_evidence import IMAGE_BACKEND, image_capability_from_config

    selected_backend = backend.get("name")
    if selected_backend not in {BACKEND, IMAGE_BACKEND} or backend.get("runtime_revision") != RUNTIME_REVISION:
        raise ValueError("Agent route requires the pinned Strata complete-model backend")
    if type(backend.get("gpu_index")) is not int or backend["gpu_index"] != 0:
        raise ValueError("native Agent currently binds only the measured physical GPU 0")
    gpu_pool = backend.get("gpu_pool")
    if gpu_pool not in {"vram:0", "vram"}:
        raise ValueError("Strata GPU pool must name physical GPU 0")
    for forbidden in ("args", "config_file", "mcp_servers", "before_load", "vision"):
        if backend.get(forbidden) is not None:
            raise ValueError(f"Strata Agent entry refuses {forbidden}")

    if selected_backend == BACKEND and backend.get("image_route") is not None:
        raise ValueError("text Strata route cannot carry an image capability")
    capability = image_capability_from_config(backend) if selected_backend == IMAGE_BACKEND else None

    source = ArtifactManifest.from_dict(backend["artifact_manifest"])
    runtime = ArtifactManifest.from_dict(backend["runtime_manifest"])
    prepared = ArtifactManifest.from_dict(backend["prepared_pack_manifest"])
    if not re.fullmatch(r"[a-f0-9]{40}", source.revision) or runtime.revision != RUNTIME_REVISION:
        raise ValueError("source/runtime manifests need the pinned immutable revisions")
    conversion = backend["conversion_manifest"]
    if (
        not isinstance(conversion, Mapping)
        or conversion.get("complete") is not True
        or conversion.get("source_manifest_sha256") != source.manifest_sha256
        or conversion.get("prepared_manifest_sha256") != prepared.manifest_sha256
        or conversion.get("tool_revision") != RUNTIME_REVISION
        or not isinstance(conversion.get("conversions"), list)
    ):
        raise ValueError("prepared conversion identity is incomplete or bound to different bytes")
    for key in ("runtime_root", "artifact_root", "prepared_model_dir", "python_bin"):
        if not isinstance(backend.get(key), str) or not backend[key]:
            raise ValueError(f"Strata launch requires {key}")
    if not re.fullmatch(r"[a-f0-9]{64}", backend.get("python_sha256", "")):
        raise ValueError("Strata launch requires the native Python executable hash")
    if not isinstance(backend.get("python_environment"), Mapping):
        raise ValueError("Strata launch requires its native Python dependency identity")
    source_paths = {item.path for item in source.files}
    if backend.get("native_file") not in source_paths or backend.get("ple_file") not in source_paths:
        raise ValueError("native/PLE files must belong to the complete source manifest")

    plan = WeightTierPlan.from_dict(backend["weight_tier_plan"])
    if (
        plan.backend != selected_backend
        or plan.backend_revision != RUNTIME_REVISION
        or plan.artifact_manifest_sha256 != source.manifest_sha256
        or backend.get("route_id") != plan.route_id
    ):
        raise ValueError("tier plan differs from the launch route, checkpoint or runtime")
    if plan.cpu_block_layers or plan.cpu_expert_layers:
        raise ValueError("Strata owns expert scheduling; static CPU layer overrides are unsupported")
    spec = backend.get("spec_tokens", 0)
    prefetch = [backend.get(key, False) for key in ("ple_prefetch", "routing_prefetch", "io_prefetch")]
    if (
        type(spec) is not int
        or spec not in {0, 2, 3, 4, 5, 6, 7, 8}
        or any(type(flag) is not bool for flag in prefetch)
    ):
        raise ValueError("Strata MTP/prefetch controls must be explicitly supported values")
    if plan.mtp != bool(spec) or plan.prefetch != any(prefetch) or bool(spec) != bool(backend.get("mtp_directory")):
        raise ValueError("MTP/prefetch controls differ from the pinned tier plan")
    budget = plan.budget
    resources = launch["resource_budget"]
    capacities, demands = dict(resources["capacities"]), dict(resources["demands"])
    allowed = {"host_ram", gpu_pool, "windows_commit", "ssd"}
    if set(demands) != allowed or set(capacities) != allowed:
        raise ValueError("native Windows launch needs host/GPU/SSD/commit pools and no WSL or foreign pools")
    for pool, amount in demands.items():
        _positive(amount, f"{pool} demand")
        _positive(capacities[pool], f"{pool} capacity")
        if amount > capacities[pool]:
            raise ValueError(f"{pool} demand exceeds the preparation ceiling")
    expected = budget.resource_demands(gpu_pool=gpu_pool, include_windows_commit=True)
    if demands != expected or budget.windows_commit_peak_bytes <= 0:
        raise ValueError("native Windows claims differ from the complete tier budget/commit peak")
    if budget.ssd_artifact_bytes < source.total_size_bytes + prepared.total_size_bytes:
        raise ValueError("SSD tier budget omits source or prepared artifacts")
    if budget.cpu_expert_cache_bytes != backend.get("expert_ram_budget_bytes"):
        raise ValueError("expert cache budget differs from the launch")
    host_overhead = _positive(backend.get("host_overhead_bytes"), "host overhead")
    gpu_budget = _positive(backend.get("gpu_budget_bytes"), "GPU budget")
    gpu_total = _positive(backend.get("gpu_total_bytes"), "GPU physical total")
    if gpu_budget > gpu_total or gpu_budget != demands[gpu_pool]:
        raise ValueError("GPU claim/budget/physical total differ")
    context = _positive(backend.get("context_tokens"), "context_tokens")
    maximum = _positive(backend.get("max_new_tokens"), "max_new_tokens")
    io_bytes = _positive(backend.get("max_io_bytes"), "max_io_bytes")
    if context <= maximum + 8 or budget.host_transfer_bytes < io_bytes:
        raise ValueError("context or I/O bounds exceed the admitted plan")
    from vllm_omni.edge.agent.strata_execution_evidence import execution_route_binding

    execution = execution_route_binding(backend, runtime, plan)
    observer_workspace = execution["workspace_bytes"] if execution is not None else 0
    if execution is not None and (
        selected_backend != BACKEND
        or budget.ssd_artifact_bytes < source.total_size_bytes + prepared.total_size_bytes + runtime.total_size_bytes
    ):
        raise ValueError("combined execution observation requires the text Stage and complete runtime SSD allowance")
    if demands["host_ram"] != budget.cpu_expert_cache_bytes + host_overhead + io_bytes + observer_workspace:
        raise ValueError("host claim omits cache, loading/state/workspace overhead or I/O")
    for key in ("request_timeout_s", "start_timeout_s"):
        value = backend.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")

    # Physical identity and byte claims stay unchanged; only the controller's
    # pool spelling differs. Live ceilings will be freshly measured by Agent.
    backend["gpu_pool"] = "vram"
    normalized_demands = {("vram" if pool == gpu_pool else pool): amount for pool, amount in demands.items()}
    entry = {
        "route_id": plan.route_id,
        "artifact_id": "strata:" + _digest(backend),
        "model": source.checkpoint,
        "backend": selected_backend,
        "modalities": ["text", "image"] if capability else ["text"],
        "placement": "cpu+cuda:0",
        "requires_nvidia": True,
        "expected_device_name": expected_device_name,
        "gpu_memory_pool": "vram",
        "memory_demands": normalized_demands,
        "context_tokens": context,
        "max_new_tokens": maximum,
        "max_io_bytes": io_bytes,
        "request_timeout_s": backend["request_timeout_s"],
        "start_timeout_s": backend["start_timeout_s"],
        "backend_config": backend,
        "experimental_status": "unqualified; verified loaded configuration and Agent admission still required",
        "source_launch_sha256": _digest(launch),
    }
    if capability is not None:
        entry.update(
            image_capability=capability,
            mmproj_file=capability["projector_file"],
            mmproj_sha256=capability["projector_sha256"],
            max_image_bytes=capability["max_image_bytes"],
            max_image_pixels=capability["max_image_pixels"],
            image_token_reserve=capability["max_image_tokens"],
        )
    return entry


def agent_config_from_launch(
    launch: Mapping[str, Any], *, expected_device_name: str, experimental_bootstrap: bool = False
) -> dict[str, Any]:
    """Opt-in bootstrap uses existing experimental routing; never creates evidence."""
    if type(experimental_bootstrap) is not bool:
        raise ValueError("experimental_bootstrap must be an explicit boolean")
    entry = agent_entry_from_launch(launch, expected_device_name=expected_device_name)
    config: dict[str, Any] = {"routes": [entry], "qualification_bundles": [], "trusted_review_keys": {}}
    config["limits"] = {"max_answer_tokens": min(512, entry["max_new_tokens"])}
    if experimental_bootstrap:
        config["experimental_bootstrap_route_id"] = entry["route_id"]
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path)
    parser.add_argument("--expected-device-name", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--experimental-bootstrap",
        action="store_true",
        help="explicitly allow this visibly experimental route; does not qualify it",
    )
    args = parser.parse_args()
    config = agent_config_from_launch(
        json.loads(args.launch.read_text(encoding="utf-8")),
        expected_device_name=args.expected_device_name,
        experimental_bootstrap=args.experimental_bootstrap,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(config, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")



def agent_config_with_output_contract_from_launch(
    launch: Mapping[str, Any], *, expected_device_name: str, output_contract: Mapping[str, Any],
    route_id: str, experimental_bootstrap: bool = False,
) -> dict[str, Any]:
    """Derive a separately identified Agent consumer with one exact shared lease.

    This is metadata derivation only, never qualification or a fresh capacity
    observation. Agent and StageRuntime must still perform their live checks.
    Source/runtime/pack/cache/GPU/context/precision controls remain unchanged.
    """
    from vllm_omni.edge.agent.model_output import AgentOutputContract, validate_output_contract_entry

    contract = AgentOutputContract.from_dict(dict(output_contract))
    original = agent_entry_from_launch(launch, expected_device_name=expected_device_name)
    if not isinstance(route_id, str) or not route_id or route_id == original["route_id"]:
        raise ValueError("explicit output consumer requires a distinct declared route ID")
    derived = copy.deepcopy(dict(launch))
    backend = derived["backend"]
    plan = WeightTierPlan.from_dict(backend["weight_tier_plan"])
    contract.admit(max_io_bytes=backend["max_io_bytes"],
                   workspace_reserved_bytes=contract.workspace_budget_bytes)
    tier = plan.to_dict()
    delta = contract.workspace_budget_bytes
    for key in ("host_workspace_bytes", "host_loading_peak_bytes", "windows_commit_peak_bytes"):
        tier["budget"][key] += delta
    tier["route_id"] = route_id
    backend["route_id"] = route_id
    backend["host_overhead_bytes"] += delta
    backend["weight_tier_plan"] = tier
    new_plan = WeightTierPlan.from_dict(tier)
    claims = new_plan.budget.resource_demands(gpu_pool=backend["gpu_pool"], include_windows_commit=True)
    if any(claim > derived["resource_budget"]["capacities"][pool] for pool, claim in claims.items()):
        raise ValueError("derived Agent workspace exceeds preparation ceilings; fresh admission required")
    derived["resource_budget"]["demands"] = claims
    config = agent_config_from_launch(derived, expected_device_name=expected_device_name,
                                      experimental_bootstrap=experimental_bootstrap)
    entry = config["routes"][0]
    identity = contract.consumer_identity(entry["artifact_id"])
    entry.update(base_artifact_id=entry["artifact_id"],
                 artifact_id="strata-agent:" + identity["identity_sha256"],
                 model_output_contract=contract.to_dict(), model_output_consumer_identity=identity,
                 model_output_workspace_bytes=delta)
    validate_output_contract_entry(entry)
    config["output_contract_derivation"] = {
        "schema": "omni-agent-output-contract-derivation-v1",
        "parent_launch_sha256": _digest(launch), "derived_engine_launch_sha256": _digest(derived),
        "parent_route_id": original["route_id"], "route_id": route_id,
        "workspace_delta_bytes": delta, "workspace_owner": "Agent model-output consumer",
        "workspace_scope": "declared parser/raw retention workspace within the single shared physical RAM lease",
        "transport_delta_bytes": 0, "fresh_admission_required": True,
        "engine_neural_controls_changed": False, "qualified": False,
    }
    return config


if __name__ == "__main__":
    main()
