# SPDX-License-Identifier: Apache-2.0
"""Opt-in llama.cpp launch identity using existing Agent route/profile fields."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

SCHEMA = "omni-llamacpp-agent-launch-identity-v1"
LLAMA_TEXT_BACKEND = "external.llamacpp.text.v1"
_CONSUMER_TOP = {
    "model_output_contract", "base_artifact_id",
    "model_output_consumer_identity", "model_output_workspace_bytes",
}
_LAUNCH_KEYS = {
    "schema", "backend_config_sha256", "launch_controls",
    "runtime_artifact_manifest_sha256", "source_artifact_manifest_sha256",
    "context_tokens", "max_new_tokens", "max_io_bytes", "server_sha256",
    "model_sha256", "mmproj_sha256", "placement",
}
_CONSUMER_KEYS = {
    "base_engine_artifact_id", "model_output_consumer_identity",
    "consumer_memory_demands", "consumer_memory_overhead_bytes",
    "consumer_workspace_bytes",
}
_STRATA_PROVENANCE = {
    "weight_tier_plan", "observation_runtime", "execution_observation",
    "gpu_observer_identity", "native_io_observation",
    "expert_compute_verified", "expert_final_storage_verified",
    "cpu_compute_verified", "cpu_expert_compute_verified",
}


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def llamacpp_route_binding(entry: Mapping) -> dict | None:
    """Recompute from actual typed Stage config, never a caller hash claim.

    Absence preserves legacy identity. Partial controlled metadata refuses.
    Original checkpoint/model/artifact IDs remain independent and unchanged.
    The signed profile loader already binds the entire native configuration.
    """
    fields = {"launch_controls", "launch_controls_runtime_manifest"}
    present = fields & set(entry)
    if not present:
        return None
    if present != fields or any(entry[key] is None for key in fields):
        raise ValueError("llama.cpp controlled route requires both explicit launch fields")
    backend = entry.get(
        "backend", "external.llamacpp.multimodal.v1" if entry.get("mmproj_file") else "external.llamacpp.text.v1"
    )
    if backend not in {"external.llamacpp.text.v1", "external.llamacpp.multimodal.v1"}:
        raise ValueError("llama.cpp launch controls belong only to the llama.cpp backend")
    from vllm_omni.edge.agent.omni_backend import OmniLlamaBackend, llama_config_from_entry

    config = llama_config_from_entry(entry, capacities=llamacpp_memory_demands(entry))
    if OmniLlamaBackend(config)._stage_backend_config()["name"] != backend:
        raise ValueError("controlled llama.cpp route backend differs from its projector configuration")
    if entry.get("model_output_contract") is not None:
        return validate_llamacpp_consumer_entry(entry, config=config)
    return llamacpp_config_binding(config)


def llamacpp_config_binding(config) -> dict:
    """Bind the normalized execution config used by this Agent adapter."""
    from vllm_omni.edge.agent.omni_backend import OmniLlamaBackend

    stage = OmniLlamaBackend(config)._stage_backend_config()
    from pathlib import Path

    from vllm_omni.engine.weight_tiers import ArtifactManifest

    runtime = ArtifactManifest.from_dict(stage["launch_controls_runtime_manifest"])
    selected = next((item for item in runtime.files if item.path == Path(config.server_bin).name), None)
    if selected is None or selected.sha256 != config.server_sha256:
        raise ValueError("controlled llama.cpp launcher differs from the supported runtime manifest")
    source = (
        ArtifactManifest.from_dict(stage["artifact_manifest"]) if stage.get("artifact_manifest") is not None else None
    )
    # Paths/output destinations and all fresh observations are excluded. The
    # model identity is retained separately from normalized execution controls.
    excluded = {
        "model_file",
        "model_sha256",
        "server_bin",
        "log_file",
        "artifact_root",
        "artifact_manifest",
        "mmproj_file",
        "mmproj_sha256",
        "launch_controls_runtime_manifest",
    }
    behavior = {key: value for key, value in stage.items() if key not in excluded}
    behavior["runtime_artifact_manifest_sha256"] = runtime.manifest_sha256
    digest = hashlib.sha256(_canonical(behavior)).hexdigest()
    identity = {
        "schema": SCHEMA,
        "backend_config_sha256": digest,
        "launch_controls": stage["launch_controls"],
        "runtime_artifact_manifest_sha256": runtime.manifest_sha256,
        "source_artifact_manifest_sha256": source.manifest_sha256 if source is not None else None,
        "context_tokens": config.context_tokens,
        "max_new_tokens": config.max_new_tokens,
        "max_io_bytes": config.max_io_bytes,
        "server_sha256": config.server_sha256,
        "model_sha256": config.model_sha256,
        "mmproj_sha256": config.mmproj_sha256,
        "placement": config.placement,
    }
    return identity


def validate_llamacpp_launch_plan(plan: Mapping, binding: Mapping) -> None:
    """Check applied controls in the owned loaded plan, without promotion."""
    observed = plan.get("launch_controls")
    if not isinstance(observed, Mapping) or binding.get("schema") != SCHEMA:
        raise ValueError("controlled llama.cpp route lacks applied launch metadata")
    if (
        _canonical(observed.get("requested")) != _canonical(binding["launch_controls"])
        or observed.get("runtime_artifact_manifest_sha256") != binding["runtime_artifact_manifest_sha256"]
        or any(
            _canonical(plan.get(key)) != _canonical(binding[key])
            for key in ("context_tokens", "max_new_tokens", "max_io_bytes", "server_sha256", "model_sha256")
        )
        or plan.get("artifact_manifest_sha256") != binding["source_artifact_manifest_sha256"]
        or plan.get("mmproj_sha256") != binding["mmproj_sha256"]
        or plan.get("requested_device") != binding["placement"]
        or observed.get("memory_claim_discounted") is not False
    ):
        raise ValueError("controlled llama.cpp applied launch differs from route identity")


def validate_llamacpp_profile_binding(profile: Mapping, entry: Mapping, provenance: Mapping) -> None:
    """Require exact profile/current binding before existing signed release gates."""
    expected = llamacpp_route_binding(entry)
    recorded = profile.get("backend_identity", {})
    marked = isinstance(recorded, Mapping) and (
        recorded.get("schema") == SCHEMA
        or "launch_controls" in recorded
        or "runtime_artifact_manifest_sha256" in recorded
    )
    if expected is None:
        if marked:
            raise ValueError("profiled llama.cpp launch controls were dropped from current config")
        return
    if _canonical(recorded) != _canonical(expected) or any(
        _canonical(provenance.get(key)) != _canonical(value) for key, value in expected.items()
    ):
        raise ValueError("current llama.cpp launch controls/runtime differ from reviewed profile")


def llamacpp_profile_identity(profile: Mapping) -> dict[str, str]:
    """Extend signed identity only for the opt-in profile; legacy stays exact."""
    binding = profile.get("backend_identity", {})
    if not isinstance(binding, Mapping) or not (
        binding.get("schema") == SCHEMA or "launch_controls" in binding or "runtime_artifact_manifest_sha256" in binding
    ):
        return {}
    digest = binding.get("backend_config_sha256")
    if (
        binding.get("schema") != SCHEMA
        or type(digest) is not str
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or profile.get("backend") not in {"external.llamacpp.text.v1", "external.llamacpp.multimodal.v1"}
    ):
        raise ValueError("invalid llama.cpp profile launch identity")
    return {"llamacpp_launch_controls_sha256": digest}


def llamacpp_memory_demands(entry: Mapping) -> dict:
    """Native entries retain base demands; charge optional parser RAM/commit once."""
    workspace = entry.get("model_output_workspace_bytes", 0)
    if type(workspace) is not int or not 0 <= workspace <= 32 << 20:
        raise ValueError("invalid llama.cpp consumer workspace")
    demands = dict(entry["memory_demands"])
    if workspace:
        if any(type(demands.get(pool)) is not int or demands[pool] <= 0
               for pool in ("host_ram", "windows_commit")):
            raise ValueError("llama.cpp consumer requires explicit base RAM and commit claims")
        for pool in ("host_ram", "windows_commit"):
            demands[pool] += workspace
    return demands


def _consumer_material(launch: Mapping, demands: Mapping, overhead: int, workspace: int) -> dict:
    return {"launch": dict(launch), "memory_demands": dict(demands),
            "memory_overhead_bytes": overhead, "workspace_bytes": workspace}


def llamacpp_consumer_binding(config, contract) -> dict:
    """Bind the existing consumer to a controlled text plan, not a new checkpoint."""
    from vllm_omni.edge.agent.model_output import AgentOutputContract
    from vllm_omni.engine.weight_tiers import ArtifactManifest

    if (not isinstance(contract, AgentOutputContract) or config.mmproj_file is not None
            or config.launch_controls is None or config.artifact_manifest is None
            or config.placement.startswith("Vulkan_Host+")
            or config.model_output_workspace_bytes != contract.workspace_budget_bytes):
        raise ValueError("llama.cpp consumer requires controlled text, source manifest and exact workspace")
    source = ArtifactManifest.from_dict(dict(config.artifact_manifest))
    if not any(item.role == "weights" for item in source.files):
        raise ValueError("llama.cpp consumer source manifest has no weights")
    contract.admit(max_io_bytes=config.max_io_bytes,
                   workspace_reserved_bytes=config.model_output_workspace_bytes)
    launch = llamacpp_config_binding(config)
    material = _consumer_material(launch, config.demands, config.memory_overhead_bytes,
                                  config.model_output_workspace_bytes)
    base = "llamacpp:" + hashlib.sha256(_canonical(material)).hexdigest()
    return {
        **launch, "base_engine_artifact_id": base,
        "model_output_consumer_identity": contract.consumer_identity(base),
        "consumer_memory_demands": dict(config.demands),
        "consumer_memory_overhead_bytes": config.memory_overhead_bytes,
        "consumer_workspace_bytes": config.model_output_workspace_bytes,
    }


def validate_llamacpp_consumer_entry(entry: Mapping, *, config=None) -> dict:
    """Recompute all consumer claims from the actual native config translation."""
    from vllm_omni.edge.agent.model_output import AgentOutputContract
    from vllm_omni.edge.agent.omni_backend import llama_config_from_entry

    if (entry.get("backend") != LLAMA_TEXT_BACKEND or entry.get("mmproj_file") is not None
            or "backend_config" in entry or _STRATA_PROVENANCE & set(entry)
            or not _CONSUMER_TOP <= set(entry)):
        raise ValueError("explicit llama.cpp consumer requires its supported native text representation")
    contract = AgentOutputContract.from_dict(entry["model_output_contract"])
    if config is None:
        config = llama_config_from_entry(entry, capacities=llamacpp_memory_demands(entry))
    binding = llamacpp_consumer_binding(config, contract)
    consumer = binding["model_output_consumer_identity"]
    if (entry["base_artifact_id"] != binding["base_engine_artifact_id"]
            or entry["model_output_consumer_identity"] != consumer
            or entry["artifact_id"] != "llamacpp-agent:" + consumer["identity_sha256"]
            or type(entry["model_output_workspace_bytes"]) is not int
            or entry["model_output_workspace_bytes"] != contract.workspace_budget_bytes):
        raise ValueError("llama.cpp Agent consumer identity or workspace differs")
    return binding


def llamacpp_agent_config_with_output_contract(entry: Mapping, contract) -> dict:
    """Derive one opt-in entry; its memory_demands remain the unmodified base claim."""
    from copy import deepcopy

    from vllm_omni.edge.agent.omni_backend import llama_config_from_entry

    if (_CONSUMER_TOP & set(entry)
            or str(entry.get("artifact_id", "")).startswith(("strata-agent:", "llamacpp-agent:"))):
        raise ValueError("cannot apply an Agent output contract twice")
    result = deepcopy(dict(entry))
    result["model_output_contract"] = contract.to_dict()
    result["model_output_workspace_bytes"] = contract.workspace_budget_bytes
    config = llama_config_from_entry(result, capacities=llamacpp_memory_demands(result))
    binding = llamacpp_consumer_binding(config, contract)
    result["base_artifact_id"] = binding["base_engine_artifact_id"]
    result["model_output_consumer_identity"] = binding["model_output_consumer_identity"]
    result["artifact_id"] = "llamacpp-agent:" + binding["model_output_consumer_identity"]["identity_sha256"]
    validate_llamacpp_consumer_entry(result, config=config)
    return result


def validate_llamacpp_consumer_binding(binding: Mapping):
    """Validate the finite recorded profile representation without model I/O."""
    from vllm_omni.edge.agent.model_output import AgentOutputContract

    if not isinstance(binding, Mapping) or set(binding) != _LAUNCH_KEYS | _CONSUMER_KEYS:
        raise ValueError("llama.cpp consumer binding has unknown or missing provenance")
    consumer = binding["model_output_consumer_identity"]
    if not isinstance(consumer, Mapping) or not isinstance(consumer.get("contract"), Mapping):
        raise ValueError("missing llama.cpp consumer contract")
    contract = AgentOutputContract.from_dict(dict(consumer["contract"]))
    workspace, demands = binding["consumer_workspace_bytes"], binding["consumer_memory_demands"]
    overhead = binding["consumer_memory_overhead_bytes"]
    if (binding["schema"] != SCHEMA or binding["mmproj_sha256"] is not None
            or not isinstance(binding["source_artifact_manifest_sha256"], str)
            or len(binding["source_artifact_manifest_sha256"]) != 64
            or not isinstance(binding["placement"], str)
            or binding["placement"].startswith("Vulkan_Host+")
            or type(workspace) is not int or workspace != contract.workspace_budget_bytes
            or type(overhead) is not int or overhead < workspace
            or not isinstance(demands, Mapping)
            or any(type(value) is not int or value < 0 for value in demands.values())
            or any(demands.get(pool, 0) < workspace for pool in ("host_ram", "windows_commit"))):
        raise ValueError("invalid llama.cpp consumer memory/source binding")
    launch = {key: binding[key] for key in _LAUNCH_KEYS}
    base = "llamacpp:" + hashlib.sha256(
        _canonical(_consumer_material(launch, demands, overhead, workspace))
    ).hexdigest()
    expected = contract.consumer_identity(base)
    contract.admit(max_io_bytes=binding["max_io_bytes"], workspace_reserved_bytes=workspace)
    if binding["base_engine_artifact_id"] != base or _canonical(consumer) != _canonical(expected):
        raise ValueError("llama.cpp consumer/base plan identity differs")
    return contract, expected


def validate_llamacpp_consumer_plan(plan: Mapping, binding: Mapping) -> None:
    """Startup-log placement plus owned Stage identity; no native compute/I/O claim."""
    validate_llamacpp_consumer_binding(binding)
    validate_llamacpp_launch_plan(plan, binding)
    if (plan.get("backend") != LLAMA_TEXT_BACKEND
            or plan.get("observed_model_placement") != binding["placement"]
            or plan.get("placement_evidence_level") != "startup_log"
            or _canonical(plan.get("reserved_bytes")) != _canonical(binding["consumer_memory_demands"])
            or plan.get("memory_overhead_bytes") != binding["consumer_memory_overhead_bytes"]
            or not isinstance(plan.get("worker_generation"), str) or not plan["worker_generation"]
            or type(plan.get("stage_id")) is not int or plan["stage_id"] < 0
            or type(plan.get("worker_pid")) is not int or plan["worker_pid"] <= 0
            or any(plan.get(key) is not None for key in _STRATA_PROVENANCE)
            or plan.get("expert_compute_verified") is not None
            or plan.get("expert_final_storage_verified") is not None):
        raise ValueError("llama.cpp consumer lacks its loaded reservation/Stage provenance")


def validate_llamacpp_consumer_request_evidence(evidence: Mapping, route: Mapping) -> None:
    from vllm_omni.edge.agent.placement import evidence_sha256

    if route.get("backend") != LLAMA_TEXT_BACKEND:
        raise ValueError("llama.cpp consumer evidence belongs only to its text backend")
    binding, plan = route["backend_identity"], evidence.get("execution_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("llama.cpp consumer lacks a loaded plan")
    validate_llamacpp_consumer_plan(plan, binding)
    if (evidence.get("loaded_plan_sha256") != evidence_sha256(dict(plan))
            or evidence.get("placement_verification_scope") != "loaded_configuration_and_terminal_stage_identity"
            or any(evidence.get(key) is not None for key in _STRATA_PROVENANCE)):
        raise ValueError("llama.cpp consumer request evidence differs from its limited loaded-plan scope")
