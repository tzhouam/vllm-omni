# SPDX-License-Identifier: Apache-2.0
"""Opt-in llama.cpp launch identity using existing Agent route/profile fields."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

SCHEMA = "omni-llamacpp-agent-launch-identity-v1"


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

    config = llama_config_from_entry(entry, capacities=entry["memory_demands"])
    if OmniLlamaBackend(config)._stage_backend_config()["name"] != backend:
        raise ValueError("controlled llama.cpp route backend differs from its projector configuration")
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
