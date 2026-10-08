# SPDX-License-Identifier: Apache-2.0
"""Fail-closed bindings for the experimental CPU image + Strata text stage.

Configuration is not an execution proof. A completed image request proves only
the owned PNG -> ENC -> SVE -> GENI chain, not placement of every operator,
physical SSD traffic, image task quality, or a qualified Agent default.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

IMAGE_BACKEND = "external.strata.multimodal.v1"
IMAGE_PROOF_SCHEMA = "omni-agent-strata-image-proof-v1"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _positive(value: Any) -> bool:
    return type(value) is int and value > 0


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def image_capability_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the declared projector/transport/bounds; never read model files."""
    from vllm_omni.engine.weight_tiers import ArtifactManifest, WeightTierPlan

    if config.get("name") != IMAGE_BACKEND:
        raise ValueError("an image capability requires the distinct image StageClient")
    image = config.get("image_route")
    fields = {
        "schema",
        "projector_root",
        "projector_manifest",
        "projector_file",
        "text_artifact_manifest_sha256",
        "encoder_file",
        "encoder_build_manifest_file",
        "embedding_width",
        "max_image_bytes",
        "max_image_pixels",
        "max_image_tokens",
        "encoder_threads",
        "allow_cpu_fallback",
        "vision_host_bytes",
        "vision_gpu_bytes",
        "vision_scratch_bytes",
        "encoder_device",
    }
    if not isinstance(image, Mapping) or set(image) != fields or image["schema"] != "omni-strata-image-route-v1":
        raise ValueError("image capability needs an exact image route descriptor")
    if (
        image["encoder_device"] != "cpu"
        or image["allow_cpu_fallback"] is not False
        or type(image["vision_gpu_bytes"]) is not int
        or image["vision_gpu_bytes"] != 0
    ):
        raise ValueError("only explicit CPU image encoding is enabled")
    source = ArtifactManifest.from_dict(config["artifact_manifest"])
    runtime = ArtifactManifest.from_dict(config["runtime_manifest"])
    projector = ArtifactManifest.from_dict(image["projector_manifest"])
    tier = WeightTierPlan.from_dict(config["weight_tier_plan"])
    if (
        tier.backend != IMAGE_BACKEND
        or tier.route_id != config.get("route_id")
        or tier.artifact_manifest_sha256 != source.manifest_sha256
        or image["text_artifact_manifest_sha256"] != source.manifest_sha256
    ):
        raise ValueError("image route belongs to another checkpoint, backend or tier plan")
    records = {item.path: item for item in runtime.files}
    for key in ("encoder_file", "encoder_build_manifest_file"):
        path = PurePosixPath(image[key])
        if path.is_absolute() or ".." in path.parts or "\\" in image[key] or image[key] not in records:
            raise ValueError("encoder/build file must belong to the complete runtime manifest")
    if (
        not isinstance(image["projector_root"], str)
        or not image["projector_root"]
        or len(projector.files) != 1
        or projector.files[0].path != image["projector_file"]
        or projector.files[0].role != "vision_projector"
    ):
        raise ValueError("exactly one manifest-bound image projector is required")
    for key in (
        "embedding_width",
        "max_image_bytes",
        "max_image_pixels",
        "max_image_tokens",
        "encoder_threads",
        "vision_host_bytes",
        "vision_scratch_bytes",
    ):
        if not _positive(image[key]):
            raise ValueError(f"invalid image bound: {key}")
    if (
        image["max_image_bytes"] > 16 << 20
        or image["max_image_pixels"] > 4 * 1024 * 1024
        or image["max_image_tokens"] > 4096
        or image["embedding_width"] > 65536
        or image["encoder_threads"] > 64
    ):
        raise ValueError("image route exceeds the reviewed prototype bounds")
    host_min = projector.total_size_bytes + 32 * image["max_image_pixels"] + 2 * image["max_image_bytes"]
    scratch_min = 2 * (20 + 4 * image["embedding_width"] * image["max_image_tokens"]) + image["max_image_bytes"]
    if (
        image["vision_host_bytes"] < host_min
        or image["vision_scratch_bytes"] < scratch_min
        or tier.budget.host_workspace_bytes < image["vision_host_bytes"] + image["vision_scratch_bytes"]
        or tier.budget.ssd_temporary_bytes < image["vision_scratch_bytes"]
        or image["max_image_tokens"] + config["max_new_tokens"] + 8 >= config["context_tokens"]
        or config.get("spec_tokens", 0) != 0
    ):
        raise ValueError("image coexistence, scratch or context budget is incomplete")
    return {
        "schema": "omni-agent-strata-image-capability-v1",
        "backend": IMAGE_BACKEND,
        "image_route_sha256": digest(image),
        "projector_file": str(PurePosixPath(image["projector_root"]) / image["projector_file"]),
        "projector_sha256": projector.files[0].sha256,
        "projector_manifest_sha256": projector.manifest_sha256,
        "transport": "data:image/png;base64",
        "max_image_bytes": image["max_image_bytes"],
        "max_image_pixels": image["max_image_pixels"],
        "max_image_tokens": image["max_image_tokens"],
        "embedding_width": image["embedding_width"],
        "encoder_device": "cpu",
        "qualification": "experimental_opt_in_only",
    }


def validate_image_load_plan_for_config(plan: Mapping[str, Any], config: Mapping[str, Any], requested: str) -> None:
    """Use the actual image gate, then compare it to the selected Agent config."""
    from vllm_omni.engine.backends.strata_multimodal import validate_strata_multimodal_load_plan
    from vllm_omni.engine.weight_tiers import ArtifactManifest

    validate_strata_multimodal_load_plan(plan, requested)
    capability = image_capability_from_config(config)
    declared = config["image_route"]
    loaded = plan["image_route"]
    runtime = ArtifactManifest.from_dict(config["runtime_manifest"])
    records = {row.path: row for row in runtime.files}
    source = ArtifactManifest.from_dict(config["artifact_manifest"])
    prepared = ArtifactManifest.from_dict(config["prepared_pack_manifest"])
    expected = {
        "runtime_manifest_sha256": runtime.manifest_sha256,
        "text_artifact_manifest_sha256": source.manifest_sha256,
        "projector_manifest_sha256": capability["projector_manifest_sha256"],
        "projector_sha256": capability["projector_sha256"],
        "encoder_sha256": records[declared["encoder_file"]].sha256,
        "encoder_build_manifest_sha256": records[declared["encoder_build_manifest_file"]].sha256,
        "text_artifact_size_bytes": source.total_size_bytes,
        "prepared_artifact_size_bytes": prepared.total_size_bytes,
        "runtime_artifact_size_bytes": runtime.total_size_bytes,
        "projector_size_bytes": ArtifactManifest.from_dict(declared["projector_manifest"]).total_size_bytes,
    }
    bounds = {
        key: declared[key]
        for key in (
            "embedding_width",
            "max_image_bytes",
            "max_image_pixels",
            "max_image_tokens",
            "encoder_threads",
            "vision_host_bytes",
            "vision_gpu_bytes",
            "vision_scratch_bytes",
        )
    }
    if (
        any(loaded.get(key) != value for key, value in expected.items())
        or loaded.get("bounds_and_declared_budgets") != bounds
        or plan.get("weight_tier_plan") != config["weight_tier_plan"]
        or any(
            plan.get(key) != config[key]
            for key in (
                "context_tokens",
                "max_new_tokens",
                "max_io_bytes",
                "host_overhead_bytes",
                "gpu_budget_bytes",
                "gpu_total_bytes",
                "gpu_pool",
                "expert_ram_budget_bytes",
            )
        )
        or plan.get("prepared_manifest_sha256") != prepared.manifest_sha256
        or plan.get("conversion_manifest") != config["conversion_manifest"]
    ):
        raise RuntimeError("loaded image pipeline differs from the selected Agent artifact/configuration")


def image_input_identity(data_url: str, capability: Mapping[str, Any]) -> dict[str, Any]:
    """Check bounded PNG framing/dimensions without decoding pixels or resizing."""
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("image input must use the admitted inline PNG transport")
    encoded = data_url[len(prefix) :]
    if len(encoded) > 4 * ((capability["max_image_bytes"] + 2) // 3):
        raise ValueError("encoded image exceeds the admitted byte bound")
    raw = base64.b64decode(encoded, validate=True)
    if (
        base64.b64encode(raw).decode("ascii") != encoded
        or len(raw) > capability["max_image_bytes"]
        or len(raw) < 33
        or raw[:8] != b"\x89PNG\r\n\x1a\n"
        or raw[12:16] != b"IHDR"
        or struct.unpack(">I", raw[8:12])[0] != 13
    ):
        raise ValueError("invalid or oversized canonical PNG transport")
    width, height = struct.unpack(">II", raw[16:24])
    if not width or not height or width * height > capability["max_image_pixels"]:
        raise ValueError("image dimensions exceed the admitted pixel bound")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "width": width,
        "height": height,
        "transport": capability["transport"],
    }


def validate_image_terminal(
    plan: Mapping[str, Any],
    metrics: Mapping[str, Any],
    stage: Mapping[str, Any],
    *,
    request_id: str,
    input_identity: Mapping[str, Any] | None,
    seen_states: set[tuple[Any, ...]],
) -> dict[str, Any] | None:
    """Validate one real inner request; update replay protection only on success.

    Callers must first pass the strict load-plan gate. No text-only request can
    satisfy image proof. Incomplete/cancelled evidence is diagnostic, never pass.
    """
    telemetry = metrics["runtime_telemetry"]
    report = telemetry.get("image_observation")
    if input_identity is None:
        if report is not None:
            raise ValueError("text-only request unexpectedly contains image-chain evidence")
        return None
    image = plan["image_route"]
    bounds = image["bounds_and_declared_budgets"]
    if (
        not _sha(input_identity.get("sha256"))
        or not all(_positive(input_identity.get(key)) for key in ("size_bytes", "width", "height"))
        or input_identity["size_bytes"] > bounds["max_image_bytes"]
        or input_identity["width"] * input_identity["height"] > bounds["max_image_pixels"]
        or input_identity.get("transport") != "data:image/png;base64"
    ):
        raise ValueError("submitted image identity exceeds its admitted transport/dimensions")
    if (
        stage.get("terminal") is not True
        or stage.get("kind") != "text"
        or stage.get("request_id") != request_id
        or stage.get("stage_id") != plan["stage_id"]
        or stage.get("worker_generation") != plan["worker_generation"]
        or not isinstance(stage.get("worker_generation"), str)
        or not stage["worker_generation"]
        or not _positive(stage.get("epoch"))
        or not _positive(stage.get("seq"))
    ):
        raise ValueError("image terminal is not bound to this request, stage and worker state")
    if (
        not isinstance(report, Mapping)
        or report.get("schema") != "omni-strata-image-request-observation-v1"
        or report.get("status") != "complete"
        or report.get("request_id") != request_id
        or report.get("epoch") != stage["epoch"]
        or report.get("generation") != stage["worker_generation"]
        or report.get("input_sha256") != input_identity["sha256"]
        or report.get("scope") != "owned_native_encoder_then_pinned_server_SVE_GENI_chain"
        or report.get("owned_encoder") != plan["owned_encoder_at_load"]
        or report.get("backend_selection") != plan["encoder_backend_selection_at_load"]
        or report.get("backend_selection")
        != {
            "schema": "strata-vision-backend-v1",
            "primary_backend": "CPU",
            "device_type": "CPU",
            "gpu_requested": False,
            "cpu_fallback_available": True,
        }
        or report.get("all_encoder_operators_gpu_verified") is not False
        or report.get("whole_model_placement") is not None
        or report.get("physical_ssd_read_bytes") is not None
        or report.get("release_qualified") is not False
        or report.get("reasons") != []
    ):
        raise ValueError("owned CPU image chain is incomplete or belongs to another request/artifact")
    result, dispatch = report["encode_result"], report["language_dispatch"]
    bounds = image["bounds_and_declared_budgets"]
    language = plan["selected_native_module_audit_at_load"]
    io = telemetry.get("native_io_observation")
    if (
        not _positive(report.get("native_encoder_request_seq"))
        or not isinstance(result, Mapping)
        or not isinstance(dispatch, Mapping)
        or not all(_positive(result.get(key)) for key in ("image_tokens", "nx", "ny", "sve_bytes"))
        or result["image_tokens"] > bounds["max_image_tokens"]
        or result["nx"] * result["ny"] != result["image_tokens"]
        or result["sve_bytes"] != 20 + 4 * bounds["embedding_width"] * result["image_tokens"]
        or not _sha(result.get("sve_sha256"))
        or type(result.get("native_encoder_ms")) not in (int, float)
        or not math.isfinite(result["native_encoder_ms"])
        or result["native_encoder_ms"] < 0
        or dispatch.get("command") != "GENI"
        or dispatch.get("sve_sha256") != result["sve_sha256"]
        or dispatch.get("language_pid") != language["pid"]
        or dispatch.get("language_creation_filetime_100ns") != language["creation_filetime_100ns"]
        or not isinstance(io, Mapping)
        or io.get("schema") != "omni-strata-request-io-observation-v1"
        or io.get("status") != "complete"
        or io.get("request_id") != request_id
        or io.get("epoch") != stage["epoch"]
        or io.get("generation") != stage["worker_generation"]
        or io.get("native_pid") != dispatch["language_pid"]
        or io.get("creation_filetime_100ns") != dispatch["language_creation_filetime_100ns"]
        or io.get("runtime_identity_sha256") != plan["observation_runtime"]["identity_sha256"]
        or not _positive(io.get("native_request_seq"))
        or report.get("text_native_io_request_seq") != io["native_request_seq"]
        or report.get("text_native_io_scope") != io.get("scope")
        or io.get("scope") != "native_FileExpertSource_and_PLE_counters_excludes_loading"
        or io.get("physical_ssd_read_bytes") is not None
        or io.get("loading_covered") is not False
        or io.get("three_tier_memory_qualified") is not False
        or telemetry.get("native_io_excludes_encoder") is not True
        or telemetry.get("encoder_all_operators_placement") is not None
    ):
        raise ValueError("ENC/SVE/GENI sizes, hashes, owners or native terminal bindings differ")
    states = {
        ("request", stage["worker_generation"], request_id),
        ("state", stage["worker_generation"], stage["epoch"]),
        ("encoder", stage["worker_generation"], report["native_encoder_request_seq"]),
        ("language", stage["worker_generation"], io["native_request_seq"]),
    }
    if states & seen_states:
        raise ValueError("image request reuses a prior epoch or native request sequence")
    previous_epochs = [row[2] for row in seen_states if row[0] == "state" and row[1] == stage["worker_generation"]]
    if previous_epochs and stage["epoch"] <= max(previous_epochs):
        raise ValueError("image terminal epoch is stale or unordered")
    for role, current in (("encoder", report["native_encoder_request_seq"]), ("language", io["native_request_seq"])):
        previous = [row[2] for row in seen_states if row[0] == role and row[1] == stage["worker_generation"]]
        if previous and (current <= max(previous) or (role == "encoder" and current != max(previous) + 1)):
            raise ValueError("image native request sequence is stale, skipped or unordered")
    seen_states.update(states)
    return {
        "schema": IMAGE_PROOF_SCHEMA,
        "scope": "owned_image_functional_chain_not_task_qualification",
        "loaded_plan_sha256": digest(plan),
        "image_identity_sha256": image["identity_sha256"],
        "request_id": request_id,
        "input": copy.deepcopy(dict(input_identity)),
        "stage_event": copy.deepcopy(dict(stage)),
        "image_observation": copy.deepcopy(dict(report)),
        "native_io_observation": copy.deepcopy(dict(io)),
        "whole_model_placement": None,
        "physical_ssd_read_bytes": None,
        "release_qualified": False,
    }
