# SPDX-License-Identifier: Apache-2.0
"""Backend-specific profile evidence, separate from release qualification.

Strata's native load configuration and DONE counters have different scopes.
Neither establishes the placement of every model operator. Generic backends
retain the existing exact placement comparison. Imports of the Strata stage
are deferred so profiling a CPU/llama route needs no optional stage runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

STRATA_BACKEND = "external.strata.text.v1"
STRATA_REVISION = "d5ea7133741e67743c0e886bb426c0ce8d69cf6c"
STRATA_SCOPE = "routed_decode_experts_only"
_CACHE_SCHEMA = "omni-strata-explicit-cache-v2"


def evidence_sha256(value: Any, *, ascii: bool = False) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=ascii,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def native_gpu_sample_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Recompute separate process/GPU-generation peaks from retained OS samples."""
    rows = [sample["native_process_gpu_memory"] for sample in samples if "native_process_gpu_memory" in sample]
    if not rows:
        return None
    from vllm_omni.edge.windows_gpu_memory import ProcessGpuPeakTracker

    tracker = ProcessGpuPeakTracker()
    for row in rows:
        if row.get("status") == "observed" and any(
            type(row.get(name)) is not int or row[name] < 0
            for name in ("local_current_usage_bytes", "nonlocal_current_usage_bytes")
        ):
            raise ValueError("observed native GPU accounting needs nonnegative integer byte readings")
        tracker.add(row)
    return {
        "sampled_peaks_by_generation": tracker.snapshot(),
        "unknown_samples": sum(row.get("status") != "observed" for row in rows),
        "sample_count": len(rows),
        "scope": "separate sampled process-local/nonlocal WDDM accounting; startup coverage not inferred",
        "hard_cap_verified": False,
        "nonlocal_is_additional_ram_pool": False,
    }


def strata_route_binding(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Bind all source shards, packed conversions, runtime and admitted controls.

    This reads manifest metadata only. The backend verifies actual file bytes
    before startup; a lineage author assertion remains separately reviewed.
    """
    from vllm_omni.engine.weight_tiers import ArtifactManifest, WeightTierPlan

    config = entry["backend_config"]
    if entry.get("backend") != STRATA_BACKEND or config.get("name") != STRATA_BACKEND:
        raise ValueError("Strata profile has a mismatched backend")
    if config.get("runtime_revision") != STRATA_REVISION:
        raise ValueError("Strata profile needs the pinned runtime revision")
    source = ArtifactManifest.from_dict(config["artifact_manifest"])
    runtime = ArtifactManifest.from_dict(config["runtime_manifest"])
    packed = ArtifactManifest.from_dict(config["prepared_pack_manifest"])
    tier = WeightTierPlan.from_dict(config["weight_tier_plan"])
    conversion = config["conversion_manifest"]
    if (
        entry.get("artifact_id") != "strata:" + evidence_sha256(config)
        or entry.get("model") != source.checkpoint
        or runtime.revision != STRATA_REVISION
        or tier.artifact_manifest_sha256 != source.manifest_sha256
        or tier.backend != STRATA_BACKEND
        or tier.backend_revision != STRATA_REVISION
        or conversion.get("complete") is not True
        or conversion.get("source_manifest_sha256") != source.manifest_sha256
        or conversion.get("prepared_manifest_sha256") != packed.manifest_sha256
        or conversion.get("tool_revision") != STRATA_REVISION
    ):
        raise ValueError("Strata profile artifact/conversion/runtime identity differs")
    layout = next((item for item in packed.files if item.path == "native_experts.txt"), None)
    engine = next((item for item in runtime.files if item.path == config["engine_file"]), None)
    environment = config["python_environment"]
    if (
        layout is None
        or engine is None
        or not isinstance(environment, Mapping)
        or environment.get("executable_sha256") != config["python_sha256"]
        or not isinstance(environment.get("sys_version"), str)
        or not environment["sys_version"]
        or not isinstance(environment.get("dependencies"), Mapping)
        or set(environment["dependencies"]) != {"numpy", "jinja2", "regex", "PyYAML", "psutil", "Pillow", "gguf"}
    ):
        raise ValueError("Strata layout, binary or Python environment identity is missing")
    spec = config.get("spec_tokens", 0)
    controls = {
        "schema": _CACHE_SCHEMA,
        "runtime_revision": STRATA_REVISION,
        "runtime_manifest_sha256": runtime.manifest_sha256,
        "artifact_manifest_sha256": source.manifest_sha256,
        "prepared_manifest_sha256": packed.manifest_sha256,
        "ram_expert_cache_budget_bytes": config["expert_ram_budget_bytes"],
        "gpu_budget_bytes": config["gpu_budget_bytes"],
        "vram_reserve_mib": max(
            config.get("vram_reserve_mib", 1024),
            math.ceil((config["gpu_total_bytes"] - config["gpu_budget_bytes"]) / (1 << 20)),
        ),
        "context_tokens": config["context_tokens"],
        "kv_type": config.get("kv_type", "int8"),
        "spec_tokens": spec,
        "native_verify_window": max(spec, 2),
        "prefetch": {
            "ple": config.get("ple_prefetch", False),
            "routing": config.get("routing_prefetch", False),
            "io": config.get("io_prefetch", False),
        },
        "io_mode": config.get("io_mode", "auto"),
        "ple_io": config.get("ple_io", "direct"),
        "expert_profile_file": config.get("expert_profile_file"),
    }
    observed_binding = None
    descriptor = config.get("observation_runtime")
    expected_observed_identity = config.get("observation_runtime_identity_sha256")
    if descriptor is not None:
        runtime_files = {item.path: item for item in runtime.files}
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("schema") != "omni-strata-observation-runtime-v1"
            or descriptor.get("base_revision") != STRATA_REVISION
            or descriptor.get("native_io_schema") != "strata-omni-io-v1"
            or descriptor.get("engine_file") != config["engine_file"]
            or not isinstance(expected_observed_identity, str)
            or re.fullmatch(r"[a-f0-9]{64}", expected_observed_identity) is None
        ):
            raise ValueError("Strata observation descriptor needs its registered exact runtime identity")
        references = {
            "native_executable_sha256": "engine_file",
            "patch_sha256": "patch_file",
            "build_receipt_sha256": "build_receipt_file",
            "dependency_provenance_sha256": "dependency_provenance_file",
            "runtime_dependencies_sha256": "runtime_dependencies_file",
        }
        if any(descriptor.get(key) not in runtime_files for key in references.values()):
            raise ValueError("Strata observation provenance is absent from its runtime manifest")
        prefix = descriptor.get("patched_sources_dir", "") + "/"
        patched_sources = {
            item.path[len(prefix) :]: item.sha256 for item in runtime.files if item.path.startswith(prefix)
        }
        if len(patched_sources) != 7:
            raise ValueError("Strata observation route needs seven manifest-bound patched sources")
        observed_binding = {
            "identity_sha256": expected_observed_identity,
            "descriptor_sha256": evidence_sha256(descriptor),
            "dependency_revision": descriptor.get("dependency_revision"),
            "dependency_tree": descriptor.get("dependency_tree"),
            "files": {field: runtime_files[descriptor[key]].sha256 for field, key in references.items()},
            "patched_sources": patched_sources,
        }
    elif expected_observed_identity is not None:
        raise ValueError("Strata observation identity cannot exist without a runtime descriptor")
    return {
        "schema": "omni-strata-profile-binding-v1",
        "backend_config_sha256": evidence_sha256(config),
        "checkpoint": source.checkpoint,
        "checkpoint_revision": source.revision,
        "artifact_manifest_sha256": source.manifest_sha256,
        "runtime_manifest_sha256": runtime.manifest_sha256,
        "prepared_manifest_sha256": packed.manifest_sha256,
        "runtime_revision": runtime.revision,
        "engine_sha256": engine.sha256,
        "conversion_manifest_sha256": evidence_sha256(conversion),
        "declared_python_environment_sha256": evidence_sha256(environment),
        # The backend's fresh subprocess probe verifies exactly these fields.
        # Declaration-only executable paths/probe-recipe metadata stay bound
        # through the full backend config without becoming observed facts.
        "python_environment_sha256": evidence_sha256(
            {key: environment[key] for key in ("sys_version", "dependencies", "executable_sha256")}
        ),
        "weight_tier_plan_sha256": evidence_sha256(tier.to_dict()),
        "native_layout_sha256": layout.sha256,
        "gpu_expert_cache_budget_bytes": tier.budget.gpu_expert_cache_bytes,
        "expected_controls": controls,
        "observation_runtime": observed_binding,
    }


def validate_strata_profile_plan(plan: Mapping[str, Any], route: Mapping[str, Any]) -> None:
    """Validate observed load proof and immutable identities; not a peak gate."""
    from vllm_omni.engine.backends.strata import _verify_cache_bounds, validate_strata_load_plan

    binding = route["backend_identity"]
    if (
        route["backend"] != STRATA_BACKEND
        or binding.get("schema") != "omni-strata-profile-binding-v1"
        or route["artifact_id"] != "strata:" + binding["backend_config_sha256"]
        or route["artifact_sha256"] != binding["artifact_manifest_sha256"]
        or route["model_id"] != binding["checkpoint"]
        or route["checkpoint_revision"] != binding["checkpoint_revision"]
    ):
        raise ValueError("Strata route source binding differs")
    validate_strata_load_plan(dict(plan), route["expected_placement"])
    if plan.get("observed_model_placement") is not None or plan.get("observed_compute_units") is not None:
        raise ValueError("Strata load evidence cannot claim whole-model compute placement")
    for key in ("artifact_manifest_sha256", "runtime_manifest_sha256", "prepared_manifest_sha256", "runtime_revision"):
        if plan.get(key) != binding[key]:
            raise ValueError(f"Strata loaded {key} differs from the source route")
    for field in ("conversion_manifest", "python_environment", "weight_tier_plan"):
        if evidence_sha256(plan.get(field)) != binding[field + "_sha256"]:
            raise ValueError(f"Strata loaded {field} identity differs")
    controls = plan["route_controls"]
    if (
        plan.get("route_controls_sha256") != evidence_sha256(controls, ascii=True)
        or {k: v for k, v in controls.items() if k not in {"gpu_expert_cache", "observation_runtime"}}
        != binding["expected_controls"]
    ):
        raise ValueError("Strata cache/control identity is missing, changed or unbound")
    observed_binding = binding.get("observation_runtime")
    observed_identity = plan.get("observation_runtime")
    if observed_binding is None:
        if observed_identity is not None or controls.get("observation_runtime") is not None:
            raise ValueError("Strata unregistered observation runtime appeared in the loaded plan")
    elif (
        not isinstance(observed_identity, Mapping)
        or observed_identity.get("schema") != "omni-strata-observed-runtime-v1"
        or observed_identity.get("identity_sha256") != observed_binding["identity_sha256"]
        or evidence_sha256(
            {key: value for key, value in observed_identity.items() if key != "identity_sha256"}, ascii=True
        )
        != observed_binding["identity_sha256"]
        or observed_identity.get("base_revision") != STRATA_REVISION
        or observed_identity.get("native_io_schema") != "strata-omni-io-v1"
        or observed_identity.get("dependency_revision") != observed_binding["dependency_revision"]
        or observed_identity.get("dependency_tree") != observed_binding["dependency_tree"]
        or any(observed_identity.get(key) != value for key, value in observed_binding["files"].items())
        or {key: value.get("patched_sha256") for key, value in observed_identity.get("patch_source_hashes", {}).items()}
        != observed_binding["patched_sources"]
        or observed_identity.get("three_tier_memory_qualified") is not False
        or observed_identity.get("supervisor_bootstrap_sha256") != plan.get("supervisor_bootstrap_sha256")
        or controls.get("observation_runtime") != observed_identity
    ):
        raise ValueError("Strata observed runtime/build/bootstrap identity differs from its registered route")
    control = controls["gpu_expert_cache"]
    names = (
        "layout_version",
        "layers",
        "max_blob_bytes",
        "aligned_max_blob_bytes",
        "budget_bytes",
        "requested_slots",
        "allocation_upper_bytes",
    )
    if (
        control.get("schema") != _CACHE_SCHEMA
        or control.get("layout_sha256") != binding["native_layout_sha256"]
        or any(type(control.get(name)) is not int or control[name] <= 0 for name in names)
        or control["layout_version"] not in {1, 2, 3, 4}
        or control["layers"] > 4096
        or control.get("alignment_bytes") != 256
        or control["aligned_max_blob_bytes"] != (control["max_blob_bytes"] + 255) // 256 * 256
        or control["budget_bytes"] != binding["gpu_expert_cache_budget_bytes"]
        or control["requested_slots"] != control["budget_bytes"] // control["aligned_max_blob_bytes"]
        or control["allocation_upper_bytes"] != control["requested_slots"] * control["aligned_max_blob_bytes"]
        or plan.get("gpu_expert_cache_control") != control
    ):
        raise ValueError("Strata positive native cache control is not valid for this artifact/budget")
    for field, source in (
        ("context_tokens", "context_tokens"),
        ("kv_type", "kv_type"),
        ("spec_tokens", "spec_tokens"),
        ("native_verify_window", "native_verify_window"),
        ("gpu_budget_bytes", "gpu_budget_bytes"),
        ("expert_ram_budget_bytes", "ram_expert_cache_budget_bytes"),
    ):
        if plan.get(field) != controls[source]:
            raise ValueError("Strata loaded controls contradict the plan")
    reconstructed = _verify_cache_bounds(
        plan["execution_configuration_evidence"]["engine_info"],
        control,
        controls["ram_expert_cache_budget_bytes"],
        profiled=bool(controls["expert_profile_file"]),
    )
    observed = plan["expert_cache_component_bounds"]
    if any(observed.get(key) != value for key, value in reconstructed.items()):
        raise ValueError("Strata native INFO component bounds differ")


def validate_strata_request_evidence(
    evidence: Mapping[str, Any], route: Mapping[str, Any], *, events: Sequence[Mapping[str, Any]] | None = None
) -> None:
    """Check every model step's DONE counts, terminal identity and load binding."""
    plan = evidence["execution_plan"]
    validate_strata_profile_plan(plan, route)
    if evidence.get("loaded_plan_sha256") != evidence_sha256(plan):
        raise ValueError("Strata full loaded plan hash differs or is missing")
    terminals = evidence["terminal_model_metrics"]
    if not isinstance(terminals, list) or not terminals:
        raise ValueError("Strata request has no native terminal metrics")
    if type(evidence.get("model_prompt_step_count")) is not int or evidence["model_prompt_step_count"] != len(
        terminals
    ):
        raise ValueError("Strata model-step count differs from native terminal metrics")
    request_id = None
    if events is not None:
        observed = [event.get("payload", {}) for event in events if event.get("kind") == "model_metrics"]
        if terminals != observed:
            raise ValueError("Strata terminal evidence differs from ordered Agent events")
        identity = next(event["payload"] for event in events if event.get("kind") == "route")
        if (
            identity.get("actual_placement") is not None
            or identity.get("verified_execution_configuration") != route["expected_placement"]
            or identity.get("execution_configuration_evidence") != plan["execution_configuration_evidence"]
        ):
            raise ValueError("Strata Agent route event differs from the loaded configuration")
        request_id = events[0]["request_id"]
    seen = set()
    for index, terminal in enumerate(terminals):
        if type(terminal.get("step")) is not int or terminal["step"] != index:
            raise ValueError("Strata model steps are duplicated or unordered")
        metrics = terminal["metrics"]
        stage = metrics["stage_event"]
        if (
            stage.get("terminal") is not True
            or stage.get("kind") != "text"
            or stage.get("stage_id") != plan["stage_id"]
            or stage.get("worker_generation") != plan["worker_generation"]
            or not isinstance(stage.get("worker_generation"), str)
            or not stage["worker_generation"]
            or type(stage.get("epoch")) is not int
            or stage["epoch"] < 1
            or type(stage.get("seq")) is not int
            or stage["seq"] < 1
            or not isinstance(stage.get("request_id"), str)
            or not stage["request_id"]
            or (request_id is not None and stage["request_id"] != f"{request_id}-step-{index}")
        ):
            raise ValueError("Strata terminal metrics have a mismatched request/state identity")
        state = (stage["epoch"], stage["worker_generation"])
        if state in seen:
            raise ValueError("Strata terminal state is reused")
        seen.add(state)
        telemetry = metrics["backend_metrics"]["runtime_telemetry"]
        io_observation = telemetry.get("native_io_observation")
        if io_observation is not None:
            observed_binding = route["backend_identity"].get("observation_runtime")
            if (
                observed_binding is None
                or not isinstance(io_observation, Mapping)
                or io_observation.get("schema") != "omni-strata-request-io-observation-v1"
                or io_observation.get("runtime_identity_sha256") != observed_binding["identity_sha256"]
                or io_observation.get("generation") != stage["worker_generation"]
                or io_observation.get("request_id") != stage["request_id"]
                or io_observation.get("epoch") != stage["epoch"]
                or io_observation.get("scope") != "native_FileExpertSource_and_PLE_counters_excludes_loading"
                or io_observation.get("status") not in {"complete", "incomplete"}
                or io_observation.get("physical_ssd_read_bytes") is not None
                or io_observation.get("loading_covered") is not False
                or io_observation.get("three_tier_memory_qualified") is not False
            ):
                raise ValueError("Strata native I/O observation differs from its runtime/request identity or scope")
        if (
            telemetry.get("logical_file_read_bytes") is not None
            and telemetry.get("logical_file_read_scope") != "decode_only_excludes_prefill_and_loading"
        ):
            raise ValueError("Strata logical file reads lack their decode-only measurement scope")
        compute = telemetry["native_compute"]
        counters = compute["counters"]
        if (
            compute.get("scope") != STRATA_SCOPE
            or compute.get("evidence") != "native_DONE; pinned native-pack miss dispatch; no remote GPU routes"
            or not isinstance(counters, Mapping)
            or set(counters) != {"generated", "prompt_tokens", "hits", "lookups", "offloaded", "prompt_read"}
            or any(type(value) is not int or value < 0 for value in counters.values())
            or counters["generated"] < 1
            or counters["hits"] > counters["lookups"]
        ):
            raise ValueError("Strata native DONE decode evidence is unknown or malformed")
        cpu, gpu = counters["lookups"] - counters["hits"], counters["hits"] + counters["offloaded"]
        units = (["cpu"] if cpu else []) + ([route["expected_placement"].split("+", 1)[1]] if gpu else [])
        if (
            not units
            or type(compute.get("cpu_expert_entries")) is not int
            or type(compute.get("gpu_expert_entries")) is not int
            or compute["cpu_expert_entries"] != cpu
            or compute["gpu_expert_entries"] != gpu
            or compute.get("units") != units
        ):
            raise ValueError("Strata CPU/CUDA decode counters contradict their scoped compute report")


def result_placement_matches(result: Mapping[str, Any], route: Mapping[str, Any]) -> bool:
    if (
        result.get("model_id") != route["model_id"]
        or result.get("artifact_id") != route["artifact_id"]
        or result.get("backend") != route["backend"]
    ):
        return False
    if route["backend"] == "external.strata.multimodal.v1":
        # Image functional receipts do not inherit text-suite qualification.
        return False
    if route["backend"] != STRATA_BACKEND:
        return result.get("actual_placement") == route["expected_placement"]
    try:
        if result.get("actual_placement") is not None:
            return False
        validate_strata_request_evidence(result["placement_evidence"], route)
        return True
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError):
        return False


def preparation_placement_matches(preparation: Mapping[str, Any], route: Mapping[str, Any]) -> bool:
    if not preparation.get("cold_start_confirmed") or preparation.get("artifact_id") != route["artifact_id"]:
        return False
    if route["backend"] == "external.strata.multimodal.v1":
        return False
    if route["backend"] != STRATA_BACKEND:
        return preparation.get("actual_placement") == route["expected_placement"]
    try:
        details = preparation["details"]
        plan = details["execution_plan"]
        if preparation.get("actual_placement") is not None or details.get("loaded_plan_sha256") != evidence_sha256(
            plan
        ):
            return False
        validate_strata_profile_plan(plan, route)
        return True
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError):
        return False
