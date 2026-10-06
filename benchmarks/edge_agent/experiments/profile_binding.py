"""Bind a later native diagnostic to one audited batch-one Agent profile.

This records provenance only. It cannot sign a gate or qualify a route. A
pre-profile or older probe has no binding and must be rerun after the profile.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from vllm_omni.edge.agent.native_app import _fingerprint
from vllm_omni.edge.agent.qualification import audit_summary
from vllm_omni.edge.agent.runtime_identity import (
    imported_omni_source_sha256, loaded_runtime_sha256,
)


SCHEMA = "omni-agent-profile-binding-v1"
HARDWARE_IDENTITY_KEYS = (
    "os", "machine", "cpu", "host_ram_total_bytes", "gpu_name",
    "gpu_driver", "power_condition",
)


class ProfileBindingError(ValueError):
    """The diagnostic does not match the audited profile and live runtime."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ProfileBindingError(reason)


def bind_profile_to_live(
    profile_index: Path, *, source_config_bytes: bytes, route_id: str,
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify raw profile, then compare its exact route/runtime/hardware.

    Memory availability is deliberately excluded from stable hardware identity:
    admission measures live free bytes again. CPU, GPU, OS, driver, total RAM,
    and power condition must match the profiled machine exactly.
    """
    index_bytes = profile_index.read_bytes()
    index = json.loads(index_bytes)
    _require(isinstance(index, dict), "profile index must be a JSON object")
    results = index.get("results")
    _require(index.get("protocol") == "full_20x3_and_30m" and
             isinstance(results, list) and len(results) == 1 and
             isinstance(results[0], dict) and results[0].get("route_id") == route_id,
             "profile must have one full-protocol result for the diagnostic route")
    config_sha = hashlib.sha256(source_config_bytes).hexdigest()
    _require(config_sha == index.get("source_config_sha256"),
             "profile source config differs from diagnostic config")
    summary_path = Path(results[0]["summary"])
    audit = audit_summary(summary_path)
    _require(audit.internally_valid and audit.trace_verified and
             audit.protocol_compliant and audit.route_id == route_id and
             audit.raw_sha256 == results[0].get("raw_sha256"),
             "profile raw requests failed audit or route identity")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    conditions = index.get("conditions")
    _require(isinstance(conditions, dict) and
             summary.get("conditions") == conditions and
             summary.get("raw_sha256") == audit.raw_sha256,
             "profile index conditions differ from audited summary")
    profiled_hardware = index.get("hardware")
    _require(isinstance(profiled_hardware, dict),
             "profile index lacks hardware identity")
    identity = {key: hardware.get(key) for key in HARDWARE_IDENTITY_KEYS}
    _require(all(identity[key] == profiled_hardware.get(key)
                 for key in HARDWARE_IDENTITY_KEYS),
             "live hardware differs from profiled CPU, GPU, OS, RAM, driver, or power")
    _require(conditions.get("os_version") == identity["os"] and
             conditions.get("driver_versions", {}).get("nvidia") == str(identity["gpu_driver"]) and
             conditions.get("power_condition") == identity["power_condition"] and
             conditions.get("hardware_id") == (
                 f"{identity['cpu']} | {identity['gpu_name']} | "
                 f"RAM {identity['host_ram_total_bytes']} bytes"
             ), "profile hardware, OS, driver, or power conditions disagree")
    runtime = conditions.get("runtime_versions")
    _require(isinstance(runtime, dict), "profile has no runtime identity")
    source_sha = imported_omni_source_sha256()
    loaded_sha = loaded_runtime_sha256()
    environment = _fingerprint(dict(hardware))
    _require(runtime.get("vllm_omni_imported_source_sha256") == source_sha and
             runtime.get("agent_runtime_identity_sha256") == loaded_sha and
             conditions.get("environment_fingerprint") == environment,
             "live Omni source, runtime, or environment differs from profile")
    return {
        "schema": SCHEMA,
        "profile_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "profile_raw_sha256": audit.raw_sha256,
        "source_config_sha256": config_sha,
        "route_id": route_id,
        "environment_fingerprint": environment,
        "imported_omni_source_sha256": source_sha,
        "loaded_runtime_sha256": loaded_sha,
        "hardware_identity": identity,
    }
