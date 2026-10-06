# SPDX-License-Identifier: Apache-2.0
"""Assemble private, unsigned Gemma Agent gate observations from raw runs.

This is a transcription and consistency check, not an independent review.
Only evidence that matches the audited batch-one profile is transcribed into
``*_evidence_v1`` raw sources. No pass receipt, review key, signature, router
qualification, or public aggregate is created. The report names every missing
input and limitation, including stale profile source and unsigned observations.

Run under the native Windows Agent Python environment. Input paths are
explicit; output is a new directory under private LocalAppData by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

from benchmarks.edge_agent.experiments.native_memory_admission import (
    compare_samples, verify_resident_claim,
)
from benchmarks.edge_agent.download import load_manifest
from benchmarks.edge_agent.experiments.probe_gemma_lineage import (
    verify_record as verify_lineage_record,
)
from benchmarks.edge_agent.experiments.snapshot_runtime_placement import (
    _parse_log, _request_proof, verify_record as verify_placement_record,
)
from vllm_omni.edge.agent.qualification import (
    REQUIRED_GATES, _identity, _verify_gate_observations, audit_summary,
    verify_fixed_suite_cases,
)
from vllm_omni.edge.agent.runtime_identity import (
    imported_omni_source_sha256, loaded_runtime_sha256,
)


ASSEMBLY_SCHEMA = "omni-agent-unsigned-gate-assembly-v1"
PROFILE_BINDING_SCHEMA = "omni-agent-profile-binding-v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_DIAGNOSTIC_ONLY = {
    "memory_admission": (
        "sampled_global_peak_limit",
        "whole-host periodic samples can miss transient allocations and include unrelated processes",
    ),
    "runtime_placement": (
        "compute_placement_unverified",
        "startup offload and model buffers do not prove per-operation CPU/GPU compute placement",
    ),
}


class EvidenceBlocked(ValueError):
    """One raw source cannot support a gate observation."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise EvidenceBlocked(code, detail)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "invalid_json_object", "expected a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    _require(bool(rows) and all(isinstance(row, dict) for row in rows),
             "invalid_jsonl", "expected nonempty JSONL objects")
    return rows


def _only(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    found = [row for row in rows if row.get("record_type") == kind]
    _require(len(found) == 1, "record_count", f"expected exactly one {kind} record")
    return found[0]


def _ordered_cancel_events(events: list[dict[str, Any]]) -> None:
    """Require the raw capture order to be two noninterleaved Agent turns."""
    _require(bool(events) and all(isinstance(row.get("kind"), str) and
                                  isinstance(row.get("request_id"), str) and
                                  type(row.get("epoch")) is int and
                                  type(row.get("seq")) is int for row in events),
             "cancel_event_order", "cancel trace contains a non-event row")
    epochs = [row["epoch"] for row in events]
    _require(epochs == sorted(epochs) and set(epochs) == {1, 2},
             "cancel_event_order", "cancel and recovery epochs are interleaved or absent")
    for epoch in (1, 2):
        turn = [row for row in events if row["epoch"] == epoch]
        _require(len({row["request_id"] for row in turn}) == 1 and
                 [row["seq"] for row in turn] == list(range(1, len(turn) + 1)),
                 "cancel_event_order", "raw request events are duplicated or out of order")


def _file_ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve(strict=True)), "sha256": _sha(path)}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _same_route(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    # An isolated probe chooses a private log path; the executable route and
    # all its budgets, artifact hashes, limits and placement must still match.
    return ({key: value for key, value in observed.items() if key != "log_file"}
            == {key: value for key, value in expected.items() if key != "log_file"})


def _clean_ledger(snapshot: Mapping[str, Any]) -> bool:
    ledger = snapshot.get("ledger", snapshot)
    return (snapshot.get("resident_route") is None and
            ledger.get("owners") == [] and ledger.get("quarantined") == [] and
            isinstance(ledger.get("reserved"), dict) and
            all(value == 0 for value in ledger["reserved"].values()))


def identity_binding_gaps(manifest: Mapping[str, Any], hardware: Mapping[str, Any],
                          ctx: Mapping[str, Any]) -> list[str]:
    """Name missing exact profile/runtime bindings in independent run records.

    A matching route/config and similar hardware are insufficient: a probe
    cannot borrow the profile's runtime digest or raw hash after the fact.
    Older probes lacked these fields and remain useful diagnostics only.
    """
    identity = ctx["identity"]
    conditions = ctx["summary"]["conditions"]
    profiled_runtime = conditions.get("runtime_versions", {})
    binding = manifest.get("profile_binding")
    if not isinstance(binding, dict):
        return ["profile_binding"]
    required = {
        "schema": PROFILE_BINDING_SCHEMA,
        "profile_index_sha256": _sha(ctx["index_path"]),
        "profile_raw_sha256": identity["profile_raw_sha256"],
        "source_config_sha256": ctx["index"]["source_config_sha256"],
        "route_id": identity["route_id"],
        "environment_fingerprint": identity["environment_fingerprint"],
        "imported_omni_source_sha256": profiled_runtime.get(
            "vllm_omni_imported_source_sha256"),
        "loaded_runtime_sha256": profiled_runtime.get(
            "agent_runtime_identity_sha256"),
    }
    gaps = [name for name, expected in required.items()
            if not isinstance(expected, str) or binding.get(name) != expected]
    hardware_fields = ("os", "machine", "cpu", "host_ram_total_bytes",
                       "gpu_name", "gpu_driver", "power_condition")
    observed_hardware = {name: hardware.get(name) for name in hardware_fields}
    if binding.get("hardware_identity") != observed_hardware:
        gaps.append("hardware_identity")
    profiled_hardware = ctx["index"].get("hardware", {})
    if any(observed_hardware[name] != profiled_hardware.get(name)
           for name in hardware_fields):
        gaps.append("profile_hardware_identity")
    hardware_id = (f"{hardware.get('cpu')} | {hardware.get('gpu_name')} | "
                   f"RAM {hardware.get('host_ram_total_bytes')} bytes")
    if hardware_id != conditions.get("hardware_id"):
        gaps.append("hardware_id")
    if (hardware.get("os") != conditions.get("os_version") or
        str(hardware.get("gpu_driver")) != conditions.get("driver_versions", {}).get("nvidia") or
        hardware.get("power_condition") != identity["power_condition"]):
        gaps.append("hardware_os_driver_power")
    stable = {name: hardware.get(name) for name in
              ("os", "machine", "cpu", "gpu_name", "gpu_driver")}
    stable["loaded_agent_runtime_sha256"] = binding.get("loaded_runtime_sha256")
    calculated_fingerprint = hashlib.sha256(
        json.dumps(stable, sort_keys=True).encode()).hexdigest()
    if calculated_fingerprint != identity["environment_fingerprint"]:
        gaps.append("recomputed_environment_fingerprint")
    return gaps


def profile_context(index_path: Path) -> dict[str, Any]:
    index = _json(index_path)
    results = index.get("results")
    _require(index.get("protocol") == "full_20x3_and_30m" and
             index.get("automatic_qualification_export") is False and
             isinstance(results, list) and len(results) == 1 and
             results[0].get("task_class") == "memory",
             "profile_scope", "profile index must contain one Gemma memory route")
    summary_path = Path(results[0]["summary"]).resolve(strict=True)
    audit = audit_summary(summary_path)
    _require(audit.internally_valid and audit.trace_verified and audit.protocol_compliant,
             "profile_audit", "raw profile failed full trace or batch-one protocol audit: " +
             "; ".join(audit.errors))
    try:
        verify_fixed_suite_cases(audit)
    except ValueError as exc:
        raise EvidenceBlocked("fixed_suite_case_mismatch", str(exc)) from exc
    summary = _json(summary_path)
    _require(index.get("conditions") == summary.get("conditions") and
             results[0].get("raw_sha256") == audit.raw_sha256 and
             results[0].get("route_id") == audit.route_id,
             "profile_index_mismatch", "profile index differs from audited raw samples")
    profile = summary["routes"][audit.route_id]
    _require(profile.get("task_class") == "memory" and
             profile.get("measured_attempts") >= 60 and
             profile.get("measured_successes") == profile.get("measured_attempts"),
             "profile_quality", "profile has incomplete or failed measured requests")
    route_id = audit.route_id
    source_config = Path(index["source_config"]).resolve(strict=True)
    _require(_sha(source_config) == index.get("source_config_sha256"),
             "source_config_changed", "profile source config SHA-256 changed")
    config = _json(source_config)
    routes = [item for item in config.get("routes", ()) if item.get("route_id") == route_id]
    _require(len(routes) == 1, "route_missing", "source config has no unique profile route")
    route = routes[0]
    declared = profile["route"]
    _require(route.get("model_sha256") == declared["artifact_sha256"] and
             route.get("artifact_id") == declared["artifact_id"] and
             route.get("placement") == declared["expected_placement"],
             "route_identity", "source route differs from profile route identity")
    provenance = index.get("artifact_provenance", {}).get(route_id, {})
    _require(provenance.get("lineage_verified") is True and
             provenance.get("model_sha256") == route["model_sha256"] and
             provenance.get("server_sha256") == route["server_sha256"] and
             provenance.get("mmproj_sha256") == route.get("mmproj_sha256"),
             "profile_lineage", "profile index lacks pinned model, server or projector lineage")
    return {
        "index_path": index_path.resolve(strict=True), "index": index,
        "summary_path": summary_path, "summary": summary,
        "source_config_path": source_config, "config": config, "route": route,
        "identity": _identity(declared, summary["conditions"], audit.raw_sha256),
        "audit": audit,
    }


def memory_observations(path: Path, ctx: Mapping[str, Any]) -> dict[str, Any]:
    rows = _jsonl(path)
    manifest = rows[0]
    route, identity = ctx["route"], ctx["identity"]
    _require(sum(row.get("record_type") == "manifest" for row in rows) == 1 and
             manifest.get("record_type") == "manifest" and
             manifest.get("scope") == "independent_memory_admission_probe" and
             manifest.get("batch_size") == manifest.get("concurrency") == 1 and
             manifest.get("route_id") == identity["route_id"] and
             manifest.get("source_config_sha256") == ctx["index"]["source_config_sha256"] and
             manifest.get("model_sha256") == route["model_sha256"] and
             manifest.get("server_sha256") == route["server_sha256"] and
             manifest.get("mmproj_sha256") == route.get("mmproj_sha256"),
             "memory_identity", "memory probe is not the audited batch-one route/config")
    probe_config = Path(manifest["probe_config"]).resolve(strict=True)
    _require(_sha(probe_config) == manifest.get("probe_config_sha256") and
             _same_route(_json(probe_config)["routes"][0], route),
             "memory_probe_config", "probe config differs from the profiled route")
    hardware = _only(rows, "hardware")["hardware"]
    _require(hardware.get("os") == ctx["summary"]["conditions"]["os_version"] and
             hardware.get("gpu_driver") == ctx["summary"]["conditions"]["driver_versions"].get("nvidia") and
             hardware.get("power_condition") == identity["power_condition"],
             "memory_hardware", "memory probe OS, driver, or power condition differs")
    demands = route["memory_demands"]
    _require(set(demands) == {"host_ram", "vram"},
             "memory_pools", "this assembler expects explicit host RAM and VRAM pools")
    ceilings = {"host_ram": hardware["host_ram_available_bytes"],
                "vram": hardware["vram_available_bytes"]}
    refusals = [row for row in rows if row.get("record_type") == "over_ceiling_refusal"]
    _require(len(refusals) == len(ceilings) and {row.get("pool") for row in refusals} == set(ceilings),
             "memory_refusal", "one real ledger over-ceiling refusal per pool is required")
    refused: dict[str, int] = {}
    reasons: list[str] = []
    for row in refusals:
        pool = row["pool"]
        _require(row.get("ceiling_bytes") == ceilings[pool] and
                 type(row.get("declared_bytes")) is int and
                 row["declared_bytes"] > ceilings[pool] and
                 row.get("physical_allocation_attempted") is False and
                 isinstance(row.get("refusal"), str) and row["refusal"] and
                 _clean_ledger(row.get("ledger_after_refusal", {})),
                 "memory_refusal", f"{pool} refusal is not backed by a clean Omni ledger")
        refused[pool] = row["declared_bytes"]
        reasons.append(row["refusal"])
    _require(_only(rows, "pre_load_admission").get("admitted") is True,
             "memory_admission", "route was not admitted before cold load")
    plan = _only(rows, "loaded_plan")["plan"]
    _require(plan.get("requested_device") == identity["placement"] and
             plan.get("reserved_bytes") == demands,
             "memory_plan", "loaded Omni plan differs from declared placement or memory")
    for record_type in ("host_ledger_after_load", "host_ledger_after_request"):
        verify_resident_claim(_only(rows, record_type)["snapshot"], identity["route_id"], demands)
    _require(_clean_ledger(_only(rows, "host_ledger_after_release")["snapshot"]),
             "memory_release", "host claim remained after worker release")
    samples = [row for row in rows if row.get("record_type") == "sample"]
    ticks = [row.get("monotonic_ns") for row in samples]
    _require(bool(samples) and all(type(tick) is int and tick > 0 for tick in ticks) and
             ticks == sorted(ticks),
             "memory_samples", "memory samples are absent or out of order")
    comparison = compare_samples(samples, demands)
    _require(_only(rows, "sampled_comparison").get("comparison") == comparison,
             "memory_comparison", "stored memory peak differs from all raw samples")
    outcome = _only(rows, "outcome")
    _require(outcome.get("result") == "one_complete_agent_turn_measured" and
             outcome.get("error") is None and outcome.get("sample_count") == len(samples) and
             outcome.get("event_kinds", []).count("final") == 1 and
             not set(outcome.get("event_kinds", ())) & {"error", "refusal", "cancelled"} and
             isinstance(outcome.get("answer_sha256"), str) and
             bool(_HEX64.fullmatch(outcome["answer_sha256"])),
             "memory_request", "memory probe lacks one complete request and clean outcome")
    observations = {
        "admitted_demand_bytes": dict(demands),
        "pool_used_baseline_bytes": {pool: comparison[pool]["pre_load_used_bytes"] for pool in demands},
        "pool_used_peak_bytes": {pool: comparison[pool]["sampled_global_used_peak_bytes"] for pool in demands},
        "route_incremental_peak_bytes": {pool: comparison[pool]["sampled_global_incremental_peak_bytes"] for pool in demands},
        "available_ceiling_bytes": ceilings,
        "refused_demand_bytes": refused,
        "refusal_reason": "; ".join(reasons),
        "sample_count": len(samples),
        "measurement_scope": "sampled_whole_host_increment_not_process_allocation_or_transient_upper_bound",
    }
    _verify_gate_observations("memory_admission", observations, identity, path.parent)
    return observations


def source_binding_gaps(gate: str, path: Path,
                        ctx: Mapping[str, Any]) -> list[str]:
    """Check environment/runtime binding separately from observed values."""
    if gate == "memory_admission":
        rows = _jsonl(path)
        return identity_binding_gaps(rows[0], _only(rows, "hardware")["hardware"], ctx)
    if gate in {"cancel_recovery", "runtime_placement"}:
        rows = _jsonl(path if gate == "cancel_recovery" else
                      Path(_json(path)["source_record"]["path"]))
        return identity_binding_gaps(rows[0], rows[0].get("hardware", {}), ctx)
    if gate == "reference_quality":
        rows = _jsonl(path.parent / "raw.jsonl")
        summary = _only(rows, "summary")
        hardware_by_phase = summary.get("hardware_by_phase", {})
        hardware = hardware_by_phase.get("replay", {})
        return identity_binding_gaps(rows[0], hardware, ctx)
    return []


def cancel_observations(path: Path, ctx: Mapping[str, Any]) -> dict[str, Any]:
    rows = _jsonl(path)
    manifest, events = rows[0], rows[1:]
    route, identity = ctx["route"], ctx["identity"]
    config = manifest.get("config", {})
    _require(manifest.get("record_type") == "manifest" and
             manifest.get("scope") == "cancel_and_recover_functional_smoke" and
             manifest.get("batch_size") == manifest.get("concurrency") == 1 and
             config.get("qualification_suite_id") == identity["suite_id"] and
             len(config.get("routes", ())) == 1 and
             _same_route(config["routes"][0], route),
             "cancel_identity", "cancel/recovery trace differs from the profile route")
    _require(manifest.get("result") == "cancelled_stream_then_restarted_request_passed" and
             manifest.get("exact_answer_pass") is True,
             "cancel_outcome", "cancel/recovery experiment did not complete")
    _ordered_cancel_events(events)
    proof = _request_proof(manifest, events, config["routes"][0])
    _require(proof["recovered_epoch"] > 1,
             "cancel_recovery", "recovered request epoch did not advance")
    observations = {"events": [
        {**{key: event[key] for key in ("request_id", "epoch", "seq", "kind")},
         **({"payload": event.get("payload", {})} if event.get("kind") == "state_released" else {})}
        for event in events
    ]}
    _verify_gate_observations("cancel_recovery", observations, identity, path.parent)
    return observations


def lineage_observations(path: Path, ctx: Mapping[str, Any], *,
                         manifest_path: Path, source_path: Path) -> dict[str, Any]:
    record = _json(path)
    route, identity = ctx["route"], ctx["identity"]
    _require(verify_lineage_record(record) and
             record.get("record_type") == "checkpoint_lineage_probe_v1" and
             record.get("status") == "observed" and
             record.get("human_review_required") is True and
             record.get("route_id") == identity["route_id"] and
             record.get("native_config_sha256") == ctx["index"]["source_config_sha256"] and
             record.get("manifest_sha256") == _sha(manifest_path) and
             record.get("source_metadata_sha256") == _sha(source_path),
             "lineage_binding", "lineage observation or pinned metadata hash differs")
    source = _json(source_path)
    observation = record.get("observations_for_review", {})
    _require(observation.get("artifact_sha256") == identity["artifact_sha256"] and
             observation.get("projector_sha256") == route.get("mmproj_sha256") and
             observation.get("checkpoint_revision") == identity["checkpoint_revision"] and
             observation.get("source_repo") == source.get("source_repo") and
             observation.get("license") == source.get("license") and
             source.get("revision") == identity["checkpoint_revision"],
             "lineage_identity", "model, projector, revision, source, or license differs")
    artifacts = record.get("artifacts", [])
    pinned = load_manifest(manifest_path)
    _require(len(artifacts) == 2 and {item.get("role") for item in artifacts} == {"model", "projector"} and
             len(pinned) == 2 and
             {item.get("expected_sha256") for item in artifacts} ==
             {item.sha256 for item in pinned} and
             {item.get("expected_size_bytes") for item in artifacts} ==
             {item.size for item in pinned} and
             all(item.get("verified") is True and
                 item.get("observed_sha256") == item.get("expected_sha256") and
                 item.get("observed_size_bytes") == item.get("expected_size_bytes")
                 for item in artifacts),
             "lineage_artifacts", "local model/projector files were not verified by the probe")
    remote = record.get("remote", {})
    remote_artifacts = remote.get("remote_artifacts", ()) if isinstance(remote, dict) else ()
    _require(remote.get("checkpoint_revision") == identity["checkpoint_revision"] and
             remote.get("source_repo") == observation["source_repo"] and
             len(remote_artifacts) == 2 and
             {item.get("lfs_oid_sha256") for item in remote_artifacts} ==
             {item["expected_sha256"] for item in artifacts},
             "lineage_remote", "pinned remote model/projector metadata differs")
    observations = dict(observation)
    _verify_gate_observations("checkpoint_lineage", observations, identity, path.parent)
    return observations


def placement_observations(path: Path, ctx: Mapping[str, Any],
                           cancel_path: Path) -> dict[str, Any]:
    record = _json(path)
    route, identity = ctx["route"], ctx["identity"]
    _require(verify_placement_record(record) and
             record.get("schema") == "omni-agent-placement-snapshot-v1" and
             record.get("record_type") == "runtime_placement_observation_v1" and
             record.get("status") == "observed_not_qualified" and
             record.get("source_record", {}).get("sha256") == _sha(cancel_path),
             "placement_binding", "placement snapshot is not bound to the cancel/recovery trace")
    cancel_rows = _jsonl(cancel_path)
    cancel_manifest = cancel_rows[0]
    _ordered_cancel_events(cancel_rows[1:])
    _require(_same_route(cancel_manifest["config"]["routes"][0], route) and
             record.get("config", {}).get("route_id") == identity["route_id"] and
             record.get("config", {}).get("route_canonical_sha256") ==
             hashlib.sha256(_canonical(cancel_manifest["config"]["routes"][0])).hexdigest(),
             "placement_route", "placement snapshot route differs from profile/cancel route")
    proof = _request_proof(cancel_manifest, cancel_rows[1:], cancel_manifest["config"]["routes"][0])
    _require(record.get("request_proof") == proof,
             "placement_request", "placement snapshot recovery proof differs from raw events")
    log_ref = record.get("startup_log", {})
    log_name = log_ref.get("path")
    _require(isinstance(log_name, str) and Path(log_name).name == log_name,
             "placement_log_path", "startup log copy must be beside the placement snapshot")
    log_path = (path.parent / log_name).resolve(strict=True)
    log_bytes = log_path.read_bytes()
    _require(_sha(log_path) == log_ref.get("sha256") and
             len(log_bytes) == log_ref.get("size_bytes") and
             _parse_log(log_bytes, cancel_manifest["config"]["routes"][0]) == record.get("placement"),
             "placement_log", "startup log differs from the verified snapshot")
    artifacts = record.get("artifacts", {})
    _require(all(artifacts.get(role, {}).get("sha256") == route[field]
                 for role, field in (("model", "model_sha256"),
                                     ("server", "server_sha256"),
                                     ("projector", "mmproj_sha256"))) and
             record["placement"].get("reported_placement") == identity["placement"],
             "placement_artifacts", "placement snapshot artifact or device differs")
    observations = {
        "reported_placement": record["placement"]["reported_placement"],
        "artifact_sha256": identity["artifact_sha256"],
        "startup_log": _file_ref(log_path),
        "offloaded_layers": record["placement"]["offloaded_layers"],
        "total_layers": record["placement"]["total_layers"],
        "compute_placement_verified": False,
    }
    _verify_gate_observations("runtime_placement", observations, identity, path.parent, route)
    return observations


def quality_observations(path: Path, ctx: Mapping[str, Any]) -> dict[str, Any]:
    index = _json(path)
    raw_path = path.parent / "raw.jsonl"
    _require(_sha(raw_path) == index.get("raw_sha256") and
             index.get("status") == "completed" and
             index.get("review_status") == "unreviewed_private_experiment" and
             index.get("qualifies_default") is False,
             "quality_index", "quality index is missing, failed, or not bound to raw data")
    rows = _jsonl(raw_path)
    manifest = rows[0]
    identity = ctx["identity"]
    _require(manifest.get("record_type") == "manifest" and
             manifest.get("batch_size") == manifest.get("concurrency") == 1 and
             manifest.get("route", {}).get("route_id") == identity["route_id"] and
             manifest.get("route", {}).get("artifact_sha256") == identity["artifact_sha256"] and
             manifest.get("native_config_sha256") == ctx["index"]["source_config_sha256"] and
             manifest.get("suite_id") != identity["suite_id"] and
             manifest.get("loaded_runtime_sha256") == loaded_runtime_sha256(),
             "quality_identity", "quality run has another route, source config, suite, or runtime")
    cases = [row for row in rows if row.get("record_type") == "quality_case"]
    summary = _only(rows, "summary")
    _require(len(cases) >= 2 and {row.get("language") for row in cases} >= {"en-US", "zh-CN"} and
             summary.get("cases") == len(cases) and summary.get("passed") == len(cases) and
             index.get("summary") == {key: value for key, value in summary.items()
                                      if key not in {"record_type", "hardware_by_phase"}},
             "quality_summary", "quality run lacks a fully passing bilingual case set")
    observations_cases = []
    for row in cases:
        checks = row.get("checks", {})
        recall = row.get("recall", {})
        deletion = row.get("deletion", {})
        source_id, distractor_id = row.get("source_event_id"), row.get("distractor_event_id")
        reference = row.get("reference")
        answer = recall.get("answer")
        _require(isinstance(reference, str) and isinstance(answer, str) and
                 row.get("reference_sha256") == hashlib.sha256(reference.encode()).hexdigest() and
                 recall.get("answer_sha256") == hashlib.sha256(answer.encode()).hexdigest() and
                 answer == reference and
                 checks.get("passed") is True and all(value is True for value in checks.values()) and
                 recall.get("trace_ok") is True and row.get("after_delete", {}).get("trace_ok") is True and
                 row["after_delete"].get("answer") == "UNKNOWN" and
                 source_id in recall.get("final_derived_from", ()) and
                 distractor_id in recall.get("final_derived_from", ()) and
                 source_id in row.get("before_retrieved_event_ids", ()) and
                 distractor_id in row.get("before_retrieved_event_ids", ()) and
                 source_id not in deletion.get("after_retrieved_event_ids", ()) and
                 deletion.get("source_absent") is True and
                 deletion.get("distractor_retained") is True and
                 deletion.get("derived_final_absent") is True and
                 deletion.get("answer_index_absent") is True,
                 "quality_case", f"quality case {row.get('case_id')} failed raw answer or provenance checks")
        observations_cases.append({
            "case_id": row["case_id"], "language": row["language"],
            "task_class": "memory", "passed": True,
            "comparison": "exact_sha256",
            "reference_sha256": row["reference_sha256"],
            "answer_sha256": recall["answer_sha256"],
        })
    observations = {"task_class": "memory", "reference_suite_id": manifest["suite_id"],
                    "cases": observations_cases}
    _verify_gate_observations("reference_quality", observations, identity, path.parent)
    return observations


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False,
                  allow_nan=False)
        stream.write("\n")


def assemble(
    *, profile_index: Path, output_root: Path,
    memory_probe: Path | None = None, cancel_record: Path | None = None,
    lineage_probe: Path | None = None, download_manifest: Path | None = None,
    source_metadata: Path | None = None, placement_snapshot: Path | None = None,
    quality_index: Path | None = None,
) -> Path:
    """Create one private no-clobber draft directory and return its report."""
    _require(output_root.is_absolute() and
             "public_evidence" not in {part.casefold() for part in output_root.parts},
             "private_output_required", "gate drafts must use an absolute private output root")
    ctx = profile_context(profile_index)
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / f"gate_assembly_{uuid.uuid4().hex}"
    out.mkdir(mode=0o700)
    identity = ctx["identity"]
    report: dict[str, Any] = {
        "schema": ASSEMBLY_SCHEMA, "identity": identity,
        "review_status": "unsigned_human_review_required",
        "router_qualification_created": False,
        "profile": {"index": _file_ref(ctx["index_path"]),
                    "summary": _file_ref(ctx["summary_path"]),
                    "raw": _file_ref(ctx["audit"].raw_jsonl),
                    "task_class": "memory", "protocol_compliant": True,
                    "measured_attempts": ctx["audit"].measured_attempts,
                    "measured_successes": ctx["audit"].measured_successes},
        "gate_drafts": {}, "diagnostics": {}, "blockers": [],
    }
    conditions = ctx["summary"]["conditions"]
    runtime = conditions.get("runtime_versions", {})
    if (runtime.get("vllm_omni_imported_source_sha256") != imported_omni_source_sha256() or
        runtime.get("agent_runtime_identity_sha256") != loaded_runtime_sha256()):
        report["blockers"].append({
            "gate": "profile", "code": "profile_runtime_stale",
            "detail": "profile source/dependency digest differs from the currently imported runtime; rerun the complete profile",
        })
    inputs = {
        "memory_admission": (memory_probe, lambda path: memory_observations(path, ctx)),
        "cancel_recovery": (cancel_record, lambda path: cancel_observations(path, ctx)),
        "checkpoint_lineage": (lineage_probe, lambda path: lineage_observations(
            path, ctx, manifest_path=download_manifest, source_path=source_metadata)),
        "runtime_placement": (placement_snapshot, lambda path: placement_observations(
            path, ctx, cancel_record)),
        "reference_quality": (quality_index, lambda path: quality_observations(path, ctx)),
    }
    required_companions = {
        "checkpoint_lineage": (download_manifest, source_metadata),
        "runtime_placement": (cancel_record,),
    }
    for gate in sorted(REQUIRED_GATES):
        path, extract = inputs[gate]
        if path is None or any(item is None for item in required_companions.get(gate, ())):
            report["blockers"].append({
                "gate": gate, "code": "source_missing",
                "detail": "raw source or required companion path was not supplied",
            })
            continue
        try:
            path = path.resolve(strict=True)
            observations = extract(path)
            gaps = source_binding_gaps(gate, path, ctx)
            if gaps or gate in _DIAGNOSTIC_ONLY:
                diagnostic = {
                    "record_type": "unsigned_gate_diagnostic_v1",
                    "gate": gate, "comparison_target_identity": identity,
                    "observations": observations, "source": _file_ref(path),
                    "not_gate_evidence": True,
                    "missing_or_mismatched_profile_bindings": gaps,
                    "measurement_limitation": _DIAGNOSTIC_ONLY[gate][1]
                    if gate in _DIAGNOSTIC_ONLY else None,
                }
                diagnostic_path = out / f"{gate}.diagnostic.json"
                _write_json(diagnostic_path, diagnostic)
                report["diagnostics"][gate] = {
                    "path": str(diagnostic_path), "sha256": _sha(diagnostic_path),
                    "source": _file_ref(path),
                }
                if gaps:
                    report["blockers"].append({
                        "gate": gate, "code": "profile_identity_unbound",
                        "detail": "raw run does not independently record exact profile bindings: " +
                                  ", ".join(gaps),
                    })
                if gate in _DIAGNOSTIC_ONLY:
                    code, detail = _DIAGNOSTIC_ONLY[gate]
                    report["blockers"].append({"gate": gate, "code": code,
                                               "detail": detail})
                continue
            source = {
                "record_type": f"{gate}_evidence_v1", "identity": identity,
                "observations": observations,
                "source": _file_ref(path),
                "review_status": "unsigned_raw_observation",
                "qualification_outcome": None,
            }
            draft = out / f"{gate}.raw.json"
            _write_json(draft, source)
            report["gate_drafts"][gate] = {"path": str(draft),
                                            "sha256": _sha(draft),
                                            "source": _file_ref(path)}
        except EvidenceBlocked as exc:
            report["blockers"].append({"gate": gate, "code": exc.code,
                                       "detail": exc.detail})
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
            report["blockers"].append({
                "gate": gate, "code": "source_invalid",
                "detail": f"{type(exc).__name__}: raw source did not verify",
            })
    # These are real limitations of the current evidence even if every draft
    # can be transcribed. A signature must come from a separate human review.
    report["blockers"].append({
        "gate": "review", "code": "independent_signed_review_missing",
        "detail": "raw observations require independent inspection and a trusted Ed25519 review; this assembler cannot promote a route",
    })
    report["all_raw_sources_present"] = set(report["gate_drafts"]) == REQUIRED_GATES
    report_path = out / "report.json"
    _write_json(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-index", type=Path, required=True)
    parser.add_argument("--memory-probe", type=Path)
    parser.add_argument("--cancel-record", type=Path)
    parser.add_argument("--lineage-probe", type=Path)
    parser.add_argument("--download-manifest", type=Path)
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--placement-snapshot", type=Path)
    parser.add_argument("--quality-index", type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.environ.get("LOCALAPPDATA", "")) /
                        "OmniEdgeAgent" / "gate-assemblies")
    args = parser.parse_args()
    if not args.output_root.is_absolute():
        parser.error("output root must be an absolute private directory")
    result = assemble(
        profile_index=args.profile_index, output_root=args.output_root,
        memory_probe=args.memory_probe, cancel_record=args.cancel_record,
        lineage_probe=args.lineage_probe, download_manifest=args.download_manifest,
        source_metadata=args.source_metadata,
        placement_snapshot=args.placement_snapshot, quality_index=args.quality_index,
    )
    report = _json(result)
    print(json.dumps({"private_report": str(result),
                      "drafts": sorted(report["gate_drafts"]),
                      "blocker_codes": [item["code"] for item in report["blockers"]],
                      "router_qualification_created": False}, sort_keys=True))


if __name__ == "__main__":
    main()
