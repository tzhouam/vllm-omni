# SPDX-License-Identifier: Apache-2.0
"""Publish allowlisted, aggregate native Agent evidence from private raw runs.

Run on the Windows profiling host, where each index's local paths resolve.
The full JSONL, prompts, screens, page observations and tool payloads remain
private. SHA-256 values bind the public report to that local evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from benchmarks.edge_agent.evidence import audit_summary


_LENGTHS = ("short", "medium", "long")
_TASK_CLASSES = frozenset({
    "browser_text", "browser_vision", "windows_settings", "memory", "code_tools",
})
_STATUSES = frozenset({
    "evidence_recorded", "failed_or_blocked", "blocked_missing_vision_projector",
})
_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .,+()_=-]{0,159}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"native_[0-9a-f]{32}\Z")
_FAILURE_TYPES = {
    "AttributeError": "backend_attribute_error",
    "FileExistsError": "fixture_memory_path_exists",
    "OperationalError": "fixture_database_error",
    "TargetClosedError": "browser_target_closed",
    "ValueError": "profile_input_error",
}
_EXACT_FAILURES = {
    "RuntimeError: Edge fixture window could not be verified in foreground":
        "fixture_foreground_unverified",
}
_PRE_FIX_MEMORY_RUNS = frozenset({
    "native_1d0455bb82a74515b70cf0e07805e63c",
    "native_3ea19e21291c44c2a77f78bdc94c0f82",
    "native_37b71069bd36416aaa57d33bbbe49639",
    "native_5cd04bf300bd4616af88a773ef401fbc",
    "native_8dce8147ad6944279879752fbe699233",
    "native_b0c1704b274d448b89e0562f4674f059",
    "native_b2bb58f3c045419b9e5f0b75b1f14e1b",
})
_POST_FIX_MEMORY_RUNS = frozenset({
    "native_e1d90b1e97604ce7ae22aee28c2e950f",
    "native_5c400dad75f44554873c7aaa178261bd",
    "native_a165341a6b4d4843a219473929833cbe",
    "native_95bccd7f0c1948a6aa36ff0f7e1392a4",
    "native_37aad46efd4f4893a497a7bfaa83599d",
    "native_4bba0301e4d04f4f9af9dd45e1c6deea",
})
_PRE_CURRENT_CODE_RUNS = frozenset({
    "native_5c400dad75f44554873c7aaa178261bd",
    "native_a165341a6b4d4843a219473929833cbe",
    "native_37aad46efd4f4893a497a7bfaa83599d",
    "native_4bba0301e4d04f4f9af9dd45e1c6deea",
})
_LOADER_LOG_LINES = tuple(re.compile(pattern) for pattern in (
    r"llama_model_load: using device Vulkan[0-9]+ \([A-Za-z0-9 .,+()_=-]+\)",
    r"load_tensors: layer [0-9]+ assigned to device Vulkan[0-9]+",
    r"tensor blk\.[0-9]+\.ffn_(?:down|gate|up)_exps\.weight \(redacted\) buffer type overridden to Vulkan_Host",
    r"load_tensors: offloaded [0-9]+/[0-9]+ layers to GPU",
    r"load_tensors: (?:CPU|Vulkan[0-9]+) model buffer size = +[0-9]+\.[0-9]+ MiB",
    r"llama\.cpp capacity: (?:CPU|Vulkan[0-9]+|Vulkan_Host) (?:KV|compute) buffer size = +[0-9]+\.[0-9]+ MiB",
    r"clip_ctx: CLIP using Vulkan[0-9]+ backend",
))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _config_snapshot_verified(raw_path: Path, claimed_sha256: str) -> bool:
    snapshot = raw_path.with_suffix(".config.json")
    return snapshot.is_file() and _sha256(snapshot) == claimed_sha256


def _safe_label(value: Any, name: str) -> str:
    """Accept ordinary public model/hardware labels, never paths or URLs."""
    if not isinstance(value, str) or not _SAFE_LABEL.fullmatch(value):
        raise ValueError(f"{name} is not a safe public label")
    return value


def _safe_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} is not a safe public identifier")
    return value


def _sha_label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} is not a SHA-256 value")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} is not a nonnegative integer")
    return value


def _failure_code(error: Any) -> str:
    # Exception messages may contain local paths, prompts or page text.
    # Publish only a known exception class; all other messages stay private.
    if not isinstance(error, str):
        return "unclassified_failure"
    if error in _EXACT_FAILURES:
        return _EXACT_FAILURES[error]
    return _FAILURE_TYPES.get(error.partition(":")[0], "unclassified_failure")


def _audit_codes(errors: tuple[str, ...]) -> list[str]:
    codes: set[str] = set()
    for error in errors:
        if "claimed complete Agent trace is not reconstructible" in error:
            codes.add("claimed_trace_not_reconstructible")
        elif error == "one or more full Agent event traces are unavailable or invalid":
            codes.add("incomplete_agent_trace")
        else:
            codes.add("integrity_mismatch")
    return sorted(codes)


def _public_conditions(index: dict[str, Any]) -> dict[str, Any]:
    conditions = index["conditions"]
    hardware = index["hardware"]
    if (conditions["os_version"] != hardware["os"] or
            conditions["power_condition"] != hardware["power_condition"]):
        raise ValueError("index hardware and profile conditions disagree")
    if conditions["suite_id"] != "edge-agent-fixed-local-fixtures-v1":
        raise ValueError("index does not use the fixed local Agent suite")
    runtime = conditions["runtime_versions"]
    versions = {}
    if "python" in runtime:
        versions["python"] = _safe_label(runtime["python"], "Python runtime")
    if "vllm" in runtime:
        versions["vllm_installed_distribution"] = _safe_label(
            runtime["vllm"], "installed vLLM distribution")
    if "vllm_omni_installed_distribution" in runtime:
        # New indexes distinguish the imported checkout from wheel metadata.
        versions["vllm_omni_installed_distribution"] = _safe_label(
            runtime["vllm_omni_installed_distribution"],
            "installed Omni distribution")
        versions["vllm_omni_loaded_source_version"] = _safe_label(
            runtime["vllm_omni"], "loaded Omni source version")
    else:
        # Historic indexes recorded package metadata even when PYTHONPATH
        # loaded this checkout. It is not an executed-source version.
        if "vllm_omni" in runtime:
            versions["vllm_omni_installed_distribution"] = _safe_label(
                runtime["vllm_omni"], "installed Omni distribution")
        versions["vllm_omni_loaded_source_version"] = None
    if "llama_server_sha256" in runtime:
        versions["llama_server_sha256"] = _sha_label(
            runtime["llama_server_sha256"], "llama-server hash")
    for field in ("vllm_omni_imported_source_sha256",
                  "agent_runtime_identity_sha256"):
        versions[field] = (_sha_label(runtime[field], field)
                           if field in runtime else None)
    drivers = {
        _safe_id(key, "driver name"): _safe_label(value, "driver version")
        for key, value in conditions["driver_versions"].items()
    }
    power = conditions["power_condition"]
    if power not in {"AC", "battery", "unknown"}:
        raise ValueError("unknown power condition")
    return {
        "hardware": {
            "cpu": _safe_label(hardware["cpu"], "CPU"),
            "gpu": _safe_label(hardware["gpu_name"], "GPU")
            if hardware.get("gpu_name") else None,
            "host_ram_total_bytes": _nonnegative_int(
                hardware["host_ram_total_bytes"], "total host RAM"),
            "host_ram_available_bytes_at_start": _nonnegative_int(
                hardware["host_ram_available_bytes"], "available host RAM"),
            "vram_available_bytes_at_start": _nonnegative_int(
                hardware["vram_available_bytes"], "available VRAM")
            if hardware.get("vram_available_bytes") is not None else None,
        },
        "os_version": _safe_label(conditions["os_version"], "OS version"),
        "driver_versions": drivers,
        "runtime_versions": versions,
        "power_condition": power,
        "suite_id": conditions["suite_id"],
        "environment_fingerprint": _sha_label(
            conditions["environment_fingerprint"], "environment fingerprint"),
        "batch_size": 1,
        "concurrency": 1,
    }


def _public_routes(index: dict[str, Any]) -> dict[str, Any]:
    routes: dict[str, Any] = {}
    for route_id, provenance in index["artifact_provenance"].items():
        route_id = _safe_id(route_id, "route ID")
        revision = provenance["checkpoint_revision"]
        routes[route_id] = {
            "checkpoint_revision": _safe_label(revision, "checkpoint revision")
            if provenance["lineage_verified"] else None,
            "lineage_verified": provenance["lineage_verified"] is True,
            "model_sha256": _sha_label(provenance["model_sha256"], "model hash"),
            "mmproj_sha256": _sha_label(provenance["mmproj_sha256"], "projector hash")
            if provenance.get("mmproj_sha256") else None,
            "server_sha256": _sha_label(provenance["server_sha256"], "server hash"),
            "precision": _safe_label(provenance["precision"], "precision"),
        }
    return routes


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("profile timing is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("profile timing is invalid")
    return number


def _length_stats(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        bucket: {
            "measured_requests": len(profile["answer_latency_s"][bucket]),
            "first_token_samples": len(profile["ttft_s"][bucket]),
            "warmup_requests": profile["warmups_per_length"][bucket],
            "answer_p50_s": _number(profile["answer_p50_s"][bucket]),
            "answer_p95_s": _number(profile["answer_p95_s"][bucket]),
            "ttft_p50_s": _number(profile["ttft_p50_s"][bucket]),
            "ttft_p95_s": _number(profile["ttft_p95_s"][bucket]),
        }
        for bucket in _LENGTHS
    }


def _input_shapes(raw_path: Path) -> dict[str, Any]:
    """Derive lengths from raw prompts without copying any prompt or payload."""
    shapes: dict[str, set[tuple[int, int]]] = {bucket: set() for bucket in _LENGTHS}
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") != "request" or row.get("phase") != "measured":
                continue
            case = row["case"]
            prompt = case["prompt"]
            shapes[case["length"]].add((len(prompt), len(prompt.encode("utf-8"))))
    return {
        bucket: {
            "prompt_chars_min_max": [min(item[0] for item in values),
                                     max(item[0] for item in values)],
            "prompt_utf8_bytes_min_max": [min(item[1] for item in values),
                                          max(item[1] for item in values)],
        } if values else None
        for bucket, values in shapes.items()
    }


def _vision_kind_counts(raw_path: Path) -> dict[str, Any]:
    """Separate desktop capture from browser image reading in the fixed suite."""
    counts = {kind: {"attempts": 0, "raw_recomputed_successes": 0}
              for kind in ("desktop_screen", "screen_vision")}
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") != "request" or row.get("phase") != "measured":
                continue
            kind = row["case"]["metadata"]["kind"]
            if kind not in counts:
                raise ValueError("unexpected browser vision case kind")
            counts[kind]["attempts"] += 1
            counts[kind]["raw_recomputed_successes"] += bool(
                row.get("e2e_complete") and row.get("placement_matches")
                and (row.get("evaluation") or {}).get("success"))
    return counts


def _telemetry_peaks(raw_path: Path) -> dict[str, Any]:
    """Summarize request samples; loading and sub-interval spikes are excluded."""
    fields = ("ram_used_bytes_peak", "vram_used_bytes_peak", "gpu_power_w_peak",
              "system_power_w_peak", "gpu_temp_c_peak")
    maxima: dict[str, float | None] = {field: None for field in fields}
    sample_count = 0
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (row.get("record_type") != "request" or
                    row.get("phase") not in {"measured", "endurance"}):
                continue
            summary = row["telemetry"]["summary"]
            sample_count += int(summary["sample_count"])
            for field in fields:
                if field in summary:
                    value = _number(summary[field])
                    if value is not None:
                        maxima[field] = value if maxima[field] is None else max(
                            maxima[field], value)
    return {
        "sample_count": sample_count,
        "peaks": maxima,
        "peak_is_sampled_lower_bound": True,
        "scope": "measured_and_endurance_requests_only_excludes_cold_load",
    }


def _index_matches_summary(index: dict[str, Any], row: dict[str, Any],
                           summary: dict[str, Any], raw_sha256: str,
                           protocol_compliant: bool) -> bool:
    route_id = row["route_id"]
    profile = summary.get("routes", {}).get(route_id, {})
    route = profile.get("route", {})
    provenance = index["artifact_provenance"][route_id]
    return (
        summary.get("conditions") == index["conditions"]
        and summary.get("raw_sha256") == raw_sha256
        and row.get("raw_sha256") == raw_sha256
        and row.get("protocol_compliant") is protocol_compliant
        and row.get("release_qualified") is False
        and profile.get("task_class") == row["task_class"]
        and route.get("route_id") == route_id
        and route.get("checkpoint_revision") == provenance["checkpoint_revision"]
        and route.get("artifact_sha256") == provenance["model_sha256"]
        and route.get("precision") == provenance["precision"]
    )


def summarize_navigation_retest(path: Path) -> dict[str, Any]:
    """Publish only the checked fields of a one-request functional retest."""
    path = path.resolve(strict=True)
    retest = json.loads(path.read_text(encoding="utf-8"))
    if retest["scope"] != "one_request_navigation_fix_smoke_unqualified":
        raise ValueError("unexpected supplemental smoke scope")
    sequence = retest["tool_sequence"]
    if sequence == ["browser_open", "browser_read"]:
        task_class = "browser_text"
        auto_observation = retest.get("navigation_auto_read_recorded") is True
    elif sequence == ["browser_open", "browser_screenshot"]:
        task_class = "browser_vision"
        auto_observation = retest.get("navigation_auto_observation_recorded") is True
    else:
        raise ValueError("unexpected supplemental smoke tool sequence")
    if retest.get("task_class", task_class) != task_class:
        raise ValueError("supplemental smoke task class differs from tools")
    return {
        "retest_id": _safe_id(path.stem, "navigation retest ID"),
        "scope": retest["scope"],
        "task_class": task_class,
        "evidence_sha256": _sha256(path),
        "config_sha256": _sha_label(retest["config_sha256"], "config hash"),
        "case_prompt_sha256": _sha_label(
            retest["case_prompt_sha256"], "case prompt hash"),
        "duration_s_including_cold_load": _number(retest["duration_s"]),
        "final_answer_present": retest["final_answer_present"] is True,
        "reference_matched": retest["reference_matched"] is True,
        "tool_sequence": sequence,
        "navigation_auto_observation_recorded": auto_observation,
        "seq_ordered": retest["seq_ordered"] is True,
        "release_qualified": False,
    }


def summarize_load_refusal(path: Path) -> dict[str, Any]:
    """Bind a failed one-request load without exporting model paths or prompts."""
    path = path.resolve(strict=True)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if (len(rows) != 3 or rows[0].get("record_type") != "manifest" or
            rows[0].get("scope") != "one_complete_agent_request_smoke" or
            rows[0].get("batch_size") != 1 or rows[0].get("concurrency") != 1 or
            [row.get("kind") for row in rows[1:]] != ["user_observation", "error"]):
        raise ValueError("load-refusal smoke has an unexpected record sequence")
    error = rows[2].get("payload", {})
    if (error.get("type") == "ResourceUnavailable" and
            error.get("message") ==
            "actual hybrid model buffers exceed the declared per-pool weight budgets"):
        code = "declared_weight_budget_exceeded"
    elif (error.get("type") == "RuntimeError" and
          error.get("message") ==
          "REFUSE_DEVICE_PLACEMENT: CPU-expert tensor override was not verified"):
        code = "cpu_experts_placement_unverified"
    else:
        code = "unclassified_load_refusal"
    return {
        "smoke_id": _safe_id(path.stem, "load smoke ID"),
        "scope": "one_complete_agent_request_smoke",
        "raw_sha256": _sha256(path),
        "config_sha256": _sha_label(rows[0]["config_sha256"], "config hash"),
        "config_snapshot_verified": _config_snapshot_verified(
            path, rows[0]["config_sha256"]),
        "batch_size": 1,
        "concurrency": 1,
        "failure_code": code,
        "completed_model_request": False,
        "release_qualified": False,
    }


def summarize_qwen_host_mapped_smoke(path: Path) -> dict[str, Any]:
    """Bind a single successful, experimental Qwen request without its prompt."""
    path = path.resolve(strict=True)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if (len(rows) != 6 or rows[0].get("record_type") != "manifest" or
            rows[0].get("scope") != "one_complete_agent_request_smoke" or
            rows[0].get("batch_size") != 1 or rows[0].get("concurrency") != 1 or
            [row.get("kind") for row in rows[1:]] != [
                "user_observation", "route", "text_delta", "model_metrics", "final"] or
            [row.get("seq") for row in rows[1:]] != list(range(1, 6))):
        raise ValueError("host-mapped smoke has an unexpected event sequence")
    route = rows[2]["payload"]
    final = rows[5]["payload"]
    if (route.get("actual_placement") != "Vulkan_Host+Vulkan0" or
            route.get("experimental") is not True or
            final.get("experimental") is not True or
            final.get("answer") != "ready" or
            final.get("streamed") is not True):
        raise ValueError("host-mapped smoke lacks the expected narrow outcome")
    return {
        "smoke_id": _safe_id(path.stem, "Qwen smoke ID"),
        "scope": "one_complete_agent_request_smoke",
        "raw_sha256": _sha256(path),
        "config_sha256": _sha_label(rows[0]["config_sha256"], "config hash"),
        "config_snapshot_verified": _config_snapshot_verified(
            path, rows[0]["config_sha256"]),
        "batch_size": 1,
        "concurrency": 1,
        "route_id": _safe_id(route["route_id"], "route ID"),
        "model": _safe_label(route["model"], "model"),
        "backend": _safe_label(route["backend"], "backend"),
        "reported_placement": "Vulkan_Host+Vulkan0",
        "ordered_streamed_final": True,
        "final_answer_exact_ready": True,
        "completed_model_request": True,
        "expert_storage_and_compute_verified": False,
        "release_qualified": False,
    }


def summarize_loader_log(path: Path) -> dict[str, Any]:
    """Bind a reviewed, allowlisted loader log and its buffer measurements."""
    path = path.resolve(strict=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not any(pattern.fullmatch(line) for pattern in _LOADER_LOG_LINES)
                        for line in lines):
        raise ValueError("loader log contains non-allowlisted content")
    cpu = [re.fullmatch(r"load_tensors: CPU model buffer size = +([0-9]+\.[0-9]+) MiB", line)
           for line in lines]
    vulkan = [re.fullmatch(r"load_tensors: Vulkan0 model buffer size = +([0-9]+\.[0-9]+) MiB", line)
              for line in lines]
    cpu_values = [float(match.group(1)) for match in cpu if match]
    vulkan_values = [float(match.group(1)) for match in vulkan if match]
    if len(cpu_values) != 1 or len(vulkan_values) != 1:
        raise ValueError("loader log lacks unique CPU and Vulkan0 buffers")
    host_mapped_overrides = sum(
        line.startswith("tensor blk.") and line.endswith(
            "buffer type overridden to Vulkan_Host") for line in lines)
    return {
        "filename": _safe_id(path.name, "public loader log name"),
        "sha256": _sha256(path),
        "allowlisted_line_count": len(lines),
        "cpu_model_buffer_mib": cpu_values[0],
        "vulkan0_model_buffer_mib": vulkan_values[0],
        "host_mapped_expert_override_lines": host_mapped_overrides,
        "scope": "loader_buffer_and_placement_observation",
    }


def summarize_index(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not _RUN_ID.fullmatch(path.parent.name):
        raise ValueError("index must be in a native run directory")
    index = json.loads(path.read_text(encoding="utf-8"))
    if (index["scope"] != "whole_agent_batch1_paired_local_fixture" or
            index["protocol"] not in {"smoke_incomplete", "full_20x3_and_30m"}):
        raise ValueError("index has an unknown Agent profile scope or protocol")
    public: dict[str, Any] = {
        "run_id": path.parent.name,
        "index_sha256": _sha256(path),
        "source_config_sha256": _sha_label(index["source_config_sha256"], "config hash"),
        "lineage_manifest_sha256": _sha_label(index["lineage_sha256"], "lineage hash"),
        "scope": index["scope"],
        "protocol": index["protocol"],
        "conditions": _public_conditions(index),
        "routes": _public_routes(index),
        "results": [],
    }
    for row in index["results"]:
        task = row["task_class"]
        route_id = row["route_id"]
        status = row["status"]
        if task not in _TASK_CLASSES or route_id not in public["routes"] or status not in _STATUSES:
            raise ValueError("index contains an unknown task, route, or status")
        result: dict[str, Any] = {
            "task_class": task,
            "route_id": route_id,
            "status": status,
            "release_qualified": False,
        }
        if path.parent.name in _PRE_CURRENT_CODE_RUNS:
            result["current_release_code_status"] = (
            "profiled_before_current_imported_omni_source")
        if task == "memory":
            # A prior fixture seeded an event kind that retrieval did not
            # search. Corrected runs remain scoped to the fixed fixture.
            if path.parent.name in _PRE_FIX_MEMORY_RUNS:
                result["memory_evidence_scope"] = "invalid_pre_fix_seed_kind"
            elif path.parent.name in _POST_FIX_MEMORY_RUNS:
                result["memory_evidence_scope"] = (
                    "corrected_seed_fixed_fixture_profile"
                    if index["protocol"] == "full_20x3_and_30m"
                    else "corrected_seed_fixed_fixture_smoke")
            else:
                result["memory_evidence_scope"] = "unreviewed_memory_fixture"
        if status != "evidence_recorded":
            result["failure_code"] = (_failure_code(row.get("error"))
                                      if status == "failed_or_blocked"
                                      else "missing_vision_projector")
            public["results"].append(result)
            continue
        try:
            summary_path = Path(row["summary"]).resolve(strict=True)
            if not summary_path.is_relative_to(path.parent / "raw"):
                raise ValueError("summary is outside the private run directory")
            audit = audit_summary(summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            consistent = _index_matches_summary(
                index, row, summary, audit.raw_sha256, audit.protocol_compliant)
            codes = _audit_codes(audit.errors)
            if not consistent:
                codes.append("index_summary_mismatch")
            result.update({
                "summary_sha256": _sha256(summary_path),
                "raw_sha256": audit.raw_sha256,
                "raw_internally_valid": audit.internally_valid and consistent,
                "trace_verified": audit.trace_verified,
                "protocol_compliant": audit.protocol_compliant,
                "audit_error_codes": sorted(set(codes)),
            })
            if (consistent and set(codes) <= {
                    "claimed_trace_not_reconstructible", "incomplete_agent_trace"}):
                # These are reconstructed raw evaluation counts, but the
                # missing full traces prevent a qualified task claim.
                result["raw_recomputed_measured_attempts"] = audit.measured_attempts
                result["raw_recomputed_measured_successes"] = audit.measured_successes
                if task == "browser_vision":
                    result["raw_recomputed_vision_kinds"] = _vision_kind_counts(
                        audit.raw_jsonl)
            if audit.internally_valid and consistent and audit.trace_verified:
                profile = summary["routes"][route_id]
                result.update({
                    "backend": _safe_label(profile["route"]["backend"], "backend"),
                    "placement": _safe_label(
                        profile["route"]["expected_placement"], "placement"),
                    "measured_successes": profile["measured_successes"],
                    "measured_attempts": profile["measured_attempts"],
                    "lengths": _length_stats(profile),
                    "input_shapes": _input_shapes(audit.raw_jsonl),
                    "telemetry": _telemetry_peaks(audit.raw_jsonl),
                    "telemetry_interval_seconds": _number(
                        row["telemetry_interval_seconds"]),
                    "cold_start_s": _number(profile["cold_start_s"]),
                    "endurance_requests": profile["endurance_requests"],
                    "sustained_seconds": _number(profile["sustained_seconds"]),
                    "endurance_wall_seconds": _number(
                        profile["endurance_wall_seconds"]),
                    "correctness_pass": profile["correctness_pass"] is True,
                    "tool_safety_pass": profile["tool_safety_pass"] is True,
                    "stability_pass": profile["stability_pass"] is True,
                    "e2e_trace_pass": profile["e2e_trace_pass"] is True,
                    "actual_placement_matches": (
                        profile["actual_placement_matches"] is True),
                    "placement_evidence_present": (
                        profile["placement_evidence_present"] is True),
                    "telemetry_present": profile["telemetry_present"] is True,
                })
        except Exception:
            # Do not serialize the exception: it may contain a private path or
            # the raw content that caused the parser to fail.
            result.update({
                "raw_internally_valid": False,
                "trace_verified": False,
                "protocol_compliant": False,
                "audit_error_codes": ["audit_exception"],
            })
        public["results"].append(result)
    return public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--navigation-retest", type=Path, action="append")
    parser.add_argument("--load-refusal", type=Path, action="append")
    parser.add_argument("--completed-qwen-smoke", type=Path, action="append")
    parser.add_argument("--loader-log", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indexes = [path.resolve(strict=True) for path in args.index]
    if len(indexes) != len(set(indexes)):
        parser.error("duplicate index")
    report = {
        "schema": "omni-agent-public-aggregate-v2",
        "raw_evidence_location": "private on the profiling host; SHA-256 bound below",
        "latency_statistic": "nearest-rank within each recorded input length",
        "qualification_minimums": {
            "batch_size": 1,
            "concurrency": 1,
            "measured_requests_per_length": 20,
            "active_endurance_seconds": 1800,
            "independent_gate_review_required": True,
        },
        "runs": [summarize_index(path) for path in indexes],
    }
    if args.navigation_retest:
        report["supplemental_smokes"] = [
            summarize_navigation_retest(path) for path in args.navigation_retest]
    if args.load_refusal:
        report["load_refusals"] = [
            summarize_load_refusal(path) for path in args.load_refusal]
    if args.completed_qwen_smoke:
        report["completed_qwen_smokes"] = [
            summarize_qwen_host_mapped_smoke(path)
            for path in args.completed_qwen_smoke]
    if args.loader_log:
        report["loader_logs"] = [
            summarize_loader_log(path) for path in args.loader_log]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        output.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
