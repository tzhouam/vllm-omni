# SPDX-License-Identifier: Apache-2.0
"""Bind optional Strata observation metadata without granting qualification.

These checks join an Agent route to the backend's separately verified runtime
and scoped reports. They do not replace native ownership/module verification,
prove physical SSD traffic, or establish whole-model placement.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()


def _member(value, files):
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or ":" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or value not in files
    ):
        raise ValueError("execution observation metadata is absent from its runtime manifest")
    return files[value].sha256


def execution_route_binding(config, runtime, tier):
    """Metadata only; Stage verifies all actual bytes before loading a model."""
    settings = config.get("execution_observation")
    if settings is None:
        return None
    if (
        not isinstance(settings, Mapping)
        or set(settings)
        != {
            "schema",
            "descriptor_file",
            "static_identity_sha256",
            "source_context_file",
            "receipt_root",
            "workspace_bytes",
            "receipt_storage_bytes",
        }
        or settings["schema"] != "omni-strata-execution-observation-config-v1"
        or config.get("observation_runtime") is not None
        or config.get("observation_runtime_identity_sha256") is not None
        or type(settings["static_identity_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", settings["static_identity_sha256"]) is None
        or type(settings["receipt_root"]) is not str
        or not settings["receipt_root"]
    ):
        raise ValueError("combined execution observation needs its separate exact route identity")
    for key, minimum, maximum in (
        ("workspace_bytes", 512 << 20, 2 << 30),
        ("receipt_storage_bytes", 16 << 20, 64 << 20),
    ):
        if type(settings[key]) is not int or not minimum <= settings[key] <= maximum:
            raise ValueError("execution observer budget differs from the Stage contract")
    if (
        tier.budget.host_workspace_bytes < settings["workspace_bytes"]
        or tier.budget.ssd_temporary_bytes < settings["receipt_storage_bytes"]
    ):
        raise ValueError("execution observer is missing its independent resource allowance")
    files = {item.path: item for item in runtime.files}
    provenance = config.get("runtime_provenance")
    member_manifest = provenance.get("combined_member_manifest") if isinstance(provenance, Mapping) else None
    if (
        not isinstance(member_manifest, Mapping)
        or set(member_manifest) != {"file", "sha256"}
        or _member(member_manifest["file"], files) != member_manifest["sha256"]
    ):
        raise ValueError("combined route must bind its distinct raw member manifest")
    from vllm_omni.engine.backends import strata_io

    return {
        "schema": "omni-strata-execution-route-binding-v1",
        "static_identity_sha256": settings["static_identity_sha256"],
        "descriptor_sha256": _member(settings["descriptor_file"], files),
        "source_context_sha256": _member(settings["source_context_file"], files),
        "member_manifest_file": member_manifest["file"],
        "member_manifest_sha256": member_manifest["sha256"],
        "adapter_sha256": {
            name: _member("adapter/" + name + ".py", files)
            for name in ("strata_exec", "strata_exec_bridge", "strata_exec_runtime", "strata_exec_live")
        },
        "io_adapter_sha256": strata_io.adapter_source_sha256(),
        "workspace_bytes": settings["workspace_bytes"],
        "receipt_storage_bytes": settings["receipt_storage_bytes"],
    }


def validate_execution_plan(plan, binding):
    from vllm_omni.engine.backends import strata_io

    execution = binding["execution_observation"]
    static = plan.get("observation_runtime")
    # This validates the exact static-v2 shape/lineage and its self-hash. Actual
    # byte/build/OS verification remains the backend's responsibility.
    strata_io.validate_combined_io_identity(static)
    if (
        static["identity_sha256"] != execution["static_identity_sha256"]
        or static["runtime_manifest_sha256"] != execution["member_manifest_sha256"]
        or static["native_executable_sha256"] != binding["engine_sha256"]
        or static["descriptor_sha256"] != execution["descriptor_sha256"]
        or static["parser_sha256"] != execution["adapter_sha256"]["strata_exec"]
        or plan["route_controls"].get("observation_runtime") != static
    ):
        raise ValueError("combined static identity differs from the registered Agent route")
    expected = {
        "schema": "omni-strata-execution-observation-config-v1",
        "static_identity_sha256": execution["static_identity_sha256"],
        "adapter_sha256": execution["adapter_sha256"],
        "workspace_bytes": execution["workspace_bytes"],
        "receipt_storage_bytes": execution["receipt_storage_bytes"],
        "workspace_scope": "separate_declared_observer_allowance_not_a_measured_heap_hard_cap",
        "Agent_consumer_workspace_reused": False,
    }
    if plan["route_controls"].get("execution_observation") != expected:
        raise ValueError("execution observer controls or independent budgets changed")


def validate_combined_io_report(report, plan, binding):
    from vllm_omni.engine.backends import strata_io

    expected = strata_io.validate_combined_io_identity(plan["observation_runtime"]) | {
        "io_adapter_sha256": binding["execution_observation"]["io_adapter_sha256"]
    }
    if (
        not isinstance(report, Mapping)
        or report.get("schema") != "omni-strata-combined-request-io-observation-v1"
        or report.get("runtime_identity_schema") != "omni-strata-combined-static-runtime-identity-v2"
        or report.get("combined_io_binding") != expected
    ):
        raise ValueError("combined I/O report has a different runtime or adapter identity")


def validate_execution_report(report, stage, plan, binding, io_report):
    """Validate the reported scope/identity; this is no new device attestation."""
    execution = binding["execution_observation"]
    if (
        not isinstance(report, Mapping)
        or report.get("schema") != "omni-private-strata-owned-execution-report-v1"
        or report.get("status") != "complete_scoped_observation"
        or report.get("complete") is not True
        or report.get("channel_errors") != []
        or report.get("engine_abi_binding_established") is not True
        or _digest(report.get("request_binding"))
        != _digest({"request_id": stage["request_id"], "epoch": stage["epoch"]})
        or report.get("static_runtime_identity_sha256") != execution["static_identity_sha256"]
        or report.get("lifecycle_outcome") != "normal"
        or report.get("reader_healthy") is not True
        or report.get("omni_completed") is not True
        or report.get("cancellation_requested") is not False
        or type(report.get("pending_dispatch_count_at_finish")) is not int
        or report["pending_dispatch_count_at_finish"] != 0
    ):
        raise ValueError("execution observation is incomplete or belongs to another request")
    native = report.get("native_observation")
    if (
        not isinstance(native, Mapping)
        or native.get("schema") != "omni-private-strata-execution-observation-v1"
        or native.get("status") != "complete_scoped_observation"
        or native.get("complete") is not True
        or native.get("reasons") != []
        or native.get("request_id") != stage["request_id"]
        or type(native.get("epoch")) is not int
        or native["epoch"] != stage["epoch"]
        or native.get("omni_completed") is not True
        or native.get("cancellation_requested") is not False
        or native.get("binding_verified_by_parser") is not False
    ):
        raise ValueError("native execution observation identity or completion differs")
    from vllm_omni.engine.backends.strata_execution import strata_exec as parser

    owner = parser._owner(native.get("owner_binding"))
    runtime = parser._runtime(native.get("runtime_binding"))
    observed = plan.get("gpu_observer_identity")
    if (
        not isinstance(owner, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(observed, Mapping)
        or observed.get("status") != "verified"
        or any(
            owner.get(key) != observed.get(key)
            for key in ("worker_generation", "pid", "creation_filetime_100ns", "gpu")
        )
        or owner.get("stage_id") != stage["stage_id"]
        or owner.get("worker_generation") != stage["worker_generation"]
        or native.get("owner_binding_sha256") != _digest(owner)
        or native.get("runtime_binding_sha256") != _digest(runtime)
        or report.get("runtime_binding_sha256") != _digest(runtime)
        or runtime.get("runtime_manifest_sha256") != execution["member_manifest_sha256"]
        or runtime.get("native_executable_sha256") != binding["engine_sha256"]
        or runtime.get("owner_adapter_sha256") != execution["adapter_sha256"]["strata_exec_bridge"]
    ):
        raise ValueError("execution report differs from the loaded native owner/runtime")
    snapshots = native.get("snapshots")
    if (
        not isinstance(snapshots, list)
        or len(snapshots) != 3
        or native.get("snapshot_sha256") != [_digest(row) for row in snapshots]
    ):
        raise ValueError("execution report lacks its three unchanged native snapshots")
    dispatch, done = native.get("dispatch"), native.get("native_done")
    if (
        not isinstance(dispatch, Mapping)
        or dispatch.get("command") != "GEN"
        or type(dispatch.get("native_request_seq")) is not int
        or dispatch["native_request_seq"] <= 0
        or not isinstance(done, Mapping)
        or type(done.get("native_request_seq")) is not int
        or done.get("native_request_seq") != dispatch["native_request_seq"]
        or done.get("finish_reason") not in {"stop", "length"}
    ):
        raise ValueError("execution report dispatch or actual native DONE differs")
    if (
        not isinstance(io_report, Mapping)
        or io_report.get("status") != "complete"
        or io_report.get("reasons") != []
        or io_report.get("native_terminal") != done["finish_reason"]
        or type(io_report.get("native_pid")) is not int
        or io_report["native_pid"] != owner["pid"]
        or type(io_report.get("creation_filetime_100ns")) is not int
        or io_report["creation_filetime_100ns"] != owner["creation_filetime_100ns"]
        or type(io_report.get("native_request_seq")) is not int
        or io_report["native_request_seq"] != dispatch["native_request_seq"]
    ):
        raise ValueError("combined I/O and execution reports have different native ownership or sequence")
    for index, (snapshot, boundary) in enumerate(
        zip(snapshots, ("request_start", "prefill_end_decode_start", "request_end"))
    ):
        snapshot = parser.validate_snapshot(snapshot)
        if (
            snapshot.get("snapshot_seq") != index
            or snapshot.get("native_request_seq") != dispatch["native_request_seq"]
            or snapshot.get("phase") != boundary
            or snapshot["issues"]
            or snapshot["clock"]["ticks"] is None
            or snapshot["observer"] != runtime["observer_layout"]
        ):
            raise ValueError("execution snapshots are missing, reordered or cross-request")
    final = snapshots[-1]
    if (
        final["request_terminal"] != "completed"
        or final["boundary_complete"] is not True
        or any(final["inflight"].values())
    ):
        raise ValueError("native execution ended with incomplete or in-flight work")
    for value in (report, native):
        for key in (
            "aggregate_gpu_hard_cap_verified",
            "aggregate_ram_hard_cap_verified",
            "runtime_qualification",
            "default_eligible",
        ):
            if value.get(key) is not False:
                raise ValueError("scoped execution observation cannot grant memory or runtime qualification")
        if value.get("physical_ssd_read_bytes") is not None:
            raise ValueError("execution counters do not measure physical SSD traffic")
    if report.get("actual_whole_model_placement") is not None or native.get("whole_model_placement") is not None:
        raise ValueError("scoped execution counters do not establish whole-model placement")
