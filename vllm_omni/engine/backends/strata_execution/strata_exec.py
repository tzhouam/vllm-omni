"""Private, untested parser for a reviewed future Strata execution observer.

Parsing native text does not establish ownership or runtime provenance. The
caller must independently verify the combined runtime and native owner, then
construct its protected owner envelope. No production adapter uses this module.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
import threading
from collections.abc import Mapping
from typing import Any

PREFIX = "OMNI_EXEC_V1 "
NATIVE_SCHEMA = "strata-omni-exec-v1"
HEADER_V1_SHA256 = "1aab06f18a6c3068ae3de389acc3f32347092bc3dd18c415343cb8b0e8a8d81b"
HEADER_SHA256 = "148b2d3de4b3f62e8f5ebff9211d04a89d488bd00d9256cc26ef41cc5bfa2f55"
BOUNDARY_PATCH_SHA256 = "4689571b0c15873dba9e232baf3834db925c1cfb6d9f5929ef70f95c773ab583"
COMBINED_PATCH_MANIFEST_SHA256 = "b6a0546fc281491fb6171c5e201c46dad714bf7183f3b0ccbd96b872968ee98f"
SCHEMA_SHA256 = "d7ff209a7533f85e182e2be9954805c1e08d505e253a7e116c405542ea729cfc"
INCREMENTAL_PATCH_SHA256 = "884ce375cfab94756834f1e97211bfbef6471674e50e24c0bfcc03a9eca96358"
BASE_IO_PATCH_SHA256 = "31b98e5e02e21a6289d0713097036334a59aef2d15ad9e951ded80f59f4bfdcc"
FRAME_SCHEMA = "omni-private-strata-owned-exec-frame-v1"
RUNTIME_SCHEMA = "omni-private-strata-exec-runtime-binding-v1"
OWNER_SCHEMA = "omni-private-strata-exec-owner-v1"
REPORT_SCHEMA = "omni-private-strata-execution-observation-v1"
NATIVE_FRAME_BYTES = 32768  # C++ buffer, including prefix/LF/NUL.
OWNED_FRAME_BYTES = NATIVE_FRAME_BYTES + 4096
REPORT_BYTES = 262144  # Encoded detached-report policy; not a Python heap cap.
U64_MAX = (1 << 64) - 1
NATIVE_SEQUENCE_MAX = (1 << 62) - 1
ISSUE_MASK = (1 << 14) - 1
CLOCK_UNAVAILABLE = 8192
UNTRACKED_CACHE_MASK = 3
BOUNDARIES = ("request_start", "prefill_end_decode_start", "request_end")
PHASES = ("outside_request", "prefill", "decode")
CPU_FAMILIES = ("full", "split", "split_multi", "split_multi_native")
CUDA_FAMILIES = ("session_run_token", "Verifier_run")
CPU_COUNTS = (
    "submitted", "completed", "unique_jobs_submitted", "unique_jobs_completed",
    "token_expert_entries_submitted", "token_expert_entries_completed",
)
CPU_PHASE_COUNTS = ("submitted", "completed", "row_tasks_submitted", "row_tasks_completed")
CUDA_COUNTS = (
    "launch_attempts", "submitted", "completed_after_successful_existing_stream_fence",
    "launch_errors", "fence_errors",
)
MEMORY_COUNTS = (
    "allocation_attempts", "successful_allocations", "allocation_errors", "free_attempts",
    "successful_frees", "failed_frees",
)
INSTRUMENTED = (
    "CPU_pool_methods_and_row_phase_barriers", "session_run_token_stream_fence",
    "Verifier_run_stream_fence", "ordinary_ExpertCache_cudaMalloc_cudaFree",
)
EXCLUDED = (
    "VMM_segmented_cache_and_KV_handle_transfers", "prefill_direct_CUDA_cuBLAS",
    "other_session_graphs", "Verifier_commit_slots_pipeline_graphs",
    "embedding_head_PLE_direct_kernels", "graph_node_inventory",
    "CPU_activation_quantization_separate_accounting", "all_other_allocators_aliases_library_driver",
)
OVERHEAD_SCOPE = (
    "fixed_Cpp_state_and_one_owner_frame_plus_per_call_stack_ticket_excludes_CRT_stdio"
)
OBSERVER_KEYS = {
    "state_bytes", "frame_buffer_bytes", "frame_state_bytes", "ticket_bytes",
    "concurrent_ticket_count_verified", "root_registry_capacity", "overhead_scope",
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
_PCI = re.compile(r"[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\Z")


class ObservationError(ValueError):
    """A content-free protocol failure; code never contains native input text."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise ObservationError(code)


def _object(value: Any, keys: set[str], code: str = "object_shape") -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        _fail(code)
    return value


def _literal(value: Any, expected: Any, code: str = "literal") -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(code)


def _uint(value: Any, *, positive: bool = False, maximum: int = U64_MAX) -> int:
    if type(value) is not int or value < (1 if positive else 0) or value > maximum:
        _fail("unsigned_integer")
    return value


def _status(value: Any) -> None:
    # optional_status() maps only -1 to null; preserve other original int values.
    if value is not None and (type(value) is not int or value == -1 or not -(1 << 31) <= value < (1 << 31)):
        _fail("cuda_status")


def _identifier(value: Any) -> str:
    if type(value) is not str or not 1 <= len(value) <= 128 or not value.isascii():
        _fail("identifier")
    if any(ord(char) < 33 or ord(char) > 126 for char in value):
        _fail("identifier")
    return value


def _hash(value: Any) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        _fail("sha256")
    return value


def _encoded(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, RecursionError):
        _fail("json_value")


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _json(data: bytes | str, maximum: int) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail("duplicate_json_key")
            result[key] = value
        return result

    def no_float(value: str) -> None:
        del value
        _fail("non_integer_json_number")

    try:
        raw = data.encode("utf-8", errors="strict") if type(data) is str else data
        if type(raw) is not bytes or not raw or len(raw) > maximum:
            _fail("json_size")
        return json.loads(raw.decode("ascii", errors="strict"), object_pairs_hook=pairs,
                          parse_float=no_float, parse_constant=no_float)
    except ObservationError:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError):
        _fail("invalid_json")


def _observer(value: Any) -> dict[str, Any]:
    record = _object(value, OBSERVER_KEYS, "observer_shape")
    _uint(record["state_bytes"], positive=True, maximum=16384)
    _literal(record["frame_buffer_bytes"], NATIVE_FRAME_BYTES)
    _uint(record["frame_state_bytes"], positive=True)
    if record["frame_state_bytes"] < NATIVE_FRAME_BYTES:
        _fail("frame_state_bound")
    _uint(record["ticket_bytes"], positive=True)
    _literal(record["concurrent_ticket_count_verified"], False)
    _literal(record["root_registry_capacity"], 16)
    _literal(record["overhead_scope"], OVERHEAD_SCOPE)
    return record


def validate_snapshot(value: Any) -> dict[str, Any]:
    """Validate an untrusted snapshot, without authorizing it as owned evidence."""
    record = _object(value, {
        "schema", "native_request_seq", "snapshot_seq", "phase", "clock", "counter_scope",
        "snapshot_coherent", "scope_status", "request_terminal", "boundary_complete", "issues",
        "observer", "inflight", "counters", "memory", "coverage", "whole_model_placement",
        "physical_ssd_read_bytes", "aggregate_gpu_hard_cap_verified", "aggregate_ram_hard_cap_verified",
    }, "native_shape")
    if len(_encoded(record)) + len(PREFIX) + 2 > NATIVE_FRAME_BYTES:
        _fail("native_size")
    _literal(record["schema"], NATIVE_SCHEMA)
    _uint(record["native_request_seq"], positive=True, maximum=NATIVE_SEQUENCE_MAX)
    index = _uint(record["snapshot_seq"], maximum=2)
    _literal(record["phase"], BOUNDARIES[index])
    _literal(record["counter_scope"], "process_lifetime_cumulative_phase_at_submission")
    _literal(record["snapshot_coherent"], False)
    _literal(record["scope_status"], "partial_coverage")
    if index < 2:
        _literal(record["request_terminal"], None)
        _literal(record["boundary_complete"], False)
    else:
        if type(record["request_terminal"]) is not str or record["request_terminal"] not in ("completed", "cancelled"):
            _fail("native_terminal")
        if type(record["boundary_complete"]) is not bool:
            _fail("native_boundary_flag")
        if record["request_terminal"] == "cancelled":
            _literal(record["boundary_complete"], False)
    issues = _uint(record["issues"])
    if issues & ~ISSUE_MASK:
        _fail("unknown_issue_bits")
    clock = _object(record["clock"], {"source", "ticks", "frequency_hz"}, "clock_shape")
    _literal(clock["source"], "Windows_QPC")
    if clock["ticks"] is None or clock["frequency_hz"] is None:
        if clock["ticks"] is not None or clock["frequency_hz"] is not None or not issues & CLOCK_UNAVAILABLE:
            _fail("clock_unavailable_shape")
    else:
        _uint(clock["ticks"], positive=True)
        _uint(clock["frequency_hz"], positive=True)
    _observer(record["observer"])
    inflight = _object(record["inflight"], {"cpu_methods_and_phases", "cuda_replays"}, "inflight_shape")
    for number in inflight.values():
        _uint(number)
    counters = _object(record["counters"], {"cpu", "cpu_phase", "cuda"}, "counter_shape")
    arrays = (
        ("cpu", CPU_FAMILIES, CPU_COUNTS, "CPU_expert_method", ()),
        ("cpu_phase", tuple(range(1, 7)), CPU_PHASE_COUNTS, "CPU_row_partition_phase", ()),
        ("cuda", CUDA_FAMILIES, CUDA_COUNTS, "CUDA_graph_replay_including_host_copy_nodes",
         ("last_launch_status", "last_fence_status")),
    )
    for name, families, counts, unit, statuses in arrays:
        rows = counters[name]
        if type(rows) is not list or len(rows) != len(PHASES) * len(families):
            _fail("counter_array_bound")
        identity_key = "mode" if name == "cpu_phase" else "family"
        for position, (phase, family) in enumerate((phase, family) for phase in PHASES for family in families):
            row = _object(rows[position], {"phase", identity_key, "unit", *counts, *statuses}, "counter_row_shape")
            _literal(row["phase"], phase)
            _literal(row[identity_key], family)
            _literal(row["unit"], unit)
            for key in counts:
                _uint(row[key])
            for key in statuses:
                _status(row[key])
    memory = _object(record["memory"], {"ordinary_expert_cache"}, "memory_shape")
    cache = _object(memory["ordinary_expert_cache"], {
        "unit", "accounting", *MEMORY_COUNTS, "last_allocation_status", "last_free_status",
        "tracked_live_requested_bytes", "lifetime_peak_requested_bytes", "request_peak_requested_bytes",
        "untracked_cache_family_bits", "physical_resident_bytes", "vmm_reserved_bytes", "vmm_mapped_committed_bytes",
    }, "cache_shape")
    _literal(cache["unit"], "bytes")
    _literal(cache["accounting"], "successful_owned_cudaMalloc_requested_bytes")
    for key in (*MEMORY_COUNTS, "tracked_live_requested_bytes", "lifetime_peak_requested_bytes", "request_peak_requested_bytes"):
        _uint(cache[key])
    for key in ("last_allocation_status", "last_free_status"):
        _status(cache[key])
    mask = _uint(cache["untracked_cache_family_bits"])
    if mask & ~UNTRACKED_CACHE_MASK:
        _fail("unknown_cache_family_bits")
    for key in ("physical_resident_bytes", "vmm_reserved_bytes", "vmm_mapped_committed_bytes"):
        _literal(cache[key], None)
    coverage = _object(record["coverage"], {"instrumented", "excluded", "entire_model_covered"}, "coverage_shape")
    _literal(coverage["instrumented"], list(INSTRUMENTED))
    _literal(coverage["excluded"], list(EXCLUDED))
    _literal(coverage["entire_model_covered"], False)
    for key in ("whole_model_placement", "physical_ssd_read_bytes"):
        _literal(record[key], None)
    for key in ("aggregate_gpu_hard_cap_verified", "aggregate_ram_hard_cap_verified"):
        _literal(record[key], False)
    return copy.deepcopy(record)


def parse_native_line(line: bytes | str) -> dict[str, Any]:
    """Parse one exact native frame; prefix recognition alone conveys no trust.

    CRT CRLF is allowed as a transport terminator. The native buffer bound is
    applied after converting that one terminator to the emitter's LF; NUL space
    is reserved. No other whitespace/prefix/suffix line is accepted.
    """
    try:
        raw = line.encode("utf-8", errors="strict") if type(line) is str else line
        if type(raw) is not bytes or len(raw) > NATIVE_FRAME_BYTES:
            _fail("native_size")
        if raw.endswith(b"\r\n"):
            raw = raw[:-2] + b"\n"
        if not raw.startswith(PREFIX.encode("ascii")) or not raw.endswith(b"\n"):
            _fail("native_prefix_or_terminator")
        if len(raw) + 1 > NATIVE_FRAME_BYTES or b"\n" in raw[:-1] or b"\r" in raw[:-1]:
            _fail("native_size_or_multiline")
        return validate_snapshot(_json(raw[len(PREFIX):-1], NATIVE_FRAME_BYTES))
    except ObservationError:
        raise
    except UnicodeError:
        _fail("native_encoding")


def _cumulative(snapshot: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for family, fields in (("cpu", CPU_COUNTS), ("cpu_phase", CPU_PHASE_COUNTS), ("cuda", CUDA_COUNTS)):
        for row in snapshot["counters"][family]:
            identity = row["mode"] if family == "cpu_phase" else row["family"]
            for key in fields:
                result[f"{family}:{row['phase']}:{identity}:{key}"] = row[key]
    cache = snapshot["memory"]["ordinary_expert_cache"]
    for key in (*MEMORY_COUNTS, "lifetime_peak_requested_bytes"):
        result[f"ordinary_cache:{key}"] = cache[key]
    return result


def _observed_api_error(snapshot: Mapping[str, Any]) -> bool:
    # Later independent loads may expose an error after the earlier issues read.
    # Classify partial; do not reject the snapshot or impose count equality.
    cache = snapshot["memory"]["ordinary_expert_cache"]
    if cache["allocation_errors"] or cache["failed_frees"]:
        return True
    if any(cache[key] not in (None, 0) for key in ("last_allocation_status", "last_free_status")):
        return True
    return any(
        row["launch_errors"] or row["fence_errors"]
        or any(row[key] not in (None, 0) for key in ("last_launch_status", "last_fence_status"))
        for row in snapshot["counters"]["cuda"]
    )


def _runtime(value: Any) -> dict[str, Any]:
    """Shape check only; verification_receipt must be established externally."""
    record = _object(value, {
        "schema", "authorization", "native_schema", "header_sha256", "header_v1_sha256", "schema_sha256",
        "incremental_patch_sha256", "boundary_patch_sha256", "base_io_patch_sha256", "combined_patch_manifest_sha256",
        "runtime_manifest_sha256", "native_executable_sha256", "owner_adapter_sha256",
        "loaded_module_receipt_sha256", "verification_receipt_sha256", "observer_layout",
    }, "runtime_binding_shape")
    _literal(record["schema"], RUNTIME_SCHEMA)
    _literal(record["authorization"], "externally_reviewed_combined_runtime_not_established_by_parser")
    for key, expected in (
        ("native_schema", NATIVE_SCHEMA), ("header_sha256", HEADER_SHA256), ("header_v1_sha256", HEADER_V1_SHA256),
        ("schema_sha256", SCHEMA_SHA256), ("boundary_patch_sha256", BOUNDARY_PATCH_SHA256),
        ("combined_patch_manifest_sha256", COMBINED_PATCH_MANIFEST_SHA256),
        ("incremental_patch_sha256", INCREMENTAL_PATCH_SHA256), ("base_io_patch_sha256", BASE_IO_PATCH_SHA256),
    ):
        _literal(record[key], expected)
    for key in ("combined_patch_manifest_sha256", "runtime_manifest_sha256", "native_executable_sha256",
                "owner_adapter_sha256", "loaded_module_receipt_sha256", "verification_receipt_sha256"):
        _hash(record[key])
    _observer(record["observer_layout"])
    return copy.deepcopy(record)


def _owner(value: Any) -> dict[str, Any]:
    record = _object(value, {"schema", "worker_generation", "pid", "creation_filetime_100ns", "stage_id", "gpu"},
                     "owner_binding_shape")
    _literal(record["schema"], OWNER_SCHEMA)
    _identifier(record["worker_generation"])
    _uint(record["pid"], positive=True, maximum=(1 << 32) - 1)
    _uint(record["creation_filetime_100ns"], positive=True)
    _uint(record["stage_id"])
    gpu = _object(record["gpu"], {"uuid", "pci_bus_id", "name_sha256"}, "gpu_shape")
    if type(gpu["uuid"]) is not str or _GPU_UUID.fullmatch(gpu["uuid"]) is None:
        _fail("gpu_uuid")
    if type(gpu["pci_bus_id"]) is not str or _PCI.fullmatch(gpu["pci_bus_id"]) is None:
        _fail("gpu_pci")
    _hash(gpu["name_sha256"])
    return copy.deepcopy(record)


class StrataExecutionObserver:
    """Bounded per-generation state machine, for a separately verified adapter.

    The constructor cannot verify a native executable, receipt, PID or GPU.
    Its inputs are prerequisites produced by that future external verifier.
    A channel token is private adapter authentication, never native self-proof.
    Any invalid owned frame retires this accumulator; eventual OS drain does not
    repair its evidence. It retains one active and one detached last report.
    """

    def __init__(self, runtime_binding: dict[str, Any], owner_binding: dict[str, Any], *, channel_token: str):
        self._runtime = _runtime(runtime_binding)
        self._owner = _owner(owner_binding)
        self._runtime_sha = _digest(self._runtime)
        self._owner_sha = _digest(self._owner)
        self._token = _hash(channel_token)
        self._lock = threading.RLock()
        self._active: dict[str, Any] | None = None
        self._last: dict[str, Any] | None = None
        self._retired = False
        self._last_epoch = 0
        self._last_native_seq = 0
        self._baseline: dict[str, int] | None = None
        self._last_issues = 0
        self._last_cache_bits = 0
        self._clock_ticks: int | None = None
        self._clock_hz: int | None = None

    @property
    def retired(self) -> bool:
        with self._lock:
            return self._retired

    @property
    def binding_identity(self) -> dict[str, Any]:
        return {"runtime_binding_sha256": self._runtime_sha, "owner_binding_sha256": self._owner_sha}

    def _reject(self, code: str) -> None:
        self._retired = True
        if self._active is not None and code not in self._active["reasons"]:
            if len(self._active["reasons"]) < 8:
                self._active["reasons"].append(code)
        _fail(code)

    def begin(self, request_id: str, epoch: int) -> None:
        with self._lock:
            if self._retired:
                _fail("retired_generation")
            try:
                _identifier(request_id)
                _uint(epoch, positive=True)
                if self._active is not None or epoch <= self._last_epoch:
                    _fail("request_epoch_or_overlap")
            except ObservationError as error:
                self._reject(error.code)
            self._last_epoch = epoch
            self._active = {"request_id": request_id, "epoch": epoch, "dispatch": None,
                            "snapshots": [], "native_done": None, "reasons": []}

    def ingest(self, frame: bytes | str | dict[str, Any]) -> None:
        """Accept only a complete protected owner envelope, never bare stdout."""
        with self._lock:
            if self._retired:
                _fail("retired_generation")
            try:
                if self._active is None:
                    _fail("no_active_request")
                if type(frame) in (bytes, str):
                    frame = _json(frame, OWNED_FRAME_BYTES)
                frame = _object(frame, {
                    "schema", "channel_token", "runtime_binding_sha256", "owner_binding_sha256",
                    "worker_generation", "pid", "creation_filetime_100ns", "gpu_uuid", "stage_id",
                    "request_id", "epoch", "kind", "payload",
                }, "owned_frame_shape")
                if len(_encoded(frame)) > OWNED_FRAME_BYTES:
                    _fail("owned_frame_size")
                _literal(frame["schema"], FRAME_SCHEMA)
                if not hmac.compare_digest(_hash(frame["channel_token"]), self._token):
                    _fail("owned_channel")
                _literal(frame["runtime_binding_sha256"], self._runtime_sha)
                _literal(frame["owner_binding_sha256"], self._owner_sha)
                for key in ("worker_generation", "pid", "creation_filetime_100ns", "stage_id"):
                    _literal(frame[key], self._owner[key], "owned_process_identity")
                _literal(frame["gpu_uuid"], self._owner["gpu"]["uuid"], "owned_gpu_identity")
                for key in ("request_id", "epoch"):
                    _literal(frame[key], self._active[key], "owned_request_identity")
                if type(frame["kind"]) is not str or frame["kind"] not in ("dispatch", "snapshot", "native_done"):
                    _fail("owned_frame_kind")
                if frame["kind"] == "dispatch":
                    self._dispatch(frame["payload"])
                elif frame["kind"] == "snapshot":
                    self._snapshot(frame["payload"])
                else:
                    self._done(frame["payload"])
            except ObservationError as error:
                self._reject(error.code)

    def _dispatch(self, value: Any) -> None:
        active = self._active
        assert active is not None
        dispatch = _object(value, {"command", "native_request_seq"}, "dispatch_shape")
        if type(dispatch["command"]) is not str or dispatch["command"] not in ("GEN", "GENI"):
            _fail("dispatch_command")
        seq = _uint(dispatch["native_request_seq"], positive=True, maximum=NATIVE_SEQUENCE_MAX)
        if active["dispatch"] is not None or seq <= self._last_native_seq:
            _fail("stale_or_duplicate_dispatch")
        self._last_native_seq = seq
        active["dispatch"] = copy.deepcopy(dispatch)

    def _snapshot(self, value: Any) -> None:
        active = self._active
        assert active is not None
        if active["dispatch"] is None or active["native_done"] is not None:
            _fail("snapshot_dispatch_order")
        snapshot = validate_snapshot(value)
        snapshots = active["snapshots"]
        if snapshot["native_request_seq"] != active["dispatch"]["native_request_seq"]:
            _fail("native_sequence_identity")
        if len(snapshots) >= 3 or snapshot["snapshot_seq"] != len(snapshots):
            _fail("native_snapshot_order")
        if _encoded(snapshot["observer"]) != _encoded(self._runtime["observer_layout"]):
            _fail("compiled_observer_layout")
        current = _cumulative(snapshot)
        if self._baseline is not None and any(current[key] < previous for key, previous in self._baseline.items()):
            _fail("cumulative_counter_regression")
        if snapshot["issues"] | self._last_issues != snapshot["issues"]:
            _fail("sticky_issue_regression")
        cache = snapshot["memory"]["ordinary_expert_cache"]
        if cache["untracked_cache_family_bits"] | self._last_cache_bits != cache["untracked_cache_family_bits"]:
            _fail("sticky_cache_family_regression")
        if snapshots and cache["request_peak_requested_bytes"] < snapshots[-1]["memory"]["ordinary_expert_cache"]["request_peak_requested_bytes"]:
            _fail("request_peak_regression")
        clock = snapshot["clock"]
        if clock["ticks"] is not None:
            if self._clock_hz is not None and (clock["frequency_hz"] != self._clock_hz or clock["ticks"] < self._clock_ticks):
                _fail("qpc_regression_or_frequency")
            self._clock_ticks = clock["ticks"]
            self._clock_hz = clock["frequency_hz"]
        self._baseline = current
        self._last_issues = snapshot["issues"]
        self._last_cache_bits = cache["untracked_cache_family_bits"]
        snapshots.append(snapshot)

    def _done(self, value: Any) -> None:
        active = self._active
        assert active is not None
        done = _object(value, {"native_request_seq", "finish_reason"}, "native_done_shape")
        seq = _uint(done["native_request_seq"], positive=True, maximum=NATIVE_SEQUENCE_MAX)
        if active["dispatch"] is None or active["native_done"] is not None:
            _fail("native_done_order")
        if seq != active["dispatch"]["native_request_seq"]:
            _fail("native_done_sequence")
        if type(done["finish_reason"]) is not str or done["finish_reason"] not in ("stop", "length", "cancelled", "error"):
            _fail("native_done_reason")
        if active["snapshots"] and active["snapshots"][-1]["snapshot_seq"] == 2:
            terminal = active["snapshots"][-1]["request_terminal"]
            if (terminal == "cancelled") != (done["finish_reason"] == "cancelled"):
                _fail("native_terminal_mismatch")
        active["native_done"] = copy.deepcopy(done)

    def finish(self, request_id: str, epoch: int, *, omni_completed: bool, cancellation_requested: bool = False) -> dict[str, Any]:
        """Detach bounded evidence. Cancellation/OS drain cannot manufacture end."""
        with self._lock:
            if self._active is None:
                _fail("no_active_request")
            active = self._active
            try:
                _literal(request_id, active["request_id"], "finish_request_identity")
                _literal(epoch, active["epoch"], "finish_request_identity")
                if type(omni_completed) is not bool or type(cancellation_requested) is not bool:
                    _fail("finish_flag_type")
            except ObservationError as error:
                self._reject(error.code)
            reasons = list(active["reasons"])
            snapshots = active["snapshots"]
            if active["dispatch"] is None:
                reasons.append("missing_dispatch")
            if len(snapshots) != 3:
                reasons.append("missing_native_snapshots")
            done = active["native_done"]
            if done is None:
                reasons.append("missing_native_done")
            elif done["finish_reason"] not in ("stop", "length"):
                reasons.append("native_not_normally_completed")
            if not omni_completed:
                reasons.append("omni_not_completed")
            if cancellation_requested:
                reasons.append("cancelled_request")
            if snapshots:
                final = snapshots[-1]
                if final["issues"]:
                    reasons.append("native_sticky_issues")
                if not final["boundary_complete"]:
                    reasons.append("native_boundary_incomplete")
                if final["request_terminal"] != "completed":
                    reasons.append("native_request_not_completed")
                if any(final["inflight"].values()):
                    reasons.append("observed_work_inflight")
                if any(_observed_api_error(snapshot) for snapshot in snapshots):
                    reasons.append("observed_native_api_error")
                if any(snapshot["clock"]["ticks"] is None for snapshot in snapshots):
                    reasons.append("native_clock_unavailable")
            reasons = list(dict.fromkeys(reasons))
            deltas: list[dict[str, Any]] = []
            for before, after in zip(snapshots, snapshots[1:]):
                left, right = _cumulative(before), _cumulative(after)
                deltas.append({
                    "from_snapshot_seq": before["snapshot_seq"], "to_snapshot_seq": after["snapshot_seq"],
                    "scope": "per_named_cumulative_counter_difference_not_cross_counter_sum",
                    "values": {key: right[key] - left[key] for key in left
                               if key != "ordinary_cache:lifetime_peak_requested_bytes"},
                })
            report = {
                "schema": REPORT_SCHEMA,
                "status": "complete_scoped_observation" if not reasons else ("partial" if snapshots else "unavailable"),
                "complete": not reasons, "reasons": reasons,
                "request_id": active["request_id"], "epoch": active["epoch"],
                "runtime_binding": self._runtime, "runtime_binding_sha256": self._runtime_sha,
                "owner_binding": self._owner, "owner_binding_sha256": self._owner_sha,
                "binding_verified_by_parser": False,
                "dispatch": active["dispatch"], "native_done": done,
                "omni_completed": omni_completed, "cancellation_requested": cancellation_requested,
                "snapshots": snapshots,
                "snapshot_sha256": [_digest(snapshot) for snapshot in snapshots],
                "snapshot_hash_scope": "canonical_JSON_payload_not_original_stdout_bytes",
                "counter_deltas": deltas,
                "snapshot_coherent": False,
                "counter_scope": "process_lifetime_cumulative_phase_at_submission",
                "coverage": {"instrumented": list(INSTRUMENTED), "excluded": list(EXCLUDED), "entire_model_covered": False},
                "memory_scope": "ordinary_owned_cudaMalloc_requested_bytes_only_not_physical_or_aggregate",
                "physical_resident_bytes": None, "whole_model_placement": None, "physical_ssd_read_bytes": None,
                "aggregate_gpu_hard_cap_verified": False, "aggregate_ram_hard_cap_verified": False,
                "runtime_qualification": False, "default_eligible": False,
                "bounds": {"native_snapshots_max": 3, "native_frame_bytes": NATIVE_FRAME_BYTES,
                           "owned_frame_bytes": OWNED_FRAME_BYTES, "detached_report_bytes": REPORT_BYTES,
                           "retained_requests_max": 2, "observer_concurrent_stack_overhead_verified": False,
                           "python_heap_hard_cap_verified": False},
            }
            if len(_encoded(report)) > REPORT_BYTES:
                self._reject("detached_report_bound")
            # Detach first; never let callers mutate the monotonic baseline or
            # runtime/owner state. Last report is the only retained history.
            detached = copy.deepcopy(report)
            self._last = detached
            self._active = None
            if self._last_issues or any(reason in reasons for reason in (
                "missing_native_snapshots", "missing_native_done", "native_not_normally_completed",
                "omni_not_completed", "cancelled_request", "native_boundary_incomplete",
                "observed_work_inflight", "native_clock_unavailable", "observed_native_api_error",
            )):
                self._retired = True
            return copy.deepcopy(detached)

    def last_observation(self) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self._last)
