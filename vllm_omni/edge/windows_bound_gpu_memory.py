# SPDX-License-Identifier: Apache-2.0
"""Optional synchronous WDDM sampling of borrowed, already-retained handles.

No process discovery, process-handle open/duplication, polling thread, budget or
admission policy belongs here. The registry owns process handles. This provider
owns only once-bound adapter query handles, and must drain before its borrower
closes any process handle. Local/nonlocal accounting never becomes extra RAM.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

BOUND_GPU_SCHEMA = "omni-windows-bound-process-gpu-memory-v1"
_CAPABILITY_SOURCE = "native_DXGI_GetDesc1_D3D12_Architecture1"
_MAX_ADAPTERS = 16
_MAX_NODES_PER_ADAPTER = 16
_MAX_TOTAL_NODES = 64
_COUNTERS = ("current_usage_bytes", "budget_bytes", "current_reservation_bytes", "available_for_reservation_bytes")


def _error(exc: BaseException) -> str:
    try:
        message = str(exc)[:192].encode("unicode_escape").decode("ascii")[:256]
    except BaseException:
        message = "unprintable_exception"
    return type(exc).__name__[:64] + ":" + message


def _uint(value: Any, *, positive: bool = False, bits: int = 64) -> bool:
    return type(value) is int and (1 if positive else 0) <= value < (1 << bits)


@dataclass(frozen=True)
class _AdapterNode:
    adapter_luid_hex: str
    physical_adapter_index: int
    physical_adapter_count: int
    description: str
    vendor_id: int
    device_id: int
    uma: bool
    cache_coherent_uma: bool

    def metadata(self) -> dict[str, Any]:
        row = asdict(self)
        local = "host_ram" if self.uma else f"wddm_local:{self.adapter_luid_hex}:{self.physical_adapter_index}"
        row.update(
            local_physical_pool_alias=local,
            nonlocal_physical_pool_alias=None if self.uma else "host_ram",
            local_pool_kind="shared_system_ram" if self.uma else "discrete_gpu_local",
            local_omni_pool_id="host_ram" if self.uma else None,
            nonlocal_omni_pool_id=None if self.uma else "host_ram",
            nonlocal_pool_kind="no_independent_UMA_nonlocal_pool_declared" if self.uma else "shared_system_ram",
            discrete_omni_pool_mapping="requires_explicit_LUID_node_join_to_engine_inventory",
            cpu_numa_topology_observed=False,
        )
        return row


def _validated_nodes(inventory: Any) -> tuple[_AdapterNode, ...]:
    """Validate the explicit native inventory; never infer a missing node/UMA bit."""
    if (not isinstance(inventory, Sequence) or isinstance(inventory, (str, bytes))
            or not 1 <= len(inventory) <= _MAX_ADAPTERS):
        raise ValueError("native adapter inventory is empty, unavailable or over bound")
    nodes: list[_AdapterNode] = []
    luids: set[str] = set()
    for adapter in inventory:
        if not isinstance(adapter, Mapping):
            raise ValueError("invalid adapter descriptor")
        capability = adapter.get("gpu_memory_accounting")
        if (not isinstance(capability, Mapping) or capability.get("available") is not True
                or capability.get("source") != _CAPABILITY_SOURCE):
            raise ValueError("native adapter accounting capability is unavailable")
        luid = capability.get("adapter_luid_hex")
        if (type(luid) is not str or len(luid) != 16
                or any(char not in "0123456789abcdef" for char in luid) or luid in luids):
            raise ValueError("adapter LUID must be unique canonical eight-byte little-endian hex")
        luids.add(luid)
        count, declared = capability.get("physical_adapter_count"), capability.get("nodes")
        if (not _uint(count, positive=True, bits=32) or count > _MAX_NODES_PER_ADAPTER
                or type(declared) is not list or len(declared) != count):
            raise ValueError("physical adapter node inventory is incomplete or over bound")
        description = adapter.get("description")
        if type(description) is not str or not 1 <= len(description) <= 128:
            raise ValueError("bounded native adapter description is required")
        description.encode("utf-8")
        vendor, device = adapter.get("vendor_id"), adapter.get("device_id")
        if not _uint(vendor, bits=32) or not _uint(device, bits=32):
            raise ValueError("native vendor/device identity is required")
        for index, node in enumerate(declared):
            if (not isinstance(node, Mapping) or type(node.get("physical_adapter_index")) is not int
                    or node["physical_adapter_index"] != index or type(node.get("uma")) is not bool
                    or type(node.get("cache_coherent_uma")) is not bool):
                raise ValueError("node indices and native architecture bits must be exact")
            nodes.append(_AdapterNode(luid, index, count, description, vendor, device,
                                      node["uma"], node["cache_coherent_uma"]))
    if len(nodes) > _MAX_TOTAL_NODES:
        raise ValueError("total physical adapter nodes exceed observation bound")
    return tuple(nodes)


def _process_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError("registry process identity is required")
    row = {key: identity.get(key) for key in (
        "cohort_generation", "role", "pid", "creation_filetime_100ns", "image_path_sha256")}
    if (not _uint(row["pid"], positive=True, bits=32)
            or not _uint(row["creation_filetime_100ns"], positive=True)):
        raise ValueError("exact PID and birth time are required")
    for key, limit in (("cohort_generation", 256), ("role", 64)):
        if type(row[key]) is not str or not 1 <= len(row[key]) <= limit:
            raise ValueError("bounded process generation and role are required")
        row[key].encode("utf-8")
    digest = row["image_path_sha256"]
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("registry image-path hash is required")
    return row


class WindowsRetainedProcessGpuProvider:
    """One capability snapshot, synchronous borrows, explicit partial coverage.

    ``adapter_inventory`` must come from the opt-in native DXGI/D3D12 inventory
    in this source version. KMT opening cross-checks its physical-node count.
    No descriptors are silently dropped. An unavailable adapter invalidates
    binding coverage for the set. A failed segment stays unavailable while a
    successful segment remains useful, with whole-observation availability false.
    """

    def __init__(self, adapter_inventory: Any, *, transport: Any = None) -> None:
        self._lock = threading.RLock()
        self._nodes: tuple[_AdapterNode, ...] = ()
        self._adapters: dict[str, Any] = {}
        self._adapter_close_attempts: set[str] = set()
        self._adapter_verified_closed: set[str] = set()
        self._adapter_close_outcomes: dict[str, dict[str, Any]] = {}
        self._adapter_close_errors: dict[str, str] = {}
        self._transport = transport
        self._binding_error: str | None = None
        self._closed = False
        self._receipt: dict[str, Any] | None = None
        self._capability_sha256 = hashlib.sha256(b"[]").hexdigest()
        try:
            self._nodes = _validated_nodes(adapter_inventory)
            self._capability_sha256 = hashlib.sha256(json.dumps(
                [node.metadata() for node in self._nodes], sort_keys=True, ensure_ascii=True,
                separators=(",", ":")).encode("ascii")).hexdigest()
            if transport is None:
                from vllm_omni.edge.windows_gpu_memory import _WindowsTransport

                self._transport = _WindowsTransport(retained_handle_queries=True)
            else:
                self._transport = transport
            for node in self._nodes:
                if node.adapter_luid_hex in self._adapters:
                    continue
                handle = self._transport.open_adapter(node.adapter_luid_hex)
                self._adapters[node.adapter_luid_hex] = handle
                count = self._transport.physical_adapter_count(handle)
                if type(count) is not int or count != node.physical_adapter_count:
                    raise ValueError("D3D12 and KMT physical-adapter counts disagree")
        except BaseException as exc:
            self._binding_error = _error(exc)
            cleanup_interruption = self._close_adapters()
            if not isinstance(exc, Exception):
                # Clean every known resource once, then preserve the original
                # constructor cancellation even if cleanup also interrupted.
                raise
            if cleanup_interruption is not None:
                raise cleanup_interruption from exc

    def _close_adapters(self) -> BaseException | None:
        interruption: BaseException | None = None
        for luid, handle in self._adapters.items():
            if luid in self._adapter_close_attempts:
                continue
            outcome: dict[str, Any] = {"close_attempted": True, "verified_closed": False,
                                       "error_type": None, "error": None}
            self._adapter_close_outcomes[luid] = outcome
            self._adapter_close_attempts.add(luid)
            try:
                self._transport.close_adapter(handle)
                self._adapter_verified_closed.add(luid)
                outcome["verified_closed"] = True
            except BaseException as exc:
                self._adapter_verified_closed.discard(luid)
                self._adapter_close_errors[luid] = _error(exc)
                outcome.update(verified_closed=False, error_type=type(exc).__name__,
                               error=self._adapter_close_errors[luid])
                if not isinstance(exc, Exception) and interruption is None:
                    interruption = exc
        return interruption

    def capabilities(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": BOUND_GPU_SCHEMA, "enabled": True,
                "capability_sha256": self._capability_sha256,
                "capability_source": _CAPABILITY_SOURCE,
                "binding_available": self._binding_error is None and not self._closed,
                "binding_error": self._binding_error, "closed": self._closed,
                "max_adapters": _MAX_ADAPTERS, "max_nodes_per_adapter": _MAX_NODES_PER_ADAPTER,
                "max_total_nodes": _MAX_TOTAL_NODES,
                "adapter_nodes": [node.metadata() for node in self._nodes],
                "adapter_set_scope": "explicit_once_bound_native_inventory_not_dynamic_discovery",
                "hotplug_after_binding_covered": False, "polling_threads_created": 0,
                "process_handles_opened_or_duplicated": 0, "hard_cap_verified": False,
                "physical_pool_mapping_is_not_an_admission_budget": True,
            }

    def unavailable(self, identity: Mapping[str, Any], reason: str) -> dict[str, Any]:
        return {"schema": BOUND_GPU_SCHEMA, "enabled": True, "available": False,
                "status": "unavailable", "reason": reason[:256],
                "process_identity": _process_identity(identity),
                "capability_sha256": self._capability_sha256,
                "adapter_nodes": [], "hard_cap_verified": False,
                "missing_measurements_are_not_zero": True,
                "sampled_peaks_are_lower_bounds": True,
                "local_nonlocal_and_process_working_set_must_not_be_added": True}

    def sample_process(self, process_handle: Any, identity: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            owner = _process_identity(identity)
            if self._closed or self._binding_error is not None:
                return self.unavailable(owner, "provider_closed" if self._closed else self._binding_error)
            if process_handle is None or process_handle == 0:
                return self.unavailable(owner, "missing_retained_process_handle")
            started = time.monotonic_ns()
            try:
                self._transport.check_process(process_handle, owner["pid"], owner["creation_filetime_100ns"])
            except Exception as exc:
                return self.unavailable(owner, _error(exc))
            rows: list[dict[str, Any]] = []
            observed = 0
            for node in self._nodes:
                row = node.metadata()
                for segment, label in ((0, "local"), (1, "nonlocal")):
                    sample = {"status": "unavailable", "available": False,
                              **dict.fromkeys(_COUNTERS), "error": None}
                    try:
                        values = self._transport.query_segment(
                            process_handle, self._adapters[node.adapter_luid_hex],
                            node.physical_adapter_index, segment)
                        if not isinstance(values, Mapping) or any(not _uint(values.get(key)) for key in _COUNTERS):
                            raise ValueError("invalid or incomplete unsigned WDDM memory counters")
                        sample.update({key: values[key] for key in _COUNTERS})
                        sample.update(status="observed", available=True)
                        observed += 1
                    except Exception as exc:
                        sample["error"] = _error(exc)
                    row[label] = sample
                row["available"] = row["local"]["available"] and row["nonlocal"]["available"]
                rows.append(row)
            try:
                self._transport.check_process(process_handle, owner["pid"], owner["creation_filetime_100ns"])
            except Exception as exc:
                # An exit during the sample invalidates its complete measurement;
                # retained handles prevent a reused PID from becoming a new target.
                return self.unavailable(owner, "process_check_after_query:" + _error(exc))
            expected = len(self._nodes) * 2
            return {"schema": BOUND_GPU_SCHEMA, "enabled": True,
                    "available": observed == expected, "status": "observed" if observed == expected else (
                        "partial" if observed else "unavailable"),
                    "reason": None if observed == expected else "one_or_more_segments_unavailable",
                    "process_identity": owner, "capability_sha256": self._capability_sha256,
                    "started_monotonic_ns": started, "finished_monotonic_ns": time.monotonic_ns(),
                    "adapter_nodes": rows, "expected_segment_count": expected,
                    "observed_segment_count": observed, "sampled_peaks_are_lower_bounds": True,
                    "local_nonlocal_and_process_working_set_must_not_be_added": True,
                    "wddm_budget_is_not_omni_admission_or_hard_cap": True,
                    "missing_measurements_are_not_zero": True, "hard_cap_verified": False}

    def close(self) -> dict[str, Any]:
        with self._lock:
            if self._receipt is None:
                # The same lock serializes every synchronous borrow. No thread,
                # callback or request may use process handles after this point.
                self._closed = True
                interruption = self._close_adapters()
                self._receipt = {
                    "schema": BOUND_GPU_SCHEMA, "drained": True,
                    "adapter_handles_closed": all(luid in self._adapter_verified_closed for luid in self._adapters),
                    "adapter_handle_count": len(self._adapters),
                    "adapter_close_attempt_count": len(self._adapter_close_attempts),
                    "verified_adapter_close_count": len(self._adapter_verified_closed),
                    "adapter_close_outcomes": deepcopy(self._adapter_close_outcomes),
                    "adapter_close_errors": dict(self._adapter_close_errors),
                    "close_interruption_type": type(interruption).__name__ if interruption is not None else None,
                    "binding_error": self._binding_error,
                    "capability_sha256": self._capability_sha256,
                    "borrowed_process_handles_retained": False,
                    "process_handles_closed_by_provider": 0,
                    "finished_monotonic_ns": time.monotonic_ns(),
                }
                if interruption is not None:
                    # Receipt is terminal before cancellation propagates. A
                    # repeated close cannot retry or turn an unknown close green.
                    raise interruption
            return deepcopy(self._receipt)
