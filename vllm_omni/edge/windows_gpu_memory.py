# SPDX-License-Identifier: Apache-2.0
"""Read-only WDDM process memory observations bound to an owned process and GPU.

These are sampled OS accounting values, not allocation limits or CUDA kernel
placement proof. Nonlocal memory shares physical RAM and is not an extra pool.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

SCHEMA = "omni-windows-process-gpu-memory-v1"


def normalize_pci_bus_id(value: str) -> str:
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value)
    if not match:
        raise ValueError("invalid PCI bus identity")
    return ":".join(f"{int(part, 16):x}" for part in match.groups()[:3]) + "." + match[4]


def identity_key(identity: dict[str, Any]) -> tuple:
    gpu = identity.get("gpu", {})
    if (
        identity.get("status") != "verified"
        or type(identity.get("pid")) is not int
        or identity["pid"] <= 0
        or type(identity.get("creation_filetime_100ns")) is not int
        or identity["creation_filetime_100ns"] <= 0
        or not isinstance(identity.get("worker_generation"), str)
        or not identity["worker_generation"]
        or not isinstance(gpu, dict)
        or not isinstance(gpu.get("uuid"), str)
        or not gpu["uuid"].startswith("GPU-")
        or not re.fullmatch(r"[0-9a-f]{64}", gpu.get("name_sha256", ""))
    ):
        raise ValueError("complete verified owner process and GPU identity is required")
    return (
        identity["pid"],
        identity["creation_filetime_100ns"],
        identity["worker_generation"],
        gpu["uuid"],
        normalize_pci_bus_id(gpu["pci_bus_id"]),
        gpu["name_sha256"],
    )


def unknown_observation(reason: str, identity: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "unknown",
        "reason": reason,
        "unix_s": time.time(),
        "owner_identity": copy.deepcopy(identity),
        "local_current_usage_bytes": None,
        "nonlocal_current_usage_bytes": None,
        "local_budget_bytes": None,
        "nonlocal_budget_bytes": None,
        "scope": "sampled WDDM process accounting; nonlocal memory shares host RAM",
        "hard_cap_verified": False,
    }


def validate_native_gpu_pool_binding(
    *, gpu_identity: dict[str, Any], gpu_index: int, gpu_pool: str,
    ledger_capacities: dict[str, int], cuda_binding: dict[str, Any],
    provider_capabilities: dict[str, Any],
) -> dict[str, Any]:
    """Join app-owned native snapshots, never authenticate external JSON.

    No native call, allocation, admission, process lookup or budget selection.
    The caller must obtain both snapshots from the current owned native provider
    and identity resolver; strings/hashes alone are not an authority boundary.
    """
    from vllm_omni.edge.windows_bound_gpu_memory import BOUND_GPU_SCHEMA, _AdapterNode, _uint

    expected, binding, capability = (copy.deepcopy(value) for value in (
        gpu_identity, cuda_binding, provider_capabilities))
    if (not all(isinstance(value, dict) for value in (expected, binding, capability, ledger_capacities))
            or not _uint(gpu_index, bits=31)):
        raise ValueError("explicit GPU identity, configured ordinal and actual ledger are required")
    uuid, name_hash = expected.get("uuid"), expected.get("name_sha256")
    if (not isinstance(uuid, str) or not re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}", uuid)
            or not isinstance(name_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", name_hash)):
        raise ValueError("complete expected NVML UUID/PCI/name identity is required")
    pci = normalize_pci_bus_id(expected.get("pci_bus_id", ""))
    canonical_pool = f"vram:{gpu_index}"
    aliases = ["vram", canonical_pool] if gpu_index == 0 else [canonical_pool]
    if (gpu_pool not in aliases or gpu_pool not in ledger_capacities
            or not _uint(ledger_capacities[gpu_pool])
            or sum(alias in ledger_capacities for alias in aliases) != 1):
        raise ValueError("actual ledger must expose exactly one canonical GPU pool alias")
    luid, node, mask = (binding.get(key) for key in (
        "adapter_luid_hex", "physical_adapter_index", "cuda_node_mask"))
    if (binding.get("uuid") != uuid
            or normalize_pci_bus_id(binding.get("pci_bus_id", "")) != pci
            or binding.get("name_sha256") != name_hash
            or type(binding.get("cuda_device_ordinal")) is not int
            or binding["cuda_device_ordinal"] != gpu_index
            or not _uint(mask, positive=True, bits=32) or mask & (mask - 1)
            or not _uint(node, bits=32) or node != mask.bit_length() - 1
            or not isinstance(luid, str) or not re.fullmatch(r"[0-9a-f]{16}", luid)
            or binding.get("binding_source") != (
                "NVML UUID/PCI to CUDA cuDeviceGetByPCIBusId/cuDeviceGetLuid to WDDM LUID")):
        raise ValueError("CUDA UUID/PCI/ordinal/LUID/node identity is incomplete or mismatched")
    if (not _uint(binding.get("cuda_driver_version"), positive=True, bits=32)
            or not isinstance(binding.get("observer_driver_dll"), str)
            or not 1 <= len(binding["observer_driver_dll"]) <= 4096
            or not re.fullmatch(r"[0-9a-f]{64}", binding.get("observer_driver_dll_sha256", ""))):
        raise ValueError("identity resolver native driver provenance is incomplete")
    nodes, capability_hash = capability.get("adapter_nodes"), capability.get("capability_sha256")
    if (capability.get("schema") != BOUND_GPU_SCHEMA or capability.get("enabled") is not True
            or capability.get("binding_available") is not True or capability.get("closed") is not False
            or capability.get("binding_error") is not None
            or capability.get("capability_source") != "native_DXGI_GetDesc1_D3D12_Architecture1"
            or not isinstance(nodes, list) or not 1 <= len(nodes) <= 64
            or not isinstance(capability_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", capability_hash)):
        raise ValueError("current bounded native provider capabilities are unavailable")
    groups: dict[str, dict[str, Any]] = {}
    matches = []
    for row in nodes:
        if not isinstance(row, dict):
            raise ValueError("invalid provider node metadata")
        keys = ("adapter_luid_hex", "physical_adapter_index", "physical_adapter_count",
                "description", "vendor_id", "device_id", "uma", "cache_coherent_uma")
        core = {key: row.get(key) for key in keys}
        adapter, index, count = (core[key] for key in keys[:3])
        if (not isinstance(adapter, str) or not re.fullmatch(r"[0-9a-f]{16}", adapter)
                or not _uint(count, positive=True, bits=32) or count > 16
                or not _uint(index, bits=32) or index >= count
                or not isinstance(core["description"], str) or not 1 <= len(core["description"]) <= 128
                or not _uint(core["vendor_id"], bits=32) or not _uint(core["device_id"], bits=32)
                or type(core["uma"]) is not bool or type(core["cache_coherent_uma"]) is not bool
                or _AdapterNode(**core).metadata() != row):
            raise ValueError("provider node identity/topology/pool metadata is malformed")
        group_identity = (count, core["description"], core["vendor_id"], core["device_id"])
        group = groups.setdefault(adapter, {"identity": group_identity, "indices": set()})
        if group["identity"] != group_identity or index in group["indices"]:
            raise ValueError("provider adapter/node identity is ambiguous")
        group["indices"].add(index)
        if (adapter, index) == (luid, node):
            matches.append(row)
    if (len(groups) > 16 or any(group["indices"] != set(range(group["identity"][0]))
                                for group in groups.values())):
        raise ValueError("provider physical adapter nodes are incomplete or over bound")
    actual_hash = hashlib.sha256(json.dumps(nodes, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()
    if actual_hash != capability_hash or len(matches) != 1 or matches[0]["uma"] is not False:
        raise ValueError("exact discrete CUDA/provider node join is unavailable")
    return {
        "schema": "omni-windows-native-gpu-pool-binding-v1", "status": "identity_joined",
        "ledger_pool": gpu_pool, "canonical_pool": canonical_pool, "pool_aliases": aliases,
        "aliases_are_not_independent_capacities": True,
        "physical_pool_identity": f"wddm_local:{luid}:{node}",
        "provider_capability_sha256": capability_hash, "provider_node": copy.deepcopy(matches[0]),
        "gpu_identity": expected, "cuda_binding": binding,
        "nonlocal_physical_pool_alias": "host_ram", "memory_budget_selected": False,
        "memory_admission_performed": False, "native_execution_placement_verified": False,
        "hard_cap_verified": False, "qualification": False,
        "scope": "current_app_owned_native_identity_snapshots_not_external_JSON_or_execution_proof",
    }


class WindowsProcessGpuObserver:
    """Observe one native generation; any identity or platform mismatch is unknown."""

    def __init__(self, identity: dict[str, Any], *, transport: Any = None) -> None:
        self.identity = copy.deepcopy(identity)
        self.transport = transport
        self.binding = None
        self.error = None
        try:
            self.key = identity_key(self.identity)
            if transport is None:
                transport = _WindowsTransport()
            self.transport = transport
            self.binding = copy.deepcopy(transport.resolve_gpu(self.identity["gpu"]))
            self.binding_key = json.dumps(self.binding, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def sample(self) -> dict[str, Any]:
        result = unknown_observation(self.error or "not observed", self.identity)
        if self.error:
            return result
        try:
            if identity_key(self.identity) != self.key:
                raise ValueError("owner identity changed after GPU binding; observation refused")
            if json.dumps(self.binding, sort_keys=True, separators=(",", ":"), allow_nan=False) != self.binding_key:
                raise ValueError("GPU binding changed after resolution; observation refused")
            usage = self.transport.query(self.identity["pid"], self.identity["creation_filetime_100ns"], self.binding)
            for name in (
                "local_current_usage_bytes",
                "nonlocal_current_usage_bytes",
                "local_budget_bytes",
                "nonlocal_budget_bytes",
            ):
                if type(usage.get(name)) is not int or usage[name] < 0:
                    raise ValueError("incomplete WDDM memory observation")
            result.update(usage, status="observed", reason=None, gpu_binding=copy.deepcopy(self.binding))
        except Exception as exc:
            result["reason"] = f"{type(exc).__name__}: {exc}"
        return result


class ProcessGpuPeakTracker:
    """Keep separate sampled peaks for each owned process/GPU generation."""

    def __init__(self) -> None:
        self.generations: dict[tuple, dict[str, Any]] = {}

    def add(self, row: dict[str, Any]) -> None:
        if row.get("status") != "observed":
            return
        key = identity_key(row["owner_identity"])
        peak = self.generations.setdefault(
            key,
            {
                "owner_identity": copy.deepcopy(row["owner_identity"]),
                "samples": 0,
                "sampled_local_peak_bytes": 0,
                "sampled_nonlocal_peak_bytes": 0,
                "hard_cap_verified": False,
                "scope": "separate sampled WDDM local/nonlocal accounting; do not add nonlocal to host RAM",
            },
        )
        peak["samples"] += 1
        peak["sampled_local_peak_bytes"] = max(peak["sampled_local_peak_bytes"], row["local_current_usage_bytes"])
        peak["sampled_nonlocal_peak_bytes"] = max(
            peak["sampled_nonlocal_peak_bytes"], row["nonlocal_current_usage_bytes"]
        )

    def snapshot(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self.generations.values()))


class _WindowsTransport:
    def __init__(self, *, retained_handle_queries: bool = False) -> None:
        if os.name != "nt":
            raise OSError("WDDM process accounting requires native Windows")
        from ctypes import wintypes as w

        class Luid(ctypes.Structure):
            _fields_ = [("low", w.DWORD), ("high", w.LONG)]

        class Adapter(ctypes.Structure):
            _fields_ = [("handle", w.UINT), ("luid", Luid), ("sources", w.ULONG), ("precise", w.BOOL)]

        class Enumeration(ctypes.Structure):
            _fields_ = [("count", w.ULONG), ("adapters", ctypes.POINTER(Adapter))]

        class Query(ctypes.Structure):
            _fields_ = (
                [("process", w.HANDLE), ("adapter", w.UINT), ("segment", ctypes.c_int)]
                + [(name, ctypes.c_uint64) for name in ("budget", "usage", "reservation", "available")]
                + [("physical_index", w.UINT)]
            )

        class Close(ctypes.Structure):
            _fields_ = [("adapter", w.UINT)]

        self.Adapter, self.Enumeration, self.Query, self.Close = Adapter, Enumeration, Query, Close
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
        self.kernel.GetProcessTimes.restype = w.BOOL
        self.kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        self.kernel.GetExitCodeProcess.restype = w.BOOL
        self.gdi = ctypes.WinDLL("gdi32", use_last_error=True)
        for name, struct in (
            ("D3DKMTEnumAdapters2", Enumeration),
            ("D3DKMTQueryVideoMemoryInfo", Query),
            ("D3DKMTCloseAdapter", Close),
        ):
            function = getattr(self.gdi, name)
            function.argtypes = [ctypes.POINTER(struct)]
            function.restype = w.LONG
        if retained_handle_queries:
            if ctypes.sizeof(ctypes.c_void_p) != 8:
                raise OSError("retained WDDM sampling requires native Windows x64")

            class OpenAdapter(ctypes.Structure):
                _fields_ = [("luid", Luid), ("adapter", w.UINT)]

            class AdapterInfo(ctypes.Structure):
                _fields_ = [("adapter", w.UINT), ("kind", ctypes.c_int),
                            ("data", ctypes.c_void_p), ("size", w.UINT)]

            if (ctypes.sizeof(OpenAdapter), ctypes.sizeof(AdapterInfo), ctypes.sizeof(Query),
                    ctypes.sizeof(Close), Query.budget.offset, Query.physical_index.offset) != (
                    12, 24, 56, 4, 16, 48):
                raise OSError("unsupported WDDM structure ABI")
            self.w, self.Luid, self.OpenAdapter, self.AdapterInfo = w, Luid, OpenAdapter, AdapterInfo
            self.kernel.GetProcessId.argtypes, self.kernel.GetProcessId.restype = [w.HANDLE], w.DWORD
            self.kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
            self.kernel.WaitForSingleObject.restype = w.DWORD
            for name, structure in (("D3DKMTOpenAdapterFromLuid", OpenAdapter),
                                    ("D3DKMTQueryAdapterInfo", AdapterInfo)):
                function = getattr(self.gdi, name)
                function.argtypes, function.restype = [ctypes.POINTER(structure)], w.LONG

    @staticmethod
    def _check_status(status: int, operation: str) -> None:
        if status != 0:
            raise OSError(f"{operation}:NTSTATUS=0x{status & 0xFFFFFFFF:08x}")

    def open_adapter(self, luid_hex: str) -> int:
        request = self.OpenAdapter(self.Luid.from_buffer_copy(bytes.fromhex(luid_hex)), 0)
        self._check_status(self.gdi.D3DKMTOpenAdapterFromLuid(ctypes.byref(request)), "open_adapter")
        if not request.adapter:
            raise OSError("KMT returned a null adapter handle")
        return int(request.adapter)

    def physical_adapter_count(self, adapter: int) -> int:
        count = self.w.UINT()
        # KMTQAITYPE_PHYSICALADAPTERCOUNT = 30 in the pinned Microsoft SDK.
        request = self.AdapterInfo(adapter, 30, ctypes.addressof(count), ctypes.sizeof(count))
        self._check_status(self.gdi.D3DKMTQueryAdapterInfo(ctypes.byref(request)), "physical_adapter_count")
        return int(count.value)

    def check_process(self, handle: Any, pid: int, creation: int) -> None:
        observed_pid = int(self.kernel.GetProcessId(handle))
        if not observed_pid:
            raise ctypes.WinError(ctypes.get_last_error())
        times = [self.w.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        observed_birth = times[0].dwLowDateTime | times[0].dwHighDateTime << 32
        if observed_pid != pid or observed_birth != creation:
            raise ValueError("retained process PID or birth identity changed")
        status = int(self.kernel.WaitForSingleObject(handle, 0))
        if status == 0:
            raise ValueError("retained process retired before or during GPU observation")
        if status != 0x102:
            raise ctypes.WinError(ctypes.get_last_error())

    def query_segment(self, process: Any, adapter: int, node: int, segment: int) -> dict[str, int]:
        # Borrowed handle: no OpenProcess, duplication, or process CloseHandle.
        request = self.Query(process, adapter, segment, 0, 0, 0, 0, node)
        self._check_status(self.gdi.D3DKMTQueryVideoMemoryInfo(ctypes.byref(request)), "query_video_memory")
        return {"current_usage_bytes": int(request.usage), "budget_bytes": int(request.budget),
                "current_reservation_bytes": int(request.reservation),
                "available_for_reservation_bytes": int(request.available)}

    def close_adapter(self, adapter: int) -> None:
        self._check_status(self.gdi.D3DKMTCloseAdapter(ctypes.byref(self.Close(adapter))), "close_adapter")

    def resolve_gpu(self, expected: dict[str, Any], *, expected_cuda_ordinal: int | None = None) -> dict[str, Any]:
        """Bind CUDA LUID/node; optional ordinal check preserves the default observer format."""
        if expected_cuda_ordinal is not None and (
            type(expected_cuda_ordinal) is not int or not 0 <= expected_cuda_ordinal < (1 << 31)
        ):
            raise ValueError("an explicit nonnegative CUDA ordinal is required")
        import pynvml as nvml

        nvml.nvmlInit()
        try:
            handle = nvml.nvmlDeviceGetHandleByUUID(expected["uuid"])
            pci = nvml.nvmlDeviceGetPciInfo(handle).busId
            name = nvml.nvmlDeviceGetName(handle)
            pci = pci.decode() if isinstance(pci, bytes) else pci
            name = name.decode() if isinstance(name, bytes) else name
            if normalize_pci_bus_id(pci) != normalize_pci_bus_id(expected["pci_bus_id"]):
                raise ValueError("NVML UUID/PCI identity mismatch")
            if hashlib.sha256(name.strip().encode()).hexdigest() != expected["name_sha256"]:
                raise ValueError("NVML GPU name identity changed")
        finally:
            nvml.nvmlShutdown()
        directory = ctypes.create_unicode_buffer(32768)
        self.kernel.GetSystemDirectoryW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
        if not self.kernel.GetSystemDirectoryW(directory, len(directory)):
            raise ctypes.WinError(ctypes.get_last_error())
        path = Path(directory.value) / "nvcuda.dll"
        cuda = ctypes.WinDLL(str(path))
        for function, arguments in (
            ("cuInit", [ctypes.c_uint]),
            ("cuDeviceGetByPCIBusId", [ctypes.POINTER(ctypes.c_int), ctypes.c_char_p]),
            ("cuDeviceGetLuid", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_int]),
            ("cuDeviceGetName", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]),
            ("cuDriverGetVersion", [ctypes.POINTER(ctypes.c_int)]),
        ):
            getattr(cuda, function).argtypes = arguments
            getattr(cuda, function).restype = ctypes.c_int

        def check(status: int) -> None:
            if status != 0:
                raise OSError(f"CUDA driver identity query failed: {status}")

        check(cuda.cuInit(0))
        device, node, version = ctypes.c_int(), ctypes.c_uint(), ctypes.c_int()
        check(cuda.cuDeviceGetByPCIBusId(ctypes.byref(device), expected["pci_bus_id"].encode("ascii")))
        if expected_cuda_ordinal is not None:
            # CUdevice is a handle: validate ordinal through the driver, not its numeric value.
            cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            cuda.cuDeviceGet.restype = ctypes.c_int
            configured_device = ctypes.c_int()
            check(cuda.cuDeviceGet(ctypes.byref(configured_device), expected_cuda_ordinal))
            if configured_device.value != device.value:
                raise ValueError("resolved CUDA ordinal differs from the configured native device")
        luid, name = ctypes.create_string_buffer(8), ctypes.create_string_buffer(256)
        check(cuda.cuDeviceGetLuid(luid, ctypes.byref(node), device))
        check(cuda.cuDeviceGetName(name, len(name), device))
        check(cuda.cuDriverGetVersion(ctypes.byref(version)))
        if hashlib.sha256(name.value.decode().strip().encode()).hexdigest() != expected["name_sha256"]:
            raise ValueError("CUDA and NVML GPU identity disagree")
        if node.value == 0 or node.value & (node.value - 1):
            raise ValueError("CUDA physical adapter node mask is not uniquely resolved")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 << 20), b""):
                digest.update(block)
        return {
            **({"cuda_device_ordinal": expected_cuda_ordinal, "cuda_node_mask": node.value,
                "name_sha256": expected["name_sha256"]} if expected_cuda_ordinal is not None else {}),
            "uuid": expected["uuid"],
            "pci_bus_id": expected["pci_bus_id"],
            "adapter_luid_hex": luid.raw.hex(),
            "physical_adapter_index": node.value.bit_length() - 1,
            "cuda_driver_version": version.value,
            "observer_driver_dll": str(path),
            "observer_driver_dll_sha256": digest.hexdigest(),
            "binding_source": "NVML UUID/PCI to CUDA cuDeviceGetByPCIBusId/cuDeviceGetLuid to WDDM LUID",
        }

    def query(self, pid: int, creation: int, binding: dict[str, Any]) -> dict[str, int]:
        from ctypes import wintypes as w

        process = self.kernel.OpenProcess(0x400, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        adapters = (self.Adapter * 64)()
        enumeration = self.Enumeration(64, adapters)
        enumerated = False
        try:
            times = [w.FILETIME() for _ in range(4)]
            if not self.kernel.GetProcessTimes(process, *(ctypes.byref(value) for value in times)):
                raise ctypes.WinError(ctypes.get_last_error())
            observed = times[0].dwLowDateTime | times[0].dwHighDateTime << 32
            if observed != creation:
                raise ValueError("owned process creation time changed; PID reuse is refused")
            status = self.gdi.D3DKMTEnumAdapters2(ctypes.byref(enumeration))
            if status != 0 or enumeration.count > len(adapters):
                raise OSError(f"D3DKMT adapter enumeration failed: {status}")
            enumerated = True
            matches = [
                adapter
                for adapter in adapters[: enumeration.count]
                if ctypes.string_at(ctypes.byref(adapter.luid), 8).hex() == binding["adapter_luid_hex"]
            ]
            if len(matches) != 1:
                raise ValueError("CUDA LUID does not identify exactly one WDDM adapter")
            result = {}
            for segment, prefix in ((0, "local"), (1, "nonlocal")):
                query = self.Query(process, matches[0].handle, segment, 0, 0, 0, 0, binding["physical_adapter_index"])
                status = self.gdi.D3DKMTQueryVideoMemoryInfo(ctypes.byref(query))
                if status != 0:
                    raise OSError(f"D3DKMT memory query failed: {status}")
                result[prefix + "_current_usage_bytes"] = int(query.usage)
                result[prefix + "_budget_bytes"] = int(query.budget)
            exit_code = w.DWORD()
            if not self.kernel.GetExitCodeProcess(process, ctypes.byref(exit_code)) or exit_code.value != 259:
                raise ValueError("owned native process exited during observation")
            return result
        finally:
            if enumerated:
                for adapter in adapters[: enumeration.count]:
                    self.gdi.D3DKMTCloseAdapter(ctypes.byref(self.Close(adapter.handle)))
            self.kernel.CloseHandle(process)
