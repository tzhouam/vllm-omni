# SPDX-License-Identifier: Apache-2.0
"""Identity-only pool join contracts. All API responses below are synthetic."""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from vllm_omni.edge import windows_gpu_memory as gpu
from vllm_omni.edge.agent import native_app as app
from vllm_omni.edge.windows_bound_gpu_memory import WindowsRetainedProcessGpuProvider

UUID = "GPU-11111111-2222-3333-4444-555555555555"
NAME = "synthetic discrete GPU"
DISCRETE, UMA = "0200000000000000", "0100000000000000"


def identity():
    return {"uuid": UUID, "pci_bus_id": "00000000:64:00.0",
            "name_sha256": hashlib.sha256(NAME.encode()).hexdigest()}


def inventory():
    def adapter(luid, uma, count):
        return {"description": NAME if not uma else "synthetic UMA", "vendor_id": 4318,
            "device_id": 1, "gpu_memory_accounting": {"available": True,
                "source": "native_DXGI_GetDesc1_D3D12_Architecture1", "adapter_luid_hex": luid,
                "physical_adapter_count": count, "nodes": [{"physical_adapter_index": index,
                    "uma": uma, "cache_coherent_uma": False} for index in range(count)]}}
    return [adapter(UMA, True, 1), adapter(DISCRETE, False, 2)]


class AdapterApi:
    def __init__(self):
        self.closed = []

    def open_adapter(self, luid):
        return luid

    def physical_adapter_count(self, handle):
        return 1 if handle == UMA else 2

    def close_adapter(self, handle):
        self.closed.append(handle)


def provider():
    api = AdapterApi()
    return WindowsRetainedProcessGpuProvider(inventory(), transport=api), api


def binding(ordinal=0):
    return {**identity(), "adapter_luid_hex": DISCRETE, "physical_adapter_index": 1,
        "cuda_device_ordinal": ordinal, "cuda_node_mask": 2, "cuda_driver_version": 13030,
        "observer_driver_dll": r"C:\Windows\System32\nvcuda.dll",
        "observer_driver_dll_sha256": "b" * 64,
        "binding_source": "NVML UUID/PCI to CUDA cuDeviceGetByPCIBusId/cuDeviceGetLuid to WDDM LUID"}


def arguments():
    value, _api = provider()
    capabilities = value.capabilities()
    value.close()
    return {"gpu_identity": identity(), "gpu_index": 0, "gpu_pool": "vram",
        "ledger_capacities": {"host_ram": 100, "vram": 200}, "cuda_binding": binding(),
        "provider_capabilities": capabilities}


def rehash(capability):
    capability["capability_sha256"] = hashlib.sha256(json.dumps(capability["adapter_nodes"],
        sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")).hexdigest()


@pytest.mark.parametrize("pool", ["vram", "vram:0"])
def test_exact_linked_node_and_ledger_alias_join_without_budget_or_polling(monkeypatch, pool):
    args = arguments()
    args["gpu_pool"] = pool
    args["ledger_capacities"] = {"host_ram": 100, pool: 200}
    monkeypatch.setattr(gpu, "_WindowsTransport", lambda: pytest.fail("pure join made a native call"))
    joined = gpu.validate_native_gpu_pool_binding(**args)
    assert joined["ledger_pool"] == pool
    assert joined["physical_pool_identity"] == "wddm_local:" + DISCRETE + ":1"
    assert joined["pool_aliases"] == ["vram", "vram:0"]
    assert joined["provider_node"]["physical_adapter_index"] == 1
    assert joined["nonlocal_physical_pool_alias"] == "host_ram"
    assert joined["memory_budget_selected"] is joined["memory_admission_performed"] is False
    assert joined["native_execution_placement_verified"] is joined["qualification"] is False


def test_cuda_ordinal_and_node_index_are_distinct_and_node_order_is_not_identity():
    args = arguments()
    args.update(gpu_index=3, gpu_pool="vram:3", ledger_capacities={"vram:3": 200})
    args["cuda_binding"] = binding(ordinal=3)
    args["provider_capabilities"]["adapter_nodes"].reverse()
    rehash(args["provider_capabilities"])
    joined = gpu.validate_native_gpu_pool_binding(**args)
    assert joined["cuda_binding"]["cuda_device_ordinal"] == 3
    assert joined["provider_node"]["physical_adapter_index"] == 1


@pytest.mark.parametrize("field,value", [
    ("uuid", "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"), ("pci_bus_id", "0000:65:00.0"),
    ("name_sha256", "c" * 64), ("cuda_device_ordinal", 1), ("cuda_device_ordinal", False),
    ("cuda_node_mask", 0), ("cuda_node_mask", 3), ("cuda_node_mask", True),
    ("physical_adapter_index", 0), ("physical_adapter_index", True),
    ("adapter_luid_hex", "0300000000000000"), ("observer_driver_dll_sha256", "bad"),
])
def test_incomplete_or_mismatched_native_binding_is_refused(field, value):
    args = arguments()
    args["cuda_binding"][field] = value
    with pytest.raises(ValueError):
        gpu.validate_native_gpu_pool_binding(**args)


@pytest.mark.parametrize("pools,pool,index", [
    ({"vram": 1, "vram:0": 1}, "vram", 0), ({}, "vram", 0),
    ({"vram": 1}, "vram:0", 0), ({"vram": True}, "vram", 0),
    ({"vram": 1}, "vram", 1), ({"vram:1": 1}, "vram:1", 0),
])
def test_aliases_cannot_duplicate_capacity_or_select_another_ledger_device(pools, pool, index):
    args = arguments()
    args.update(ledger_capacities=pools, gpu_pool=pool, gpu_index=index)
    with pytest.raises(ValueError, match="ledger"):
        gpu.validate_native_gpu_pool_binding(**args)


@pytest.mark.parametrize("mutation", ["closed", "unavailable", "hash", "duplicate", "missing_node", "wrong_pool"])
def test_unknown_or_ambiguous_provider_snapshot_refuses_join(mutation):
    args = arguments()
    capability = args["provider_capabilities"]
    if mutation == "closed":
        capability["closed"] = True
    elif mutation == "unavailable":
        capability["binding_available"] = False
    elif mutation == "hash":
        capability["capability_sha256"] = "0" * 64
    elif mutation == "duplicate":
        capability["adapter_nodes"].append(copy.deepcopy(capability["adapter_nodes"][-1]))
        rehash(capability)
    elif mutation == "missing_node":
        capability["adapter_nodes"].pop(1)
        rehash(capability)
    else:
        capability["adapter_nodes"][-1]["local_physical_pool_alias"] = "host_ram"
        rehash(capability)
    with pytest.raises(ValueError):
        gpu.validate_native_gpu_pool_binding(**args)


def test_uma_node_cannot_be_turned_into_additional_vram():
    args = arguments()
    args["cuda_binding"].update(adapter_luid_hex=UMA, physical_adapter_index=0, cuda_node_mask=1)
    with pytest.raises(ValueError, match="discrete"):
        gpu.validate_native_gpu_pool_binding(**args)


def test_join_detaches_inputs_and_returned_metadata():
    args = arguments()
    original = copy.deepcopy(args)
    joined = gpu.validate_native_gpu_pool_binding(**args)
    args["gpu_identity"]["uuid"] = "changed"
    args["cuda_binding"]["physical_adapter_index"] = 0
    args["provider_capabilities"]["adapter_nodes"][-1]["uma"] = True
    assert joined["gpu_identity"] == original["gpu_identity"]
    assert joined["cuda_binding"] == original["cuda_binding"]
    joined["provider_node"]["uma"] = True
    assert gpu.validate_native_gpu_pool_binding(**original)["provider_node"]["uma"] is False


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def synthetic_resolver(monkeypatch, tmp_path, *, ordinal=0, device_handle=71, mask=2, ordinal_status=0):
    calls = []
    (tmp_path / "nvcuda.dll").write_bytes(b"synthetic native identity fixture")
    def directory(buffer, _size):
        buffer.value = str(tmp_path)
        return len(str(tmp_path))
    def by_pci(pointer, pci):
        calls.append(("pci", pci))
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = device_handle
        return 0
    def by_ordinal(pointer, requested):
        calls.append(("ordinal", requested))
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = (
            device_handle if requested == ordinal else device_handle + 1)
        return ordinal_status
    def luid(buffer, pointer, device):
        calls.append(("luid", device.value))
        ctypes.memmove(buffer, bytes.fromhex(DISCRETE), 8)
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint))[0] = mask
        return 0
    def name(buffer, _size, _device):
        buffer.value = NAME.encode()
        return 0
    def version(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 13030
        return 0
    cuda = SimpleNamespace(cuInit=Function(lambda _flags: 0), cuDeviceGetByPCIBusId=Function(by_pci),
        cuDeviceGet=Function(by_ordinal),
        cuDeviceGetLuid=Function(luid), cuDeviceGetName=Function(name), cuDriverGetVersion=Function(version))
    nvml = SimpleNamespace(nvmlInit=lambda: calls.append("nvml_init"),
        nvmlShutdown=lambda: calls.append("nvml_shutdown"),
        nvmlDeviceGetHandleByUUID=lambda uuid: uuid,
        nvmlDeviceGetPciInfo=lambda _handle: SimpleNamespace(busId=identity()["pci_bus_id"]),
        nvmlDeviceGetName=lambda _handle: NAME)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(ctypes, "WinDLL", lambda _path: cuda, raising=False)
    value = object.__new__(gpu._WindowsTransport)
    value.kernel = SimpleNamespace(GetSystemDirectoryW=Function(directory))
    return value, calls


def test_identity_resolver_opt_in_records_actual_ordinal_mask_and_default_is_unchanged(monkeypatch, tmp_path):
    value, calls = synthetic_resolver(monkeypatch, tmp_path, ordinal=3, mask=2)
    row = value.resolve_gpu(identity(), expected_cuda_ordinal=3)
    assert row["cuda_device_ordinal"] == 3 and row["cuda_node_mask"] == 2
    assert row["physical_adapter_index"] == 1
    assert row["name_sha256"] == identity()["name_sha256"]
    default = value.resolve_gpu(identity())
    assert not {"cuda_device_ordinal", "cuda_node_mask", "name_sha256"} & default.keys()
    assert [entry for entry in calls if isinstance(entry, tuple) and entry[0] == "luid"] == [
        ("luid", 71), ("luid", 71)]
    assert [entry for entry in calls if isinstance(entry, tuple) and entry[0] == "ordinal"] == [
        ("ordinal", 3)]  # Default path does not add the optional ordinal query.


def test_resolver_refuses_nvml_cuda_ordinal_reordering_before_luid_query(monkeypatch, tmp_path):
    value, calls = synthetic_resolver(monkeypatch, tmp_path, ordinal=1)
    with pytest.raises(ValueError, match="ordinal"):
        value.resolve_gpu(identity(), expected_cuda_ordinal=0)
    assert "nvml_shutdown" in calls
    assert not any(isinstance(row, tuple) and row[0] == "luid" for row in calls)


def test_configured_ordinal_native_failure_refuses_before_luid_query(monkeypatch, tmp_path):
    value, calls = synthetic_resolver(monkeypatch, tmp_path, ordinal_status=101)
    with pytest.raises(OSError, match="identity query failed: 101"):
        value.resolve_gpu(identity(), expected_cuda_ordinal=0)
    assert ("ordinal", 0) in calls
    assert not any(isinstance(row, tuple) and row[0] == "luid" for row in calls)


@pytest.mark.parametrize("invalid", [True, -1, 1 << 31, "0", 0.0])
def test_invalid_configured_ordinal_refuses_before_nvml(monkeypatch, tmp_path, invalid):
    value, calls = synthetic_resolver(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="ordinal"):
        value.resolve_gpu(identity(), expected_cuda_ordinal=invalid)
    assert calls == []


@pytest.mark.parametrize("mask", [0, 3])
def test_resolver_refuses_unknown_or_multiple_physical_nodes(monkeypatch, tmp_path, mask):
    value, _calls = synthetic_resolver(monkeypatch, tmp_path, mask=mask)
    with pytest.raises(ValueError, match="node mask"):
        value.resolve_gpu(identity(), expected_cuda_ordinal=0)


@pytest.mark.parametrize("change", [False, True])
def test_app_preparation_uses_exact_owned_provider_and_identity_only_resolver(monkeypatch, change):
    value, api = provider()
    calls = []
    class Resolver:
        def resolve_gpu(self, expected, *, expected_cuda_ordinal):
            calls.append((copy.deepcopy(expected), expected_cuda_ordinal))
            if change:
                value.close()
            return binding()
        def query(self, *args):
            pytest.fail("app metadata seam invoked legacy process polling")
    monkeypatch.setattr(gpu, "_WindowsTransport", Resolver)
    kwargs = dict(gpu_provider=value, gpu_identity=identity(), gpu_index=0, gpu_pool="vram",
                  ledger_capacities={"vram": 200})
    if change:
        with pytest.raises(ValueError, match="changed"):
            app._resolve_native_gpu_pool_identity(**kwargs)
    else:
        row = app._resolve_native_gpu_pool_identity(**kwargs)
        assert row["physical_pool_identity"] == "wddm_local:" + DISCRETE + ":1"
        assert api.closed == []  # Helper never takes/releases factory ownership.
        value.close()
    assert calls == [(identity(), 0)]


def test_app_refuses_non_native_or_closed_provider_without_resolver(monkeypatch):
    monkeypatch.setattr(gpu, "_WindowsTransport", lambda: pytest.fail("unavailable provider reached resolver"))
    kwargs = dict(gpu_identity=identity(), gpu_index=0, gpu_pool="vram", ledger_capacities={"vram": 200})
    with pytest.raises(ValueError, match="exact app-owned"):
        app._resolve_native_gpu_pool_identity(gpu_provider=SimpleNamespace(), **kwargs)
    value, _api = provider()
    value.close()
    with pytest.raises(ValueError, match="unavailable"):
        app._resolve_native_gpu_pool_identity(gpu_provider=value, **kwargs)
