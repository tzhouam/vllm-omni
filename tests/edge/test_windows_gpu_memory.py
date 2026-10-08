# SPDX-License-Identifier: Apache-2.0
"""The read-only observer must never turn unknown or reused identities into zero."""

import copy
import hashlib

import pytest

from vllm_omni.edge.windows_gpu_memory import (
    ProcessGpuPeakTracker,
    WindowsProcessGpuObserver,
    normalize_pci_bus_id,
)


def identity():
    return {
        "status": "verified",
        "pid": 12,
        "creation_filetime_100ns": 133000000000000000,
        "worker_generation": "owned-generation",
        "gpu": {
            "uuid": "GPU-abc",
            "pci_bus_id": "00000000:01:00.0",
            "name_sha256": hashlib.sha256(b"test GPU").hexdigest(),
        },
    }


class Transport:
    def __init__(self):
        self.local, self.shared = 123, 44
        self.creation = identity()["creation_filetime_100ns"]
        self.binding_error = self.query_error = None
        self.calls = []

    def resolve_gpu(self, expected):
        self.calls.append(("bind", expected))
        if self.binding_error:
            raise self.binding_error
        return {"adapter_luid_hex": "0100000000000000", "physical_adapter_index": 0}

    def query(self, pid, creation, binding):
        self.calls.append(("query", pid, creation, binding))
        if creation != self.creation:
            raise ValueError("PID reused")
        if self.query_error:
            raise self.query_error
        return {
            "local_current_usage_bytes": self.local,
            "nonlocal_current_usage_bytes": self.shared,
            "local_budget_bytes": 999,
            "nonlocal_budget_bytes": 333,
        }


def test_exact_process_and_gpu_binding_keep_shared_ram_separate():
    transport = Transport()
    row = WindowsProcessGpuObserver(identity(), transport=transport).sample()
    assert row["status"] == "observed"
    assert row["local_current_usage_bytes"] == 123
    assert row["nonlocal_current_usage_bytes"] == 44
    assert row["hard_cap_verified"] is False
    assert transport.calls[-1][1:3] == (12, identity()["creation_filetime_100ns"])
    assert "total" not in row


@pytest.mark.parametrize("error", [PermissionError("query denied"), ValueError("LUID mismatch"), OSError("NTSTATUS")])
def test_query_failures_are_unknown_and_never_zero(error):
    transport = Transport()
    transport.query_error = error
    row = WindowsProcessGpuObserver(identity(), transport=transport).sample()
    assert row["status"] == "unknown"
    assert row["local_current_usage_bytes"] is None
    assert row["nonlocal_current_usage_bytes"] is None


def test_pid_reuse_after_binding_refuses_measurement():
    transport = Transport()
    observer = WindowsProcessGpuObserver(identity(), transport=transport)
    transport.creation += 1
    assert observer.sample()["status"] == "unknown"
    assert "PID reused" in observer.sample()["reason"]


def test_gpu_binding_failure_does_not_query_another_adapter():
    transport = Transport()
    transport.binding_error = ValueError("NVML CUDA PCI mismatch")
    row = WindowsProcessGpuObserver(identity(), transport=transport).sample()
    assert row["status"] == "unknown"
    assert len(transport.calls) == 1


@pytest.mark.parametrize("field", ["pid", "creation_filetime_100ns", "worker_generation", "gpu", "status"])
def test_incomplete_owner_identity_never_probes(field):
    value = identity()
    value.pop(field)
    transport = Transport()
    assert WindowsProcessGpuObserver(value, transport=transport).sample()["status"] == "unknown"
    assert transport.calls == []


def test_invalid_partial_or_negative_values_are_unknown():
    transport = Transport()
    transport.local = -1
    row = WindowsProcessGpuObserver(identity(), transport=transport).sample()
    assert row["status"] == "unknown"
    assert row["local_current_usage_bytes"] is None


def test_sampled_peaks_never_merge_recovered_generations_or_add_shared_ram():
    transport = Transport()
    observer = WindowsProcessGpuObserver(identity(), transport=transport)
    peaks = ProcessGpuPeakTracker()
    peaks.add(observer.sample())
    transport.local, transport.shared = 100, 60
    peaks.add(observer.sample())
    transport.query_error = PermissionError("unknown")
    peaks.add(observer.sample())
    second = copy.deepcopy(identity())
    second.update(pid=13, worker_generation="reloaded")
    transport.query_error = None
    peaks.add(WindowsProcessGpuObserver(second, transport=transport).sample())
    first, second_peak = peaks.snapshot()
    assert first["samples"] == 2
    assert first["sampled_local_peak_bytes"] == 123
    assert first["sampled_nonlocal_peak_bytes"] == 60
    assert second_peak["samples"] == 1
    assert first["hard_cap_verified"] is False


def test_pci_domain_width_does_not_imply_device_ordinal():
    assert normalize_pci_bus_id("00000000:01:00.0") == normalize_pci_bus_id("0000:01:00.0")
    assert normalize_pci_bus_id("0000:02:00.0") != normalize_pci_bus_id("0000:01:00.0")
    with pytest.raises(ValueError):
        normalize_pci_bus_id("cuda:0")


def test_constructor_detaches_caller_identity_after_gpu_binding():
    value = identity()
    original = copy.deepcopy(value)
    transport = Transport()
    observer = WindowsProcessGpuObserver(value, transport=transport)
    value["status"] = "unverified"
    value["gpu"]["uuid"] = "GPU-another-device"
    row = observer.sample()
    assert row["status"] == "observed"
    assert row["owner_identity"] == original
    assert transport.calls[-1][1:3] == (original["pid"], original["creation_filetime_100ns"])


@pytest.mark.parametrize("mutation", ["status", "gpu", "pid", "creation_filetime_100ns", "worker_generation"])
def test_internal_identity_mutation_is_refused_before_process_query(mutation):
    transport = Transport()
    observer = WindowsProcessGpuObserver(identity(), transport=transport)
    if mutation == "gpu":
        observer.identity["gpu"]["uuid"] = "GPU-another-device"
    elif mutation in {"pid", "creation_filetime_100ns"}:
        observer.identity[mutation] += 1
    else:
        observer.identity[mutation] = "changed"
    row = observer.sample()
    assert row["status"] == "unknown"
    assert row["local_current_usage_bytes"] is None
    assert len(transport.calls) == 1, "mutated identity must not reach a process query"


def test_emitted_observation_and_peak_snapshots_are_independent_values():
    observer = WindowsProcessGpuObserver(identity(), transport=Transport())
    row = observer.sample()
    original = copy.deepcopy(row)
    peaks = ProcessGpuPeakTracker()
    peaks.add(row)
    row["owner_identity"]["gpu"]["uuid"] = "GPU-mutated-returned-row"
    row["gpu_binding"]["adapter_luid_hex"] = "mutated-binding"
    again = observer.sample()
    assert again["owner_identity"] == original["owner_identity"]
    assert again["gpu_binding"] == original["gpu_binding"]
    snapshot = peaks.snapshot()
    assert snapshot[0]["owner_identity"] == original["owner_identity"]
    snapshot[0]["owner_identity"]["gpu"]["uuid"] = "GPU-mutated-peak-snapshot"
    assert peaks.snapshot()[0]["owner_identity"] == original["owner_identity"]


def test_constructor_detaches_transport_owned_binding():
    class AliasedTransport(Transport):
        def resolve_gpu(self, expected):
            self.binding = super().resolve_gpu(expected)
            return self.binding

    transport = AliasedTransport()
    observer = WindowsProcessGpuObserver(identity(), transport=transport)
    before = observer.sample()["gpu_binding"]
    transport.binding["adapter_luid_hex"] = "mutated-transport-binding"
    assert observer.sample()["gpu_binding"] == before


@pytest.mark.parametrize("field", ["adapter_luid_hex", "physical_adapter_index"])
def test_internal_gpu_binding_mutation_is_refused_before_process_query(field):
    transport = Transport()
    observer = WindowsProcessGpuObserver(identity(), transport=transport)
    observer.binding[field] = "mutated" if field == "adapter_luid_hex" else 7
    row = observer.sample()
    assert row["status"] == "unknown"
    assert "GPU binding changed" in row["reason"]
    assert row["local_current_usage_bytes"] is None
    assert len(transport.calls) == 1, "mutated binding must not reach a process query"
