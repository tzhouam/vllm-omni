# SPDX-License-Identifier: Apache-2.0
"""Opt-in borrowed-handle GPU accounting; synthetic counters, no native probes."""

from __future__ import annotations

import copy
import ctypes
import hashlib
import threading
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.edge.windows_bound_gpu_memory import WindowsRetainedProcessGpuProvider
from vllm_omni.edge.windows_gpu_memory import _WindowsTransport
from vllm_omni.edge.windows_process_memory import WindowsProcessMemoryRegistry


def inventory() -> list[dict[str, Any]]:
    return [
        {"description": "synthetic UMA", "vendor_id": 0x1002, "device_id": 1,
         "gpu_memory_accounting": {"available": True,
             "source": "native_DXGI_GetDesc1_D3D12_Architecture1",
             "adapter_luid_hex": "0100000000000000", "physical_adapter_count": 1,
             "nodes": [{"physical_adapter_index": 0, "uma": True, "cache_coherent_uma": True}]}},
        {"description": "synthetic linked discrete", "vendor_id": 0x10DE, "device_id": 2,
         "gpu_memory_accounting": {"available": True,
             "source": "native_DXGI_GetDesc1_D3D12_Architecture1",
             "adapter_luid_hex": "0200000000000000", "physical_adapter_count": 2,
             "nodes": [{"physical_adapter_index": 0, "uma": False, "cache_coherent_uma": False},
                       {"physical_adapter_index": 1, "uma": False, "cache_coherent_uma": False}]}},
    ]


def identity() -> dict[str, Any]:
    return {"cohort_generation": "browser-fixture", "role": "edge_browser", "pid": 20,
            "creation_filetime_100ns": 40, "image_path_sha256": hashlib.sha256(b"Edge path").hexdigest()}


class GpuApi:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.counts = {101: 1, 102: 2}
        self.open_errors: set[str] = set()
        self.query_errors: set[tuple[int, int, int]] = set()
        self.close_errors: set[int] = set()
        self.open_interruptions: dict[str, BaseException] = {}
        self.count_interruptions: dict[int, BaseException] = {}
        self.close_interruptions: dict[int, BaseException] = {}
        self.check_error: Exception | None = None
        self.retire_after_query = False
        self.queried = False
        self.invalid_counter: Any = None

    def open_adapter(self, luid: str) -> int:
        self.calls.append(("open_adapter", luid))
        if luid in self.open_errors:
            raise PermissionError("adapter denied")
        if luid in self.open_interruptions:
            raise self.open_interruptions[luid]
        return 101 if luid == "0100000000000000" else 102

    def physical_adapter_count(self, handle: int) -> int:
        self.calls.append(("physical_adapter_count", handle))
        if handle in self.count_interruptions:
            raise self.count_interruptions[handle]
        return self.counts[handle]

    def check_process(self, handle: Any, pid: int, creation: int) -> None:
        self.calls.append(("check_process", handle, pid, creation))
        if self.check_error is not None:
            raise self.check_error
        if self.retire_after_query and self.queried:
            raise ValueError("retained process retired")

    def query_segment(self, process: Any, adapter: int, node: int, segment: int) -> dict[str, int]:
        self.calls.append(("query_segment", process, adapter, node, segment))
        self.queried = True
        if (adapter, node, segment) in self.query_errors:
            raise PermissionError("segment denied")
        return {"current_usage_bytes": 7 if self.invalid_counter is None else self.invalid_counter,
                "budget_bytes": 100, "current_reservation_bytes": 0, "available_for_reservation_bytes": 100}

    def close_adapter(self, handle: int) -> None:
        self.calls.append(("close_adapter", handle))
        if handle in self.close_interruptions:
            raise self.close_interruptions[handle]
        if handle in self.close_errors:
            raise OSError("adapter close failed")


class ProcessApi:
    def __init__(self) -> None:
        self.process = {"pid": 20, "creation_filetime_100ns": 40, "image_path": r"C:\Edge\msedge.exe"}
        self.handles: dict[int, dict[str, Any]] = {}
        self.opens: list[int] = []
        self.closes: Counter[int] = Counter()
        self.retired = False
        self.ram_denied = False
        self.events: list[str] = []

    def filetime_now(self) -> int:
        return 100

    def open(self, pid: int) -> int:
        self.opens.append(pid)
        assert pid == 20
        self.handles[777] = dict(self.process)
        return 777

    def identity(self, handle: int) -> dict[str, Any]:
        return dict(self.handles[handle])

    def wait(self, handle: int, timeout_ms: int) -> dict[str, Any]:
        return {"signaled": self.retired, "exit_code": 259 if self.retired else None}

    def memory(self, handle: int) -> dict[str, int]:
        if self.ram_denied:
            raise PermissionError("RAM counter denied")
        return {"working_set_bytes": 11, "private_commit_bytes": 13, "peak_working_set_bytes": 17}

    def close(self, handle: int) -> None:
        self.events.append("process_handle_close")
        self.closes[handle] += 1


def registry(api: ProcessApi, provider: Any = None) -> WindowsProcessMemoryRegistry:
    reg = WindowsProcessMemoryRegistry("browser-fixture", transport=api, gpu_provider=provider)
    reg.browser_checkpoint("cdp_membership", {"cutoff_filetime_100ns": 100,
        "process_info": [{"type": "browser", "id": 20}]})
    return reg


def test_multi_adapter_node_binding_is_once_and_shared_pools_are_aliases() -> None:
    api = GpuApi()
    original = inventory()
    provider = WindowsRetainedProcessGpuProvider(original, transport=api)
    original[0]["gpu_memory_accounting"]["nodes"][0]["uma"] = False
    row = provider.sample_process(777, identity())
    assert row["available"] and row["observed_segment_count"] == 6
    uma, first, second = row["adapter_nodes"]
    assert uma["local_physical_pool_alias"] == "host_ram"
    assert uma["nonlocal_physical_pool_alias"] is None and uma["nonlocal_omni_pool_id"] is None
    assert uma["nonlocal_pool_kind"] == "no_independent_UMA_nonlocal_pool_declared"
    assert first["local_physical_pool_alias"] != second["local_physical_pool_alias"]
    assert first["local_omni_pool_id"] is None
    assert first["nonlocal_omni_pool_id"] == "host_ram"
    assert row["hard_cap_verified"] is False
    assert all(call[1] == 777 for call in api.calls if call[0] == "query_segment")
    provider.sample_process(777, identity())
    assert len([call for call in api.calls if call[0] == "open_adapter"]) == 2
    assert "total" not in row and row["local_nonlocal_and_process_working_set_must_not_be_added"]
    provider.close()


def test_partial_segment_failure_preserves_success_and_is_not_complete_or_zero() -> None:
    api = GpuApi()
    api.query_errors.add((102, 1, 1))
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=api)
    row = provider.sample_process(777, identity())
    assert row["available"] is False and row["status"] == "partial"
    assert row["observed_segment_count"] == 5
    assert row["adapter_nodes"][-1]["local"]["current_usage_bytes"] == 7
    assert row["adapter_nodes"][-1]["nonlocal"]["current_usage_bytes"] is None
    provider.close()


@pytest.mark.parametrize("error", [PermissionError("access denied"), ValueError("PID or birth mismatch"),
                                   OSError("process wait failed")])
def test_process_check_failure_never_queries_or_reports_zero(error: Exception) -> None:
    api = GpuApi()
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=api)
    api.check_error = error
    row = provider.sample_process(777, identity())
    assert row["available"] is False and row["adapter_nodes"] == []
    assert not any(call[0] == "query_segment" for call in api.calls)
    provider.close()


def test_process_exit_during_queries_invalidates_pre_exit_complete_observation() -> None:
    api = GpuApi()
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=api)
    api.retire_after_query = True
    row = provider.sample_process(777, identity())
    assert row["available"] is False and row["adapter_nodes"] == []
    assert row["reason"].startswith("process_check_after_query:")
    provider.close()


@pytest.mark.parametrize("bad_counter", [-1, True, 1 << 64, "7"])
def test_invalid_counters_are_unavailable(bad_counter: Any) -> None:
    api = GpuApi()
    api.invalid_counter = bad_counter
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=api)
    row = provider.sample_process(777, identity())
    assert row["observed_segment_count"] == 0 and row["available"] is False
    assert all(node["local"]["current_usage_bytes"] is None for node in row["adapter_nodes"])
    provider.close()


@pytest.mark.parametrize("mutation", ["duplicate_luid", "unknown_uma", "missing_node", "wrong_node_index",
                                      "over_bound", "unknown_capability", "wrong_source"])
def test_unavailable_or_ambiguous_native_capabilities_do_not_open_adapters(mutation: str) -> None:
    value = inventory()
    if mutation == "duplicate_luid":
        value[1]["gpu_memory_accounting"]["adapter_luid_hex"] = value[0]["gpu_memory_accounting"]["adapter_luid_hex"]
    elif mutation == "unknown_uma":
        value[0]["gpu_memory_accounting"]["nodes"][0]["uma"] = None
    elif mutation == "missing_node":
        value[1]["gpu_memory_accounting"]["nodes"].pop()
    elif mutation == "wrong_node_index":
        value[1]["gpu_memory_accounting"]["nodes"][1]["physical_adapter_index"] = 0
    elif mutation == "over_bound":
        value *= 9
    elif mutation == "unknown_capability":
        value[0]["gpu_memory_accounting"]["available"] = False
    else:
        value[0]["gpu_memory_accounting"]["source"] = "caller_declaration"
    api = GpuApi()
    provider = WindowsRetainedProcessGpuProvider(value, transport=api)
    assert provider.sample_process(777, identity())["available"] is False
    assert api.calls == []
    assert provider.close()["drained"]


def test_kmt_node_count_mismatch_closes_every_opened_adapter_once() -> None:
    api = GpuApi()
    api.counts[102] = 1
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=api)
    assert provider.capabilities()["binding_available"] is False
    assert provider.sample_process(777, identity())["adapter_nodes"] == []
    provider.close()
    provider.close()
    assert [call for call in api.calls if call[0] == "close_adapter"] == [
        ("close_adapter", 101), ("close_adapter", 102)]


def test_partial_adapter_open_failure_closes_prior_resource_and_keeps_binding_unavailable() -> None:
    api = GpuApi()
    api.open_errors.add("0200000000000000")
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=api)
    assert provider.capabilities()["binding_available"] is False
    assert provider.close()["adapter_handle_count"] == 1
    assert [call for call in api.calls if call[0] == "close_adapter"] == [("close_adapter", 101)]


def test_close_is_once_immutable_and_all_future_queries_are_unavailable() -> None:
    api = GpuApi()
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=api)
    receipt = provider.close()
    frozen = copy.deepcopy(receipt)
    receipt["drained"] = False
    assert provider.close() == frozen and frozen["drained"]
    before = len(api.calls)
    assert provider.sample_process(777, identity())["reason"] == "provider_closed"
    assert len(api.calls) == before


def test_registry_default_has_no_gpu_sampling_or_capability_metadata() -> None:
    process = ProcessApi()
    reg = registry(process)
    row = reg.sample()
    assert "gpu_memory_capabilities" not in row
    assert "gpu_memory" not in row["processes"][0]
    process.retired = True
    assert "gpu_sampling_close" not in reg.close_children(timeout_seconds=0)
    assert process.closes == {777: 1}


def test_registry_borrows_existing_handle_and_still_samples_gpu_when_ram_counter_fails() -> None:
    process, gpu = ProcessApi(), GpuApi()
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=gpu)
    reg = registry(process, provider)
    process.ram_denied = True
    row = reg.sample()["processes"][0]
    assert row["available"] is False and row["gpu_memory"]["available"] is True
    assert process.opens == [20]
    assert all(call[1] == 777 for call in gpu.calls if call[0] == "query_segment")
    process.retired = True
    reg.close_children(timeout_seconds=0)


def test_registry_identity_mismatch_and_retired_reused_pid_cannot_reach_gpu_query() -> None:
    process, gpu = ProcessApi(), GpuApi()
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=gpu)
    reg = registry(process, provider)
    gpu.calls.clear()  # Owner checkpoint legitimately sampled the original handle.
    process.handles[777]["creation_filetime_100ns"] = 99
    row = reg.sample()["processes"][0]
    assert row["gpu_memory"]["available"] is False
    assert not any(call[0] == "query_segment" for call in gpu.calls)
    process.handles[777]["creation_filetime_100ns"] = 40
    process.process["creation_filetime_100ns"] = 99  # A new process reused the numeric PID.
    process.retired = True
    row = reg.sample()["processes"][0]
    assert row["creation_filetime_100ns"] == 40 and row["gpu_memory"]["available"] is False
    assert process.opens == [20]
    reg.close_children(timeout_seconds=0)


def test_owner_checkpoint_uses_same_provider_without_a_second_polling_path() -> None:
    process, gpu = ProcessApi(), GpuApi()
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=gpu)
    reg = registry(process, provider)
    view = reg.sample()
    checkpoints = view["owner_checkpoint_samples"]
    assert len(checkpoints) == 1 and checkpoints[0]["processes"][0]["gpu_memory"]["available"]
    assert view["gpu_memory_capabilities"]["polling_threads_created"] == 0
    process.retired = True
    reg.close_children(timeout_seconds=0)


def test_provider_drain_and_adapter_close_precede_process_handle_close() -> None:
    process, gpu = ProcessApi(), GpuApi()
    original_close = gpu.close_adapter

    def close_adapter(handle: int) -> None:
        process.events.append("adapter_close")
        original_close(handle)

    gpu.close_adapter = close_adapter
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=gpu)
    reg = registry(process, provider)
    process.retired = True
    receipt = reg.close_children(timeout_seconds=0)
    assert process.events == ["adapter_close", "adapter_close", "process_handle_close"]
    assert receipt["gpu_sampling_close"]["drained"] and receipt["bound_set_drain_verified"]
    reg.close_children(timeout_seconds=0)
    assert process.closes == {777: 1}


def test_adapter_close_failure_remains_quarantined_even_after_known_process_retirement() -> None:
    process, gpu = ProcessApi(), GpuApi()
    gpu.close_errors.add(101)
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=gpu)
    reg = registry(process, provider)
    process.retired = True
    receipt = reg.close_children(timeout_seconds=0)
    assert receipt["known_bound_children_retired"] is True
    assert receipt["gpu_sampling_close"]["drained"] is True
    assert receipt["gpu_sampling_close"]["adapter_handles_closed"] is False
    assert receipt["quarantine_required"] and not receipt["bound_set_drain_verified"]
    assert process.closes == {777: 1}


@pytest.mark.parametrize("failure", ["raise", "malformed", "undrained"])
def test_unverified_provider_drain_retains_process_objects_in_quarantine(failure: str) -> None:
    process, gpu = ProcessApi(), GpuApi()
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=gpu)

    def failed_close() -> dict[str, Any]:
        if failure == "raise":
            raise RuntimeError("drain failed")
        return {} if failure == "malformed" else {"drained": False, "adapter_handles_closed": False}

    provider.close = failed_close
    reg = registry(process, provider)
    process.retired = True
    receipt = reg.close_children(timeout_seconds=0)
    assert receipt["quarantine_required"] and not receipt["bound_set_drain_verified"]
    assert process.closes == {} and reg._bindings[20].handle == 777
    assert receipt["processes"][0]["gpu_sampling_drain_unverified"]


def test_provider_close_waits_for_synchronous_query_without_owning_process_handle() -> None:
    api = GpuApi()
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original_query = api.query_segment
    errors: list[BaseException] = []

    def blocked_query(*args: Any) -> dict[str, int]:
        if not entered.is_set():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test query was not released")
        return original_query(*args)

    api.query_segment = blocked_query
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=api)

    def sample() -> None:
        try:
            assert provider.sample_process(777, identity())["available"]
        except BaseException as exc:
            errors.append(exc)

    def close() -> None:
        try:
            assert provider.close()["drained"]
        except BaseException as exc:
            errors.append(exc)
        finally:
            closed.set()

    worker = threading.Thread(target=sample)
    closer = threading.Thread(target=close)
    worker.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert not closed.wait(0.01)
        assert not any(call[0] == "close_adapter" for call in api.calls)
    finally:
        release.set()
        worker.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not worker.is_alive() and not closer.is_alive() and errors == []
    assert closed.is_set()
    assert [call[0] for call in api.calls][-2:] == ["close_adapter", "close_adapter"]


@pytest.mark.parametrize("change", ["pid", "birth", "retired", "wait_failed"])
def test_actual_retained_native_identity_guard_uses_borrowed_handle_only(change: str, monkeypatch: Any) -> None:
    class Filetime(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]

    calls: list[tuple[str, int]] = []

    class Kernel:
        def GetProcessId(self, handle: int) -> int:
            calls.append(("pid", handle))
            return 21 if change == "pid" else 20

        def GetProcessTimes(self, handle: int, birth: Any, *other: Any) -> bool:
            calls.append(("times", handle))
            birth._obj.dwLowDateTime = 41 if change == "birth" else 40
            return True

        def WaitForSingleObject(self, handle: int, timeout: int) -> int:
            calls.append(("wait", handle))
            return 0 if change == "retired" else (0xFFFFFFFF if change == "wait_failed" else 0x102)

    transport = object.__new__(_WindowsTransport)
    transport.w, transport.kernel = SimpleNamespace(FILETIME=Filetime), Kernel()
    monkeypatch.setattr(ctypes, "WinError", lambda code: OSError("synthetic wait failure"), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    with pytest.raises((ValueError, OSError)):
        transport.check_process(777, 20, 40)
    assert calls and all(handle == 777 for _, handle in calls)
    assert not hasattr(transport.kernel, "OpenProcess")


def test_optional_capability_report_failure_does_not_replace_cpu_counters() -> None:
    process, gpu = ProcessApi(), GpuApi()
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=gpu)
    reg = registry(process, provider)

    def unavailable_capability() -> dict[str, Any]:
        raise PermissionError("capability metadata unavailable")

    provider.capabilities = unavailable_capability
    row = reg.sample()
    assert row["processes"][0]["working_set_bytes"] == 11
    assert row["gpu_memory_capabilities"]["binding_available"] is False
    process.retired = True
    reg.close_children(timeout_seconds=0)


def test_missing_native_capability_api_does_not_fall_back_to_another_adapter() -> None:
    gpu = GpuApi()
    gpu.physical_adapter_count = None
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=gpu)
    row = provider.sample_process(777, identity())
    assert row["available"] is False and row["adapter_nodes"] == []
    assert gpu.calls == [("open_adapter", "0100000000000000"), ("close_adapter", 101)]
    assert provider.close()["drained"]


def test_adapter_binding_success_cannot_erase_unknown_browser_membership() -> None:
    process, gpu = ProcessApi(), GpuApi()
    process.process["image_path"] = r"C:\Edge\identity_helper.exe"
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=gpu)
    reg = registry(process, provider)
    view = reg.sample()
    assert view["processes"] == [] and view["coverage"]["unknown_count"] == 1
    assert view["gpu_memory_capabilities"]["binding_available"] is True
    assert view["coverage"]["all_descendants_covered"] is False
    assert not any(call[0] == "query_segment" for call in gpu.calls)
    receipt = reg.close_children(timeout_seconds=0)
    assert receipt["coverage"]["unknown_count"] == 1


@pytest.mark.parametrize("interrupted_handle", [101, 102])
@pytest.mark.parametrize("interruption_type", [KeyboardInterrupt, SystemExit])
def test_interrupted_adapter_close_records_failure_and_never_retries_or_turns_green(
        interrupted_handle: int, interruption_type: type[BaseException]) -> None:
    gpu = GpuApi()
    interruption = interruption_type("synthetic close cancellation")
    gpu.close_interruptions[interrupted_handle] = interruption
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=gpu)
    with pytest.raises(interruption_type) as caught:
        provider.close()
    assert caught.value is interruption
    receipt = provider.close()
    assert receipt["drained"] is True and receipt["adapter_handles_closed"] is False
    assert receipt["adapter_close_attempt_count"] == 2 and receipt["verified_adapter_close_count"] == 1
    luid = "0100000000000000" if interrupted_handle == 101 else "0200000000000000"
    assert receipt["adapter_close_outcomes"][luid]["verified_closed"] is False
    assert receipt["adapter_close_outcomes"][luid]["error_type"] == interruption_type.__name__
    receipt["adapter_handles_closed"] = True
    assert provider.close()["adapter_handles_closed"] is False
    assert [call for call in gpu.calls if call[0] == "close_adapter"] == [
        ("close_adapter", 101), ("close_adapter", 102)]


@pytest.mark.parametrize("phase", ["first_count", "second_open", "second_count"])
@pytest.mark.parametrize("interruption_type", [KeyboardInterrupt, SystemExit])
def test_constructor_cancellation_cleans_every_already_opened_adapter_once(
        phase: str, interruption_type: type[BaseException]) -> None:
    gpu = GpuApi()
    interruption = interruption_type("synthetic constructor cancellation")
    if phase == "second_open":
        gpu.open_interruptions["0200000000000000"] = interruption
    else:
        gpu.count_interruptions[101 if phase == "first_count" else 102] = interruption
    # Keep the partial instance only to inspect cleanup; normal construction
    # raises and never exposes this object to a production caller.
    provider = object.__new__(WindowsRetainedProcessGpuProvider)
    with pytest.raises(interruption_type) as caught:
        provider.__init__(inventory(), transport=gpu)
    assert caught.value is interruption
    expected = [("close_adapter", 101)]
    if phase == "second_count":
        expected.append(("close_adapter", 102))
    assert [call for call in gpu.calls if call[0] == "close_adapter"] == expected
    assert provider.close()["adapter_handles_closed"] is True
    assert [call for call in gpu.calls if call[0] == "close_adapter"] == expected


def test_constructor_original_cancellation_survives_secondary_cleanup_interruption() -> None:
    gpu = GpuApi()
    original, secondary = KeyboardInterrupt("original bind"), SystemExit("secondary close")
    gpu.count_interruptions[101] = original
    gpu.close_interruptions[101] = secondary
    provider = object.__new__(WindowsRetainedProcessGpuProvider)
    with pytest.raises(KeyboardInterrupt) as caught:
        provider.__init__(inventory(), transport=gpu)
    assert caught.value is original
    receipt = provider.close()
    assert receipt["adapter_handles_closed"] is False and receipt["verified_adapter_close_count"] == 0
    assert receipt["adapter_close_outcomes"]["0100000000000000"]["error_type"] == "SystemExit"
    assert [call for call in gpu.calls if call[0] == "close_adapter"] == [("close_adapter", 101)]


def test_registry_does_not_close_process_objects_after_provider_close_cancellation() -> None:
    process, gpu = ProcessApi(), GpuApi()
    interruption = KeyboardInterrupt("provider close cancellation")
    gpu.close_interruptions[101] = interruption
    provider = WindowsRetainedProcessGpuProvider(inventory(), transport=gpu)
    reg = registry(process, provider)
    process.retired = True
    with pytest.raises(KeyboardInterrupt) as caught:
        reg.close_children(timeout_seconds=0)
    assert caught.value is interruption
    assert process.closes == {} and reg._bindings[20].handle == 777
    assert reg._children_receipt is None  # No verified registry close may be invented.
    assert provider.close()["adapter_handles_closed"] is False
