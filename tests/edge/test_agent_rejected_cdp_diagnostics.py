# SPDX-License-Identifier: Apache-2.0
"""Rejected CDP diagnostics; fake handles only, no browser, NN or OS queries."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

import pytest

from vllm_omni.edge import windows_process_memory as memory
from vllm_omni.edge.agent.browser_resources import _ownership_coverage_verified


class DiagnosticCounters:
    def __init__(self) -> None:
        self.rows = {
            20: {"pid": 20, "creation_filetime_100ns": 40, "image_path": r"C:\Edge\msedge.exe"},
            21: {"pid": 21, "creation_filetime_100ns": 41,
                 "image_path": r"C:\Windows\System32\candidate-helper.exe"},
        }
        self.handles: dict[int, dict[str, Any]] = {}
        self.opened: list[int] = []
        self.identified: list[int] = []
        self.waited: list[int] = []
        self.closed: Counter[int] = Counter()
        self.denied = self.identity_error = self.wait_error = self.close_error = False
        self.retired: set[int] = set()

    def open(self, pid: int) -> int:
        self.opened.append(pid)
        if self.denied:
            raise PermissionError("fixture open denied")
        handle = 1000 + len(self.handles)
        self.handles[handle] = dict(self.rows[pid])
        return handle

    def identity(self, handle: int) -> dict[str, Any]:
        self.identified.append(handle)
        if self.identity_error:
            raise OSError("fixture identity query failed")
        return dict(self.handles[handle])

    def wait(self, handle: int, timeout_ms: int) -> dict[str, Any]:
        self.waited.append(handle)
        if self.wait_error:
            raise OSError("fixture wait failed")
        signaled = self.handles[handle]["pid"] in self.retired
        return {"signaled": signaled, "exit_code": 0 if signaled else None}

    def memory(self, handle: int) -> dict[str, int]:
        return {"working_set_bytes": 1, "private_commit_bytes": 2, "peak_working_set_bytes": 3}

    def close(self, handle: int) -> None:
        self.closed[handle] += 1
        if self.close_error:
            raise OSError("fixture CloseHandle failed")


def _registry(api: DiagnosticCounters, **kwargs: Any) -> memory.WindowsProcessMemoryRegistry:
    return memory.WindowsProcessMemoryRegistry("rejected-diagnostic-fixture", transport=api, **kwargs)


def _cdp(registry: memory.WindowsProcessMemoryRegistry, *, pid: int = 21,
         label: str = "winrt_app_id.mojom.WinrtAppIdService", cutoff: int = 100) -> None:
    registry.browser_checkpoint("cdp_membership", {
        "process_info": [{"id": pid, "type": label}], "cutoff_filetime_100ns": cutoff,
    })


def _detail(registry: memory.WindowsProcessMemoryRegistry) -> dict[str, Any]:
    return registry.sample()["coverage"]["unknown_details"][0]


def test_rejected_image_records_only_existing_exact_query_then_one_close() -> None:
    api = DiagnosticCounters()
    api.rows[21].update(command_line="secret arguments", environment={"SECRET": "private"})
    reg = _registry(api)
    _cdp(reg)
    detail = _detail(reg)
    diagnostic = detail["rejected_cdp_candidate"]
    assert detail["reason"] == "binding_failed:ValueError:CDP process image is not the owned Edge runtime"
    assert detail["cdp_process_type"] == "winrt_app_id.mojom.WinrtAppIdService"
    assert diagnostic["schema"] == "omni-rejected-cdp-candidate-v1"
    assert diagnostic["failed_check"] == "edge_runtime_image"
    assert diagnostic["requested_pid"] == diagnostic["observed_pid"] == 21
    assert diagnostic["creation_filetime_100ns"] == 41 and diagnostic["cutoff_filetime_100ns"] == 100
    assert diagnostic["image_path"] == api.rows[21]["image_path"]
    assert diagnostic["image_path_sha256"] == hashlib.sha256(api.rows[21]["image_path"].encode("utf-8")).hexdigest()
    assert diagnostic["image_path_hash_encoding"] == "utf-8"
    assert diagnostic["query_handle_obtained"] is True and diagnostic["identity_query_returned"] is True
    assert diagnostic["liveness_check_attempted"] is False
    assert diagnostic["signaled_at_existing_liveness_check"] is None
    assert diagnostic["query_handle_close_attempted"] is True and diagnostic["query_handle_closed"] is True
    assert api.opened == [21] and api.identified == [1000] and api.waited == []
    assert api.closed == {1000: 1} and reg.sample()["processes"] == []
    serialized = json.dumps(detail, ensure_ascii=False)
    assert "command_line" not in serialized and "secret arguments" not in serialized and "SECRET" not in serialized
    reg.close()
    assert api.closed == {1000: 1}


@pytest.mark.parametrize("failure,check,identity_observed,liveness_attempted,signaled", [
    ("pid", "source_pid_and_positive_birth", True, False, None),
    ("cutoff", "pre_cdp_birth_cutoff", True, False, None),
    ("identity", "query_existing_handle_identity", False, False, None),
    ("wait", "existing_pre_adoption_liveness", True, True, None),
    ("retired", "existing_pre_adoption_liveness", True, True, True),
])
def test_rejection_phase_uses_original_checks_without_extra_queries(
    failure: str, check: str, identity_observed: bool, liveness_attempted: bool, signaled: bool | None,
) -> None:
    api = DiagnosticCounters()
    api.rows[21]["image_path"] = r"C:\Edge\msedge.exe"
    if failure == "pid":
        api.rows[21]["pid"] = 22
    elif failure == "cutoff":
        api.rows[21]["creation_filetime_100ns"] = 101
    elif failure == "identity":
        api.identity_error = True
    elif failure == "wait":
        api.wait_error = True
    else:
        api.retired.add(21)
    reg = _registry(api)
    _cdp(reg)
    diagnostic = _detail(reg)["rejected_cdp_candidate"]
    assert diagnostic["failed_check"] == check
    assert diagnostic["identity_query_returned"] is identity_observed
    assert diagnostic["liveness_check_attempted"] is liveness_attempted
    assert diagnostic["signaled_at_existing_liveness_check"] is signaled
    assert diagnostic["query_handle_closed"] is True
    assert api.opened == [21] and api.identified == [1000] and api.closed == {1000: 1}
    assert len(api.waited) == int(liveness_attempted)
    assert reg.sample()["coverage"]["unknown_count"] == 1
    assert reg.sample()["processes"] == []
    reg.close()


def test_open_failure_has_no_fabricated_identity_handle_close_or_extra_lookup() -> None:
    api = DiagnosticCounters()
    api.denied = True
    reg = _registry(api)
    _cdp(reg)
    diagnostic = _detail(reg)["rejected_cdp_candidate"]
    assert diagnostic["failed_check"] == "open_process_handle"
    assert diagnostic["query_handle_obtained"] is False
    assert diagnostic["identity_query_returned"] is False
    assert diagnostic["observed_pid"] is diagnostic["creation_filetime_100ns"] is None
    assert diagnostic["image_path"] is diagnostic["image_path_sha256"] is None
    assert diagnostic["query_handle_close_attempted"] is False and diagnostic["query_handle_closed"] is None
    assert api.opened == [21] and api.identified == api.waited == [] and api.closed == {}
    reg.close()


def test_failed_close_preserves_original_two_unknowns_and_order() -> None:
    api = DiagnosticCounters()
    api.close_error = True
    reg = _registry(api)
    _cdp(reg)
    coverage = reg.sample()["coverage"]
    assert coverage["unknown_count"] == 2 and coverage["unadopted_handle_close_failures"] == 1
    first, second = coverage["unknown_details"]
    assert first["reason"] == "binding_failed:ValueError:CDP process image is not the owned Edge runtime"
    assert second["reason"] == "unadopted_handle_close_failed:OSError"
    assert "rejected_cdp_candidate" not in second
    diagnostic = first["rejected_cdp_candidate"]
    assert diagnostic["query_handle_close_attempted"] is True and diagnostic["query_handle_closed"] is False
    assert diagnostic["query_handle_close_error_type"] == "OSError"
    assert api.identified == [1000] and api.closed == {1000: 1}
    assert reg.close()["observer_handles_closed"] is False
    assert api.closed == {1000: 1}


@pytest.mark.parametrize("failure", [RuntimeError("format failed"), KeyboardInterrupt(), SystemExit(9)])
def test_optional_formatter_failure_cannot_replace_primary_refusal_or_skip_close(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException,
) -> None:
    api = DiagnosticCounters()
    reg = _registry(api)

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(memory, "_rejected_cdp_metadata", fail)
    _cdp(reg)
    detail = _detail(reg)
    assert detail["reason"] == "binding_failed:ValueError:CDP process image is not the owned Edge runtime"
    assert detail["rejected_cdp_candidate"]["diagnostic_error_type"] == type(failure).__name__
    assert detail["rejected_cdp_candidate"]["query_handle_closed"] is True
    assert reg.sample()["coverage"]["unknown_count"] == 1
    assert api.opened == [21] and api.identified == [1000] and api.closed == {1000: 1}
    reg.close()


def test_optional_close_bookkeeping_failure_runs_only_after_required_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenBookkeeping(dict):
        def update(self, *_args: Any, **_kwargs: Any) -> None:
            raise KeyboardInterrupt()

    api = DiagnosticCounters()
    reg = _registry(api)
    original = memory._rejected_cdp_metadata

    def broken(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return BrokenBookkeeping(original(*args, **kwargs))

    monkeypatch.setattr(memory, "_rejected_cdp_metadata", broken)
    _cdp(reg)
    assert api.closed == {1000: 1}
    assert _detail(reg)["reason"] == "binding_failed:ValueError:CDP process image is not the owned Edge runtime"
    diagnostic = _detail(reg)["rejected_cdp_candidate"]
    assert diagnostic["diagnostic_error_type"] == "KeyboardInterrupt"
    assert diagnostic["query_handle_close_attempted"] is True and diagnostic["query_handle_closed"] is True
    assert reg.sample()["coverage"]["unknown_count"] == 1
    reg.close()


@pytest.mark.parametrize("image,encoding,truncated", [
    (r"C:\中文\candidate.exe", "utf-8", False),
    ("C:\\" + "界" * 1100 + "\\candidate.exe", "utf-8", True),
    ("C:\\bad\ud800\\candidate.exe", "utf-16-le-surrogatepass", True),
    ("C:\\" + "x" * 5000 + "\\candidate.exe", None, True),
])
def test_image_diagnostics_are_bounded_serializable_and_hash_encoding_is_explicit(
    image: str, encoding: str | None, truncated: bool,
) -> None:
    api = DiagnosticCounters()
    api.rows[21]["image_path"] = image
    reg = _registry(api)
    _cdp(reg)
    detail = _detail(reg)
    diagnostic = detail["rejected_cdp_candidate"]
    assert len(diagnostic["image_path"]) <= 1024
    assert diagnostic["image_path_hash_encoding"] == encoding
    assert diagnostic["image_path_truncated"] is truncated
    if encoding == "utf-8":
        expected = hashlib.sha256(image.encode("utf-8")).hexdigest()
        assert diagnostic["image_path_sha256"] == expected
    elif encoding == "utf-16-le-surrogatepass":
        expected = hashlib.sha256(image.encode("utf-16-le", errors="surrogatepass")).hexdigest()
        assert diagnostic["image_path_sha256"] == expected
        assert diagnostic["image_path_representation"] == "escaped_prefix_only"
    else:
        assert diagnostic["image_path_sha256"] is None
    json.dumps(detail, ensure_ascii=False).encode("utf-8")
    assert api.identified == [1000] and api.closed == {1000: 1}
    reg.close()


def test_invalid_out_of_uint64_fields_are_unavailable_without_changing_refusal() -> None:
    api = DiagnosticCounters()
    api.rows[21]["creation_filetime_100ns"] = 1 << 65
    reg = _registry(api)
    _cdp(reg, cutoff=(1 << 65) + 1)
    detail = _detail(reg)
    assert detail["reason"] == "binding_failed:ValueError:CDP process image is not the owned Edge runtime"
    assert detail["rejected_cdp_candidate"]["creation_filetime_100ns"] is None
    assert detail["rejected_cdp_candidate"]["cutoff_filetime_100ns"] is None
    assert api.closed == {1000: 1}
    reg.close()


def test_unknown_detail_limit_and_snapshots_keep_original_counts_and_copy_isolation() -> None:
    api = DiagnosticCounters()
    reg = _registry(api, max_unknown_details=1)
    for _ in range(3):
        _cdp(reg)
    coverage = reg.sample()["coverage"]
    assert coverage["unknown_count"] == 3 and len(coverage["unknown_details"]) == 1
    assert coverage["unknown_details_truncated"] is True
    coverage["unknown_details"][0]["rejected_cdp_candidate"]["query_handle_closed"] = False
    assert _detail(reg)["rejected_cdp_candidate"]["query_handle_closed"] is True
    assert api.opened == [21, 21, 21] and api.identified == [1000, 1001, 1002]
    assert api.closed == {1000: 1, 1001: 1, 1002: 1}
    reg.close()


def test_accepted_and_repeated_retained_paths_are_not_reopened_or_reclassified() -> None:
    api = DiagnosticCounters()
    api.rows[21]["image_path"] = r"C:\Edge\msedge.exe"
    reg = _registry(api)
    _cdp(reg)
    _cdp(reg)
    # One adoption identity plus the unchanged retained sample in each CDP
    # owner checkpoint. Diagnostics add no new query or reopened process.
    assert api.opened == [21] and api.identified == [1000, 1000, 1000] and api.closed == {}
    _cdp(reg, label="different service label")
    view = reg.sample()
    assert len(view["processes"]) == 1 and view["coverage"]["unknown_count"] == 1
    detail = view["coverage"]["unknown_details"][0]
    assert detail["reason"] == "existing_identity_unavailable:ValueError"
    assert "rejected_cdp_candidate" not in detail and api.opened == [21]
    api.retired.add(21)
    reg.close()
    assert api.closed == {1000: 1}


def test_non_cdp_refusal_does_not_gain_cdp_diagnostics() -> None:
    api = DiagnosticCounters()
    reg = _registry(api)
    assert not reg.bind_model({"pid": 21, "creation_filetime_100ns": 42, "worker_generation": "fake-v1"})
    assert "rejected_cdp_candidate" not in _detail(reg)
    assert api.closed == {1000: 1}
    reg.close()


def test_rejection_still_blocks_companion_ownership_even_after_finite_bound_set_drains() -> None:
    api = DiagnosticCounters()
    reg = _registry(api)
    reg.browser_checkpoint("browser_launch_started", {})
    _cdp(reg, pid=20, label="browser")
    _cdp(reg)
    api.retired.add(20)
    children = reg.close_children(timeout_seconds=0)
    assert children["required_owned_bindings_verified"] is True
    assert children["bound_set_drain_verified"] is True
    assert children["coverage"]["unknown_count"] == 1
    assert children["all_descendants_retired"] is False
    assert _ownership_coverage_verified(children) is False
    assert children["coverage"]["unknown_details"][0]["rejected_cdp_candidate"]["query_handle_closed"] is True
    reg.close()
    assert api.closed == {1000: 1, 1001: 1}
