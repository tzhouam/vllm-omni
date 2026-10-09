# SPDX-License-Identifier: Apache-2.0
"""Bounded, retained-handle Windows process RAM attribution.

This observer neither discovers descendants nor enforces a memory cap. CDP
membership is an observed set, not OS ancestry. Working set includes shared
pages; private commit is not physical RAM. Sampled maxima are lower bounds.
No polling thread, process termination, or admission policy belongs here.
"""

from __future__ import annotations

import ctypes
import hashlib
import ntpath
import os
import threading
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.edge.windows_cdp_helper import CdpObservedHelperCapability

PROCESS_MEMORY_SCHEMA = "omni-windows-bound-process-memory-v1"
_QUERY_RIGHTS = 0x0400 | 0x0010 | 0x00100000  # query information, VM read, synchronize
_EDGE_TYPES = {"browser": "edge_browser", "renderer": "edge_renderer", "GPU": "edge_gpu",
               "utility": "edge_utility", "other": "edge_other"}
_CHILD_ROLES = frozenset({"playwright_node", *_EDGE_TYPES.values()})
_CDP_TYPE_MAX_UTF8_BYTES = 128
_REJECTED_CDP_IMAGE_MAX_CHARS = 1024
_PROCESS_IMAGE_IDENTITY_MAX_CHARS = 4096
_NO_CDP_TYPE = object()


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _valid_process_id(value: Any) -> bool:
    # FILETIME and spawn HANDLE validation still use the generic positive helper.
    return _positive_int(value) and value <= 0xFFFFFFFF


def _bounded_cdp_type(value: Any) -> str | None:
    # CDP ProcessInfo.type is an open string, not a protocol enum. Bound before
    # encoding, and retain the exact label without case folding or aliasing.
    if (type(value) is not str or not 1 <= len(value) <= _CDP_TYPE_MAX_UTF8_BYTES
            or not value.strip() or not value.isprintable()):
        return None
    try:
        return value if len(value.encode("utf-8")) <= _CDP_TYPE_MAX_UTF8_BYTES else None
    except UnicodeEncodeError:
        return None


def _cdp_type_metadata(value: Any) -> dict[str, Any]:
    label = _bounded_cdp_type(value)
    if label is not None:
        return {"cdp_process_type_valid": True, "cdp_process_type": label,
                "cdp_process_type_sha256": hashlib.sha256(label.encode("utf-8")).hexdigest(),
                "cdp_process_type_known": label in _EDGE_TYPES}
    result: dict[str, Any] = {"cdp_process_type_valid": False,
                              "cdp_process_type_value_type": type(value).__name__[:32]}
    if type(value) is str:
        # Malformed strings are diagnostics only. Escape a bounded prefix so
        # lone surrogates/control bytes cannot break later UTF-8 publication.
        result["cdp_process_type_escaped_prefix"] = value[:64].encode("unicode_escape").decode("ascii")[:128]
        result["cdp_process_type_prefix_only"] = True
    return result


def _rejected_cdp_metadata(identity: Any, *, requested_pid: int, cutoff: Any,
                           failed_check: str, handle_obtained: bool,
                           liveness_attempted: bool, liveness: Any) -> dict[str, Any]:
    """Bounded diagnostics from one existing handle query, never authority.

    Rejected candidates are not bindings. A path hash identifies observed path
    text, not executable contents or verified ownership. No PID lookup, scan,
    additional identity/wait query, command line, environment or file read is
    performed here. The caller separately preserves every original refusal.
    """
    def filetime(value: Any) -> int | None:
        return value if type(value) is int and 0 < value <= 0xFFFFFFFFFFFFFFFF else None

    row: dict[str, Any] = {
        "schema": "omni-rejected-cdp-candidate-v1",
        "scope": "already_open_handle_diagnostic_not_ownership_or_retirement_proof",
        "requested_pid": requested_pid, "failed_check": failed_check,
        "cutoff_filetime_100ns": filetime(cutoff),
        "query_handle_obtained": handle_obtained,
        "identity_query_returned": isinstance(identity, Mapping),
        "observed_pid": None, "creation_filetime_100ns": None,
        "image_path": None, "image_path_sha256": None,
        "image_path_hash_encoding": None, "image_path_representation": None,
        "image_path_truncated": False,
        "liveness_check_attempted": liveness_attempted,
        "signaled_at_existing_liveness_check": None,
        "query_handle_close_attempted": False, "query_handle_closed": None,
        "query_handle_close_error_type": None,
    }
    if isinstance(liveness, Mapping) and type(liveness.get("signaled")) is bool:
        row["signaled_at_existing_liveness_check"] = liveness["signaled"]
    if not isinstance(identity, Mapping):
        return row
    pid = identity.get("pid")
    row["observed_pid"] = pid if _valid_process_id(pid) else None
    row["creation_filetime_100ns"] = filetime(identity.get("creation_filetime_100ns"))
    image = identity.get("image_path")
    if type(image) is not str:
        return row
    if len(image) > _PROCESS_IMAGE_IDENTITY_MAX_CHARS:
        # An unexpected transport can supply over-bound data. Do not hash or
        # serialize its complete contents, and do not claim a full path hash.
        row["image_path"] = image[:128].encode("unicode_escape").decode("ascii")[:1024]
        row["image_path_representation"] = "escaped_prefix_only"
        row["image_path_truncated"] = True
        return row
    try:
        raw = image.encode("utf-8")
    except UnicodeEncodeError:
        raw = image.encode("utf-16-le", errors="surrogatepass")
        row["image_path_hash_encoding"] = "utf-16-le-surrogatepass"
        row["image_path"] = image[:128].encode("unicode_escape").decode("ascii")[:1024]
        row["image_path_representation"] = "escaped_prefix_only"
        row["image_path_truncated"] = True
    else:
        row["image_path_hash_encoding"] = "utf-8"
        row["image_path"] = image[:_REJECTED_CDP_IMAGE_MAX_CHARS]
        row["image_path_representation"] = "observed_text"
        row["image_path_truncated"] = len(image) > _REJECTED_CDP_IMAGE_MAX_CHARS
    row["image_path_sha256"] = hashlib.sha256(raw).hexdigest()
    return row


class _Win32ProcessCounters:
    """Only query/read-counter/synchronize handles; never termination access."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows retained process handles are unavailable")
        from ctypes import wintypes as w

        self.w = w
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel.GetCurrentProcess.restype = w.HANDLE
        self.kernel.GetCurrentProcessId.restype = w.DWORD
        self.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.DuplicateHandle.argtypes = [w.HANDLE, w.HANDLE, w.HANDLE,
                                               ctypes.POINTER(w.HANDLE), w.DWORD, w.BOOL, w.DWORD]
        self.kernel.DuplicateHandle.restype = w.BOOL
        self.kernel.GetProcessId.argtypes = [w.HANDLE]
        self.kernel.GetProcessId.restype = w.DWORD
        self.kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
        self.kernel.GetProcessTimes.restype = w.BOOL
        self.kernel.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR,
                                                         ctypes.POINTER(w.DWORD)]
        self.kernel.QueryFullProcessImageNameW.restype = w.BOOL
        self.kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        self.kernel.WaitForSingleObject.restype = w.DWORD
        self.kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        self.kernel.GetExitCodeProcess.restype = w.BOOL
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.CloseHandle.restype = w.BOOL
        self.kernel.GetSystemTimePreciseAsFileTime.argtypes = [ctypes.POINTER(w.FILETIME)]
        self.kernel.GetSystemTimePreciseAsFileTime.restype = None

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", w.DWORD), ("page_fault_count", w.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working_set_bytes", "working_set_bytes", "quota_peak_paged_pool",
                    "quota_paged_pool", "quota_peak_nonpaged_pool", "quota_nonpaged_pool",
                    "pagefile_usage", "peak_pagefile_usage", "private_commit_bytes",
                )
            ]

        self.MemoryCounters = MemoryCounters
        self.psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(MemoryCounters), w.DWORD]
        self.psapi.GetProcessMemoryInfo.restype = w.BOOL

    def current_pid(self) -> int:
        return int(self.kernel.GetCurrentProcessId())

    def filetime_now(self) -> int:
        value = self.w.FILETIME()
        self.kernel.GetSystemTimePreciseAsFileTime(ctypes.byref(value))
        return int(value.dwLowDateTime | value.dwHighDateTime << 32)

    def open(self, pid: int) -> Any:
        handle = self.kernel.OpenProcess(_QUERY_RIGHTS, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def duplicate_spawn_handle(self, spawn_handle: int) -> Any:
        duplicate = self.w.HANDLE()
        current = self.kernel.GetCurrentProcess()
        if not self.kernel.DuplicateHandle(current, spawn_handle, current, ctypes.byref(duplicate),
                                           _QUERY_RIGHTS, False, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        return duplicate

    def identity(self, handle: Any) -> dict[str, Any]:
        pid = int(self.kernel.GetProcessId(handle))
        if not pid:
            raise ctypes.WinError(ctypes.get_last_error())
        times = [self.w.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        size = self.w.DWORD(32768)
        image = ctypes.create_unicode_buffer(size.value)
        if not self.kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not image.value or len(image.value) > 4096:
            raise ValueError("process image identity exceeds the observation bound")
        return {"pid": pid, "creation_filetime_100ns": int(times[0].dwLowDateTime |
                                                            times[0].dwHighDateTime << 32),
                "image_path": image.value}

    def wait(self, handle: Any, timeout_ms: int) -> dict[str, Any]:
        status = int(self.kernel.WaitForSingleObject(handle, timeout_ms))
        if status == 0x102:
            return {"signaled": False, "exit_code": None}
        if status != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = self.w.DWORD()
        if not self.kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"signaled": True, "exit_code": int(code.value)}

    def memory(self, handle: Any) -> dict[str, int]:
        counters = self.MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not self.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {name: int(getattr(counters, name)) for name in (
            "working_set_bytes", "private_commit_bytes", "peak_working_set_bytes")}

    def close(self, handle: Any) -> None:
        if not self.kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


@dataclass
class _Binding:
    role: str
    pid: int
    creation: int
    image_path: str
    source: str
    handle: Any
    first_seen_monotonic_ns: int
    last_seen_monotonic_ns: int
    worker_generation: str | None = None
    cdp_process_type: str | None = None
    helper_capability_sha256: str | None = None
    last_sample: dict[str, Any] = field(default_factory=dict)
    sampled_working_set_max: int | None = None
    sampled_private_commit_max: int | None = None
    sample_count: int = 0
    close_attempted: bool = False
    handle_closed: bool = False
    closure: dict[str, Any] | None = None


class WindowsProcessMemoryRegistry:
    """Exact bound-set observer shared by owner callbacks and telemetry calls.

    A retained handle never retargets a reused PID. Repeated CDP membership
    does not prove ancestry or all descendants. Every failure remains unknown.
    """

    def __init__(self, cohort_generation: str, *, transport: Any = None,
                 max_bindings: int = 128, max_unknown_details: int = 32,
                 gpu_provider: Any = None,
                 cdp_helper_capability: CdpObservedHelperCapability | None = None) -> None:
        if not isinstance(cohort_generation, str) or not 1 <= len(cohort_generation) <= 256:
            raise ValueError("cohort generation is required and bounded")
        if type(max_bindings) is not int or not 1 <= max_bindings <= 128:
            raise ValueError("process binding limit must be between 1 and 128")
        if type(max_unknown_details) is not int or not 1 <= max_unknown_details <= 32:
            raise ValueError("unknown detail limit must be between 1 and 32")
        if gpu_provider is not None and any(not callable(getattr(gpu_provider, name, None))
                for name in ("sample_process", "unavailable", "capabilities", "close")):
            raise ValueError("optional GPU provider contract is incomplete")
        if cdp_helper_capability is not None:
            if type(cdp_helper_capability) is not CdpObservedHelperCapability:
                raise ValueError("an explicit app-owned CDP helper capability is required")
            cdp_helper_capability.verify_named_artifacts()
        self._cdp_helper_capability = cdp_helper_capability
        self._gpu_provider = gpu_provider
        self._gpu_close_receipt: dict[str, Any] | None = None
        self.generation = cohort_generation
        self._transport = transport if transport is not None else _Win32ProcessCounters()
        self._max_bindings = max_bindings
        self._max_unknown_details = max_unknown_details
        self._lock = threading.RLock()
        self._bindings: dict[int, _Binding] = {}
        self._unknown_count = 0
        self._unknown: list[dict[str, Any]] = []
        self._overflow = False
        self._unadopted_handle_close_failures = 0
        self._children_closing = False
        self._closing = False
        self._browser_launch_attempted = False
        self._node_start_attempted = False
        self._node_verified = False
        self._browser_root_verified = False
        self._cdp_checkpoints = 0
        self._checkpoint_samples: list[dict[str, Any]] = []
        self._checkpoint_queue_overflow = False
        self._children_receipt: dict[str, Any] | None = None
        self._receipt: dict[str, Any] | None = None

    def _unknown_event(self, reason: str, *, pid: Any = None, source: str = "observer",
                       count: int = 1, cdp_process_type: Any = _NO_CDP_TYPE,
                       rejected_cdp_candidate: dict[str, Any] | None = None) -> dict[str, Any] | None:
        self._unknown_count += count
        if len(self._unknown) < self._max_unknown_details:
            row = {"reason": reason[:256], "pid": pid if _valid_process_id(pid) else None,
                   "source": source[:64], "count": count,
                   "monotonic_ns": time.monotonic_ns(),
                   **(_cdp_type_metadata(cdp_process_type)
                      if cdp_process_type is not _NO_CDP_TYPE else {})}
            if rejected_cdp_candidate is not None:
                row["rejected_cdp_candidate"] = rejected_cdp_candidate
            self._unknown.append(row)
            # _adopt runs under this registry's RLock. Only its diagnostic
            # CloseHandle outcome is filled before that lock is released;
            # every external sample/receipt returns a locked deepcopy.
            return row
        return None

    def _metadata(self, binding: _Binding) -> dict[str, Any]:
        return {"cohort_generation": self.generation, "role": binding.role, "pid": binding.pid,
                "creation_filetime_100ns": binding.creation,
                "image_path": binding.image_path,
                "image_path_sha256": hashlib.sha256(binding.image_path.encode("utf-8")).hexdigest(),
                "identity_source": binding.source,
                "worker_generation": binding.worker_generation,
                "first_seen_monotonic_ns": binding.first_seen_monotonic_ns,
                "last_seen_monotonic_ns": binding.last_seen_monotonic_ns,
                **({"observed_cdp_helper": self._cdp_helper_capability.metadata()}
                   if binding.helper_capability_sha256 is not None else {}),
                **(_cdp_type_metadata(binding.cdp_process_type)
                   if binding.cdp_process_type is not None else {})}

    def _verify_observed_helper(self, *, pid: int, image: str, label: str | None) -> str:
        """Only the exact app declaration broadens the image check.

        No CDP label, basename, historical receipt or Job absence supplies
        authority. The existing original handle is retained by _adopt; this
        method never opens a PID or grants ancestry/termination rights.
        """
        capability = self._cdp_helper_capability
        roots = [row for row in self._bindings.values()
                 if row.role == "edge_browser" and row.source == "owned_browser_cdp"
                 and row.cdp_process_type == "browser" and not row.close_attempted]
        helpers = [row for row in self._bindings.values() if row.helper_capability_sha256 is not None]
        if (capability is None or len(roots) != 1 or any(row.pid != pid for row in helpers)
                or not capability.matches(browser_image_path=roots[0].image_path,
                                          helper_image_path=image, cdp_process_type=label)):
            raise ValueError("CDP helper does not match its exact declared browser installation and type")
        root = roots[0]
        identity = self._transport.identity(root.handle)
        if (identity["pid"] != root.pid or identity["creation_filetime_100ns"] != root.creation
                or ntpath.normcase(identity["image_path"]) != ntpath.normcase(root.image_path)
                or self._transport.wait(root.handle, 0)["signaled"]):
            raise ValueError("declared helper browser root is no longer a verified live retained object")
        capability.verify_named_artifacts()
        return capability.capability_sha256

    def _adopt(self, *, pid: Any, role: str, source: str, expected_creation: Any = None,
               cutoff: Any = None, spawn_handle: Any = None, expected_image: str | None = None,
               worker_generation: str | None = None, cdp_process_type: str | None = None) -> bool:
        def unknown(reason: str, *, pid: Any = None,
                    diagnostic: dict[str, Any] | None = None) -> dict[str, Any] | None:
            return self._unknown_event(reason, pid=pid, source=source,
                                       cdp_process_type=(cdp_process_type if cdp_process_type is not None
                                                         else _NO_CDP_TYPE),
                                       rejected_cdp_candidate=diagnostic)

        if self._closing or self._children_closing or self._receipt is not None or self._children_receipt is not None:
            unknown("registration_after_closure_refused", pid=pid)
            return False
        if not _valid_process_id(pid):
            unknown("invalid_pid")
            return False
        previous = self._bindings.get(pid)
        if previous is not None:
            duplicate = None
            try:
                retired = self._transport.wait(previous.handle, 0)["signaled"]
                if (retired or previous.close_attempted or previous.role != role or previous.source != source
                        or previous.worker_generation != worker_generation
                        or previous.cdp_process_type != cdp_process_type
                        or (expected_creation is not None and previous.creation != expected_creation)
                        or (cutoff is not None and (not _positive_int(cutoff) or previous.creation > cutoff))
                        or (expected_image is not None and
                            ntpath.normcase(previous.image_path) != ntpath.normcase(expected_image))):
                    raise ValueError("PID reuse or contradictory source identity is refused")
                if previous.helper_capability_sha256 is not None:
                    if previous.helper_capability_sha256 != self._verify_observed_helper(
                            pid=pid, image=previous.image_path, label=cdp_process_type):
                        raise ValueError("retained helper capability identity changed")
                if spawn_handle is not None:
                    duplicate = self._transport.duplicate_spawn_handle(spawn_handle)
                    observed = self._transport.identity(duplicate)
                    if (observed["pid"] != previous.pid or observed["creation_filetime_100ns"] != previous.creation
                            or ntpath.normcase(observed["image_path"]) != ntpath.normcase(previous.image_path)
                            or self._transport.wait(duplicate, 0)["signaled"]):
                        raise ValueError("repeated spawn source does not match retained identity")
                previous.last_seen_monotonic_ns = time.monotonic_ns()
                return True
            except Exception as exc:
                unknown("existing_identity_unavailable:" + type(exc).__name__, pid=pid)
                return False
            except BaseException as exc:
                if self._cdp_helper_capability is not None and source == "owned_browser_cdp":
                    unknown("declared_cdp_validation_interrupted:" + type(exc).__name__, pid=pid)
                raise
            finally:
                if duplicate is not None:
                    try:
                        self._transport.close(duplicate)
                    except Exception as exc:
                        self._unadopted_handle_close_failures += 1
                        unknown("unadopted_handle_close_failed:" + type(exc).__name__, pid=pid)
        if len(self._bindings) >= self._max_bindings:
            self._overflow = True
            unknown("binding_registry_overflow", pid=pid)
            return False
        handle = None
        adopted = False
        identity = liveness = None
        liveness_attempted = False
        failed_check = "open_process_handle"
        rejected_detail = None
        helper_capability_sha256 = None
        try:
            handle = (self._transport.duplicate_spawn_handle(spawn_handle)
                      if spawn_handle is not None else self._transport.open(pid))
            failed_check = "query_existing_handle_identity"
            identity = self._transport.identity(handle)
            birth = identity["creation_filetime_100ns"]
            image = identity["image_path"]
            failed_check = "source_pid_and_positive_birth"
            if identity["pid"] != pid or not _positive_int(birth):
                raise ValueError("retained process identity does not match its source")
            failed_check = "expected_creation_time"
            if expected_creation is not None and (not _positive_int(expected_creation) or birth != expected_creation):
                raise ValueError("creation time mismatch; PID reuse is refused")
            failed_check = "pre_cdp_birth_cutoff"
            if cutoff is not None and (not _positive_int(cutoff) or birth > cutoff):
                raise ValueError("birth is after the pre-CDP cutoff; membership is unverified")
            failed_check = "owned_spawn_image"
            if expected_image is not None and ntpath.normcase(image) != ntpath.normcase(expected_image):
                raise ValueError("process image differs from its owned spawn source")
            failed_check = "edge_runtime_image"
            declared_helper_label = (self._cdp_helper_capability is not None
                                     and cdp_process_type == self._cdp_helper_capability.cdp_process_type)
            if role.startswith("edge_") and (ntpath.basename(image).casefold() != "msedge.exe"
                                            or declared_helper_label):
                if self._cdp_helper_capability is None:
                    raise ValueError("CDP process image is not the owned Edge runtime")
                failed_check = "declared_cdp_helper_capability"
                if source != "owned_browser_cdp" or role != "edge_other":
                    raise ValueError("declared helper requires its exact observed CDP membership")
                helper_capability_sha256 = self._verify_observed_helper(
                    pid=pid, image=image, label=cdp_process_type)
            failed_check = "existing_pre_adoption_liveness"
            liveness_attempted = True
            liveness = self._transport.wait(handle, 0)
            if liveness["signaled"]:
                raise ValueError("process exited before identity adoption")
            failed_check = "retain_verified_binding"
            now = time.monotonic_ns()
            self._bindings[pid] = _Binding(role, pid, birth, image, source, handle, now, now,
                                          worker_generation=worker_generation,
                                          cdp_process_type=cdp_process_type,
                                          helper_capability_sha256=helper_capability_sha256)
            adopted = True
            return True
        except Exception as exc:
            diagnostic = None
            if source == "owned_browser_cdp":
                try:
                    diagnostic = _rejected_cdp_metadata(
                        identity, requested_pid=pid, cutoff=cutoff, failed_check=failed_check,
                        handle_obtained=handle is not None, liveness_attempted=liveness_attempted,
                        liveness=liveness,
                    )
                except BaseException as diagnostic_failure:
                    # Formatting may fail independently. Preserve the exact
                    # primary refusal/count/order and still close its handle.
                    diagnostic = {"schema": "omni-rejected-cdp-candidate-v1",
                        "scope": "diagnostic_formatting_failed_not_ownership_or_retirement_proof",
                        "failed_check": failed_check,
                        "diagnostic_error_type": type(diagnostic_failure).__name__[:64],
                        "query_handle_obtained": handle is not None,
                        "query_handle_close_attempted": False, "query_handle_closed": None,
                        "query_handle_close_error_type": None}
            rejected_detail = unknown("binding_failed:" + type(exc).__name__ + ":" + str(exc)[:128],
                                      pid=pid, diagnostic=diagnostic)
            return False
        except BaseException as exc:
            if self._cdp_helper_capability is not None and source == "owned_browser_cdp":
                unknown("declared_cdp_validation_interrupted:" + type(exc).__name__, pid=pid)
            raise
        finally:
            if handle is not None and not adopted:
                diagnostic = (rejected_detail.get("rejected_cdp_candidate")
                              if rejected_detail is not None else None)
                close_succeeded = False
                close_error_type = None
                try:
                    self._transport.close(handle)
                    close_succeeded = True
                except Exception as exc:
                    close_error_type = type(exc).__name__[:64]
                    self._unadopted_handle_close_failures += 1
                    unknown("unadopted_handle_close_failed:" + type(exc).__name__, pid=pid)
                finally:
                    if diagnostic is not None:
                        try:
                            diagnostic.update(query_handle_close_attempted=True,
                                              query_handle_closed=close_succeeded,
                                              query_handle_close_error_type=close_error_type)
                        except BaseException as diagnostic_failure:
                            # Optional output bookkeeping runs only after the
                            # required close and must not mask either refusal.
                            try:
                                rejected_detail["rejected_cdp_candidate"] = {
                                    "schema": "omni-rejected-cdp-candidate-v1",
                                    "scope": "diagnostic_formatting_failed_not_ownership_or_retirement_proof",
                                    "failed_check": failed_check,
                                    "diagnostic_error_type": type(diagnostic_failure).__name__[:64],
                                    "query_handle_obtained": True,
                                    "query_handle_close_attempted": True,
                                    "query_handle_closed": close_succeeded,
                                    "query_handle_close_error_type": close_error_type,
                                }
                            except BaseException:
                                pass

    def bind_current_agent(self) -> bool:
        with self._lock:
            return self._adopt(pid=self._transport.current_pid(), role="agent",
                               source="current_profiler_process")

    def bind_model(self, identity: Mapping[str, Any] | None) -> bool:
        with self._lock:
            if not isinstance(identity, Mapping) or not _positive_int(identity.get("creation_filetime_100ns")):
                self._unknown_event("backend_owned_model_birth_unavailable", source="backend_loaded_plan")
                return False
            generation = identity.get("worker_generation")
            if not isinstance(generation, str) or not 1 <= len(generation) <= 256:
                self._unknown_event("backend_owned_model_generation_unavailable", source="backend_loaded_plan")
                return False
            return self._adopt(pid=identity.get("pid"), role="model", source="backend_loaded_plan",
                               expected_creation=identity["creation_filetime_100ns"],
                               worker_generation=generation)

    def _queue_owner_checkpoint(self, checkpoint: str) -> None:
        # The owner callback already holds the registry lock. This performs
        # only retained-handle counter queries, never Playwright/CDP calls.
        if len(self._checkpoint_samples) >= 16:
            self._checkpoint_queue_overflow = True
            self._unknown_event("owner_checkpoint_queue_overflow", source="browser_owner_worker")
            self._checkpoint_samples.pop(0)
        self._checkpoint_samples.append({
            "checkpoint": checkpoint[:64],
            "monotonic_ns": time.monotonic_ns(), "wall_time_ns": time.time_ns(),
            "coverage": self._coverage(),
            "processes": deepcopy([self._sample_binding(binding) for binding in self._bindings.values()]),
        })

    def browser_checkpoint(self, action: str, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Called on the browser owner worker, except final drain after its join."""
        if action == "children_closed":
            return self.close_children(timeout_seconds=5.0)
        with self._lock:
            if self._closing or self._children_closing or self._receipt is not None:
                self._unknown_event("owner_callback_after_closure_refused", source="browser_owner_worker")
                return None
            if action == "cdp_begin":
                return {"cutoff_filetime_100ns": self._transport.filetime_now()}
            if action == "node_start_attempted":
                self._node_start_attempted = True
                self._queue_owner_checkpoint("before_node_start")
            elif action == "browser_launch_started":
                self._browser_launch_attempted = True
                self._queue_owner_checkpoint("before_browser_launch")
            elif action == "node_spawn":
                if (not _positive_int(payload.get("spawn_handle"))
                        or not isinstance(payload.get("expected_image"), str)
                        or not ntpath.isabs(payload["expected_image"])
                        or ntpath.basename(payload["expected_image"]).casefold() != "node.exe"
                        or payload.get("playwright_version") != "1.63.0"
                        or payload.get("python_version") != "3.12.10"):
                    self._unknown_event("owned_node_spawn_handle_or_pin_unavailable",
                                        source="playwright_owned_spawn_handle")
                    return None
                self._node_verified = self._adopt(
                    pid=payload.get("pid"), role="playwright_node", source="playwright_owned_spawn_handle",
                    spawn_handle=payload.get("spawn_handle"), expected_image=payload.get("expected_image"),
                )
            elif action == "cdp_membership":
                self._cdp_checkpoints += 1
                rows = payload.get("process_info")
                cutoff = payload.get("cutoff_filetime_100ns")
                if not isinstance(rows, list) or not _positive_int(cutoff):
                    self._unknown_event("invalid_cdp_membership", source="owned_browser_cdp")
                else:
                    if len(rows) > self._max_bindings:
                        self._overflow = True
                        self._unknown_event("cdp_membership_overflow", source="owned_browser_cdp",
                                            count=len(rows) - self._max_bindings)
                    for item in rows[:self._max_bindings]:
                        if not isinstance(item, Mapping):
                            self._unknown_event("malformed_cdp_process_info", source="owned_browser_cdp")
                            continue
                        label, pid = _bounded_cdp_type(item.get("type")), item.get("id")
                        if not _valid_process_id(pid) or label is None:
                            self._unknown_event(
                                "invalid_cdp_process_id" if not _valid_process_id(pid) else "invalid_cdp_process_type",
                                pid=pid, source="owned_browser_cdp", cdp_process_type=item.get("type"))
                            continue
                        role = _EDGE_TYPES.get(label, "edge_other")
                        ok = self._adopt(pid=pid, role=role, source="owned_browser_cdp",
                                         cutoff=cutoff, cdp_process_type=label)
                        if role == "edge_browser" and ok:
                            self._browser_root_verified = True
                self._queue_owner_checkpoint(str(payload.get("checkpoint", "unlabelled")))
            else:
                self._unknown_event(str(payload.get("reason", action))[:256], source="browser_owner_worker")
            return None

    def _sample_binding(self, binding: _Binding) -> dict[str, Any]:
        row = self._metadata(binding)
        row.update(monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(),
                   handle_closed=binding.handle_closed, available=False)
        if binding.close_attempted:
            row.update(last_counters=binding.last_sample, closure=binding.closure)
            return row
        identity_verified = False
        try:
            identity = self._transport.identity(binding.handle)
            if (identity["pid"] != binding.pid or identity["creation_filetime_100ns"] != binding.creation
                    or ntpath.normcase(identity["image_path"]) != ntpath.normcase(binding.image_path)):
                raise ValueError("retained handle identity changed")
            identity_verified = True
            state = self._transport.wait(binding.handle, 0)
            row.update(state)
            if state["signaled"]:
                row["counter_error"] = "process_retired_before_sample"
            else:
                memory = self._transport.memory(binding.handle)
                if any(type(memory.get(key)) is not int or memory[key] < 0 for key in (
                        "working_set_bytes", "private_commit_bytes", "peak_working_set_bytes")):
                    raise ValueError("invalid process memory counters")
                row.update(memory, available=True,
                           peak_working_set_scope="OS_lifetime_process_peak_not_cohort_sample_peak")
                binding.sample_count += 1
                binding.sampled_working_set_max = max(binding.sampled_working_set_max or 0,
                                                      memory["working_set_bytes"])
                binding.sampled_private_commit_max = max(binding.sampled_private_commit_max or 0,
                                                         memory["private_commit_bytes"])
        except Exception as exc:
            row["counter_error"] = type(exc).__name__ + ":" + str(exc)[:192]
        if self._gpu_provider is not None:
            try:
                if identity_verified and row.get("signaled") is False:
                    row["gpu_memory"] = self._gpu_provider.sample_process(
                        binding.handle, self._metadata(binding))
                else:
                    row["gpu_memory"] = self._gpu_provider.unavailable(
                        self._metadata(binding), "registry_identity_or_live_state_unverified")
            except Exception as exc:
                row["gpu_memory"] = {"enabled": True, "available": False,
                    "status": "unavailable", "reason": type(exc).__name__,
                    "missing_measurements_are_not_zero": True, "adapter_nodes": []}
        binding.last_sample = dict(row)
        return row

    def _coverage(self) -> dict[str, Any]:
        return {"scope": "exact_retained_handle_bound_set_only", "all_descendants_covered": False,
                "all_startup_peaks_covered": False, "short_lived_processes_may_be_unobserved": True,
                "cdp_membership_is_not_ancestry": True, "binding_registry_overflow": self._overflow,
                "cdp_process_type_max_utf8_bytes": _CDP_TYPE_MAX_UTF8_BYTES,
                "cdp_process_type_scope": "observed_owned_CDP_label_not_verified_process_function",
                "unrecognized_cdp_type_binding_count": sum(
                    row.cdp_process_type is not None and row.cdp_process_type not in _EDGE_TYPES
                    for row in self._bindings.values()),
                "unknown_count": self._unknown_count, "unknown_details": deepcopy(self._unknown),
                "unknown_details_truncated": self._unknown_count > len(self._unknown),
                "cdp_checkpoint_count": self._cdp_checkpoints,
                "owner_checkpoint_queue_limit": 16,
                "owner_checkpoint_queue_overflow": self._checkpoint_queue_overflow,
                "browser_launch_attempted": self._browser_launch_attempted,
                "node_start_attempted": self._node_start_attempted,
                "node_spawn_identity_verified": self._node_verified,
                "browser_root_identity_verified": self._browser_root_verified,
                "agent_identity_verified": any(row.role == "agent" for row in self._bindings.values()),
                "model_identity_verified": any(row.role == "model" for row in self._bindings.values()),
                "unadopted_handle_close_failures": self._unadopted_handle_close_failures,
                **({"observed_cdp_helper_capability": self._cdp_helper_capability.metadata(),
                    "observed_cdp_helper_binding_count": sum(
                        row.helper_capability_sha256 is not None for row in self._bindings.values())}
                   if self._cdp_helper_capability is not None else {})}

    def _gpu_capabilities(self) -> dict[str, Any]:
        try:
            value = self._gpu_provider.capabilities()
            if not isinstance(value, Mapping):
                raise ValueError("GPU capability report is incomplete")
            return deepcopy(dict(value))
        except Exception as exc:
            return {"enabled": True, "binding_available": False,
                    "error_type": type(exc).__name__, "missing_measurements_are_not_zero": True}

    def sample(self) -> dict[str, Any]:
        with self._lock:
            checkpoints = deepcopy(self._checkpoint_samples)
            self._checkpoint_samples.clear()
            return {"schema": PROCESS_MEMORY_SCHEMA, "cohort_generation": self.generation,
                    "monotonic_ns": time.monotonic_ns(), "wall_time_ns": time.time_ns(),
                    "working_set_includes_shared_pages": True,
                    "private_commit_is_not_physical_residency": True,
                    "sampled_maxima_are_lower_bounds": True, "coverage": self._coverage(),
                    "owner_checkpoint_samples": checkpoints,
                    "processes": deepcopy([self._sample_binding(binding) for binding in self._bindings.values()]),
                    **({"gpu_memory_capabilities": self._gpu_capabilities(),
                        "gpu_memory_scope": "bound_set_only_ownership_unknowns_preserved_in_coverage"}
                       if self._gpu_provider is not None else {})}

    def _close_bindings(self, roles: frozenset[str], timeout_seconds: float) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + timeout_seconds
        rows: list[dict[str, Any]] = []
        pending: list[tuple[_Binding, Any, dict[str, Any]]] = []
        with self._lock:
            for binding in self._bindings.values():
                if binding.role not in roles:
                    continue
                if binding.close_attempted:
                    rows.append(dict(binding.closure or {}))
                    continue
                last = self._sample_binding(binding)
                outcome = self._metadata(binding)
                outcome.update(last_counters=last, wait_method="WaitForSingleObject_retained_handle",
                               timeout_seconds=timeout_seconds, signaled=None, exit_code=None,
                               observer_handle_closed=False, sampled_maxima_are_lower_bounds=True,
                               sample_count=binding.sample_count,
                               sampled_working_set_max_bytes=binding.sampled_working_set_max,
                               sampled_private_commit_max_bytes=binding.sampled_private_commit_max)
                # Samples now use last_counters instead of the handle. No
                # registry lock is held during bounded waits or CloseHandle.
                binding.close_attempted = True
                pending.append((binding, binding.handle, outcome))
            if self._gpu_provider is not None and self._gpu_close_receipt is None:
                try:
                    receipt = self._gpu_provider.close()
                    if (not isinstance(receipt, Mapping) or type(receipt.get("drained")) is not bool
                            or type(receipt.get("adapter_handles_closed")) is not bool):
                        raise ValueError("GPU provider close receipt is incomplete")
                    self._gpu_close_receipt = deepcopy(dict(receipt))
                except Exception as exc:
                    self._gpu_close_receipt = {"drained": False, "adapter_handles_closed": False,
                                               "error_type": type(exc).__name__}
            gpu_drained = self._gpu_provider is None or self._gpu_close_receipt.get("drained") is True
        for binding, handle, outcome in pending:
            if not gpu_drained:
                outcome["gpu_sampling_drain_unverified"] = True
                with self._lock:
                    binding.closure = dict(outcome)
                rows.append(outcome)
                continue  # Preserve borrowed process objects when drain is unverified.
            try:
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                outcome.update(self._transport.wait(handle, remaining_ms))
            except Exception as exc:
                outcome["wait_error"] = type(exc).__name__ + ":" + str(exc)[:192]
            finally:
                try:
                    self._transport.close(handle)
                    outcome["observer_handle_closed"] = True
                except Exception as exc:
                    outcome["handle_close_error"] = type(exc).__name__ + ":" + str(exc)[:192]
                with self._lock:
                    binding.handle_closed = outcome["observer_handle_closed"]
                    binding.handle = None
                    binding.closure = dict(outcome)
            rows.append(outcome)
        return {"started_monotonic_ns": int(started * 1e9),
                "deadline_monotonic_ns": int(deadline * 1e9),
                "finished_monotonic_ns": time.monotonic_ns(), "processes": rows}

    def close_children(self, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
        if (type(timeout_seconds) not in {int, float} or not 0 <= timeout_seconds <= 10):
            raise ValueError("bound-set close timeout must be finite and at most ten seconds")
        with self._lock:
            if self._children_receipt is not None:
                return deepcopy(self._children_receipt)
            if self._children_closing:
                raise RuntimeError("bound-set close is already in progress")
            self._children_closing = True
        observed = self._close_bindings(_CHILD_ROLES, float(timeout_seconds))
        with self._lock:
            required = ((not self._node_start_attempted or self._node_verified)
                        and (not self._browser_launch_attempted or self._browser_root_verified))
            retired = (not self._unadopted_handle_close_failures and
                       all(row.get("signaled") is True and row.get("observer_handle_closed") is True
                           for row in observed["processes"]))
            gpu_closed = (self._gpu_provider is None or (self._gpu_close_receipt.get("drained") is True
                          and self._gpu_close_receipt.get("adapter_handles_closed") is True))
            self._children_receipt = {"schema": PROCESS_MEMORY_SCHEMA,
                "cohort_generation": self.generation, "coverage": self._coverage(), **observed,
                "owner_checkpoint_samples": deepcopy(self._checkpoint_samples),
                "required_owned_bindings_verified": required,
                "known_bound_children_retired": retired,
                "bound_set_drain_verified": required and retired and gpu_closed,
                "all_descendants_retired": False,
                "quarantine_required": not (required and retired and gpu_closed),
                **({"gpu_sampling_close": deepcopy(self._gpu_close_receipt)}
                   if self._gpu_provider is not None else {})}
            self._checkpoint_samples.clear()
            return deepcopy(self._children_receipt)

    def close(self) -> dict[str, Any]:
        with self._lock:
            if self._receipt is not None:
                return deepcopy(self._receipt)
            if self._closing:
                raise RuntimeError("process-memory registry close is already in progress")
            self._closing = True
        children = self.close_children()
        # Agent is expected alive. Model is expected retired after controller.close().
        others = self._close_bindings(frozenset({"agent", "model"}), 0.0)
        with self._lock:
            models = [row for row in others["processes"] if row.get("role") == "model"]
            agents = [row for row in others["processes"] if row.get("role") == "agent"]
            self._receipt = {"schema": PROCESS_MEMORY_SCHEMA, "cohort_generation": self.generation,
                "children": children, "agent_and_model": others, "coverage": self._coverage(),
                "agent_alive_expected": True,
                "agent_alive_observed": bool(agents) and all(row.get("signaled") is False for row in agents),
                "model_retirement_verified": bool(models) and all(row.get("signaled") is True
                    and row.get("observer_handle_closed") is True for row in models),
                "observer_handles_closed": (not self._unadopted_handle_close_failures
                    and all(binding.handle_closed for binding in self._bindings.values())),
                "all_descendants_retired": False}
            return deepcopy(self._receipt)
