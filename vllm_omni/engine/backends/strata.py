# SPDX-License-Identifier: Apache-2.0
"""A bounded whole-model Strata stage; expert placement remains backend-private.

The supervisor creates its own minimal upstream server configuration. It never
loads a user's Strata config (which may contain MCP commands or before_load).
An in-flight cancellation retires the worker tree rather than claiming an HTTP
disconnect has drained native I/O, DMA and recurrent state.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from omni_stage_contracts import StageEvent, StageRequest
from vllm.outputs import CompletionOutput

from vllm_omni.engine.backends import strata_io
from vllm_omni.engine.resource_ledger import ResourceUnavailable
from vllm_omni.engine.stage_client import StageClientBase
from vllm_omni.outputs import OmniRequestOutput

PINNED_STRATA_REVISION = "d5ea7133741e67743c0e886bb426c0ce8d69cf6c"
PINNED_STRATA_VERSION = "0.1.40.3"
BACKEND_NAME = "external.strata.text.v1"
_MAX_EVENTS = 4096
_MAX_DELTA_BYTES = 8192
_CACHE_CONTROL_SCHEMA = "omni-strata-explicit-cache-v2"

# Optional complete-model observation is source-pinned and has its own route
# admission. These hashes are reviewed adapter preimages, not native eligibility.
_EXECUTION_ADAPTER_SHA256 = {
    "strata_exec": "c653d6eb422ed1f5ca970d5db1d0d3c8d08139a1e085eabd171497490a7663bf",
    "strata_exec_bridge": "5911125621e1d27cdcc9e12f92683e7e481e411cfe099fe6ce95b42c423e63fe",
    "strata_exec_runtime": "efd2c7fd077a84d50bb2c0e2ed4d3cdf12b0a91380e78ec5673d9ad8f416138c",
    "strata_exec_live": "a4b56f159d4fc986c1e82b536c65e589327b6f25f910d7da98ac3bf8fc4a996a",
}
_EXECUTION_MODULE_LOCK = threading.RLock()
_EXECUTION_LOADED_MODULES = {}
_EXECUTION_CONFIG_SCHEMA = "omni-strata-execution-observation-config-v1"
_EXECUTION_WORKSPACE_MIN = 512 << 20
_EXECUTION_RECEIPT_MIN = 16 << 20


def _load_execution_adapters(runtime: Path, verified_files: set[Path]) -> dict:
    """Load exact reviewed bytes; neither config nor a file chooses Python code."""
    modules = {}
    with _EXECUTION_MODULE_LOCK:
        for name, expected in _EXECUTION_ADAPTER_SHA256.items():
            path = _contained(runtime, "adapter/" + name + ".py")
            if path not in verified_files or path.stat().st_size > 256 << 10:
                raise ValueError("execution adapter is absent or exceeds its source bound")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("execution adapter is not a reviewed source preimage")
            cached = sys.modules.get(name)
            owned = _EXECUTION_LOADED_MODULES.get(name)
            if cached is not None:
                if (
                    owned is None
                    or owned[0] is not cached
                    or owned[1] != expected
                    or getattr(cached, "__file__", None) != owned[2]
                    or _sha256(Path(cached.__file__)) != expected
                ):
                    raise ValueError("execution adapter module name is not owned by the reviewed loader")
                modules[name] = cached
                continue
            if owned is not None:
                raise ValueError("owned execution adapter was removed from the interpreter")
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None:
                raise ValueError("execution adapter import specification is unavailable")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                # Execute precisely the verified preimage, without rereading the
                # pathname or consulting an uncontrolled module search path.
                exec(compile(raw, str(path), "exec"), module.__dict__)
            except BaseException:
                if sys.modules.get(name) is module:
                    del sys.modules[name]
                raise
            _EXECUTION_LOADED_MODULES[name] = (module, expected, module.__file__)
            modules[name] = module
    return modules


def _execution_settings(value, runtime: Path) -> dict:
    keys = {
        "schema",
        "descriptor_file",
        "static_identity_sha256",
        "source_context_file",
        "receipt_root",
        "workspace_bytes",
        "receipt_storage_bytes",
    }
    if type(value) is not dict or set(value) != keys or value["schema"] != _EXECUTION_CONFIG_SCHEMA:
        raise ValueError("invalid execution observation configuration")
    for key in ("descriptor_file", "source_context_file"):
        name = value[key]
        if type(name) is not str or "\\" in name or ".." in Path(name).parts:
            raise ValueError("execution metadata paths must be contained relative POSIX paths")
        _contained(runtime, name)
    if type(value["static_identity_sha256"]) is not str or not re.fullmatch(
        r"[0-9a-f]{64}", value["static_identity_sha256"]
    ):
        raise ValueError("execution route requires its exact static identity")
    for key, minimum, maximum in (
        ("workspace_bytes", _EXECUTION_WORKSPACE_MIN, 2 << 30),
        ("receipt_storage_bytes", _EXECUTION_RECEIPT_MIN, 64 << 20),
    ):
        if type(value[key]) is not int or not minimum <= value[key] <= maximum:
            raise ValueError("execution observer requires a separately admitted bounded budget")
    directory = Path(value["receipt_root"])
    if directory.is_symlink():
        raise ValueError("execution receipt root cannot be a symlink")
    directory = directory.resolve(strict=True)
    if not directory.is_dir() or directory.is_relative_to(runtime):
        raise ValueError("execution receipts require a stable root outside the runtime")
    result = copy.deepcopy(value)
    result["receipt_root"] = str(directory)
    return result


def _native_cache_control(pack: Path, verified_files: set[Path], budget_bytes: int) -> dict:
    """Convert a pinned native layout into the upstream positive slot control.

    Native uniform slots use max_blob bytes; profile-sized slots round each
    blob to 256 bytes. Use the aligned maximum for both, so neither layout can
    grow past the requested byte bound. Zero has upstream auto semantics with
    a profile and is never a valid explicit bound for this verifier.
    """
    layout = (pack / "native_experts.txt").resolve(strict=True)
    if layout not in verified_files:
        raise ValueError("Strata cache bound requires hash-verified native_experts.txt")
    if layout.stat().st_size > 1 << 20:
        raise ValueError("Strata native expert layout exceeds the parsing bound")
    text = layout.read_text(encoding="utf-8")
    version = re.search(r"^# strata native experts v([1-4]):", text, re.MULTILINE)
    if version is None:
        raise ValueError("Strata cache bound requires a supported native expert layout")
    blobs = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if (
            len(columns) not in {5, 8, 9}
            or not all(re.fullmatch(r"[0-9]+", value) for value in columns[:5])
            or int(columns[0]) != len(blobs)
            or int(columns[4]) <= 0
        ):
            raise ValueError("Strata cache bound found a malformed native expert layout")
        blobs.append(int(columns[4]))
    if not blobs or len(blobs) > 4096:
        raise ValueError("Strata cache bound requires a nonempty bounded native expert layout")
    largest = max(blobs)
    aligned = (largest + 255) // 256 * 256
    slots = budget_bytes // aligned
    if not 0 < slots <= 2**31 - 1:
        raise ResourceUnavailable("Strata GPU expert cache budget cannot admit positive native cache slots")
    return {
        "schema": _CACHE_CONTROL_SCHEMA,
        "layout_sha256": _sha256(layout),
        "layout_version": int(version[1]),
        "layers": len(blobs),
        "max_blob_bytes": largest,
        "alignment_bytes": 256,
        "aligned_max_blob_bytes": aligned,
        "budget_bytes": budget_bytes,
        "requested_slots": slots,
        "allocation_upper_bytes": slots * aligned,
        "source_contract": ["kernels/cpu/expert_layout.cpp:311-415", "core/expert_cache.cpp:469-585"],
    }


def _verify_cache_bounds(info: dict, control: dict, ram_budget: int, *, profiled: bool) -> dict:
    """Combine enforced native controls with floor-MiB observations.

    The intervals are INFO's precision, not process peaks or complete memory
    observations. Native allocation bounds constrain their upper endpoint.
    """
    slots, cache_mib, arena_mib = (info.get(key) for key in ("expert_slots", "expert_cache_mib", "arena_mib"))
    if any(type(value) is not int or value < 0 for value in (slots, cache_mib, arena_mib)) or slots == 0:
        raise ResourceUnavailable("Strata native INFO cannot verify bounded expert caches")
    upper = control["allocation_upper_bytes"]
    if cache_mib * (1 << 20) > upper or arena_mib * (1 << 20) > ram_budget:
        raise ResourceUnavailable("Strata native INFO expert cache exceeds its declared component bound")
    if not profiled:
        exact = control["requested_slots"] * control["max_blob_bytes"]
        if slots != control["requested_slots"] or cache_mib != exact >> 20:
            raise ResourceUnavailable("Strata native INFO disagrees with the explicit uniform cache control")
        lower = upper = exact
    else:
        lower = cache_mib * (1 << 20)
        upper = min(upper, ((cache_mib + 1) << 20) - 1)
    return {
        "evidence": "native_control_bound_with_floor_MiB_INFO_consistency",
        "gpu_expert_cache": {
            "observed_slots": slots,
            "observed_mib_floor": cache_mib,
            "allocation_lower_bytes": lower,
            "allocation_upper_bytes": upper,
            "declared_budget_bytes": control["budget_bytes"],
            "profile_sized": profiled,
        },
        "ram_expert_cache": {
            "observed_mib_floor": arena_mib,
            "allocation_lower_bytes": arena_mib << 20,
            "allocation_upper_bytes": min(ram_budget, ((arena_mib + 1) << 20) - 1),
            "declared_budget_bytes": ram_budget,
            "source_contract": "core/expert_source.cpp:2101-2147; program/generate.cpp:7892",
        },
        "aggregate_gpu_hard_cap_verified": False,
        "total_process_peak_verified": False,
    }


def _probe_memory(gpu_index: int) -> dict:
    """Measure immediately before loading; no persisted launch ceiling is proof.

    NVML bytes are preferred. The fallback's MiB granularity is explicit and
    must still match the declared total exactly (never round a claim upward).
    """
    import psutil

    snapshot = {
        "host_ram_available_bytes": int(psutil.virtual_memory().available),
        "host_ram_source": "psutil.virtual_memory.available",
        "gpu_total_bytes": None,
        "gpu_free_bytes": None,
        "gpu_source": None,
        "gpu_name_sha256": None,
        "gpu_uuid": None,
        "gpu_pci_bus_id": None,
        "windows_commit_available_bytes": None,
        "windows_host_available_bytes": None,
        "wsl_ram_available_bytes": None,
        "is_wsl": False,
        "windows_host_verification": "not_applicable",
        "captured_unix_s": time.time(),
    }
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total_phys",
                    "avail_phys",
                    "total_page",
                    "avail_page",
                    "total_virtual",
                    "avail_virtual",
                    "avail_extended_virtual",
                )
            ]

        state = MemoryStatus()
        state.length = ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            raise ResourceUnavailable("Windows physical RAM/commit measurement failed")
        snapshot.update(
            host_ram_available_bytes=int(state.avail_phys),
            host_ram_source="GlobalMemoryStatusEx.ullAvailPhys",
            windows_host_available_bytes=int(state.avail_phys),
            windows_commit_available_bytes=int(state.avail_page),
            windows_host_verification="measured",
        )
    elif "microsoft" in platform.release().lower():
        snapshot["is_wsl"] = True
        snapshot["wsl_ram_available_bytes"] = snapshot["host_ram_available_bytes"]
        snapshot["windows_host_verification"] = "unverified"
        # WSL has its own quota. A successful Windows-side observation adds a
        # distinct host constraint; failure is retained explicitly as unknown.
        powershell = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        try:
            probe = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            free = int(probe.stdout.strip()) * 1024
            if free <= 0:
                raise ValueError("invalid host observation")
            snapshot.update(windows_host_available_bytes=free, windows_host_verification="measured")
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    nvml = None
    try:
        import pynvml as nvml

        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        name = nvml.nvmlDeviceGetName(handle)
        name = name.decode("utf-8", "strict") if isinstance(name, bytes) else str(name)
        snapshot.update(gpu_total_bytes=int(info.total), gpu_free_bytes=int(info.free), gpu_source="NVML exact bytes")
        snapshot["gpu_name_sha256"] = hashlib.sha256(name.strip().encode()).hexdigest()
        # Optional observer identity must not weaken the independent memory
        # admission probe when an older NVML lacks either identity API.
        try:
            gpu_uuid = nvml.nvmlDeviceGetUUID(handle)
            pci = nvml.nvmlDeviceGetPciInfo(handle).busId
            snapshot["gpu_uuid"] = gpu_uuid.decode() if isinstance(gpu_uuid, bytes) else str(gpu_uuid)
            snapshot["gpu_pci_bus_id"] = pci.decode() if isinstance(pci, bytes) else str(pci)
        except Exception:
            pass
    except Exception:
        # nvidia-smi exists in native Windows and the WSL NVIDIA integration.
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_index),
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            name, total_text, free_text = probe.stdout.strip().rsplit(",", 2)
            total, free = int(total_text.strip()), int(free_text.strip())
            if total <= 0 or not 0 <= free <= total:
                raise ValueError("invalid GPU observation")
            snapshot.update(
                gpu_total_bytes=total << 20,
                gpu_free_bytes=free << 20,
                gpu_source="nvidia-smi integer MiB; rounded observation",
                gpu_name_sha256=hashlib.sha256(name.strip().encode()).hexdigest(),
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    finally:
        if nvml is not None:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
    return snapshot


def _check_live_memory(snapshot: dict, demands: dict, gpu_pool: str, gpu_total: int) -> None:
    """Check the full not-yet-loaded claim once, including explicit headroom."""
    failures = []
    if snapshot.get("gpu_total_bytes") != gpu_total:
        failures.append("declared GPU total differs from fresh observation or GPU is unverified")
    if snapshot.get("is_wsl") is True and snapshot.get("windows_host_available_bytes") is None:
        failures.append("Windows physical RAM is unverified from WSL; host admission is blocked")
    checks = [("host_ram", "host_ram_available_bytes"), (gpu_pool, "gpu_free_bytes")]
    if os.name == "nt":
        # A RAM-only claim still requires commit; paging capacity is no license
        # to exceed physical RAM because both checks apply independently.
        checks.append(("host_ram", "windows_commit_available_bytes"))
    if "windows_commit" in demands:
        checks.append(("windows_commit", "windows_commit_available_bytes"))
    if "wsl_ram" in demands:
        checks.append(("wsl_ram", "wsl_ram_available_bytes"))
    if snapshot.get("windows_host_available_bytes") is not None:
        checks.append(("host_ram", "windows_host_available_bytes"))
    for pool, field in checks:
        required, available = demands.get(pool, 0), snapshot.get(field)
        if type(available) is not int or available < required:
            failures.append(f"{pool}: required={required}, {field}={available}")
    if failures:
        raise ResourceUnavailable(
            "Strata fresh admission refused: "
            + "; ".join(failures)
            + "; snapshot="
            + json.dumps(snapshot, sort_keys=True)
        )


def _verify_python_environment(python: Path, expected: dict | None) -> dict | None:
    if expected is None:
        return None
    if not isinstance(expected, dict):
        raise ValueError("python_environment must be a captured environment object")
    dependencies = expected.get("dependencies")
    names = ("numpy", "jinja2", "regex", "PyYAML", "psutil", "Pillow", "gguf")
    if not isinstance(dependencies, dict) or set(dependencies) != set(names):
        raise ValueError("python_environment dependency set differs from the pinned probe")
    source = (
        "import importlib.metadata as m,json,sys\n"
        "def version(n):\n"
        " try:return m.version(n)\n"
        " except m.PackageNotFoundError:return None\n"
        f"print(json.dumps(dict(sys_version=sys.version,dependencies={{n:version(n) for n in {names!r}}})))"
    )
    probe = subprocess.run(
        [str(python), "-I", "-B", "-X", "utf8", "-c", source],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    actual = json.loads(probe.stdout)
    if (
        actual.get("sys_version") != expected.get("sys_version")
        or actual.get("dependencies") != dependencies
        or _sha256(python) != expected.get("executable_sha256")
    ):
        raise ValueError("Strata Python environment changed since artifact preparation")
    return actual | {"executable_sha256": expected["executable_sha256"]}


def _windows_process_creation_filetime(pid: int) -> int | None:
    """Exact OS identity; psutil's float creation time can round FILETIME."""
    if os.name != "nt" or type(pid) is not int or pid <= 0:
        return None
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:  # STILL_ACTIVE
            return None
        creation, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in (creation, exited, system, user))):
            return None
        return (creation.dwHighDateTime << 32) | creation.dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def _sanitize_diagnostic(line: str, *, observer_nonce: str | None = None) -> str | None:
    """Content-free codes only; upstream logs can contain uncontrolled paths."""
    # The native child and Python supervisor each use platform text streams;
    # Windows pipe bytes therefore end in CRLF. Normalize line terminators
    # before the same exact content patterns used on POSIX.
    line = line.rstrip("\r\n")
    match = re.fullmatch(r"strata supervisor: native identity ([0-9a-f]{32}) (\{.{1,256}\})", line)
    if match:
        if observer_nonce is None or match[1] != observer_nonce:
            return None
        try:
            identity = json.loads(match[2])
            if set(identity) != {"pid", "creation_filetime_100ns"} or any(
                type(value) is not int or value <= 0 for value in identity.values()
            ):
                return None
            return "strata supervisor: native identity " + json.dumps(identity, sort_keys=True)
        except (ValueError, TypeError):
            return None
    if line.strip() == "strata supervisor: native contained in Windows kill-on-close job":
        return line.strip()
    if line.strip() == "strata supervisor: native process started":
        return line.strip()
    if line.startswith("strata supervisor: native done "):
        try:
            counters = json.loads(line.split("native done ", 1)[1])
            names = {"generated", "prompt_tokens", "hits", "lookups", "offloaded", "prompt_read"}
            if (
                set(counters) != names
                or any(type(v) is not int or v < 0 for v in counters.values())
                or counters["hits"] > counters["lookups"]
            ):
                return None
            return "strata supervisor: native done " + json.dumps(counters, sort_keys=True)
        except (ValueError, TypeError):
            return None
    match = re.fullmatch(r"strata generate: GPU ([0-9]+): (.{1,256}), compute capability ([0-9]+)\.([0-9]+)\n?", line)
    if match and int(match[3]) > 0 and match[2] != "(an unnamed GPU)":
        return "strata load_evidence: gpu " + json.dumps(
            {
                "local_index": int(match[1]),
                "compute_capability": f"{match[3]}.{match[4]}",
                "name_sha256": hashlib.sha256(match[2].strip().encode()).hexdigest(),
            },
            sort_keys=True,
        )
    match = re.fullmatch(
        r"strata generate: CPU pool tasks/phase: ([0-9]+)(?: \(automatic\)| \(capped by rows\))?"
        r", participating threads: ([0-9]+)\n?",
        line,
    )
    if match:
        return "strata load_evidence: cpu_pool " + json.dumps(
            {"tasks_per_phase": int(match[1]), "participating_threads": int(match[2])}, sort_keys=True
        )
    match = re.fullmatch(r"strata generate: ([0-9]+) expert-pool workers( \+ the host thread)?\n?", line)
    if match:
        return "strata load_evidence: expert_workers " + json.dumps(
            {"workers": int(match[1]), "host_thread": bool(match[2])}, sort_keys=True
        )
    if re.fullmatch(
        r"strata generate: native pack: .+ experts \(largest blob [0-9.]+ MB\)"
        r", token embedding [A-Za-z0-9_]+ in mapped host memory \([0-9.]+ MiB, [0-9.]+ s\)\n?",
        line,
    ):
        return "strata load_evidence: native_pack"
    if line.startswith("ready: http://127.0.0.1:"):
        return "strata status: ready"
    if line.startswith("loading the model ("):
        return "strata status: loading"
    match = re.match(r"^strata generate: expert arena read (unbuffered|through the file cache) \(", line)
    if match:
        return "strata arena_load_io_mode: " + ("unbuffered" if match[1] == "unbuffered" else "buffered")
    match = re.match(
        r"^strata generate: the file tier reads (unbuffered|through the file cache)(?: \(changed\))? \(", line
    )
    if match:
        return "strata file_tier_io_policy: " + ("unbuffered" if match[1] == "unbuffered" else "buffered")
    if re.match(r"^FileExpertSource: --resident-budget-gib [0-9.]+ exceeds available ", line):
        return "strata diagnostic: resident_budget_clamped"
    # Require the native logger prefix; prompt/body words are not diagnostics.
    if not line.startswith(("strata generate:", "FileExpertSource:", "[strata]")):
        return None
    for needles, code in (
        (("out of memory", "allocation failed"), "allocation_failed"),
        (("RAM budget", "cannot be kept"), "resident_budget_not_kept"),
        (("unbuffered I/O",), "unbuffered_file_tier_mentioned"),
        (("failed", "error"), "backend_error"),
    ):
        if code == "resident_budget_not_kept":
            matches = all(needle in line for needle in needles)
        else:
            matches = any(needle.lower() in line.lower() for needle in needles)
        if matches:
            return "strata diagnostic: " + code
    return None


class _DiagnosticLog:
    def __init__(
        self,
        pipe,
        path: Path | None,
        *,
        observer_nonce: str | None = None,
        frame_callback=None,
        execution_token: str | None = None,
        execution_module=None,
    ):
        if (execution_token is None) != (execution_module is None):
            raise ValueError("Strata execution diagnostics require a verified module and token")
        self.pipe, self.path = pipe, path
        self.observer_nonce = observer_nonce
        self.frame_callback = frame_callback
        self.native_identity = None
        self.native_identity_conflict = False
        self.records: list[str] = []
        self.file_tier_io_policy = None
        self.native_starts = 0
        self.native_done_count = 0
        self.native_done = None
        self.io_observer = None
        self.load_evidence: dict[str, Any] = {}
        self._changed = threading.Condition()
        self.failed = False
        self._execution_bridge = None
        self._execution_token = execution_token
        self._execution_failure = None
        self._execution_reader = (
            execution_module.AuthenticatedControlLineReader(
                execution_token,
                on_exec=self._consume_execution_frame,
                on_ordinary=self._consume_ordinary,
                on_failure=self._observation_failure,
            )
            if execution_module is not None
            else None
        )
        self._diagnostic_handle = None
        self._thread = threading.Thread(target=self._drain, name="strata-diagnostics", daemon=True)
        self._thread.start()

    def _observation_failure(self, code):
        with self._changed:
            self.failed = True
            self._execution_failure = code
            # The two channels fail independently; failure in one callback must
            # never prevent the other from retaining a partial observation.
            try:
                if self._execution_bridge is not None:
                    self._execution_bridge.channel_failure(code)
            finally:
                try:
                    if self.io_observer is not None:
                        self.io_observer.mark_incomplete(code)
                finally:
                    self._changed.notify_all()

    def _consume_execution_frame(self, frame):
        with self._changed:
            try:
                if self._execution_bridge is None:
                    self._observation_failure("invalid_owned_exec_frame")
                    return
                self._execution_bridge.ingest(frame, channel_token=self._execution_token)
            finally:
                self._changed.notify_all()

    def _consume_ordinary(self, row):
        decoded = row.decode("utf-8", "replace")
        if self.frame_callback is not None:
            self.frame_callback(decoded)
        safe = _sanitize_diagnostic(decoded, observer_nonce=self.observer_nonce)
        with self._changed:
            if self.io_observer is not None:
                self.io_observer.ingest(decoded)
            if safe == "strata supervisor: native process started":
                self.native_starts += 1
            elif safe is not None and safe.startswith("strata supervisor: native done "):
                self.native_done_count += 1
                self.native_done = json.loads(safe.split("native done ", 1)[1])
            elif safe is not None and safe.startswith("strata supervisor: native identity "):
                identity = json.loads(safe.split("native identity ", 1)[1])
                if self.native_identity is not None and self.native_identity != identity:
                    self.native_identity_conflict = True
                self.native_identity = identity
            elif safe == "strata load_evidence: native_pack":
                self.load_evidence["native_pack"] = True
            elif safe is not None and safe.startswith("strata load_evidence: "):
                kind, value = safe.split("strata load_evidence: ", 1)[1].split(" ", 1)
                parsed = json.loads(value)
                previous = self.load_evidence.get(kind)
                if previous is not None and previous != parsed:
                    self.load_evidence["contradictory"] = True
                self.load_evidence[kind] = parsed
            self._changed.notify_all()
        if safe is not None and safe.startswith("strata file_tier_io_policy: "):
            self.file_tier_io_policy = safe.rsplit(": ", 1)[1]
        if safe is not None and safe not in self.records and len(self.records) < 64:
            self.records.append(safe)
            if self._diagnostic_handle:
                self._diagnostic_handle.write(safe + "\n")

    def _drain(self):
        try:
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._diagnostic_handle = self.path.open("w", encoding="utf-8", buffering=1)
            if self._execution_reader is not None:
                self._execution_reader.drain(self.pipe)
            else:
                # Preserve the legacy bounded ordinary/IO path byte for byte.
                skipping = False
                while row := self.pipe.readline(4097):
                    complete = row.endswith(b"\n")
                    if skipping or len(row) > 4096 or not complete:
                        if self.io_observer is not None:
                            self.io_observer.ingest(row.decode("utf-8", "replace"))
                        skipping = not complete
                        continue
                    self._consume_ordinary(row)
        except Exception:
            self._observation_failure("diagnostic_read_failed")
        finally:
            if self._diagnostic_handle:
                self._diagnostic_handle.close()
                self._diagnostic_handle = None

    def join(self):
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            self.pipe.close()
        return not self.failed and not self._thread.is_alive()

    def wait_load_evidence(self, timeout: float = 5) -> dict:
        with self._changed:
            self._changed.wait_for(
                lambda: all(key in self.load_evidence for key in ("gpu", "cpu_pool", "expert_workers", "native_pack")),
                timeout=timeout,
            )
            return dict(self.load_evidence) | {"native_starts": self.native_starts}

    def completed_native_request(self, expected: int, timeout: float = 5) -> dict | None:
        with self._changed:
            self._changed.wait_for(lambda: self.native_done_count >= expected, timeout=timeout)
            if self.native_starts != 1 or self.native_done_count != expected:
                return None
            return None if self.native_done is None else dict(self.native_done)

    def owned_native_identity(self) -> dict | None:
        with self._changed:
            if self.native_starts != 1 or self.native_identity_conflict or self.native_identity is None:
                return None
            return dict(self.native_identity)

    def bind_io_observer(self, generation: str, runtime_identity: dict) -> None:
        with self._changed:
            identity = self.owned_native_identity()
            if identity is None:
                raise RuntimeError("Strata I/O observer needs an authoritative owned native identity")
            if runtime_identity.get("schema") == "omni-strata-combined-static-runtime-identity-v2":
                self.io_observer = strata_io.StrataIoObserver.from_combined_runtime(
                    self.observer_nonce,
                    generation,
                    identity["pid"],
                    identity["creation_filetime_100ns"],
                    runtime_identity,
                )
            else:
                self.io_observer = strata_io.StrataIoObserver(
                    self.observer_nonce,
                    generation,
                    identity["pid"],
                    identity["creation_filetime_100ns"],
                    runtime_identity,
                )
            if self._execution_failure is not None:
                self.io_observer.mark_incomplete(self._execution_failure)

    def finish_io_request(self, request: StageRequest, *, completed: bool, reason: str | None = None) -> dict:
        with self._changed:
            observer = self.io_observer
            if reason:
                observer.mark_incomplete(reason)
            if completed:
                self._changed.wait_for(
                    lambda: observer.retired or observer.active is not None and observer.active["terminal"] is not None,
                    timeout=5,
                )
            return observer.finish(request.request_id, request.epoch, completed=completed)

    def bind_execution_bridge(self, bridge) -> None:
        with self._changed:
            if self._execution_reader is None or self._execution_bridge is not None:
                raise RuntimeError("Strata execution reader is unavailable or already bound")
            if type(bridge.token) is not str or bridge.token != self._execution_token:
                raise RuntimeError("Strata execution reader token does not match its bridge")
            self._execution_bridge = bridge
            if self._execution_failure is not None:
                bridge.channel_failure(self._execution_failure)

    def begin_execution_request(self, request: StageRequest) -> None:
        with self._changed:
            if self._execution_bridge is None:
                raise RuntimeError("Strata execution bridge is not bound")
            self._execution_bridge.begin(request.request_id, request.epoch)

    def finish_execution_request(
        self,
        request: StageRequest,
        *,
        omni_completed: bool,
        cancellation_requested: bool,
        lifecycle_outcome: str,
        reader_joined: bool | None = None,
    ) -> dict:
        with self._changed:
            bridge = self._execution_bridge
            if bridge is None:
                raise RuntimeError("Strata execution bridge is not bound")
            if lifecycle_outcome == "normal":
                self._changed.wait_for(
                    lambda: self.failed or bridge.ready_to_finish(request.request_id, request.epoch), timeout=5
                )
            actual_reader_joined = not self._thread.is_alive()
            if reader_joined is not None and (type(reader_joined) is not bool or reader_joined != actual_reader_joined):
                raise ValueError("Strata diagnostic reader join override contradicts its actual thread")
            return bridge.finish(
                request.request_id,
                request.epoch,
                omni_completed=omni_completed,
                cancellation_requested=cancellation_requested,
                lifecycle_outcome=lifecycle_outcome,
                reader_healthy=not self.failed and not self._execution_reader.failed,
                reader_joined=actual_reader_joined,
            )


# Patch only the child stderr sink, not its weights/execution/protocol. This
# avoids persisting a raw native log. The shim is included in execution identity.
_BOOTSTRAP = r"""
import hashlib, importlib.util, json, os, runpy, subprocess, sys, threading
native, server = sys.argv[1:3]
sys.argv = [server] + sys.argv[3:]
observer_nonce = os.environ.pop('OMNI_STRATA_OBSERVER_NONCE', '')
io_adapter_path = os.environ.pop('OMNI_STRATA_IO_ADAPTER', '')
io_adapter_sha256 = os.environ.pop('OMNI_STRATA_IO_ADAPTER_SHA256', '')
io_generation = os.environ.pop('OMNI_STRATA_IO_GENERATION', '')
io_module = None
system_directory = None
if io_adapter_path:
    if os.name != 'nt': raise RuntimeError('Observed Strata runtime currently requires Windows')
    with open(io_adapter_path, 'rb') as source:
        if hashlib.sha256(source.read()).hexdigest() != io_adapter_sha256:
            raise RuntimeError('Strata observer adapter changed before bootstrap import')
    spec = importlib.util.spec_from_file_location('omni_strata_owned_io', io_adapter_path)
    io_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(io_module)
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.SetDllDirectoryW.argtypes = (wintypes.LPCWSTR,); kernel.SetDllDirectoryW.restype = wintypes.BOOL
    kernel.GetSystemDirectoryW.argtypes = (wintypes.LPWSTR, wintypes.UINT)
    kernel.GetSystemDirectoryW.restype = wintypes.UINT
    system = ctypes.create_unicode_buffer(32768)
    count = kernel.GetSystemDirectoryW(system, len(system))
    if not count or count >= len(system) or not kernel.SetDllDirectoryW(''):
        raise RuntimeError('Strata native dependency search isolation failed')
    system_directory = system.value
original = subprocess.Popen
def launch(args, *a, **kw):
    if os.path.realpath(str(args[0])) != os.path.realpath(native):
        return original(args, *a, **kw)
    kw['stderr'] = subprocess.PIPE
    if io_module is not None:
        # Application-directory pinned cuBLAS plus System32. SetDllDirectory('')
        # above removes CWD from inherited standard DLL search; OS/driver modules
        # remain machine-specific and are not claimed pinned by this policy.
        native_env = dict(kw.get('env') or os.environ)
        native_env['PATH'] = os.path.dirname(os.path.realpath(native)) + os.pathsep + system_directory
        kw['env'] = native_env
    p = original(args, *a, **kw)
    owned_io = None
    sys.stderr.write('strata supervisor: native process started\n'); sys.stderr.flush()
    if os.name == 'nt':
        from serve.winjob import contain
        if not contain(p):
            p.kill(); p.wait(timeout=5)
            raise RuntimeError('Strata native Windows containment failed')
        sys.stderr.write('strata supervisor: native contained in Windows kill-on-close job\n')
        sys.stderr.flush()
        if observer_nonce:
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
            kernel.GetProcessTimes.restype = wintypes.BOOL
            times = [wintypes.FILETIME() for _ in range(4)]
            if kernel.GetProcessTimes(int(p._handle), *(ctypes.byref(t) for t in times)):
                created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
                if created > 0:
                    identity = {'pid': p.pid, 'creation_filetime_100ns': created}
                    sys.stderr.write('strata supervisor: native identity '+observer_nonce+' '+json.dumps(identity)+'\n')
                    sys.stderr.flush()
                    if io_module is not None:
                        def emit_owned(row): sys.stderr.write(row); sys.stderr.flush()
                        owned_io = io_module.OwnedNativeFrameWriter(
                            observer_nonce, io_generation, p.pid, created, emit_owned)
    if io_module is not None and owned_io is None:
        p.kill(); p.wait(timeout=5)
        raise RuntimeError('Strata I/O observer did not obtain owned native identity')
    class NativeInput:
        def __init__(self, pipe): self.pipe = pipe
        def write(self, row):
            text = row.decode('utf-8', 'strict') if isinstance(row, bytes) else row
            owned_io.observe_input(text)
            return self.pipe.write(row)
        def __getattr__(self, name): return getattr(self.pipe, name)
    if owned_io is not None and p.stdin is not None: p.stdin = NativeInput(p.stdin)
    # Observe counters only; return each exact line unchanged to the upstream
    # parser. Token IDs, prompts and tensor contents are never copied to logs.
    class NativeOutput:
        def __init__(self, pipe): self.pipe = pipe
        def __iter__(self): return self
        def __next__(self):
            row = self.readline()
            if not row: raise StopIteration
            return row
        def readline(self, *args):
            row = self.pipe.readline(*args)
            text = row.decode('utf-8', 'replace') if isinstance(row, bytes) else row
            if owned_io is not None: owned_io.observe_output(text)
            if text.startswith('DONE '):
                fields = text.split()
                try:
                    if len(fields) == 16:
                        indexes = {'generated':1, 'prompt_tokens':2, 'hits':9, 'lookups':10,
                                   'prompt_read':14, 'offloaded':15}
                        counters = {k:int(fields[i]) for k,i in indexes.items()}
                        if all(v >= 0 for v in counters.values()) and counters['hits'] <= counters['lookups']:
                            sys.stderr.write('strata supervisor: native done '+json.dumps(counters)+'\n')
                            sys.stderr.flush()
                except (ValueError, IndexError): pass
            return row
        def __getattr__(self, name): return getattr(self.pipe, name)
    if p.stdout is not None: p.stdout = NativeOutput(p.stdout)
    def drain():
        skipping = False
        while True:
            row = p.stderr.readline(4097)
            if not row: break
            if isinstance(row, bytes): row = row.decode('utf-8', 'replace')
            complete = row.endswith('\n')
            if skipping or len(row) > 4096 or not complete:
                skipping = not complete; continue
            # Only native diagnostics go up this pipe; parent applies a strict
            # content-free canonical filter before retention.
            sys.stderr.write(row); sys.stderr.flush()
        p.stderr.close()
    threading.Thread(target=drain, daemon=True).start()
    return p
subprocess.Popen = launch
runpy.run_path(server, run_name='__main__')
"""


_EXECUTION_BOOTSTRAP = r"""
import hashlib, importlib.util, json, os, runpy, subprocess, sys, threading
native, server = sys.argv[1:3]
sys.argv = [server] + sys.argv[3:]
observer_nonce = os.environ.pop('OMNI_STRATA_OBSERVER_NONCE', '')
io_adapter_path = os.environ.pop('OMNI_STRATA_IO_ADAPTER', '')
io_adapter_sha256 = os.environ.pop('OMNI_STRATA_IO_ADAPTER_SHA256', '')
io_generation = os.environ.pop('OMNI_STRATA_IO_GENERATION', '')
# Remove all private context before importing adapters or launching any native
# process. Only the opt-in switch below may reach the native child's environment.
exec_keys = {
    'OMNI_STRATA_EXEC_TOKEN', 'OMNI_STRATA_EXEC_GENERATION', 'OMNI_STRATA_EXEC_ADAPTER',
    'OMNI_STRATA_EXEC_ADAPTER_SHA256', 'OMNI_STRATA_EXEC_PARSER', 'OMNI_STRATA_EXEC_PARSER_SHA256',
}
exec_values = {name: os.environ.pop(name, '') for name in exec_keys}
exec_unknown = [name for name in os.environ if name.startswith('OMNI_STRATA_EXEC_')]
for name in exec_unknown: os.environ.pop(name)
if exec_unknown: raise RuntimeError('Unrecognized Strata execution observer environment')
os.environ.pop('STRATA_OMNI_EXEC_OBSERVER', None)
exec_module = None
exec_configuration = None
def load_execution_source(name, path, expected):
    if name in sys.modules: raise RuntimeError('Strata execution module already imported')
    with open(path, 'rb') as source: data = source.read(131073)
    if not 0 < len(data) <= 131072 or hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError('Strata execution source changed before bootstrap import')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        # Execute the exact verified bytes, without loader reopening the path.
        exec(compile(data, path, 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
if any(exec_values.values()):
    if not all(type(value) is str and value for value in exec_values.values()):
        raise RuntimeError('Partial Strata execution observer environment')
    if os.name != 'nt': raise RuntimeError('Observed Strata execution currently requires Windows')
    if (exec_values['OMNI_STRATA_EXEC_PARSER_SHA256'] !=
            'c653d6eb422ed1f5ca970d5db1d0d3c8d08139a1e085eabd171497490a7663bf'):
        raise RuntimeError('Strata execution parser identity differs')
    load_execution_source('strata_exec', exec_values['OMNI_STRATA_EXEC_PARSER'],
                          exec_values['OMNI_STRATA_EXEC_PARSER_SHA256'])
    exec_module = load_execution_source('omni_strata_owned_exec', exec_values['OMNI_STRATA_EXEC_ADAPTER'],
                                        exec_values['OMNI_STRATA_EXEC_ADAPTER_SHA256'])
    exec_configuration = exec_module.take_bootstrap_configuration(exec_values)
    sys.stderr = exec_module.SerializedDiagnosticSink(sys.stderr)
io_module = None
system_directory = None
if io_adapter_path:
    if os.name != 'nt': raise RuntimeError('Observed Strata runtime currently requires Windows')
    with open(io_adapter_path, 'rb') as source:
        if hashlib.sha256(source.read()).hexdigest() != io_adapter_sha256:
            raise RuntimeError('Strata observer adapter changed before bootstrap import')
    spec = importlib.util.spec_from_file_location('omni_strata_owned_io', io_adapter_path)
    io_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(io_module)
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.SetDllDirectoryW.argtypes = (wintypes.LPCWSTR,); kernel.SetDllDirectoryW.restype = wintypes.BOOL
    kernel.GetSystemDirectoryW.argtypes = (wintypes.LPWSTR, wintypes.UINT)
    kernel.GetSystemDirectoryW.restype = wintypes.UINT
    system = ctypes.create_unicode_buffer(32768)
    count = kernel.GetSystemDirectoryW(system, len(system))
    if not count or count >= len(system) or not kernel.SetDllDirectoryW(''):
        raise RuntimeError('Strata native dependency search isolation failed')
    system_directory = system.value
if exec_module is not None and io_module is None:
    raise RuntimeError('Execution observation requires the verified native I/O owner channel')
original = subprocess.Popen
def launch(args, *a, **kw):
    if os.path.realpath(str(args[0])) != os.path.realpath(native):
        return original(args, *a, **kw)
    kw['stderr'] = subprocess.PIPE
    if io_module is not None:
        # Application-directory pinned cuBLAS plus System32. SetDllDirectory('')
        # above removes CWD from inherited standard DLL search; OS/driver modules
        # remain machine-specific and are not claimed pinned by this policy.
        native_env = dict(kw.get('env') or os.environ)
        native_env['PATH'] = os.path.dirname(os.path.realpath(native)) + os.pathsep + system_directory
        kw['env'] = native_env
    if exec_module is not None:
        kw = exec_module.prepare_native_kwargs(kw, os.environ, opt_in=True)
    elif kw.get('env') is not None:
        native_env = dict(kw['env'])
        for name in tuple(native_env):
            if name.startswith('OMNI_STRATA_EXEC_'): native_env.pop(name)
        native_env.pop('STRATA_OMNI_EXEC_OBSERVER', None)
        kw['env'] = native_env
    p = original(args, *a, **kw)
    owned_io = None
    sys.stderr.write('strata supervisor: native process started\n'); sys.stderr.flush()
    if os.name == 'nt':
        from serve.winjob import contain
        if not contain(p):
            p.kill(); p.wait(timeout=5)
            raise RuntimeError('Strata native Windows containment failed')
        sys.stderr.write('strata supervisor: native contained in Windows kill-on-close job\n')
        sys.stderr.flush()
        if observer_nonce:
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
            kernel.GetProcessTimes.restype = wintypes.BOOL
            times = [wintypes.FILETIME() for _ in range(4)]
            if kernel.GetProcessTimes(int(p._handle), *(ctypes.byref(t) for t in times)):
                created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
                if created > 0:
                    identity = {'pid': p.pid, 'creation_filetime_100ns': created}
                    sys.stderr.write('strata supervisor: native identity '+observer_nonce+' '+json.dumps(identity)+'\n')
                    sys.stderr.flush()
                    if io_module is not None:
                        def emit_owned(row):
                            if exec_module is not None: sys.stderr.emit_control(row)
                            else: sys.stderr.write(row); sys.stderr.flush()
                        owned_io = io_module.OwnedNativeFrameWriter(
                            observer_nonce, io_generation, p.pid, created, emit_owned)
    if io_module is not None and owned_io is None:
        p.kill(); p.wait(timeout=5)
        raise RuntimeError('Strata I/O observer did not obtain owned native identity')
    class NativeInput:
        def __init__(self, pipe): self.pipe = pipe
        def write(self, row):
            text = row.decode('utf-8', 'strict') if isinstance(row, bytes) else row
            owned_io.observe_input(text)
            return self.pipe.write(row)
        def __getattr__(self, name): return getattr(self.pipe, name)
    if owned_io is not None and p.stdin is not None: p.stdin = NativeInput(p.stdin)
    # Observe counters only; return each exact line unchanged to the upstream
    # parser. Token IDs, prompts and tensor contents are never copied to logs.
    class NativeOutput:
        def __init__(self, pipe): self.pipe = pipe
        def __iter__(self): return self
        def __next__(self):
            row = self.readline()
            if not row: raise StopIteration
            return row
        def readline(self, *args):
            row = self.pipe.readline(*args)
            text = row.decode('utf-8', 'replace') if isinstance(row, bytes) else row
            if owned_io is not None: owned_io.observe_output(text)
            if text.startswith('DONE '):
                fields = text.split()
                try:
                    if len(fields) == 16:
                        indexes = {'generated':1, 'prompt_tokens':2, 'hits':9, 'lookups':10,
                                   'prompt_read':14, 'offloaded':15}
                        counters = {k:int(fields[i]) for k,i in indexes.items()}
                        if all(v >= 0 for v in counters.values()) and counters['hits'] <= counters['lookups']:
                            sys.stderr.write('strata supervisor: native done '+json.dumps(counters)+'\n')
                            sys.stderr.flush()
                except (ValueError, IndexError): pass
            return row
        def __getattr__(self, name): return getattr(self.pipe, name)
    if p.stdout is not None: p.stdout = NativeOutput(p.stdout)
    if exec_module is not None:
        # Existing IO wrappers and containment/GetProcessTimes are already bound.
        # The outer wrapper observes metadata and returns original stdout objects.
        try:
            owned_exec = exec_module.OwnedExecutionFrameWriter(
                exec_configuration['OMNI_STRATA_EXEC_TOKEN'],
                exec_configuration['OMNI_STRATA_EXEC_GENERATION'], p.pid, created, sys.stderr.emit_control)
            exec_module.attach_execution_pipes(p, owned_exec)
        except BaseException:
            p.kill(); p.wait(timeout=5)
            raise
    def drain():
        skipping = False
        while True:
            row = p.stderr.readline(4097)
            if not row: break
            if isinstance(row, bytes): row = row.decode('utf-8', 'replace')
            complete = row.endswith('\n')
            if skipping or len(row) > 4096 or not complete:
                skipping = not complete; continue
            # Only native diagnostics go up this pipe; parent applies a strict
            # content-free canonical filter before retention.
            sys.stderr.write(row); sys.stderr.flush()
        p.stderr.close()
    threading.Thread(target=drain, daemon=True).start()
    return p
subprocess.Popen = launch
runpy.run_path(server, run_name='__main__')
"""


def _verify_load_configuration(
    evidence: dict, info: Any, memory: dict, *, gpu: int, context: int, kv: str, verify_window: int
) -> dict:
    """Verify a loaded execution configuration, never claim per-request work."""
    reasons = []
    info = info if isinstance(info, dict) else {}
    expected = {
        "engine": PINNED_STRATA_VERSION,
        "context": context,
        "kv": kv,
        "spec": verify_window,
        "lookup": 0,
        "conversation_cache_mib": 0,
        "conversation_cache_slots": 0,
    }
    for key, value in expected.items():
        if info.get(key) != value:
            reasons.append(f"native INFO {key} does not verify the requested value")
    observed_gpu = evidence.get("gpu", {})
    if (
        observed_gpu.get("local_index") != 0
        or not memory.get("gpu_name_sha256")
        or observed_gpu.get("name_sha256") != memory.get("gpu_name_sha256")
    ):
        reasons.append("native CUDA identity does not match the measured physical GPU")
    pool, workers = evidence.get("cpu_pool", {}), evidence.get("expert_workers", {})
    if (
        type(workers.get("workers")) is not int
        or type(pool.get("tasks_per_phase")) is not int
        or pool.get("tasks_per_phase", 0) <= 0
        or pool.get("participating_threads", 0) <= 0
        or pool.get("participating_threads") != workers.get("workers", 0) + int(workers.get("host_thread", False))
        or info.get("pool_workers") != workers.get("workers")
    ):
        reasons.append("native CPU expert dispatch pool is unverified")
    for key in ("expert_slots", "expert_cache_mib", "arena_mib"):
        if type(info.get(key)) is not int or info[key] < 0:
            reasons.append(f"native INFO {key} is unavailable")
    if evidence.get("native_pack") is not True or evidence.get("native_starts") != 1 or evidence.get("contradictory"):
        reasons.append("one native pack/process generation was not verified")
    return {
        "status": "verified" if not reasons else "unverified",
        "reasons": reasons,
        "scope": "loaded_backend_execution_configuration_not_per_request_compute",
        "physical_gpu_index": gpu,
        "cuda_device": observed_gpu,
        "cpu_expert_pool": pool | workers,
        "native_pack": evidence.get("native_pack") is True,
        "native_starts": evidence.get("native_starts"),
        "source_revision": PINNED_STRATA_REVISION,
        "engine_info": {
            key: info.get(key) for key in (*expected, "pool_workers", "expert_slots", "expert_cache_mib", "arena_mib")
        },
        "cpu_expert_dispatch_capability": "pinned_native_pack_CPU_kernels_for_non_GPU_expert_entries",
        "source_contract": [
            "core/expert_source.cpp:3154-3196",
            "program/generate.cpp:4980",
            "kernels/cpu/pool.cpp:759",
        ],
    }


def _native_compute_observation(counters: dict | None, *, verified_native_pack: bool, gpu: int) -> dict:
    """DONE lookups exclude GPU PCIe entries; misses are native CPU entries.

    This is specific to the pinned native-pack verify path, with one GPU and no
    remote expert flags. It covers routed decode experts, not all model stages.
    """
    if not verified_native_pack or counters is None:
        return {
            "scope": "routed_decode_experts_only",
            "counters": counters,
            "cpu_expert_entries": None,
            "gpu_expert_entries": None,
            "units": None,
        }
    cpu_entries = counters["lookups"] - counters["hits"]
    gpu_entries = counters["hits"] + counters["offloaded"]
    units = (["cpu"] if cpu_entries else []) + ([f"cuda:{gpu}"] if gpu_entries else [])
    return {
        "scope": "routed_decode_experts_only",
        "counters": counters,
        "cpu_expert_entries": cpu_entries,
        "gpu_expert_entries": gpu_entries,
        "units": units,
        "evidence": "native_DONE; pinned native-pack miss dispatch; no remote GPU routes",
    }


def validate_strata_load_plan(plan: Any, requested: str) -> None:
    """Agent load gate: verify configuration without inventing compute use.

    This permits an explicitly selected experimental route to load. It neither
    qualifies its task quality/latency nor selects it as an automatic default.
    """
    _validate_strata_loaded_config(plan, requested, expected_backend=BACKEND_NAME)


def _validate_strata_loaded_config(plan: Any, requested: str, *, expected_backend: str) -> None:
    """Shared native-language load proof; public entrypoints fix their own name."""
    try:
        if expected_backend not in {BACKEND_NAME, "external.strata.multimodal.v1"}:
            raise ValueError("unsupported internal Strata backend identity")
        if not isinstance(plan, dict) or not re.fullmatch(r"cpu\+cuda:[0-9]+", requested):
            raise ValueError("invalid Strata requested route")
        if expected_backend == BACKEND_NAME and (
            plan.get("image_route") is not None
            or plan.get("owned_encoder_at_load") is not None
            or "image" in plan.get("declared_modalities", [])
            or plan.get("route_controls", {}).get("image_route") is not None
        ):
            raise ValueError("text load gate refuses image capabilities")
        report = plan["execution_configuration_evidence"]
        if (
            plan.get("backend") != expected_backend
            or plan.get("runtime_revision") != PINNED_STRATA_REVISION
            or plan.get("requested_device") != requested
            or plan.get("verified_execution_configuration") != requested
            or plan.get("placement_evidence_level") != "native_loaded_configuration"
            or report.get("status") != "verified"
            or report.get("scope") != "loaded_backend_execution_configuration_not_per_request_compute"
            or report.get("source_revision") != PINNED_STRATA_REVISION
            or report.get("physical_gpu_index") != int(requested.rsplit(":", 1)[1])
        ):
            raise ValueError("native loaded execution configuration is not verified")
        pool = report["cpu_expert_pool"]
        reconstructed = {
            "gpu": report["cuda_device"],
            "cpu_pool": pool,
            "expert_workers": pool,
            "native_pack": report["native_pack"],
            "native_starts": report["native_starts"],
        }
        verified = _verify_load_configuration(
            reconstructed,
            report["engine_info"],
            plan["fresh_memory_admission"],
            gpu=report["physical_gpu_index"],
            context=plan["context_tokens"],
            kv=plan["kv_type"],
            verify_window=plan["native_verify_window"],
        )
        if verified["status"] != "verified":
            raise ValueError("native load proof is incomplete or contradictory")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError("Strata did not verify its loaded backend execution configuration") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("manifest file names must be relative")
    result = (root / name).resolve(strict=True)
    if not result.is_relative_to(root):
        raise ValueError("file escapes its declared artifact root")
    return result


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _json_request(url: str, body: dict | None, *, token: str, timeout: float, limit: int = 65536) -> Any:
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    # Loopback IPC must not be redirected through a user's HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ResourceUnavailable("Strata response exceeds admitted I/O bound")
    return json.loads(raw)


def _stream_request(
    url: str,
    body: dict,
    *,
    token: str,
    timeout: float,
    limit: int,
    cancelled: threading.Event,
    on_delta: Callable[[str, str], None],
) -> dict:
    """Consume bounded SSE, retaining content/reasoning separately and no tools.

    A stop/length marker followed by [DONE] is required. EOF and tool execution
    extensions cannot masquerade as a completed text request.
    """
    payload = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    parts: dict[str, list[str]] = {"content": [], "reasoning_content": []}
    size = 0
    wire_size = 0
    deadline = time.monotonic() + timeout
    finish = None
    usage = timings = None
    with opener.open(req, timeout=timeout) as response:
        while True:
            if cancelled.is_set():
                raise RuntimeError("Strata request cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("Strata whole-request deadline exceeded")
            row = response.readline(min(limit * 4, 1 << 20) + 1)
            if not row:
                raise RuntimeError("Strata SSE ended before terminal [DONE]")
            wire_size += len(row)
            # Includes JSON overhead, heartbeat lines and empty deltas.
            if len(row) > min(limit * 4, 1 << 20) or wire_size > limit * 16 + (1 << 20):
                raise ResourceUnavailable("Strata SSE exceeds admitted wire bound")
            if not row.startswith(b"data:"):
                continue
            raw = row[5:].strip()
            if raw == b"[DONE]":
                if finish not in {"stop", "length"}:
                    raise RuntimeError("Strata SSE has no complete text finish marker")
                return {key: "".join(value) for key, value in parts.items()} | {
                    "finish_reason": finish,
                    "usage": usage,
                    "timings": timings,
                }
            obj = json.loads(raw)
            if not isinstance(obj, dict) or "error" in obj or obj.get("strata_mcp"):
                raise RuntimeError("Strata returned an error or unsupported tool extension")
            if obj.get("usage") is not None:
                usage = obj["usage"]
            if obj.get("timings") is not None:
                timings = obj["timings"]
            choices = obj.get("choices", [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise RuntimeError("Strata returned more than one choice")
            for choice in choices:
                delta = choice.get("delta") or {}
                if delta.get("tool_calls") or delta.get("function_call"):
                    raise RuntimeError("Strata text stage returned unsupported tool calls")
                for kind in parts:
                    part = delta.get(kind)
                    if part is None or part == "":
                        continue
                    if finish is not None:
                        raise RuntimeError("Strata emitted content after its finish marker")
                    if not isinstance(part, str):
                        raise RuntimeError("Strata returned a non-text delta")
                    size += len(part.encode("utf-8"))
                    if size > limit:
                        raise ResourceUnavailable("Strata output exceeds admitted I/O bound")
                    parts[kind].append(part)
                    # UTF-8 has at most four bytes per code point.
                    for start in range(0, len(part), _MAX_DELTA_BYTES // 4):
                        on_delta(kind, part[start : start + _MAX_DELTA_BYTES // 4])
                if choice.get("finish_reason") is not None:
                    if finish is not None:
                        raise RuntimeError("Strata returned duplicate finish markers")
                    finish = choice["finish_reason"]


def _safe_telemetry(status: Any) -> dict:
    names = ("hit_rate", "pcie_share", "ram_blobs", "file_blobs", "file_mb", "drafts_offered", "drafts_accepted")
    result = {name: None for name in names}
    if isinstance(status, dict):
        for name in names:
            value = status.get(name)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                if name in {"hit_rate", "pcie_share"} and value > 1:
                    continue
                result[name] = value
    result.update(physical_ssd_read_bytes=None, ssd_wait_s=None, actual_io_mode=None, observed_compute_units=None)
    result["logical_file_read_bytes"] = round(result["file_mb"] * 1_000_000) if result["file_mb"] is not None else None
    # The pinned native DONE baseline is captured after prefill (generate.cpp:
    # 9792-9793). Its rounded decimal-MB counter excludes prefill and loading;
    # it cannot establish either whole-request reads or physical drive I/O.
    result["logical_file_read_scope"] = "decode_only_excludes_prefill_and_loading"
    return result


class StrataTextStageClient(StageClientBase):
    """One complete request at a time, with bounded streaming and terminal ACK."""

    _backend_name = BACKEND_NAME
    _supports_images = False

    def __init__(self, metadata, config: dict, ledger, reservation) -> None:
        for name, value in vars(metadata).items():
            setattr(self, name, value)
        self.stage_type = "graph"
        self._ledger, self._reservation = ledger, reservation
        self._generation = uuid.uuid4().hex
        self._epoch = 0
        self._closed = False
        self._active = None
        self._task = None
        self._output = None
        self._agent_stream = None
        self._cancel = threading.Event()
        self._proc = None
        self._diagnostics = None
        self._temporary = None
        self._transport_lock = asyncio.Lock()
        self._ack_pending = None
        self._terminal_published = False
        self._completed_requests = 0
        self._retirement_lock = threading.Lock()
        self._known_children: dict[int, float] = {}
        self._observation_runtime = None
        self._execution_settings = None
        self._execution_modules = None
        self._execution_bridge = None
        self._execution_verifier = None
        self._execution_token = None
        self._execution_report_lock = threading.RLock()
        self._execution_report_request = None
        self._execution_active_request = None
        self._last_execution_report = None
        self._native_module_audit = None
        self._io_request = None
        self._last_io_report = None
        self._io_report_lock = threading.RLock()
        try:
            # StageRuntime passes the backend through StageConfig's OmegaConf
            # representation. Normalize that transport at this boundary so
            # strict manifest validators still inspect real dict/list values.
            if OmegaConf.is_config(config):
                config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
            if not isinstance(config, dict):
                raise ValueError("Strata backend configuration must be a mapping")
            self._timeout = float(config.get("request_timeout_s", 300))
            self._max_io_bytes = _positive(config.get("max_io_bytes", 1 << 20), "max_io_bytes")
            self._max_new_tokens = _positive(config.get("max_new_tokens", 128), "max_new_tokens")
            self._context_tokens = _positive(config.get("context_tokens", 4096), "context_tokens")
            self._start(config)
        except BaseException:
            self.shutdown()
            raise

    def _start(self, config: dict) -> None:
        from vllm_omni.engine.weight_tiers import (
            ArtifactManifest,
            BackendCapabilities,
            PlacementReport,
            WeightTierPlan,
        )

        if config.get("name", self._backend_name) != self._backend_name:
            raise ValueError("unexpected Strata backend name")
        if config.get("runtime_revision") != PINNED_STRATA_REVISION:
            raise ValueError("Strata runtime revision is not the validated protocol revision")
        if not math.isfinite(self._timeout) or self._timeout <= 0 or self._context_tokens <= self._max_new_tokens + 8:
            raise ValueError("invalid Strata timeout or context bound")
        start_timeout = float(config.get("start_timeout_s", 900))
        if not math.isfinite(start_timeout) or start_timeout <= 0:
            raise ValueError("Strata start timeout must be finite and positive")
        for forbidden in ("args", "config_file", "mcp_servers", "before_load", "vision"):
            if config.get(forbidden) is not None:
                raise ValueError(f"Strata supervisor does not accept {forbidden}")
        runtime = Path(config["runtime_root"]).resolve(strict=True)
        artifact_root = Path(config["artifact_root"]).resolve(strict=True)
        pack = Path(config["prepared_model_dir"]).resolve(strict=True)
        runtime_manifest = ArtifactManifest.from_dict(config["runtime_manifest"])
        artifacts = ArtifactManifest.from_dict(config["artifact_manifest"])
        prepared = ArtifactManifest.from_dict(config["prepared_pack_manifest"])
        runtime_files = set(runtime_manifest.verify(runtime))
        if runtime_manifest.revision != PINNED_STRATA_REVISION:
            raise ValueError("runtime file manifest has a different revision")
        script = _contained(runtime, config.get("server_script", "serve/server.py"))
        engine = _contained(
            runtime, config.get("engine_file", "engine/strata.exe" if os.name == "nt" else "engine/strata")
        )
        if script not in runtime_files or engine not in runtime_files:
            raise ValueError("Strata entry points are absent from the runtime hash manifest")
        # Imports are code too. Reject an unpinned helper silently added beside
        # an otherwise pinned server or a different DLL found beside the exe.
        for directory in (runtime / "serve", runtime / "tools", engine.parent):
            if directory.exists():
                for path in directory.rglob("*"):
                    if path.is_file() and path.suffix.lower() in {".py", ".pyc", ".dll", ".so", ".exe"}:
                        if path.resolve() not in runtime_files:
                            raise ValueError("Strata runtime has an unpinned code/dependency file")
        if config.get("execution_observation") is not None:
            if os.name != "nt" or type(self) is not StrataTextStageClient:
                raise ValueError("execution observation currently requires the native Windows text Stage")
            if (
                config.get("observation_runtime") is not None
                or config.get("observation_runtime_identity_sha256") is not None
            ):
                raise ValueError("combined execution and legacy I/O descriptors cannot be mixed")
            if config.get("weight_tier_plan") is None or not config.get("route_id"):
                raise ValueError("execution observation requires an explicit independently admitted tier route")
            self._execution_settings = _execution_settings(config["execution_observation"], runtime)
            minimum_host = (
                _positive(config["expert_ram_budget_bytes"], "expert_ram_budget_bytes")
                + _positive(config["host_overhead_bytes"], "host_overhead_bytes")
                + self._max_io_bytes
                + self._execution_settings["workspace_bytes"]
            )
            if minimum_host > self._reservation.demands.get("host_ram", 0):
                raise ResourceUnavailable("execution observer workspace is not admitted before verification")
            self._execution_modules = _load_execution_adapters(runtime, runtime_files)
            self._observation_runtime = self._execution_modules["strata_exec_runtime"].verify_combined_runtime(
                self._execution_settings["descriptor_file"], runtime
            )
            if self._observation_runtime["identity_sha256"] != self._execution_settings["static_identity_sha256"]:
                raise ValueError("execution runtime changed after route registration")
        elif config.get("observation_runtime") is not None:
            self._observation_runtime = strata_io.verify_observation_runtime(
                config["observation_runtime"], runtime, runtime_files, engine, _BOOTSTRAP
            )
            if (
                config.get("observation_runtime_identity_sha256", self._observation_runtime["identity_sha256"])
                != self._observation_runtime["identity_sha256"]
            ):
                raise ValueError("Strata observation runtime changed after route registration")
        elif config.get("observation_runtime_identity_sha256") is not None:
            raise ValueError("Strata observation identity requires its runtime descriptor")
        # Reject invalid executable/import roots before reading a potentially
        # hundred-GiB model. Full source and converted-pack verification is
        # still mandatory before any native process can start.
        source_files = set(artifacts.verify(artifact_root))
        pack_files = set(prepared.verify(pack))
        conversion = config.get("conversion_manifest")
        if (
            not isinstance(conversion, dict)
            or conversion.get("complete") is not True
            or conversion.get("source_manifest_sha256") != artifacts.manifest_sha256
            or conversion.get("prepared_manifest_sha256") != prepared.manifest_sha256
            or conversion.get("tool_revision") != PINNED_STRATA_REVISION
            or not isinstance(conversion.get("conversions"), list)
        ):
            raise ValueError("prepared pack needs a complete, source-bound conversion manifest")
        for path in pack.rglob("*"):
            if path.is_file() and path.resolve() not in pack_files:
                raise ValueError("prepared model directory contains an unpinned file")
        tokenizer = pack / "tokenizer"
        for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja"):
            if (tokenizer / name).resolve(strict=True) not in pack_files:
                raise ValueError("Strata tokenizer/template must be part of the prepared pack")
        native = _contained(artifact_root, config["native_file"])
        ple = _contained(artifact_root, config["ple_file"])
        if native not in source_files or ple not in source_files:
            raise ValueError("Strata native and PLE files must be in the complete artifact manifest")
        python = Path(config["python_bin"]).resolve(strict=True)
        if _sha256(python) != config["python_sha256"]:
            raise ValueError("Strata Python executable hash mismatch")
        python_environment = _verify_python_environment(python, config.get("python_environment"))
        expert_budget = _positive(config["expert_ram_budget_bytes"], "expert_ram_budget_bytes")
        host_overhead = _positive(config["host_overhead_bytes"], "host_overhead_bytes")
        observer_workspace = self._execution_settings["workspace_bytes"] if self._execution_settings else 0
        observer_receipts = self._execution_settings["receipt_storage_bytes"] if self._execution_settings else 0
        gpu_budget = _positive(config["gpu_budget_bytes"], "gpu_budget_bytes")
        gpu_total = _positive(config["gpu_total_bytes"], "gpu_total_bytes")
        tier_plan = (
            WeightTierPlan.from_dict(config["weight_tier_plan"]) if config.get("weight_tier_plan") is not None else None
        )
        cache_budget = _positive(
            tier_plan.budget.gpu_expert_cache_bytes if tier_plan is not None else config.get("gpu_expert_cache_bytes"),
            "gpu_expert_cache_bytes",
        )
        if config.get("gpu_expert_cache_bytes", cache_budget) != cache_budget:
            raise ValueError("Strata explicit GPU cache budget differs from its typed tier plan")
        if cache_budget > gpu_budget:
            raise ResourceUnavailable("Strata GPU expert cache budget exceeds aggregate GPU reservation")
        cache_control = _native_cache_control(pack, pack_files, cache_budget)
        if gpu_budget > gpu_total:
            raise ResourceUnavailable("Strata GPU budget exceeds physical VRAM")
        gpu_pool = config.get("gpu_pool", "vram:0")
        if expert_budget + host_overhead + self._max_io_bytes + observer_workspace > self._reservation.demands.get(
            "host_ram", 0
        ):
            raise ResourceUnavailable("Strata experts, loading/state/workspace and I/O exceed host reservation")
        if gpu_budget > self._reservation.demands.get(gpu_pool, 0):
            raise ResourceUnavailable("Strata GPU weights/state/workspace exceed GPU reservation")
        gpu = config.get("gpu_index", 0)
        reserve = config.get("vram_reserve_mib", 1024)
        workers = config.get("pool_workers", 8)
        if type(gpu) is not int or gpu < 0 or type(reserve) is not int or reserve < 0:
            raise ValueError("invalid Strata GPU index or reserve")
        # The native Agent's single-GPU shared ledger uses the legacy `vram`
        # name. It is a GPU0 alias only, never a claim on another device.
        if gpu_pool != f"vram:{gpu}" and not (gpu == 0 and gpu_pool == "vram"):
            raise ValueError("Strata GPU pool must match the observed physical GPU index")
        reserve = max(reserve, math.ceil((gpu_total - gpu_budget) / (1 << 20)))
        _positive(workers, "pool_workers")
        kv = config.get("kv_type", "int8")
        if kv not in {"int8", "fp16"}:
            raise ValueError("Strata KV precision must be explicitly supported")
        # No arbitrary engine args: in particular no SAVE, tool server, session
        # persistence, additional GPU, context fitting, or hidden recovery.
        spec = config.get("spec_tokens", 0)
        if type(spec) is not int or spec not in {0, 2, 3, 4, 5, 6, 7, 8}:
            raise ValueError("spec_tokens must be zero (MTP off), or from 2 through 8")
        mtp = config.get("mtp_directory")
        if bool(spec) != bool(mtp):
            raise ValueError("MTP requires both a pinned draft directory and spec_tokens")
        ple_io = config.get("ple_io", "direct")
        if ple_io not in {"direct", "mmap"}:
            raise ValueError("ple_io must be direct or mmap; a resident table needs a separate full-table budget")
        # The native serve verifier needs a >=2-row window even for plain
        # one-token rounds. With no draft model and lookup disabled this is
        # genuinely MTP off; a zero native window cannot start this revision.
        native_verify_window = max(spec, 2)
        args = [
            "--pack",
            str(pack),
            "--gpu",
            str(gpu),
            "--native",
            str(native),
            "--ple-gguf",
            str(ple),
            "--ple-io",
            ple_io,
            "--expert-cache",
            str(cache_control["requested_slots"]),
            "--prefill",
            "auto",
            "--spec",
            str(native_verify_window),
            "--suffix-draft",
            "0",
            "--lookup-chain",
            "0",
            "--max-context",
            str(self._context_tokens),
            "--kv",
            kv,
            "--vram-reserve-mib",
            str(reserve),
            "--mmap-experts",
            "--resident-budget-gib",
            str(expert_budget / (1 << 30)),
            "--pool-workers",
            str(workers),
            "--prompt-cache",
            "0",
            "--conversation-cache-mib",
            "0",
            "--conversation-cache-slots",
            "0",
        ]
        if mtp:
            mtp_path = _contained(pack, mtp)
            if not mtp_path.is_dir() or not any(path.is_relative_to(mtp_path) for path in pack_files):
                raise ValueError("MTP draft files must be part of the pinned prepared manifest")
            args += ["--mtp", str(mtp_path)]
        ple_prefetch = config.get("ple_prefetch", False)
        if type(ple_prefetch) is not bool:
            raise ValueError("ple_prefetch must be boolean")
        if not ple_prefetch:
            args.append("--no-ple-prefetch")
        profile = config.get("expert_profile_file")
        if profile:
            profile_path = _contained(runtime, profile)
            if profile_path not in runtime_files:
                raise ValueError("Strata expert profile is not pinned")
            args += ["--expert-profile", str(profile_path)]
        self._prepare_extension(
            config, runtime, runtime_manifest, runtime_files, artifacts, native, tier_plan, host_overhead, gpu_budget
        )
        self._temporary = tempfile.TemporaryDirectory(prefix="omni-strata-")
        self._token = uuid.uuid4().hex + uuid.uuid4().hex
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self._base_url = f"http://127.0.0.1:{port}"
        self._model_alias = config.get("model_alias", "qwen3.8-flash-next")
        if not isinstance(self._model_alias, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,256}", self._model_alias):
            raise ValueError("invalid Strata model alias")
        server_cfg = {
            "exe": str(engine),
            "args": args,
            "cwd": str(runtime),
            "tokenizer": str(tokenizer),
            "model_name": self._model_alias,
            "gpu": gpu,
            "parallel": 1,
            "api_key": self._token,
            "api_monitor": False,
            "fit_max_tokens": False,
            "tool_call_recovery": False,
            "reasoning_loop_recovery": False,
            "reasoning_close_retry": False,
            "open_browser": False,
            "idle_unload_s": 0,
            "mcp_servers": {},
        }
        self._configure_extension(server_cfg)
        config_path = Path(self._temporary.name) / "server.json"
        config_path.write_text(json.dumps(server_cfg), encoding="utf-8")
        bootstrap_path = Path(self._temporary.name) / "bootstrap.py"
        bootstrap_path.write_text(self._bootstrap_source(), encoding="utf-8")
        env = os.environ.copy()
        for name in tuple(env):
            if name.startswith(("STRATA_", "PYTHON", "OMNI_STRATA_")):
                env.pop(name)
        env["STRATA_ENGINE_READY_S"] = str(start_timeout)
        env["STRATA_RESIDENT_HEADROOM_GIB"] = str(host_overhead / (1 << 30))
        io_prefetch = config.get("io_prefetch", False)
        if type(io_prefetch) is not bool:
            raise ValueError("io_prefetch must be boolean")
        routing_prefetch = config.get("routing_prefetch", False)
        if type(routing_prefetch) is not bool:
            raise ValueError("routing_prefetch must be boolean")
        env["STRATA_LOOKAHEAD"] = "1" if routing_prefetch else "0"
        env["STRATA_IO_PREFETCH"] = "1" if io_prefetch else "0"
        env["STRATA_IO_PF_AHEAD"] = "1" if io_prefetch else "0"
        io_mode = config.get("io_mode", "auto")
        if io_mode not in {"auto", "buffered", "direct"}:
            raise ValueError("io_mode must be auto, buffered or direct")
        if io_prefetch and (os.name == "nt" or io_mode != "buffered"):
            raise ValueError("pinned Strata I/O worker prefetch requires Linux buffered mode")
        if io_mode != "auto":
            env["STRATA_UNBUFFERED_LOAD"] = "1" if io_mode == "direct" else "0"
        if tier_plan is not None:
            if (
                tier_plan.artifact_manifest_sha256 != artifacts.manifest_sha256
                or tier_plan.backend != self._backend_name
                or tier_plan.backend_revision != PINNED_STRATA_REVISION
            ):
                raise ValueError("Strata tier plan is bound to a different artifact or runtime")
            if config.get("route_id") is not None and tier_plan.route_id != config["route_id"]:
                raise ValueError("Strata tier plan route identity differs from its launch configuration")
            if tier_plan.cpu_block_layers or tier_plan.cpu_expert_layers:
                raise ValueError("Strata owns dynamic expert compute; fixed CPU layer controls are unsupported")
            tier_plan.check_capabilities(
                BackendCapabilities(
                    cpu_block_offload=False,
                    cpu_expert_offload=True,
                    bounded_ssd_expert_cache=True,
                    lookup_tables_on_demand=True,
                    vision=self._supports_images,
                    mtp=True,
                )
            )
            budget = tier_plan.budget
            expected = budget.resource_demands(
                gpu_pool=gpu_pool,
                include_wsl="wsl_ram" in self._reservation.demands,
                include_windows_commit="windows_commit" in self._reservation.demands,
            )
            if expected != dict(self._reservation.demands):
                raise ValueError("Strata tier plan and exact stage resource lease differ")
            if budget.cpu_expert_cache_bytes != expert_budget:
                raise ValueError("Strata expert RAM budget differs from the tier plan")
            if budget.host_transfer_bytes < self._max_io_bytes:
                raise ValueError("Strata tier plan does not cover admitted host I/O")
            if max(budget.gpu_steady_bytes, budget.gpu_loading_peak_bytes) > gpu_budget:
                raise ValueError("Strata GPU budget does not cover its tier plan components/peak")
            host_peak = max(budget.host_steady_bytes, budget.host_loading_peak_bytes)
            if host_overhead + self._max_io_bytes + observer_workspace < host_peak - expert_budget:
                raise ValueError("Strata overhead does not cover loading/state/lookup/workspace and I/O")
            if self._execution_settings is not None:
                if budget.host_workspace_bytes < observer_workspace or budget.ssd_temporary_bytes < observer_receipts:
                    raise ResourceUnavailable("tier plan omits separate execution observer workspace/receipts")
                if (
                    budget.ssd_artifact_bytes
                    < artifacts.total_size_bytes + prepared.total_size_bytes + runtime_manifest.total_size_bytes
                ):
                    raise ResourceUnavailable("combined observer SSD admission omits runtime provenance/adapter files")
            if budget.ssd_artifact_bytes < artifacts.total_size_bytes + prepared.total_size_bytes:
                raise ValueError("Strata SSD budget omits source shards or prepared pack files")
            if tier_plan.mtp != bool(spec) or tier_plan.prefetch != any((ple_prefetch, routing_prefetch, io_prefetch)):
                raise ValueError("Strata MTP/prefetch switches differ from its tier plan")
        memory_snapshot = _probe_memory(gpu)
        _check_live_memory(memory_snapshot, dict(self._reservation.demands), gpu_pool, gpu_total)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        observer_nonce = uuid.uuid4().hex
        env["OMNI_STRATA_OBSERVER_NONCE"] = observer_nonce
        if self._observation_runtime is not None:
            adapter_path = Path(self._temporary.name) / "strata_io.py"
            adapter_path.write_bytes(Path(strata_io.__file__).read_bytes())
            expected_io = (
                self._execution_modules["strata_exec_live"].ENGINE_ROLE_IO_SHA
                if self._execution_settings
                else self._observation_runtime["io_adapter_sha256"]
            )
            if _sha256(adapter_path) != expected_io:
                raise RuntimeError("Strata trusted I/O adapter changed before launch")
            env["OMNI_STRATA_IO_ADAPTER"] = str(adapter_path)
            env["OMNI_STRATA_IO_ADAPTER_SHA256"] = expected_io
            env["OMNI_STRATA_IO_GENERATION"] = self._generation
        if self._execution_settings is not None:
            import secrets

            self._execution_token = secrets.token_hex(32)
            for name, prefix in (("strata_exec", "PARSER"), ("strata_exec_bridge", "ADAPTER")):
                module = self._execution_modules[name]
                path = Path(self._temporary.name) / (name + ".py")
                path.write_bytes(Path(module.__file__).read_bytes())
                if _sha256(path) != _EXECUTION_ADAPTER_SHA256[name]:
                    raise RuntimeError("execution adapter changed before supervisor launch")
                env["OMNI_STRATA_EXEC_" + prefix] = str(path)
                env["OMNI_STRATA_EXEC_" + prefix + "_SHA256"] = _EXECUTION_ADAPTER_SHA256[name]
            env["OMNI_STRATA_EXEC_TOKEN"] = self._execution_token
            env["OMNI_STRATA_EXEC_GENERATION"] = self._generation
        self._extension_environment(env)
        self._proc = subprocess.Popen(
            [
                str(python),
                "-I",
                "-X",
                "utf8",
                "-X",
                "pycache_prefix=" + str(Path(self._temporary.name) / "pycache"),
                str(bootstrap_path),
                str(engine),
                str(script),
                "--engine",
                "strata",
                "--config",
                str(config_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=runtime,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            start_new_session=os.name != "nt",
        )
        self._diagnostics = self._new_diagnostics(
            self._proc.stdout,
            Path(config["log_file"]) if config.get("log_file") else None,
            observer_nonce=observer_nonce,
            **(
                {
                    "execution_token": self._execution_token,
                    "execution_module": self._execution_modules["strata_exec_bridge"],
                }
                if self._execution_settings is not None
                else {}
            ),
        )
        deadline = time.monotonic() + start_timeout
        while True:
            self._snapshot_children()
            if self._proc.poll() is not None:
                raise RuntimeError(f"Strata worker exited during model loading: exit_code={self._proc.returncode}")
            try:
                health = _json_request(self._base_url + "/health", None, token=self._token, timeout=1)
                if health.get("loaded") is True and health.get("status") == "ok":
                    break
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Strata worker did not become ready")
            time.sleep(0.1)
        props = _json_request(self._base_url + "/props", None, token=self._token, timeout=5)
        if (
            health.get("service") != "strata"
            or health.get("images") is not self._supports_images
            or props.get("total_slots") != 1
            or props.get("model_alias") != self._model_alias
            or props.get("default_generation_settings", {}).get("n_ctx") != self._context_tokens
        ):
            raise RuntimeError("Strata did not load the declared single-slot text configuration")
        if any(
            code in self._diagnostics.records
            for code in ("strata diagnostic: resident_budget_not_kept", "strata diagnostic: resident_budget_clamped")
        ):
            raise ResourceUnavailable("Strata could not keep its declared RAM expert budget")
        if (
            os.name == "nt"
            and "strata supervisor: native contained in Windows kill-on-close job" not in self._diagnostics.records
        ):
            raise RuntimeError("Strata did not prove Windows native process containment")
        native_evidence = self._diagnostics.wait_load_evidence()
        native_metrics = _json_request(self._base_url + "/metrics", None, token=self._token, timeout=5)
        native_info = native_metrics.get("engine", {}) if isinstance(native_metrics, dict) else {}
        for key, expected in {
            "engine": PINNED_STRATA_VERSION,
            "context": self._context_tokens,
            "kv": kv,
            "spec": native_verify_window,
            "lookup": 0,
            "conversation_cache_mib": 0,
            "conversation_cache_slots": 0,
        }.items():
            if key in native_info and native_info[key] != expected:
                raise RuntimeError(f"Strata native INFO disagrees with the pinned/requested {key}")
        cache_bounds = _verify_cache_bounds(native_info, cache_control, expert_budget, profiled=bool(profile))
        if self._observation_runtime is not None and self._execution_settings is None:
            identity = self._diagnostics.owned_native_identity()
            self._native_module_audit = strata_io.audit_selected_loaded_modules(
                identity, runtime, self._observation_runtime
            )
            if self._native_module_audit["status"] != "verified":
                raise RuntimeError(
                    "Strata observed runtime selected module binding is unverified: "
                    + json.dumps(self._native_module_audit, sort_keys=True)
                )
            self._diagnostics.bind_io_observer(self._generation, self._observation_runtime)
        if tier_plan is not None:
            fixed_gpu = tier_plan.budget.gpu_steady_bytes - cache_budget
            aggregate_estimate = fixed_gpu + cache_bounds["gpu_expert_cache"]["allocation_upper_bytes"]
            if aggregate_estimate > gpu_budget:
                raise ResourceUnavailable(
                    "Strata controlled cache plus declared fixed GPU components exceeds reservation"
                )
            cache_bounds["aggregate_gpu_declared_components_upper_bytes"] = aggregate_estimate
            cache_bounds["aggregate_gpu_budget_bytes"] = gpu_budget
        route_controls = {
            "schema": _CACHE_CONTROL_SCHEMA,
            "runtime_revision": PINNED_STRATA_REVISION,
            "runtime_manifest_sha256": runtime_manifest.manifest_sha256,
            "artifact_manifest_sha256": artifacts.manifest_sha256,
            "prepared_manifest_sha256": prepared.manifest_sha256,
            "gpu_expert_cache": cache_control,
            "ram_expert_cache_budget_bytes": expert_budget,
            "gpu_budget_bytes": gpu_budget,
            "vram_reserve_mib": reserve,
            "context_tokens": self._context_tokens,
            "kv_type": kv,
            "spec_tokens": spec,
            "native_verify_window": native_verify_window,
            "prefetch": {"ple": ple_prefetch, "routing": routing_prefetch, "io": io_prefetch},
            "io_mode": io_mode,
            "ple_io": ple_io,
            "expert_profile_file": profile,
        }
        if self._observation_runtime is not None:
            route_controls["observation_runtime"] = self._observation_runtime
        if self._execution_settings is not None:
            route_controls["execution_observation"] = {
                "schema": _EXECUTION_CONFIG_SCHEMA,
                "static_identity_sha256": self._observation_runtime["identity_sha256"],
                "adapter_sha256": dict(_EXECUTION_ADAPTER_SHA256),
                "workspace_bytes": observer_workspace,
                "receipt_storage_bytes": observer_receipts,
                "workspace_scope": "separate_declared_observer_allowance_not_a_measured_heap_hard_cap",
                "Agent_consumer_workspace_reused": False,
            }
        route_controls_sha256 = hashlib.sha256(
            json.dumps(route_controls, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._load_configuration = _verify_load_configuration(
            native_evidence,
            native_info,
            memory_snapshot,
            gpu=gpu,
            context=self._context_tokens,
            kv=kv,
            verify_window=native_verify_window,
        )
        self._physical_gpu = gpu
        verified_configuration = self._load_configuration["status"] == "verified"
        self.execution_plan = {
            "backend": self._backend_name,
            "stage_id": self.stage_id,
            "worker_generation": self._generation,
            "runtime_revision": PINNED_STRATA_REVISION,
            "python_environment": python_environment,
            "fresh_memory_admission": memory_snapshot,
            "supervisor_bootstrap_sha256": hashlib.sha256(self._bootstrap_source().encode()).hexdigest(),
            "observation_runtime": self._observation_runtime,
            "selected_native_module_audit_at_load": self._native_module_audit,
            "runtime_manifest_sha256": runtime_manifest.manifest_sha256,
            "artifact_manifest_sha256": artifacts.manifest_sha256,
            "prepared_manifest_sha256": prepared.manifest_sha256,
            "conversion_manifest": conversion,
            "model_alias": self._model_alias,
            "requested_device": f"cpu+cuda:{gpu}",
            "observed_model_placement": None,
            "observed_compute_units": None,
            "verified_execution_configuration": f"cpu+cuda:{gpu}" if verified_configuration else None,
            "execution_configuration_evidence": self._load_configuration,
            "actual_io_mode": None,
            "expert_ram_budget_bytes": expert_budget,
            "host_overhead_bytes": host_overhead,
            "gpu_budget_bytes": gpu_budget,
            "gpu_expert_cache_control": cache_control,
            "expert_cache_component_bounds": cache_bounds,
            "route_controls": route_controls,
            "route_controls_sha256": route_controls_sha256,
            "measurement_identity_note": "explicit cache v2 differs from historical auto-cache routes; remeasure",
            "gpu_aggregate_hard_cap_verified": False,
            "gpu_total_bytes": gpu_total,
            "vram_reserve_mib": reserve,
            "gpu_pool": gpu_pool,
            "reserved_bytes": dict(self._reservation.demands),
            "worker_pid": self._proc.pid,
            "context_tokens": self._context_tokens,
            "max_new_tokens": self._max_new_tokens,
            "kv_type": kv,
            "max_io_bytes": self._max_io_bytes,
            "request_capacity": 1,
            "batch_size": 1,
            "mtp_enabled": bool(spec),
            "spec_tokens": spec,
            "native_verify_window": native_verify_window,
            "suffix_drafting_enabled": False,
            "lookup_chain_enabled": False,
            "ple_prefetch_requested": ple_prefetch,
            "routing_prefetch_requested": routing_prefetch,
            "io_prefetch_requested": io_prefetch,
            "io_mode_requested": io_mode,
            "ple_io_requested": ple_io,
            "file_tier_io_policy_at_load": self._diagnostics.file_tier_io_policy,
            "file_tier_io_policy_note": "observed policy; individual direct-read fallback is not exposed by HTTP",
            "file_cache_peak_bytes": None,
            "ssd_cache_budget_verified": False,
            "three_tier_memory_qualified": False,
            "prompt_cache_enabled": False,
            "stateful_session": "Strata-owned; prompt/conversation caches disabled",
            "placement_evidence_level": "native_loaded_configuration" if verified_configuration else "unverified",
            "evidence": "B",
            "qualification": "experimental_backend_loaded",
            "diagnostics": list(self._diagnostics.records),
            "weight_tier_plan": tier_plan.to_dict() if tier_plan is not None else None,
            "weight_tier_evidence": "declared_admission_budget_not_measured_allocation",
            "requested_storage": {
                "gpu_hot_expert_cache": "gpu",
                "bounded_cpu_expert_cache": "ram",
                "source_weight_files": "ssd",
                "ple_table_files": "ssd_direct_reads" if ple_io == "direct" else "ssd_with_ram_file_pages",
            },
            "placement_report": PlacementReport().to_dict(),
        }
        # Resident snapshot only. Observers must recheck the exact PID creation
        # time while sampling; this does not cover the earlier loading peak.
        self.execution_plan["gpu_observer_identity"] = self.gpu_observer_identity()
        self._finish_extension_plan()
        if self._execution_settings is not None:
            settings = self._execution_settings
            verifier = self._execution_modules["strata_exec_live"].create_live_binding_verifier(
                self,
                settings["descriptor_file"],
                runtime,
                settings["source_context_file"],
                Path(settings["receipt_root"]) / ("generation-" + self._generation),
            )
            self._execution_verifier = verifier
            bridge = self._execution_modules["strata_exec_bridge"].ParentExecutionBridge(
                settings["descriptor_file"],
                runtime,
                expected_owner=verifier.owner,
                token=self._execution_token,
                verify_combined_runtime=verifier.verify_static_identity,
                verify_live_binding=verifier,
            )
            if verifier.static != self._observation_runtime:
                raise RuntimeError("execution runtime changed between admission and actual load")
            self._execution_bridge = bridge
            verifier.attach_bridge(bridge)
            self._diagnostics.bind_execution_bridge(bridge)
            self._diagnostics.bind_io_observer(self._generation, self._observation_runtime)

    # Small optional complete-model hooks; the text route retains its exact
    # bootstrap bytes, controls and request serialization.
    def _prepare_extension(
        self, config, runtime, runtime_manifest, runtime_files, artifacts, native, tier_plan, host_overhead, gpu_budget
    ):
        if config.get("image_route") is not None:
            raise ValueError("text stage refuses image route capabilities")

    def _configure_extension(self, server_cfg):
        pass

    def _bootstrap_source(self):
        return _EXECUTION_BOOTSTRAP if self._execution_settings is not None else _BOOTSTRAP

    def _extension_environment(self, env):
        pass

    def _new_diagnostics(self, *args, **kwargs):
        return _DiagnosticLog(*args, **kwargs)

    def _finish_extension_plan(self):
        pass

    def _prompt_fields(self):
        return {"text", "max_tokens", "temperature", "stream_agent"}

    def _prepare_extension_prompt(self, prompt):
        pass

    def _http_content(self, text):
        return text

    def _extension_telemetry(self, request, telemetry):
        pass

    def _check_extension_health(self):
        pass

    def _extension_drained(self):
        return True

    def gpu_observer_identity(self) -> dict | None:
        """Owned native process + verified GPU identity for read-only metrics.

        No process-name scan or supervisor PID is a substitute for the native
        Popen identity observed by the bootstrap. This proves identity only.
        """
        if (
            os.name != "nt"
            or self._closed
            or self._proc is None
            or self._proc.poll() is not None
            or self._diagnostics is None
            or self._load_configuration.get("status") != "verified"
        ):
            return None
        identity = self._diagnostics.owned_native_identity()
        memory = self.execution_plan["fresh_memory_admission"]
        if (
            identity is None
            or identity["pid"] == self._proc.pid
            or _windows_process_creation_filetime(identity["pid"]) != identity["creation_filetime_100ns"]
            or not all(memory.get(key) for key in ("gpu_uuid", "gpu_pci_bus_id", "gpu_name_sha256"))
        ):
            return None
        return identity | {
            "status": "verified",
            "worker_generation": self._generation,
            "gpu": {
                "uuid": memory["gpu_uuid"],
                "pci_bus_id": memory["gpu_pci_bus_id"],
                "name_sha256": memory["gpu_name_sha256"],
            },
        }

    def last_execution_observation(self) -> dict | None:
        with self._execution_report_lock:
            return copy.deepcopy(self._last_execution_report)

    def _finish_execution_observation(self, request, *, completed, lifecycle_outcome):
        if self._execution_bridge is None or request is None:
            return None
        key = (request.request_id, request.epoch, request.worker_generation)
        with self._execution_report_lock:
            if self._execution_report_request == key:
                return copy.deepcopy(self._last_execution_report)
            if self._execution_active_request != key:
                return None  # Startup/begin failure cannot invent an active request.
            failure = None
            try:
                report = self._diagnostics.finish_execution_request(
                    request,
                    omni_completed=completed and not self._cancel.is_set(),
                    cancellation_requested=self._cancel.is_set(),
                    lifecycle_outcome=lifecycle_outcome,
                )
                detached = copy.deepcopy(report)
            except Exception as error:
                failure = error
                report = {
                    "schema": "omni-strata-stage-execution-observation-failure-v1",
                    "status": "unavailable",
                    "complete": False,
                    "request_binding": {"request_id": request.request_id, "epoch": request.epoch},
                    "worker_generation": request.worker_generation,
                    "reason": "execution_report_failed",
                    "failure_type": type(error).__name__,
                    "native_observation": None,
                    "physical_ssd_read_bytes": None,
                    "runtime_qualification": False,
                    "default_eligible": False,
                }
                detached = copy.deepcopy(report)
            self._execution_report_request = key
            self._execution_active_request = None
            self._last_execution_report = detached
            if failure is not None and lifecycle_outcome == "normal":
                # Retire the worker through the existing error path. A broken
                # observation channel cannot silently authorize the next turn.
                raise RuntimeError("Strata execution report could not be detached") from failure
            return report

    def _retire_execution_observer(self, drained):
        # Reader/process drain is proved by the existing retained-handle path.
        # Never clear a verifier still reachable from an active reader.
        with self._execution_report_lock:
            if drained and self._execution_verifier is not None:
                try:
                    self._execution_verifier.close()
                except Exception:
                    return False  # Retain ownership if observer cleanup failed.
                self._execution_verifier = None
            return drained

    def _finish_failure_observations(self, request, *, reason, lifecycle_outcome, drained):
        io_report = execution_report = None
        try:
            try:
                io_report = self._finish_io_observation(request, completed=False, reason=reason)
            except Exception as error:
                io_report = {
                    "schema": "omni-strata-stage-io-observation-failure-v1",
                    "status": "unavailable",
                    "reason": "io_report_failed",
                    "failure_type": type(error).__name__,
                    "native_observation": None,
                    "physical_ssd_read_bytes": None,
                    "runtime_qualification": False,
                }
            try:
                execution_report = self._finish_execution_observation(
                    request, completed=False, lifecycle_outcome=lifecycle_outcome
                )
            except Exception as error:
                execution_report = {
                    "schema": "omni-strata-stage-execution-observation-failure-v1",
                    "status": "unavailable",
                    "reason": "execution_cleanup_report_failed",
                    "failure_type": type(error).__name__,
                    "native_observation": None,
                    "physical_ssd_read_bytes": None,
                    "runtime_qualification": False,
                    "default_eligible": False,
                }
        finally:
            drained = self._retire_execution_observer(drained)
        return io_report, execution_report, drained

    def last_io_observation(self) -> dict | None:
        """Bounded last request report, retained after drain; loaded plan stays immutable."""
        with self._io_report_lock:
            return copy.deepcopy(self._last_io_report)

    def _begin_io_observation(self, request: StageRequest) -> None:
        if self._observation_runtime is None:
            return
        with self._io_report_lock:
            self._io_request = request
            observer = self._diagnostics.io_observer
            if observer is not None and not observer.retired:
                observer.begin(request.request_id, request.epoch)

    def _finish_io_observation(
        self, request: StageRequest | None, *, completed: bool, reason: str | None = None
    ) -> dict | None:
        if self._observation_runtime is None or request is None:
            return None
        with self._io_report_lock:
            if self._last_io_report is not None and all(
                self._last_io_report.get(key) == value
                for key, value in (
                    ("request_id", request.request_id),
                    ("epoch", request.epoch),
                    ("generation", request.worker_generation),
                )
            ):
                return copy.deepcopy(self._last_io_report)
            observer = self._diagnostics.io_observer if self._diagnostics is not None else None
            if observer is not None and observer.active is not None:
                if completed and _windows_process_creation_filetime(observer.pid) != observer.created:
                    completed, reason = False, "owned_native_identity_unverified"
                report = self._diagnostics.finish_io_request(request, completed=completed, reason=reason)
            else:
                # The neural output may still complete after this observation
                # generation was retired. Never borrow its previous intervals.
                report = {
                    "schema": "omni-strata-request-io-observation-v1",
                    "status": "incomplete",
                    "generation": request.worker_generation,
                    "request_id": request.request_id,
                    "epoch": request.epoch,
                    "runtime_identity_sha256": self._observation_runtime["identity_sha256"],
                    "native_request_seq": None,
                    "scope": strata_io.SCOPE,
                    "raw_snapshots": [],
                    "intervals": {},
                    "physical_ssd_read_bytes": None,
                    "three_tier_memory_qualified": False,
                    "loading_covered": False,
                    "reasons": [reason or "observer_generation_unavailable"],
                }
            if (
                self._observation_runtime.get("schema") == "omni-strata-combined-static-runtime-identity-v2"
                and report.get("schema") == "omni-strata-request-io-observation-v1"
            ):
                # A retired/missing combined observer keeps the combined scope.
                report["schema"] = "omni-strata-combined-request-io-observation-v1"
                report["runtime_identity_schema"] = self._observation_runtime["schema"]
                report["combined_io_binding"] = strata_io.validate_combined_io_identity(self._observation_runtime) | {
                    "io_adapter_sha256": strata_io.adapter_source_sha256()
                }
            self._last_io_report = copy.deepcopy(report)
            return report

    def _snapshot_children(self) -> None:
        if self._proc is None:
            return
        import psutil

        try:
            for child in psutil.Process(self._proc.pid).children(recursive=True):
                self._known_children[child.pid] = child.create_time()
        except psutil.NoSuchProcess:
            pass

    async def add_request_async(self, request_id: str, prompt: Any, params: Any = None) -> None:
        async with self._transport_lock:
            self.check_health()
            if self._active is not None:
                raise ResourceUnavailable("Strata has an unacknowledged request; capacity is one")
            health = await asyncio.to_thread(
                _json_request, self._base_url + "/health", None, token=self._token, timeout=5
            )
            self.check_health()
            if health.get("loaded") is not True:
                # Upstream can auto-restart a dead native child. Require a new
                # Omni route generation instead of silently reusing this lease.
                self.shutdown()
                raise RuntimeError("Strata native worker exited; explicit route reload required")
            if not isinstance(prompt, dict) or not isinstance(prompt.get("text"), str) or not prompt["text"]:
                raise ValueError("Strata requires nonempty model-adapter text")
            if set(prompt) - self._prompt_fields():
                raise ValueError("Strata text stage refuses unsupported prompt fields")
            text = prompt["text"]
            if len(text.encode()) > self._max_io_bytes:
                raise ResourceUnavailable("Strata prompt exceeds admitted I/O bound")
            maximum = prompt.get("max_tokens", self._max_new_tokens)
            if type(maximum) is not int or not 0 < maximum <= self._max_new_tokens:
                raise ValueError("Strata token limit exceeds admitted maximum")
            if prompt.get("temperature", 0) != 0:
                raise ValueError("Strata v1 supports deterministic temperature=0")
            streaming = prompt.get("stream_agent", False)
            if type(streaming) is not bool:
                raise ValueError("stream_agent must be boolean")
            self._prepare_extension_prompt(prompt)
            self._epoch += 1
            request = StageRequest(request_id, self.stage_id, self._epoch, self._generation)
            self._active = request_id
            self._terminal_published = False
            self._cancel.clear()
            try:
                self._begin_io_observation(request)
                if self._execution_bridge is not None:
                    self._diagnostics.begin_execution_request(request)
                    self._execution_active_request = (request.request_id, request.epoch, request.worker_generation)
            except BaseException:
                self.shutdown()
                raise
            self._agent_stream = asyncio.Queue(maxsize=64) if streaming else None
            self._task = asyncio.create_task(self._run(request, text, maximum, self._agent_stream))
            self._task.add_done_callback(lambda _: self._apply_pending_ack())

    async def _run(self, request, text: str, maximum: int, queue) -> None:
        from vllm_omni.engine.weight_tiers import ComputePlacement, PlacementReport

        started = time.perf_counter()
        times, events = [], []
        visible_times = []
        loop = asyncio.get_running_loop()

        def delta(kind: str, part: str) -> None:
            if self._cancel.is_set() or request.epoch != self._epoch or self._closed:
                raise RuntimeError("retired Strata generation")
            if len(events) >= _MAX_EVENTS:
                raise ResourceUnavailable("Strata output exceeds admitted event count")
            stamp = time.perf_counter()
            times.append(stamp - started)
            if kind == "content":
                visible_times.append(stamp - started)
            event = StageEvent(
                request.request_id,
                self.stage_id,
                request.epoch,
                len(events) + 1,
                "text",
                self._generation,
                payload_nbytes=len(part.encode()),
                started_monotonic_ns=int(started * 1e9),
                emitted_monotonic_ns=int(stamp * 1e9),
            )
            events.append(dataclasses.asdict(event) | {"channel": kind, "bytes": len(part.encode())})
            if queue is not None and kind == "content":
                ticket = asyncio.run_coroutine_threadsafe(queue.put((part, stamp)), loop)
                try:
                    while True:
                        try:
                            ticket.result(timeout=min(0.1, self._timeout))
                            break
                        except concurrent.futures.TimeoutError:
                            if self._cancel.is_set() or self._closed:
                                raise RuntimeError("Strata stream cancelled under backpressure")
                            if time.perf_counter() - started > self._timeout:
                                raise TimeoutError("Strata consumer did not acknowledge stream credits")
                finally:
                    if not ticket.done():
                        ticket.cancel()

        try:
            self._snapshot_children()
            result = await asyncio.to_thread(
                _stream_request,
                self._base_url + "/v1/chat/completions",
                {
                    "model": self._model_alias,
                    "messages": [{"role": "user", "content": self._http_content(text)}],
                    "max_tokens": maximum,
                    "temperature": 0,
                    "n": 1,
                    "strata_mcp": False,
                    "strata_checkpoint": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                token=self._token,
                timeout=self._timeout,
                limit=self._max_io_bytes,
                cancelled=self._cancel,
                on_delta=delta,
            )
            # Native server rejects prompt+generation beyond context; fitting is
            # disabled, so refusal propagates instead of silently truncating.
            stats = await asyncio.to_thread(
                _json_request, self._base_url + "/metrics", None, token=self._token, timeout=5, limit=self._max_io_bytes
            )
            self._completed_requests += 1
            if (
                not isinstance(stats, dict)
                or stats.get("totals", {}).get("requests") != self._completed_requests
                or not isinstance(stats.get("requests"), list)
                or not stats["requests"]
            ):
                raise RuntimeError("Strata metrics do not identify the completed single-slot request")
            status = stats["requests"][0]
            native_counters = await asyncio.to_thread(
                self._diagnostics.completed_native_request, self._completed_requests
            )
            self.check_health()
            io_report = await asyncio.to_thread(
                self._finish_io_observation,
                request,
                completed=native_counters is not None,
                reason=None if native_counters is not None else "native_done_not_verified",
            )
            terminal = StageEvent(
                request.request_id,
                self.stage_id,
                request.epoch,
                len(events) + 1,
                "text",
                self._generation,
                terminal=True,
            )
            telemetry = _safe_telemetry(status)
            telemetry["file_tier_io_policy"] = self._diagnostics.file_tier_io_policy
            telemetry["direct_read_fallback_count"] = None
            native_compute = _native_compute_observation(
                native_counters,
                verified_native_pack=self._load_configuration["status"] == "verified",
                gpu=self._physical_gpu,
            )
            telemetry["native_compute"] = native_compute
            telemetry["native_io_observation"] = io_report
            self._extension_telemetry(request, telemetry)
            output = OmniRequestOutput(
                request_id=request.request_id,
                prompt=text,
                stage_id=self.stage_id,
                final_output_type="text",
                outputs=[CompletionOutput(0, result["content"], [], None, None, finish_reason=result["finish_reason"])],
                _custom_output={
                    "stage_event": dataclasses.asdict(terminal),
                    "reasoning_content": result["reasoning_content"],
                },
                metrics={
                    "strata_wall_s": time.perf_counter() - started,
                    "first_token_s": times[0] if times else None,
                    "delta_timestamps_s": times,
                    "first_visible_token_s": visible_times[0] if visible_times else None,
                    "timestamp_note": "SSE delta receipt after stage worker submission, not individual token clocks",
                    "stage_events": events,
                    "usage": result["usage"],
                    "runtime_timings": result["timings"],
                    "runtime_telemetry": telemetry,
                    "placement_report": PlacementReport(
                        compute=(
                            ComputePlacement(
                                "routed_decode_experts", tuple(native_compute["units"]), native_compute["evidence"]
                            ),
                        )
                        if native_compute["units"]
                        else (),
                        logical_read_bytes=telemetry["logical_file_read_bytes"],
                        expert_cache_hit_ratio=telemetry["hit_rate"],
                    ).to_dict()
                    | {"logical_read_scope": telemetry["logical_file_read_scope"]},
                },
            )
            if self._execution_bridge is not None:
                execution_report = await asyncio.to_thread(
                    self._finish_execution_observation, request, completed=True, lifecycle_outcome="normal"
                )
                telemetry["native_execution_observation"] = execution_report
                output.metrics["strata_wall_s"] = time.perf_counter() - started
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A truncated stream/error might leave native work active. Retire
            # the route before its reservation can be reused.
            self._cancel.set()
            self._closed = True
            drained = await asyncio.to_thread(self._terminate)
            io_report, execution_report, drained = self._finish_failure_observations(
                request, reason="request_failed", lifecycle_outcome="error", drained=drained
            )
            self._ledger.release(self._reservation, drained=drained)
            output = OmniRequestOutput.from_error(request.request_id, f"Strata request failed: {type(exc).__name__}")
            if io_report is not None:
                output.metrics = {"runtime_telemetry": {"native_io_observation": io_report}}
            if execution_report is not None:
                if output.metrics is None:
                    output.metrics = {}
                output.metrics.setdefault("runtime_telemetry", {})["native_execution_observation"] = execution_report
            output.stage_id = self.stage_id
            output._custom_output = {
                "stage_event": dataclasses.asdict(
                    StageEvent(
                        request.request_id,
                        self.stage_id,
                        request.epoch,
                        len(events) + 1,
                        "error",
                        self._generation,
                        terminal=True,
                    )
                )
            }
        if request.epoch != self._epoch:
            return
        output._stage_release = lambda: loop.call_soon_threadsafe(
            self.acknowledge, request.request_id, request.epoch, request.worker_generation
        )
        self._terminal_published = True
        self._output = output
        if queue is not None:
            await queue.put(None)

    async def receive_agent_delta(self, request_id: str) -> tuple[str, float] | None:
        if self._active != request_id or self._agent_stream is None:
            raise ValueError("no active Strata stream for this request")
        return await self._agent_stream.get()

    def get_graph_output_nowait(self) -> OmniRequestOutput | None:
        output, self._output = self._output, None
        return output

    def acknowledge(self, request_id: str, epoch: int, generation: str) -> None:
        if self._active != request_id or epoch != self._epoch or generation != self._generation:
            return
        if not self._terminal_published:
            return
        if self._task is not None and not self._task.done():
            self._ack_pending = (request_id, epoch, generation)
            return
        self._active = self._task = self._agent_stream = self._ack_pending = None

    def _apply_pending_ack(self) -> None:
        if self._ack_pending:
            self.acknowledge(*self._ack_pending)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if self._active not in request_ids:
            return
        self._cancel.set()
        self._closed = True
        self._epoch += 1
        self._output = None
        if self._agent_stream is not None:
            while not self._agent_stream.empty():
                self._agent_stream.get_nowait()
            self._agent_stream.put_nowait(None)
        drained = await asyncio.to_thread(self._terminate)
        _, _, drained = self._finish_failure_observations(
            self._io_request, reason="request_cancelled", lifecycle_outcome="cancelled", drained=drained
        )
        if self._task is not None:
            done, _ = await asyncio.wait({self._task}, timeout=5)
            if not done:
                self._ledger.release(self._reservation, drained=False)
                self._task.add_done_callback(lambda _: self._ledger.release(self._reservation, drained=drained))
            else:
                self._ledger.release(self._reservation, drained=drained)
        else:
            self._ledger.release(self._reservation, drained=drained)
        self._active = self._agent_stream = self._ack_pending = None

    def _terminate(self) -> bool:
        with self._retirement_lock:
            proc = self._proc
            if proc is None:
                return True
            try:
                import psutil
            except ImportError:
                return False
            try:
                self._snapshot_children()
            except (OSError, psutil.Error):
                # Unknown descendants must not become a successful release.
                return False
            targets = []
            try:
                for pid, created in self._known_children.items():
                    try:
                        child = psutil.Process(pid)
                        if child.create_time() == created:
                            targets.append(child)
                    except psutil.NoSuchProcess:
                        pass
                if os.name != "nt":
                    # The group leader may exit before child inventory. Its
                    # reparented native child still owns weights and the group;
                    # a dead leader alone never proves the route is drained.
                    for child in psutil.process_iter():
                        try:
                            if child.pid != proc.pid and os.getpgid(child.pid) == proc.pid:
                                if child.status() != psutil.STATUS_ZOMBIE:
                                    targets.append(child)
                        except (ProcessLookupError, psutil.NoSuchProcess):
                            pass
            except (OSError, psutil.Error):
                return False
            # The server contains the native child with an upstream Windows job.
            # Also retain descendant PID+creation-time identities independently.
            try:
                if os.name != "nt":
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                elif proc.poll() is None:
                    proc.terminate()
                for child in targets:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                _, alive = psutil.wait_procs(targets, timeout=5)
                for child in alive:
                    child.kill()
                _, alive = psutil.wait_procs(alive, timeout=5)
                alive = [child for child in alive if child.status() != psutil.STATUS_ZOMBIE]
                if os.name != "nt":
                    # Catch any child created between inventory and retirement.
                    for child in psutil.process_iter():
                        try:
                            if os.getpgid(child.pid) == proc.pid and child.status() != psutil.STATUS_ZOMBIE:
                                alive.append(child)
                        except (ProcessLookupError, psutil.NoSuchProcess):
                            pass
                drained = proc.poll() is not None and not alive
            except (OSError, psutil.Error, subprocess.TimeoutExpired):
                drained = False
            if self._diagnostics is not None:
                drained = self._diagnostics.join() and drained
            drained = self._extension_drained() and drained
            if drained and self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
            return drained

    def check_health(self) -> None:
        self._check_extension_health()
        if (
            self._closed
            or self._proc is None
            or self._proc.poll() is not None
            or self._diagnostics is not None
            and self._diagnostics.native_starts > 1
        ):
            from vllm.v1.engine.exceptions import EngineDeadError

            raise EngineDeadError()

    async def collective_rpc_async(self, method, timeout=None, args=(), kwargs=None):
        raise NotImplementedError(f"Strata does not implement collective RPC {method}")

    def shutdown(self) -> None:
        self._closed = True
        self._cancel.set()
        self._epoch += 1
        self._output = None
        if self._agent_stream is not None:
            while not self._agent_stream.empty():
                self._agent_stream.get_nowait()
            self._agent_stream.put_nowait(None)
        drained = self._terminate()
        _, _, drained = self._finish_failure_observations(
            self._io_request, reason="route_shutdown", lifecycle_outcome="drained", drained=drained
        )
        if self._task is not None and not self._task.done():
            self._ledger.release(self._reservation, drained=False)
            self._task.add_done_callback(lambda _: self._ledger.release(self._reservation, drained=drained))
        else:
            self._ledger.release(self._reservation, drained=drained)
