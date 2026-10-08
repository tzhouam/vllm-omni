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
import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vllm.outputs import CompletionOutput

from omni_stage_contracts import StageEvent, StageRequest
from vllm_omni.engine.resource_ledger import ResourceUnavailable
from vllm_omni.engine.stage_client import StageClientBase
from vllm_omni.outputs import OmniRequestOutput

PINNED_STRATA_REVISION = "d5ea7133741e67743c0e886bb426c0ce8d69cf6c"
BACKEND_NAME = "external.strata.text.v1"
_MAX_EVENTS = 4096
_MAX_DELTA_BYTES = 8192


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
        info = nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(gpu_index))
        snapshot.update(gpu_total_bytes=int(info.total), gpu_free_bytes=int(info.free), gpu_source="NVML exact bytes")
    except Exception:
        # nvidia-smi exists in native Windows and the WSL NVIDIA integration.
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_index),
                    "--query-gpu=memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            total, free = (int(value.strip()) for value in probe.stdout.strip().split(","))
            if total <= 0 or not 0 <= free <= total:
                raise ValueError("invalid GPU observation")
            snapshot.update(
                gpu_total_bytes=total << 20,
                gpu_free_bytes=free << 20,
                gpu_source="nvidia-smi integer MiB; rounded observation",
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


def _sanitize_diagnostic(line: str) -> str | None:
    """Content-free codes only; upstream logs can contain uncontrolled paths."""
    if line.strip() == "strata supervisor: native contained in Windows kill-on-close job":
        return line.strip()
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
    def __init__(self, pipe, path: Path | None):
        self.pipe, self.path = pipe, path
        self.records: list[str] = []
        self.file_tier_io_policy = None
        self.failed = False
        self._thread = threading.Thread(target=self._drain, name="strata-diagnostics", daemon=True)
        self._thread.start()

    def _drain(self):
        handle = None
        try:
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                handle = self.path.open("w", encoding="utf-8", buffering=1)
            skipping = False
            while row := self.pipe.readline(4097):
                complete = row.endswith(b"\n")
                if skipping or len(row) > 4096 or not complete:
                    skipping = not complete
                    continue
                safe = _sanitize_diagnostic(row.decode("utf-8", "replace"))
                if safe is not None and safe.startswith("strata file_tier_io_policy: "):
                    self.file_tier_io_policy = safe.rsplit(": ", 1)[1]
                if safe is not None and safe not in self.records and len(self.records) < 64:
                    self.records.append(safe)
                    if handle:
                        handle.write(safe + "\n")
        except Exception:
            self.failed = True
        finally:
            if handle:
                handle.close()

    def join(self):
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            self.pipe.close()
        return not self.failed and not self._thread.is_alive()


# Patch only the child stderr sink, not its weights/execution/protocol. This
# avoids persisting a raw native log. The shim is included in execution identity.
_BOOTSTRAP = r"""
import os, runpy, subprocess, sys, threading
native, server = sys.argv[1:3]
sys.argv = [server] + sys.argv[3:]
original = subprocess.Popen
def launch(args, *a, **kw):
    if os.path.realpath(str(args[0])) != os.path.realpath(native):
        return original(args, *a, **kw)
    kw['stderr'] = subprocess.PIPE
    p = original(args, *a, **kw)
    if os.name == 'nt':
        from serve.winjob import contain
        if not contain(p):
            p.kill(); p.wait(timeout=5)
            raise RuntimeError('Strata native Windows containment failed')
        sys.stderr.write('strata supervisor: native contained in Windows kill-on-close job\n')
        sys.stderr.flush()
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
    # file_mb is the native logical file-tier counter, not physical drive I/O.
    return result


class StrataTextStageClient(StageClientBase):
    """One complete request at a time, with bounded streaming and terminal ACK."""

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
        try:
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

        if config.get("name", BACKEND_NAME) != BACKEND_NAME:
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
        source_files = set(artifacts.verify(artifact_root))
        pack_files = set(prepared.verify(pack))
        if runtime_manifest.revision != PINNED_STRATA_REVISION:
            raise ValueError("runtime file manifest has a different revision")
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
        gpu_budget = _positive(config["gpu_budget_bytes"], "gpu_budget_bytes")
        gpu_total = _positive(config["gpu_total_bytes"], "gpu_total_bytes")
        if gpu_budget > gpu_total:
            raise ResourceUnavailable("Strata GPU budget exceeds physical VRAM")
        gpu_pool = config.get("gpu_pool", "vram:0")
        if expert_budget + host_overhead + self._max_io_bytes > self._reservation.demands.get("host_ram", 0):
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
            "--native",
            str(native),
            "--ple-gguf",
            str(ple),
            "--ple-io",
            ple_io,
            "--expert-cache",
            "auto",
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
        config_path = Path(self._temporary.name) / "server.json"
        config_path.write_text(json.dumps(server_cfg), encoding="utf-8")
        bootstrap_path = Path(self._temporary.name) / "bootstrap.py"
        bootstrap_path.write_text(_BOOTSTRAP, encoding="utf-8")
        env = os.environ.copy()
        for name in tuple(env):
            if name.startswith(("STRATA_", "PYTHON")):
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
        tier_plan = None
        if config.get("weight_tier_plan") is not None:
            tier_plan = WeightTierPlan.from_dict(config["weight_tier_plan"])
            if (
                tier_plan.artifact_manifest_sha256 != artifacts.manifest_sha256
                or tier_plan.backend != BACKEND_NAME
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
                    vision=False,
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
            if host_overhead + self._max_io_bytes < host_peak - expert_budget:
                raise ValueError("Strata overhead does not cover loading/state/lookup/workspace and I/O")
            if budget.ssd_artifact_bytes < artifacts.total_size_bytes + prepared.total_size_bytes:
                raise ValueError("Strata SSD budget omits source shards or prepared pack files")
            if tier_plan.mtp != bool(spec) or tier_plan.prefetch != any((ple_prefetch, routing_prefetch, io_prefetch)):
                raise ValueError("Strata MTP/prefetch switches differ from its tier plan")
        memory_snapshot = _probe_memory(gpu)
        _check_live_memory(memory_snapshot, dict(self._reservation.demands), gpu_pool, gpu_total)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
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
        self._diagnostics = _DiagnosticLog(
            self._proc.stdout, Path(config["log_file"]) if config.get("log_file") else None
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
            or health.get("images") is not False
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
        self.execution_plan = {
            "backend": BACKEND_NAME,
            "stage_id": self.stage_id,
            "worker_generation": self._generation,
            "runtime_revision": PINNED_STRATA_REVISION,
            "python_environment": python_environment,
            "fresh_memory_admission": memory_snapshot,
            "supervisor_bootstrap_sha256": hashlib.sha256(_BOOTSTRAP.encode()).hexdigest(),
            "runtime_manifest_sha256": runtime_manifest.manifest_sha256,
            "artifact_manifest_sha256": artifacts.manifest_sha256,
            "prepared_manifest_sha256": prepared.manifest_sha256,
            "conversion_manifest": conversion,
            "model_alias": self._model_alias,
            "requested_device": f"cpu+cuda:{gpu}",
            "observed_model_placement": None,
            "observed_compute_units": None,
            "actual_io_mode": None,
            "expert_ram_budget_bytes": expert_budget,
            "host_overhead_bytes": host_overhead,
            "gpu_budget_bytes": gpu_budget,
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
            "placement_evidence_level": "unverified",
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
            if set(prompt) - {"text", "max_tokens", "temperature", "stream_agent"}:
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
            self._epoch += 1
            request = StageRequest(request_id, self.stage_id, self._epoch, self._generation)
            self._active = request_id
            self._terminal_published = False
            self._cancel.clear()
            self._agent_stream = asyncio.Queue(maxsize=64) if streaming else None
            self._task = asyncio.create_task(self._run(request, text, maximum, self._agent_stream))
            self._task.add_done_callback(lambda _: self._apply_pending_ack())

    async def _run(self, request, text: str, maximum: int, queue) -> None:
        from vllm_omni.engine.weight_tiers import PlacementReport

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
                    "messages": [{"role": "user", "content": text}],
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
                        logical_read_bytes=telemetry["logical_file_read_bytes"],
                        expert_cache_hit_ratio=telemetry["hit_rate"],
                    ).to_dict(),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A truncated stream/error might leave native work active. Retire
            # the route before its reservation can be reused.
            self._cancel.set()
            self._closed = True
            drained = await asyncio.to_thread(self._terminate)
            self._ledger.release(self._reservation, drained=drained)
            output = OmniRequestOutput.from_error(request.request_id, f"Strata request failed: {type(exc).__name__}")
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
            if drained and self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
            return drained

    def check_health(self) -> None:
        if self._closed or self._proc is None or self._proc.poll() is not None:
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
        if self._task is not None and not self._task.done():
            self._ledger.release(self._reservation, drained=False)
            self._task.add_done_callback(lambda _: self._ledger.release(self._reservation, drained=drained))
        else:
            self._ledger.release(self._reservation, drained=drained)
