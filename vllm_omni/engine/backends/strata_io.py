# SPDX-License-Identifier: Apache-2.0
"""Content-free, owned Strata I/O observations; no inference or physical-I/O claims.

Pure stdlib so the same code can run inside the isolated native-process bootstrap.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path

BASE_REVISION = "d5ea7133741e67743c0e886bb426c0ce8d69cf6c"
DEPENDENCY_REVISION = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
PATCH_SHA256 = "31b98e5e02e21a6289d0713097036334a59aef2d15ad9e951ded80f59f4bfdcc"
DEPENDENCY_TREE = "d255198f04f9b8349f1dff24501d513f83cbfeda"
PATCH_SOURCES = {
    "include/strata/core/expert_source.hpp",
    "include/strata/kernels/ngram.hpp",
    "include/strata/ngram/ple_reader.hpp",
    "src/core/expert_source.cpp",
    "src/kernels/ngram.cpp",
    "src/ngram/ple_reader.cpp",
    "src/program/generate.cpp",
}
PREFIX = "strata supervisor: native io "
FRAME_SCHEMA = "omni-strata-owned-io-frame-v1"
PHASES = ("request_start", "prefill_end_decode_start", "request_end")
SCOPE = "native_FileExpertSource_and_PLE_counters_excludes_loading"
EXPERT_BOOLS = {"source_active", "unbuffered_active", "direct_counters_available"}
EXPERT_FLOATS = {"file_thread_us"}
EXPERT_INTS = {
    "logical_file_bytes",
    "logical_blob_bytes",
    "ram_blobs",
    "file_blobs",
    "mapped_fallback_blobs",
    "direct_submitted",
    "direct_completed",
    "direct_completed_aligned_bytes",
    "direct_submit_errors",
    "direct_completion_errors",
    "direct_short_reads",
}
PLE_BOOLS = {"direct_active"}
PLE_FLOATS = {"collect_wait_us", "submit_us", "read_latency_sum_us"}
PLE_INTS = {
    "logical_table_bytes",
    "rows_requested",
    "cache_hits",
    "dedup_rows",
    "demand_reads_issued",
    "demand_completed_bytes",
    "keepalive_reads",
    "keepalive_bytes",
    "all_completed_reads",
    "all_completed_aligned_bytes",
    "submit_errors",
    "completion_errors",
    "short_reads",
}
ERROR_KEYS = {
    "expert": {
        "direct_submit_errors",
        "direct_completion_errors",
        "direct_short_reads",
    },
    "ple": {"submit_errors", "completion_errors", "short_reads"},
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj

    return json.loads(
        raw,
        object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
    )


def _uint(value, *, positive=False):
    if type(value) is not int or not (int(positive) <= value <= 2**64 - 1):
        raise ValueError("invalid unsigned counter")
    return value


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def adapter_source_sha256():
    return _digest(__file__)


def production_bootstrap_source():
    """Read a fixed trusted sibling literal without importing the inference stack."""
    source = Path(__file__).with_name("strata.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_BOOTSTRAP" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError("trusted Strata bootstrap literal is unavailable")


def verify_observation_runtime(descriptor, runtime_root, verified_files, native_executable, bootstrap_source=None):
    """Verify additional build evidence after the complete runtime manifest.

    All paths refer to already hash-verified files, never arbitrary executable
    code or caller-supplied bootstrap/adapter hashes. No model bytes are read.
    """
    keys = {
        "schema",
        "native_io_schema",
        "base_revision",
        "dependency_revision",
        "dependency_tree",
        "engine_file",
        "patch_file",
        "patch_manifest_file",
        "build_receipt_file",
        "dependency_provenance_file",
        "patched_sources_dir",
        "build_evidence_dir",
        "runtime_dependencies_file",
    }
    actual_bootstrap = production_bootstrap_source()
    if bootstrap_source is not None and bootstrap_source != actual_bootstrap:
        raise ValueError("loaded Strata bootstrap differs from current trusted source")
    bootstrap_source = actual_bootstrap
    if not isinstance(descriptor, dict) or set(descriptor) != keys:
        raise ValueError("invalid Strata observation_runtime descriptor")
    if (
        descriptor["schema"] != "omni-strata-observation-runtime-v1"
        or descriptor["native_io_schema"] != "strata-omni-io-v1"
        or descriptor["base_revision"] != BASE_REVISION
        or descriptor["dependency_revision"] != DEPENDENCY_REVISION
        or descriptor["dependency_tree"] != DEPENDENCY_TREE
    ):
        raise ValueError("unreviewed Strata observation runtime/source identity")
    root = Path(runtime_root).resolve(strict=True)
    verified = {Path(path).resolve(strict=True) for path in verified_files}

    def member(relative, *, directory=False):
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("observation provenance path must be relative POSIX")
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("observation provenance path escapes runtime")
        path = (root / rel).resolve(strict=True)
        if not path.is_relative_to(root) or (directory and not path.is_dir()):
            raise ValueError("observation provenance path escapes runtime")
        if not directory and (not path.is_file() or path not in verified):
            raise ValueError("observation provenance is absent from runtime manifest")
        return path

    def load(relative):
        path = member(relative)
        if path.stat().st_size > 2 << 20:
            raise ValueError("observation provenance JSON is too large")
        value = _json(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("observation provenance must be an object")
        return value

    def evidence(relative, sha256, size=None):
        path = member(relative)
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256) or _digest(path) != sha256:
            raise ValueError("observation provenance byte hash mismatch")
        if size is not None and (type(size) is not int or size != path.stat().st_size):
            raise ValueError("observation provenance byte size mismatch")
        return path

    engine = member(descriptor["engine_file"])
    if engine != Path(native_executable).resolve(strict=True):
        raise ValueError("observation runtime identifies a different native executable")
    patch = evidence(descriptor["patch_file"], PATCH_SHA256)
    patch_manifest = load(descriptor["patch_manifest_file"])
    if patch_manifest.get("base_revision") != BASE_REVISION or patch_manifest.get("patch_sha256") != PATCH_SHA256:
        raise ValueError("observation patch manifest identity mismatch")
    records = patch_manifest.get("files")
    if (
        not isinstance(records, list)
        or len(records) != 7
        or any(not isinstance(item, dict) for item in records)
        or {item.get("path") for item in records} != PATCH_SOURCES
    ):
        raise ValueError("observation patch must identify seven distinct source files")
    member(descriptor["patched_sources_dir"], directory=True)
    source_hashes = {}
    for item in records:
        relative = descriptor["patched_sources_dir"] + "/" + item["path"]
        evidence(relative, item["patched_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("base_sha256", "")):
            raise ValueError("observation patch omits source lineage hash")
        source_hashes[item["path"]] = {"base_sha256": item["base_sha256"], "patched_sha256": item["patched_sha256"]}
    dependency = load(descriptor["dependency_provenance_file"])
    if (
        dependency.get("revision") != DEPENDENCY_REVISION
        or dependency.get("tree") != DEPENDENCY_TREE
        or dependency.get("git_status_porcelain") != ""
    ):
        raise ValueError("observation dependency source/tree is unverified")
    receipt = load(descriptor["build_receipt_file"])
    if (
        receipt.get("schema") != "omni-strata-private-build-v1"
        or receipt.get("status") != "built_not_installed_not_neurally_qualified"
        or receipt.get("base_revision") != BASE_REVISION
        or receipt.get("dependency_revision") != DEPENDENCY_REVISION
        or receipt.get("patch_sha256") != PATCH_SHA256
        or receipt.get("parallelism") != 2
        or not isinstance(receipt.get("steps"), list)
        or len(receipt["steps"]) != 2
        or any(type(step.get("exit_code")) is not int or step["exit_code"] != 0 for step in receipt["steps"])
    ):
        raise ValueError("observation build did not verify the reviewed source and successful steps")
    build_dir = descriptor["build_evidence_dir"]
    member(build_dir, directory=True)
    for index, step in enumerate(receipt["steps"]):
        evidence(f"{build_dir}/step-{index}.log", step["sha256"])
    for name, record in receipt.get("files", {}).items():
        if name not in {"strata.exe", "CMakeCache.txt", "compile_commands.json", "build.ninja"}:
            raise ValueError("unexpected observation build output")
        evidence(
            descriptor["engine_file"] if name == "strata.exe" else f"{build_dir}/{name}",
            record["sha256"],
            record["size_bytes"],
        )
    if set(receipt.get("files", {})) != {"strata.exe", "CMakeCache.txt", "compile_commands.json", "build.ninja"}:
        raise ValueError("observation build evidence is incomplete")
    tools = receipt.get("tools", {})
    expected_versions = {"nvcc": b"13.4", "cmake": b"3.31.6", "ninja": b"1.12.1", "cl": b"19.44.35229"}
    if set(tools) != set(expected_versions):
        raise ValueError("observation compiler tool identities are incomplete")
    for name, version in expected_versions.items():
        tool = tools[name]
        path = evidence(f"{build_dir}/{name}-version.log", tool["log_sha256"])
        if tool.get("exit_code") not in ({0, 2} if name == "cl" else {0}) or version not in path.read_bytes():
            raise ValueError("observation compiler tool/version differs from reviewed build")
    required_cache = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": "120",
        "CMAKE_CUDA_RUNTIME_LIBRARY": "Static",
        "STRATA_ENABLE_CUDA": "ON",
        "STRATA_PORTABLE": "ON",
        "STRATA_NATIVE_EXPERTS": "ON",
        "STRATA_MMQ_KQUANTS": "OFF",
        "STRATA_BUILD_TESTS": "OFF",
    }
    cache = member(f"{build_dir}/CMakeCache.txt").read_text(encoding="utf-8")
    for key, value in required_cache.items():
        if not re.search(r"(?m)^" + re.escape(key) + r":[^=]+=" + re.escape(value) + r"$", cache):
            raise ValueError("observation CMake build flags differ from reviewed build")
    configure = receipt["steps"][0].get("command", [])
    build_command = receipt["steps"][1].get("command", [])
    if (
        not isinstance(configure, list)
        or not all(f"-D{key}={value}" in configure for key, value in required_cache.items())
        or not isinstance(build_command, list)
        or build_command[-4:] != ["--target", "strata", "--parallel", "2"]
        or receipt.get("commands") != [configure, build_command]
    ):
        raise ValueError("observation configure/build command provenance mismatch")
    commands = _json(member(f"{build_dir}/compile_commands.json").read_text(encoding="utf-8"))
    if not isinstance(commands, list) or any(
        not any(
            isinstance(command, dict) and str(command.get("file", "")).replace("\\", "/").endswith("/" + path)
            for command in commands
        )
        for path in source_hashes
        if path.endswith(".cpp")
    ):
        raise ValueError("observation compile commands omit patched translation units")
    dependencies = load(descriptor["runtime_dependencies_file"])
    if dependencies.get("schema") != "omni-strata-runtime-dependencies-v1":
        raise ValueError("observation native runtime dependencies are unverified")
    dlls = dependencies.get("files")
    if not isinstance(dlls, list) or {item.get("path") for item in dlls} != {
        "engine/cublas64_13.dll",
        "engine/cublasLt64_13.dll",
    }:
        raise ValueError("observation native CUDA dependency closure is incomplete")
    for item in dlls:
        evidence(item["path"], item["sha256"], item["size_bytes"])
    imports = {
        "engine/strata.exe": {"cublas64_13.dll", "kernel32.dll", "advapi32.dll"},
        "engine/cublas64_13.dll": {"cublaslt64_13.dll", "kernel32.dll"},
        "engine/cublasLt64_13.dll": {"kernel32.dll"},
    }
    declared_imports = dependencies.get("static_imports")
    if not isinstance(declared_imports, dict) or set(declared_imports) != set(imports):
        raise ValueError("observation native static dependency closure is unverified")
    observer = dependencies.get("static_import_observer", {})
    if observer.get("tool") != "dumpbin /DEPENDENTS" or not re.fullmatch(
        r"[0-9a-f]{64}", observer.get("tool_sha256", "")
    ):
        raise ValueError("observation native import observer identity is unavailable")
    member(observer.get("raw_logs_dir"), directory=True)
    for path, expected in imports.items():
        declared = declared_imports[path]
        if not isinstance(declared, list) or {name.lower() for name in declared} != expected:
            raise ValueError("observation native static dependency closure differs from reviewed build")
        log = member(observer["raw_logs_dir"] + "/" + Path(path).name + ".dependents.txt")
        if log.stat().st_size > 1 << 20:
            raise ValueError("observation native import log is too large")
        observed = {name.lower() for name in re.findall(r"(?im)^\s+([a-z0-9_]+\.dll)\s*$", log.read_text())}
        if observed != expected:
            raise ValueError("observation native import log does not verify the declared closure")
    policy = dependencies.get("native_dll_search_policy", {})
    if policy.get("required_directories") != ["engine", "System32"] or policy.get("required_at_launch") is not True:
        raise ValueError("observation native dependency search policy is unverified")
    identity = {
        "schema": "omni-strata-observed-runtime-v1",
        "base_revision": BASE_REVISION,
        "dependency_revision": DEPENDENCY_REVISION,
        "dependency_tree": DEPENDENCY_TREE,
        "native_executable_sha256": _digest(engine),
        "patch_sha256": _digest(patch),
        "patch_source_hashes": source_hashes,
        "build_receipt_sha256": _digest(member(descriptor["build_receipt_file"])),
        "dependency_provenance_sha256": _digest(member(descriptor["dependency_provenance_file"])),
        "runtime_dependencies_sha256": _digest(member(descriptor["runtime_dependencies_file"])),
        "supervisor_bootstrap_sha256": hashlib.sha256(bootstrap_source.encode()).hexdigest(),
        "io_adapter_sha256": adapter_source_sha256(),
        "native_io_schema": "strata-omni-io-v1",
        "measurement_identity_changed": True,
        "three_tier_memory_qualified": False,
        "native_dll_search_policy": "isolated_engine_and_system32",
        "live_loaded_module_paths_verified": False,
        "native_dependency_files": [
            {"path": item["path"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]} for item in dlls
        ],
    }
    return identity | {"identity_sha256": hashlib.sha256(_canonical(identity).encode()).hexdigest()}


def audit_selected_loaded_modules(native_identity, runtime_root, runtime_identity):
    """Read the exact owned process's selected native/CUDA modules after load.

    This is a selected-module snapshot, not every OS module or a future promise.
    Driver bytes are observed machine-specific identity, not pinned engine code.
    """
    report = {
        "schema": "omni-strata-selected-module-audit-v1",
        "status": "unverified",
        "scope": "selected_engine_cublas_and_system32_driver_at_load",
        "modules": [],
        "all_os_modules_covered": False,
        "reasons": [],
    }
    if os.name != "nt":
        report["reasons"] = ["Windows module observation unavailable"]
        return report
    import ctypes
    from ctypes import wintypes

    try:
        pid = _uint(native_identity["pid"], positive=True)
        created = _uint(native_identity["creation_filetime_100ns"], positive=True)
        report["pid"], report["creation_filetime_100ns"] = pid, created
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.GetSystemDirectoryW.argtypes = (wintypes.LPWSTR, wintypes.UINT)
        kernel.GetSystemDirectoryW.restype = wintypes.UINT
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        psapi.EnumProcessModulesEx.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.HMODULE),
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.DWORD,
        )
        psapi.EnumProcessModulesEx.restype = wintypes.BOOL
        psapi.GetModuleFileNameExW.argtypes = (wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD)
        psapi.GetModuleFileNameExW.restype = wintypes.DWORD
        handle = kernel.OpenProcess(0x410, False, pid)  # query information + VM read
        if not handle:
            raise ValueError("owned native process module handle unavailable")
        try:

            def verify_process():
                times = [wintypes.FILETIME() for _ in range(4)]
                code = wintypes.DWORD()
                if not kernel.GetProcessTimes(
                    handle, *(ctypes.byref(t) for t in times)
                ) or not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                    raise ValueError("owned native process identity unavailable")
                actual = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
                if actual != created or code.value != 259:
                    raise ValueError("owned native process exited or identity changed")

            verify_process()
            modules = (wintypes.HMODULE * 8192)()
            needed = wintypes.DWORD()
            if not psapi.EnumProcessModulesEx(
                handle, modules, ctypes.sizeof(modules), ctypes.byref(needed), 3
            ) or needed.value > ctypes.sizeof(modules):
                raise ValueError("owned native module enumeration incomplete")
            root = Path(runtime_root).resolve(strict=True)
            expected = {
                "strata.exe": {
                    "path": root / "engine/strata.exe",
                    "sha256": runtime_identity["native_executable_sha256"],
                },
                **{
                    Path(item["path"]).name.lower(): {"path": root / item["path"], "sha256": item["sha256"]}
                    for item in runtime_identity["native_dependency_files"]
                },
            }
            system = ctypes.create_unicode_buffer(32768)
            count = kernel.GetSystemDirectoryW(system, len(system))
            if not count or count >= len(system):
                raise ValueError("Windows System32 identity unavailable")
            expected["nvcuda.dll"] = {"path": Path(system.value) / "nvcuda.dll", "sha256": None}
            found = {}
            for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
                path_buffer = ctypes.create_unicode_buffer(32768)
                count = psapi.GetModuleFileNameExW(handle, module, path_buffer, len(path_buffer))
                if not count or count >= len(path_buffer):
                    raise ValueError("owned native module path enumeration incomplete")
                path = Path(path_buffer.value).resolve(strict=True)
                name = path.name.lower()
                if name not in expected:
                    continue
                if name in found or path != expected[name]["path"].resolve(strict=True):
                    raise ValueError("selected native module path mismatch or duplicate")
                digest = _digest(path)
                if expected[name]["sha256"] is not None and digest != expected[name]["sha256"]:
                    raise ValueError("selected native module hash mismatch")
                found[name] = {
                    "name": name,
                    "path": str(path),
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "identity_source": "machine_System32_driver"
                    if name == "nvcuda.dll"
                    else "verified_runtime_manifest",
                }
            verify_process()
            if set(found) != set(expected):
                raise ValueError("selected native CUDA modules are missing")
            report["modules"] = [found[name] for name in sorted(found)]
            report["status"] = "verified"
        finally:
            kernel.CloseHandle(handle)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        report["reasons"] = [str(exc)]
    return report


def validate_snapshot(value):
    keys = {
        "schema",
        "request_seq",
        "phase",
        "scope",
        "qpc",
        "qpc_hz",
        "physical_ssd_read_bytes",
        "expert",
        "ple",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("unknown native snapshot fields")
    if value["schema"] != "strata-omni-io-v1" or value["phase"] not in PHASES or value["scope"] != SCOPE:
        raise ValueError("unknown native snapshot contract")
    if value["physical_ssd_read_bytes"] is not None:
        raise ValueError("native counters cannot establish physical SSD bytes")
    for key in ("request_seq", "qpc", "qpc_hz"):
        _uint(value[key], positive=True)
    for group, ints, floats, bools in (
        ("expert", EXPERT_INTS, EXPERT_FLOATS, EXPERT_BOOLS),
        ("ple", PLE_INTS, PLE_FLOATS, PLE_BOOLS),
    ):
        values = value[group]
        if not isinstance(values, dict) or set(values) != ints | floats | bools:
            raise ValueError("unknown native counter fields")
        for name in ints:
            _uint(values[name])
        for name in floats:
            number = values[name]
            if type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 2**64 - 1:
                raise ValueError("invalid native counter time")
        if any(type(values[name]) is not bool for name in bools):
            raise ValueError("invalid native mode flag")
    return copy.deepcopy(value)


class OwnedNativeFrameWriter:
    """Bootstrap-only frame writer. The nonce is removed before native Popen.

    Install input/output wrappers around the exact owned Popen pipes. Never
    forward the raw GEN body, token IDs, ERR text, prompts or tensors to logs.
    """

    def __init__(self, nonce, generation, pid, creation_filetime_100ns, emit):
        if not re.fullmatch(r"[0-9a-f]{32}", nonce or "") or not generation:
            raise ValueError("invalid owner context")
        self.nonce, self.generation = nonce, generation
        self.pid, self.created = (
            _uint(pid, positive=True),
            _uint(creation_filetime_100ns, positive=True),
        )
        self.emit, self.sequence = emit, 0
        self.lock = threading.RLock()

    def _emit(self, kind, payload=None):
        frame = {
            "schema": FRAME_SCHEMA,
            "generation": self.generation,
            "pid": self.pid,
            "creation_filetime_100ns": self.created,
            "dispatch_seq": self.sequence,
            "kind": kind,
            "payload": payload,
        }
        row = PREFIX + self.nonce + " " + _canonical(frame)
        if len(row.encode()) > 4096:
            raise ValueError("observer frame too large")
        self.emit(row + "\n")

    def observe_input(self, text):
        # Upstream writes one full GEN/GENI command. Reject batching in this
        # serial-only observer; this validation is not the native parser.
        with self.lock:
            lines = text.splitlines()
            if len(lines) != 1:
                self._emit("error", "unexpected_native_command_framing")
                return
            command = lines[0].split(" ", 1)[0]
            if command in {"GEN", "GENI"}:
                self.sequence += 1
                self._emit("dispatch", {"command": command})
            elif command.startswith("BGEN"):
                self._emit("error", "unsupported_native_batch_dispatch")

    def observe_output(self, text):
        with self.lock:
            if text.startswith("OMNI_IO_V1 "):
                try:
                    if len(text.encode()) > 3072:
                        raise ValueError("native snapshot too large")
                    value = validate_snapshot(_json(text[len("OMNI_IO_V1 ") :]))
                    self._emit("snapshot", value)
                except (ValueError, TypeError, OverflowError):
                    self._emit("error", "invalid_native_io_snapshot")
            elif text.startswith("ERR"):
                self._emit("error", "native_request_error")
            elif text.startswith("DONE "):
                fields = text.split()
                if len(fields) == 16 and fields[5] in {"stop", "length", "cancel"}:
                    self._emit("terminal", {"finish": fields[5]})
                else:
                    self._emit("error", "invalid_native_terminal")


class StrataIoObserver:
    """Bounded serial observer; late/foreign frames cannot become another request.

    begin() happens before dispatching the authenticated single-slot HTTP call.
    finish() happens after SSE, exact native DONE, metrics and health checks.
    Invalid/incomplete observations retire this observer generation; they do not
    silently turn a successful neural output into complete I/O qualification.
    """

    def __init__(self, nonce, generation, pid, creation_filetime_100ns, runtime_identity):
        if not re.fullmatch(r"[0-9a-f]{32}", nonce or "") or not generation:
            raise ValueError("invalid observer context")
        identity = {key: value for key, value in runtime_identity.items() if key != "identity_sha256"}
        if (
            identity.get("schema") != "omni-strata-observed-runtime-v1"
            or identity.get("base_revision") != BASE_REVISION
            or identity.get("dependency_revision") != DEPENDENCY_REVISION
            or identity.get("patch_sha256") != PATCH_SHA256
            or identity.get("native_io_schema") != "strata-omni-io-v1"
            or identity.get("three_tier_memory_qualified") is not False
            or runtime_identity.get("identity_sha256") != hashlib.sha256(_canonical(identity).encode()).hexdigest()
        ):
            raise ValueError("patched runtime identity is required")
        self.nonce, self.generation = nonce, generation
        self.pid, self.created = (
            _uint(pid, positive=True),
            _uint(creation_filetime_100ns, positive=True),
        )
        self.runtime_identity = copy.deepcopy(runtime_identity)
        self.active, self.last_seq, self.last_epoch, self.last_snapshot = (
            None,
            0,
            0,
            None,
        )
        self.retired = False
        self.lock = threading.RLock()

    def begin(self, request_id, epoch):
        with self.lock:
            if self.active is not None or self.retired or not isinstance(request_id, str) or not request_id:
                raise ValueError("observer has no available request slot")
            _uint(epoch, positive=True)
            if epoch <= self.last_epoch:
                raise ValueError("Omni request epoch is stale")
            self.active = {
                "request_id": request_id,
                "epoch": epoch,
                "snapshots": [],
                "reasons": [],
                "dispatch_seq": None,
                "terminal": None,
            }

    def _invalidate(self, reason):
        self.retired = True
        if self.active is not None and reason not in self.active["reasons"] and len(self.active["reasons"]) < 16:
            self.active["reasons"].append(reason)

    def mark_incomplete(self, reason):
        with self.lock:
            if not isinstance(reason, str) or not re.fullmatch(r"[a-z_]{1,64}", reason):
                reason = "observation_incomplete"
            self._invalidate(reason)

    def ingest(self, row):
        with self.lock:
            marker = PREFIX + self.nonce + " "
            if not isinstance(row, str) or not row.startswith(marker):
                return False
            if len(row.encode()) > 4096:
                self._invalidate("oversized_owned_observation")
                return False
            try:
                frame = _json(row[len(marker) :])
                if set(frame) != {
                    "schema",
                    "generation",
                    "pid",
                    "creation_filetime_100ns",
                    "dispatch_seq",
                    "kind",
                    "payload",
                }:
                    raise ValueError("invalid_owned_frame")
                if (
                    frame["schema"] != FRAME_SCHEMA
                    or frame["generation"] != self.generation
                    or type(frame["pid"]) is not int
                    or frame["pid"] != self.pid
                    or type(frame["creation_filetime_100ns"]) is not int
                    or frame["creation_filetime_100ns"] != self.created
                ):
                    raise ValueError("foreign_native_identity")
                sequence = _uint(frame["dispatch_seq"], positive=True)
                if sequence <= self.last_seq:
                    return False  # bounded late frame; never attach to a later request
                active = self.active
                if active is None:
                    raise ValueError("native_work_without_active_omni_request")
                if sequence != self.last_seq + 1:
                    raise ValueError("native_sequence_gap")
                kind, payload = frame["kind"], frame["payload"]
                if kind == "dispatch":
                    if active["dispatch_seq"] is not None or payload not in (
                        {"command": "GEN"},
                        {"command": "GENI"},
                    ):
                        raise ValueError("duplicate_or_unknown_native_dispatch")
                    active["dispatch_seq"] = sequence
                elif active["dispatch_seq"] != sequence:
                    raise ValueError("native_observation_before_owned_dispatch")
                elif kind == "snapshot":
                    snapshot = validate_snapshot(payload)
                    if snapshot["request_seq"] != sequence or len(active["snapshots"]) >= 3:
                        raise ValueError("native_request_sequence_mismatch")
                    if snapshot["phase"] != PHASES[len(active["snapshots"])] or active["terminal"] is not None:
                        raise ValueError("native_phase_order_mismatch")
                    previous = self.last_snapshot
                    if previous is not None:
                        if snapshot["qpc_hz"] != previous["qpc_hz"] or snapshot["qpc"] < previous["qpc"]:
                            raise ValueError("native_clock_changed_or_decreased")
                        for group, names in (
                            ("expert", EXPERT_INTS | EXPERT_FLOATS),
                            ("ple", PLE_INTS | PLE_FLOATS),
                        ):
                            if any(snapshot[group][name] < previous[group][name] for name in names):
                                raise ValueError("native_counter_decreased")
                    active["snapshots"].append(snapshot)
                    self.last_snapshot = snapshot
                elif kind == "terminal":
                    if (
                        len(active["snapshots"]) != 3
                        or active["terminal"] is not None
                        or payload
                        not in (
                            {"finish": "stop"},
                            {"finish": "length"},
                            {"finish": "cancel"},
                        )
                    ):
                        raise ValueError("invalid_or_duplicate_native_terminal")
                    active["terminal"] = payload["finish"]
                    if payload["finish"] == "cancel":
                        self._invalidate("native_request_cancelled")
                elif kind == "error":
                    self._invalidate("native_or_observer_error")
                else:
                    raise ValueError("unknown_owned_frame_kind")
                return True
            except (ValueError, TypeError, KeyError, OverflowError):
                self._invalidate("invalid_owned_observation")
                return False

    def finish(self, request_id, epoch, *, completed):
        with self.lock:
            _uint(epoch, positive=True)
            active = self.active
            if active is None or active["request_id"] != request_id or active["epoch"] != epoch:
                raise ValueError("stale Omni request completion")
            if completed is not True:
                self._invalidate("omni_request_not_completed")
            if len(active["snapshots"]) != 3 or active["terminal"] not in {
                "stop",
                "length",
            }:
                self._invalidate("missing_complete_native_phase_set_or_terminal")
            intervals = {}
            if len(active["snapshots"]) == 3:
                start, prefill, end = active["snapshots"]
                for name, first, last in (
                    ("prefill", start, prefill),
                    ("decode", prefill, end),
                    ("whole_native_interval", start, end),
                ):
                    intervals[name] = {
                        group: {key: last[group][key] - first[group][key] for key in names}
                        for group, names in (
                            ("expert", EXPERT_INTS | EXPERT_FLOATS),
                            ("ple", PLE_INTS | PLE_FLOATS),
                        )
                    } | {"boundary_wall_s": (last["qpc"] - first["qpc"]) / first["qpc_hz"]}
                if any(
                    intervals["whole_native_interval"][group][key] for group, keys in ERROR_KEYS.items() for key in keys
                ):
                    self._invalidate("native_io_error_counters_increased")
            report = {
                "schema": "omni-strata-request-io-observation-v1",
                "status": "complete" if not active["reasons"] else "incomplete",
                "generation": self.generation,
                "native_pid": self.pid,
                "creation_filetime_100ns": self.created,
                "request_id": request_id,
                "epoch": epoch,
                "native_request_seq": active["dispatch_seq"],
                "runtime_identity_sha256": self.runtime_identity["identity_sha256"],
                "scope": SCOPE,
                "attribution": "counter_differences_during_owned_boundary_intervals_may_include_background_work",
                "snapshot_consistency": "independent_atomic_counters_not_global_transaction",
                "loading_covered": False,
                "physical_ssd_read_bytes": None,
                "three_tier_memory_qualified": False,
                "raw_snapshots": copy.deepcopy(active["snapshots"]),
                "intervals": intervals,
                "native_terminal": active["terminal"],
                "reasons": list(active["reasons"]),
                "io_modes": self._io_modes(active["snapshots"], intervals),
            }
            if active["dispatch_seq"] is not None:
                self.last_seq = active["dispatch_seq"]
            self.last_epoch = epoch
            self.active = None
            return report

    @staticmethod
    def _io_modes(snapshots, intervals):
        if len(snapshots) != 3:
            return {"expert": "unverified", "ple": "unverified", "strict_direct_path_verified": False}
        whole = intervals["whole_native_interval"]
        expert, ple = whole["expert"], whole["ple"]
        if not all(row["expert"]["source_active"] for row in snapshots):
            expert_mode = "inactive_or_changed_file_source"
        elif expert["mapped_fallback_blobs"]:
            expert_mode = "unbuffered_with_mapped_fallback"
        elif not expert["logical_file_bytes"] and not expert["direct_completed_aligned_bytes"]:
            expert_mode = "no_expert_file_transfer_observed"
        elif not all(row["expert"]["unbuffered_active"] for row in snapshots):
            expert_mode = "mapped_or_buffered_policy_observed"
        elif expert["direct_completed_aligned_bytes"] and all(
            row["expert"]["direct_counters_available"] for row in snapshots
        ):
            expert_mode = "windows_unbuffered_completions_observed"
        else:
            expert_mode = "unbuffered_policy_without_completed_transfer_evidence"
        if not ple["logical_table_bytes"] and not ple["all_completed_aligned_bytes"]:
            ple_mode = "no_ple_table_transfer_observed"
        elif all(row["ple"]["direct_active"] for row in snapshots):
            ple_mode = (
                "direct_completions_observed"
                if ple["all_completed_aligned_bytes"]
                else "direct_policy_without_completed_transfer_evidence"
            )
        else:
            ple_mode = "mapped_or_changed_table_policy_observed"
        return {
            "expert": expert_mode,
            "ple": ple_mode,
            "strict_direct_path_verified": False,
            "note": "OS transfer observations do not prove model-attributed physical SSD I/O or aggregate cache bounds",
        }
