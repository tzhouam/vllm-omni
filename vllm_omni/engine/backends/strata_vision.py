"""Private prototype: bounded image contract and authoritative encoder observations.

No inference, scheduler, default qualification, or physical SSD measurement lives here.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import os
import re
import struct
import threading
import warnings
from pathlib import Path

SCHEMA = "omni-strata-owned-vision-frame-v1"
PREFIX = "strata supervisor: owned vision "
NATIVE_PREFIX = "OMNI_VISION_V1 "
BASE_REVISION = "d5ea7133741e67743c0e886bb426c0ce8d69cf6c"
DEPENDENCY_REVISION = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
DEPENDENCY_TREE = "d255198f04f9b8349f1dff24501d513f83cbfeda"

REVIEWED_VISION_EVIDENCE = {
    "strata_patch": "77ec1e3c80f19c56c64f5143417268ea36f2e993441542c83a6a489a75e0c8c5",
    "dependency_patch": "95e84404980a62488c57101e5b21e6d0b0564bd184dfbb6aec747882be3dcc74",
    "patched_vision_source": "455bd8edb1a60e12960542e55116c0f98022cfc70ac604fe14dff59b23300787",
    "patched_clip_source": "09015eb469da190858ec6cd50cd442ca5d9eb59c1ff8ab9e7c240c8b98cea163",
}


# CUDA v3 is separately reviewed: the v2 extra graph split corrupts embeddings.
# These are raw native source bytes, including their recorded line endings.
REVIEWED_CUDA_VISION_EVIDENCE = {
    "strata_patch": REVIEWED_VISION_EVIDENCE["strata_patch"],
    "dependency_patch": REVIEWED_VISION_EVIDENCE["dependency_patch"],
    "device_binding_patch": "ff7d8fb0a84dc731a88ba121a31aa9e6e4e1b0a36cdb68cf4cbebfdbdb8d4160",
    "warmup_patch": "4936a6f5cc07510f2c10b0226b7a966b8598ee1bd75bf828d0bb7b10517d92df",
    "tensor_budget_patch": "cf23bbd219bd012ad1bc473cf37bdb64771630fcf2a3a2cf48b44749ff87e4e7",
    "scheduler_budget_patch": "2fdc4c7ca69da6c84d9f96b7e713428849da2547653f9500b5cd8d52784f92e1",
    "patched_vision_source": "20e1bc8f5a59f01f3184357ab89d64eab6e4ff1d1ee941ff842957e4ed5f66f2",
    "patched_clip_source": "c1f6aa2d3ee25c63ead54a71080d95dd75fe74ed1273535787062155fdc4a877",
    "patched_scheduler_source": "d28407c78b8ba0d7504bba73c3db08c8670decb40d8ff491b44eb785409644e4",
    "device_binding_header": "46dea254506ec1cc00a48567eaedaa0021c15f1edc07bde6c42938c0c4756c76",
}
REVIEWED_CUDA_ENCODER_SHA256 = "b7ee3ad280e0dc094283a335f6b97e22b48d5e374a0cd6f1e8f6fe589b0e1a27"


def reviewed_vision_build(device, encoder_sha256):
    """Select reviewed provenance, without granting runtime or model qualification."""
    if device == "cpu":
        return dict(REVIEWED_VISION_EVIDENCE), 256 << 20
    if device != "cuda" or encoder_sha256 != REVIEWED_CUDA_ENCODER_SHA256:
        raise ValueError("unreviewed CUDA encoder binary; repaired v3 build required")
    return dict(REVIEWED_CUDA_VISION_EVIDENCE), 512 << 20


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _uint(value, positive=False):
    if type(value) is not int or not int(positive) <= value < 2**63:
        raise ValueError("invalid vision integer")
    return value


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate vision JSON key")
            result[key] = value
        return result

    return json.loads(
        raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON"))
    )


def _windows_system_directory():
    if os.name != "nt":
        raise ValueError("PE closure requires the actual Windows System32 directory")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetSystemDirectoryW.argtypes = (wintypes.LPWSTR, wintypes.UINT)
    kernel.GetSystemDirectoryW.restype = wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32768)
    count = kernel.GetSystemDirectoryW(buffer, len(buffer))
    if not count or count >= len(buffer):
        raise ValueError("System32 path query failed")
    return Path(buffer.value).resolve(strict=True)


def pe_imports(path, *, max_file_bytes=256 << 20):
    """Bounded PE32/PE32+ normal and delay-import scan; never executes code."""
    import mmap

    path = Path(path)
    if type(max_file_bytes) is not int or not 64 <= max_file_bytes <= 1 << 30:
        raise ValueError("invalid explicit PE file bound")
    if not 64 <= path.stat().st_size <= max_file_bytes:
        raise ValueError("PE input size outside reviewed bound")
    with path.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:

        def unpack(format_, offset):
            if offset < 0 or offset + struct.calcsize(format_) > len(data):
                raise ValueError("PE structure escapes file")
            return struct.unpack_from(format_, data, offset)

        if data[:2] != b"MZ":
            raise ValueError("invalid DOS signature")
        pe = unpack("<I", 60)[0]
        if pe < 64 or data[pe : pe + 4] != b"PE\0\0":
            raise ValueError("invalid PE signature")
        machine, sections = unpack("<HH", pe + 4)
        optional_size = unpack("<H", pe + 20)[0]
        optional = pe + 24
        magic = unpack("<H", optional)[0]
        if machine != 0x8664 or magic != 0x20B or not 1 <= sections <= 96 or optional_size < 112:
            raise ValueError("only bounded Windows x64 PE32+ is admitted")
        image_base = unpack("<Q", optional + 24)[0]
        headers = unpack("<I", optional + 60)[0]
        directory_count = unpack("<I", optional + 108)[0]
        if optional_size < 112 + 8 * min(directory_count, 16) or headers > len(data):
            raise ValueError("invalid PE optional header extent")
        section_table = optional + optional_size
        ranges = []
        for index in range(sections):
            virtual_size, address, size, offset = unpack("<IIII", section_table + index * 40 + 8)
            if offset + size > len(data):
                raise ValueError("PE section raw bytes escape file")
            ranges.append((address, max(virtual_size, size), offset, size))

        def raw(rva, size):
            if not 0 < rva < 2**32 or size < 1:
                raise ValueError("invalid PE import RVA")
            if rva < headers and rva + size <= headers:
                return rva
            matches = [
                (offset + rva - address, extent - (rva - address))
                for address, virtual, offset, extent in ranges
                if address <= rva < address + virtual
            ]
            if len(matches) != 1 or size > matches[0][1]:
                raise ValueError("PE import points outside one raw section")
            return matches[0][0]

        def name(rva):
            result = bytearray()
            for index in range(256):
                value = data[raw(rva + index, 1)]
                if not value:
                    break
                result.append(value)
            else:
                raise ValueError("unterminated PE import name")
            try:
                value = result.decode("ascii").lower()
            except UnicodeDecodeError as exc:
                raise ValueError("non-ASCII PE import name") from exc
            if not re.fullmatch(r"[a-z0-9_-][a-z0-9_.-]{0,126}\.dll", value) or ".." in value:
                raise ValueError("invalid PE DLL name")
            return value

        result = {}
        for kind, directory, row_bytes in (("normal", 1, 20), ("delay", 13, 32)):
            names = set()
            if directory >= directory_count:
                result[kind] = []
                continue
            rva, extent = unpack("<II", optional + 112 + directory * 8)
            if not rva and not extent:
                result[kind] = []
                continue
            if not rva or not row_bytes <= extent <= 1 << 20:
                raise ValueError("invalid PE import directory extent")
            for index in range(min(extent // row_bytes, 4096)):
                row = unpack("<" + "I" * (row_bytes // 4), raw(rva + index * row_bytes, row_bytes))
                if not any(row):
                    break
                if kind == "normal":
                    name_rva = row[3]
                else:
                    if row[0] not in (0, 1):
                        raise ValueError("unknown delay-import address mode")
                    name_rva = row[1] if row[0] == 1 else row[1] - image_base
                names.add(name(name_rva))
            else:
                raise ValueError("PE import directory has no bounded terminator")
            result[kind] = sorted(names)
        return result


def scan_pe_closure(executable, runtime_root, verified_files, *, max_file_bytes=256 << 20):
    """Recompute deployed non-system normal/delay closure from PE bytes.

    Explicit LoadLibrary calls are covered only by the separate live snapshot.
    System32 dependencies are classified by path, not claimed bundled/pinned.
    """
    root = Path(runtime_root).resolve(strict=True)
    executable = Path(executable).resolve(strict=True)
    verified = {Path(p).resolve(strict=True) for p in verified_files}
    system = _windows_system_directory()
    queue = [executable]
    nodes, required = {}, {executable.name.lower()}
    while queue:
        path = queue.pop()
        relative = path.relative_to(root).as_posix()
        if relative in nodes:
            continue
        if path not in verified or path.parent != executable.parent:
            raise ValueError("PE dependency is outside verified isolated engine directory")
        imports = pe_imports(path, max_file_bytes=max_file_bytes)
        node = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
            "normal": imports["normal"],
            "delay": imports["delay"],
            "non_system": [],
            "system": [],
        }
        for library in sorted(set(imports["normal"] + imports["delay"])):
            local = path.parent / library
            if local.exists():
                local = local.resolve(strict=True)
                if local not in verified or local.parent != executable.parent:
                    raise ValueError("unbound or redirected local PE dependency")
                node["non_system"].append(library)
                queue.append(local)
            elif (system / library).is_file() or library.startswith(("api-ms-win-", "ext-ms-win-")):
                node["system"].append(library)
            else:
                raise ValueError("PE dependency missing from isolated appdir or System32: " + library)
        nodes[relative] = node
        if len(nodes) > 128:
            raise ValueError("PE closure exceeds reviewed bound")
    by_name = {Path(row["path"]).name.lower(): row for row in nodes.values()}
    changed = True
    while changed:
        changed = False
        for library in tuple(required):
            row = by_name[library]
            for child in set(row["normal"]) & set(row["non_system"]):
                if child not in required:
                    required.add(child)
                    changed = True
    closure = {
        "schema": "omni-strata-pe-closure-v1",
        "scope": "recursive_PE_normal_and_delay_imports_live_LoadLibrary_requires_separate_snapshot",
        "files": sorted(nodes.values(), key=lambda row: row["path"]),
        "required_at_load": sorted(required),
        "all_dynamic_loads_covered": False,
        "system_dependencies_pinned": False,
    }
    if max_file_bytes != 256 << 20:
        closure["explicit_max_file_bytes"] = max_file_bytes
    closure["identity_sha256"] = hashlib.sha256(canonical(closure).encode()).hexdigest()
    return closure


class OwnedWindowsProcess:
    """Keep the authoritative process object open through retirement.

    Access/query errors are unknown. Only an exact retained handle signalling
    exit, or OpenProcess ERROR_INVALID_PARAMETER for a missing PID, proves drain.
    """

    def __init__(self, identity):
        if not isinstance(identity, dict) or set(identity) != {"pid", "creation_filetime_100ns"}:
            raise ValueError("invalid exact encoder process identity")
        self.identity = copy.deepcopy(identity)
        self.pid = _uint(identity["pid"], True)
        if self.pid >= 2**32:
            raise ValueError("encoder PID exceeds Windows DWORD")
        self.created = _uint(identity["creation_filetime_100ns"], True)
        self._handle = None
        self._missing = False
        self._kernel = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel = kernel
        self._handle = kernel.OpenProcess(0x1000 | 0x100000, False, self.pid)
        if not self._handle:
            self._missing = ctypes.get_last_error() == 87  # ERROR_INVALID_PARAMETER, nonexistent process ID

    def state(self):
        """Return alive, retired or unknown; never turn query denial into exit."""
        if self._missing:
            return "retired"
        if self._handle is None or not self._handle or self._kernel is None:
            return "unknown"
        import ctypes
        from ctypes import wintypes

        times = [wintypes.FILETIME() for _ in range(4)]
        code = wintypes.DWORD()
        if not self._kernel.GetProcessTimes(self._handle, *(ctypes.byref(t) for t in times)):
            return "unknown"
        created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        if created != self.created:
            # A reused PID proves the old identity is gone but is not the owned
            # process object. Fail closed instead of adopting the new process.
            return "unknown"
        if not self._kernel.GetExitCodeProcess(self._handle, ctypes.byref(code)):
            return "unknown"
        waited = self._kernel.WaitForSingleObject(self._handle, 0)
        if waited == 0:  # WAIT_OBJECT_0: even a process whose exit code is STILL_ACTIVE has exited
            return "retired"
        if waited == 258 and code.value == 259:  # WAIT_TIMEOUT and STILL_ACTIVE
            return "alive"
        return "unknown"

    def close_retired(self):
        if self.state() != "retired":
            return False
        if self._handle:
            self._kernel.CloseHandle(self._handle)
            self._handle = None
        self._missing = True
        return True


def png_input(value, *, max_bytes, max_pixels, max_io_bytes, text_bytes=0):
    """Decode one complete PNG with bounds checked before decompressing pixels."""
    for bound in (max_bytes, max_pixels, max_io_bytes):
        _uint(bound, True)
    prefix = "data:image/png;base64,"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError("only an inline PNG data URL is accepted")
    # Bound allocation before base64 decode, including encoded transport bytes.
    if len(value) + text_bytes > max_io_bytes or len(value) - len(prefix) > 4 * ((max_bytes + 2) // 3):
        raise ValueError("encoded image exceeds admitted I/O")
    try:
        data = base64.b64decode(value[len(prefix) :], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("invalid PNG base64") from exc
    if len(data) > max_bytes or len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid or oversized PNG")
    if data[12:16] != b"IHDR" or struct.unpack(">I", data[8:12])[0] != 13:
        raise ValueError("PNG does not start with IHDR")
    width, height = struct.unpack(">II", data[16:24])
    if not width or not height or width * height > max_pixels:
        raise ValueError("PNG pixel bound exceeded")
    from PIL import Image

    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.size != (width, height) or getattr(image, "n_frames", 1) != 1:
                raise ValueError("animated or inconsistent PNG")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()  # Verify decompression/truncation before neural submission.
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "width": width,
        "height": height,
        "pixels": width * height,
    }


def verify_vision_route(config, runtime, runtime_manifest, runtime_files, artifacts, *, manifest_type):
    """Bind a separate projector and encoder to this exact text artifact.

    Runtime manifest verification is performed by the existing StageClient first.
    Build provenance remains byte-bound recorded evidence, not proof of execution.
    """
    required = {
        "schema",
        "projector_root",
        "projector_manifest",
        "projector_file",
        "text_artifact_manifest_sha256",
        "encoder_file",
        "encoder_build_manifest_file",
        "embedding_width",
        "max_image_bytes",
        "max_image_pixels",
        "max_image_tokens",
        "encoder_threads",
        "allow_cpu_fallback",
        "vision_host_bytes",
        "vision_gpu_bytes",
        "vision_scratch_bytes",
        "encoder_device",
    }
    if not isinstance(config, dict) or set(config) != required or config["schema"] != "omni-strata-image-route-v1":
        raise ValueError("invalid image route descriptor")
    if config["text_artifact_manifest_sha256"] != artifacts.manifest_sha256:
        raise ValueError("projector route belongs to another text artifact")
    if config["encoder_device"] not in {"cpu", "cuda"}:
        raise ValueError("encoder_device must explicitly be cpu or cuda")
    if type(config["allow_cpu_fallback"]) is not bool:
        raise ValueError("allow_cpu_fallback must be explicit boolean")
    if config["encoder_device"] == "cpu" and config["allow_cpu_fallback"] is not False:
        raise ValueError("a CPU encoder route cannot declare GPU fallback")
    numeric = {
        key: _uint(config[key], key != "vision_gpu_bytes")
        for key in required
        if key
        in {
            "embedding_width",
            "max_image_bytes",
            "max_image_pixels",
            "max_image_tokens",
            "encoder_threads",
            "vision_host_bytes",
            "vision_gpu_bytes",
            "vision_scratch_bytes",
        }
    }
    if (
        numeric["max_image_pixels"] > 4 * 1024 * 1024
        or numeric["max_image_tokens"] > 4096
        or numeric["max_image_bytes"] > 16 << 20
        or numeric["embedding_width"] > 65536
        or numeric["encoder_threads"] > 64
    ):
        raise ValueError("prototype image bound exceeds reviewed limit")
    root = Path(runtime).resolve(strict=True)
    verified = {Path(p).resolve(strict=True) for p in runtime_files}

    def member(relative):
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("vision path must be relative POSIX")
        part = Path(relative)
        path = (root / part).resolve(strict=True)
        if part.is_absolute() or ".." in part.parts or not path.is_relative_to(root) or path not in verified:
            raise ValueError("unbound vision runtime member")
        return path

    encoder = member(config["encoder_file"])
    build_file = member(config["encoder_build_manifest_file"])
    if build_file.stat().st_size > 2 << 20:
        raise ValueError("vision build provenance exceeds bound")
    build = _json(build_file.read_text(encoding="utf-8"))
    if not isinstance(build, dict) or build.get("schema") != "omni-strata-vision-build-v1":
        raise ValueError("missing reviewed vision build provenance")
    for key, expected in {
        "base_revision": BASE_REVISION,
        "dependency_revision": DEPENDENCY_REVISION,
        "dependency_tree": DEPENDENCY_TREE,
        "encoder_sha256": digest(encoder),
        "native_observation_schema": "strata-vision-backend-v1",
    }.items():
        if build.get(key) != expected:
            raise ValueError("vision source/dependency/binary identity mismatch")
    if build.get("configure_exit_code") != 0 or build.get("build_exit_code") != 0:
        raise ValueError("vision native build was not successful")
    cmake_options = {
        "STRATA_VISION_CUDA": "ON" if config["encoder_device"] == "cuda" else "OFF",
        "STRATA_PORTABLE": "ON",
        "CMAKE_BUILD_TYPE": "Release",
    }
    if config["encoder_device"] == "cuda":
        cmake_options["CMAKE_CUDA_ARCHITECTURES"] = "120"
    if build.get("cmake_options") != cmake_options:
        raise ValueError("unreviewed vision build options")
    reviewed, pe_file_limit = reviewed_vision_build(config["encoder_device"], digest(encoder))
    # Each provenance byte is also covered by the full runtime manifest. The
    # patch is separately reviewed when this prototype is applied, before build.
    proof_files = build.get("evidence_files")
    if not isinstance(proof_files, list) or len(proof_files) < 6:
        raise ValueError("vision build evidence is incomplete")
    roles = set()
    for item in proof_files:
        if not isinstance(item, dict) or set(item) != {"role", "path", "sha256"}:
            raise ValueError("invalid vision build evidence file")
        if item["role"] in roles:
            raise ValueError("duplicate vision provenance role")
        roles.add(item["role"])
        if item["role"] in reviewed and item["sha256"] != reviewed[item["role"]]:
            raise ValueError("vision patch/source differs from reviewed prototype")
        if digest(member(item["path"])) != item["sha256"]:
            raise ValueError("vision provenance bytes changed")
    if not set(reviewed) | {"configure_log", "build_log"} <= roles:
        raise ValueError("vision build/source lineage missing")
    dependencies = build.get("native_dependency_files")
    if not isinstance(dependencies, list):
        raise ValueError("vision selected DLL closure is missing")
    dependency_names = {encoder.name.lower()}
    for item in dependencies:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError("invalid vision selected dependency")
        path = member(item["path"])
        if (
            path.parent != encoder.parent
            or path.suffix.lower() != ".dll"
            or path.name.lower() in dependency_names
            or digest(path) != item["sha256"]
            or path.stat().st_size != item["size_bytes"]
        ):
            raise ValueError("vision selected dependency hash/path mismatch")
        dependency_names.add(path.name.lower())
    pe_closure = scan_pe_closure(encoder, root, verified, max_file_bytes=pe_file_limit)
    if build.get("pe_dependency_closure") != pe_closure:
        raise ValueError("encoder actual PE normal/delay closure differs from bound build receipt")
    observed_dependencies = {
        row["path"]: {"path": row["path"], "sha256": row["sha256"], "size_bytes": row["size_bytes"]}
        for row in pe_closure["files"]
        if row["path"] != encoder.relative_to(root).as_posix()
    }
    if {item["path"]: item for item in dependencies} != observed_dependencies:
        raise ValueError("declared encoder DLLs do not equal actual recursive non-system PE closure")
    projector_root = Path(config["projector_root"]).resolve(strict=True)
    projector = manifest_type.from_dict(config["projector_manifest"])
    projector_files = set(projector.verify(projector_root))
    relative = Path(config["projector_file"])
    projection = (projector_root / relative).resolve(strict=True)
    if relative.is_absolute() or ".." in relative.parts or not projection.is_relative_to(projector_root):
        raise ValueError("projector path escapes its manifest root")
    if projection not in projector_files or len(projector_files) != 1:
        raise ValueError("exactly one separately bound projector is required")
    records = [f for f in projector.files if (projector_root / f.path).resolve() == projection]
    if len(records) != 1 or records[0].role != "vision_projector":
        raise ValueError("projector manifest has the wrong file role")
    # One persistent SVE plus one combined SVE, image data and encoded HTTP.
    sve_bytes = 20 + 4 * numeric["embedding_width"] * numeric["max_image_tokens"]
    minimum_scratch = 2 * sve_bytes + numeric["max_image_bytes"]
    warmup_pixels = 2048 * 2048 if config["encoder_device"] == "cuda" else 0
    # The pinned GPU encoder always encodes a 2048-square dummy before READY,
    # irrespective of a route's smaller admitted image bound. CPU skips it.
    host_preprocessing = max(numeric["max_image_pixels"], warmup_pixels) * 32 + 2 * numeric["max_image_bytes"]
    host_preprocessing += warmup_pixels * 3
    if numeric["vision_host_bytes"] < host_preprocessing:
        raise ValueError("vision host budget omits bounded PNG decode/preprocessing")
    if config["encoder_device"] == "cuda" and numeric["vision_gpu_bytes"] < projector.total_size_bytes:
        raise ValueError("vision GPU budget is below the projector artifact size")
    if config["encoder_device"] == "cpu" and numeric["vision_gpu_bytes"] != 0:
        raise ValueError("CPU encoder weights/preprocessing must fit the host budget; GPU claim must be zero")
    if (config["encoder_device"] == "cpu" or config["allow_cpu_fallback"]) and (
        numeric["vision_host_bytes"] < projector.total_size_bytes + host_preprocessing
    ):
        raise ValueError("CPU encoder/fallback weights and preprocessing workspace must fit simultaneous host budget")
    if numeric["vision_scratch_bytes"] < minimum_scratch:
        raise ValueError("vision scratch budget omits image/SVE files")
    if config["encoder_device"] == "cuda":
        raise ValueError(
            "CUDA encoder route disabled pending authoritative physical GPU binding and native warmup admission"
        )
    identity = {
        "schema": "omni-strata-image-identity-v1",
        "runtime_manifest_sha256": runtime_manifest.manifest_sha256,
        "text_artifact_manifest_sha256": artifacts.manifest_sha256,
        "text_artifact_size_bytes": artifacts.total_size_bytes,
        "runtime_artifact_size_bytes": runtime_manifest.total_size_bytes,
        "projector_manifest_sha256": projector.manifest_sha256,
        "projector_sha256": digest(projection),
        "projector_size_bytes": projector.total_size_bytes,
        "encoder_sha256": digest(encoder),
        "encoder_build_manifest_sha256": digest(build_file),
        "encoder_pe_closure_sha256": pe_closure["identity_sha256"],
        "cache_policy": "disabled_reencode_each_image",
        "scratch_lifecycle": "owned_stage_temporary_drained_before_cleanup",
        "encoder_device": config["encoder_device"],
        "allow_cpu_fallback": config["allow_cpu_fallback"],
        "native_startup_warmup_pixels": warmup_pixels,
        "encoder_physical_gpu_identity": None,
        "bounds_and_declared_budgets": numeric,
        "full_model_placement": None,
        "all_encoder_operators_gpu_verified": False,
        "release_qualified": False,
        "build_proof_scope": "hash_bound_recorded_build_evidence_not_independent_rebuild",
    }
    identity["identity_sha256"] = hashlib.sha256(canonical(identity).encode()).hexdigest()
    return {
        "identity": identity,
        "encoder": str(encoder),
        "projector": str(projection),
        "config": copy.deepcopy(config),
        "native_dependency_files": dependencies,
        "pe_dependency_closure": pe_closure,
        "runtime_root": str(root),
    }


def bind_cuda_vision_lease(policy, bounds, workspaces, reservation, memory, gpu_index, gpu_pool):
    """Bind encoder sub-budgets to a fresh GPU probe and the shared lease.

    Called after the text stage's live admission check, before spawning workers.
    """
    keys = {"contract", "gpu_overhead_bytes", "host_overhead_bytes",
            "language_gpu_workspace_bytes", "language_host_workspace_bytes", "flash_attention"}
    if not isinstance(policy, dict) or set(policy) != keys or policy["flash_attention"] not in {"auto", "off"}:
        raise ValueError("invalid CUDA execution policy")
    evidence = CudaEncoderEvidence(policy["contract"])
    c = evidence.contract
    for key in keys - {"contract", "flash_attention"}:
        _uint(policy[key], True)
    _uint(gpu_index)
    if gpu_pool != f"vram:{gpu_index}" and not (gpu_index == 0 and gpu_pool == "vram"):
        raise ValueError("CUDA encoder pool differs from language GPU index")
    pci = memory.get("gpu_pci_bus_id")
    if not isinstance(pci, str):
        raise ValueError("CUDA physical PCI identity unavailable")
    pci = pci.lower()
    if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]", pci):
        pci = "0000" + pci
    uuid = memory.get("gpu_uuid")
    name_hash = memory.get("gpu_name_sha256")
    if (pci != c["pci_bus_id"] or not isinstance(uuid, str)
            or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid)
            or not isinstance(name_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", name_hash)):
        raise ValueError("CUDA encoder physical GPU binding mismatch")
    if c["max_image_tokens"] != bounds["max_image_tokens"]:
        raise ValueError("CUDA lease token bound mismatch")
    preprocessing = max(bounds["max_image_pixels"], 2048 * 2048) * 32
    preprocessing += 2048 * 2048 * 3 + 2 * bounds["max_image_bytes"]
    gpu = bounds["vision_gpu_bytes"]
    host = bounds["vision_host_bytes"]
    scratch = bounds["vision_scratch_bytes"]
    for value in (gpu, host, scratch, workspaces["gpu"], workspaces["host"],
                  reservation.get(gpu_pool, 0), reservation.get("host_ram", 0)):
        _uint(value)
    if (gpu < c["gpu_tensor_bytes"] + policy["gpu_overhead_bytes"]
            or host < c["host_tensor_bytes"] + preprocessing + policy["host_overhead_bytes"]
            or workspaces["gpu"] < gpu + policy["language_gpu_workspace_bytes"]
            or workspaces["host"] < host + scratch + policy["language_host_workspace_bytes"]
            or reservation.get(gpu_pool, 0) < workspaces["gpu"]
            or reservation.get("host_ram", 0) < workspaces["host"]):
        raise ValueError("shared CUDA lease omits simultaneous encoder/language resources")
    return {"contract": c, "physical_gpu_index": gpu_index, "gpu_pool": gpu_pool,
            "gpu_identity": {"uuid": uuid, "pci_bus_id": pci, "name_sha256": name_hash},
            "policy": copy.deepcopy(policy),
            "memory_probe_sha256": hashlib.sha256(canonical(memory).encode()).hexdigest(),
            "scope": "shared_lease_with_separate_tensor_and_declared_overhead_budgets"}


class CudaEncoderEvidence:
    """Validate native CUDA startup against a caller-owned tensor sub-budget.

    This proves reported explicit buffers, not total process/GPU memory or quality.
    The caller must separately bind this contract to the physical resource lease.
    """

    def __init__(self, contract):
        required = {"pci_bus_id", "gpu_tensor_bytes", "host_tensor_bytes", "max_image_tokens"}
        if not isinstance(contract, dict) or set(contract) != required:
            raise ValueError("invalid CUDA encoder evidence contract")
        pci = contract["pci_bus_id"]
        if not isinstance(pci, str) or not re.fullmatch(r"[0-9a-f]{8}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]", pci):
            raise ValueError("invalid canonical CUDA PCI identity")
        for key in ("gpu_tensor_bytes", "host_tensor_bytes", "max_image_tokens"):
            _uint(contract[key], key != "host_tensor_bytes")
        if contract["max_image_tokens"] > 4096:
            raise ValueError("CUDA token bound exceeds reviewed limit")
        self.contract = copy.deepcopy(contract)
        self.device = self.memory = self.warmup = None
        self.memory_samples = 0
        self.sealed = self.failed = False

    def observe(self, event, payload, *, request_active=False):
        try:
            self._observe(event, payload, request_active=request_active)
        except (ValueError, TypeError, KeyError):
            self.failed = True
            raise

    def _observe(self, event, payload, *, request_active):
        if self.failed or not isinstance(payload, dict):
            raise ValueError("invalid or retired CUDA evidence")
        c = self.contract
        if event == "cuda_device":
            if self.device is not None or self.sealed or payload != {
                "schema": "strata-vision-device-v1", "pci_bus_id": c["pci_bus_id"],
                "scope": "selected_primary_backend",
            }:
                raise ValueError("CUDA device identity mismatch or replay")
            self.device = copy.deepcopy(payload)
        elif event == "cuda_memory":
            if self.device is None or (self.sealed and not request_active) or (not self.sealed and self.warmup is not None):
                raise ValueError("CUDA memory evidence out of order")
            if set(payload) != {"schema", "scope", "gpu_bytes", "host_bytes", "gpu_limit", "host_limit", "phase"}:
                raise ValueError("invalid CUDA memory fields")
            if (payload["schema"] != "strata-vision-memory-v1"
                    or payload["scope"] != "explicit_ggml_tensor_buffers"
                    or payload["phase"] != "after_admitted_graph_allocation"):
                raise ValueError("unverified CUDA allocation report")
            for key in ("gpu_bytes", "host_bytes", "gpu_limit", "host_limit"):
                _uint(payload[key])
            if (payload["gpu_limit"] != c["gpu_tensor_bytes"]
                    or payload["host_limit"] != c["host_tensor_bytes"]
                    or payload["gpu_bytes"] > payload["gpu_limit"]
                    or payload["host_bytes"] > payload["host_limit"]):
                raise ValueError("CUDA allocation exceeds or changes admitted budget")
            self.memory = copy.deepcopy(payload)
            self.memory_samples += 1
        elif event == "cuda_warmup":
            if self.device is None or self.memory is None or self.warmup is not None or self.sealed:
                raise ValueError("CUDA warmup evidence out of order or replay")
            if set(payload) != {"schema", "pixels", "image_tokens", "max_image_tokens", "elapsed_ms", "status"}:
                raise ValueError("invalid CUDA warmup fields")
            for key in ("pixels", "image_tokens", "max_image_tokens"):
                _uint(payload[key], True)
            elapsed = payload["elapsed_ms"]
            if (payload["schema"] != "strata-vision-warmup-v1" or payload["status"] != "passed"
                    or payload["pixels"] != 2048 * 2048
                    or payload["max_image_tokens"] != c["max_image_tokens"]
                    or payload["image_tokens"] > c["max_image_tokens"]
                    or type(elapsed) not in (int, float) or not 0 <= elapsed <= 2**63 - 1
                    or not math.isfinite(elapsed)):
                raise ValueError("CUDA warmup bound or status mismatch")
            self.warmup = copy.deepcopy(payload)
        else:
            raise ValueError("unknown CUDA evidence event")

    def require_ready(self):
        if self.failed or self.sealed or self.device is None or self.memory is None or self.warmup is None:
            self.failed = True
            raise ValueError("CUDA READY lacks complete admitted startup evidence")
        self.sealed = True


class OwnedEncoderWriter:
    """Called only by the trusted bootstrap around an exact Popen object."""

    def __init__(self, nonce, generation, pid, created, emit, *, cwd, bounds, contained, cuda_contract=None):
        if not re.fullmatch(r"[0-9a-f]{32}", nonce) or not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise ValueError("invalid encoder ownership context")
        self.nonce, self.generation, self.pid, self.created = nonce, generation, _uint(pid, True), _uint(created, True)
        self.emit, self.cwd, self.bounds = emit, Path(cwd).resolve(strict=True), bounds
        self.cuda = CudaEncoderEvidence(cuda_contract) if cuda_contract is not None else None
        if self.cuda is not None and self.cuda.contract["max_image_tokens"] != bounds["max_image_tokens"]:
            raise ValueError("CUDA evidence token bound differs from image bound")
        self.sequence = self.request_seq = 0
        self.active = self.backend = self.last_end = None
        self.dispatched_seq = 0
        self.ready = False
        self._lock = threading.RLock()
        if contained is not True:
            raise ValueError("encoder containment unverified")
        self.frame("owner", {"contained": True})

    def frame(self, event, payload):
        with self._lock:
            self.sequence += 1
            row = {
                "schema": SCHEMA,
                "role": "encoder",
                "generation": self.generation,
                "pid": self.pid,
                "creation_filetime_100ns": self.created,
                "sequence": self.sequence,
                "request_seq": self.request_seq,
                "event": event,
                "payload": payload,
            }
            framed = PREFIX + self.nonce + " " + canonical(row) + "\n"
            if len(framed.encode()) > 4096:
                raise ValueError("encoder observation frame exceeds bound")
            self.emit(framed)

    def observe_input(self, line):
        with self._lock:
            if line == "QUIT\n":
                return
            if not self.ready or self.active is not None or not line.startswith("ENC ") or line.count("\n") != 1:
                raise ValueError("invalid/repeated encoder dispatch")
            fields = line.strip().split(" ")
            if len(fields) != 3:
                raise ValueError("encoder paths must not contain spaces")
            paths = [(self.cwd / p).resolve(strict=False) for p in fields[1:]]
            if any(not p.is_relative_to(self.cwd) or Path(raw).is_absolute() for p, raw in zip(paths, fields[1:])):
                raise ValueError("encoder path escapes its owned scratch directory")
            if paths[0].stat().st_size > self.bounds["max_image_bytes"]:
                raise ValueError("encoder image file exceeds admitted bound")
            data = paths[0].read_bytes()
            image = png_input(
                "data:image/png;base64," + base64.b64encode(data).decode(),
                max_bytes=self.bounds["max_image_bytes"],
                max_pixels=self.bounds["max_image_pixels"],
                max_io_bytes=self.bounds["max_io_bytes"],
            )
            self.request_seq += 1
            self.active = (paths[1], image)
            self.frame("encode_start", {"image_sha256": image["sha256"], "input_bytes": image["size_bytes"]})

    def observe_output(self, line):
        """Return True only for our observation line, hidden from upstream parser."""
        with self._lock:
            for prefix, event in (("OMNI_VISION_DEVICE_V1 ", "cuda_device"),
                                  ("OMNI_VISION_MEMORY_V1 ", "cuda_memory"),
                                  ("OMNI_VISION_WARMUP_V1 ", "cuda_warmup")):
                if line.startswith(prefix):
                    if self.cuda is None or len(line.encode()) > 2048 or not line.endswith("\n"):
                        raise ValueError("unexpected or oversized CUDA encoder evidence")
                    payload = _json(line[len(prefix):])
                    self.cuda.observe(event, payload, request_active=self.active is not None)
                    self.frame(event, payload)
                    return True
            if line.startswith(NATIVE_PREFIX):
                if self.backend is not None or self.ready or len(line.encode()) > 2048:
                    raise ValueError("repeated/oversized encoder backend observation")
                payload = _json(line[len(NATIVE_PREFIX) :])
                if set(payload) != {
                    "schema",
                    "primary_backend",
                    "device_type",
                    "gpu_requested",
                    "cpu_fallback_available",
                }:
                    raise ValueError("invalid native encoder backend observation")
                if (
                    payload["schema"] != "strata-vision-backend-v1"
                    or payload["device_type"] not in {"CPU", "GPU", "IGPU"}
                    or type(payload["gpu_requested"]) is not bool
                    or payload["cpu_fallback_available"] is not True
                    or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", payload["primary_backend"])
                ):
                    raise ValueError("invalid native encoder backend identity")
                if self.cuda is not None and (self.cuda.warmup is None or payload["device_type"] != "GPU"
                        or payload["gpu_requested"] is not True
                        or not re.fullmatch(r"CUDA[0-9]+", payload["primary_backend"])):
                    raise ValueError("CUDA encoder selected an unadmitted backend")
                self.backend = payload
                self.frame("backend", payload)
                return True
            if line.startswith("READY "):
                parts = line.split()
                if (
                    self.backend is None
                    or self.ready
                    or len(parts) != 2
                    or int(parts[1]) != self.bounds["embedding_width"]
                ):
                    raise ValueError("encoder READY missing verified backend/width")
                if self.cuda is not None:
                    self.cuda.require_ready()
                self.ready = True
                self.frame("ready", {"embedding_width": int(parts[1])})
            elif line.startswith("OK "):
                parts = line.split()
                if self.active is None or len(parts) != 5:
                    raise ValueError("encoder OK without matching ENC")
                n, nx, ny = (int(value) for value in parts[1:4])
                elapsed = float(parts[4])
                if not 0 < n <= self.bounds["max_image_tokens"] or min(nx, ny) <= 0 or nx * ny != n:
                    raise ValueError("encoder token/grid bound exceeded")
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("invalid encoder clock")
                output, _image = self.active
                size = 20 + n * self.bounds["embedding_width"] * 4
                if output.stat().st_size != size:
                    raise ValueError("encoder SVE size mismatch")
                with output.open("rb") as stream:
                    if struct.unpack("<5i", stream.read(20)) != (0x31455653, n, nx, ny, self.bounds["embedding_width"]):
                        raise ValueError("encoder SVE header mismatch")
                self.last_end = {
                    "image_tokens": n,
                    "nx": nx,
                    "ny": ny,
                    "sve_bytes": size,
                    "sve_sha256": digest(output),
                    "native_encoder_ms": elapsed,
                }
                self.frame("encode_end", self.last_end)
                self.active = None
            elif line.startswith("ERR"):
                self.frame("error", {"reason": "native_encoder_error"})
            return False

    def observe_language_input(self, line, pid, created):
        with self._lock:
            if not line.startswith("GENI "):
                return
            if (
                line.count("\n") != 1
                or self.active is not None
                or self.last_end is None
                or self.dispatched_seq == self.request_seq
            ):
                raise ValueError("GENI does not follow a completed owned encode")
            fields = line.strip().split(" ")
            if len(fields) < 4:
                raise ValueError("invalid GENI framing")
            combined = Path(fields[-2]).resolve(strict=True)
            if (
                not combined.is_relative_to(self.cwd)
                or combined.stat().st_size != self.last_end["sve_bytes"]
                or digest(combined) != self.last_end["sve_sha256"]
            ):
                raise ValueError("GENI SVE differs from the owned encoder output")
            self.frame(
                "language_dispatch",
                {
                    "command": "GENI",
                    "language_pid": _uint(pid, True),
                    "language_creation_filetime_100ns": _uint(created, True),
                    "sve_sha256": self.last_end["sve_sha256"],
                },
            )
            self.dispatched_seq = self.request_seq


class EncoderObserver:
    """Bounded state machine for nonce-authenticated role/phase observations."""

    def __init__(self, nonce, generation, *, allow_cpu_fallback, embedding_width, is_live, requested_device="cpu", cuda_contract=None):
        self.nonce, self.generation = nonce, generation
        if (requested_device not in {"cpu", "cuda"} or allow_cpu_fallback is not False
                or (requested_device == "cuda" and cuda_contract is None)
                or (requested_device == "cpu" and cuda_contract is not None)):
            raise ValueError(
                "only an explicit CPU encoder is enabled; CUDA/fallback awaits physical identity and warmup admission"
            )
        self.requested_device = requested_device
        self.cuda = CudaEncoderEvidence(cuda_contract) if cuda_contract is not None else None
        self.allow_cpu_fallback, self.embedding_width, self.is_live = allow_cpu_fallback, embedding_width, is_live
        self.identity = self.backend = self.language_identity = None
        self.sequence = self.last_request_seq = 0
        self.ready = self.retired = False
        self.active = self.last = None
        self.reasons = []
        self._lock = threading.RLock()

    def fail(self, reason):
        self.retired = True
        if reason not in self.reasons and len(self.reasons) < 8:
            self.reasons.append(reason)

    def ingest(self, line):
        marker = PREFIX + self.nonce + " "
        if not line.startswith(marker):
            return
        with self._lock:
            try:
                if len(line.encode()) > 4096 or not line.endswith("\n"):
                    raise ValueError("invalid_encoder_frame_bound")
                value = _json(line[len(marker) :])
                if set(value) != {
                    "schema",
                    "role",
                    "generation",
                    "pid",
                    "creation_filetime_100ns",
                    "sequence",
                    "request_seq",
                    "event",
                    "payload",
                }:
                    raise ValueError("invalid_encoder_frame_shape")
                if value["schema"] != SCHEMA or value["role"] != "encoder" or value["generation"] != self.generation:
                    raise ValueError("foreign_encoder_generation")
                identity = {key: _uint(value[key], True) for key in ("pid", "creation_filetime_100ns")}
                seq, reqseq = _uint(value["sequence"], True), _uint(value["request_seq"])
                if self.retired:
                    return
                if seq != self.sequence + 1:
                    raise ValueError("encoder_sequence_gap_or_replay")
                event, payload = value["event"], value["payload"]
                if event == "owner":
                    if self.identity is not None or seq != 1 or reqseq != 0 or payload != {"contained": True}:
                        raise ValueError("encoder_owner_restarted_or_uncontained")
                    self.identity = identity
                elif self.identity != identity:
                    raise ValueError("encoder_owner_mismatch")
                elif event in {"cuda_device", "cuda_memory", "cuda_warmup"}:
                    if self.cuda is None:
                        raise ValueError("unexpected_CUDA_encoder_event")
                    active = self.active is not None and self.active["start"] is not None
                    if self.ready:
                        if not active or reqseq != self.active["native_encoder_request_seq"]:
                            raise ValueError("CUDA_memory_not_bound_to_request")
                    elif reqseq != 0:
                        raise ValueError("CUDA_startup_request_sequence_invalid")
                    self.cuda.observe(event, payload, request_active=active)
                elif event == "backend":
                    if self.cuda is not None and self.cuda.warmup is None:
                        raise ValueError("CUDA_backend_before_verified_warmup")
                    if self.backend is not None or self.ready or reqseq != 0 or not isinstance(payload, dict):
                        raise ValueError("encoder_backend_repeated")
                    if (
                        set(payload)
                        != {"schema", "primary_backend", "device_type", "gpu_requested", "cpu_fallback_available"}
                        or payload.get("schema") != "strata-vision-backend-v1"
                        or payload.get("device_type") not in {"CPU", "GPU", "IGPU"}
                        or payload.get("gpu_requested") is not (self.requested_device == "cuda")
                        or payload.get("cpu_fallback_available") is not True
                        or not isinstance(payload.get("primary_backend"), str)
                        or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", payload["primary_backend"])
                    ):
                        raise ValueError("invalid_encoder_backend_observation")
                    self.backend = copy.deepcopy(payload)
                    if self.requested_device == "cpu" and (
                        payload["device_type"] != "CPU" or payload["primary_backend"] != "CPU"
                    ):
                        raise ValueError("CPU_encoder_route_selected_accelerator")
                    if (
                        self.requested_device == "cuda"
                        and payload["device_type"] != "CPU"
                        and not re.fullmatch(r"CUDA[0-9]+", payload["primary_backend"])
                    ):
                        raise ValueError("CUDA_encoder_selected_unverified_backend_family")
                    if (
                        payload.get("device_type") == "CPU"
                        and payload.get("gpu_requested")
                        and not self.allow_cpu_fallback
                    ):
                        raise ValueError("encoder_cpu_fallback_refused")
                elif event == "ready":
                    if (
                        self.backend is None
                        or self.ready
                        or reqseq != 0
                        or payload != {"embedding_width": self.embedding_width}
                    ):
                        raise ValueError("encoder_ready_unverified")
                    if self.cuda is not None:
                        self.cuda.require_ready()
                    self.ready = True
                elif event == "encode_start":
                    if self.active is None or self.active["start"] is not None or reqseq != self.last_request_seq + 1:
                        raise ValueError("encoder_dispatch_not_bound_to_active_request")
                    if (
                        not isinstance(payload, dict)
                        or set(payload) != {"image_sha256", "input_bytes"}
                        or type(payload.get("input_bytes")) is not int
                        or payload["input_bytes"] <= 0
                        or payload.get("image_sha256") != self.active["image_sha256"]
                    ):
                        raise ValueError("encoder_image_identity_mismatch")
                    self.active["start"] = copy.deepcopy(payload)
                    self.active["native_encoder_request_seq"] = reqseq
                elif event == "encode_end":
                    if (
                        self.active is None
                        or self.active["start"] is None
                        or self.active["end"] is not None
                        or reqseq != self.active["native_encoder_request_seq"]
                    ):
                        raise ValueError("encoder_completion_without_dispatch")
                    if (
                        not isinstance(payload, dict)
                        or set(payload) != {"image_tokens", "nx", "ny", "sve_bytes", "sve_sha256", "native_encoder_ms"}
                        or any(
                            type(payload[key]) is not int or payload[key] <= 0
                            for key in ("image_tokens", "nx", "ny", "sve_bytes")
                        )
                        or payload["nx"] * payload["ny"] != payload["image_tokens"]
                        or payload["sve_bytes"] != 20 + payload["image_tokens"] * self.embedding_width * 4
                        or not isinstance(payload["sve_sha256"], str)
                        or not re.fullmatch(r"[0-9a-f]{64}", payload["sve_sha256"])
                        or type(payload["native_encoder_ms"]) not in (int, float)
                        or not math.isfinite(payload["native_encoder_ms"])
                        or payload["native_encoder_ms"] < 0
                    ):
                        raise ValueError("invalid_encoder_completion")
                    self.active["end"] = copy.deepcopy(payload)
                elif event == "language_dispatch":
                    if (
                        self.active is None
                        or self.active["end"] is None
                        or self.active["dispatch"] is not None
                        or reqseq != self.active["native_encoder_request_seq"]
                        or not isinstance(payload, dict)
                        or set(payload) != {"command", "language_pid", "language_creation_filetime_100ns", "sve_sha256"}
                        or payload["command"] != "GENI"
                        or payload["sve_sha256"] != self.active["end"]["sve_sha256"]
                        or self.language_identity
                        != {
                            "pid": payload["language_pid"],
                            "creation_filetime_100ns": payload["language_creation_filetime_100ns"],
                        }
                    ):
                        raise ValueError("GENI_dispatch_owner_or_SVE_mismatch")
                    self.active["dispatch"] = copy.deepcopy(payload)
                elif event == "error":
                    raise ValueError("native_encoder_error")
                else:
                    raise ValueError("unknown_encoder_event")
                self.sequence = seq
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                self.fail(str(exc) or "invalid_encoder_frame")

    def check(self):
        with self._lock:
            if self.retired or self.identity is None or not self.ready or not self.is_live(self.identity):
                self.fail("owned_encoder_not_live_or_ready")
                raise RuntimeError("owned encoder is not live/ready")

    def begin(self, request_id, epoch, image):
        with self._lock:
            self.check()
            if self.active is not None:
                raise RuntimeError("encoder already has an active request")
            self.active = {
                "request_id": request_id,
                "epoch": _uint(epoch, True),
                "image_sha256": image["sha256"],
                "start": None,
                "end": None,
                "dispatch": None,
                "native_encoder_request_seq": None,
            }

    def finish(self, request_id, epoch, *, completed, reason=None):
        with self._lock:
            if self.last and (self.last["request_id"], self.last["epoch"]) == (request_id, epoch):
                return copy.deepcopy(self.last)
            if self.active is None or (self.active["request_id"], self.active["epoch"]) != (request_id, epoch):
                return None
            active = self.active
            if reason:
                self.fail(reason)
            if completed and not self.is_live(self.identity):
                self.fail("encoder_died_before_request_terminal")
            complete = (
                completed
                and not self.retired
                and active["start"] is not None
                and active["end"] is not None
                and active["dispatch"] is not None
            )
            if not complete and not self.reasons:
                self.reasons.append("image_chain_incomplete")
            self.last = {
                "schema": "omni-strata-image-request-observation-v1",
                "status": "complete" if complete else "incomplete",
                "generation": self.generation,
                "request_id": request_id,
                "epoch": epoch,
                "owned_encoder": copy.deepcopy(self.identity),
                "native_encoder_request_seq": active["native_encoder_request_seq"],
                "backend_selection": copy.deepcopy(self.backend),
                "input_sha256": active["image_sha256"],
                "encode_result": active["end"],
                "language_dispatch": active["dispatch"],
                "scope": "owned_native_encoder_then_pinned_server_SVE_GENI_chain",
                "all_encoder_operators_gpu_verified": False,
                "whole_model_placement": None,
                "physical_ssd_read_bytes": None,
                "release_qualified": False,
                "reasons": list(self.reasons),
            }
            if active["native_encoder_request_seq"] is not None:
                self.last_request_seq = active["native_encoder_request_seq"]
            self.active = None
            return copy.deepcopy(self.last)


# The text-route bootstrap literal stays byte-for-byte unchanged. This separately
# hashed composite adds only encoder ownership, SVE binding and private server hooks.
_VISION_PREAMBLE = r"""
import hashlib, importlib.util, json, os, subprocess, sys, threading
_v_path = os.environ.pop('OMNI_STRATA_VISION_ADAPTER')
_v_sha = os.environ.pop('OMNI_STRATA_VISION_ADAPTER_SHA256')
with open(_v_path, 'rb') as _source:
    if hashlib.sha256(_source.read()).hexdigest() != _v_sha:
        raise RuntimeError('Vision trusted adapter changed before bootstrap import')
_v_spec = importlib.util.spec_from_file_location('omni_strata_owned_vision', _v_path)
_v = importlib.util.module_from_spec(_v_spec); _v_spec.loader.exec_module(_v)
_v_nonce = os.environ['OMNI_STRATA_OBSERVER_NONCE']
_v_generation = os.environ['OMNI_STRATA_IO_GENERATION']
_v_args = json.loads(os.environ.pop('OMNI_STRATA_VISION_ARGS'))
_v_bounds = json.loads(os.environ.pop('OMNI_STRATA_VISION_BOUNDS'))
_v_cuda = json.loads(os.environ.pop('OMNI_STRATA_VISION_CUDA', 'null'))
if _v_cuda is not None:
    _v.CudaEncoderEvidence(_v_cuda['contract'])
    if _v_cuda['policy']['flash_attention'] not in ('auto', 'off'):
        raise RuntimeError('Unadmitted vision attention mode')

_v_scratch = os.path.realpath(os.environ.pop('OMNI_STRATA_VISION_SCRATCH'))
if (os.path.dirname(_v_scratch) != os.path.realpath(os.path.dirname(__file__))
    or os.path.basename(_v_scratch) != 'vision-scratch' or ' ' in _v_scratch
    or not os.path.isdir(_v_scratch)):
    raise RuntimeError('Vision scratch does not belong to this stage temporary directory')
_v_original = subprocess.Popen
_v_owned_encoder = None

def _v_launch(args, *a, **kw):
    global _v_owned_encoder
    if os.path.realpath(str(args[0])) != os.path.realpath(_v_args[0]):
        return _v_original(args, *a, **kw)
    if (list(map(str, args)) != _v_args or os.name != 'nt'
        or os.path.realpath(str(kw.get('cwd', ''))) != _v_scratch):
        raise RuntimeError('Vision child arguments/platform are not the pinned route')
    # No raw child output can acquire the supervisor nonce. Both selected native
    # binaries share the already pinned application directory and System32 policy.
    _env = dict(kw.get('env') or os.environ)
    _env.pop('OMNI_STRATA_OBSERVER_NONCE', None)
    _env.pop('OMNI_STRATA_IO_GENERATION', None)
    for _key in ('OMNI_VISION_EXPECTED_PCI_BUS_ID', 'OMNI_VISION_GGML_GPU_BUDGET_BYTES',
                 'OMNI_VISION_GGML_HOST_BUDGET_BYTES'):
        _env.pop(_key, None)
    if _v_cuda is not None:
        _contract = _v_cuda['contract']
        _env['CUDA_VISIBLE_DEVICES'] = _v_cuda['gpu_identity']['uuid']
        _env['OMNI_VISION_EXPECTED_PCI_BUS_ID'] = _contract['pci_bus_id']
        _env['OMNI_VISION_GGML_GPU_BUDGET_BYTES'] = str(_contract['gpu_tensor_bytes'])
        _env['OMNI_VISION_GGML_HOST_BUDGET_BYTES'] = str(_contract['host_tensor_bytes'])
        args = [*args, '--flash-attn', _v_cuda['policy']['flash_attention']]
    _env['PATH'] = os.path.dirname(os.path.realpath(args[0])) + os.pathsep + system_directory
    kw['env'] = _env
    kw['stderr'] = subprocess.DEVNULL
    p = _v_original(args, *a, **kw)
    try:
        from serve.winjob import contain
        if not contain(p): raise RuntimeError('Vision child containment failed')
        import ctypes
        from ctypes import wintypes
        _k = ctypes.WinDLL('kernel32', use_last_error=True)
        _k.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        _k.GetProcessTimes.restype = wintypes.BOOL
        _times = [wintypes.FILETIME() for _ in range(4)]
        if not _k.GetProcessTimes(int(p._handle), *(ctypes.byref(t) for t in _times)):
            raise RuntimeError('Vision child creation time is unknown')
        _created = (_times[0].dwHighDateTime << 32) | _times[0].dwLowDateTime
        def _emit(row): sys.stderr.write(row); sys.stderr.flush()
        _owned = _v.OwnedEncoderWriter(_v_nonce, _v_generation, p.pid, _created, _emit,
                                       cwd=kw['cwd'], bounds=_v_bounds, contained=True,
                                       cuda_contract=_v_cuda['contract'] if _v_cuda is not None else None)
        _v_owned_encoder = _owned
        class _Input:
            def __init__(self, pipe): self.pipe = pipe
            def write(self, row):
                _owned.observe_input(row.decode('utf-8', 'strict') if isinstance(row, bytes) else row)
                return self.pipe.write(row)
            def __getattr__(self, name): return getattr(self.pipe, name)
        class _Output:
            def __init__(self, pipe): self.pipe = pipe
            def readline(self, *args):
                while True:
                    row = self.pipe.readline(4097)
                    if not row: return row
                    if len(row) > 4096 or not row.endswith("\n" if isinstance(row, str) else b"\n"):
                        raise RuntimeError("Encoder output framing bound exceeded")
                    text = row.decode('utf-8', 'strict') if isinstance(row, bytes) else row
                    if not _owned.observe_output(text): return row
            def __getattr__(self, name): return getattr(self.pipe, name)
        p.stdin, p.stdout = _Input(p.stdin), _Output(p.stdout)
        return p
    except BaseException:
        p.kill(); p.wait(timeout=5)
        raise
subprocess.Popen = _v_launch
"""

_VISION_MAIN = r"""
_v_server = runpy.run_path(server, run_name='omni_strata_vision_server')
_v_class = _v_server['Vision']
_v_class.work_dir = staticmethod(lambda: _v_server['Path'](_v_scratch))
_v_encode = _v_class.encode

def _v_uncached_encode(self, source, keep=()):
    # Single server slot owns this call. Clear only idle native SVE cache files;
    # never delete embeddings in use by a language request. No host neural stage.
    if keep: raise RuntimeError('Only one image is admitted by this route')
    with self.lock:
        for path, _tokens in self.cache.values(): path.unlink(missing_ok=True)
        self.cache.clear()
    return _v_encode(self, source, keep=())
_v_class.encode = _v_uncached_encode
sys.exit(_v_server['main']())
"""


def bootstrap_source(text_bootstrap):
    terminal = "runpy.run_path(server, run_name='__main__')"
    if text_bootstrap.count(terminal) != 1:
        raise ValueError("unreviewed text bootstrap shape")
    # The original literal remains untouched in strata.py. The composite has a
    # separate hash, and its base-text observer identity is explicitly scoped.
    bridge = "owned_io.observe_input(text)\n            _v_owned_encoder.observe_language_input(text, p.pid, created)"
    if text_bootstrap.count("owned_io.observe_input(text)") != 1:
        raise ValueError("unreviewed native input wrapper shape")
    return _VISION_PREAMBLE + text_bootstrap.replace(terminal, _VISION_MAIN).replace(
        "owned_io.observe_input(text)", bridge
    )
