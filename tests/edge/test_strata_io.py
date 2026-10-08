# SPDX-License-Identifier: Apache-2.0
"""Mock protocol/counter tests only: no model loading, GPU work or qualification."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from vllm_omni.engine.backends import strata_io
from vllm_omni.engine.backends.strata_io import (
    BASE_REVISION,
    DEPENDENCY_REVISION,
    EXPERT_BOOLS,
    EXPERT_FLOATS,
    EXPERT_INTS,
    PATCH_SHA256,
    PHASES,
    PLE_BOOLS,
    PLE_FLOATS,
    PLE_INTS,
    PREFIX,
    SCOPE,
    OwnedNativeFrameWriter,
    StrataIoObserver,
)

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]

NONCE = "a" * 32
GENERATION = "owned-test-generation"
PID = 1234
CREATED = 1234567890123
IDENTITY = {
    "schema": "omni-strata-observed-runtime-v1",
    "base_revision": BASE_REVISION,
    "dependency_revision": DEPENDENCY_REVISION,
    "patch_sha256": PATCH_SHA256,
    "native_io_schema": "strata-omni-io-v1",
    "three_tier_memory_qualified": False,
    "test_only": "mock identity descriptor; no built/neural runtime claim",
}
IDENTITY["identity_sha256"] = hashlib.sha256(
    json.dumps(IDENTITY, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def snapshot(phase, sequence=1, count=0):
    row = {
        "schema": "strata-omni-io-v1",
        "request_seq": sequence,
        "phase": phase,
        "scope": SCOPE,
        "qpc": 100 + count * 10,
        "qpc_hz": 100,
        "physical_ssd_read_bytes": None,
        "expert": {key: count for key in EXPERT_INTS | EXPERT_FLOATS},
        "ple": {key: count for key in PLE_INTS | PLE_FLOATS},
    }
    row["expert"].update({key: True for key in EXPERT_BOOLS})
    row["ple"].update({key: True for key in PLE_BOOLS})
    for group, keys in (
        (
            "expert",
            {
                "direct_submit_errors",
                "direct_completion_errors",
                "direct_short_reads",
                "mapped_fallback_blobs",
            },
        ),
        ("ple", {"submit_errors", "completion_errors", "short_reads"}),
    ):
        row[group].update({key: 0 for key in keys})
    return row


def setup_pair():
    rows = []
    writer = OwnedNativeFrameWriter(NONCE, GENERATION, PID, CREATED, rows.append)
    observer = StrataIoObserver(NONCE, GENERATION, PID, CREATED, IDENTITY)
    return rows, writer, observer


def flush(rows, observer):
    result = [observer.ingest(row) for row in rows]
    rows.clear()
    return result


def run_complete(rows, writer, observer, *, request_id="request-a", epoch=1, sequence=1, base_count=0):
    observer.begin(request_id, epoch)
    writer.observe_input("GEN 128 PRIVATE_TOKEN_IDS\n")
    for index, phase in enumerate(PHASES):
        writer.observe_output("OMNI_IO_V1 " + json.dumps(snapshot(phase, sequence, base_count + index)))
    writer.observe_output("DONE 128 30 1.0 2.0 length 0 0 0 1 2 0 0 0.0 30 0\n")
    assert "PRIVATE_TOKEN_IDS" not in "".join(rows)
    assert flush(rows, observer) == [True] * 5
    return observer.finish(request_id, epoch, completed=True)


def test_complete_owner_bound_phase_report_has_exact_deltas_and_no_physical_claim():
    rows, writer, observer = setup_pair()
    report = run_complete(rows, writer, observer)
    assert report["status"] == "complete"
    assert report["native_request_seq"] == 1
    assert report["epoch"] == 1
    assert report["intervals"]["prefill"]["expert"]["logical_file_bytes"] == 1
    assert report["intervals"]["whole_native_interval"]["ple"]["keepalive_bytes"] == 2
    assert report["intervals"]["whole_native_interval"]["boundary_wall_s"] == 0.2
    assert report["physical_ssd_read_bytes"] is None
    assert report["loading_covered"] is False
    assert report["three_tier_memory_qualified"] is False
    assert "background_work" in report["attribution"]
    assert all("PRIVATE_TOKEN_IDS" not in row for row in rows)


def test_relaxed_atomic_snapshot_does_not_require_cross_counter_inequality():
    rows, writer, observer = setup_pair()
    observer.begin("request", 1)
    writer.observe_input("GEN 4 PRIVATE\n")
    for index, phase in enumerate(PHASES):
        value = snapshot(phase, count=index)
        # Concurrent field loads can observe a later completion than the
        # submission field. Only each individual cumulative counter is ordered.
        value["expert"]["direct_completed"] = 5 + index
        value["expert"]["direct_submitted"] = index
        writer.observe_output("OMNI_IO_V1 " + json.dumps(value))
    writer.observe_output("DONE 4 3 1 1 length 0 0 0 0 0 0 0 0 0 0\n")
    assert all(flush(rows, observer))
    assert observer.finish("request", 1, completed=True)["status"] == "complete"


def test_late_frames_are_ignored_and_new_request_has_its_own_sequence_epoch():
    rows, writer, observer = setup_pair()
    first = run_complete(rows, writer, observer)
    stale = (
        PREFIX
        + NONCE
        + " "
        + json.dumps(
            {
                "schema": "omni-strata-owned-io-frame-v1",
                "generation": GENERATION,
                "pid": PID,
                "creation_filetime_100ns": CREATED,
                "dispatch_seq": 1,
                "kind": "snapshot",
                "payload": first["raw_snapshots"][-1],
            }
        )
    )
    observer.begin("request-b", 3)
    assert observer.ingest(stale) is False
    with pytest.raises(ValueError, match="stale"):
        observer.finish("request-a", 1, completed=True)
    writer.observe_input("GEN 4 PRIVATE_B\n")
    for index, phase in enumerate(PHASES):
        writer.observe_output("OMNI_IO_V1 " + json.dumps(snapshot(phase, 2, 3 + index)))
    writer.observe_output("DONE 4 3 1 1 stop 0 0 0 0 0 0 0 0 0 0\n")
    assert all(flush(rows, observer))
    second = observer.finish("request-b", 3, completed=True)
    assert second["status"] == "complete"
    assert second["native_request_seq"] == 2
    assert second["request_id"] == "request-b"


@pytest.mark.parametrize(
    "mutation",
    [
        "pid",
        "creation",
        "generation",
        "sequence",
        "phase",
        "decreased",
        "duplicate",
        "physical",
        "bool_counter",
    ],
)
def test_invalid_owned_evidence_remains_incomplete_and_retires_generation(mutation):
    rows, writer, observer = setup_pair()
    observer.begin("request", 1)
    writer.observe_input("GEN 4 PRIVATE\n")
    writer.observe_output("OMNI_IO_V1 " + json.dumps(snapshot(PHASES[0], count=2)))
    assert all(flush(rows, observer))
    writer.observe_output("OMNI_IO_V1 " + json.dumps(snapshot(PHASES[1], count=3)))
    frame = json.loads(rows.pop()[len(PREFIX + NONCE + " ") :])
    if mutation == "pid":
        frame["pid"] += 1
    elif mutation == "creation":
        frame["creation_filetime_100ns"] += 1
    elif mutation == "generation":
        frame["generation"] = "stale-generation"
    elif mutation == "sequence":
        frame["payload"]["request_seq"] = 2
    elif mutation == "phase":
        frame["payload"]["phase"] = PHASES[2]
    elif mutation == "decreased":
        frame["payload"]["expert"]["logical_file_bytes"] = 1
    elif mutation == "duplicate":
        frame["payload"]["phase"] = PHASES[0]
    elif mutation == "physical":
        frame["payload"]["physical_ssd_read_bytes"] = 123
    elif mutation == "bool_counter":
        frame["payload"]["expert"]["direct_completed"] = True
    assert observer.ingest(PREFIX + NONCE + " " + json.dumps(frame)) is False
    result = observer.finish("request", 1, completed=True)
    assert result["status"] == "incomplete"
    assert result["physical_ssd_read_bytes"] is None
    with pytest.raises(ValueError, match="available"):
        observer.begin("request-b", 2)


def test_wrong_nonce_and_native_raw_lines_cannot_authenticate_observations():
    rows, writer, observer = setup_pair()
    observer.begin("request", 1)
    writer.observe_input("GEN 4 PRIVATE\n")
    dispatch = rows.pop()
    assert observer.ingest(dispatch.replace(NONCE, "c" * 32)) is False
    assert observer.ingest("OMNI_IO_V1 " + json.dumps(snapshot(PHASES[0]))) is False
    assert observer.ingest(dispatch) is True
    assert observer.retired is False


@pytest.mark.parametrize("ending", ["absent_end", "cancel", "error", "host_cancel", "io_error"])
def test_failure_or_missing_boundary_never_yields_complete_observation(ending):
    rows, writer, observer = setup_pair()
    observer.begin("request", 1)
    writer.observe_input("GEN 4 PRIVATE\n")
    for index, phase in enumerate(PHASES):
        if ending == "absent_end" and phase == "request_end":
            break
        value = snapshot(phase, count=index)
        if ending == "io_error" and phase == "request_end":
            value["ple"]["completion_errors"] = 1
        writer.observe_output("OMNI_IO_V1 " + json.dumps(value))
    if ending == "error":
        writer.observe_output("ERR PRIVATE_ERROR_TEXT\n")
    else:
        finish = "cancel" if ending == "cancel" else "length"
        writer.observe_output(f"DONE 4 3 1 1 {finish} 0 0 0 0 0 0 0 0 0 0\n")
    flush(rows, observer)
    report = observer.finish("request", 1, completed=ending != "host_cancel")
    assert report["status"] == "incomplete"
    assert report["physical_ssd_read_bytes"] is None
    assert observer.retired is True
    assert "PRIVATE_ERROR_TEXT" not in json.dumps(report)


def test_unknown_or_duplicate_json_fields_are_rejected():
    rows, writer, observer = setup_pair()
    observer.begin("request", 1)
    writer.observe_input("GEN 4 PRIVATE\n")
    dispatch = rows.pop().replace('"dispatch_seq":1', '"dispatch_seq":1,"dispatch_seq":1')
    assert observer.ingest(dispatch) is False
    assert observer.finish("request", 1, completed=False)["status"] == "incomplete"


@pytest.fixture
def observation_bundle(tmp_path, monkeypatch):
    """Synthetic build bytes/metadata; never an actual native or neural build."""
    root = tmp_path

    def write(path, value):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value if isinstance(value, bytes) else value.encode())
        return target

    def record(path):
        file = root / path
        return {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "size_bytes": file.stat().st_size}

    def dump(path, value):
        write(path, json.dumps(value))

    engine = write("engine/strata.exe", b"FAKE EXECUTABLE: metadata tests only")
    patch = write("provenance/runtime.patch", b"FAKE PATCH: metadata tests only")
    patch_sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    monkeypatch.setattr(strata_io, "PATCH_SHA256", patch_sha)
    sources = [
        "include/strata/core/expert_source.hpp",
        "include/strata/kernels/ngram.hpp",
        "include/strata/ngram/ple_reader.hpp",
        "src/core/expert_source.cpp",
        "src/kernels/ngram.cpp",
        "src/ngram/ple_reader.cpp",
        "src/program/generate.cpp",
    ]
    source_records = []
    for name in sources:
        write("provenance/sources/" + name, "FAKE SOURCE " + name)
        source_records.append(
            {"path": name, "base_sha256": "0" * 64, "patched_sha256": record("provenance/sources/" + name)["sha256"]}
        )
    dump("provenance/patch.json", {"base_revision": BASE_REVISION, "patch_sha256": patch_sha, "files": source_records})
    dump(
        "provenance/dependency.json",
        {"revision": DEPENDENCY_REVISION, "tree": strata_io.DEPENDENCY_TREE, "git_status_porcelain": ""},
    )
    flags = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": "120",
        "CMAKE_CUDA_RUNTIME_LIBRARY": "Static",
        "STRATA_ENABLE_CUDA": "ON",
        "STRATA_PORTABLE": "ON",
        "STRATA_NATIVE_EXPERTS": "ON",
        "STRATA_MMQ_KQUANTS": "OFF",
        "STRATA_BUILD_TESTS": "OFF",
    }
    write("provenance/build/CMakeCache.txt", "\n".join(f"{k}:STRING={v}" for k, v in flags.items()) + "\n")
    dump(
        "provenance/build/compile_commands.json",
        [{"file": "C:/fixture/" + name} for name in sources if name.endswith(".cpp")],
    )
    write("provenance/build/build.ninja", "FAKE NINJA: metadata only")
    tools = {}
    for name, version in {"nvcc": "13.4", "cmake": "3.31.6", "ninja": "1.12.1", "cl": "19.44.35229"}.items():
        write(f"provenance/build/{name}-version.log", version)
        tools[name] = {
            "log_sha256": record(f"provenance/build/{name}-version.log")["sha256"],
            "exit_code": 2 if name == "cl" else 0,
        }
    commands = [
        ["cmake", *[f"-D{k}={v}" for k, v in flags.items()]],
        ["cmake", "--build", "build", "--target", "strata", "--parallel", "2"],
    ]
    steps = []
    for index, command in enumerate(commands):
        write(f"provenance/build/step-{index}.log", "FAKE SUCCESS LOG")
        steps.append(
            {"command": command, "exit_code": 0, "sha256": record(f"provenance/build/step-{index}.log")["sha256"]}
        )
    receipt = {
        "schema": "omni-strata-private-build-v1",
        "status": "built_not_installed_not_neurally_qualified",
        "base_revision": BASE_REVISION,
        "dependency_revision": DEPENDENCY_REVISION,
        "patch_sha256": patch_sha,
        "parallelism": 2,
        "steps": steps,
        "commands": commands,
        "tools": tools,
        "files": {
            name: record("engine/strata.exe" if name == "strata.exe" else "provenance/build/" + name)
            for name in ("strata.exe", "CMakeCache.txt", "compile_commands.json", "build.ninja")
        },
    }
    dump("provenance/build/receipt.json", receipt)
    dlls = []
    for name in ("cublas64_13.dll", "cublasLt64_13.dll"):
        path = "engine/" + name
        write(path, "FAKE DLL: " + name)
        dlls.append({"path": path, **record(path)})
    imports = {
        "engine/strata.exe": ["cublas64_13.dll", "KERNEL32.dll", "ADVAPI32.dll"],
        "engine/cublas64_13.dll": ["cublasLt64_13.dll", "KERNEL32.dll"],
        "engine/cublasLt64_13.dll": ["KERNEL32.dll"],
    }
    for path, names in imports.items():
        write("provenance/imports/" + Path(path).name + ".dependents.txt", "\n".join("    " + name for name in names))
    dump(
        "provenance/dlls.json",
        {
            "schema": "omni-strata-runtime-dependencies-v1",
            "files": dlls,
            "static_imports": imports,
            "static_import_observer": {
                "tool": "dumpbin /DEPENDENTS",
                "tool_sha256": "0" * 64,
                "raw_logs_dir": "provenance/imports",
            },
            "native_dll_search_policy": {"required_directories": ["engine", "System32"], "required_at_launch": True},
        },
    )
    descriptor = {
        "schema": "omni-strata-observation-runtime-v1",
        "native_io_schema": "strata-omni-io-v1",
        "base_revision": BASE_REVISION,
        "dependency_revision": DEPENDENCY_REVISION,
        "dependency_tree": strata_io.DEPENDENCY_TREE,
        "engine_file": "engine/strata.exe",
        "patch_file": "provenance/runtime.patch",
        "patch_manifest_file": "provenance/patch.json",
        "build_receipt_file": "provenance/build/receipt.json",
        "dependency_provenance_file": "provenance/dependency.json",
        "patched_sources_dir": "provenance/sources",
        "build_evidence_dir": "provenance/build",
        "runtime_dependencies_file": "provenance/dlls.json",
    }
    return root, descriptor, {path.resolve() for path in root.rglob("*") if path.is_file()}, engine


def test_observed_runtime_verifies_complete_byte_bound_build_and_actual_host_sources(observation_bundle):
    root, descriptor, files, engine = observation_bundle
    identity = strata_io.verify_observation_runtime(descriptor, root, files, engine)
    assert (
        identity["supervisor_bootstrap_sha256"]
        == hashlib.sha256(strata_io.production_bootstrap_source().encode()).hexdigest()
    )
    assert identity["io_adapter_sha256"] == hashlib.sha256(Path(strata_io.__file__).read_bytes()).hexdigest()
    assert len(identity["patch_source_hashes"]) == 7
    assert identity["live_loaded_module_paths_verified"] is False
    assert identity["three_tier_memory_qualified"] is False
    with pytest.raises(ValueError, match="bootstrap"):
        strata_io.verify_observation_runtime(descriptor, root, files, engine, "caller-supplied bootstrap")


@pytest.mark.parametrize(
    "target", ["source", "tool_log", "build_log", "cache", "compile", "dll", "engine", "patch", "dependency"]
)
def test_observed_runtime_rejects_changed_build_source_dependency_or_native_bytes(observation_bundle, target):
    root, descriptor, files, engine = observation_bundle
    paths = {
        "source": "provenance/sources/src/core/expert_source.cpp",
        "tool_log": "provenance/build/cl-version.log",
        "build_log": "provenance/build/step-1.log",
        "cache": "provenance/build/CMakeCache.txt",
        "compile": "provenance/build/compile_commands.json",
        "dll": "engine/cublasLt64_13.dll",
        "engine": "engine/strata.exe",
        "patch": "provenance/runtime.patch",
    }
    if target == "dependency":
        file = root / descriptor["dependency_provenance_file"]
        value = json.loads(file.read_text())
        value["tree"] = "0" * 40
        file.write_text(json.dumps(value))
    else:
        (root / paths[target]).write_bytes(b"CHANGED BY TEST")
    with pytest.raises(ValueError):
        strata_io.verify_observation_runtime(descriptor, root, files, engine)


def test_observed_runtime_rejects_failed_step_missing_manifest_member_or_semantic_flags(observation_bundle):
    root, descriptor, files, engine = observation_bundle
    with pytest.raises(ValueError, match="absent from runtime manifest"):
        strata_io.verify_observation_runtime(descriptor, root, files - {engine.resolve()}, engine)
    receipt_path = root / descriptor["build_receipt_file"]
    receipt = json.loads(receipt_path.read_text())
    receipt["steps"][1]["exit_code"] = 1
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="successful steps"):
        strata_io.verify_observation_runtime(descriptor, root, files, engine)
    receipt["steps"][1]["exit_code"] = 0
    cache = root / descriptor["build_evidence_dir"] / "CMakeCache.txt"
    cache.write_text(cache.read_text().replace("STRATA_PORTABLE:STRING=ON", "STRATA_PORTABLE:STRING=OFF"))
    receipt["files"]["CMakeCache.txt"] = {
        "size_bytes": cache.stat().st_size,
        "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="build flags"):
        strata_io.verify_observation_runtime(descriptor, root, files, engine)


def test_selected_module_audit_never_claims_missing_native_identity(observation_bundle):
    root, descriptor, files, engine = observation_bundle
    identity = strata_io.verify_observation_runtime(descriptor, root, files, engine)
    result = strata_io.audit_selected_loaded_modules(None, root, identity)
    assert result["status"] == "unverified"
    assert result["all_os_modules_covered"] is False


@pytest.mark.skipif(os.name != "nt", reason="actual GetProcessTimes/module query is Windows-specific")
def test_selected_module_audit_refuses_live_wrong_module_set_and_reused_identity(observation_bundle):
    import ctypes
    from ctypes import wintypes

    root, descriptor, files, engine = observation_bundle
    identity = strata_io.verify_observation_runtime(descriptor, root, files, engine)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    times = [wintypes.FILETIME() for _ in range(4)]
    assert kernel.GetProcessTimes(kernel.GetCurrentProcess(), *(ctypes.byref(t) for t in times))
    created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    owned = {"pid": os.getpid(), "creation_filetime_100ns": created}
    # This test process is Python, not the configured engine; knowing a live
    # PID/creation time cannot substitute for the selected runtime module set.
    result = strata_io.audit_selected_loaded_modules(owned, root, identity)
    assert result["status"] == "unverified" and result["all_os_modules_covered"] is False
    assert "missing" in result["reasons"][0] or "mismatch" in result["reasons"][0]
    stale = strata_io.audit_selected_loaded_modules(owned | {"creation_filetime_100ns": created + 1}, root, identity)
    assert stale["status"] == "unverified" and "identity changed" in stale["reasons"][0]


@pytest.mark.parametrize("change", ["different_sources", "extra_import", "changed_import_log"])
def test_observed_runtime_refuses_unreviewed_source_set_or_native_import_closure(observation_bundle, change):
    root, descriptor, files, engine = observation_bundle
    if change == "different_sources":
        path = root / descriptor["patch_manifest_file"]
        value = json.loads(path.read_text())
        value["files"][0]["path"] = "other.hpp"
    elif change == "extra_import":
        path = root / descriptor["runtime_dependencies_file"]
        value = json.loads(path.read_text())
        value["static_imports"]["engine/strata.exe"].append("unreviewed.dll")
    else:
        path = root / "provenance/imports/strata.exe.dependents.txt"
        path.write_text(path.read_text() + "\n    unreviewed.dll\n")
    if change != "changed_import_log":
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source files|dependency closure|declared closure"):
        strata_io.verify_observation_runtime(descriptor, root, files, engine)


def test_owned_oversized_frame_and_boolean_epoch_refuse():
    rows, writer, observer = setup_pair()
    observer.begin("request", 1)
    assert observer.ingest(PREFIX + NONCE + " " + "x" * 5000) is False
    with pytest.raises(ValueError, match="unsigned"):
        observer.finish("request", True, completed=False)
    assert observer.finish("request", 1, completed=False)["status"] == "incomplete"
