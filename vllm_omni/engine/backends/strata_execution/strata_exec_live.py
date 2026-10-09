"""Private Stage-owned live execution binding candidate; not deployed.

No module enumeration is implemented here. The actual existing Windows module
auditor and native process-birth helper remain authoritative. Source preparation
does not execute any of the imports, Windows probes or byte-verification below.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath

import strata_exec as parser
import strata_exec_runtime as static_verifier

STATIC_VERIFIER_SHA = "273b81b115756988ef990d4a0ffde86104141787f1658bd0e90ec596f9753111"
PARSER_SHA = "c653d6eb422ed1f5ca970d5db1d0d3c8d08139a1e085eabd171497490a7663bf"
BRIDGE_SHA = "5911125621e1d27cdcc9e12f92683e7e481e411cfe099fe6ce95b42c423e63fe"
ENGINE_ROLE_IO_SHA = "8dd65287a447e20bc3096408ae7606b38db4aeb44216ea678d57c09e406fde43"
MAX_RECEIPT_BYTES = 131072
MAX_CONTEXT_BYTES = 65536
MAX_SOURCE_BYTES = 262144
MAX_RECEIPTS = 8
MODULE_NAMES = {
    "stage": "vllm_omni.engine.backends.strata",
    "io": "vllm_omni.engine.backends.strata_io",
    "bridge": "strata_exec_bridge", "parser": "strata_exec",
    "static": "strata_exec_runtime", "live": "strata_exec_live",
}


class LiveBindingError(ValueError):
    """Only bounded stable reason codes may enter a failure receipt."""


def require(condition, code):
    if not condition:
        raise LiveBindingError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def source_bytes(path):
    return static_verifier.file_bytes(Path(path), MAX_SOURCE_BYTES)


def live_clock_identity():
    info = time.get_clock_info("perf_counter")
    require(info.monotonic is True and info.adjustable is False, "live_high_resolution_monotonic_clock_unavailable")
    return {"source": "time.perf_counter_ns", "implementation": info.implementation,
            "monotonic": info.monotonic, "adjustable": info.adjustable, "resolution_s": info.resolution,
            "scope": "fresh_verifier_callback_timestamp_not_native_compute_or_latency_measurement"}


def _reopen_static_bundle(root, descriptor, static):
    """Reuse the exact descriptor-bound source exceptions, never arbitrary roots."""
    bundle = static_verifier.Bundle(root, descriptor["manifest_file"],
        dependency_source_roots=static_verifier.dependency_source_roots(descriptor),
        strata_source_roots=static_verifier.strata_source_roots(descriptor))
    require(bundle.manifest_sha256 == static["runtime_manifest_sha256"], "live_static_manifest_binding")
    return bundle


def _module_preimages(bundle, context_file):
    """Read actual archived source bytes and the actual loaded module locations."""
    context = static_verifier.object_(static_verifier.json_(bundle.read(context_file, MAX_CONTEXT_BYTES)),
        {"schema", "modules", "bootstrap_member", "bootstrap_sha256", "scope"}, "live_source_context_shape")
    require(context["schema"] == "omni-strata-execution-live-source-context-v1"
            and context["scope"] == "reviewed_stage_adapter_preimages_not_native_execution_proof", "live_source_context_scope")
    rows = context["modules"]
    require(type(rows) is list and len(rows) == len(MODULE_NAMES), "live_source_module_count")
    modules, evidence = {}, {}
    for row in rows:
        static_verifier.object_(row, {"role", "module", "member", "sha256"}, "live_source_module_record")
        role = row["role"]
        require(role in MODULE_NAMES and role not in modules and row["module"] == MODULE_NAMES[role], "live_source_module_role")
        expected = static_verifier.digest(row["sha256"])
        bundle.expected(row["member"], expected, read=False)
        archived = bundle.read(row["member"], MAX_SOURCE_BYTES)
        module = importlib.import_module(row["module"])
        actual = source_bytes(module.__file__)
        require(actual == archived, "loaded_adapter_source_preimage_differs")
        modules[role] = module
        evidence[role] = {"member": row["member"], "sha256": expected, "size_bytes": len(actual),
                          "loaded_source_path_sha256": digest(str(Path(module.__file__).resolve(strict=True)).encode("utf-8"))}
        del archived, actual
    require(evidence["parser"]["sha256"] == PARSER_SHA
            and evidence["bridge"]["sha256"] == BRIDGE_SHA
            and evidence["static"]["sha256"] == STATIC_VERIFIER_SHA
            and evidence["io"]["sha256"] == ENGINE_ROLE_IO_SHA, "reviewed_adapter_source_pin")
    require(modules["parser"] is parser and modules["static"] is static_verifier
            and modules["live"].LiveExecutionBindingVerifier is LiveExecutionBindingVerifier, "loaded_adapter_module_object")
    bundle.expected(context["bootstrap_member"], static_verifier.digest(context["bootstrap_sha256"]), read=False)
    bootstrap = bundle.read(context["bootstrap_member"], MAX_SOURCE_BYTES)
    require(len(bootstrap) <= MAX_SOURCE_BYTES, "bootstrap_source_bound")
    return modules, evidence, bootstrap


def _recursive_engine_closure(static):
    """Adapt verified PE bytes to the existing auditor's v1 closure contract."""
    pe = static["pe"]
    preimage = {key: value for key, value in pe.items() if key != "identity_sha256"}
    require(pe.get("schema") == "omni-strata-static-PE-closure-v2"
            and pe.get("identity_sha256") == digest(canonical(preimage))
            and pe.get("all_dynamic_loads_covered") is False
            and pe.get("current_system_dependencies_verified") is False, "static_PE_preimage")
    files = pe["files"]
    require(type(files) is list and 0 < len(files) <= 128, "engine_PE_file_count")
    by_name = {}
    for row in files:
        name = PurePosixPath(row["path"]).name.casefold()
        require(name not in by_name, "engine_PE_basename_alias")
        by_name[name] = row
    engine = next((row for row in files if row["sha256"] == static["native_executable_sha256"]
                   and PurePosixPath(row["path"]).name.casefold() == "strata.exe"), None)
    require(engine is not None, "engine_PE_executable_missing")
    required, pending = set(), [PurePosixPath(engine["path"]).name.casefold()]
    while pending:
        name = pending.pop()
        if name in required:
            continue
        require(name in by_name, "engine_PE_normal_dependency_missing")
        required.add(name)
        node = by_name[name]
        pending.extend(library.casefold() for library in node["normal"] if library.casefold() in by_name)
    closure = {"schema": "omni-strata-pe-closure-v1",
        "scope": "verified_static_engine_recursive_normal_and_delay_imports_not_future_dynamic_loads",
        "files": copy.deepcopy(files), "required_at_load": sorted(required),
        "all_dynamic_loads_covered": False, "system_dependencies_pinned": False}
    closure["identity_sha256"] = digest(canonical(closure))
    runtime = {"native_executable_sha256": static["native_executable_sha256"],
        "native_dependency_files": [{key: row[key] for key in ("path", "size_bytes", "sha256")}
                                    for row in files if row is not engine]}
    return engine["path"], runtime, closure


class LiveExecutionBindingVerifier:
    """One actual native generation; immutable binding after the first frame.

    Construction must occur inside the Stage integration after actual load.
    No arbitrary owner/module/GPU callbacks are accepted. The source-only bridge
    integration must install stage._execution_bridge before attach_bridge().
    """

    def __init__(self, stage, descriptor_file, runtime_root, source_context_file, receipt_directory, *, verified_snapshot=None):
        require(os.name == "nt", "native_Windows_live_binding_required")
        require(digest(source_bytes(static_verifier.__file__)) == STATIC_VERIFIER_SHA
                and digest(source_bytes(parser.__file__)) == PARSER_SHA, "live_verifier_import_preimages")
        self._root = Path(runtime_root).resolve(strict=True)
        self._descriptor_file = static_verifier.relative(descriptor_file)
        self._directory = Path(receipt_directory).resolve(strict=True)
        require(self._directory.is_dir() and not self._directory.is_relative_to(self._root)
                and not self._directory.is_symlink() and not list(self._directory.iterdir()), "fresh_external_live_receipt_directory")
        self._lock = threading.RLock()
        self._clock = live_clock_identity()
        self._failed, self._closed, self._bridge = False, False, None
        self._runtime, self._first_frame = None, None
        self._receipt_count, self._checks, self._last_ns = 0, 0, 0
        self._verified_snapshot = verified_snapshot
        self._bundle = None
        if verified_snapshot is None:
            self._static = copy.deepcopy(static_verifier.verify_combined_runtime(descriptor_file, self._root))
        else:
            require(type(verified_snapshot) is static_verifier._VerifiedCombinedRuntimeSnapshot
                    and stage._verified_execution_snapshot is verified_snapshot,
                    "actual_Stage_owned_verification_snapshot_required")
            self._static, self._bundle = verified_snapshot.claim(
                stage, descriptor_file, self._root, source_context_file)
        self._static_key = digest(canonical(self._static))
        require(self.static["runtime_binding"] is None and self.static["compiled_engine_ABI_verified"] is False
                and self.static["observer_layout_scope"] == "compiled_standalone_fixture_reference_only", "no_static_engine_ABI_authority")
        descriptor_raw = source_bytes(static_verifier.contained_file(self._root, self._descriptor_file))
        require(digest(descriptor_raw) == self.static["descriptor_sha256"], "live_descriptor_reread_changed")
        descriptor = static_verifier.json_(descriptor_raw)
        del descriptor_raw
        if verified_snapshot is None:
            self._bundle = _reopen_static_bundle(self._root, descriptor, self.static)
        self._modules, self._sources, self._bootstrap = _module_preimages(self._bundle, source_context_file)
        engine_file, self._module_runtime, self._closure = _recursive_engine_closure(self.static)
        self._engine_file = engine_file
        self._stage_module, self._io_module = self._modules["stage"], self._modules["io"]
        require(type(stage) is self._stage_module.StrataTextStageClient
                and type(stage._diagnostics) is self._stage_module._DiagnosticLog,
                "actual_text_Stage_and_owned_diagnostics_required")
        self._stage = stage
        require(isinstance(stage._proc, subprocess.Popen) and stage._proc.poll() is None, "actual_retained_supervisor_required")
        self._supervisor = stage._proc
        self._supervisor_birth = self._stage_module._windows_process_creation_filetime(stage._proc.pid)
        require(type(self._supervisor_birth) is int and self._supervisor_birth > 0, "supervisor_birth_unavailable")
        self._generation, self._stage_id = stage._generation, stage.stage_id
        self._physical_gpu = stage._physical_gpu
        require(type(self._physical_gpu) is int and self._physical_gpu >= 0, "actual_physical_GPU_index_required")
        require(stage._load_configuration.get("status") == "verified"
                and stage.execution_plan.get("worker_generation") == self._generation
                and stage.execution_plan.get("stage_id") == self._stage_id
                and stage.execution_plan.get("verified_execution_configuration") == "cpu+cuda:" + str(self._physical_gpu),
                "actual_Stage_loaded_configuration_required")
        self._plan_hash = digest(canonical(stage.execution_plan))
        self._verify_bootstrap()
        self._owner = self._fresh_owner()
        self._owner_key = digest(canonical(self._owner))
        self._load_audit = self._audit_modules(self.owner)
        self._load_receipt = self._write("load-modules.json", {
            "schema": "omni-strata-execution-owned-load-verification-v1", "owner": self.owner,
            "static_identity_sha256": self.static["identity_sha256"], "source_preimages": self._sources,
            "stage_plan_sha256": self._plan_hash,
            "stage_typed_runtime_manifest_sha256": stage.execution_plan["runtime_manifest_sha256"],
            "observer_member_manifest_sha256": self.static["runtime_manifest_sha256"], "module_audit": self._load_audit,
            "verification_clock": copy.deepcopy(self._clock),
            "scope": "actual_owned_native_load_snapshot_not_future_dynamic_module_attestation",
            "all_future_dynamic_loads_covered": False, "runtime_binding": None})

    @property
    def static(self):
        return copy.deepcopy(self._static)

    @property
    def owner(self):
        return copy.deepcopy(self._owner)

    @property
    def load_verification(self):
        """Detached actual load audit/receipt for the owning Stage plan only."""
        with self._lock:
            if self._closed or self._failed:
                return None
            return {"module_audit": copy.deepcopy(self._load_audit),
                    "receipt": copy.deepcopy(self._load_receipt), "source_preimages": copy.deepcopy(self._sources),
                    "scope": "actual_owned_native_load_snapshot_not_future_dynamic_module_attestation"}

    def verify_static_identity(self, descriptor_file, runtime_root):
        """Bridge constructor callback; reuse this exact already checked input."""
        with self._lock:
            require((self._verified_snapshot is None
                     or self._verified_snapshot._valid
                     and self._stage._verified_execution_snapshot is self._verified_snapshot)
                    and not self._failed and not self._closed
                    and static_verifier.relative(descriptor_file) == self._descriptor_file
                    and Path(runtime_root).resolve(strict=True) == self._root
                    and digest(canonical(self._static)) == self._static_key,
                    "bridge_static_verifier_input_or_identity_changed")
            return self.static

    def _write(self, name, value):
        require(self._receipt_count < MAX_RECEIPTS and re.fullmatch(r"[a-z0-9-]+\.json", name), "live_receipt_count_or_name")
        raw = canonical(value) + b"\n"
        require(len(raw) <= MAX_RECEIPT_BYTES, "live_receipt_encoded_bound")
        path = self._directory / name
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self._receipt_count += 1
        return {"file": name, "size_bytes": len(raw), "sha256": digest(raw)}

    def _verify_bootstrap(self):
        stage = self._stage
        require(stage._bootstrap_source().encode("utf-8") == self._bootstrap, "actual_Stage_bootstrap_preimage_differs")
        path = Path(stage._temporary.name).resolve(strict=True) / "bootstrap.py"
        raw = source_bytes(path)
        require(raw == self._bootstrap or b"\0" not in raw and raw.replace(b"\r\n", b"\n") == self._bootstrap,
                "actual_supervisor_bootstrap_file_differs")
        require(type(self._supervisor.args) is list and str(path) in self._supervisor.args
                and str(self._root / self._engine_file) in self._supervisor.args, "actual_supervisor_command_binding")
        require(stage.execution_plan.get("supervisor_bootstrap_sha256") == digest(self._bootstrap), "loaded_Stage_bootstrap_digest")

    def _fresh_owner(self):
        stage = self._stage
        require(not self._closed and not self._failed and not stage._closed and stage._proc is self._supervisor
                and self._supervisor.poll() is None and stage._generation == self._generation and stage.stage_id == self._stage_id,
                "owned_Stage_generation_retired_or_changed")
        require(self._stage_module._windows_process_creation_filetime(self._supervisor.pid) == self._supervisor_birth,
                "owned_supervisor_birth_changed")
        actual = stage._diagnostics.owned_native_identity()
        require(type(actual) is dict and set(actual) == {"pid", "creation_filetime_100ns"}
                and type(actual["pid"]) is int and actual["pid"] != self._supervisor.pid
                and self._stage_module._windows_process_creation_filetime(actual["pid"]) == actual["creation_filetime_100ns"],
                "owned_native_birth_unavailable_or_changed")
        fresh = self._stage_module._probe_memory(self._physical_gpu)
        require(fresh.get("gpu_source") == "NVML exact bytes"
                and all(type(fresh.get(key)) is str and fresh[key] for key in ("gpu_uuid", "gpu_pci_bus_id", "gpu_name_sha256")),
                "fresh_physical_NVML_GPU_identity_unavailable")
        owner = parser._owner({"schema": parser.OWNER_SCHEMA, "worker_generation": self._generation,
            "pid": actual["pid"], "creation_filetime_100ns": actual["creation_filetime_100ns"], "stage_id": self._stage_id,
            "gpu": {"uuid": fresh["gpu_uuid"], "pci_bus_id": fresh["gpu_pci_bus_id"], "name_sha256": fresh["gpu_name_sha256"]}})
        loaded_gpu = stage.execution_plan["fresh_memory_admission"]
        require(owner["gpu"] == {"uuid": loaded_gpu["gpu_uuid"], "pci_bus_id": loaded_gpu["gpu_pci_bus_id"],
                                 "name_sha256": loaded_gpu["gpu_name_sha256"]}, "fresh_GPU_differs_from_loaded_stage")
        require(self._stage_module._windows_process_creation_filetime(owner["pid"]) == owner["creation_filetime_100ns"],
                "owned_native_changed_during_GPU_query")
        if hasattr(self, "_owner"):
            require(digest(canonical(self._owner)) == self._owner_key
                    and canonical(owner) == canonical(self._owner), "native_owner_or_GPU_changed")
        return owner

    def _audit_modules(self, owner):
        audit = self._io_module.audit_selected_loaded_modules(owner, self._root, self._module_runtime,
            native_file=self._engine_file, include_cuda_driver=True, role="engine", non_system_pe_closure=self._closure)
        require(audit.get("status") == "verified" and audit.get("role") == "engine"
                and audit.get("observed_non_system_module_closure_verified") is True
                and audit.get("pe_closure_identity_sha256") == self._closure["identity_sha256"]
                and audit.get("pid") == owner["pid"]
                and audit.get("creation_filetime_100ns") == owner["creation_filetime_100ns"]
                and audit.get("all_future_dynamic_loads_covered") is False, "actual_engine_module_audit_unverified")
        require(canonical(self._fresh_owner()) == canonical(owner), "owner_changed_during_module_audit")
        return copy.deepcopy(audit)

    def attach_bridge(self, bridge):
        """Trusted Stage callsite, before begin/dispatch; does not authorize data."""
        with self._lock:
            require(self._bridge is None and type(bridge) is self._modules["bridge"].ParentExecutionBridge
                    and bridge.verify_live_binding is self and self._stage._execution_bridge is bridge,
                    "actual_Stage_owned_bridge_required")
            require(canonical(bridge.static) == canonical(self.static) and canonical(bridge.owner) == canonical(self.owner),
                    "bridge_static_or_owner_preimage_differs")
            self._bridge = bridge

    def __call__(self, static_identity, expected_owner, challenge, first_snapshot):
        with self._lock:
            try:
                require((self._verified_snapshot is None
                     or self._verified_snapshot._valid
                     and self._stage._verified_execution_snapshot is self._verified_snapshot)
                    and not self._failed and not self._closed and self._bridge is not None, "live_binding_retired_or_unattached")
                require(digest(canonical(self._static)) == self._static_key, "retained_static_identity_mutated")
                require(type(challenge) is str and re.fullmatch(r"[0-9a-f]{32}", challenge), "fresh_bridge_challenge_shape")
                require(canonical(static_identity) == canonical(self.static)
                        and canonical(parser._owner(expected_owner)) == canonical(self.owner), "live_call_static_or_owner_changed")
                require(self._stage._execution_bridge is self._bridge and self._bridge._lock._is_owned()
                        and self._bridge._active is not None, "actual_owned_reader_callsite_required")
                request = self._stage._io_request
                require(request is not None and request.request_id == self._bridge._active["request_id"]
                        and request.epoch == self._bridge._active["epoch"] and request.stage_id == self._stage_id
                        and request.worker_generation == self._generation, "actual_Stage_request_binding")
                owner = self._fresh_owner()
                if self._runtime is None and first_snapshot is not None:
                    snapshot = parser.validate_snapshot(first_snapshot)
                    pending = self._bridge._pending_dispatch
                    require(pending is not None and snapshot["snapshot_seq"] == 0
                            and snapshot["native_request_seq"] == pending["native_request_seq"], "first_owned_engine_snapshot_order")
                    require(canonical(snapshot["observer"]) == canonical(self.static["observer_layout"]), "actual_engine_ABI_reference_mismatch")
                    self._establish(snapshot, owner, challenge)
                elif self._runtime is not None:
                    require(first_snapshot is None, "duplicate_engine_ABI_establishment")
                stamp = time.perf_counter_ns()
                require(stamp > self._last_ns, "live_verifier_monotonic_clock_unavailable")
                self._last_ns, self._checks = stamp, min(self._checks + 1, (1 << 64) - 1)
                return {"runtime_binding": copy.deepcopy(self._runtime), "owner_binding": copy.deepcopy(owner),
                        "challenge": challenge, "monotonic_ns": stamp}
            except BaseException as error:
                self._failed = True
                if self._verified_snapshot is not None:
                    self._verified_snapshot.invalidate()
                self._runtime = None
                code = str(error) if isinstance(error, LiveBindingError) else "live_verification_failed"
                try:
                    self._write("failure.json", {"schema": "omni-strata-execution-live-binding-failure-v1",
                        "reason_code": code, "failure_type": type(error).__name__, "runtime_binding": None,
                        "owner": copy.deepcopy(self.owner), "checks": self._checks,
                        "verification_clock": copy.deepcopy(self._clock), "runtime_qualification": False})
                except BaseException:
                    pass  # Original refusal is never replaced by a disk failure.
                raise

    def _establish(self, snapshot, owner, challenge):
        # Full native bytes are hashed at load and again at the first snapshot,
        # never on every frame. This has no future dynamic-load attestation.
        modules = self._audit_modules(owner)
        module_receipt = self._write("first-frame-modules.json", {
            "schema": "omni-strata-execution-first-frame-module-verification-v1", "owner": owner,
            "module_audit": modules, "load_module_receipt": self._load_receipt,
            "scope": "actual_first_owned_frame_snapshot_only", "all_future_dynamic_loads_covered": False})
        frame_receipt = self._write("first-engine-frame.json", {
            "schema": "omni-strata-execution-owned-engine-ABI-frame-v1", "owner": owner, "challenge": challenge,
            "actual_snapshot": snapshot, "actual_snapshot_sha256": digest(canonical(snapshot)),
            "fixture_reference_receipt_sha256": self.static["ABI_reference"]["fixture_receipt_sha256"],
            "static_identity_sha256": self.static["identity_sha256"], "first_frame_module_receipt": module_receipt,
            "compiled_engine_geometry_matches_reference": True,
            "authentication_scope": "Stage_owned_bridge_authenticated_reader_callback_not_native_JSON_self_authentication",
            "model_execution_completed": False, "runtime_qualification": False})
        verification = self._write("binding-verification.json", {
            "schema": "omni-strata-execution-generated-live-verification-v1", "owner": owner,
            "source_preimages": self._sources, "bootstrap_sha256": digest(self._bootstrap),
            "verification_clock": copy.deepcopy(self._clock),
            "static_identity_sha256": self.static["identity_sha256"], "runtime_manifest_sha256": self.static["runtime_manifest_sha256"],
            "native_executable_sha256": self.static["native_executable_sha256"], "first_engine_frame_receipt": frame_receipt,
            "loaded_module_receipt": module_receipt, "load_module_receipt": self._load_receipt,
            "scope": "owned_first_frame_ABI_and_at_load_first_frame_module_snapshots",
            "all_future_dynamic_loads_covered": False, "runtime_qualification": False, "default_eligible": False,
            "physical_ssd_read_bytes": None, "aggregate_memory_hard_caps_verified": False})
        binding = {"schema": parser.RUNTIME_SCHEMA,
            "authorization": "externally_reviewed_combined_runtime_not_established_by_parser",
            "native_schema": parser.NATIVE_SCHEMA, "header_sha256": parser.HEADER_SHA256,
            "header_v1_sha256": parser.HEADER_V1_SHA256, "schema_sha256": parser.SCHEMA_SHA256,
            "incremental_patch_sha256": parser.INCREMENTAL_PATCH_SHA256, "boundary_patch_sha256": parser.BOUNDARY_PATCH_SHA256,
            "base_io_patch_sha256": parser.BASE_IO_PATCH_SHA256, "combined_patch_manifest_sha256": parser.COMBINED_PATCH_MANIFEST_SHA256,
            "runtime_manifest_sha256": self.static["runtime_manifest_sha256"], "native_executable_sha256": self.static["native_executable_sha256"],
            "owner_adapter_sha256": self._sources["bridge"]["sha256"], "loaded_module_receipt_sha256": module_receipt["sha256"],
            "verification_receipt_sha256": verification["sha256"], "observer_layout": copy.deepcopy(snapshot["observer"])}
        self._runtime = parser._runtime(binding)
        self._first_frame = frame_receipt

    def close(self):
        """Retire this verifier only; the Stage remains responsible for OS drain."""
        with self._lock:
            self._closed = True
            if self._verified_snapshot is not None:
                self._verified_snapshot.invalidate()
                self._verified_snapshot = None
            self._runtime = None
            self._bridge = None
            self._bundle = None
            self._stage = None
            self._supervisor = None
            self._bootstrap = b""
            self._load_audit = None
            self._modules = {}


def create_live_binding_verifier(stage, descriptor_file, runtime_root, source_context_file, receipt_directory, *,
                                 verified_snapshot=None):
    """Future trusted Stage factory; a failed preparation gets its own receipt."""
    root = Path(runtime_root).resolve(strict=True)
    directory = Path(receipt_directory).absolute()
    require(directory.parent.is_dir() and not directory.exists()
            and not directory.resolve(strict=False).is_relative_to(root), "fresh_external_live_receipt_directory")
    directory.mkdir()
    try:
        return LiveExecutionBindingVerifier(stage, descriptor_file, root, source_context_file, directory,
                                            verified_snapshot=verified_snapshot)
    except BaseException as error:
        if type(verified_snapshot) is static_verifier._VerifiedCombinedRuntimeSnapshot:
            verified_snapshot.invalidate()
        code = str(error) if isinstance(error, LiveBindingError) else "live_binding_preparation_unavailable"
        failure = {"schema": "omni-strata-execution-live-binding-preparation-failure-v1", "reason_code": code,
                   "failure_type": type(error).__name__, "runtime_binding": None, "compiled_engine_ABI_verified": False,
                   "verifier_source_sha256": digest(source_bytes(__file__)), "runtime_qualification": False}
        try:
            raw = canonical(failure) + b"\n"
            require(len(raw) <= MAX_RECEIPT_BYTES, "live_failure_receipt_bound")
            with (directory / "preparation-failure.json").open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            pass
        raise

