"""Private, unexecuted owned-channel integration prototype.

Runtime byte verification, OS/module/GPU verification and resource retirement
are injected external prerequisites, not implemented or inferred by this code.
Install only in a future reviewed bootstrap/Stage patch; production is unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import secrets
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import strata_exec as parser

PRODUCTION_STRATA_SHA256 = "8d80e7e7dc43f0f6c0f3b3198e0a295ab2c235996b619fd2ddfab4c99da7d015"
PRODUCTION_IO_SHA256 = "60ca72b35b7f2722c4157c1619fa72363855d925f2f5b98e362d4ea512ad686e"
PARSER_SHA256 = "c653d6eb422ed1f5ca970d5db1d0d3c8d08139a1e085eabd171497490a7663bf"
CONTROL_PREFIX = "strata supervisor: native exec "
CONTROL_NAMESPACE = b"strata supervisor: native exec"
BOOTSTRAP_SCHEMA = "omni-private-strata-bootstrap-exec-v1"
STATIC_SCHEMA = "omni-strata-combined-static-runtime-identity-v2"
BRIDGE_REPORT_SCHEMA = "omni-private-strata-owned-execution-report-v1"
ORDINARY_BYTES = 4096
CONTROL_BYTES = 36864  # Whole authenticated control line, including terminator.
STATIC_IDENTITY_BYTES = 65536
BOOTSTRAP_KEYS = {
    "OMNI_STRATA_EXEC_TOKEN", "OMNI_STRATA_EXEC_GENERATION", "OMNI_STRATA_EXEC_ADAPTER",
    "OMNI_STRATA_EXEC_ADAPTER_SHA256", "OMNI_STRATA_EXEC_PARSER", "OMNI_STRATA_EXEC_PARSER_SHA256",
}
WRITER_ERRORS = {
    "unexpected_native_command_framing", "unsupported_native_batch_dispatch", "overlapping_native_dispatch",
    "native_snapshot_without_dispatch", "invalid_native_execution_snapshot", "unknown_native_execution_version",
    "native_request_error", "invalid_native_terminal", "native_terminal_without_dispatch",
    "native_write_failed", "native_stdout_eof", "owner_channel_emit_failed",
}


def _source_sha(path: str | Path) -> str:
    data = Path(path).read_bytes()
    if not 0 < len(data) <= 131072:
        raise parser.ObservationError("adapter_source_bound")
    return hashlib.sha256(data).hexdigest()


def validate_adapter_sources() -> None:
    """Verify actual import preimages; a future launcher must pin this module too."""
    if _source_sha(parser.__file__) != PARSER_SHA256:
        raise parser.ObservationError("parser_import_source_changed")


def take_bootstrap_configuration(environment: dict[str, str]) -> dict[str, str] | None:
    """Remove every private exec value BEFORE any native Popen, including failure."""
    values = {name: environment.pop(name, "") for name in BOOTSTRAP_KEYS}
    # Refuse unknown exec context while still removing it from native inheritance.
    extra = [name for name in environment if name.startswith("OMNI_STRATA_EXEC_")]
    for name in extra:
        environment.pop(name)
    if extra:
        raise parser.ObservationError("unknown_bootstrap_exec_environment")
    if not any(values.values()):
        return None
    if not all(type(value) is str and value for value in values.values()):
        raise parser.ObservationError("partial_bootstrap_exec_configuration")
    parser._hash(values["OMNI_STRATA_EXEC_TOKEN"])
    parser._identifier(values["OMNI_STRATA_EXEC_GENERATION"])
    parser._hash(values["OMNI_STRATA_EXEC_ADAPTER_SHA256"])
    if values["OMNI_STRATA_EXEC_PARSER_SHA256"] != PARSER_SHA256:
        raise parser.ObservationError("bootstrap_parser_identity")
    # This checks bytes only. It cannot make an arbitrary runtime/receipt eligible.
    if _source_sha(values["OMNI_STRATA_EXEC_ADAPTER"]) != values["OMNI_STRATA_EXEC_ADAPTER_SHA256"]:
        raise parser.ObservationError("bootstrap_adapter_identity")
    if _source_sha(values["OMNI_STRATA_EXEC_PARSER"]) != PARSER_SHA256:
        raise parser.ObservationError("bootstrap_parser_identity")
    return values


def prepare_native_kwargs(kwargs: Mapping[str, Any], inherited_environment: Mapping[str, str], *, opt_in: bool) -> dict:
    """Call at the existing original(args,*a,**kw) boundary, never after launch."""
    if type(opt_in) is not bool:
        raise parser.ObservationError("native_observation_opt_in_type")
    result = dict(kwargs)
    source = result.get("env")
    environment = dict(inherited_environment if source is None else source)
    for name in tuple(environment):
        if name.startswith("OMNI_STRATA_EXEC_"):
            environment.pop(name)
    # The switch requests native records; it is not trusted runtime provenance.
    environment.pop("STRATA_OMNI_EXEC_OBSERVER", None)
    if opt_in:
        environment["STRATA_OMNI_EXEC_OBSERVER"] = "1"
    result["env"] = environment
    return result


class SerializedDiagnosticSink:
    """One shared sink for existing IO, ordinary diagnostics and new exec writes.

    Replace only the bootstrap's sys.stderr proxy before native launch, retaining
    its original underlying stream. No content/history is retained. Existing
    ordinary/IO writers must use this same sink to avoid interleaving long rows.
    """

    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.RLock()

    def write(self, text):
        with self.lock:
            return self.stream.write(text)

    def flush(self):
        with self.lock:
            return self.stream.flush()

    def emit_control(self, row):
        with self.lock:
            self.stream.write(row)
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class OwnedExecutionFrameWriter:
    """Bootstrap-only metadata writer around the actual owned native pipes."""

    def __init__(self, token: str, generation: str, pid: int, birth: int, emit: Callable[[str], None]):
        self.token = parser._hash(token)
        self.generation = parser._identifier(generation)
        self.pid = parser._uint(pid, positive=True, maximum=(1 << 32) - 1)
        self.birth = parser._uint(birth, positive=True)
        self.emit = emit
        self.sequence = 0
        self.active = False
        self.snapshot_count = 0
        self.failed = False
        self.lock = threading.RLock()

    def _emit(self, kind: str, payload: Any) -> None:
        frame = {"schema": BOOTSTRAP_SCHEMA, "generation": self.generation, "pid": self.pid,
                 "creation_filetime_100ns": self.birth, "dispatch_seq": self.sequence,
                 "kind": kind, "payload": payload}
        row = CONTROL_PREFIX + self.token + " " + parser._encoded(frame).decode("ascii") + "\n"
        if len(row.encode("ascii")) > CONTROL_BYTES:
            self.failed = True
            raise parser.ObservationError("owned_writer_frame_bound")
        try:
            self.emit(row)
        except BaseException:
            self.failed = True
            raise

    def error(self, code: str) -> None:
        with self.lock:
            if code not in WRITER_ERRORS:
                raise parser.ObservationError("unknown_writer_error")
            if self.failed:
                return
            self.failed = True
            self._emit("error", {"code": code})

    def observe_input(self, row: bytes | str) -> None:
        # Inspect framing/prefix only, never split/copy/retain the raw GEN body.
        with self.lock:
            if self.failed:
                return
            if type(row) not in (bytes, str):
                self.error("unexpected_native_command_framing")
                return
            lf = b"\n" if type(row) is bytes else "\n"
            names = ((b"GEN ", b"GENI ", b"BGEN") if type(row) is bytes else ("GEN ", "GENI ", "BGEN"))
            if row.startswith(names[2]):
                self.error("unsupported_native_batch_dispatch")
                return
            if not row.startswith(names[:2]):
                return
            if not row.endswith(lf) or row.count(lf) != 1:
                self.error("unexpected_native_command_framing")
                return
            if self.active:
                self.error("overlapping_native_dispatch")
                return
            if self.sequence >= parser.NATIVE_SEQUENCE_MAX:
                self.error("overlapping_native_dispatch")
                return
            command = "GENI" if row.startswith(names[1]) else "GEN"
            # No token IDs, prompt slices or tensor paths enter this frame.
            self.sequence += 1
            self.active = True
            self.snapshot_count = 0
            self._emit("dispatch", {"command": command})

    def observe_output(self, row: bytes | str) -> None:
        with self.lock:
            if self.failed:
                return
            if type(row) not in (bytes, str):
                self.error("invalid_native_execution_snapshot")
                return
            is_bytes = type(row) is bytes
            prefix = parser.PREFIX.encode("ascii") if is_bytes else parser.PREFIX
            namespace = b"OMNI_EXEC_" if is_bytes else "OMNI_EXEC_"
            done = b"DONE " if is_bytes else "DONE "
            error = b"ERR" if is_bytes else "ERR"
            if row.startswith(namespace):
                if not row.startswith(prefix):
                    self.error("unknown_native_execution_version")
                    return
                if not self.active:
                    self.error("native_snapshot_without_dispatch")
                    return
                try:
                    snapshot = parser.parse_native_line(row)
                    if snapshot["native_request_seq"] != self.sequence or snapshot["snapshot_seq"] != self.snapshot_count:
                        raise parser.ObservationError("bootstrap_native_sequence")
                    self._emit("snapshot", snapshot)
                    self.snapshot_count += 1
                except (ValueError, TypeError, OverflowError):
                    self.error("invalid_native_execution_snapshot")
            elif row.startswith(error):
                self.error("native_request_error")
            elif row.startswith(done):
                if not self.active:
                    self.error("native_terminal_without_dispatch")
                    return
                if len(row) > ORDINARY_BYTES:
                    self.error("invalid_native_terminal")
                    return
                try:
                    text = row.decode("ascii", "strict") if is_bytes else row
                    fields = text.split()
                    if len(fields) != 16 or fields[5] not in ("stop", "length", "cancel"):
                        raise ValueError
                    finish = "cancelled" if fields[5] == "cancel" else fields[5]
                    self._emit("native_done", {"finish_reason": finish})
                    self.active = False
                except (UnicodeError, ValueError, TypeError):
                    self.error("invalid_native_terminal")


class ExecutionInput:
    def __init__(self, pipe, writer: OwnedExecutionFrameWriter):
        self.pipe, self.writer = pipe, writer

    def write(self, row):
        try:
            self.writer.observe_input(row)
        except Exception:
            self.writer.failed = True
        try:
            return self.pipe.write(row)
        except BaseException:
            try:
                self.writer.error("native_write_failed")
            except Exception:
                self.writer.failed = True
            raise

    def __getattr__(self, name):
        return getattr(self.pipe, name)


class ExecutionOutput:
    def __init__(self, pipe, writer: OwnedExecutionFrameWriter):
        self.pipe, self.writer = pipe, writer

    def readline(self, *args):
        row = self.pipe.readline(*args)
        try:
            if row:
                self.writer.observe_output(row)
            elif self.writer.active:
                self.writer.error("native_stdout_eof")
        except Exception:
            self.writer.failed = True
        return row  # Exact original row/object goes to upstream server unchanged.

    def __iter__(self):
        return self

    def __next__(self):
        row = self.readline()
        if not row:
            raise StopIteration
        return row

    def __getattr__(self, name):
        return getattr(self.pipe, name)


def attach_execution_pipes(process, writer: OwnedExecutionFrameWriter) -> None:
    """After existing containment/GetProcessTimes/IO wrapping, before server use."""
    if process.stdin is None or process.stdout is None:
        raise parser.ObservationError("owned_native_pipes_required")
    process.stdin = ExecutionInput(process.stdin, writer)
    process.stdout = ExecutionOutput(process.stdout, writer)


class AuthenticatedControlLineReader:
    """Bounded replacement read loop; does not itself spawn a thread or log data."""

    def __init__(self, token: str, *, on_exec: Callable[[dict], None], on_ordinary: Callable[[bytes], None],
                 on_failure: Callable[[str], None]):
        self.token = parser._hash(token)
        self.prefix = (CONTROL_PREFIX + self.token + " ").encode("ascii")
        self.on_exec, self.on_ordinary, self.on_failure = on_exec, on_ordinary, on_failure
        self.failed = False
        self.complete_lines = 0
        self.errors = 0
        self.failure_callback_failed = False

    def _failure(self, code: str) -> None:
        self.failed = True
        self.errors = min(self.errors + 1, parser.U64_MAX)
        try:
            self.on_failure(code)  # Content-free; also notify independent IO path.
        except Exception:
            # Retain only a flag and keep draining the shared stderr pipe.
            self.failure_callback_failed = True

    def _discard_to_newline(self, pipe, initial: bytes) -> None:
        chunk = initial
        while chunk and not chunk.endswith(b"\n"):
            chunk = pipe.readline(ORDINARY_BYTES + 1)
            if type(chunk) is not bytes:
                raise parser.ObservationError("diagnostic_pipe_type")

    def drain(self, pipe) -> None:
        try:
            self._drain(pipe)
        except Exception:
            # Initial, continuation and discard reads share one terminal guard.
            # Never retain exception text or leave an apparently healthy reader.
            self._failure("diagnostic_read_failed")
            return

    def _drain(self, pipe) -> None:
        while True:
            first = pipe.readline(ORDINARY_BYTES + 1)
            if type(first) is not bytes:
                self._failure("diagnostic_pipe_type")
                return
            if not first:
                return
            authenticated = first.startswith(self.prefix)
            is_exec = first.startswith(CONTROL_NAMESPACE)
            if authenticated:
                row = first
                while not row.endswith(b"\n") and len(row) <= CONTROL_BYTES:
                    rest = pipe.readline(CONTROL_BYTES + 1 - len(row))
                    if type(rest) is not bytes:
                        self._failure("diagnostic_pipe_type")
                        return
                    if not rest:
                        break
                    row += rest
                if not row.endswith(b"\n") or len(row) > CONTROL_BYTES:
                    self._failure("oversize_or_partial_exec_control")
                    self._discard_to_newline(pipe, row)
                    continue
                try:
                    payload = parser._json(row[len(self.prefix):], CONTROL_BYTES)
                    self.on_exec(payload)
                except Exception:
                    self._failure("invalid_authenticated_exec_control")
                self.complete_lines = min(self.complete_lines + 1, parser.U64_MAX)
            elif is_exec:
                self._failure("unauthenticated_exec_control")
                self._discard_to_newline(pipe, first)
            elif len(first) > ORDINARY_BYTES or not first.endswith(b"\n"):
                self._failure("oversize_or_partial_ordinary_diagnostic")
                self._discard_to_newline(pipe, first)
            else:
                # Same <=4096 ordinary/IO bytes reach the existing sanitizer/IO
                # observer unchanged. No new prefix is added to its allowlist.
                try:
                    self.on_ordinary(first)
                except Exception:
                    self._failure("ordinary_diagnostic_callback_failed")
                    continue
                self.complete_lines = min(self.complete_lines + 1, parser.U64_MAX)


class ParentExecutionBridge:
    """Future owned-channel adapter; injected verifiers remain prerequisites.

    Static standalone fixture layout is only a reference. No parser binding is
    created until the first authenticated actual engine snapshot has matched that
    reference AND the independent live verifier has verified current native
    EXE/modules/PID/birth/GPU. At most one typed dispatch waits for that snapshot.
    """

    def __init__(self, descriptor_file: str, runtime_root: Path, *, expected_owner: dict, token: str,
                 verify_combined_runtime: Callable[[str, Path], dict],
                 verify_live_binding: Callable[[dict, dict, str, dict | None], dict]):
        validate_adapter_sources()
        if not callable(verify_combined_runtime) or not callable(verify_live_binding):
            raise TypeError("independent_static_and_live_verifiers_required")
        self.static = copy.deepcopy(verify_combined_runtime(descriptor_file, runtime_root))
        if type(self.static) is not dict or len(parser._encoded(self.static)) > STATIC_IDENTITY_BYTES:
            raise parser.ObservationError("combined_static_identity")
        parser._literal(self.static.get("schema"), STATIC_SCHEMA)
        identity = {key: value for key, value in self.static.items() if key != "identity_sha256"}
        parser._literal(self.static.get("identity_sha256"), parser._digest(identity))
        parser._literal(self.static.get("parser_sha256"), PARSER_SHA256)
        parser._literal(self.static.get("native_execution_schema"), parser.NATIVE_SCHEMA)
        parser._literal(self.static.get("runtime_binding"), None)
        parser._literal(self.static.get("observer_layout_scope"), "compiled_standalone_fixture_reference_only")
        for key in ("live_loaded_module_paths_verified", "current_OS_identity_verified", "owner_adapter_verified",
                    "runtime_qualification", "default_eligible", "compiled_engine_ABI_verified"):
            parser._literal(self.static.get(key), False)
        parser._observer(self.static.get("observer_layout"))
        self.owner = parser._owner(expected_owner)
        self.token = parser._hash(token)
        self.verify_live_binding = verify_live_binding
        self._lock = threading.RLock()
        self._active: dict | None = None
        self._last: dict | None = None
        self._pending_dispatch: dict | None = None
        self._last_epoch = 0
        self._retired = False
        self._errors: list[str] = []
        self._last_live_ns = 0
        self._native_done_received = False
        self._live_checks = 0
        self.runtime: dict | None = None
        self.observer: parser.StrataExecutionObserver | None = None
        # No live verification, engine ABI claim or parser construction here.

    def _verified_live(self, first_snapshot: dict | None) -> dict:
        challenge = secrets.token_hex(16)
        result = self.verify_live_binding(copy.deepcopy(self.static), copy.deepcopy(self.owner),
                                          challenge, copy.deepcopy(first_snapshot))
        result = parser._object(result, {"runtime_binding", "owner_binding", "challenge", "monotonic_ns"},
                                "live_binding_shape")
        parser._literal(result["challenge"], challenge, "fresh_owner_challenge")
        owner = parser._owner(result["owner_binding"])
        parser._literal(parser._encoded(owner), parser._encoded(self.owner), "fresh_owner_identity")
        stamp = parser._uint(result["monotonic_ns"], positive=True)
        if stamp <= self._last_live_ns:
            raise parser.ObservationError("stale_live_owner_check")
        # An actual verifier may establish a binding ONLY for the first owned
        # snapshot. A static receipt or dispatch cannot supply an engine ABI.
        if self.runtime is None and first_snapshot is None:
            parser._literal(result["runtime_binding"], None, "premature_engine_abi_binding")
        elif self.runtime is not None:
            parser._literal(parser._encoded(result["runtime_binding"]), parser._encoded(self.runtime),
                            "fresh_runtime_binding_changed")
        else:
            runtime = parser._runtime(result["runtime_binding"])
            for key in ("runtime_manifest_sha256", "native_executable_sha256", "combined_patch_manifest_sha256",
                        "header_sha256", "schema_sha256"):
                parser._literal(runtime[key], self.static.get(key), "static_live_runtime_identity")
            parser._literal(parser._encoded(runtime["observer_layout"]),
                            parser._encoded(self.static["observer_layout"]), "static_live_compiled_layout")
            parser._literal(parser._encoded(first_snapshot["observer"]),
                            parser._encoded(runtime["observer_layout"]), "actual_engine_layout_mismatch")
            parser._literal(runtime["owner_adapter_sha256"], _source_sha(__file__), "actual_owner_adapter_source")
            result = dict(result, runtime_binding=runtime)
        self._last_live_ns = stamp
        self._live_checks = min(self._live_checks + 1, parser.U64_MAX)
        return copy.deepcopy(result)

    def channel_failure(self, code: str) -> None:
        with self._lock:
            known = {
                "diagnostic_pipe_type", "oversize_or_partial_exec_control", "invalid_authenticated_exec_control",
                "unauthenticated_exec_control", "oversize_or_partial_ordinary_diagnostic", "live_owner_verification_failed",
                "invalid_owned_exec_frame", "owned_writer_error", "control_reader_failed", "control_stream_unjoined",
                "ordinary_diagnostic_callback_failed", "diagnostic_read_failed",
            }
            if code not in known:
                raise parser.ObservationError("unknown_channel_failure")
            self._retired = True
            if code not in self._errors and len(self._errors) < 8:
                self._errors.append(code)

    def control_reader(self, *, on_ordinary: Callable[[bytes], None],
                       on_independent_io_failure: Callable[[str], None]) -> AuthenticatedControlLineReader:
        """Wire both observation channels to failures; ordinary behavior stays external."""
        def failure(code: str) -> None:
            self.channel_failure(code)
            on_independent_io_failure(code)
        return AuthenticatedControlLineReader(
            self.token, on_exec=lambda frame: self.ingest(frame, channel_token=self.token),
            on_ordinary=on_ordinary, on_failure=failure)
    def begin(self, request_id: str, epoch: int) -> None:
        with self._lock:
            if self._retired:
                raise parser.ObservationError("retired_owner_bridge")
            parser._identifier(request_id)
            parser._uint(epoch, positive=True)
            if self._active is not None or epoch <= self._last_epoch:
                raise parser.ObservationError("bridge_request_epoch_or_overlap")
            if self.observer is not None:
                self.observer.begin(request_id, epoch)
            self._active = {"request_id": request_id, "epoch": epoch}
            self._last_epoch = epoch
            self._pending_dispatch = None
            self._native_done_received = False

    def _owned(self, kind: str, payload: dict) -> dict:
        assert self.observer is not None and self._active is not None
        return {"schema": parser.FRAME_SCHEMA, "channel_token": self.token, **self.observer.binding_identity,
                "worker_generation": self.owner["worker_generation"], "pid": self.owner["pid"],
                "creation_filetime_100ns": self.owner["creation_filetime_100ns"], "gpu_uuid": self.owner["gpu"]["uuid"],
                "stage_id": self.owner["stage_id"], **self._active, "kind": kind, "payload": payload}

    def ingest(self, frame: dict, *, channel_token: str) -> None:
        """Only the authenticated reader callback supplies this separate token."""
        with self._lock:
            if self._retired:
                return  # Continue draining elsewhere; never repair prior errors.
            try:
                if not hmac.compare_digest(parser._hash(channel_token), self.token):
                    raise parser.ObservationError("owned_channel")
            except (ValueError, TypeError):
                self.channel_failure("unauthenticated_exec_control")
                return
            try:
                if self._active is None:
                    raise parser.ObservationError("exec_frame_without_stage_request")
                frame = parser._object(frame, {"schema", "generation", "pid", "creation_filetime_100ns",
                                              "dispatch_seq", "kind", "payload"}, "bootstrap_frame_shape")
                if len(parser._encoded(frame)) > CONTROL_BYTES:
                    raise parser.ObservationError("bootstrap_frame_bound")
                parser._literal(frame["schema"], BOOTSTRAP_SCHEMA)
                for source, target in (("generation", "worker_generation"), ("pid", "pid"),
                                       ("creation_filetime_100ns", "creation_filetime_100ns")):
                    parser._literal(frame[source], self.owner[target], "bootstrap_owner_identity")
                seq = parser._uint(frame["dispatch_seq"], positive=True, maximum=parser.NATIVE_SEQUENCE_MAX)
                kind = frame["kind"]
                if type(kind) is not str or kind not in ("dispatch", "snapshot", "native_done", "error"):
                    raise parser.ObservationError("bootstrap_frame_kind")
                payload = frame["payload"]
                if kind == "error":
                    payload = parser._object(payload, {"code"}, "bootstrap_error_shape")
                    if type(payload["code"]) is not str or payload["code"] not in WRITER_ERRORS:
                        raise parser.ObservationError("bootstrap_error_code")
                elif kind == "dispatch":
                    payload = parser._object(payload, {"command"}, "bootstrap_dispatch_shape")
                    if type(payload["command"]) is not str or payload["command"] not in ("GEN", "GENI"):
                        raise parser.ObservationError("bootstrap_dispatch_command")
                    payload = {"command": payload["command"], "native_request_seq": seq}
                    if self.observer is None and self._pending_dispatch is not None:
                        raise parser.ObservationError("duplicate_pending_dispatch")
                elif kind == "native_done":
                    payload = parser._object(payload, {"finish_reason"}, "bootstrap_done_shape")
                    if type(payload["finish_reason"]) is not str or payload["finish_reason"] not in (
                            "stop", "length", "cancelled", "error"):
                        raise parser.ObservationError("bootstrap_done_finish")
                    payload = {"finish_reason": payload["finish_reason"], "native_request_seq": seq}
                    if self.observer is None:
                        raise parser.ObservationError("native_terminal_before_initial_snapshot")
                else:
                    payload = parser.validate_snapshot(payload)
                    parser._literal(payload["native_request_seq"], seq, "bootstrap_native_sequence")
                    if self.observer is None:
                        if self._pending_dispatch is None:
                            raise parser.ObservationError("initial_snapshot_without_dispatch")
                        parser._literal(payload["snapshot_seq"], 0, "missing_initial_engine_snapshot")
                        parser._literal(payload["native_request_seq"], self._pending_dispatch["native_request_seq"],
                                        "initial_snapshot_dispatch_sequence")
                        parser._literal(parser._encoded(payload["observer"]),
                                        parser._encoded(self.static["observer_layout"]),
                                        "actual_engine_fixture_layout_mismatch")
            except Exception:
                self.channel_failure("invalid_owned_exec_frame")
                return
            try:
                first_snapshot = payload if self.observer is None and kind == "snapshot" else None
                live = self._verified_live(first_snapshot)
            except Exception:
                self.channel_failure("live_owner_verification_failed")
                return
            try:
                if kind == "error":
                    self.channel_failure("owned_writer_error")
                    return
                if self.observer is None:
                    if kind == "dispatch":
                        # One small typed record only; no prompt/model tokens.
                        self._pending_dispatch = copy.deepcopy(payload)
                        return
                    # First actual owned snapshot plus independent fresh verifier
                    # creates the parser; replay the saved dispatch in order.
                    self.runtime = copy.deepcopy(live["runtime_binding"])
                    self.observer = parser.StrataExecutionObserver(self.runtime, self.owner, channel_token=self.token)
                    assert self._active is not None and self._pending_dispatch is not None
                    self.observer.begin(**self._active)
                    self.observer.ingest(self._owned("dispatch", self._pending_dispatch))
                    self._pending_dispatch = None
                self.observer.ingest(self._owned(kind, payload))
                if kind == "native_done":
                    self._native_done_received = True
            except Exception:
                self.channel_failure("invalid_owned_exec_frame")

    def ready_to_finish(self, request_id: str, epoch: int) -> bool:
        """Only a matching parser-accepted DONE or retired error ends the wait.

        This is synchronization metadata, never evidence of clean completion or
        authoritative process retirement. Missing DONE remains partial at finish.
        """
        with self._lock:
            if self._active is None:
                raise parser.ObservationError("bridge_no_active_request")
            parser._literal(request_id, self._active["request_id"], "bridge_finish_request")
            parser._literal(epoch, self._active["epoch"], "bridge_finish_request")
            return self._retired or self._native_done_received

    def finish(self, request_id: str, epoch: int, *, omni_completed: bool, cancellation_requested: bool,
               lifecycle_outcome: str, reader_healthy: bool, reader_joined: bool) -> dict:
        with self._lock:
            if self._active is None:
                raise parser.ObservationError("bridge_no_active_request")
            parser._literal(request_id, self._active["request_id"], "bridge_finish_request")
            parser._literal(epoch, self._active["epoch"], "bridge_finish_request")
            if type(omni_completed) is not bool or type(cancellation_requested) is not bool:
                raise parser.ObservationError("bridge_finish_flags")
            if type(lifecycle_outcome) is not str or lifecycle_outcome not in ("normal", "error", "cancelled", "drained"):
                raise parser.ObservationError("bridge_lifecycle_outcome")
            if type(reader_joined) is not bool or type(reader_healthy) is not bool:
                raise parser.ObservationError("bridge_reader_join_type")
            if not reader_healthy:
                self.channel_failure("control_reader_failed")
            if lifecycle_outcome != "normal" and not reader_joined:
                self.channel_failure("control_stream_unjoined")
            native = (self.observer.finish(request_id, epoch, omni_completed=omni_completed,
                                          cancellation_requested=cancellation_requested)
                      if self.observer is not None else None)
            complete = (native is not None and native["complete"] and not self._errors
                        and lifecycle_outcome in ("normal", "drained"))
            unavailable = native is None
            report = {
                "schema": BRIDGE_REPORT_SCHEMA,
                "status": "unavailable" if unavailable else ("complete_scoped_observation" if complete else "partial"),
                "complete": complete, "channel_errors": list(self._errors), "native_observation": native,
                "unavailable_reason": "no_verified_initial_owned_engine_snapshot" if unavailable else None,
                "request_binding": copy.deepcopy(self._active), "engine_abi_binding_established": not unavailable,
                "runtime_binding_sha256": parser._digest(self.runtime) if self.runtime is not None else None,
                "static_runtime_identity_sha256": self.static["identity_sha256"],
                "static_layout_scope": "compiled_standalone_fixture_reference_only",
                "verification_scope": "independent_injected_static_and_live_verifiers_required_not_implemented_here",
                "live_verifier_result_count": self._live_checks, "last_live_verifier_monotonic_ns": self._last_live_ns,
                "lifecycle_outcome": lifecycle_outcome, "reader_healthy": reader_healthy, "reader_joined": reader_joined,
                "omni_completed": omni_completed, "cancellation_requested": cancellation_requested,
                "pending_dispatch_count_at_finish": int(self._pending_dispatch is not None),
                "resource_retirement_verified_by_bridge": False, "omni_ledger_changed": False,
                "actual_whole_model_placement": None, "physical_ssd_read_bytes": None,
                "aggregate_gpu_hard_cap_verified": False, "aggregate_ram_hard_cap_verified": False,
                "runtime_qualification": False, "default_eligible": False,
                "channel_frame_bytes": CONTROL_BYTES, "ordinary_diagnostic_bytes": ORDINARY_BYTES,
                "observer_overhead_charged_to_Agent_consumer_workspace": False,
                "observer_total_heap_or_stack_bound_verified": False,
            }
            if len(parser._encoded(report)) > parser.REPORT_BYTES + 4096:
                raise parser.ObservationError("bridge_report_bound")
            self._last = copy.deepcopy(report)
            self._active = None
            self._pending_dispatch = None
            self._retired = self._retired or unavailable or (self.observer is not None and self.observer.retired)
            return copy.deepcopy(report)

    def last_observation(self) -> dict | None:
        with self._lock:
            return copy.deepcopy(self._last)
