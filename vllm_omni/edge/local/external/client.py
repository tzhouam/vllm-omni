# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The engine's half of an out-of-process stage.

Binds a listener, starts the worker, waits for it to connect *back*, and then
speaks :mod:`~vllm_omni.edge.local.external.protocol` over that socket. The
connect-back direction is what keeps this deployable: WSL reaching into Windows
needs an inbound firewall rule, while Windows reaching into WSL does not.

Two things here are load-bearing rather than defensive.

**The placement gate.** :meth:`ExternalWorker.load` returns a
:class:`LoadReport` carrying the per-provider node counts ORT wrote while the
graph actually ran. ``fraction_on_target`` is the number the planner refuses
on. It has to be measured this way because every cheaper check is wrong: a
VitisAI session lists the EP in ``get_providers()`` whether or not it took any
nodes, and when it takes none the outputs are bit-identical to the CPU's -- so
neither the provider list nor the numbers can tell a real NPU run from a silent
fallback. Only the node assignment can.

**The worker's own memory.** The engine's sampler walks a psutil process tree,
which stops at the WSL boundary; a Windows worker's resident set is invisible
to it. Since both AMD devices allocate out of the *same host RAM* the CPU
stage is budgeted against, an unreported worker is a hole in the admission
ledger, not a rounding error. So the worker reports its own RSS and
:meth:`stats` is how that reaches the ledger.
"""

from __future__ import annotations

import json
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.edge.local.external import launch as _launch
from vllm_omni.edge.local.external import protocol as _proto
from vllm_omni.edge.local.external.protocol import ProtocolError, WorkerError
from vllm_omni.host.process import spawn_worker, terminate_worker

DEFAULT_START_TIMEOUT_S = 120.0
"""Generous on purpose: the first VitisAI session on a cold machine spends
over a second in ``InferenceSession`` alone, and a Windows interpreter starting
off a ``\\\\wsl.localhost`` share is slower still."""

DEFAULT_CALL_TIMEOUT_S = 300.0

DEFAULT_CLOSE_TIMEOUT_S = 10.0
"""Bounded, for the same reason M0 bounds its cancellation drain: a worker that
will not exit must not hold the engine's shutdown open."""


@dataclass(frozen=True)
class LoadReport:
    """What the worker did when it opened the graph, and where it put it."""

    ep: str
    graph_path: str
    device_name: str = ""
    placement_granularity: str = "graph_nodes"
    session_providers: tuple[str, ...] = ()
    available_providers: tuple[str, ...] = ()
    node_counts: dict[str, int] = field(default_factory=dict)
    total_nodes: int = 0
    target_nodes: int = 0
    fraction_on_target: float | None = None
    """``None`` means *unverified*, which is not the same as zero and must not
    be rounded into it: no run happened, so ORT wrote no assignments."""
    session_create_s: float = 0.0
    warmup_s: float = 0.0
    rss_bytes: int = 0
    inputs: tuple[dict[str, Any], ...] = ()
    outputs: tuple[dict[str, Any], ...] = ()
    profile_path: str = ""
    ep_directory: str = ""
    requested_provider_missing: str = ""
    fallback_note: str = ""
    onnxruntime: str = ""
    device_id_requested: int | None = None
    note: str = ""

    @classmethod
    def from_body(cls, body: dict[str, Any], *, ep: str, graph_path: str) -> LoadReport:
        return cls(
            ep=ep,
            graph_path=graph_path,
            device_name=str(body.get("device_name") or "").rstrip("\x00"),
            placement_granularity=str(body.get("placement_granularity") or "graph_nodes"),
            session_providers=tuple(body.get("session_providers") or ()),
            available_providers=tuple(body.get("available_providers") or ()),
            node_counts=dict(body.get("node_counts") or {}),
            total_nodes=int(body.get("total_nodes") or 0),
            target_nodes=int(body.get("target_nodes") or 0),
            fraction_on_target=body.get("fraction_on_target"),
            session_create_s=float(body.get("session_create_s") or 0.0),
            warmup_s=float(body.get("warmup_s") or 0.0),
            rss_bytes=int(body.get("rss_bytes") or 0),
            inputs=tuple(body.get("inputs") or ()),
            outputs=tuple(body.get("outputs") or ()),
            profile_path=str(body.get("profile_path") or ""),
            ep_directory=str(body.get("ep_directory") or ""),
            requested_provider_missing=str(body.get("requested_provider_missing") or ""),
            fallback_note=str(body.get("fallback_note") or ""),
            onnxruntime=str(body.get("onnxruntime") or ""),
            device_id_requested=body.get("device_id_requested"),
            note=str(body.get("placement_unknown_reason") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if self.fraction_on_target is None:
            placement = f"placement unverified ({self.note or 'no profiled run'})"
        else:
            split = ", ".join(f"{k}={v}" for k, v in sorted(self.node_counts.items()))
            counted = "outputs" if self.placement_granularity == "output_device" else "nodes"
            placement = f"{self.target_nodes}/{self.total_nodes} {counted} on {self.ep} ({split})"
        device_suffix = f" ({self.device_name})" if self.device_name else ""
        return (
            f"{Path(self.graph_path).name} on {self.ep}{device_suffix}: {placement}; "
            f"session {self.session_create_s:.3f}s, warmup {self.warmup_s:.3f}s, "
            f"worker rss {self.rss_bytes / 2**20:.0f} MiB"
        )


@dataclass(frozen=True)
class RunTiming:
    """Where one forward's wall time went.

    Split because the composition rule needs it split: a stage only earns its
    place on a second device when the end-to-end gain beats the cost of getting
    there, and ``transport_s`` is that cost measured rather than assumed.
    """

    worker_s: float
    """Inside ``session.run`` on the far side."""
    round_trip_s: float
    """Wall time here, from send to the last byte of the reply."""

    @property
    def transport_s(self) -> float:
        """Serialisation, both socket traversals, and the worker's dispatch."""
        return max(0.0, self.round_trip_s - self.worker_s)

    def to_dict(self) -> dict[str, float]:
        return {
            "worker_s": self.worker_s,
            "round_trip_s": self.round_trip_s,
            "transport_s": self.transport_s,
        }


class WorkerStartError(RuntimeError):
    """The worker never connected back. Carries whatever it printed."""


def _default_gateway() -> str | None:
    """The far side of the default route, i.e. the Windows host under WSL2 NAT."""
    try:
        lines = Path("/proc/net/route").read_text().splitlines()[1:]
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) > 2 and fields[1] == "00000000":
            packed = int(fields[2], 16)
            return ".".join(str((packed >> shift) & 0xFF) for shift in (0, 8, 16, 24))
    return None


def _bind_address(*, is_windows: bool) -> str:
    """The address the worker should dial.

    A WSL worker gets loopback, which nothing outside this machine can reach. A
    native-Windows worker cannot use loopback -- under WSL2's NAT the two live
    on different network stacks -- so it needs this distro's address on the
    host-only network (``172.21.141.90`` here), which is the source address the
    routing table picks for the *default gateway*.

    The gateway specifically, not any address that happens to be handy. The
    obvious candidate is the nameserver in ``/etc/resolv.conf``, and it is
    wrong: WSL puts ``10.255.255.254`` on a loopback alias, so a probe aimed at
    it reports ``10.255.255.254`` as the source, the listener binds an address
    only this distro can reach, and the Windows worker gets ECONNREFUSED after
    a silent start. That is exactly the failure this comment exists to prevent
    a future edit from reintroducing.
    """
    if not is_windows or _launch.is_wsl() is False:
        return "127.0.0.1"
    # 1.1.1.1 is a fallback route target, not a destination: UDP ``connect``
    # sends no packets, it only resolves which local address would be used.
    for target in (_default_gateway(), "1.1.1.1"):
        if not target:
            continue
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect((target, 53))
            address = str(probe.getsockname()[0])
        except OSError:
            continue
        finally:
            probe.close()
        if not address.startswith("127."):
            return address
    return "127.0.0.1"


class ExternalWorker:
    """A stage running in another interpreter, possibly on another OS."""

    def __init__(
        self,
        route: _launch.Route,
        *,
        start_timeout_s: float = DEFAULT_START_TIMEOUT_S,
        call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
        max_payload_bytes: int = _proto.MAX_PAYLOAD_BYTES,
    ) -> None:
        if not route.available:
            raise WorkerStartError(f"route {route.name} is not available: {route.reason}")
        self.route = route
        self.start_timeout_s = start_timeout_s
        self.call_timeout_s = call_timeout_s
        self.max_payload_bytes = max_payload_bytes
        self._log = None
        self._lifecycle_lock = threading.RLock()
        self._drained = False
        self._process: subprocess.Popen[bytes] | None = None
        self._sock: socket.socket | None = None
        self._listener: socket.socket | None = None
        self.hello: dict[str, Any] = {}
        self.report: LoadReport | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> dict[str, Any]:
        """Launch the worker and complete the handshake. Returns its ``hello``."""
        if self._process is not None:
            raise WorkerStartError("worker instances cannot be started twice")
        token = secrets.token_hex(16)
        host = _bind_address(is_windows=self.route.is_windows)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, 0))
        listener.listen(1)
        listener.settimeout(self.start_timeout_s)
        self._listener = listener
        port = listener.getsockname()[1]

        argv = [
            str(self.route.interpreter),
            _launch.to_worker_path(
                Path(__file__).parents[3] / "host/worker_bootstrap.py", is_windows=self.route.is_windows
            ),
            _launch.to_worker_path(self.route.worker, is_windows=self.route.is_windows),
            "--host",
            host,
            "--port",
            str(port),
            "--token",
            token,
        ]
        self._log = tempfile.TemporaryFile()
        try:
            self._process = spawn_worker(argv, stdout=self._log, stderr=self._log)
        except BaseException:
            self.close()
            raise

        try:
            conn, _ = listener.accept()
        except TimeoutError as exc:
            raise WorkerStartError(
                f"worker did not connect back within {self.start_timeout_s:.0f}s: {self._drain_process()}"
            ) from exc
        finally:
            listener.close()
            self._listener = None

        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(self.call_timeout_s)
        self._sock = conn

        try:
            op, body, _ = _proto.recv_message(conn, max_payload_bytes=0)
        except BaseException:
            self.terminate()
            raise
        if op != _proto.OP_HELLO:
            self.close()
            raise WorkerStartError(f"first message was {op!r}, expected {_proto.OP_HELLO!r}")
        if body.get("token") != token:
            # Somebody else reached the listener first. Refuse rather than
            # speak to it: the whole point of the token.
            self.close()
            raise WorkerStartError("connect-back presented the wrong token; refusing the peer")
        self.hello = body
        return body

    def _drain_process(self) -> str:
        with self._lifecycle_lock:
            return self._drain_process_locked()

    def _drain_process_locked(self) -> str:
        process = self._process
        if process is None:
            return "worker was never started"
        if not terminate_worker(process):
            return f"worker pid {process.pid} did not exit"
        text = ""
        if self._log is not None:
            self._log.seek(0, 2)
            self._log.seek(max(0, self._log.tell() - 2000))
            text = self._log.read().decode("utf-8", "replace").strip()
        return f"exit={process.returncode} output={text[:2000]!r}" if text else f"exit={process.returncode}"

    def terminate(self) -> bool:
        """Interrupt a blocking native call without sending on its data socket.

        The caller retains reservations if the process tree did not drain.
        This method may run concurrently with the one in-flight `_call`.
        """
        with self._lifecycle_lock:
            return self._terminate_locked()

    def _terminate_locked(self) -> bool:
        if self._drained:
            return True
        native_drained = True
        if self._process is not None and self.route.is_windows and _launch.is_wsl():
            identity = self.hello.get("process_identity") or {}
            native_drained = False
            if type(identity.get("pid")) is int and type(identity.get("created_filetime")) is int:
                try:
                    helper = Path(__file__).parents[3] / "host/windows_process.py"
                    result = subprocess.run(
                        [
                            str(self.route.interpreter),
                            _launch.to_worker_path(helper, is_windows=True),
                            str(identity["pid"]),
                            str(identity["created_filetime"]),
                        ],
                        capture_output=True,
                        timeout=20,
                        text=True,
                    )
                    native_drained = result.returncode == 0 and json.loads(result.stdout).get("drained") is True
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
        # Keep the data connection open until native tree retirement completes:
        # closing it first lets an idle worker exit before we can verify its tree.
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        drained = (self._process is None or terminate_worker(self._process)) and native_drained
        self._drained = drained
        if drained and self._log is not None:
            self._log.close()
            self._log = None
        return drained

    def close(self, timeout_s: float = DEFAULT_CLOSE_TIMEOUT_S) -> None:
        """Ask the worker to exit; kill it if it will not. Safe to call twice."""
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.settimeout(min(timeout_s, 5.0))
                _proto.send_message(sock, _proto.OP_CLOSE, {})
                _proto.recv_message(sock)
            except (OSError, ProtocolError, WorkerError):
                pass
            finally:
                sock.close()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                if not self.terminate():
                    raise WorkerStartError("worker tree did not drain during close")
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> ExternalWorker:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- calls --------------------------------------------------------------

    def _call(
        self,
        op: str,
        body: dict[str, Any] | None = None,
        tensors: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        sock = self._sock
        if sock is None:
            raise WorkerStartError("worker is not running; call start() first")
        try:
            _proto.send_message(sock, op, body, tensors)
            reply_op, reply_body, reply_tensors = _proto.recv_message(sock, max_payload_bytes=self.max_payload_bytes)
        except (OSError, ProtocolError) as exc:
            # The socket is the only liveness signal we have; a dead worker
            # shows up as a broken pipe or a short read, and the useful part of
            # the diagnosis is on its stderr.
            detail = self._drain_process()
            sock.close()
            self._sock = None
            raise WorkerStartError(f"worker connection failed ({exc}); {detail}") from exc
        if reply_op != _proto.OP_OK:
            raise ProtocolError(f"unexpected reply {reply_op!r} to {op!r}")
        return reply_body, reply_tensors

    def load(
        self,
        graph_path: str | Path,
        *,
        example_inputs: dict[str, np.ndarray] | None = None,
        profile: bool = True,
        profile_prefix: str | Path | None = None,
        ep_dir: str | Path | None = None,
        intra_op_num_threads: int | None = None,
        device_id: int | None = None,
        artifact_files: list[Path] | None = None,
    ) -> LoadReport:
        """Open a graph and measure where its nodes actually ran.

        ``example_inputs`` are not optional in practice: without a run ORT
        writes no node assignments, so the report comes back with
        ``fraction_on_target is None`` and the planner treats the placement as
        unverified rather than assuming it is fine.
        """
        body: dict[str, Any] = {
            "graph_path": _launch.to_worker_path(graph_path, is_windows=self.route.is_windows),
            "ep": self.route.ep,
            "profile": bool(profile),
        }
        if profile_prefix is not None:
            body["profile_prefix"] = _launch.to_worker_path(profile_prefix, is_windows=self.route.is_windows)
        if ep_dir is not None:
            body["ep_dir"] = str(ep_dir)
        if intra_op_num_threads is not None:
            body["intra_op_num_threads"] = int(intra_op_num_threads)
        if device_id is not None:
            if self.route.ep != "dml" or device_id < 0:
                raise ValueError("device_id requires a nonnegative DirectML route")
            body["device_id"] = int(device_id)
        if artifact_files is not None:
            body["artifact_files"] = [
                _launch.to_worker_path(path, is_windows=self.route.is_windows) for path in artifact_files
            ]

        reply, _ = self._call(_proto.OP_LOAD, body, example_inputs or {})
        self.report = LoadReport.from_body(reply, ep=self.route.ep, graph_path=str(graph_path))
        return self.report

    def run(self, inputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], RunTiming]:
        """One forward, with the transport cost separated from the compute."""
        started = time.perf_counter()
        reply, outputs = self._call(_proto.OP_RUN, {}, inputs)
        round_trip = time.perf_counter() - started
        return outputs, RunTiming(worker_s=float(reply.get("run_s", 0.0)), round_trip_s=round_trip)

    def stats(self) -> dict[str, Any]:
        """Counters and the worker's own resident set, for the ledger."""
        reply, _ = self._call(_proto.OP_STATS, {})
        return reply


def start_worker(route_name: str, **kwargs: Any) -> ExternalWorker:
    """Resolve a route by name and start it."""
    worker = ExternalWorker(_launch.resolve(route_name), **kwargs)
    worker.start()
    return worker
