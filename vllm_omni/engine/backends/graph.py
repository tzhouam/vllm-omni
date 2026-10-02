# SPDX-License-Identifier: Apache-2.0
"""One bounded, non-preemptible graph stage using the existing worker.

The runtime owns startup and capacity. This adapter owns one in-flight call,
including its output until acknowledged by the consumer. No token scheduler,
provider fallback, or automatic worker replay is introduced here.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from omni_stage_contracts import ArtifactManifest, BufferRef, StageEvent, StageRequest
from vllm_omni.edge.local.external.client import ExternalWorker
from vllm_omni.engine.resource_ledger import ResourceUnavailable
from vllm_omni.engine.stage_client import StageClientBase
from vllm_omni.host.routes import resolve_route
from vllm_omni.outputs import OmniRequestOutput


class GraphStageClient(StageClientBase):
    def __init__(self, metadata, config, ledger, reservation) -> None:
        for name, value in vars(metadata).items():
            setattr(self, name, value)
        self.stage_type = "graph"
        self._ledger, self._reservation = ledger, reservation
        self._generation = uuid.uuid4().hex
        self._config = dict(config)
        self._worker = None
        self._closed = False
        self._active: str | None = None
        self._task: asyncio.Task | None = None
        self._output: OmniRequestOutput | None = None
        self._epoch = 0
        self._max_io_bytes = int(config.get("max_io_bytes", 8 << 20))
        if self._max_io_bytes <= 0 or self._max_io_bytes > min(reservation.demands.values()):
            raise ValueError("graph I/O limit must be positive and fit the stage reservation")
        try:
            manifest_path = Path(config["manifest"]).resolve()
            manifest = ArtifactManifest.read(manifest_path)
            descriptor = manifest.artifact_metadata()
            if descriptor is not None and not descriptor.qualified:
                raise ValueError("graph artifact failed numerical or task validation")
            if descriptor is not None:
                raise ValueError(
                    "external.graph.v1 cannot bind v2 artifact ABI, runtime, layout and shape to its worker"
                )
            graph_name = manifest.metadata["graph_file"]
            examples_name = manifest.metadata["example_inputs_file"]
            if graph_name not in manifest.files or examples_name not in manifest.files:
                raise ValueError("graph and placement examples must be hashed artifact payloads")
            if sum((manifest_path.parent / n).stat().st_size for n in manifest.files) > min(
                reservation.demands.values()
            ):
                raise ResourceUnavailable("artifact files exceed stage memory reservation")
            route = resolve_route(config["route"])
            self._worker = ExternalWorker(
                route, start_timeout_s=float(config.get("start_timeout_s", 120)), max_payload_bytes=self._max_io_bytes
            )
            self._worker.start()
            with np.load(manifest_path.parent / examples_name, allow_pickle=False) as example:
                inputs = {name: example[name] for name in example.files}
            self._describe(inputs)
            self.load_report = self._worker.load(
                manifest_path.parent / graph_name,
                example_inputs=inputs,
                ep_dir=config.get("ep_dir"),
                intra_op_num_threads=config.get("intra_op_num_threads"),
                profile_prefix=config.get("profile_prefix"),
                artifact_files=[manifest_path.parent / name for name in manifest.files],
            )
            report = self.load_report
            minimum = float(config.get("min_fraction_on_target", 1.0))
            if not 0 < minimum <= 1:
                raise ValueError("placement threshold must be in (0, 1]")
            if (
                report.requested_provider_missing
                or report.fraction_on_target is None
                or not math.isfinite(report.fraction_on_target)
                or not 0 < report.fraction_on_target <= 1
                or report.target_nodes <= 0
                or report.fraction_on_target < minimum
            ):
                raise RuntimeError(f"REFUSE_EP_PLACEMENT: {report.summary()}")
            allowed = set(config.get("allowed_providers", report.session_providers if route.ep == "cpu" else ()))
            if not allowed or set(report.node_counts) - allowed:
                raise RuntimeError("REFUSE_EP_PLACEMENT: declare all permitted execution providers")
            if report.rss_bytes > min(reservation.demands.values()):
                raise ResourceUnavailable("loaded worker RSS exceeds stage memory reservation")
            self._input_schema = {name: (value.dtype.str, value.shape) for name, value in inputs.items()}
            from omni_stage_contracts.types import file_digest

            self.execution_plan = {
                "backend": "external.graph.v1",
                "stage_id": self.stage_id,
                "worker_generation": self._generation,
                "artifact_sha256": file_digest(manifest_path),
                "component": manifest.component,
                "artifact_schema_version": manifest.schema_version,
                "route": route.to_dict(),
                "reserved_bytes": dict(reservation.demands),
                "max_io_bytes": self._max_io_bytes,
                "placement": report.to_dict(),
                "worker": {k: v for k, v in self._worker.hello.items() if k != "token"},
                "evidence": "B",
                "stateful": False,
            }
        except BaseException:
            self.shutdown()
            raise

    def _describe(self, tensors: dict[str, np.ndarray]) -> tuple[BufferRef, ...]:
        refs = tuple(
            BufferRef(
                name, str(self.stage_id), self._generation, array.dtype.name, tuple(array.shape), int(array.nbytes)
            )
            for name, array in tensors.items()
        )
        if sum(r.nbytes for r in refs) > self._max_io_bytes:
            raise ResourceUnavailable("graph payload exceeds admitted I/O bound")
        return refs

    async def add_request_async(self, request_id: str, prompt: Any, params: Any = None) -> None:
        self.check_health()
        if self._active is not None:
            raise ResourceUnavailable("graph stage has an unacknowledged request; capacity is one")
        if not isinstance(prompt, dict) or not isinstance(prompt.get("tensors"), dict):
            raise ValueError("graph prompt must contain a tensors mapping supplied by its model adapter")
        inputs = {name: np.asarray(value) for name, value in prompt["tensors"].items()}
        refs = self._describe(inputs)
        if {name: (value.dtype.str, value.shape) for name, value in inputs.items()} != self._input_schema:
            raise ValueError("input names/dtypes/shapes differ from the placement-verified artifact bucket")
        # Copy at submission: application mutation cannot change an admitted call.
        inputs = {name: value.copy() for name, value in inputs.items()}
        self._epoch += 1
        epoch = self._epoch
        request = StageRequest(request_id, self.stage_id, epoch, self._generation, inputs=refs)
        self._active = request_id
        self._task = asyncio.create_task(self._run(request, inputs), name=f"graph-{self.stage_id}-{request_id}")

    async def _run(self, request: StageRequest, inputs: dict[str, np.ndarray]) -> None:
        try:
            started_ns = time.monotonic_ns()
            outputs, timing = await asyncio.to_thread(self._worker.run, inputs)
            emitted_ns = time.monotonic_ns()
            refs = self._describe(outputs)
            if sum(x.nbytes for x in outputs.values()) + sum(x.nbytes for x in inputs.values()) > self._max_io_bytes:
                raise ResourceUnavailable("combined graph input/output exceeds admitted I/O bound")
            event = StageEvent(
                request.request_id, self.stage_id, request.epoch, 1, "tensor", self._generation, refs,
                terminal=True, started_monotonic_ns=started_ns, emitted_monotonic_ns=emitted_ns,
                payload_nbytes=sum(ref.nbytes for ref in refs), release_token=uuid.uuid4().hex,
            )
            output = OmniRequestOutput(
                request_id=request.request_id,
                stage_id=self.stage_id,
                final_output_type=self.final_output_type or "latent",
                _custom_output={"tensors": outputs, "stage_event": dataclasses.asdict(event)},
                metrics={"graph_timing": timing.to_dict()},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            output = OmniRequestOutput.from_error(request.request_id, str(exc))
            output.stage_id = self.stage_id
        if not self._closed and self._epoch == request.epoch:
            loop = asyncio.get_running_loop()
            output._stage_release = lambda: loop.call_soon_threadsafe(
                self.acknowledge, request.request_id, request.epoch, request.worker_generation
            )
            self._output = output

    def get_graph_output_nowait(self):
        # Delivery is separate from release: the slot stays charged until ACK.
        output, self._output = self._output, None
        return output

    def acknowledge(self, request_id: str, epoch: int, generation: str) -> None:
        if (
            self._active == request_id
            and epoch == self._epoch
            and generation == self._generation
            and (self._task is None or self._task.done())
        ):
            self._active = None
            self._task = None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if self._active not in request_ids:
            return
        self._epoch += 1
        self._output = None
        # A completed call needs no worker kill. Otherwise cancellation retires
        # this single-session worker; the runtime will report it unavailable.
        if self._task is not None and not self._task.done():
            drained = await asyncio.to_thread(self._worker.terminate)
            self._closed = True
            done, _ = await asyncio.wait({self._task}, timeout=5)
            self._ledger.release(self._reservation, drained=drained and bool(done) and not self._task.cancelled())
        self._active = None

    def check_health(self) -> None:
        if self._closed:
            from vllm.v1.engine.exceptions import EngineDeadError

            raise EngineDeadError()

    async def collective_rpc_async(self, method, timeout=None, args=(), kwargs=None):
        raise NotImplementedError(f"graph backend does not implement collective RPC {method}")

    def shutdown(self) -> None:
        self._closed = True
        self._output = None
        drained = self._worker is None or self._worker.terminate()
        task = self._task
        if task is not None and not task.done():
            # Process exit does not release Python input/transport references.
            # Keep them charged until the call coroutine actually unwinds.
            self._ledger.release(self._reservation, drained=False)
            task.add_done_callback(
                lambda done: self._ledger.release(self._reservation, drained=drained and not done.cancelled())
            )
        else:
            self._ledger.release(self._reservation, drained=drained)
