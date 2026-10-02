"""Host reference for the minimal single-request stage-controller boundary.

This module does not execute a model. A device adapter owns its weights, state,
workspaces and cancellation primitive. The controller only binds that adapter
to v2 request identity, ordered/credited events and explicit state release.
Native mobile controllers can run the same conformance fixture against their
own implementation without embedding Python or a vLLM service on device.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Mapping, Protocol

from .stream import BoundedStageEventStream, StageStreamClosed
from .types import StageEvent, StageRequest, StateHandle

EventEmitter = Callable[[StageEvent], Awaitable[None]]


class StageCancelTimeout(TimeoutError):
    """The adapter did not confirm cancellation and quiescence in time."""


class StageBackend(Protocol):
    """A stage adapter's control surface; tensor/model execution stays inside it."""

    backend: str
    instance_id: str
    worker_generation: str

    async def run(self, request: StageRequest, emit: EventEmitter) -> None: ...

    async def cancel(self, request: StageRequest) -> None: ...

    async def release(self, request: StageRequest) -> None: ...


@dataclass(frozen=True)
class ControllerLimits:
    max_events: int
    max_bytes: int

    def __post_init__(self) -> None:
        if type(self.max_events) is not int or self.max_events < 1:
            raise ValueError("max_events must be a positive integer")
        if type(self.max_bytes) is not int or self.max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")


class ControlledStageRun:
    """One backend run and its consumer-owned event credits."""

    def __init__(
        self,
        controller: ReferenceStageController,
        request: StageRequest,
        backend: StageBackend,
        limits: ControllerLimits,
    ) -> None:
        self._controller = controller
        self.request = request
        self.backend = backend
        self.stream = BoundedStageEventStream(
            request, max_events=limits.max_events, max_bytes=limits.max_bytes
        )
        self._task: asyncio.Task[None] | None = None
        self._cancelled = False
        self._cancel_unresolved = False
        self._terminal_emitted = False

    @property
    def state(self) -> StateHandle | None:
        return self.stream.state

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def cancel_unresolved(self) -> bool:
        return self._cancel_unresolved

    async def receive(self) -> StageEvent | None:
        event = await self.stream.receive()
        if event is None and not self._cancelled:
            # Draining the stream must surface a producer failure; otherwise a
            # short stream could be mistaken for a successful completion.
            await self.wait()
        return event

    async def acknowledge(self, event: StageEvent | str) -> bool:
        token = event.release_token if isinstance(event, StageEvent) else event
        return await self.stream.acknowledge(token)

    async def wait(self) -> None:
        """Wait for adapter completion; callers must consume/ACK separately."""
        assert self._task is not None
        await self._task
        self._cancel_unresolved = False

    async def cancel(self) -> None:
        await self._controller.cancel(self)


class ReferenceStageController:
    """Portable single-request controller for a mapped set of stage adapters.

    A backend may create an opaque state handle in a successful event. The
    controller remembers only its identity, never its contents. Continuations
    require the same live state and backend instance. Cancellation retires that
    state for continuation, but its memory is not reported free until an
    explicit backend ``release`` succeeds and all delivered payloads are ACKed.
    """

    def __init__(
        self, backends: Mapping[int, StageBackend], *, max_events: int,
        max_bytes: int, cancel_timeout_s: float = 5.0,
    ) -> None:
        self.backends = dict(backends)
        self.limits = ControllerLimits(max_events, max_bytes)
        if (
            isinstance(cancel_timeout_s, bool)
            or not isinstance(cancel_timeout_s, (float, int))
            or not math.isfinite(cancel_timeout_s)
            or cancel_timeout_s <= 0
        ):
            raise ValueError("cancel_timeout_s must be finite and positive")
        self.cancel_timeout_s = float(cancel_timeout_s)
        if not self.backends or any(
            type(stage_id) is not int or stage_id < 0 for stage_id in self.backends
        ):
            raise ValueError("stage backends need nonnegative stage IDs")
        if any(
            not backend.backend or not backend.instance_id or not backend.worker_generation
            for backend in self.backends.values()
        ):
            raise ValueError("stage backends need backend, instance and worker identity")
        self._active: ControlledStageRun | None = None
        self._states: dict[tuple[int, str, str, str, str], StateHandle] = {}
        self._retired: set[tuple[int, str, str, str, str]] = set()
        self._releasing: set[tuple[int, str, str, str, str]] = set()
        self._release_ids: dict[tuple[int, str, str, str, str], str] = {}
        self._state_runs: dict[tuple[int, str, str, str, str], list[ControlledStageRun]] = {}

    @staticmethod
    def _key(stage_id: int, state: StateHandle) -> tuple[int, str, str, str, str]:
        return (
            stage_id, state.backend, state.backend_instance_id,
            state.worker_generation, state.state_id,
        )

    @staticmethod
    def _check_backend(state: StateHandle, backend: StageBackend) -> None:
        if (
            state.backend != backend.backend
            or state.backend_instance_id != backend.instance_id
            or state.worker_generation != backend.worker_generation
        ):
            raise ValueError("state is bound to a different backend instance")

    def _track_state(self, run: ControlledStageRun, state: StateHandle) -> None:
        key = self._validate_state(run, state)
        self._states[key] = state
        runs = self._state_runs.setdefault(key, [])
        if run not in runs:
            runs.append(run)

    def _validate_state(
        self, run: ControlledStageRun, state: StateHandle
    ) -> tuple[int, str, str, str, str]:
        self._check_backend(state, run.backend)
        request = run.request
        if (
            state.session_id != request.session_id
            or state.artifact_id != request.artifact_id
            or state.layout_version != request.state_layout_version
            or state.epoch != request.epoch
        ):
            raise ValueError(
                "returned state differs from request session, artifact, layout or epoch"
            )
        key = self._key(request.stage_id, state)
        previous = self._states.get(key)
        if previous is not None and not previous.accepts(state):
            raise ValueError("backend reused an opaque state ID for a different state")
        if key in self._retired:
            raise ValueError("backend returned a retired state")
        return key

    def start(self, request: StageRequest) -> ControlledStageRun:
        """Start exactly one active request using its declared stage placement."""
        if request.operation != "run":
            raise ValueError("start requires a run request")
        if self._releasing:
            raise RuntimeError("a backend state release is in progress")
        if self._active is not None:
            previous = self._active
            if (
                previous._task is None
                or not previous._task.done()
                or previous.stream.outstanding_events
            ):
                raise RuntimeError(
                    "this batch-1 reference controller already has an active request"
                )
            self._active = None
        if not request.session_id or not request.artifact_id or request.state_layout_version < 1:
            raise ValueError("v2 controller requests require session, artifact and state layout")
        try:
            backend = self.backends[request.stage_id]
        except KeyError as exc:
            raise ValueError(f"no backend for stage {request.stage_id}") from exc
        if request.worker_generation != backend.worker_generation:
            raise ValueError("request worker generation differs from selected backend")
        run = ControlledStageRun(self, request, backend, self.limits)
        if request.state is not None:
            self._check_backend(request.state, backend)
            key = self._key(request.stage_id, request.state)
            live = self._states.get(key)
            if (
                live is None or key in self._retired or key in self._releasing
                or not live.accepts(request.state)
            ):
                raise ValueError("request state is not live in this controller")
            self._track_state(run, request.state)
        self._active = run
        run._task = asyncio.create_task(self._produce(run))
        return run

    async def _produce(self, run: ControlledStageRun) -> None:
        async def emit(event: StageEvent) -> None:
            if event.state is not None:
                # Refuse a bad state before publishing the event or consuming
                # credit. Register it only after the stream accepts it.
                self._validate_state(run, event.state)
            await run.stream.submit(event)
            if event.state is not None:
                self._track_state(run, event.state)
            if event.terminal:
                run._terminal_emitted = True

        try:
            if not run._cancelled:
                await run.backend.run(run.request, emit)
            if not run._cancelled and not run._terminal_emitted:
                raise RuntimeError("backend completed without a terminal stage event")
        except StageStreamClosed:
            if not run._cancelled:
                if run.state is not None:
                    key = self._key(run.request.stage_id, run.state)
                    if key in self._states:
                        self._retired.add(key)
                raise
        except BaseException:
            # A failed producer cannot certify that a previously returned
            # state is still usable. Keep it charged for explicit cleanup.
            if run.state is not None:
                key = self._key(run.request.stage_id, run.state)
                if key in self._states:
                    self._retired.add(key)
            raise
        finally:
            if not run._terminal_emitted or run._cancelled:
                await run.stream.close()

    async def cancel(self, run: ControlledStageRun) -> None:
        """Fence queued events, notify the adapter, and await its stop ACK."""
        if run._controller is not self:
            raise ValueError("run belongs to another controller")
        if not run._cancelled:
            run._cancelled = True
            await run.stream.cancel()
            if run.state is not None:
                key = self._key(run.request.stage_id, run.state)
                if key in self._states:
                    self._retired.add(key)
        assert run._task is not None
        if not run._task.done():
            cancel_request = replace(
                run.request, operation="cancel", inputs=(), state=run.state
            )
            try:
                await asyncio.wait_for(run.backend.cancel(cancel_request), self.cancel_timeout_s)
            except asyncio.TimeoutError as exc:
                run._cancel_unresolved = True
                raise StageCancelTimeout("adapter did not acknowledge cancellation") from exc
        try:
            # Shield leaves an unresponsive backend visible as active rather
            # than mistaking Python task cancellation for hardware quiescence.
            await asyncio.wait_for(asyncio.shield(run._task), self.cancel_timeout_s)
        except asyncio.TimeoutError as exc:
            run._cancel_unresolved = True
            raise StageCancelTimeout("adapter did not stop after cancellation") from exc
        run._cancel_unresolved = False

    async def release(self, stage_id: int, state: StateHandle) -> None:
        """Call the owning adapter; remove the handle only after its ACK."""
        if self._active is not None:
            active = self._active
            if active._task is None or not active._task.done() or active.stream.outstanding_events:
                raise RuntimeError(
                    "active request or unacknowledged payloads prevent state release"
                )
        try:
            backend = self.backends[stage_id]
        except KeyError as exc:
            raise ValueError(f"no backend for stage {stage_id}") from exc
        self._check_backend(state, backend)
        key = self._key(stage_id, state)
        live = self._states.get(key)
        if live is None or not live.accepts(state):
            raise ValueError("state is not live in this controller")
        if key in self._releasing:
            raise RuntimeError("state release is already in progress")
        if any(
            run._task is not None and (not run._task.done() or run.stream.outstanding_events)
            for run in self._state_runs.get(key, ())
        ):
            raise RuntimeError("state still has running work or unacknowledged payloads")
        self._releasing.add(key)
        try:
            # Keep the operation identity stable across an uncertain ACK. A
            # native adapter must make release idempotent for this request ID.
            release_id = self._release_ids.setdefault(
                key, f"release-{uuid.uuid4().hex}"
            )
            release_request = StageRequest.release(
                release_id, stage_id, state
            )
            await backend.release(release_request)
        except BaseException:
            # An adapter that did not confirm release must remain charged and
            # may be retried or diagnosed by the caller.
            raise
        else:
            del self._states[key]
            self._retired.discard(key)
            self._release_ids.pop(key, None)
            self._state_runs.pop(key, None)
        finally:
            self._releasing.discard(key)
