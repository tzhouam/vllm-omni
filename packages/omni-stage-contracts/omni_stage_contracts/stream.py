"""Portable credit accounting for one ordered Omni stage request.

The adapter owns the payload and backend state. This stream only fences event
identity/order and holds chunk/byte credit until a consumer acknowledges it.
It deliberately does not schedule model work or inspect state contents.
"""

from __future__ import annotations

import asyncio

from .types import StageEvent, StageRequest, StateHandle


class StageStreamClosed(RuntimeError):
    """A producer tried to submit to a completed or retired request."""


class BoundedStageEventStream:
    """One request's ordered event stream with strict byte and chunk limits.

    ``submit`` waits for consumer credit; ``receive`` does not return credit.
    The consumer must ``acknowledge(event.release_token)`` after it has finished
    with the payload. Cancellation fences the epoch and drops queued events;
    already delivered payloads stay charged until their consumers acknowledge.
    """

    def __init__(self, request: StageRequest, *, max_events: int, max_bytes: int) -> None:
        if type(max_events) is not int or max_events < 1 or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("event and byte bounds must be positive integers")
        if request.operation != "run":
            raise ValueError("an event stream requires a run request")
        self.request = request
        self.max_events = max_events
        self.max_bytes = max_bytes
        self._queue: asyncio.Queue[StageEvent | None] = asyncio.Queue()
        self._credit = asyncio.Condition()
        self._pending: dict[str, int] = {}
        self._seen_tokens: set[str] = set()
        self._delivered: set[str] = set()
        self._used_bytes = 0
        self._next_seq = 1
        self._last_watermark = 0
        self._epoch = request.epoch
        self._state: StateHandle | None = request.state
        self._terminal_submitted = False
        self._closed = False
        self._cancelled = False
        self.dropped_stale = 0
        self.high_water_events = 0
        self.high_water_bytes = 0

    @property
    def state(self) -> StateHandle | None:
        return self._state

    @property
    def outstanding_events(self) -> int:
        return len(self._pending)

    @property
    def outstanding_bytes(self) -> int:
        return self._used_bytes

    def _validate(self, event: StageEvent) -> int:
        request = self.request
        if (
            event.request_id != request.request_id
            or event.stage_id != request.stage_id
            or event.worker_generation != request.worker_generation
            or event.epoch != self._epoch
        ):
            raise ValueError("event request, stage, worker or epoch mismatch")
        if event.seq != self._next_seq:
            raise ValueError(f"event sequence gap or duplicate: expected {self._next_seq}")
        if event.input_watermark < self._last_watermark:
            raise ValueError("event input watermark regressed")
        if not event.started_monotonic_ns or not event.release_token:
            raise ValueError("credited events require timestamps and an acknowledgement token")
        if event.release_token in self._seen_tokens:
            raise ValueError("duplicate event acknowledgement token")
        if event.state is not None and self._state is not None and not self._state.accepts(event.state):
            raise ValueError("event returned a different backend state")
        if event.state is not None:
            if not request.session_id or event.state.session_id != request.session_id:
                raise ValueError("event returned state from another or unbound session")
            if (
                not request.artifact_id
                or event.state.artifact_id != request.artifact_id
                or event.state.layout_version != request.state_layout_version
            ):
                raise ValueError("event returned state for another artifact or layout")
        size = max(event.payload_nbytes, sum(buffer.nbytes for buffer in event.buffers))
        if size > self.max_bytes:
            raise ValueError("event exceeds the stream's admitted byte bound")
        return size

    async def submit(self, event: StageEvent) -> None:
        """Wait for both byte and event credit before publishing an event."""
        async with self._credit:
            if self._closed or self._terminal_submitted:
                raise StageStreamClosed("stage stream is closed")
            size = self._validate(event)
            await self._credit.wait_for(
                lambda: self._closed or self._terminal_submitted
                or (len(self._pending) < self.max_events and self._used_bytes + size <= self.max_bytes)
            )
            if self._closed or self._terminal_submitted:
                raise StageStreamClosed("stage stream was retired while waiting for credit")
            # A concurrent producer may have advanced the sequence while we
            # waited. Revalidate under the same lock before taking credit.
            self._validate(event)
            self._pending[event.release_token] = size
            self._seen_tokens.add(event.release_token)
            self._used_bytes += size
            self._next_seq += 1
            self._last_watermark = event.input_watermark
            if event.state is not None:
                self._state = event.state
            self.high_water_events = max(self.high_water_events, len(self._pending))
            self.high_water_bytes = max(self.high_water_bytes, self._used_bytes)
            if event.terminal:
                self._terminal_submitted = True
            self._queue.put_nowait(event)
            if event.terminal:
                self._queue.put_nowait(None)
                self._credit.notify_all()

    async def receive(self) -> StageEvent | None:
        """Take the next live event; the caller still owes an ACK."""
        while True:
            if self._closed and self._queue.empty():
                return None
            event = await self._queue.get()
            if event is None:
                # Close before any await, and leave a sentinel for another
                # consumer already blocked in queue.get(). Losing this sole
                # marker on task cancellation would hang that consumer.
                self._closed = True
                self._queue.put_nowait(None)
                async with self._credit:
                    self._credit.notify_all()
                return None
            # No await may separate dequeue from delivery registration. If a
            # consumer is cancelled while waiting for the condition lock,
            # its dequeued event disappears and its byte credit is stranded.
            # These mutations are atomic on this event loop.
            if event.epoch != self._epoch:
                self.dropped_stale += 1
                self._release(event.release_token)
                async with self._credit:
                    self._credit.notify_all()
                continue
            self._delivered.add(event.release_token)
            return event

    def _release(self, token: str) -> bool:
        size = self._pending.pop(token, None)
        if size is None:
            return False
        self._used_bytes -= size
        self._delivered.discard(token)
        return True

    async def acknowledge(self, release_token: str) -> bool:
        """Return credit once; duplicate or stale ACKs cannot free new work."""
        async with self._credit:
            if release_token not in self._delivered:
                return False
            released = self._release(release_token)
            if released:
                self._credit.notify_all()
            return released

    async def cancel(self) -> int:
        """Retire this epoch and discard undelivered events, waking waiters."""
        async with self._credit:
            if self._cancelled:
                return self._epoch
            self._cancelled = True
            self._epoch += 1
            self._closed = True
            while not self._queue.empty():
                event = self._queue.get_nowait()
                if event is not None:
                    self.dropped_stale += 1
                    self._release(event.release_token)
            self._queue.put_nowait(None)
            self._credit.notify_all()
            return self._epoch

    async def close(self) -> None:
        """Stop accepting events without claiming that delivered buffers are free."""
        async with self._credit:
            if not self._closed:
                self._closed = True
                self._queue.put_nowait(None)
                self._credit.notify_all()
