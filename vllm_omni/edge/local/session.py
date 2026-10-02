# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The three data boundaries the local engine needs, and nothing else.

The proposal's section 4 is explicit that the first version needs exactly three
contracts: **a request/event stream, a tensor-or-buffer reference, and an
opaque state handle**. This module is the first and the third. The second does
not appear in M0 because a single text stage never hands a buffer to anybody --
it is the thing that arrives in M2, when a talker hands codec frames to a
vocoder, and inventing it now would be inventing it wrong.

Two properties here are correctness, not bookkeeping:

**An epoch fences cancellation.** Cancelling a request is not "stop calling
next()". The backend has work in flight, and tokens from that work arrive after
the cancel. Every event carries the epoch it was produced under, and
:class:`BoundedEventStream` drops events from a retired epoch instead of
delivering them. Without that, an interrupted turn's tail lands in the next
turn's output -- the "重复输出和跨请求污染" the proposal names.

**A queue bounds chunks and bytes, both.** A consumer that stops reading must
slow the producer down, not grow the heap. Bounding only the chunk count is not
enough once a chunk can be a second of PCM rather than one token, so the bound
is on both from the start even though M0's chunks are small.

``StateHandle`` is deliberately opaque. The engine holds a *reference* to the
session's state; vLLM holds the actual KV blocks, the position counter and the
RNG. The handle records enough to know when that state is no longer valid --
a different artifact, a different layout version, a retired epoch -- and
nothing that would let a caller reach inside it.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from typing import Any

STATE_LAYOUT_VERSION = 1
"""Bumped when the meaning of a handle's state changes. A handle from a
different version is not migrated, it is rejected: the proposal's item 10 says
to treat state as non-migratable until a specific backend pair has been
measured, and a version that silently matches would hide that."""


from omni_stage_contracts.legacy import ChunkEvent, StateHandle  # noqa: E402, F401


class StreamClosed(RuntimeError):  # noqa: N818 - retained public API
    """Raised on a stream that has been closed while a consumer waited."""


class EventTooLarge(ValueError):
    """A producer must fragment an event before it can enter this stream."""


class BoundedEventStream:
    """An async queue with a credit bound on chunks *and* bytes.

    Producers ``await put(...)``, which blocks while the consumer is behind:
    that is the backpressure. A delivered event retains its credit until the
    consumer acknowledges it. For existing ``async for``/``get`` callers, the
    next pull acknowledges the previous event owned by that task. A consumer
    which stops pulling can call ``await acknowledge(event)`` explicitly. A
    wrapper such as ``asyncio.wait_for(stream.get(), ...)`` uses a helper task;
    its caller must acknowledge explicitly before continuing that stream.
    """

    def __init__(self, *, max_chunks: int = 64, max_bytes: int = 4 << 20) -> None:
        if max_chunks < 1 or max_bytes < 1:
            raise ValueError("bounds must be positive")
        self.max_chunks = max_chunks
        self.max_bytes = max_bytes
        self._queue: asyncio.Queue[tuple[ChunkEvent, int] | None] = asyncio.Queue()
        # Queued and delivered-but-unacknowledged events both consume credit.
        self._bytes = 0
        self._chunks = 0
        self._delivered: dict[str, tuple[ChunkEvent, int, asyncio.Task[Any] | None]] = {}
        self._retired_delivery_tokens: set[str] = set()
        self._task_delivery: dict[asyncio.Task[Any], str] = {}
        self._credit = asyncio.Condition()
        self._closed = False
        self._epoch = 0
        self.dropped_stale = 0
        """Events discarded because their epoch had been retired. Reported
        rather than silently swallowed: a non-zero count on a run with no
        cancellation is a bug worth seeing."""
        self.high_water_chunks = 0
        self.high_water_bytes = 0
        self.acknowledged_events = 0
        self.retired_delivered = 0

    # -- producer side ------------------------------------------------------
    @staticmethod
    def _payload_bytes(payload: Any) -> int:
        """Count the logical bytes of the supported, copyable wire payloads.

        Unknown objects must be adapted to a buffer reference before queuing;
        assigning them an arbitrary small size would defeat the byte bound.
        """
        if payload is None:
            return 0
        if isinstance(payload, memoryview):
            return payload.nbytes
        if isinstance(payload, (bytes, bytearray)):
            return len(payload)
        if isinstance(payload, str):
            return len(payload.encode("utf-8"))
        if isinstance(payload, bool):
            return 1
        if isinstance(payload, (int, float)):
            return 8
        if isinstance(payload, (tuple, list)):
            return sum(BoundedEventStream._payload_bytes(item) for item in payload)
        if isinstance(payload, dict):
            if not all(isinstance(key, str) for key in payload):
                raise TypeError("event payload dictionary keys must be strings")
            return sum(
                len(key.encode("utf-8"))
                + BoundedEventStream._payload_bytes(value)
                for key, value in payload.items()
            )
        raise TypeError(f"unsupported event payload type: {type(payload).__name__}")

    @staticmethod
    def _copy_payload(payload: Any) -> Any:
        """Copy allowed payloads so their size stays fixed while queued."""
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return bytes(payload)
        if payload is None or isinstance(payload, (str, bool, int, float)):
            return payload
        if isinstance(payload, (tuple, list)):
            copied = [BoundedEventStream._copy_payload(item) for item in payload]
            return tuple(copied) if isinstance(payload, tuple) else copied
        if isinstance(payload, dict):
            if not all(isinstance(key, str) for key in payload):
                raise TypeError("event payload dictionary keys must be strings")
            return {
                key: BoundedEventStream._copy_payload(value)
                for key, value in payload.items()
            }
        raise TypeError(f"unsupported event payload type: {type(payload).__name__}")

    @staticmethod
    def _snapshot(event: ChunkEvent) -> ChunkEvent:
        """Keep a producer from enlarging a mutable payload after admission."""
        return replace(
            event,
            payload=BoundedEventStream._copy_payload(event.payload),
            # The stream, not the producer, owns the release token. It is
            # unique even if a producer repeats a sequence number by mistake.
            release_token=uuid.uuid4().hex,
        )

    def _release_credit(self, size: int) -> None:
        self._chunks -= 1
        self._bytes -= size
        assert self._chunks >= 0 and self._bytes >= 0

    def _acknowledge_locked(self, token: str) -> bool:
        """Return one delivery's credit while ``_credit`` is held."""
        delivery = self._delivered.pop(token, None)
        if delivery is None:
            return False
        _, size, owner = delivery
        self._retired_delivery_tokens.discard(token)
        if owner is not None and self._task_delivery.get(owner) == token:
            del self._task_delivery[owner]
        self._release_credit(size)
        self.acknowledged_events += 1
        self._credit.notify_all()
        return True

    async def acknowledge(self, event: ChunkEvent | str | None = None) -> bool:
        """Return a delivered event's credit exactly once.

        Pass the event (or its opaque ``release_token``) when handing it to
        another task. With no argument, acknowledge this task's last delivery.
        Returns ``False`` if it was already acknowledged or retired.
        """
        if isinstance(event, ChunkEvent):
            token = event.release_token
        elif isinstance(event, str):
            token = event
        elif event is None:
            task = asyncio.current_task()
            token = self._task_delivery.get(task, "") if task is not None else ""
        else:
            raise TypeError("acknowledge expects a ChunkEvent, release token, or None")
        if not token:
            return False
        async with self._credit:
            return self._acknowledge_locked(token)

    async def put(self, event: ChunkEvent) -> None:
        """Enqueue, waiting for credit. Stale-epoch events are dropped here."""
        if self._closed:
            raise StreamClosed("stream is closed")
        event = self._snapshot(event)
        size = self._payload_bytes(event.payload)
        if size > self.max_bytes:
            raise EventTooLarge(
                f"event payload is {size} bytes; limit is {self.max_bytes}; "
                "fragment the event or increase the stream limit"
            )
        async with self._credit:
            await self._credit.wait_for(
                lambda: (
                    self._closed
                    or event.epoch < self._epoch
                    or (
                        self._chunks < self.max_chunks
                        and self._bytes + size <= self.max_bytes
                    )
                )
            )
            if self._closed:
                raise StreamClosed("stream closed while producing")
            if event.epoch < self._epoch:
                self.dropped_stale += 1
                return
            self._chunks += 1
            self._bytes += size
            self.high_water_chunks = max(self.high_water_chunks, self._chunks)
            self.high_water_bytes = max(self.high_water_bytes, self._bytes)
            # The queue is unbounded; enqueue in the same critical section as
            # the reservation so cancellation cannot strand allocated credit.
            self._queue.put_nowait((event, size))

    async def close(self) -> None:
        """End production, preserving queued events for normal drain."""
        async with self._credit:
            if self._closed:
                return
            self._closed = True
            self._credit.notify_all()
            self._queue.put_nowait(None)

    async def _notify_credit(self) -> None:
        async with self._credit:
            self._credit.notify_all()

    def _schedule_credit_wakeup(self) -> None:
        try:
            asyncio.get_running_loop().create_task(self._notify_credit())
        except RuntimeError:
            # No blocked async producer can be running without an event loop.
            pass

    def retire_epoch(self, epoch: int) -> int:
        """Fence queued events; delivered payloads still require their ACK.

        This is synchronous for the engine's cancellation API. All mutations
        run without an ``await`` on the event loop, so a producer cannot race
        the queue drain. A wakeup task resumes producers waiting for credit.
        """
        new_epoch = max(self._epoch, epoch + 1)
        if new_epoch == self._epoch:
            return self.dropped_stale
        self._epoch = new_epoch
        survivors: list[tuple[ChunkEvent, int] | None] = []
        while True:
            try:
                queued = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if queued is None:
                survivors.append(None)
            elif queued[0].epoch < new_epoch:
                self._release_credit(queued[1])
                self.dropped_stale += 1
            else:
                survivors.append(queued)
        for queued in survivors:
            self._queue.put_nowait(queued)
        for token, (event, _, _) in self._delivered.items():
            if event.epoch < new_epoch and token not in self._retired_delivery_tokens:
                # The caller may still hold this copied payload. Cancellation
                # prevents further delivery but cannot infer consumption, so
                # its byte/chunk credit stays charged until explicit ACK (or
                # the owning task's next pull).
                self._retired_delivery_tokens.add(token)
                self.retired_delivered += 1
        self._schedule_credit_wakeup()
        return self.dropped_stale

    @property
    def epoch(self) -> int:
        return self._epoch

    # -- consumer side ------------------------------------------------------
    async def get(self) -> ChunkEvent | None:
        """Acknowledge this task's previous event, then get the next one."""
        owner = asyncio.current_task()
        if owner is not None:
            async with self._credit:
                token = self._task_delivery.get(owner)
                if token is not None:
                    self._acknowledge_locked(token)
        while True:
            queued = await self._queue.get()
            if queued is not None:
                event, size = queued
                # Do not await after dequeuing and before recording delivery:
                # cancellation at such an await would strand its credit. All
                # mutations below are synchronous on this event loop.
                if event.epoch < self._epoch:
                    self._release_credit(size)
                    self.dropped_stale += 1
                    self._schedule_credit_wakeup()
                    continue
                self._delivered[event.release_token] = (event, size, owner)
                if owner is not None:
                    self._task_delivery[owner] = event.release_token
                return event
            # Keep the terminal marker available for another waiter or a
            # later ``get``. It holds no chunk or byte credit.
            self._queue.put_nowait(None)
            return None

    def __aiter__(self) -> BoundedEventStream:
        return self

    async def __anext__(self) -> ChunkEvent:
        event = await self.get()
        if event is None:
            raise StopAsyncIteration
        return event

    def stats(self) -> dict[str, int]:
        return {
            "max_chunks": self.max_chunks,
            "max_bytes": self.max_bytes,
            "high_water_chunks": self.high_water_chunks,
            "high_water_bytes": self.high_water_bytes,
            "dropped_stale": self.dropped_stale,
            "retired_delivered": self.retired_delivered,
            "acknowledged_events": self.acknowledged_events,
            "outstanding_chunks": self._chunks,
            "outstanding_bytes": self._bytes,
            "delivered_unacknowledged": len(self._delivered),
            "epoch": self._epoch,
        }


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]
