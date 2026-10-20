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
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

STATE_LAYOUT_VERSION = 1
"""Bumped when the meaning of a handle's state changes. A handle from a
different version is not migrated, it is rejected: the proposal's item 10 says
to treat state as non-migratable until a specific backend pair has been
measured, and a version that silently matches would hide that."""


@dataclass(frozen=True)
class StateHandle:
    """A reference to session state the backend owns.

    The engine never reads the state. It reads this, to decide whether a
    request may continue a session or must start a new one.
    """

    session_id: str
    backend: str
    artifact_id: str
    """Ties the state to the checkpoint that produced it. The same shapes from
    a different quantization are not the same state."""
    layout_version: int = STATE_LAYOUT_VERSION
    epoch: int = 0
    """Incremented on every cancel. Events from a lower epoch are stale."""
    replayable: bool = True
    """A text session can be rebuilt by re-submitting its committed prompt, so
    recovery after a backend crash is possible at an output boundary."""
    migratable: bool = False
    """Never true in M0. Moving KV between backends needs a measured layout
    conversion for the specific pair; assuming it is the failure the proposal
    warns about."""
    created_unix: float = field(default_factory=time.time)

    def next_epoch(self) -> StateHandle:
        from dataclasses import replace

        return replace(self, epoch=self.epoch + 1)

    def accepts(self, other: StateHandle) -> bool:
        """Whether ``other`` may continue this session's state."""
        return (
            other.session_id == self.session_id
            and other.artifact_id == self.artifact_id
            and other.layout_version == self.layout_version
            and other.epoch == self.epoch
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkEvent:
    """One unit of output, with everything needed to order and retire it."""

    request_id: str
    stage_id: int
    seq: int
    """Monotonic within (request, epoch). Gaps mean loss; repeats mean a bug."""
    epoch: int
    kind: str
    """``token`` | ``done`` | ``error`` | ``cancelled``."""
    payload: Any
    """For ``token``: the incremental text and token ids. Never the whole
    output so far -- a cumulative payload makes a dropped chunk invisible."""
    started_unix: float
    emitted_unix: float
    final: bool = False
    error: str | None = None
    input_watermark: int = 0
    """How much of the input the producer had consumed when this was emitted.
    One number in M0 (prompt tokens); it is what a duplex turn will align on."""
    release_token: str = ""
    """Returned by the consumer to give credit back. Empty when the event
    carries no resource the producer is holding on its behalf."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StreamClosed(RuntimeError):
    """Raised on a stream that has been closed while a consumer waited."""


class BoundedEventStream:
    """An async queue with a credit bound on chunks *and* bytes.

    Producers ``await put(...)``, which blocks while the consumer is behind:
    that is the backpressure. Consumers ``async for`` over it and the credit is
    returned as each event leaves the queue.
    """

    def __init__(self, *, max_chunks: int = 64, max_bytes: int = 4 << 20) -> None:
        if max_chunks < 1 or max_bytes < 1:
            raise ValueError("bounds must be positive")
        self.max_chunks = max_chunks
        self.max_bytes = max_bytes
        self._queue: asyncio.Queue[ChunkEvent | None] = asyncio.Queue()
        self._bytes = 0
        self._chunks = 0
        self._credit = asyncio.Condition()
        self._closed = False
        self._epoch = 0
        self.dropped_stale = 0
        """Events discarded because their epoch had been retired. Reported
        rather than silently swallowed: a non-zero count on a run with no
        cancellation is a bug worth seeing."""
        self.high_water_chunks = 0
        self.high_water_bytes = 0

    # -- producer side ------------------------------------------------------
    @staticmethod
    def _sizeof(event: ChunkEvent) -> int:
        payload = event.payload
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return len(payload)
        if isinstance(payload, str):
            return len(payload.encode("utf-8"))
        if isinstance(payload, dict):
            return sum(
                len(str(k)) + (len(v.encode("utf-8")) if isinstance(v, str) else 8 * (len(v) if isinstance(v, (list, tuple)) else 1))
                for k, v in payload.items()
            )
        return 64

    async def put(self, event: ChunkEvent) -> None:
        """Enqueue, waiting for credit. Stale-epoch events are dropped here."""
        if self._closed:
            raise StreamClosed("stream is closed")
        if event.epoch < self._epoch:
            self.dropped_stale += 1
            return
        size = self._sizeof(event)
        async with self._credit:
            await self._credit.wait_for(
                lambda: self._closed
                or (self._chunks < self.max_chunks and self._bytes + size <= self.max_bytes)
                # A single event larger than the whole byte bound must still go
                # through once the queue is empty, or it deadlocks forever.
                or (self._chunks == 0 and size > self.max_bytes)
            )
            if self._closed:
                raise StreamClosed("stream closed while producing")
            self._chunks += 1
            self._bytes += size
            self.high_water_chunks = max(self.high_water_chunks, self._chunks)
            self.high_water_bytes = max(self.high_water_bytes, self._bytes)
        await self._queue.put(event)

    async def close(self) -> None:
        """End the stream. Idempotent; wakes every waiter."""
        if self._closed:
            return
        self._closed = True
        async with self._credit:
            self._credit.notify_all()
        await self._queue.put(None)

    def retire_epoch(self, epoch: int) -> int:
        """Retire everything at or below ``epoch``; return what was discarded.

        Called by cancel. Queued events from the retired epoch are dropped on
        the way out rather than scanned here, so this stays O(1) and cannot
        race a producer mid-``put``.
        """
        self._epoch = max(self._epoch, epoch + 1)
        return self.dropped_stale

    @property
    def epoch(self) -> int:
        return self._epoch

    # -- consumer side ------------------------------------------------------
    async def get(self) -> ChunkEvent | None:
        """Next live event, or ``None`` at end of stream."""
        while True:
            event = await self._queue.get()
            size = 0 if event is None else self._sizeof(event)
            if event is not None:
                async with self._credit:
                    self._chunks -= 1
                    self._bytes -= size
                    self._credit.notify_all()
                if event.epoch < self._epoch:
                    self.dropped_stale += 1
                    continue
            return event

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
            "epoch": self._epoch,
        }


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]
