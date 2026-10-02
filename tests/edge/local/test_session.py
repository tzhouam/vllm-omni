# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The event stream: backpressure that blocks, and an epoch that fences."""

import asyncio
from dataclasses import replace

import pytest

from vllm_omni.edge.local.session import (
    STATE_LAYOUT_VERSION,
    BoundedEventStream,
    ChunkEvent,
    EventTooLarge,
    StateHandle,
    StreamClosed,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _event(seq: int, *, epoch: int = 0, text: str = "x", kind: str = "token") -> ChunkEvent:
    return ChunkEvent(
        request_id="r", stage_id=0, seq=seq, epoch=epoch, kind=kind,
        payload={"text": text}, started_unix=0.0, emitted_unix=0.0,
    )


# -------------------------------------------------------------- state handle
def test_a_handle_carries_the_artifact_it_was_made_for():
    handle = StateHandle(session_id="s", backend="vllm:cuda", artifact_id="abc")
    assert handle.layout_version == STATE_LAYOUT_VERSION
    assert handle.accepts(handle)


def test_state_from_another_checkpoint_is_not_accepted():
    """Same shapes from a different quantization are not the same state."""
    a = StateHandle(session_id="s", backend="vllm:cuda", artifact_id="abc")
    b = StateHandle(session_id="s", backend="vllm:cuda", artifact_id="def")
    assert not a.accepts(b)


def test_state_from_a_retired_epoch_is_not_accepted():
    handle = StateHandle(session_id="s", backend="vllm:cuda", artifact_id="abc")
    assert not handle.next_epoch().accepts(handle)


def test_state_is_never_migratable_in_m0():
    handle = StateHandle(session_id="s", backend="vllm:cuda", artifact_id="abc")
    assert handle.migratable is False
    assert handle.replayable is True
    assert not handle.accepts(replace(handle, migratable=True))
    assert not handle.accepts(replace(handle, replayable=False))


# ---------------------------------------------------------------- ordering
async def test_events_arrive_in_order():
    stream = BoundedEventStream()
    for i in range(5):
        await stream.put(_event(i))
    await stream.close()
    assert [e.seq async for e in stream] == [0, 1, 2, 3, 4]


async def test_closing_ends_the_iteration():
    stream = BoundedEventStream()
    await stream.put(_event(0))
    await stream.close()
    seen = [e async for e in stream]
    assert len(seen) == 1


async def test_producing_after_close_raises():
    stream = BoundedEventStream()
    await stream.close()
    with pytest.raises(StreamClosed):
        await stream.put(_event(0))


# ------------------------------------------------------------ backpressure
async def test_a_full_queue_blocks_the_producer():
    """The bound has to actually stop the producer. A queue that grows instead
    is how a slow consumer turns into an OOM."""
    stream = BoundedEventStream(max_chunks=2, max_bytes=1 << 20)
    await stream.put(_event(0))
    await stream.put(_event(1))
    blocked = asyncio.create_task(stream.put(_event(2)))
    await asyncio.sleep(0.05)
    assert not blocked.done()
    delivered = await stream.get()
    assert delivered is not None
    await asyncio.sleep(0.05)
    assert not blocked.done(), "dequeue must not release credit before handling ends"
    assert await stream.acknowledge(delivered)
    await asyncio.wait_for(blocked, timeout=1.0)
    assert stream.high_water_chunks == 2
    assert stream.stats()["outstanding_chunks"] == 2


async def test_the_byte_bound_also_blocks():
    stream = BoundedEventStream(max_chunks=100, max_bytes=64)
    await stream.put(_event(0, text="a" * 40))
    blocked = asyncio.create_task(stream.put(_event(1, text="b" * 40)))
    await asyncio.sleep(0.05)
    assert not blocked.done()
    delivered = await stream.get()
    assert delivered is not None
    await asyncio.sleep(0.05)
    assert not blocked.done()
    assert await stream.acknowledge(delivered.release_token)
    await asyncio.wait_for(blocked, timeout=1.0)


async def test_async_iteration_auto_acknowledges_on_next_pull_with_one_credit():
    """A one-slot stream must progress without explicit profiler call-site ACKs."""
    stream = BoundedEventStream(max_chunks=1, max_bytes=64)
    first_handled = asyncio.Event()
    release_first = asyncio.Event()
    producer_progress = asyncio.Event()

    async def produce() -> None:
        await stream.put(_event(0))
        await stream.put(_event(1))
        producer_progress.set()
        await stream.put(_event(2))
        await stream.close()

    async def consume() -> list[int]:
        seen = []
        async for event in stream:
            seen.append(event.seq)
            if event.seq == 0:
                first_handled.set()
                await release_first.wait()
        return seen

    producer = asyncio.create_task(produce())
    consumer = asyncio.create_task(consume())
    await asyncio.wait_for(first_handled.wait(), timeout=1.0)
    assert not producer_progress.is_set()
    assert stream.stats()["outstanding_chunks"] == 1
    release_first.set()
    assert await asyncio.wait_for(consumer, timeout=1.0) == [0, 1, 2]
    await asyncio.wait_for(producer, timeout=1.0)
    assert stream.stats()["outstanding_chunks"] == 0
    assert stream.stats()["acknowledged_events"] == 3


async def test_explicit_ack_is_idempotent_and_requires_the_delivery_token():
    stream = BoundedEventStream(max_chunks=1, max_bytes=64)
    event = _event(0)
    await stream.put(event)
    delivered = await stream.get()
    assert delivered is not None
    assert delivered.release_token and delivered.release_token != event.release_token
    assert not await stream.acknowledge(event)
    assert await stream.acknowledge(delivered)
    assert not await stream.acknowledge(delivered)
    assert stream.stats()["outstanding_chunks"] == 0


async def test_wait_for_wrapper_does_not_ack_before_caller_handles_event():
    stream = BoundedEventStream(max_chunks=1, max_bytes=64)
    await stream.put(_event(0))
    delivered = await asyncio.wait_for(stream.get(), timeout=1.0)
    assert delivered is not None
    # wait_for's internal get task has exited, but handling has not finished.
    assert stream.stats()["outstanding_chunks"] == 1
    blocked = asyncio.create_task(stream.put(_event(1)))
    await asyncio.sleep(0.05)
    assert not blocked.done()
    assert await stream.acknowledge(delivered)
    await asyncio.wait_for(blocked, timeout=1.0)
    await stream.close()
    assert [event.seq async for event in stream] == [1]
    assert await stream.get() is None


async def test_an_oversized_event_is_rejected_even_on_an_empty_queue():
    """A producer must split PCM/text before it can exceed the byte credit."""
    stream = BoundedEventStream(max_chunks=4, max_bytes=8)
    with pytest.raises(EventTooLarge, match="fragment"):
        await asyncio.wait_for(stream.put(_event(0, text="x" * 400)), timeout=1.0)
    assert stream.high_water_bytes == 0


async def test_nested_binary_payload_is_counted_and_snapshotted():
    stream = BoundedEventStream(max_chunks=2, max_bytes=64)
    payload = {"pcm": [bytearray(24), memoryview(bytes(24))]}
    event = ChunkEvent(
        request_id="r", stage_id=0, seq=0, epoch=0, kind="audio",
        payload=payload, started_unix=0.0, emitted_unix=0.0,
    )
    await stream.put(event)
    payload["pcm"][0].extend(bytes(200))
    got = await stream.get()
    assert got is not None
    assert len(got.payload["pcm"][0]) == 24
    assert stream.high_water_bytes == len("pcm") + 48
    assert stream._bytes == len("pcm") + 48
    assert await stream.acknowledge(got)
    assert stream._bytes == 0


async def test_unknown_payload_cannot_bypass_byte_accounting():
    stream = BoundedEventStream(max_bytes=1)
    event = ChunkEvent(
        request_id="r", stage_id=0, seq=0, epoch=0, kind="audio",
        payload=object(), started_unix=0.0, emitted_unix=0.0,
    )
    with pytest.raises(TypeError, match="unsupported event payload"):
        await stream.put(event)


async def test_closing_wakes_a_blocked_producer():
    stream = BoundedEventStream(max_chunks=1, max_bytes=1 << 20)
    await stream.put(_event(0))
    blocked = asyncio.create_task(stream.put(_event(1)))
    await asyncio.sleep(0.05)
    await stream.close()
    with pytest.raises(StreamClosed):
        await asyncio.wait_for(blocked, timeout=1.0)
    # Normal close does not discard events that were already accepted.
    assert [event.seq async for event in stream] == [0]


async def test_retirement_keeps_delivered_credit_until_ack_and_fences_late_events():
    stream = BoundedEventStream(max_chunks=2, max_bytes=64)
    await stream.put(_event(0, epoch=0))
    delivered = await stream.get()
    assert delivered is not None
    await stream.put(_event(1, epoch=0))
    stale_waiter = asyncio.create_task(stream.put(_event(2, epoch=0)))
    await asyncio.sleep(0.05)
    assert not stale_waiter.done()

    stream.retire_epoch(0)
    await asyncio.wait_for(stale_waiter, timeout=1.0)
    assert stream.stats()["outstanding_chunks"] == 1
    assert stream.stats()["retired_delivered"] == 1
    assert stream.dropped_stale == 2  # queued plus blocked producer
    assert await stream.acknowledge(delivered)
    assert stream.stats()["outstanding_chunks"] == 0

    await stream.put(_event(0, epoch=1, text="recovered"))
    await stream.close()
    recovered = [event async for event in stream]
    assert [(event.epoch, event.payload["text"]) for event in recovered] == [
        (1, "recovered")
    ]


async def test_cancelled_delivery_still_blocks_new_epoch_until_explicit_ack():
    stream = BoundedEventStream(max_chunks=1, max_bytes=64)
    await stream.put(_event(0, epoch=0))
    held = await stream.get()
    assert held is not None
    stream.retire_epoch(0)
    stream.retire_epoch(0)
    assert stream.stats()["retired_delivered"] == 1
    pending = asyncio.create_task(stream.put(_event(0, epoch=1, text="next")))
    await asyncio.sleep(0)
    assert not pending.done()
    assert stream.stats()["outstanding_chunks"] == 1

    assert await stream.acknowledge(held)
    await asyncio.wait_for(pending, timeout=1.0)
    await stream.close()
    next_event = await stream.get()
    assert next_event is not None and next_event.epoch == 1
    assert await stream.acknowledge(next_event)
    assert await stream.get() is None


async def test_cancelled_stream_wakes_waiting_consumer_without_late_event():
    stream = BoundedEventStream(max_chunks=1, max_bytes=64)
    await stream.put(_event(0))
    delivered = await stream.get()
    assert delivered is not None
    waiting = asyncio.create_task(stream.get())
    await asyncio.sleep(0)
    stream.retire_epoch(0)
    await stream.close()
    assert await asyncio.wait_for(waiting, timeout=1.0) is None
    assert stream.stats()["outstanding_chunks"] == 1
    assert await stream.acknowledge(delivered)
    assert stream.stats()["outstanding_chunks"] == 0


# ------------------------------------------------------------------- epochs
async def test_a_retired_epoch_cannot_be_produced_into():
    stream = BoundedEventStream()
    stream.retire_epoch(0)
    await stream.put(_event(0, epoch=0))
    await stream.close()
    assert [e async for e in stream] == []
    assert stream.dropped_stale == 1


async def test_events_already_queued_from_a_retired_epoch_are_not_delivered():
    """The real cancellation race: the backend had tokens in flight when the
    cancel landed, and they must not reach the consumer."""
    stream = BoundedEventStream()
    for i in range(3):
        await stream.put(_event(i, epoch=0))
    stream.retire_epoch(0)
    await stream.put(_event(0, epoch=1, text="new"))
    await stream.close()
    delivered = [e async for e in stream]
    assert [e.epoch for e in delivered] == [1]
    assert stream.dropped_stale == 3


async def test_retiring_is_monotonic():
    stream = BoundedEventStream()
    stream.retire_epoch(3)
    stream.retire_epoch(0)
    assert stream.epoch == 4


async def test_stats_report_the_bounds_and_what_was_dropped():
    stream = BoundedEventStream(max_chunks=8, max_bytes=256)
    await stream.put(_event(0))
    stats = stream.stats()
    assert stats["max_chunks"] == 8
    assert stats["max_bytes"] == 256
    assert stats["high_water_chunks"] == 1
    assert stats["dropped_stale"] == 0
