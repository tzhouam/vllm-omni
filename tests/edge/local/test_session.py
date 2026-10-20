# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The event stream: backpressure that blocks, and an epoch that fences."""

import asyncio

import pytest

from vllm_omni.edge.local.session import (
    STATE_LAYOUT_VERSION,
    BoundedEventStream,
    ChunkEvent,
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
    await stream.get()          # consumer drains one; credit returns
    await asyncio.wait_for(blocked, timeout=1.0)
    assert stream.high_water_chunks == 2


async def test_the_byte_bound_also_blocks():
    stream = BoundedEventStream(max_chunks=100, max_bytes=64)
    await stream.put(_event(0, text="a" * 40))
    blocked = asyncio.create_task(stream.put(_event(1, text="b" * 40)))
    await asyncio.sleep(0.05)
    assert not blocked.done()
    await stream.get()
    await asyncio.wait_for(blocked, timeout=1.0)


async def test_an_oversized_event_still_gets_through_on_an_empty_queue():
    """Otherwise one chunk larger than the whole bound deadlocks forever."""
    stream = BoundedEventStream(max_chunks=4, max_bytes=8)
    await asyncio.wait_for(stream.put(_event(0, text="x" * 400)), timeout=1.0)
    assert (await stream.get()).seq == 0


async def test_closing_wakes_a_blocked_producer():
    stream = BoundedEventStream(max_chunks=1, max_bytes=1 << 20)
    await stream.put(_event(0))
    blocked = asyncio.create_task(stream.put(_event(1)))
    await asyncio.sleep(0.05)
    await stream.close()
    with pytest.raises(StreamClosed):
        await asyncio.wait_for(blocked, timeout=1.0)


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
