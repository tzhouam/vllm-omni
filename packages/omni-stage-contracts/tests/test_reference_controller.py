"""Host conformance for the mobile/embedded stage-controller boundary."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from omni_stage_contracts import (
    ReferenceStageController,
    StageCancelTimeout,
    StageEvent,
    StageRequest,
    StateHandle,
)


FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "conformance/v2.json").read_text()
)
STATE = StateHandle(**FIXTURE["request"]["state"])


def _request(request_id: str = "request-2", *, state: StateHandle | None = None) -> StageRequest:
    return StageRequest(
        request_id=request_id, stage_id=3, epoch=7,
        worker_generation="worker-A", artifact_id="sha256:artifact-A",
        state_layout_version=2, session_id="session-2", state=state,
    )


def _event(
    request_id: str, seq: int, *, nbytes: int = 0,
    state: StateHandle | None = None, terminal: bool = False,
) -> StageEvent:
    return StageEvent(
        request_id=request_id, stage_id=3, epoch=7, seq=seq,
        kind="done" if terminal else "chunk", worker_generation="worker-A",
        state=state, terminal=terminal, started_monotonic_ns=100 + seq,
        emitted_monotonic_ns=101 + seq, input_watermark=seq,
        payload_nbytes=nbytes, release_token=f"{request_id}-ack-{seq}",
    )


class FixtureBackend:
    backend = "qnn"
    instance_id = "device-process-A"
    worker_generation = "worker-A"

    def __init__(self) -> None:
        self.cancels: list[StageRequest] = []
        self.releases: list[StageRequest] = []

    async def run(self, request: StageRequest, emit) -> None:
        if request.request_id == "seed":
            await emit(_event("seed", 1, state=STATE, terminal=True))
        else:
            for raw in FIXTURE["events"]:
                await emit(StageEvent.from_dict(raw))

    async def cancel(self, request: StageRequest) -> None:
        self.cancels.append(request)

    async def release(self, request: StageRequest) -> None:
        self.releases.append(request)


def test_reference_controller_replays_v2_fixture_and_explicitly_releases_state():
    async def exercise():
        backend = FixtureBackend()
        controller = ReferenceStageController({3: backend}, **FIXTURE["stream_limits"])
        seed = controller.start(_request("seed"))
        await seed.wait()
        produced = await seed.receive()
        assert produced is not None and produced.state == STATE
        with pytest.raises(RuntimeError, match="active request"):
            controller.start(StageRequest.from_dict(FIXTURE["request"]))
        assert await seed.acknowledge(produced)
        assert await seed.receive() is None

        run = controller.start(StageRequest.from_dict(FIXTURE["request"]))
        await run.wait()
        with pytest.raises(RuntimeError, match="active request"):
            controller.start(_request("next-while-unacked"))
        for raw in FIXTURE["events"]:
            event = await run.receive()
            assert event is not None and event.seq == raw["seq"]
            assert await run.acknowledge(event)
        assert await run.receive() is None
        assert run.stream.high_water_events == 2
        assert run.stream.high_water_bytes == 5
        assert run.state == STATE

        await controller.release(3, STATE)
        assert len(backend.releases) == 1
        assert backend.releases[0].operation == "release"
        assert backend.releases[0].state == STATE
        with pytest.raises(ValueError, match="not live"):
            controller.start(_request("after-release", state=STATE))

    asyncio.run(exercise())


def test_controller_propagates_ack_credit_without_exceeding_byte_bound():
    class ThreeChunkBackend(FixtureBackend):
        async def run(self, request: StageRequest, emit) -> None:
            await emit(_event(request.request_id, 1, nbytes=3))
            await emit(_event(request.request_id, 2, nbytes=2))
            await emit(_event(request.request_id, 3, nbytes=1, terminal=True))

    async def exercise():
        controller = ReferenceStageController({3: ThreeChunkBackend()}, max_events=2, max_bytes=5)
        run = controller.start(_request())
        first = await asyncio.wait_for(run.receive(), 1)
        second = await asyncio.wait_for(run.receive(), 1)
        assert (first.seq, second.seq) == (1, 2)
        waiter = asyncio.create_task(run.wait())
        await asyncio.sleep(0)
        assert not waiter.done()
        assert run.stream.outstanding_bytes == 5
        assert run.stream.high_water_bytes <= 5
        assert await run.acknowledge(first)
        await asyncio.wait_for(waiter, 1)
        third = await run.receive()
        assert third is not None and third.seq == 3 and third.terminal
        assert await run.receive() is None
        assert await run.acknowledge(second)
        assert await run.acknowledge(third)
        assert run.stream.outstanding_bytes == 0

    asyncio.run(exercise())


def test_cancel_retires_state_and_release_waits_for_delivered_payload_ack():
    class StallingBackend(FixtureBackend):
        async def run(self, request: StageRequest, emit) -> None:
            await emit(_event(request.request_id, 1, nbytes=3, state=STATE))
            await emit(_event(request.request_id, 2, nbytes=3, terminal=True))

    async def exercise():
        backend = StallingBackend()
        controller = ReferenceStageController({3: backend}, max_events=1, max_bytes=3)
        run = controller.start(_request())
        delivered = await asyncio.wait_for(run.receive(), 1)
        assert delivered is not None and delivered.state == STATE
        await run.cancel()
        assert backend.cancels and backend.cancels[0].operation == "cancel"
        assert backend.cancels[0].state == STATE
        assert run.stream.outstanding_bytes == 3
        assert await run.receive() is None
        with pytest.raises(RuntimeError, match="active request"):
            controller.start(_request("stale-continuation", state=STATE))
        with pytest.raises(RuntimeError, match="unacknowledged"):
            await controller.release(3, STATE)
        assert await run.acknowledge(delivered)
        with pytest.raises(ValueError, match="not live"):
            controller.start(_request("stale-continuation", state=STATE))
        await controller.release(3, STATE)
        assert len(backend.releases) == 1

    asyncio.run(exercise())


def test_controller_refuses_wrong_backend_state_and_missing_terminal():
    class BadStateBackend(FixtureBackend):
        async def run(self, request: StageRequest, emit) -> None:
            await emit(_event(request.request_id, 1,
                              state=replace(STATE, backend_instance_id="other"),
                              terminal=True))

    class NoTerminalBackend(FixtureBackend):
        async def run(self, request: StageRequest, emit) -> None:
            await emit(_event(request.request_id, 1))

    class OutOfOrderBackend(FixtureBackend):
        async def run(self, request: StageRequest, emit) -> None:
            await emit(_event(request.request_id, 2, terminal=True))

    class FailedAfterStateBackend(FixtureBackend):
        async def run(self, request: StageRequest, emit) -> None:
            await emit(_event(request.request_id, 1, state=STATE))
            raise RuntimeError("backend reset")

    async def exercise():
        wrong = ReferenceStageController({3: BadStateBackend()}, max_events=2, max_bytes=5)
        run = wrong.start(_request())
        with pytest.raises(ValueError, match="different backend instance"):
            await run.wait()
        with pytest.raises(ValueError, match="different backend instance"):
            await run.receive()

        incomplete = ReferenceStageController({3: NoTerminalBackend()}, max_events=2, max_bytes=5)
        run = incomplete.start(_request())
        first = await run.receive()
        assert first is not None and first.seq == 1
        assert await run.acknowledge(first)
        with pytest.raises(RuntimeError, match="without a terminal"):
            await run.wait()
        with pytest.raises(RuntimeError, match="without a terminal"):
            await run.receive()

        out_of_order = ReferenceStageController({3: OutOfOrderBackend()}, max_events=2, max_bytes=5)
        run = out_of_order.start(_request())
        with pytest.raises(ValueError, match="sequence gap"):
            await run.wait()
        with pytest.raises(ValueError, match="sequence gap"):
            await run.receive()

        failed = ReferenceStageController({3: FailedAfterStateBackend()}, max_events=2, max_bytes=5)
        run = failed.start(_request())
        event = await run.receive()
        assert event is not None and event.state == STATE
        with pytest.raises(RuntimeError, match="backend reset"):
            await run.wait()
        assert await run.acknowledge(event)
        with pytest.raises(ValueError, match="not live"):
            failed.start(_request("after-reset", state=STATE))

    asyncio.run(exercise())


def test_cancel_timeout_retains_active_request_and_state_until_adapter_stops():
    class UnresponsiveBackend(FixtureBackend):
        def __init__(self) -> None:
            super().__init__()
            self.stop = asyncio.Event()

        async def run(self, request: StageRequest, emit) -> None:
            await emit(_event(request.request_id, 1, nbytes=3, state=STATE))
            await self.stop.wait()

    async def exercise():
        backend = UnresponsiveBackend()
        controller = ReferenceStageController(
            {3: backend}, max_events=1, max_bytes=3, cancel_timeout_s=0.01
        )
        run = controller.start(_request())
        delivered = await run.receive()
        assert delivered is not None
        with pytest.raises(StageCancelTimeout, match="did not stop"):
            await run.cancel()
        assert run.cancel_unresolved
        assert backend.cancels and backend.cancels[0].operation == "cancel"
        with pytest.raises(RuntimeError, match="active request"):
            controller.start(_request("cannot-overlap"))
        with pytest.raises(RuntimeError, match="active request"):
            await controller.release(3, STATE)
        backend.stop.set()
        await run.wait()
        assert not run.cancel_unresolved
        assert await run.acknowledge(delivered)
        await controller.release(3, STATE)
        assert len(backend.releases) == 1

    asyncio.run(exercise())


def test_release_is_single_flight_and_blocks_new_backend_work():
    class SlowReleaseBackend(FixtureBackend):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.finish = asyncio.Event()

        async def release(self, request: StageRequest) -> None:
            self.started.set()
            await self.finish.wait()
            await super().release(request)

    async def exercise():
        backend = SlowReleaseBackend()
        controller = ReferenceStageController({3: backend}, max_events=1, max_bytes=5)
        seed = controller.start(_request("seed"))
        await seed.wait()
        event = await seed.receive()
        assert event is not None and await seed.acknowledge(event)

        pending = asyncio.create_task(controller.release(3, STATE))
        await backend.started.wait()
        with pytest.raises(RuntimeError, match="already in progress"):
            await controller.release(3, STATE)
        with pytest.raises(RuntimeError, match="release is in progress"):
            controller.start(_request("overlap"))
        backend.finish.set()
        await pending
        assert len(backend.releases) == 1

    asyncio.run(exercise())


def test_release_retry_keeps_operation_identity_until_adapter_ack():
    class UncertainReleaseBackend(FixtureBackend):
        async def release(self, request: StageRequest) -> None:
            self.releases.append(request)
            if len(self.releases) == 1:
                raise RuntimeError("release acknowledgement lost")

    async def exercise():
        backend = UncertainReleaseBackend()
        controller = ReferenceStageController({3: backend}, max_events=1, max_bytes=5)
        seed = controller.start(_request("seed"))
        await seed.wait()
        event = await seed.receive()
        assert event is not None and await seed.acknowledge(event)

        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            await controller.release(3, STATE)
        await controller.release(3, STATE)
        assert len(backend.releases) == 2
        assert backend.releases[0].request_id == backend.releases[1].request_id

    asyncio.run(exercise())
