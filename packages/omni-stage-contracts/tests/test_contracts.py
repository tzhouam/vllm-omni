"""Contract tests that also run without importing Omni or an accelerator SDK."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import struct
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from omni_stage_contracts import (
    ArtifactManifest,
    BoundedStageEventStream,
    StageEvent,
    StageRequest,
    StageStreamClosed,
    StateHandle,
    negotiate,
)


FIXTURES = Path(__file__).resolve().parents[1] / "conformance"


def _state(**overrides):
    values = dict(
        session_id="session", backend="qnn", artifact_id="sha256:checkpoint",
        layout_version=3, epoch=4, backend_instance_id="device-process",
        worker_generation="worker-a", state_id="opaque-123",
    )
    values.update(overrides)
    return StateHandle(**values)


def _event(seq, *, nbytes=3, token=None, state=None, terminal=False, epoch=4, watermark=0):
    now = time.monotonic_ns()
    return StageEvent(
        "request", 2, epoch, seq, "pcm", "worker-a", terminal=terminal,
        state=state, started_monotonic_ns=now, emitted_monotonic_ns=now + 1,
        payload_nbytes=nbytes, release_token=token or f"release-{seq}",
        input_watermark=watermark,
    )


def _request(state=None):
    return StageRequest(
        "request", 2, 4, "worker-a", state=state,
        artifact_id=state.artifact_id if state else "",
        state_layout_version=state.layout_version if state else 0,
        session_id=state.session_id if state else "",
    )


def test_persistent_state_is_bound_to_worker_artifact_layout_and_epoch():
    state = _state()
    request = _request(state)
    assert request.state == state
    assert StageRequest.release("cleanup", 2, state).operation == "release"
    assert not state.accepts(_state(state_id="other"))
    assert not state.accepts(_state(replayable=True))
    assert not state.accepts(_state(migratable=True))
    for stale in (
        _state(epoch=3), _state(worker_generation="worker-b"),
    ):
        with pytest.raises(ValueError):
            StageRequest("request", 2, 4, "worker-a", state=stale,
                         artifact_id=state.artifact_id, state_layout_version=state.layout_version,
                         session_id=state.session_id)
    for stale in (_state(artifact_id="sha256:other"), _state(layout_version=2)):
        with pytest.raises(ValueError, match="artifact or layout"):
            StageRequest("request", 2, 4, "worker-a", state=stale,
                         artifact_id=state.artifact_id, state_layout_version=state.layout_version,
                         session_id=state.session_id)
    with pytest.raises(ValueError, match="opaque ID"):
        _request(_state(state_id=""))
    with pytest.raises(ValueError, match="requires a state handle"):
        StageRequest("request", 2, 4, "worker-a", operation="release")
    with pytest.raises(ValueError, match="state session"):
        StageRequest("request", 2, 4, "worker-a", state=state,
                     artifact_id=state.artifact_id, state_layout_version=state.layout_version,
                     session_id="other-session")


def test_event_timestamps_and_state_are_validated():
    event = _event(1, state=_state())
    assert asdict(event)["state"]["state_id"] == "opaque-123"
    with pytest.raises(ValueError, match="precedes start"):
        replace(event, emitted_monotonic_ns=event.started_monotonic_ns - 1)
    with pytest.raises(ValueError, match="worker generation"):
        replace(event, state=_state(worker_generation="worker-b"))


def test_native_conformance_fixture_round_trip_and_rejection():
    fixture = json.loads((FIXTURES / "v2.json").read_text())
    negotiate(fixture["protocol_version"], fixture["required_features"])
    negotiate(1, ["host-copy", "request-epochs"])
    with pytest.raises(ValueError, match="unsupported required stage features"):
        negotiate(1, ["backend-state"])
    with pytest.raises(ValueError, match="unsupported stage protocol version"):
        negotiate(3)
    request = StageRequest.from_dict(fixture["request"])
    assert StageRequest.from_dict(request.to_dict()) == request

    async def exercise():
        stream = BoundedStageEventStream(request, **fixture["stream_limits"])
        for raw in fixture["events"]:
            event = StageEvent.from_dict(raw)
            assert StageEvent.from_dict(event.to_dict()) == event
            await stream.submit(event)
        for raw in fixture["events"]:
            event = await stream.receive()
            assert event.seq == raw["seq"]
            assert await stream.acknowledge(event.release_token)
        assert await stream.receive() is None
        for override in fixture["reject_event_overrides"]:
            fresh = BoundedStageEventStream(request, **fixture["stream_limits"])
            event = StageEvent.from_dict({**fixture["events"][0], **override})
            with pytest.raises(ValueError):
                await fresh.submit(event)
    asyncio.run(exercise())


def test_stateless_graph_wire_rejects_v2_contract_frames():
    pytest.importorskip("numpy")
    from omni_stage_contracts.wire import ProtocolError, recv_message

    sender, receiver = socket.socketpair()
    try:
        header = json.dumps({"version": 2, "required_features": ["backend-state"],
                             "op": "run", "body": {}, "tensors": []}).encode()
        sender.sendall(struct.pack(">I", len(header)) + header)
        with pytest.raises(ProtocolError, match="unsupported graph wire protocol version"):
            recv_message(receiver)
    finally:
        sender.close()
        receiver.close()


def test_v2_artifact_requires_hashed_provenance_and_validation(tmp_path):
    for name in ("graph.onnx", "calibration.json"):
        (tmp_path / name).write_text(name)
    descriptor = {
        "checkpoint_id": "org/model", "checkpoint_revision": "fixed-commit",
        "precision": "w8a16", "runtime": "QNN", "runtime_version": "2.x",
        "target_abi": "android-arm64", "adapter_version": "spark:1",
        "exporter_version": "omni-export:1", "compiler_version": "qnn-compile:2.x",
        "state_layout_version": 3,
        "shape_bucket": "prefill-512", "calibration_file": "calibration.json",
        "numerical_validation": {"file": "numeric.json", "passed": True},
        "task_validation": {"file": "task.json", "passed": True},
    }
    def write_report(kind, *, passing=True):
        report = {
            "schema_version": 1, "kind": kind,
            "checkpoint_id": descriptor["checkpoint_id"],
            "checkpoint_revision": descriptor["checkpoint_revision"],
            "precision": descriptor["precision"],
            "shape_bucket": descriptor["shape_bucket"],
            "target_abi": descriptor["target_abi"],
            "checks": [{"name": "reference_error" if kind == "numerical" else "task_score",
                        "observed": 0.01 if passing else 0.1,
                        "comparison": "<=", "limit": 0.05}],
        }
        filename = "numeric.json" if kind == "numerical" else "task.json"
        (tmp_path / filename).write_text(json.dumps(report))

    write_report("numerical")
    write_report("task")
    files = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
             for name in ("graph.onnx", "numeric.json", "task.json", "calibration.json")}
    payload = {"schema_version": 2, "component": "spark-prefill", "files": files,
               "metadata": {"artifact": descriptor}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    manifest = ArtifactManifest.read(path)
    assert manifest.artifact_metadata().qualified
    assert not ArtifactManifest(**payload).artifact_metadata().qualified  # unverified construction
    descriptor["task_validation"]["passed"] = False
    write_report("task", passing=False)
    files["task.json"] = hashlib.sha256((tmp_path / "task.json").read_bytes()).hexdigest()
    path.write_text(json.dumps(payload))
    assert not ArtifactManifest.read(path).artifact_metadata().qualified
    descriptor["task_validation"]["passed"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differs from verified report"):
        ArtifactManifest.read(path)
    write_report("task")
    files["task.json"] = hashlib.sha256((tmp_path / "task.json").read_bytes()).hexdigest()
    (tmp_path / "numeric.json").write_text("{}")
    files["numeric.json"] = hashlib.sha256((tmp_path / "numeric.json").read_bytes()).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported schema"):
        ArtifactManifest.read(path)
    write_report("numerical")
    files["numeric.json"] = hashlib.sha256((tmp_path / "numeric.json").read_bytes()).hexdigest()
    descriptor["numerical_validation"]["file"] = "unhashed.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hashed file"):
        ArtifactManifest.read(path)


def test_stream_backpressure_waits_for_ack_and_never_exceeds_byte_bound():
    async def exercise():
        request = _request(_state())
        stream = BoundedStageEventStream(request, max_events=2, max_bytes=5)
        await stream.submit(_event(1, nbytes=3, state=_state()))
        first = await stream.receive()
        assert first.seq == 1
        assert not await stream.acknowledge("unknown")
        await stream.submit(_event(2, nbytes=2))
        third = asyncio.create_task(stream.submit(_event(3, nbytes=1, terminal=True)))
        await asyncio.sleep(0)
        assert not third.done()
        assert stream.high_water_bytes == 5
        assert await stream.acknowledge(first.release_token)
        await asyncio.wait_for(third, 1)
        second = await stream.receive()
        final = await stream.receive()
        assert (second.seq, final.seq) == (2, 3)
        assert await stream.receive() is None
        assert await stream.acknowledge(second.release_token)
        assert await stream.acknowledge(final.release_token)
        assert not await stream.acknowledge(final.release_token)
        assert stream.outstanding_events == 0
        assert stream.outstanding_bytes == 0
        assert stream.high_water_bytes <= stream.max_bytes
    asyncio.run(exercise())


def test_stream_refuses_oversize_duplicate_out_of_order_and_wrong_state():
    async def exercise():
        request = _request(_state())
        stream = BoundedStageEventStream(request, max_events=1, max_bytes=5)
        with pytest.raises(ValueError, match="byte bound"):
            await stream.submit(_event(1, nbytes=6))
        with pytest.raises(ValueError, match="sequence"):
            await stream.submit(_event(2))
        with pytest.raises(ValueError, match="different backend state"):
            await stream.submit(_event(1, state=_state(state_id="wrong")))
        with pytest.raises(ValueError, match="epoch mismatch"):
            await stream.submit(_event(1, epoch=3))
        await stream.submit(_event(1, token="one"))
        assert not await stream.acknowledge("one")  # not yet delivered
        first = await stream.receive()
        assert await stream.acknowledge(first.release_token)
        with pytest.raises(ValueError, match="duplicate event acknowledgement token"):
            await stream.submit(_event(2, token="one"))
        with pytest.raises(ValueError, match="sequence"):
            await stream.submit(_event(1, token="another"))
        await stream.submit(_event(2, token="two", watermark=5))
        with pytest.raises(ValueError, match="watermark regressed"):
            await stream.submit(_event(3, token="three", watermark=4))
        await stream.cancel()
    asyncio.run(exercise())


def test_first_event_state_requires_request_session_binding():
    async def exercise():
        unbound = BoundedStageEventStream(_request(), max_events=1, max_bytes=5)
        with pytest.raises(ValueError, match="unbound session"):
            await unbound.submit(_event(1, state=_state()))
        bound = BoundedStageEventStream(
            StageRequest("request", 2, 4, "worker-a", session_id="session",
                         artifact_id="sha256:checkpoint", state_layout_version=3),
            max_events=1, max_bytes=5,
        )
        with pytest.raises(ValueError, match="another or unbound session"):
            await bound.submit(_event(1, state=_state(session_id="other")))
        with pytest.raises(ValueError, match="another artifact or layout"):
            await bound.submit(_event(1, state=_state(artifact_id="sha256:other")))
        with pytest.raises(ValueError, match="another artifact or layout"):
            await bound.submit(_event(1, state=_state(layout_version=2)))
        await bound.submit(_event(1, state=_state()))
        event = await bound.receive()
        assert bound.state.accepts(_state())
        assert await bound.acknowledge(event.release_token)
    asyncio.run(exercise())


def test_cancel_fences_queued_events_and_wakes_blocked_producer():
    async def exercise():
        stream = BoundedStageEventStream(StageRequest("request", 2, 4, "worker-a"),
                                         max_events=1, max_bytes=3)
        await stream.submit(_event(1))
        producer = asyncio.create_task(stream.submit(_event(2, terminal=True)))
        await asyncio.sleep(0)
        assert not producer.done()
        assert await stream.cancel() == 5
        with pytest.raises(StageStreamClosed):
            await asyncio.wait_for(producer, 1)
        assert stream.dropped_stale == 1
        assert stream.outstanding_events == 0
        assert await stream.receive() is None
        with pytest.raises(StageStreamClosed):
            await stream.submit(_event(1))
    asyncio.run(exercise())


def test_cancel_keeps_delivered_payload_charged_until_ack():
    async def exercise():
        stream = BoundedStageEventStream(_request(), max_events=1, max_bytes=3)
        await stream.submit(_event(1))
        delivered = await stream.receive()
        await stream.cancel()
        assert stream.outstanding_events == 1
        assert stream.outstanding_bytes == 3
        assert await stream.acknowledge(delivered.release_token)
        assert stream.outstanding_bytes == 0
        assert await stream.receive() is None
    asyncio.run(exercise())


def test_cancel_after_producer_close_still_discards_queued_epoch():
    async def exercise():
        stream = BoundedStageEventStream(_request(), max_events=2, max_bytes=6)
        await stream.submit(_event(1))
        await stream.close()
        assert stream.outstanding_events == 1
        assert await stream.cancel() == 5
        assert await stream.cancel() == 5
        assert stream.outstanding_events == 0
        assert stream.dropped_stale == 1
        assert await stream.receive() is None

    asyncio.run(exercise())


def test_dequeue_registers_delivery_without_waiting_for_credit_lock():
    async def exercise():
        stream = BoundedStageEventStream(_request(), max_events=1, max_bytes=3)
        await stream.submit(_event(1))
        async with stream._credit:
            receiving = asyncio.create_task(stream.receive())
            await asyncio.sleep(0)
            # If receive awaited the credit lock after dequeue, cancelling it
            # here would strand the event's admitted byte credit forever.
            assert receiving.done()
            delivered = receiving.result()
            assert delivered is not None and delivered.seq == 1
        assert await stream.acknowledge(delivered.release_token)
        assert stream.outstanding_bytes == 0

    asyncio.run(exercise())


def test_terminal_marker_survives_consumer_cancellation_and_wakes_peers():
    async def exercise():
        stream = BoundedStageEventStream(_request(), max_events=1, max_bytes=3)
        await stream.close()
        async with stream._credit:
            first = asyncio.create_task(stream.receive())
            second = asyncio.create_task(stream.receive())
            await asyncio.sleep(0)
            first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await asyncio.wait_for(second, 1) is None
        assert await stream.receive() is None

    asyncio.run(exercise())
