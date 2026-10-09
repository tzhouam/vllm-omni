# SPDX-License-Identifier: Apache-2.0
"""Preparation failure cannot race a real startup thread with resource close."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from benchmarks.edge_agent import native_profile as native
from benchmarks.edge_agent.profile import ProfileRoute
from vllm_omni.edge.agent import native_app


class Startup:
    def __init__(self, error=None):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.error = error
        self.execution_plan = None
        self.close_count = 0

    def start(self):
        self.entered.set()
        try:
            if not self.release.wait(5):
                raise TimeoutError("test did not release startup")
            if self.error is not None:
                raise self.error
            self.execution_plan = {"requested_device": "cpu"}
        finally:
            self.finished.set()

    def close(self):
        assert self.finished.is_set(), "resource close raced the startup thread"
        self.close_count += 1
        return True


class Tools:
    def close(self):
        pass


class Controller:
    def __init__(self, backend):
        self.backends = {"route": backend}
        self.routes = [SimpleNamespace(route_id="route")]
        self.tools = Tools()
        self.limits = SimpleNamespace(max_model_steps=6)
        self.close_count = 0

    def add_listener(self, _listener):
        pass

    def admit(self, _route):
        return SimpleNamespace(admitted=True)

    def close(self):
        for backend in self.backends.values():
            backend.close()
        self.close_count += 1


class Telemetry:
    def __init__(self, error=None):
        self.error = error

    def sample(self):
        if self.error is not None:
            raise self.error
        return {"ram_used_bytes": 10, "vram_used_bytes": 20}


def bridge_fixture(tmp_path, monkeypatch, *, sampler_error=None, loader_error=None):
    backend = Startup(loader_error)
    controller = Controller(backend)
    monkeypatch.setattr(native_app, "build_controller", lambda _path: (controller, {}))
    monkeypatch.setattr(native, "_FixtureForegroundScreen", lambda: object())
    monkeypatch.setattr(native, "ManagedEdgeBrowser", lambda **_kw: object())
    monkeypatch.setattr(native, "ReadOnlyFixtureTools", lambda *_a, **_kw: Tools())
    route = ProfileRoute("route", "fixture", "artifact", "revision", "a" * 64, "Q4", "external.llamacpp.text.v1", "cpu")
    bridge = native.NativeProfileBridge(
        native_config={"routes": [{"route_id": "route"}]},
        config_root=tmp_path / "configs",
        private_root=tmp_path,
        fixture_origin="http://127.0.0.1:1",
        telemetry=Telemetry(sampler_error),
    )
    return bridge, route, backend, controller


async def entered(backend):
    assert await asyncio.to_thread(backend.entered.wait, 2)


async def still_loading(task, backend, controller):
    await asyncio.sleep(0.01)
    assert not backend.finished.is_set()
    assert not task.done(), "preparation unwound before its startup thread returned"
    assert controller.close_count == backend.close_count == 0


@pytest.mark.asyncio
async def test_sampler_failure_waits_for_exact_startup_before_close(tmp_path, monkeypatch):
    failure = RuntimeError("sampler failed")
    bridge, route, backend, controller = bridge_fixture(tmp_path, monkeypatch, sampler_error=failure)
    task = asyncio.create_task(bridge.prepare(route))
    try:
        await entered(backend)
        await still_loading(task, backend, controller)
    finally:
        backend.release.set()
    with pytest.raises(RuntimeError) as caught:
        await task
    assert caught.value is failure
    reference = bridge.startup_samples_evidence
    assert reference["complete"] is False and reference["error_type"] == "RuntimeError"
    assert reference["sample_count"] == reference["attempted_sample_count"] == 0
    assert reference["bytes"] == 0
    assert bridge._startup_journal._file is None
    bridge.close()
    assert backend.finished.is_set() and controller.close_count == backend.close_count == 1


@pytest.mark.asyncio
async def test_cancellation_and_repeated_cancellation_cannot_cancel_loader_wrapper(tmp_path, monkeypatch):
    bridge, route, backend, controller = bridge_fixture(tmp_path, monkeypatch)
    task = asyncio.create_task(bridge.prepare(route))
    try:
        await entered(backend)
        task.cancel("original cancellation")
        await still_loading(task, backend, controller)
        for _ in range(3):
            task.cancel("repeated cancellation")
            await still_loading(task, backend, controller)
    finally:
        backend.release.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("original cancellation",)
    assert bridge.startup_samples_evidence["complete"] is False
    assert bridge.startup_samples_evidence["error_type"] == "CancelledError"
    assert bridge._startup_journal._file is None
    bridge.close()
    assert backend.finished.is_set() and controller.close_count == backend.close_count == 1


@pytest.mark.asyncio
async def test_sampler_error_survives_repeated_cancel_during_drain(tmp_path, monkeypatch):
    failure = RuntimeError("sampler failed first")
    bridge, route, backend, controller = bridge_fixture(tmp_path, monkeypatch, sampler_error=failure)
    task = asyncio.create_task(bridge.prepare(route))
    try:
        await entered(backend)
        for _ in range(3):
            task.cancel("cancel while draining")
            await still_loading(task, backend, controller)
    finally:
        backend.release.set()
    with pytest.raises(RuntimeError) as caught:
        await task
    assert caught.value is failure
    bridge.close()
    assert controller.close_count == backend.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("original", ["sampler", "cancel"])
async def test_loader_failure_is_observed_as_cause_of_original_failure(tmp_path, monkeypatch, original):
    load_error = ValueError("loader failed")
    sample_error = RuntimeError("sampler failed") if original == "sampler" else None
    bridge, route, backend, controller = bridge_fixture(
        tmp_path,
        monkeypatch,
        sampler_error=sample_error,
        loader_error=load_error,
    )
    task = asyncio.create_task(bridge.prepare(route))
    try:
        await entered(backend)
        if original == "cancel":
            task.cancel("original cancellation")
        await still_loading(task, backend, controller)
    finally:
        backend.release.set()
    expected = RuntimeError if original == "sampler" else asyncio.CancelledError
    with pytest.raises(expected) as caught:
        await task
    assert caught.value.__cause__ is load_error
    if sample_error is not None:
        assert caught.value is sample_error
    else:
        assert caught.value.args == ("original cancellation",)
    bridge.close()
    assert backend.finished.is_set() and controller.close_count == backend.close_count == 1


@pytest.mark.asyncio
async def test_loader_failure_alone_is_preserved_without_self_chaining(tmp_path, monkeypatch):
    failure = ValueError("loader failed alone")
    bridge, route, backend, controller = bridge_fixture(tmp_path, monkeypatch, loader_error=failure)
    task = asyncio.create_task(bridge.prepare(route))
    await entered(backend)
    backend.release.set()
    with pytest.raises(ValueError) as caught:
        await task
    assert caught.value is failure and caught.value.__cause__ is None
    bridge.close()
    assert controller.close_count == backend.close_count == 1


@pytest.mark.asyncio
async def test_success_still_waits_for_resident_startup_and_closes_once(tmp_path, monkeypatch):
    bridge, route, backend, controller = bridge_fixture(tmp_path, monkeypatch)
    task = asyncio.create_task(bridge.prepare(route))
    try:
        await entered(backend)
        await still_loading(task, backend, controller)
    finally:
        backend.release.set()
    preparation = await task
    assert preparation.cold_start_confirmed
    assert preparation.details["execution_plan"] == {"requested_device": "cpu"}
    assert preparation.details["startup_samples"] == bridge.startup_samples_evidence
    assert bridge.startup_samples_evidence["complete"] is True
    assert bridge.startup_samples_evidence["sample_count"] == preparation.details["sample_count"]
    assert "bound_process_cpu_load_samples" not in preparation.details
    bridge.close()
    assert controller.close_count == backend.close_count == 1


@pytest.mark.asyncio
async def test_sidecar_append_failure_drains_loader_before_file_close(tmp_path, monkeypatch):
    bridge, route, backend, controller = bridge_fixture(tmp_path, monkeypatch)
    failure = OSError("synthetic startup sidecar write failure")
    original_finish = native._StartupSampleJournal.finish

    def fail_append(journal, _sample):
        journal.attempted_sample_count += 1
        assert not backend.finished.is_set()
        raise failure

    def finish(journal, primary=None):
        assert backend.finished.is_set(), "journal closed before exact loader reached terminal"
        assert controller.close_count == backend.close_count == 0
        return original_finish(journal, primary)

    monkeypatch.setattr(native._StartupSampleJournal, "append", fail_append)
    monkeypatch.setattr(native._StartupSampleJournal, "finish", finish)
    task = asyncio.create_task(bridge.prepare(route))
    try:
        await entered(backend)
        await still_loading(task, backend, controller)
        assert bridge._startup_journal._file is not None
    finally:
        backend.release.set()
    with pytest.raises(OSError) as caught:
        await task
    assert caught.value is failure
    reference = bridge.startup_samples_evidence
    assert reference["complete"] is False and reference["error_type"] == "OSError"
    assert reference["sample_count"] == 0 and reference["attempted_sample_count"] == 1
    assert bridge._startup_journal._file is None
    bridge.close()
    assert controller.close_count == backend.close_count == 1
