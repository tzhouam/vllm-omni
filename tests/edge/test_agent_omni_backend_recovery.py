"""A failed Omni model request must not leave a dead llama.cpp route cached."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.edge.agent.omni_backend import OmniLlamaBackend, OmniLlamaConfig


def _config() -> OmniLlamaConfig:
    return OmniLlamaConfig(
        route_id="test-cpu", model_file="unused.gguf", model_sha256="a" * 64,
        server_bin="unused-server", server_sha256="b" * 64,
        log_file="unused.log", placement="cpu",
        capacities={"host_ram": 4096}, demands={"host_ram": 1024},
        memory_overhead_bytes=128,
    )


def test_failed_request_unloads_dead_worker_and_next_turn_restarts(monkeypatch) -> None:
    import vllm_omni.engine.stage_runtime as runtime_module

    created = []

    class Ledger:
        def __init__(self):
            self.owners = ["(0, 0)"]

        def snapshot(self):
            return {"owners": self.owners, "quarantined": []}

    class Client:
        execution_plan = {"requested_device": "cpu"}

        def __init__(self):
            self.healthy = True

        def check_health(self):
            if not self.healthy:
                raise RuntimeError("worker exited")

        async def receive_agent_delta(self, request_id):
            raise RuntimeError("simulated inference failure")

    class Pool:
        def __init__(self):
            self.stage_client = Client()
            self.aborted = []

        async def submit_initial(self, request_id, state, payload):
            return 0

        async def abort_requests(self, request_ids):
            self.aborted.extend(request_ids)
            self.stage_client.healthy = False
            return []

    class Runtime:
        def __init__(self, *args, **kwargs):
            self.resource_ledger = Ledger()
            self.stage_pools = [Pool()]
            self.shutdown_calls = 0
            created.append(self)

        def initialize(self):
            pass

        def shutdown(self):
            self.shutdown_calls += 1
            self.stage_pools[0].stage_client.healthy = False
            self.resource_ledger.owners.clear()

    monkeypatch.setattr(runtime_module, "StageRuntime", Runtime)
    backend = OmniLlamaBackend(_config())
    backend.start()
    assert backend.resident

    async def fail_request():
        async for _ in backend.generate("prompt", request_id="first", max_tokens=8):
            pass

    with pytest.raises(RuntimeError, match="simulated inference failure"):
        asyncio.run(fail_request())
    assert created[0].stage_pools[0].aborted == ["first"]
    assert created[0].shutdown_calls >= 1
    assert not backend.resident
    assert backend.execution_plan is None

    backend.start()
    assert len(created) == 2
    assert backend.resident
    assert backend.close()


def test_quarantined_worker_refuses_restart(monkeypatch) -> None:
    import vllm_omni.engine.stage_runtime as runtime_module

    class Ledger:
        def snapshot(self):
            return {"owners": ["(0, 0)"], "quarantined": ["(0, 0)"]}

    class Client:
        execution_plan = {"requested_device": "cpu"}

        def check_health(self):
            raise RuntimeError("worker exited")

    class Runtime:
        def __init__(self, *args, **kwargs):
            self.stage_pools = [type("Pool", (), {"stage_client": Client()})()]
            self.resource_ledger = Ledger()

        def initialize(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(runtime_module, "StageRuntime", Runtime)
    backend = OmniLlamaBackend(_config())
    backend.start()
    assert not backend.close()
    assert not backend.resident
    with pytest.raises(RuntimeError, match="reserved or quarantined"):
        backend.start()


def test_direct_cancel_drains_waiting_generator_and_resets_worker() -> None:
    entered = asyncio.Event()

    class Proc:
        pid = 1234
        exited = False

        def poll(self):
            return 0 if self.exited else None

    class Ledger:
        def __init__(self):
            self.owners = ["(0, 0)"]

        def snapshot(self):
            return {"owners": self.owners, "quarantined": []}

    class Client:
        def __init__(self):
            self._proc = Proc()

        async def receive_agent_delta(self, request_id):
            entered.set()
            await asyncio.Event().wait()

    class Pool:
        def __init__(self):
            self.stage_client = Client()
            self.aborted = []

        async def submit_initial(self, request_id, state, payload):
            return 0

        async def abort_requests(self, request_ids):
            self.aborted.extend(request_ids)
            return []

    class Runtime:
        def __init__(self, pool):
            self.resource_ledger = Ledger()
            self.pool = pool
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1
            self.pool.stage_client._proc.exited = True
            self.resource_ledger.owners.clear()

    backend = OmniLlamaBackend(_config())
    pool = Pool()
    runtime = Runtime(pool)
    backend._pool, backend._runtime = pool, runtime
    backend.execution_plan = {"requested_device": "cpu"}

    async def run():
        async def consume():
            async for _ in backend.generate("prompt", request_id="cancel-me", max_tokens=8):
                pass

        task = asyncio.create_task(consume())
        await entered.wait()
        await backend.cancel("cancel-me")
        assert task.done()
        assert task.cancelled()

    asyncio.run(run())
    assert pool.aborted == ["cancel-me"]
    assert runtime.shutdown_calls >= 1
    assert backend._runtime is None
    assert backend.request_state_released("cancel-me")
    assert backend.release_evidence["worker_pid_before"] == 1234
    assert not backend.request_state_released("another-request")


def test_cancel_after_model_step_drains_resident_worker_before_release() -> None:
    class Proc:
        pid = 5678
        exited = False

        def poll(self):
            return 0 if self.exited else None

    class Ledger:
        def __init__(self):
            self.owners = ["(0, 0)"]

        def snapshot(self):
            return {"owners": self.owners, "quarantined": []}

    class Runtime:
        def __init__(self, *, drain: bool, proc):
            self.resource_ledger = Ledger()
            self.drain = drain
            self.proc = proc
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1
            self.proc.exited = True
            if self.drain:
                self.resource_ledger.owners.clear()

    for drain in (True, False):
        backend = OmniLlamaBackend(_config())
        proc = Proc()
        runtime = Runtime(drain=drain, proc=proc)
        backend._runtime = runtime
        backend._pool = SimpleNamespace(stage_client=SimpleNamespace(_proc=proc))
        backend._last_turn_request_id = "cancel-during-tool"
        backend.execution_plan = {"requested_device": "cpu"}
        assert backend.request_state_released("cancel-during-tool") is drain
        assert runtime.shutdown_calls == 1
        assert (backend._runtime is None) is drain
        if not drain:
            assert backend._recovery_blocked_reason is not None


def test_failed_initialization_keeps_unverified_memory_claim(monkeypatch) -> None:
    import vllm_omni.engine.stage_runtime as runtime_module

    class Ledger:
        def snapshot(self):
            return {"owners": ["(0, 0)"], "quarantined": ["(0, 0)"]}

    class Runtime:
        def __init__(self, *args, **kwargs):
            self.resource_ledger = Ledger()
            self.stage_pools = []

        def initialize(self):
            raise RuntimeError("model load failed")

        def shutdown(self):
            pass

    monkeypatch.setattr(runtime_module, "StageRuntime", Runtime)
    backend = OmniLlamaBackend(_config())
    with pytest.raises(RuntimeError, match="model load failed"):
        backend.start()
    assert not backend.close()
    assert not backend.resident
    assert not backend.request_state_released("failed-load")
    with pytest.raises(RuntimeError, match="reserved or quarantined"):
        backend.start()
