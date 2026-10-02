from types import SimpleNamespace

import pytest

from vllm_omni.engine.resource_ledger import ResourceLedger, ResourceUnavailable
from vllm_omni.engine.stage_admission import StageAdmissionError, check_native_resource_budget
from vllm_omni.engine.stage_runtime import StageRuntime


def test_reserve_many_is_atomic_across_shared_pools():
    ledger = ResourceLedger({"host_ram": 100, "wsl_ram": 70, "vram": 40})
    with pytest.raises(ResourceUnavailable, match="host_ram"):
        ledger.reserve_many({
            "native": {"host_ram": 55, "wsl_ram": 55},
            "graph": {"host_ram": 50},
        })
    assert ledger.snapshot()["owners"] == []
    reservations = ledger.reserve_many({
        "native": {"host_ram": 45, "wsl_ram": 45},
        "graph": {"host_ram": 50},
    })
    assert ledger.snapshot()["reserved"] == {"host_ram": 95, "wsl_ram": 45, "vram": 0}
    assert not ledger.release(reservations["native"], drained=False)
    assert ledger.snapshot()["quarantined"] == ["native"]
    assert ledger.release(reservations["native"], drained=True)
    assert ledger.release(reservations["graph"], drained=True)


def _native(stage_id: int, demands: dict[str, int] | None, model_root):
    model_dir = model_root / f"native-{stage_id}"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "model.safetensors").write_bytes(b"x" * 10)
    budget = None if demands is None else {"capacities": {"host_ram": 100}, "demands": demands}
    replica = SimpleNamespace(
        metadata=SimpleNamespace(stage_id=stage_id, stage_type="llm", runtime_cfg={"devices": "cpu"}),
        replica_id=0,
        launch_mode="local",
        stage_cfg=SimpleNamespace(engine_args={"resource_budget": budget}),
        stage_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model=str(model_dir), enforce_eager=True),
            cache_config=SimpleNamespace(kv_cache_memory_bytes=20),
        ),
    )
    return SimpleNamespace(replicas=[replica])


def test_native_stages_share_the_controller_ledger_when_budgeted(tmp_path):
    runtime = StageRuntime([], "model", "", stage_init_timeout=1, async_chunk=False)
    runtime._reserve_stage_resources([
        _native(0, {"host_ram": 45}, tmp_path), _native(1, {"host_ram": 50}, tmp_path)
    ])
    assert runtime.resource_ledger.snapshot()["reserved"]["host_ram"] == 95
    assert len(runtime._resource_reservations) == 2


def test_budgeted_native_plan_fails_closed_on_missing_or_excess_demand(tmp_path):
    runtime = StageRuntime([], "model", "", stage_init_timeout=1, async_chunk=False)
    with pytest.raises(ValueError, match="every stage"):
        runtime._reserve_stage_resources([_native(0, {"host_ram": 45}, tmp_path), _native(1, None, tmp_path)])
    assert runtime.resource_ledger is None
    with pytest.raises(ResourceUnavailable, match="host_ram"):
        runtime._reserve_stage_resources([
            _native(0, {"host_ram": 55}, tmp_path), _native(1, {"host_ram": 50}, tmp_path)
        ])
    assert runtime.resource_ledger.snapshot()["owners"] == []
    assert runtime._resource_reservations == {}


def test_budgeted_native_plan_cannot_skip_admission_with_a_remote_replica(tmp_path):
    local = _native(0, {"host_ram": 45}, tmp_path)
    remote = _native(1, {"host_ram": 45}, tmp_path)
    remote.replicas[0].launch_mode = "remote"
    runtime = StageRuntime([], "model", "", stage_init_timeout=1, async_chunk=False)
    with pytest.raises(ValueError, match="one local host controller"):
        runtime._reserve_stage_resources([local, remote])
    assert runtime.resource_ledger is None


def test_duplicate_replica_identity_cannot_collapse_budget_claim(tmp_path):
    first = _native(0, {"host_ram": 45}, tmp_path)
    duplicate = _native(0, {"host_ram": 45}, tmp_path)
    runtime = StageRuntime([], "model", "", stage_init_timeout=1, async_chunk=False)
    with pytest.raises(ValueError, match="duplicate stage/replica"):
        runtime._reserve_stage_resources([first, duplicate])
    assert runtime.resource_ledger.snapshot()["owners"] == []


def test_native_claim_below_independent_weight_and_kv_floor_is_rejected(tmp_path):
    replica = _native(0, {"host_ram": 29}, tmp_path).replicas[0]
    with pytest.raises(StageAdmissionError, match="below the independently known"):
        check_native_resource_budget(replica, replica.stage_cfg.engine_args["resource_budget"])
    replica.stage_cfg.engine_args["resource_budget"]["demands"] = {"host_ram": 0}
    with pytest.raises(StageAdmissionError, match="below the independently known"):
        check_native_resource_budget(replica, replica.stage_cfg.engine_args["resource_budget"])


def test_native_gpu_claim_must_cover_vllm_utilization_and_host_staging(tmp_path):
    replica = _native(0, {"host_ram": 10, "vram:0": 40}, tmp_path).replicas[0]
    replica.metadata.runtime_cfg = {"devices": "0"}
    replica.stage_vllm_config.cache_config.gpu_memory_utilization = 0.5
    replica.stage_cfg.engine_args["resource_budget"]["capacities"] = {"host_ram": 100, "vram:0": 100}
    replica.stage_cfg.engine_args["resource_budget"]["native_physical_vram_bytes"] = 100
    with pytest.raises(StageAdmissionError, match="vram:0"):
        check_native_resource_budget(replica, replica.stage_cfg.engine_args["resource_budget"])
    replica.stage_cfg.engine_args["resource_budget"]["demands"] = {"host_ram": 0, "vram:0": 50}
    with pytest.raises(StageAdmissionError, match="host_ram"):
        check_native_resource_budget(replica, replica.stage_cfg.engine_args["resource_budget"])


def test_native_gpu_claim_uses_physical_vram_not_controller_ceiling(tmp_path):
    replica = _native(0, {"host_ram": 10, "vram:0": 40}, tmp_path).replicas[0]
    replica.metadata.runtime_cfg = {"devices": "0"}
    replica.stage_vllm_config.cache_config.gpu_memory_utilization = 0.5
    budget = replica.stage_cfg.engine_args["resource_budget"]
    budget["capacities"] = {"host_ram": 100, "vram:0": 60}

    with pytest.raises(StageAdmissionError, match="native_physical_vram_bytes"):
        check_native_resource_budget(replica, budget)
    budget["native_physical_vram_bytes"] = 100
    with pytest.raises(StageAdmissionError, match="vram:0"):
        check_native_resource_budget(replica, budget)
    budget["demands"]["vram:0"] = 50
    assert check_native_resource_budget(replica, budget)["vram:0"] == 50
    budget["capacities"]["vram:0"] = 101
    with pytest.raises(StageAdmissionError, match="exceeds physical device total VRAM"):
        check_native_resource_budget(replica, budget)


def test_native_budget_cannot_invent_more_host_capacity_than_observed(tmp_path, monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=50))
    runtime = StageRuntime([], "model", "", stage_init_timeout=1, async_chunk=False)
    with pytest.raises(ValueError, match="exceeds currently available host RAM"):
        runtime._reserve_stage_resources([_native(0, {"host_ram": 45}, tmp_path)])
    assert runtime.resource_ledger is None
