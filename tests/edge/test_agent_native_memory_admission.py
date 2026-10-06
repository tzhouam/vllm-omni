# SPDX-License-Identifier: Apache-2.0
"""Independent memory-probe accounting and no-allocation refusal tests."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.edge_agent.experiments.native_memory_admission import (
    compare_samples, over_ceiling_ledger_refusal, private_record_path,
    run_probe, sample_operation, verify_resident_claim,
)
from vllm_omni.edge.agent.router import Route


def _sample(phase: str, ram: int, vram: int | None = None) -> dict:
    values = {"ram_used_bytes": ram}
    if vram is not None:
        values["vram_used_bytes"] = vram
    return {"phase": phase, "values": values}


def test_over_ceiling_uses_real_ledger_and_keeps_no_claim():
    capacities = {"host_ram": 100, "vram": 50}
    for pool in capacities:
        evidence = over_ceiling_ledger_refusal(capacities, pool)
        assert evidence["declared_bytes"] == capacities[pool] + 1
        assert evidence["physical_allocation_attempted"] is False
        assert ">" in evidence["refusal"]
        assert evidence["ledger_after_refusal"]["reserved"] == {
            "host_ram": 0, "vram": 0,
        }
        assert evidence["ledger_after_refusal"]["owners"] == []


def test_comparison_uses_one_baseline_and_both_load_and_request_phases():
    samples = [
        _sample("pre_load", 100, 20),
        _sample("cold_load", 135, 48),
        _sample("complete_request", 142, 44),
        _sample("post_request", 130, 39),
    ]
    result = compare_samples(samples, {"host_ram": 50, "vram": 25})
    assert result["host_ram"]["sampled_global_incremental_peak_bytes"] == 42
    assert result["host_ram"]["reservation_minus_sampled_incremental_bytes"] == 8
    assert result["vram"]["sampled_global_incremental_peak_bytes"] == 28
    assert result["vram"]["reservation_minus_sampled_incremental_bytes"] == -3
    assert result["vram"]["sample_count"] == 4


def test_comparison_refuses_missing_phases_and_vram_sensor():
    with pytest.raises(ValueError, match="cold-load"):
        compare_samples([_sample("pre_load", 1), _sample("complete_request", 2)],
                        {"host_ram": 1})
    with pytest.raises(ValueError, match="vram measurement unavailable"):
        compare_samples([_sample("pre_load", 1), _sample("cold_load", 2),
                         _sample("complete_request", 3)], {"vram": 1})


def test_private_record_name_is_basename_only(tmp_path):
    assert private_record_path(root=tmp_path, name="probe.jsonl") == tmp_path / "probe.jsonl"
    for name in ("../probe.jsonl", "sub/probe.jsonl", "probe.json", "../outside.jsonl"):
        with pytest.raises(ValueError):
            private_record_path(root=tmp_path, name=name)


def test_sampling_waits_for_one_operation_and_records_raw_values():
    future: Future[str] = Future()
    seen = []

    class Sampler:
        def sample(self):
            future.set_result("done")
            return {"ram_used_bytes": 123}

    result = sample_operation(
        future, phase="cold_load", telemetry=Sampler(),
        write_sample=lambda phase, values: seen.append((phase, values)),
        interval_s=.01, timeout_s=1,
    )
    assert result == "done"
    assert seen == [("cold_load", {"ram_used_bytes": 123})]


def test_sampling_flushes_deadline_marker_before_a_hung_operation_unwinds():
    future: Future[str] = Future()
    deadlines = []

    class Sampler:
        def sample(self):
            return {"ram_used_bytes": 123}

    with pytest.raises(TimeoutError, match="cold_load"):
        sample_operation(
            future, phase="cold_load", telemetry=Sampler(),
            write_sample=lambda phase, values: None,
            write_deadline=deadlines.append,
            interval_s=.01, timeout_s=.02,
        )
    assert deadlines == ["cold_load"]
    assert not future.done()


def test_coordinator_snapshot_must_bind_both_physical_pool_claims():
    snapshot = {
        "resident_route": "gemma",
        "ledger": {"owners": ["gemma"], "quarantined": [],
                   "reserved": {"host_ram": 20, "vram": 10}},
    }
    verify_resident_claim(snapshot, "gemma", {"host_ram": 20, "vram": 10})
    with pytest.raises(AssertionError, match="does not own"):
        verify_resident_claim(snapshot, "gemma", {"host_ram": 20, "vram": 11})
    snapshot["ledger"]["quarantined"] = ["gemma"]
    with pytest.raises(AssertionError, match="does not own"):
        verify_resident_claim(snapshot, "gemma", {"host_ram": 20, "vram": 10})


@pytest.mark.parametrize("profile_bound", [False, True])
def test_native_probe_isolated_config_and_complete_turn_are_recorded_privately(
    tmp_path, monkeypatch, profile_bound,
):
    import benchmarks.edge_agent.experiments.native_memory_admission as memory_probe
    import benchmarks.edge_agent.native_profile as native_profile
    import vllm_omni.edge.agent.native_app as native_app

    config = {"routes": [{
        "route_id": "gemma", "model_sha256": "a" * 64,
        "server_sha256": "b" * 64, "memory_demands": {"host_ram": 20, "vram": 10},
        "placement": "cpu+Vulkan0", "start_timeout_s": 1,
        "request_timeout_s": 1, "log_file": "user-original.log",
    }], "qualification_bundles": ["do-not-load.json"],
       "trusted_review_keys": {"k": "fake"}}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    build_configs = []

    class FakeBackend:
        execution_plan = None

        def start(self):
            time.sleep(.02)
            self.execution_plan = {
                "requested_device": "cpu+Vulkan0",
                "observed_model_placement": "cpu+Vulkan0",
                "reserved_bytes": {"host_ram": 20, "vram": 10},
            }

    backend = FakeBackend()

    class FakeCoordinator:
        resident = True

        def snapshot(self):
            return {
                "resident_route": "gemma" if self.resident else None,
                "ledger": {"owners": ["gemma"] if self.resident else [],
                           "quarantined": [],
                           "reserved": {"host_ram": 20 if self.resident else 0,
                                        "vram": 10 if self.resident else 0}},
            }

    backend._coordinator = FakeCoordinator()

    class FakeController:
        routes = [Route("gemma", "artifact", "Gemma", "backend",
                        frozenset({"text"}), "cpu+Vulkan0",
                        {"host_ram": 20, "vram": 10})]
        backends = {"gemma": backend}

        def __init__(self):
            self.listeners = []
            self.closed = False

        def admit(self, route):
            return SimpleNamespace(admitted=True, reason="live capacity gate passed")

        def add_listener(self, callback):
            self.listeners.append(callback)

        def submit(self, prompt):
            assert prompt == "Reply with the single word ready."
            for listener in self.listeners:
                listener({"kind": "final", "epoch": 1, "seq": 1, "request_id": "req"})
            future = Future()
            future.set_result("ready")
            return future

        def cancel(self):
            pass

        def close(self):
            self.closed = True
            backend._coordinator.resident = False

    controller = FakeController()

    def fake_build(path):
        build_configs.append(json.loads(Path(path).read_text(encoding="utf-8")))
        return controller, {
            "host_ram_available_bytes": 100, "vram_available_bytes": 50,
            "power_condition": "AC",
        }

    class FakeTelemetry:
        def __init__(self, expected_power):
            assert expected_power == "AC"

        def sample(self):
            return {"ram_used_bytes": 60, "vram_used_bytes": 20}

        def close(self):
            pass

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(native_app, "build_controller", fake_build)
    monkeypatch.setattr(native_profile, "WindowsTelemetry", FakeTelemetry)
    profile_index = tmp_path / "profile-index.json" if profile_bound else None
    binding = {"schema": "omni-agent-profile-binding-v1",
               "profile_raw_sha256": "a" * 64}
    if profile_bound:
        profile_index.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(memory_probe, "bind_profile_to_live", lambda path, **kwargs:
                            binding if path == profile_index and kwargs["route_id"] == "gemma"
                            else None)
    record = run_probe(config_path=source, route_id="gemma", record_root=tmp_path / "private",
                       record_name="probe.jsonl", interval_s=.01,
                       profile_index=profile_index)
    rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
    kinds = [row["record_type"] for row in rows]
    assert kinds.count("over_ceiling_refusal") == 2
    assert kinds.count("sample") >= 3
    assert {row["phase"] for row in rows if row["record_type"] == "sample"} >= {
        "pre_load", "cold_load", "complete_request",
    }
    assert rows[-1]["result"] == "one_complete_agent_turn_measured"
    assert rows[-1]["signed_qualification_gate"] is False
    assert next(row for row in rows if row["record_type"] == "loaded_plan")[
        "model_placement_verified"] is True
    assert rows[0]["profile_binding_requested"] is profile_bound
    assert rows[0]["profile_binding"] == (binding if profile_bound else None)
    assert controller.closed
    assert "qualification_bundles" not in build_configs[0]
    assert build_configs[0]["memory_file"].startswith(str(tmp_path / "private"))
    assert build_configs[0]["routes"][0]["log_file"].startswith(str(tmp_path / "private"))
    with pytest.raises(FileExistsError):
        run_probe(config_path=source, route_id="gemma", record_root=tmp_path / "private",
                  record_name="probe.jsonl", interval_s=.01,
                  profile_index=profile_index)


def test_failed_native_build_keeps_private_raw_outcome(tmp_path, monkeypatch):
    import vllm_omni.edge.agent.native_app as native_app

    source = tmp_path / "source.json"
    source.write_text(json.dumps({"routes": [{
        "route_id": "gemma", "model_sha256": "a" * 64,
        "server_sha256": "b" * 64, "memory_demands": {"host_ram": 20},
        "log_file": "unused.log",
    }]}), encoding="utf-8")

    def failed_build(_path):
        raise RuntimeError("native admission refused before worker load")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(native_app, "build_controller", failed_build)
    with pytest.raises(RuntimeError, match="native admission refused"):
        run_probe(config_path=source, route_id="gemma", record_root=tmp_path / "private",
                  record_name="failed.jsonl")
    record = tmp_path / "private" / "failed.jsonl"
    rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
    assert [row["record_type"] for row in rows] == ["manifest", "outcome"]
    assert rows[-1]["result"] == "failed"
    assert "native admission refused" in rows[-1]["error"]


def test_profile_binding_mismatch_refuses_before_cold_load(tmp_path, monkeypatch):
    import benchmarks.edge_agent.experiments.native_memory_admission as memory_probe
    import vllm_omni.edge.agent.native_app as native_app

    source = tmp_path / "source.json"
    source.write_text(json.dumps({"routes": [{
        "route_id": "gemma", "model_sha256": "a" * 64,
        "server_sha256": "b" * 64, "memory_demands": {"host_ram": 20},
        "log_file": "unused.log",
    }]}), encoding="utf-8")
    profile = tmp_path / "profile-index.json"
    profile.write_text("{}", encoding="utf-8")
    seen = []

    class Controller:
        def close(self):
            seen.append("closed")

        def cancel(self):
            seen.append("cancelled")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(native_app, "build_controller", lambda _path: (Controller(), {"os": "test"}))

    def reject(*_args, **_kwargs):
        seen.append("binding_checked")
        raise ValueError("profile runtime mismatch")

    monkeypatch.setattr(memory_probe, "bind_profile_to_live", reject)
    with pytest.raises(ValueError, match="profile runtime mismatch"):
        run_probe(config_path=source, route_id="gemma", record_root=tmp_path / "private",
                  record_name="mismatch.jsonl", profile_index=profile)
    assert seen == ["binding_checked", "cancelled", "closed"]
    rows = [json.loads(line) for line in
            (tmp_path / "private" / "mismatch.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["record_type"] for row in rows] == ["manifest", "outcome"]
    assert rows[0]["profile_binding_requested"] is True
    assert rows[0]["profile_binding"] is None
    assert rows[-1]["result"] == "failed"
