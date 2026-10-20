"""Tests for vllm_omni/engine/init_timeline.py (CPU, no engine)."""

from __future__ import annotations

import json
import os
import time

import pytest

from vllm_omni.engine import init_timeline as it

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _reset_timeline(monkeypatch):
    it.InitTimeline.reset()
    monkeypatch.delenv(it.ENV_DIR, raising=False)
    monkeypatch.delenv(it.ENV_STAGE_ID, raising=False)
    monkeypatch.delenv(it.ENV_REPLICA_ID, raising=False)
    yield
    it.InitTimeline.reset()


def test_disabled_timeline_records_nothing():
    tl = it.InitTimeline.start(enabled=False)
    with tl.phase("x"):
        pass
    tl.mark("m")
    assert tl.phases == []
    assert it.ENV_DIR not in os.environ
    with it.worker_phase("w"):
        pass  # no-op without env / enabled timeline


def test_phase_nesting_and_durations(tmp_path):
    tl = it.InitTimeline.start(enabled=True, run_dir=tmp_path / "run")
    with tl.phase("outer", stage_id=0) as info:
        time.sleep(0.01)
        with tl.phase("inner", stage_id=0, n=3):
            time.sleep(0.01)
        info["k"] = "v"
    names = [p.name for p in tl.phases]
    assert names == ["inner", "outer"]  # inner closes first
    outer = tl.phases[1]
    assert outer.dur_s >= 0.02 and outer.extra == {"k": "v"}
    assert tl.phases[0].extra == {"n": 3}
    assert os.environ[it.ENV_DIR] == str(tmp_path / "run")


def test_worker_helpers_write_jsonl_and_ingest(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    tl = it.InitTimeline.start(enabled=True, run_dir=run_dir)
    # Simulate a worker: no in-process timeline, env set by the parent.
    it.InitTimeline.reset()
    monkeypatch.setenv(it.ENV_STAGE_ID, "1")
    monkeypatch.setenv(it.ENV_REPLICA_ID, "2")
    with it.worker_phase("load_weights", gib=1.5):
        time.sleep(0.005)
    it.worker_mark("proc_entry")
    files = list(run_dir.glob("worker_*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert {r["name"] for r in rows} == {"load_weights", "proc_entry"}
    assert all(r["stage_id"] == 1 and r["replica_id"] == 2 and r["role"] == "worker" for r in rows)
    # Back in the engine process: ingest (twice -> idempotent).
    it.InitTimeline._current = tl
    assert tl.ingest_workers() == 2
    assert tl.ingest_workers() == 0
    assert [p.name for p in tl.phases] == sorted(
        [p.name for p in tl.phases], key=lambda n: [p.start for p in tl.phases if p.name == n][0]
    )


def test_table_dump_and_suggested_timeouts(tmp_path):
    tl = it.InitTimeline(enabled=True, run_dir=None, t0=1000.0)
    tl.record(it.InitPhase("g2_spawn", 1000.0, 1010.0, stage_id=0, role="engine", pid=1))
    tl.record(it.InitPhase("load_weights", 1002.0, 1200.0, stage_id=0, role="worker", pid=2))
    tl.record(it.InitPhase("load_weights", 1210.0, 1260.0, stage_id=1, role="worker", pid=3))
    tl.t_end = 1500.0
    assert tl.total_s == pytest.approx(500.0)
    table = tl.format_table()
    assert "load_weights" in table and "g2_spawn" in table and "total=500.00s" in table
    d = tl.to_dict()
    assert d["schema"] == 1 and len(d["phases"]) == 3
    out = tmp_path / "tl.json"
    tl.dump(out)
    assert json.loads(out.read_text())["total_s"] == pytest.approx(500.0)
    # stage 0 span = 1000..1200 = 200 s -> 300 (== default floor); total 500 s -> 750
    sug = tl.suggest_timeouts()
    assert sug == {"stage_init_timeout": 300, "init_timeout": 750}
    # short runs floor at the defaults
    short = it.InitTimeline(enabled=True, t0=0.0)
    short.record(it.InitPhase("x", 0.0, 1.0, stage_id=0))
    short.t_end = 2.0
    assert short.suggest_timeouts() == {"stage_init_timeout": 300, "init_timeout": 600}


def test_timed_phase_decorator_records_in_engine_process(tmp_path):
    tl = it.InitTimeline.start(enabled=True, run_dir=tmp_path / "run")

    class Runner:
        @it.timed_phase("capture")
        def capture(self, x):
            return x * 2

    assert Runner().capture(21) == 42
    assert [p.name for p in tl.phases] == ["capture"]
    assert tl.phases[0].role == "engine"
