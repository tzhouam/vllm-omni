"""WP8 per-step overhead attribution counters (CPU)."""

from __future__ import annotations

import json

import pytest

from vllm_omni.edge import step_stats as ss

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_disabled_stats_are_noops(monkeypatch):
    monkeypatch.delenv(ss.ENV_DIR, raising=False)
    st = ss.StepStats.reset(None, role="x")
    assert not st.enabled
    st.add("a", 1.0)
    with st.timed("b"):
        pass
    fn = st.wrap("c", lambda: 3)
    assert fn() == 3 and not hasattr(fn, "__wrapped__")
    assert st.to_dict()["counters"] == {} and st.dump() is None


def test_series_summary_and_merge(tmp_path):
    st = ss.StepStats.reset(str(tmp_path), role="stage0")
    for v in (1.0, 2.0, 3.0, 4.0, 100.0):
        st.add("sched.schedule_ms", v)
    with st.timed("core.step_ms"):
        pass
    path = st.dump()
    assert path and path.endswith(f"stage0_{st.pid}.json")
    d = json.loads(open(path).read())
    s = d["counters"]["sched.schedule_ms"]
    assert s["count"] == 5 and s["max_ms"] == 100.0 and s["mean_ms"] == pytest.approx(22.0) and s["p50_ms"] == 3.0
    # a second process with the same role merges by count/sum and worst p95
    (tmp_path / "stage0_999.json").write_text(
        json.dumps(
            {
                "role": "stage0",
                "counters": {
                    "sched.schedule_ms": {
                        "count": 5,
                        "sum_ms": 5.0,
                        "mean_ms": 1.0,
                        "p50_ms": 1.0,
                        "p95_ms": 1.0,
                        "max_ms": 1.0,
                    }
                },
            }
        )
    )
    merged = ss.merge_dir(tmp_path)
    m = merged["roles"]["stage0"]["sched.schedule_ms"]
    assert m["count"] == 10 and m["sum_ms"] == pytest.approx(115.0) and m["p95_ms"] == 100.0
    rows = merged["attribution"]
    assert rows[0]["counter"] == "sched.schedule_ms" and rows[0]["share_of_frame"] == pytest.approx(11.5 / 80)
    assert "share of 80 ms" in ss.format_table(rows)


def test_instrument_engine_core_wraps_scheduler_and_step(tmp_path):
    ss.StepStats.reset(str(tmp_path), role="stage1")

    class Sched:
        def schedule(self, throttle_prefills=False):
            return "out"

        def update_from_output(self, a, b):
            return {}

    class Core:
        def __init__(self):
            self.scheduler = Sched()
            self.step_fn = lambda: ({}, True)

    core = Core()
    ss.instrument_engine_core(core)
    ss.instrument_engine_core(core)  # idempotent
    assert core.scheduler.schedule(True) == "out" and core.scheduler.update_from_output(1, 2) == {}
    assert core.step_fn() == ({}, True)
    counters = ss.StepStats.get().to_dict()["counters"]
    assert {"sched.schedule_ms", "sched.update_from_output_ms", "core.step_ms"} <= set(counters)
    assert counters["core.step_ms"]["count"] == 1


def test_shm_lockfile_age_missing_is_none():
    assert ss.shm_lockfile_age_ms("definitely_missing_key_xyz") is None
