"""Tests for the benchmark harness's judgement, not its plumbing.

The summariser is where a shared machine turns into a wrong claim, so that is
what is pinned here: spread reported, noisy arms flagged, single passes
refused as results, contended passes labelled.
"""

import json
import os

import pytest

from bench_harness import (
    Arm,
    HostLock,
    check_expectations,
    indistinguishable,
    Observation,
    core_utilization,
    format_summary,
    metric_is_lower_better,
    parse_cores,
    preflight,
    provenance,
    run_grid,
    summarize,
)


def _obs(tag, p, value, load=1.0, ok=True):
    return Observation(tag, p, {"tg_tps": value} if ok else {}, load, load, 1.0, ok)


def test_reports_best_median_and_spread():
    s = summarize([_obs("a", 1, 80), _obs("a", 2, 100), _obs("a", 3, 90)], "tg_tps")[0]
    assert s.best == 100 and s.median == 90
    assert s.spread_pct == pytest.approx(20.0)
    assert s.n == 3


def test_flags_an_arm_whose_passes_disagree():
    """A 30% swing between identical runs is the failure mode that produced
    three wrong conclusions in this study."""
    noisy = summarize([_obs("a", 1, 60), _obs("a", 2, 90)], "tg_tps")[0]
    assert noisy.noisy
    clean = summarize([_obs("b", 1, 88), _obs("b", 2, 90)], "tg_tps", noise_threshold=8.0)[0]
    assert not clean.noisy


def test_single_pass_is_not_presented_as_a_result():
    text = format_summary(summarize([_obs("a", 1, 80)], "tg_tps"))
    assert "single pass" in text


def test_contended_passes_are_counted_not_hidden():
    rows = [_obs("a", 1, 80, load=1.0), _obs("a", 2, 40, load=40.0)]
    s = summarize(rows, "tg_tps")[0]
    assert s.n_busy == 1
    assert "under load" in format_summary([s])


def test_failed_runs_do_not_enter_the_summary():
    rows = [_obs("a", 1, 80), _obs("a", 2, 0, ok=False)]
    s = summarize(rows, "tg_tps")[0]
    assert s.n == 1


def test_arm_order_rotates_between_passes(monkeypatch):
    """Interleaving alone leaves arm 0 permanently first, so whatever the host
    is doing at the start of a pass always lands on the same arm. This test
    previously asserted the un-rotated order, which is how that survived."""
    order = []

    def fake_run(arm, pass_no, timeout, keys):
        order.append((pass_no, arm.tag))
        return _obs(arm.tag, pass_no, 50.0)

    monkeypatch.setattr("bench_harness.run_once", fake_run)
    arms = [Arm("a", ["true"]), Arm("b", ["true"]), Arm("c", ["true"])]
    run_grid(arms, passes=3, metric="tg_tps", timeout=1, keys=["tg_tps"])
    assert order == [(1, "a"), (1, "b"), (1, "c"),
                     (2, "b"), (2, "c"), (2, "a"),
                     (3, "c"), (3, "a"), (3, "b")]
    # every arm still runs once per pass
    for p in (1, 2, 3):
        assert sorted(t for q, t in order if q == p) == ["a", "b", "c"]


def test_best_is_the_minimum_for_a_memory_metric():
    """Taking the maximum for every metric published the worst memory pass as
    the headline figure."""
    rows = [Observation("a", 1, {"rss_anon_mb": 2802.0}, 1.0, 1.0, 1.0, True),
            Observation("a", 2, {"rss_anon_mb": 2814.0}, 1.0, 1.0, 1.0, True)]
    s = summarize(rows, "rss_anon_mb")[0]
    assert s.lower_is_better is True
    assert s.best == 2802.0


def test_best_arm_sorts_first_in_both_directions():
    mem = [Observation("small", 1, {"rss_anon_mb": 2802.0}, 1.0, 1.0, 1.0, True),
           Observation("big", 1, {"rss_anon_mb": 3542.0}, 1.0, 1.0, 1.0, True)]
    assert [s.tag for s in summarize(mem, "rss_anon_mb")] == ["small", "big"]
    assert [s.tag for s in summarize([_obs("slow", 1, 40.0), _obs("fast", 1, 86.6)],
                                     "tg_tps")] == ["fast", "slow"]


@pytest.mark.parametrize("metric,lower", [
    ("tg_tps", False), ("pp_tps", False),
    ("rss_anon_mb", True), ("peak_rss_anon_mb", True),
    ("init_s", True), ("ready_s", True), ("step_ms", True),
])
def test_metric_direction_is_read_from_the_name(metric, lower):
    assert metric_is_lower_better(metric) is lower


def test_a_custom_noise_threshold_does_not_leak_into_the_next_call():
    """The threshold used to be written into a module global after the rows
    were built, so one call's setting silently changed the next one's."""
    loose = summarize([_obs("a", 1, 60), _obs("a", 2, 90)], "tg_tps",
                      noise_threshold=50.0)[0]
    assert not loose.noisy
    strict = summarize([_obs("a", 1, 60), _obs("a", 2, 90)], "tg_tps")[0]
    assert strict.noisy


def test_provenance_records_what_makes_a_number_retractable():
    p = provenance({"model": "spark2_5-int4-g128"})
    assert p["model"] == "spark2_5-int4-g128"
    assert set(p) >= {"recorded_at", "host", "commits", "env", "n_affinity"}
    assert set(p["commits"]) == {"embedding_infer", "vllm_omni", "vllm"}


def test_scrapes_the_last_json_object_a_run_printed():
    from bench_harness import _scrape

    stdout = 'INFO noise {"not": "it"}\nmore logs\n{"tg_tps": 86.6, "pp_tps": 3347}\n'
    assert _scrape(stdout, ["tg_tps", "pp_tps"]) == {"tg_tps": 86.6, "pp_tps": 3347.0}


def test_end_to_end_against_a_real_subprocess(tmp_path):
    """The arm really is a subprocess, so a trivial one must round-trip."""
    script = tmp_path / "emit.py"
    script.write_text('print("noise"); print(\'{"tg_tps": 42.0}\')')
    arm = Arm("trivial", ["python3", str(script)])
    obs = run_grid([arm], passes=2, metric="tg_tps", timeout=60,
                   keys=["tg_tps"], settle_s=0)
    assert all(o.ok for o in obs)
    s = summarize(obs, "tg_tps")[0]
    assert s.best == 42.0 and s.n == 2 and s.spread_pct == 0.0


def test_a_failing_arm_is_recorded_with_its_error(tmp_path):
    arm = Arm("broken", ["python3", "-c", "import sys; sys.exit(3)"])
    obs = run_grid([arm], passes=1, metric="tg_tps", timeout=60,
                   keys=["tg_tps"], settle_s=0)
    assert obs[0].ok is False
    assert summarize(obs, "tg_tps") == []


# ------------------------------------------------------- contention guard

def test_parse_cores_handles_ranges_and_singletons():
    assert parse_cores("0-3,8") == [0, 1, 2, 3, 8]
    assert parse_cores("5") == [5]
    assert parse_cores("2-1") == []          # empty range, not an error
    assert parse_cores("") == []


def test_core_utilization_reports_the_requested_cores():
    """Load average is a whole-host number and this host has 224 cores, so it
    says nothing about the 24 a run is pinned to."""
    usage = core_utilization([0, 1], window_s=0.05)
    assert set(usage) <= {0, 1}
    assert all(0.0 <= v <= 100.0 for v in usage.values())


def test_preflight_refuses_a_sustained_straggler(monkeypatch):
    """One heavily loaded core gates every OpenMP barrier, so it costs far more
    than its 1/24 share of capacity."""
    monkeypatch.setattr("bench_harness.core_utilization",
                        lambda cores, window_s=0.75: {0: 3.0, 1: 91.0})
    check = preflight([0, 1])
    assert check["clear"] is False
    assert check["busy_cores"] == {1: 91.0}


def test_preflight_passes_on_idle_cores(monkeypatch):
    monkeypatch.setattr("bench_harness.core_utilization",
                        lambda cores, window_s=0.75: {0: 1.0, 1: 2.0})
    assert preflight([0, 1])["clear"] is True


def test_preflight_ignores_a_transient_spike(monkeypatch):
    """A flat per-core threshold over one window refused three consecutive
    launches on cores that were idle seconds earlier. Load is the minimum of
    two windows, so a blip in one of them is not load."""
    windows = iter([{0: 1.0, 1: 30.0}, {0: 1.0, 1: 1.0}])
    monkeypatch.setattr("bench_harness.core_utilization",
                        lambda cores, window_s=0.75: next(windows))
    check = preflight([0, 1])
    assert check["clear"] is True
    assert check["utilization"][1] == 1.0


def test_preflight_refuses_when_aggregate_capacity_is_stolen(monkeypatch):
    """No single straggler, but a third of every core is someone else's."""
    monkeypatch.setattr("bench_harness.core_utilization",
                        lambda cores, window_s=0.75: {c: 33.0 for c in cores})
    check = preflight(list(range(24)))
    assert check["clear"] is False
    assert check["stolen_pct"] == 33.0
    assert check["busy_cores"] == {}          # none individually over 50%


def test_one_blipping_core_out_of_many_does_not_block_a_run(monkeypatch):
    """1 core of 24 at 30% is ~1.25% of capacity; refusing on that makes the
    guard useless rather than strict."""
    monkeypatch.setattr("bench_harness.core_utilization",
                        lambda cores, window_s=0.75: {c: (30.0 if c == 11 else 1.0)
                                                      for c in cores})
    assert preflight(list(range(24)))["clear"] is True


def test_host_lock_excludes_a_second_benchmark(tmp_path):
    """Two sessions benchmarking the same cores wrecked several measurements."""
    path = str(tmp_path / "bench.lock")
    with HostLock(path, note="first"):
        with pytest.raises(SystemExit) as excinfo:
            with HostLock(path, note="second"):
                pass
        assert "another benchmark holds" in str(excinfo.value)
    # released again afterwards
    with HostLock(path, note="third"):
        pass


def test_host_lock_records_who_holds_it(tmp_path):
    path = str(tmp_path / "bench.lock")
    with HostLock(path, note="peak_rss_anon_mb"):
        held = open(path).read()
    assert "peak_rss_anon_mb" in held and str(os.getpid()) in held


# ------------------------------------------------------- regression gates

def _summary(tag, metric, median, lower):
    from bench_harness import ArmSummary
    return ArmSummary(tag=tag, metric=metric, best=median, median=median,
                      spread_pct=0.0, n=3, n_busy=0, lower_is_better=lower)


EXPECT = {"arms": {"w4a16": {"rss_anon_mb": 3541.8, "tg_tps": 80.0}},
          "bands_pct": {"rss_anon_mb": 1.5}}


def test_memory_regression_outside_the_band_fails():
    rows = check_expectations([_summary("w4a16", "rss_anon_mb", 3900.0, True)], EXPECT)
    assert rows[0]["status"] == "REGRESSION"
    assert rows[0]["drift_pct"] > 1.5


def test_memory_inside_the_band_passes():
    rows = check_expectations([_summary("w4a16", "rss_anon_mb", 3560.0, True)], EXPECT)
    assert rows[0]["status"] == "ok"


def test_a_large_improvement_is_not_reported_as_a_regression():
    rows = check_expectations([_summary("w4a16", "rss_anon_mb", 2800.0, True)], EXPECT)
    assert rows[0]["status"] == "improved"


def test_throughput_is_not_gated_because_this_host_cannot_resolve_it():
    """tg_tps swings ~30% run to run here, so any band wide enough not to fire
    constantly is too wide to catch a real regression."""
    rows = check_expectations([_summary("w4a16", "tg_tps", 55.0, False)], EXPECT)
    assert rows[0]["status"] == "ungateable"


def test_an_unlisted_arm_is_reported_not_silently_passed():
    rows = check_expectations([_summary("brand-new", "rss_anon_mb", 100.0, True)], EXPECT)
    assert rows[0]["status"] == "unlisted"


# ---------------------------------------- refusing to rank inside the noise

def test_a_difference_inside_the_noise_is_not_presented_as_a_ranking():
    """An 8% tg_tps gap on a host that swings 30% is not a result; publishing
    one is how the W4A8/W4A16 ordering flipped between grids."""
    rows = [_obs("a", 1, 86.0), _obs("a", 2, 80.0),
            _obs("b", 1, 84.0), _obs("b", 2, 78.0)]
    text = format_summary(summarize(rows, "tg_tps"))
    assert "NOT A RANKING" in text
    assert "profile_cpu_step.py" in text


def test_a_real_difference_is_still_reported_as_one():
    rows = [_obs("fast", 1, 86.0), _obs("fast", 2, 86.2),
            _obs("slow", 1, 41.0), _obs("slow", 2, 41.1)]
    text = format_summary(summarize(rows, "tg_tps"))
    assert "NOT A RANKING" not in text


def test_memory_arms_that_really_differ_are_ranked():
    rows = [Observation("small", p, {"rss_anon_mb": v}, 1.0, 1.0, 1.0, True)
            for p, v in ((1, 2802.0), (2, 2814.0))]
    rows += [Observation("big", p, {"rss_anon_mb": v}, 1.0, 1.0, 1.0, True)
             for p, v in ((1, 3541.0), (2, 3542.0))]
    assert indistinguishable(summarize(rows, "rss_anon_mb")) is None
