# SPDX-License-Identifier: Apache-2.0
"""Observer correctness: cached RSS is aged, unknown failures stay unknown."""
import sys
from types import SimpleNamespace

import pytest

from benchmarks.edge_harness import strata_profile as profile


@pytest.fixture
def clock():
    return SimpleNamespace(value=0.0)


def sampler(clock, collect, interval=1.0):
    return profile._ProcessTreeSample(
        collect, interval, monotonic=lambda: clock.value, wall=lambda: 1000 + clock.value
    )


def test_tree_refresh_preserves_age_and_reobserves_changed_membership(clock):
    calls = []

    def collect():
        calls.append(True)
        return 10 * len(calls)

    observer = sampler(clock, collect)
    first = observer.sample()
    clock.value = 0.5
    cached = observer.sample()
    assert len(calls) == 1
    assert cached["process_tree_sample_age_s"] == 0.5
    assert cached["process_tree_rss_sum_bytes"] == 10
    assert cached["process_tree_refreshed"] is False
    first["process_tree_rss_sum_bytes"] = 999
    assert observer.sample()["process_tree_rss_sum_bytes"] == 10
    clock.value = 1.0
    assert observer.sample()["process_tree_rss_sum_bytes"] == 20
    assert len(calls) == 2


def test_failed_refresh_does_not_reuse_success_or_retry_in_a_hot_loop(clock):
    def collect():
        if clock.value:
            raise PermissionError()
        return 42

    observer = sampler(clock, collect)
    observer.sample()
    clock.value = 1.0
    failed = observer.sample()
    assert failed["process_tree_rss_sum_bytes"] is None
    assert failed["process_tree_error"] == "PermissionError"
    clock.value = 1.1
    assert observer.sample()["process_tree_refreshed"] is False


def test_slow_collection_counts_toward_age_and_schedules_after_completion(clock):
    def collect():
        clock.value += 0.6
        return 1

    observer = sampler(clock, collect)
    row = observer.sample()
    assert row["process_tree_collection_s"] == 0.6
    assert row["process_tree_sample_age_s"] == 0.6
    clock.value = 1.0
    assert observer.sample()["process_tree_refreshed"] is False
    clock.value = 1.6
    assert observer.sample()["process_tree_refreshed"] is True


def test_zero_rss_is_not_an_unknown_observation(clock):
    row = sampler(clock, lambda: 0).sample()
    assert row["process_tree_rss_sum_bytes"] == 0
    assert row["process_tree_error"] is None


@pytest.mark.parametrize("value", [-1, True, None, float("nan")])
def test_invalid_rss_is_unknown(clock, value):
    row = sampler(clock, lambda: value).sample()
    assert row["process_tree_rss_sum_bytes"] is None
    assert row["process_tree_error"] == "ValueError"


@pytest.mark.parametrize("interval", [0, -1, True, float("inf"), float("nan"), "1"])
def test_invalid_tree_interval_is_refused(clock, interval):
    with pytest.raises(ValueError):
        sampler(clock, lambda: 0, interval)


@pytest.mark.parametrize("fail_tree", [False, True])
def test_worker_keeps_host_and_gpu_cadence_when_tree_is_cached_or_unknown(
    tmp_path, monkeypatch, clock, fail_tree
):
    calls, rows = [], []
    # Initialize real optional imports before replacing only the worker's sensors.
    observer = profile.Telemetry(tmp_path / "unused.jsonl", 0.1)
    tree_class = profile._ProcessTreeSample
    monkeypatch.setattr(
        profile, "_ProcessTreeSample",
        lambda collect, interval: tree_class(
            collect, interval, monotonic=lambda: clock.value, wall=lambda: 1000 + clock.value
        ),
    )

    class Process:
        def children(self, recursive):
            assert recursive is True
            calls.append(True)
            if fail_tree:
                raise PermissionError()
            return [Process()]

        def is_running(self):
            return True

        def memory_info(self):
            return SimpleNamespace(rss=10)

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=Process,
        virtual_memory=lambda: SimpleNamespace(total=100, available=70),
        disk_io_counters=lambda: SimpleNamespace(read_bytes=123),
        swap_memory=lambda: SimpleNamespace(used=0),
    ))
    monkeypatch.setitem(sys.modules, "pynvml", None)

    class Stop:
        steps = 0

        def is_set(self):
            return self.steps >= 6

        def wait(self, delay):
            self.steps += 1
            clock.value += delay

        def set(self):
            self.steps = 6

    class Thread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    observer.stop_event = Stop()
    monkeypatch.setattr(profile.threading, "Thread", Thread)
    monkeypatch.setattr(observer, "_write", rows.append)
    monkeypatch.setattr(observer, "sample_process_gpu", lambda: {"status": "unknown"})
    observer.start()
    report = observer.stop()
    assert len(rows) == 6 and len(calls) == 1
    assert all(row["host"]["available_ram_bytes"] == 70 for row in rows)
    assert all(row["process_gpu"] == {"status": "unknown"} for row in rows)
    assert rows[0]["host"]["process_tree_refreshed"] is True
    assert rows[-1]["host"]["process_tree_refreshed"] is False
    assert rows[-1]["host"]["process_tree_sample_age_s"] == pytest.approx(0.5)
    assert rows[-1]["host"]["process_tree_rss_sum_bytes"] == (None if fail_tree else 20)
    assert report["process_tree_refresh_interval_s"] == 1.0
    assert report["sampling_thread_stopped"] is True
    if fail_tree:
        assert report["errors"]["host_tree_sample"] == "PermissionError"
