# SPDX-License-Identifier: Apache-2.0
"""Offline tests of evidence, resume, complete-stream and timing boundaries."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.edge_harness.strata_profile import (
    RUNTIME_REVISION,
    ProfileOptions,
    cache_variant_launches,
    canonical_hash,
    check_quality,
    collect_request,
    default_suite,
    nearest_rank,
    normalize_io,
    read_records,
    require_drained_ledger,
    run_profile,
    summarize,
    validate_launch,
    validate_target,
    verify_target_files,
)


def target() -> dict:
    return {
        "target_id": "test-q2",
        "runtime": {"revision": RUNTIME_REVISION},
        "artifact_manifest": {
            "schema": "omni-weight-artifacts-v1",
            "checkpoint": "test/model",
            "revision": "a" * 40,
            "files": [
                {
                    "path": f"model-{index:05d}-of-00002.gguf",
                    "size_bytes": 1,
                    "sha256": hashlib.sha256(bytes([index])).hexdigest(),
                    "role": "weights",
                }
                for index in (1, 2)
            ],
        },
    }


class FakeDriver:
    def __init__(self, *, bad_sequence=False, fail_call=None):
        self.bad_sequence, self.fail_call = bad_sequence, fail_call
        self.calls = 0
        self.closed = False
        self.cancelled = []
        self.active = False

    async def start(self):
        return {"execution_plan": {"backend": "fake", "evidence": "test only"}}

    async def stream(self, request_id, prompt):
        assert not self.active, "more than one request became active"
        self.active = True
        self.calls += 1
        try:
            if prompt.get("max_tokens") == 512:
                await asyncio.sleep(10)
            if self.calls == self.fail_call:
                raise RuntimeError("injected request failure")
            for sequence, text in ((1, "3"), (1 if self.bad_sequence else 2, "6")):
                yield {"kind": "delta", "request_id": request_id, "sequence": sequence, "text": text}
            yield {
                "kind": "terminal",
                "request_id": request_id,
                "text": "36",
                "finished": True,
                "finish_reason": "stop",
                "stage_event": {
                    "request_id": request_id,
                    "terminal": True,
                    "seq": 1,
                    "epoch": 0,
                    "worker_generation": "fake-worker",
                },
                "metrics": {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            }
        finally:
            self.active = False

    async def cancel(self, request_id):
        self.cancelled.append(request_id)
        return {"requires_route_reload": True, "drained": True}

    def usage(self):
        return {"active_requests": int(self.active)}

    async def close(self):
        self.closed = True
        return {"drained": True}


def options(**overrides):
    return ProfileOptions(
        **dict(
            {"repeats": 1, "sustained_seconds": 0, "telemetry_interval_s": 0, "cancel_after_seconds": 0.001},
            **overrides,
        )
    )


def test_nearest_rank_keeps_units_and_rejects_nonfinite():
    assert nearest_rank(list(range(1, 21)), 0.5) == 10
    assert nearest_rank(list(range(1, 21)), 0.95) == 19
    assert nearest_rank([], 0.95) is None
    with pytest.raises(ValueError):
        nearest_rank([float("nan")], 0.5)


@pytest.mark.parametrize(
    "settings",
    [{"batch_size": 2}, {"concurrency": 2}, {"batch_size": True}, {"warmups": 0}, {"request_timeout_s": float("inf")}],
)
def test_protocol_rejects_multi_request_and_invalid_settings(settings):
    with pytest.raises(ValueError):
        options(**settings).validate()


def test_verifies_every_shard_and_rejects_incomplete_or_escaped_manifest(tmp_path):
    pinned = target()
    for index in (1, 2):
        (tmp_path / f"model-{index:05d}-of-00002.gguf").write_bytes(bytes([index]))
    receipt = verify_target_files(pinned, tmp_path)
    assert receipt["total_bytes"] == 2
    (tmp_path / "model-00002-of-00002.gguf").write_bytes(b"x")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_target_files(pinned, tmp_path)
    pinned["artifact_manifest"]["files"].pop()
    with pytest.raises(ValueError, match="missing a GGUF shard"):
        validate_target(pinned)
    pinned = target()
    pinned["artifact_manifest"]["files"][0]["path"] = "../weights.gguf"
    with pytest.raises(ValueError, match="relative POSIX"):
        validate_target(pinned)


def test_complete_request_records_chunks_without_inventing_token_timings():
    row = asyncio.run(collect_request(FakeDriver(), default_suite()["cases"][0], "measured", 0, options()))
    assert row["status"] == "completed"
    assert row["output_text"] == "36"
    assert row["output_tokens"] == 2
    assert len(row["deltas"]) == 2
    assert row["output_token_ids"] is None
    assert row["output_token_timestamps_s"] is None
    assert row["io"]["physical_ssd_read_bytes"] is None
    assert row["quality"]["passed"] is True
    assert row["full_response_s"] >= row["first_output_s"]


def test_duplicate_stream_output_fails_and_cleans_up():
    driver = FakeDriver(bad_sequence=True)
    row = asyncio.run(collect_request(driver, default_suite()["cases"][0], "measured", 0, options()))
    assert row["status"] == "failed"
    assert "duplicate or out-of-order" in row["error"]
    assert driver.cancelled == [row["request_id"]]
    assert not driver.active


def test_quality_and_reference_are_separate():
    assert check_quality('{"answer":36}', {"kind": "json_equal", "expected": {"answer": 36}})["passed"]
    result = check_quality(
        "36", {"kind": "exact_text", "expected": "36"}, {"output_text": "Thirty six", "output_token_ids": [36]}
    )
    assert result["passed"] is True
    assert result["reference"]["output_text_equal"] is False
    assert result["reference"]["token_ids_equal"] is None


def test_successful_resume_does_not_repeat_completed_measurements(tmp_path):
    out = tmp_path / "run"
    drivers = []

    def factory():
        driver = FakeDriver()
        drivers.append(driver)
        return driver

    report = asyncio.run(run_profile(target(), {}, default_suite(), out, options(), driver_factory=factory))
    assert report["status"] == "completed"
    assert report["cancel"]["status"] == "cancelled"
    assert report["summary"]["timing_protocol_complete"] is False
    assert report["summary"]["qualification"] == "not release-qualified by profiling alone"
    assert all(driver.closed for driver in drivers)
    before = read_records(out)
    asyncio.run(run_profile(target(), {}, default_suite(), out, options(), resume=True, driver_factory=factory))
    after = read_records(out)
    assert len([row for row in before if row["phase"] == "measured"]) == 3
    assert len([row for row in after if row["phase"] == "measured"]) == 3
    assert len([row for row in after if row["phase"] == "warmup"]) == 6
    with pytest.raises(ValueError, match="resume identity changed"):
        asyncio.run(
            run_profile(target(), {}, default_suite(), out, options(repeats=2), resume=True, driver_factory=factory)
        )


def test_failed_sample_is_preserved_and_retry_is_a_new_request(tmp_path):
    out = tmp_path / "run"
    initial = asyncio.run(
        run_profile(target(), {}, default_suite(), out, options(), driver_factory=lambda: FakeDriver(fail_call=2))
    )
    assert initial["status"] == "failed"
    failure = next(row for row in read_records(out) if row["status"] == "failed")
    final = asyncio.run(
        run_profile(target(), {}, default_suite(), out, options(), resume=True, driver_factory=FakeDriver)
    )
    assert final["status"] == "completed"
    retries = [row for row in read_records(out) if row["sample_key"] == failure["sample_key"]]
    assert len(retries) == 2
    assert len({row["request_id"] for row in retries}) == 2
    assert final["summary"]["failed_attempts"] == 1


def test_summary_excludes_warmup_and_downtime_and_refuses_duplicate_success():
    records = []
    for band in ("short", "medium", "long"):
        records.append({"sample_key": f"warm-{band}", "status": "completed", "phase": "warmup", "length_band": band})
        for index in range(20):
            records.append(
                {
                    "sample_key": f"measure-{band}-{index}",
                    "status": "completed",
                    "phase": "measured",
                    "length_band": band,
                    "full_response_s": index + 1,
                    "first_output_s": 0.5,
                    "quality": {"passed": True},
                }
            )
    records.append(
        {
            "sample_key": "sustained-0",
            "status": "completed",
            "phase": "sustained",
            "length_band": "medium",
            "full_response_s": 1800,
            "quality": {"passed": True},
            "started_unix": 1,
            "finished_unix": 100000,
        }
    )
    summary = summarize(records, options(), default_suite())
    assert summary["groups"]["short"]["full_response_p50_s"] == 10
    assert summary["groups"]["short"]["full_response_p95_s"] == 19
    assert summary["sustained_active_request_s"] == 1800
    assert summary["timing_protocol_complete"] is True
    with pytest.raises(ValueError, match="duplicate successful"):
        summarize(records + [records[-1]], options(), default_suite())


def test_interrupted_sustained_segments_are_not_added_into_a_thermal_pass():
    records = []
    for index in range(2):
        records.append(
            {
                "sample_key": f"sustained-{index}",
                "status": "completed",
                "phase": "sustained",
                "length_band": "medium",
                "full_response_s": 1000,
                "quality": {"passed": True},
                "sustained_segment_id": f"attempt-{index}",
            }
        )
    summary = summarize(records, options(), default_suite())
    assert summary["sustained_active_request_s"] == 2000
    assert summary["longest_uninterrupted_sustained_active_s"] == 1000
    assert summary["timing_protocol_complete"] is False


def test_manifest_identity_covers_cache_budget_and_switches():
    first = {"expert_ram_budget_bytes": 24 << 30, "speculative_tokens": 0, "prefetch": False}
    second = dict(first, expert_ram_budget_bytes=32 << 30)
    assert canonical_hash(first) != canonical_hash(second)
    assert canonical_hash(first) != canonical_hash(dict(first, prefetch=True))


def test_cache_variants_preserve_capacity_and_account_for_expert_cache():
    launch = {
        "backend": {"expert_ram_budget_bytes": 24 << 30, "context_tokens": 4096},
        "resource_budget": {"capacities": {"host_ram": 48 << 30}, "demands": {"host_ram": 32 << 30}},
    }
    variants = cache_variant_launches(launch)
    assert len(variants) == 3
    assert variants["ram-40gib"]["resource_budget"]["demands"]["host_ram"] == 48 << 30
    assert variants["ram-40gib"]["resource_budget"]["capacities"] == launch["resource_budget"]["capacities"]
    assert launch["backend"]["expert_ram_budget_bytes"] == 24 << 30


def test_launch_requires_same_checkpoint_and_all_pinned_shards():
    pinned = target()
    launch = {
        "backend": {
            "name": "external.strata.text.v1",
            "runtime_revision": RUNTIME_REVISION,
            "artifact_manifest": pinned["artifact_manifest"],
        }
    }
    validate_launch(pinned, launch)
    launch["backend"]["artifact_manifest"] = dict(pinned["artifact_manifest"], files=[])
    with pytest.raises(ValueError, match="every pinned weight shard"):
        validate_launch(pinned, launch)


def test_actual_factory_reuses_ledger_and_refuses_recovery_after_quarantine(tmp_path, monkeypatch):
    import benchmarks.edge_harness.strata_profile as module

    created = []

    class QuarantinedDriver(FakeDriver):
        def __init__(self, launch, *, resource_ledger):
            super().__init__()
            self.ledger = resource_ledger
            self.lease = None
            created.append(self)

        async def start(self):
            self.lease = self.ledger.reserve("old-worker", {"host_ram": 1})
            return {"ledger": self.ledger.snapshot()}

        async def cancel(self, request_id):
            self.ledger.release(self.lease, drained=False)
            return {"drained": False, "ledger": self.ledger.snapshot()}

        async def close(self):
            return {"ledger_after_shutdown": self.ledger.snapshot()}

    monkeypatch.setattr(module, "OmniStageDriver", QuarantinedDriver)
    pinned = target()
    launch = {
        "backend": {
            "name": "external.strata.text.v1",
            "runtime_revision": RUNTIME_REVISION,
            "artifact_manifest": pinned["artifact_manifest"],
            "max_new_tokens": 512,
        },
        "resource_budget": {"capacities": {"host_ram": 10}, "demands": {"host_ram": 1}},
    }
    report = asyncio.run(run_profile(pinned, launch, default_suite(), tmp_path / "run", options()))
    assert report["status"] == "failed"
    assert report["cancel"]["status"] == "failed"
    assert len(created) == 1
    assert report["shared_ledger_final"]["quarantined"] == ["old-worker"]
    assert report["shared_ledger_final"]["reserved"]["host_ram"] == 1
    with pytest.raises(RuntimeError, match="retains or quarantines"):
        asyncio.run(run_profile(pinned, launch, default_suite(), tmp_path / "run", options(), resume=True))
    assert len(created) == 1


def test_raw_output_tampering_is_detected_on_resume(tmp_path):
    out = tmp_path / "run"
    asyncio.run(run_profile(target(), {}, default_suite(), out, options(), driver_factory=FakeDriver))
    path = next((out / "requests").glob("*.json"))
    row = json.loads(path.read_text(encoding="utf-8"))
    row["output_text"] += " altered"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_records(out)


@pytest.mark.parametrize(
    "snapshot",
    [
        {},
        None,
        {"owners": [], "quarantined": []},
        {"owners": None, "quarantined": [], "reserved": {}},
        {"owners": [], "quarantined": [], "reserved": {"host_ram": "0"}},
        {"owners": [], "quarantined": [], "reserved": {"host_ram": -1}},
        {"owners": [], "quarantined": [], "reserved": {"host_ram": 1}},
    ],
)
def test_recovery_requires_complete_empty_ledger_proof(snapshot):
    with pytest.raises(RuntimeError, match="recovery is refused"):
        require_drained_ledger(snapshot)
    require_drained_ledger({"owners": [], "quarantined": [], "reserved": {"host_ram": 0}})


def test_io_normalization_distinguishes_logical_bytes_from_physical_ssd():
    result = normalize_io({"logical_file_read_bytes": 1_500_000, "hit_rate": 0.75, "ssd_wait_s": None})
    assert result["logical_weight_read_bytes"] == 1_500_000
    assert result["expert_cache_hit_ratio"] == 0.75
    assert result["physical_ssd_read_bytes"] is None
    assert result["io_wait_s"] is None
    assert result["cache_hits"] is None


def test_shipped_target_configs_are_pinned_complete_and_have_real_sizes():
    directory = Path(__file__).parent / "configs" / "strata"
    paths = list(directory.glob("*.json"))
    assert len(paths) == 3
    for path in paths:
        pinned = json.loads(path.read_text(encoding="utf-8"))
        validate_target(pinned)
        assert pinned["artifact_manifest"]["hash_origin"] == "declared"
        assert "upstream Hugging Face LFS" in pinned["hash_source"]
        assert all(item["size_bytes"] > 1_000_000 for item in pinned["artifact_manifest"]["files"])
