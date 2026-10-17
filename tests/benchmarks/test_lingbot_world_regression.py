# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A completed benchmark is not a passing performance or quality regression."""

from __future__ import annotations

import copy
import importlib.metadata
import math
import sys
from pathlib import Path

import pytest

from benchmarks.lingbot_world.regression import (
    RegressionError,
    calibrate,
    check_binary_activity,
    check_performance,
    check_quality,
    check_video_precision,
    performance_metrics,
    read_json,
    sha256,
    verify_environment,
    verify_reference_calibration_contract,
    write_json,
)
from benchmarks.lingbot_world.workload import ChunkRecord, compute_metrics

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def document(interval=0.5, num_chunks=40):
    records = [ChunkRecord(i, (i + 1) * interval, interval, 1024, 9 if i == 0 else 12) for i in range(num_chunks)]
    config = {"target_fps": 12, "warmup_chunks": 6, "chunk_slo_ms": 1000, "playback_buffer_chunks": 1}
    metrics = compute_metrics(records, fps=12, warmup_chunks=6, slo_ms=1000, ttfc_s=interval)
    return {"config": config, "sessions": [{"chunks": [r.to_dict() for r in records], "metrics": metrics}]}


def baseline():
    measured = performance_metrics(document(), num_chunks=40, min_steady=30)
    return calibrate([measured] * 3, contract={"sku": "H100", "cards": 1}, reference_revision="a" * 40)


def test_successful_but_slow_benchmark_fails():
    measured = performance_metrics(document(interval=0.7), num_chunks=40, min_steady=30)
    result = check_performance(measured, baseline(), contract={"sku": "H100", "cards": 1})
    assert not result["passed"]
    assert any("mean_ms" in reason for reason in result["violations"])


@pytest.mark.parametrize("change", ["missing", "reordered", "frames", "nonfinite", "interval", "wall"])
def test_invalid_population_cannot_pass(change):
    data = document()
    chunks = data["sessions"][0]["chunks"]
    if change == "missing":
        chunks.pop()
    elif change == "reordered":
        chunks[1], chunks[2] = chunks[2], chunks[1]
    elif change == "frames":
        chunks[0]["num_frames"] = 12
    elif change == "nonfinite":
        chunks[7]["arrival_s"] = float("nan")
    elif change == "interval":
        chunks[7]["inter_arrival_ms"] = 0.01
    else:
        data["sessions"][0]["metrics"]["wall_seconds"] = 1
    with pytest.raises(RegressionError):
        performance_metrics(data, num_chunks=40, min_steady=30)


def test_raw_chunk_times_override_forged_aggregate():
    data = document(interval=0.8)
    data["sessions"][0]["metrics"]["interval_steady"]["mean_ms"] = 1
    assert performance_metrics(data, num_chunks=40, min_steady=30)["mean_ms"] == pytest.approx(800)


def test_baseline_is_tied_to_hardware_and_configuration():
    measured = performance_metrics(document(), num_chunks=40, min_steady=30)
    with pytest.raises(RegressionError, match="contract mismatch"):
        check_performance(measured, baseline(), contract={"sku": "910B1", "cards": 4})


def test_calibration_rejects_insufficient_or_noisy_reference():
    measured = performance_metrics(document(), num_chunks=40, min_steady=30)
    with pytest.raises(RegressionError, match="three independent"):
        calibrate([measured] * 2, contract={}, reference_revision="a" * 40)
    noisy = copy.deepcopy(measured)
    noisy["mean_ms"] *= 2
    with pytest.raises(RegressionError, match="Unstable reference"):
        calibrate([measured, measured, noisy], contract={}, reference_revision="a" * 40)


def test_short_run_reports_p99_without_gating_it():
    measured = performance_metrics(document(), num_chunks=40, min_steady=30)
    assert measured["p99_ms"] == 500
    assert "p99_ms" not in baseline()["limits"]
    gate = baseline()
    gate["limits"]["p99_ms"] = {"direction": "max", "value": 1000}
    with pytest.raises(RegressionError, match="1000 steady"):
        check_performance(measured, gate, contract=gate["contract"])


def test_quality_regression_cannot_hide_behind_improved_aesthetics():
    reference = [{"imaging": 70, "aesthetic": 0.5}] * 3
    result = check_quality(reference, {"imaging": 60, "aesthetic": 0.9}, tolerances={"imaging": 1, "aesthetic": 0.01})
    assert not result["passed"]
    assert result["dimensions"]["aesthetic"]["passed"]
    assert not result["dimensions"]["imaging"]["passed"]


def test_quality_requires_repeatability_and_complete_dimensions():
    with pytest.raises(RegressionError, match="Unstable reference"):
        check_quality([{"imaging": v} for v in (70, 65, 72)], {"imaging": 71}, tolerances={"imaging": 1})
    with pytest.raises(RegressionError, match="exact dimension"):
        check_quality([{"imaging": 70}] * 3, {}, tolerances={"imaging": 1})


def test_nonfinite_baseline_json_is_rejected(tmp_path: Path):
    path = tmp_path / "baseline.json"
    path.write_text('{"mean": NaN}')
    with pytest.raises(RegressionError, match="Nonfinite"):
        read_json(path)


def test_evaluator_dependency_lock_is_checked_in_selected_interpreter(tmp_path: Path):
    path = tmp_path / "requirements.txt"
    version = importlib.metadata.version("pytest")
    path.write_text(f"pytest=={version}\n")
    assert verify_environment(Path(sys.executable), path) == {"pytest": version}
    path.write_text("pytest==0.0.0\n")
    with pytest.raises(RegressionError, match="environment differs"):
        verify_environment(Path(sys.executable), path)


def test_unpinned_evaluator_dependencies_are_rejected(tmp_path: Path):
    path = tmp_path / "requirements.txt"
    path.write_text("pytest>=8\n")
    with pytest.raises(RegressionError, match="not exactly pinned"):
        verify_environment(Path(sys.executable), path)


@pytest.mark.parametrize(
    "metric,reference_values,candidate_value",
    [
        ("ttfc_ms", [1501.201, 1458.253, 1549.494], 1646.096),
        ("stall_ms", [159.172, 112.634, 90.709], 201.321),
    ],
)
def test_h100_transient_jitter_uses_only_independent_reference_values(metric, reference_values, candidate_value):
    # Actual H100 build 17394 transients; cadence remains subject to the same
    # 3% CV and 5% regression rules. Candidate values never enter calibration.
    row = performance_metrics(document(), num_chunks=40, min_steady=30)
    references = [dict(row, **{metric: value}) for value in reference_values]
    gate = calibrate(references, contract={}, reference_revision="a" * 40)
    strict = calibrate(references, contract={}, reference_revision="a" * 40, transient_noise_sigma=0)
    candidate = dict(row, **{metric: candidate_value})
    assert not check_performance(candidate, strict, contract={})["passed"]
    assert check_performance(candidate, gate, contract={})["passed"]
    original = copy.deepcopy(gate)
    candidate[metric] = gate["limits"][metric]["value"] + 1
    assert not check_performance(candidate, gate, contract={})["passed"]
    candidate[metric] = candidate_value
    candidate["mean_ms"] *= 1.06
    assert not check_performance(candidate, gate, contract={})["passed"]
    assert gate == original


@pytest.mark.parametrize("sigma", [-1, float("nan"), 3.01])
def test_transient_noise_multiplier_is_finite_nonnegative_and_bounded(sigma):
    row = performance_metrics(document(), num_chunks=40, min_steady=30)
    with pytest.raises(RegressionError):
        calibrate([row] * 3, contract={}, reference_revision="a" * 40, transient_noise_sigma=sigma)


def test_recorded_reference_startup_noise_is_covered_without_a_steady_stall_allowance():
    root = Path(__file__).resolve().parents[2] / "tests/dfx/lingbot_world"
    evidence = read_json(root / "calibration/cuda_nightly_reference.json")["startup_stall_calibration"]
    required = 0
    base = performance_metrics(document(), num_chunks=40, min_steady=30)
    for cohorts in evidence["cases"].values():
        for source, heldout in ((cohorts[0], cohorts[1]), (cohorts[1], cohorts[0])):
            assert len(source) == len(heldout) == 3
            assert all(row["steady_stall_ms"] == 0 for row in [*source, *heldout])
            required = max(
                required,
                max(row["startup_stall_ms"] for row in heldout) - max(row["startup_stall_ms"] for row in source),
            )
            reference = [
                dict(
                    base,
                    stall_ms=row["startup_stall_ms"],
                    **{key: row[key] for key in ("startup_stall_ms", "steady_stall_ms")},
                )
                for row in source
            ]
            gate = calibrate(
                reference,
                contract={},
                reference_revision="a" * 40,
                startup_stall_jitter_floor_ms=evidence["selected_floor_ms"],
            )
            for row in heldout:
                measured = dict(
                    base,
                    stall_ms=row["startup_stall_ms"],
                    **{key: row[key] for key in ("startup_stall_ms", "steady_stall_ms")},
                )
                assert check_performance(measured, gate, contract={})["passed"]
            assert gate["limits"]["steady_stall_ms"]["value"] == 0
    assert required == pytest.approx(evidence["maximum_reference_startup_excess_ms"])
    assert evidence["selected_floor_ms"] == math.ceil(required * 1.25)


def test_startup_noise_allowance_cannot_hide_a_later_stall_or_forged_summary():
    def measured(first_gap_jitter=0.0, later_gap_jitter=0.0):
        doc = document(interval=0.8)
        session = doc["sessions"][0]
        for index, row in enumerate(session["chunks"]):
            row["arrival_s"] += (first_gap_jitter if index >= 1 else 0) + (later_gap_jitter if index >= 6 else 0)
            previous = session["chunks"][index - 1]["arrival_s"] if index else 0
            row["inter_arrival_ms"] = (row["arrival_s"] - previous) * 1000
        session["metrics"]["wall_seconds"] = session["chunks"][-1]["arrival_s"]
        session["metrics"]["playback"].update(total_stall_ms=0, startup_stall_ms=0, steady_stall_ms=0)
        return performance_metrics(doc, num_chunks=40, min_steady=30)

    reference = [measured(jitter) for jitter in (0, 0.02, 0.04)]
    candidate = measured(later_gap_jitter=1.02)
    legacy = calibrate(reference, contract={}, reference_revision="a" * 40)
    assert check_performance(candidate, legacy, contract={})["passed"]
    gate = calibrate(reference, contract={}, reference_revision="a" * 40, startup_stall_jitter_floor_ms=135)
    assert candidate["steady_stall_ms"] == pytest.approx(20)
    check = check_performance(candidate, gate, contract={})
    assert not check["passed"]
    assert len(check["violations"]) == 1 and "steady_stall_ms" in check["violations"][0]
    startup_only = measured(first_gap_jitter=0.1)
    assert check_performance(startup_only, gate, contract={})["passed"]
    startup_only["stall_ms"] = gate["limits"]["stall_ms"]["value"] + 1
    assert not check_performance(startup_only, gate, contract={})["passed"]


@pytest.mark.parametrize("metric", ["mean_ms", "p50_ms", "p95_ms", "p99_ms", "ttfc_ms", "generated_fps"])
def test_startup_noise_floor_does_not_change_other_performance_limits(metric):
    row = performance_metrics(document(num_chunks=1020), num_chunks=1020, min_steady=1000)
    gate = calibrate([row] * 3, contract={}, reference_revision="a" * 40, startup_stall_jitter_floor_ms=135)
    original = calibrate([row] * 3, contract={}, reference_revision="a" * 40)
    assert gate["limits"][metric] == original["limits"][metric]
    candidate = dict(row)
    limit = gate["limits"][metric]
    candidate[metric] = limit["value"] + (1 if limit["direction"] == "max" else -1)
    assert not check_performance(candidate, gate, contract={})["passed"]


@pytest.mark.parametrize("floor", [-1, True, float("nan"), float("inf")])
def test_invalid_startup_stall_floor_fails_closed(floor):
    row = performance_metrics(document(), num_chunks=40, min_steady=30)
    with pytest.raises(RegressionError):
        calibrate([row] * 3, contract={}, reference_revision="a" * 40, startup_stall_jitter_floor_ms=floor)


def test_startup_stall_floor_requires_separate_raw_steady_measurements():
    row = performance_metrics(document(), num_chunks=40, min_steady=30)
    del row["steady_stall_ms"]
    with pytest.raises(RegressionError, match="separately recomputed steady"):
        calibrate([row] * 3, contract={}, reference_revision="a" * 40, startup_stall_jitter_floor_ms=135)


def test_regression_rejects_a_changed_playback_buffer():
    from tests.dfx.lingbot_world.runner import _measured_metrics

    doc = document()
    doc["sessions"][0]["request_id"] = "test-session"
    doc["sessions"][0]["session_index"] = 0
    doc["config"]["playback_buffer_chunks"] = 2
    with pytest.raises(RegressionError, match="one-chunk playback buffer"):
        _measured_metrics(doc, {"num_chunks": 40, "min_steady_samples": 30, "target_fps": 12, "warmup_chunks": 6})


@pytest.mark.parametrize(
    "reference,candidate,passed",
    [
        ([0, 1, 0], [0, 1, 0], True),
        ([0, 1, 0], [1, 0, 0], True),
        ([0, 0, 0], [0, 1, 0], True),
        ([0, 0, 0], [1, 1, 0], False),
        ([1, 1, 1], [1, 0, 0], False),
        ([0, 1, 0], [1, 1, 1], False),
        ([0, 0, 0], [1, 1, 1], False),
        ([1, 1, 1], [0, 0, 0], False),
    ],
)
def test_binary_activity_keeps_one_trial_budget_and_rejects_larger_changes(reference, candidate, passed):
    # Actual H100 build 17431 repeats are [0, 1, 0] in both roles.
    result = check_binary_activity(reference, candidate, tolerance=0.34)
    assert result["passed"] is passed
    assert result["allowed_count_change"] == 1
    assert result["reference"] == reference and result["candidate"] == candidate
    strict = check_binary_activity(reference, candidate, tolerance=0)
    assert strict["passed"] is (sum(reference) == sum(candidate))


@pytest.mark.parametrize("invalid", [[], [0, 1], [0, 1, 0, 1], [0, 0.5, 1], [0, float("nan"), 1], [0, True, 1]])
def test_binary_activity_rejects_invalid_native_classifications_or_missing_starts(invalid):
    for reference, candidate in ((invalid, [0, 0, 0]), ([0, 0, 0], invalid)):
        with pytest.raises(RegressionError):
            check_binary_activity(reference, candidate, tolerance=0.34)


@pytest.mark.parametrize("tolerance", [-1, 1, float("inf"), float("nan"), True])
def test_binary_activity_budget_cannot_disable_the_gate(tolerance):
    with pytest.raises(RegressionError):
        check_binary_activity([0, 0, 0], [1, 1, 1], tolerance=tolerance)


def test_recorded_independent_h100_reference_cohorts_fit_the_binary_noise_budget():
    root = Path(__file__).resolve().parents[2] / "tests/dfx/lingbot_world"
    record = read_json(root / "calibration/cuda_nightly_reference.json")
    first, second = record["reference_cohorts"]
    assert first["reference_scores"].keys() == second["reference_scores"].keys()
    tolerance = read_json(root / "quality.json")["candidate_tolerances"]["cuda"]["vbench"]["dynamic_degree"]
    changes = []
    for case, scores in first["reference_scores"].items():
        check = check_binary_activity(
            [row["dynamic_degree"] for row in scores],
            [row["dynamic_degree"] for row in second["reference_scores"][case]],
            tolerance=tolerance,
        )
        assert check["passed"], case
        changes.append(abs(check["reference_active_count"] - check["candidate_active_count"]))
    assert max(changes) == 1


def test_cuda_quality_ceilings_reproduce_reference_only_calibration_and_keep_npu_separate():
    root = Path(__file__).resolve().parents[2]
    suite = root / "tests/dfx/lingbot_world"
    record = read_json(suite / "calibration/cuda_nightly_reference.json")
    limits = read_json(suite / "quality.json")
    assert record["workload"] == read_json(suite / "configs/cuda_nightly.json")
    assert record["deploy_sha256"] == sha256(root / record["workload"]["deploy_config"])
    assert record["evaluator_lock_sha256"] == sha256(suite / "evaluators.lock.json")
    assert record["asset_lock_sha256"] == sha256(suite / "assets.lock.json")
    assert record["evaluator_requirements_sha256"] == sha256(root / "requirements/lingbot-quality.txt")
    calibration = record["continuous_reference_calibration"]
    for name, cohorts in (("vbench", record["reference_cohorts"]), ("wbench", record["wbench_reference_cohorts"])):
        for dimension, evidence in calibration["dimensions"][name].items():
            ranges = []
            for case in cohorts[0]["reference_scores"]:
                values = [row[dimension] for cohort in cohorts for row in cohort["reference_scores"][case]]
                assert len(values) == 6
                ranges.append(max(values) - min(values))
            rounded = math.ceil(max(ranges) * (1 + calibration["margin_fraction"]) / calibration["round_up_step"])
            expected = max(evidence["previous_ceiling"], rounded * calibration["round_up_step"])
            if name == "wbench" and dimension == "hpsv3_quality":
                expected = max(expected, record["target_h100_hps_reference_calibration"]["selected_ceiling"])
            if name == "wbench" and dimension == "imaging_quality":
                expected = max(expected, record["target_h100_long_imaging_reference_calibration"]["selected_ceiling"])
            assert limits["max_reference_spreads"]["cuda"][name][dimension] == pytest.approx(expected)
            assert limits["max_reference_spreads"]["npu"][name][dimension] == evidence["previous_ceiling"]


def test_h100_hps_reference_ceiling_uses_only_reference_scores_and_preserves_candidate_budget(tmp_path):
    suite = Path(__file__).resolve().parents[2] / "tests/dfx/lingbot_world"
    record = read_json(suite / "calibration/cuda_nightly_reference.json")
    evidence = record["target_h100_hps_reference_calibration"]
    limits = read_json(suite / "quality.json")
    assert evidence["evaluator_hardware"]["devices"][0]["name"] == "NVIDIA H100 80GB HBM3"
    assert evidence["reference_revision"] == record["reference_revision"]
    assert evidence["evaluator_lock_sha256"] == record["evaluator_lock_sha256"]
    profile = tmp_path / "profile.json"
    write_json(profile, record["workload"])
    assert sha256(profile) == evidence["profile_sha256"]
    ranges = []
    tolerance = limits["candidate_tolerances"]["cuda"]["wbench"]["hpsv3_quality"]
    assert tolerance == evidence["unchanged_candidate_tolerance"] == 0.02
    for case, scores in evidence["reference_scores"].items():
        values = scores["scores"]
        assert len(values) == len(scores["video_sha256"]) == evidence["independent_reference_starts"] == 3
        spread = max(values) - min(values)
        assert spread == evidence["reference_ranges"][case]
        ranges.append(spread)
        reference = [{"hpsv3_quality": value} for value in values]
        options = {
            "tolerances": {"hpsv3_quality": tolerance},
            "max_reference_spreads": {"hpsv3_quality": evidence["selected_ceiling"]},
        }
        minimum = min(values)
        assert check_quality(reference, {"hpsv3_quality": minimum}, **options)["passed"]
        assert not check_quality(reference, {"hpsv3_quality": minimum - tolerance - 0.001}, **options)["passed"]
        if case == "boardwalk-42":
            with pytest.raises(RegressionError, match="Unstable reference"):
                check_quality(
                    reference,
                    {"hpsv3_quality": minimum},
                    tolerances=options["tolerances"],
                    max_reference_spreads={"hpsv3_quality": evidence["previous_ceiling"]},
                )
    selected = (
        math.ceil(max(ranges) * (1 + evidence["margin_fraction"]) / evidence["round_up_step"])
        * evidence["round_up_step"]
    )
    assert selected == pytest.approx(evidence["selected_ceiling"])
    assert limits["max_reference_spreads"]["cuda"]["wbench"]["hpsv3_quality"] == evidence["selected_ceiling"]
    assert limits["max_reference_spreads"]["npu"]["wbench"]["hpsv3_quality"] == evidence["previous_ceiling"]


def test_h100_long_imaging_calibration_preserves_candidate_budget_and_rejects_unstable_references():
    root = Path(__file__).resolve().parents[2]
    suite = root / "tests/dfx/lingbot_world"
    record = read_json(suite / "calibration/cuda_nightly_reference.json")
    evidence = record["target_h100_long_imaging_reference_calibration"]
    limits = read_json(suite / "quality.json")
    assert evidence["profile_sha256"] == sha256(root / evidence["profile"])
    assert evidence["deploy_sha256"] == record["deploy_sha256"]
    assert evidence["reference_revision"] == record["reference_revision"]
    assert evidence["evaluator_lock_sha256"] == record["evaluator_lock_sha256"]
    assert evidence["evaluator_requirements_sha256"] == record["evaluator_requirements_sha256"]
    assert evidence["evaluator_hardware"]["devices"][0]["name"] == "NVIDIA H100 80GB HBM3"
    assert evidence["evaluator_torch_version"] == "2.4.0+cu124"
    values = evidence["reference_scores"]["boardwalk-42"]["scores"]
    assert len(values) == evidence["independent_reference_starts"] == 3
    spread = max(values) - min(values)
    assert spread == evidence["maximum_reference_range"]
    expected = math.ceil(spread * (1 + evidence["margin_fraction"]) / evidence["round_up_step"])
    assert expected * evidence["round_up_step"] == pytest.approx(evidence["selected_ceiling"])
    dimension = "imaging_quality"
    tolerance = limits["candidate_tolerances"]["cuda"]["wbench"][dimension]
    assert tolerance == evidence["unchanged_candidate_tolerance"] == 0.01
    assert limits["max_reference_spreads"]["cuda"]["wbench"][dimension] == evidence["selected_ceiling"]
    assert limits["max_reference_spreads"]["npu"]["wbench"][dimension] == 0.01
    reference = [{dimension: value} for value in values]
    with pytest.raises(RegressionError, match="Unstable reference"):
        check_quality(
            reference,
            {dimension: min(values)},
            tolerances={dimension: tolerance},
            max_reference_spreads={dimension: evidence["previous_ceiling"]},
        )
    options = {
        "tolerances": {dimension: tolerance},
        "max_reference_spreads": {dimension: evidence["selected_ceiling"]},
    }
    assert check_quality(reference, {dimension: min(values)}, **options)["passed"]
    assert not check_quality(reference, {dimension: min(values) - tolerance - 0.001}, **options)["passed"]
    with pytest.raises(RegressionError, match="Unstable reference"):
        check_quality([{dimension: value} for value in (0.8, 0.82, 0.84)], {dimension: 0.8}, **options)


def test_aesthetic_budget_comes_from_reference_holdouts_and_still_rejects_degradation():
    root = Path(__file__).resolve().parents[2] / "tests/dfx/lingbot_world"
    record = read_json(root / "calibration/cuda_nightly_reference.json")
    limits = read_json(root / "quality.json")
    evidence = record["candidate_budget_calibration"]["cuda_vbench_aesthetic_quality"]
    tolerance = limits["candidate_tolerances"]["cuda"]["vbench"]["aesthetic_quality"]
    first, second = record["reference_cohorts"]
    deficits = []
    for source, heldout in ((first, second), (second, first)):
        for case, scores in source["reference_scores"].items():
            reference = [{"aesthetic_quality": row["aesthetic_quality"]} for row in scores]
            options = {
                "tolerances": {"aesthetic_quality": tolerance},
                "max_reference_spreads": {"aesthetic_quality": 0.03},
            }
            minimum = min(row["aesthetic_quality"] for row in scores)
            for row in heldout["reference_scores"][case]:
                deficits.append(max(0, minimum - row["aesthetic_quality"]))
                assert check_quality(reference, {"aesthetic_quality": row["aesthetic_quality"]}, **options)["passed"]
            assert not check_quality(reference, {"aesthetic_quality": minimum - tolerance - 0.001}, **options)["passed"]
    assert max(deficits) > evidence["previous_tolerance"]
    assert max(deficits) == pytest.approx(evidence["maximum_reference_only_deficit"])
    assert tolerance == pytest.approx(math.ceil(max(deficits) * 1.25 / 0.005) * 0.005)
    assert limits["candidate_tolerances"]["npu"]["vbench"]["aesthetic_quality"] == evidence["previous_tolerance"]


def test_ssim_budget_covers_independent_reference_holdouts_and_remains_bounded():
    root = Path(__file__).resolve().parents[2] / "tests/dfx/lingbot_world"
    evidence = read_json(root / "calibration/cuda_nightly_reference.json")["candidate_budget_calibration"][
        "cuda_pixels"
    ]
    limits = read_json(root / "quality.json")
    budget = limits["pixel_budgets"]["cuda"]

    def row(ssim):
        distance = {"psnr_db": 40.0, "psnr_is_infinite": False}
        return {
            "total": distance,
            "windows": [{"index": 0, "start_frame": 0, "frame_count": 32, **distance}],
            "ssim": {"mean": ssim},
        }

    deficits = []
    for samples in evidence["reference_ssim_samples"].values():
        reference = [row(value) for value in samples["reference_ssim"]]
        minimum = min(samples["reference_ssim"])
        for value in samples["heldout_reference_ssim"]:
            deficits.append(max(0, minimum - value))
            assert check_video_precision(reference, row(value), min_psnr_db=40, **budget)["passed"]
        assert not check_video_precision(
            reference, row(minimum - budget["ssim_drop"] - 0.001), min_psnr_db=40, **budget
        )["passed"]
    assert max(deficits) > evidence["previous_ssim_drop"]
    assert max(deficits) == pytest.approx(evidence["maximum_reference_only_ssim_deficit"])
    assert budget["ssim_drop"] == pytest.approx(math.ceil(max(deficits) * 1.25 / 0.005) * 0.005)
    assert evidence["maximum_reference_only_psnr_deficit_db"] < budget["psnr_drop_db"] == 3
    assert limits["pixel_budgets"]["npu"]["ssim_drop"] == evidence["previous_ssim_drop"]


@pytest.mark.parametrize(
    "group,field",
    [
        (None, "deploy_sha256"),
        ("profile", "reference_revision"),
        ("profile", "model_revision"),
        ("profile", "height"),
        ("profile", "width"),
        ("profile", "precision"),
        ("hardware", "platform"),
        ("hardware", "cards"),
        ("hardware", "dies"),
        ("hardware", "devices"),
        ("hardware", "packages"),
    ],
)
def test_quality_calibration_cannot_be_reused_after_numerical_contract_changes(group, field):
    root = Path(__file__).resolve().parents[2] / "tests/dfx/lingbot_world"
    record = read_json(root / "calibration/cuda_nightly_reference.json")
    contract = {"deploy_sha256": record["deploy_sha256"], "profile": record["workload"], "hardware": record["hardware"]}
    verify_reference_calibration_contract(contract, record)
    heldout = copy.deepcopy(contract)
    heldout["profile"].update(num_chunks=600, cases=[{"id": "new-heldout-scene", "seed": 123}])
    verify_reference_calibration_contract(heldout, record)
    changed = copy.deepcopy(contract)
    (changed if group is None else changed[group])[field] = "unreviewed change"
    with pytest.raises(RegressionError, match="Reference calibration"):
        verify_reference_calibration_contract(changed, record)


def test_reference_quality_spread_does_not_widen_candidate_budget():
    # H100-generated full rollout scores, obtained with the pinned VBench
    # evaluator. These diagnose calibration; target-runner acceptance is separate.
    reference = [{"background": value} for value in (0.876767968, 0.898090779, 0.894397383)]
    candidate = {"background": 0.872326250}
    with pytest.raises(RegressionError, match="Unstable reference"):
        check_quality(reference, candidate, tolerances={"background": 0.01})
    options = {"tolerances": {"background": 0.01}, "max_reference_spreads": {"background": 0.03}}
    report = check_quality(reference, candidate, **options)
    assert report["passed"]
    dimension = report["dimensions"]["background"]
    assert dimension["minimum"] == pytest.approx(0.866767968)
    assert dimension["candidate_tolerance"] == 0.01
    assert dimension["max_reference_spread"] == 0.03
    assert not check_quality(reference, {"background": 0.86}, **options)["passed"]
    with pytest.raises(RegressionError, match="Unstable reference"):
        check_quality([*reference[:2], {"background": 0.93}], candidate, **options)


@pytest.mark.parametrize("limits", [{}, {"wrong": 0.03}, {"imaging": float("nan")}, {"imaging": -1}])
def test_invalid_quality_reference_spread_limits_fail_closed(limits):
    with pytest.raises(RegressionError):
        check_quality(
            [{"imaging": 0.8}] * 3, {"imaging": 0.8}, tolerances={"imaging": 0.01}, max_reference_spreads=limits
        )
