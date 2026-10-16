# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CI wiring and failure propagation, without substituting model inference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from benchmarks.lingbot_world.regression import check_quality

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


def leaves(document):
    for step in document.get("steps", []):
        if "steps" in step:
            yield from leaves(step)
        else:
            yield step


@pytest.mark.parametrize("platform", ["cuda", "npu"])
@pytest.mark.parametrize("level", ["nightly", "weekly"])
def test_generation_and_gpu_evaluation_are_blocking_and_connected(platform, level):
    name = f"test-{level}.yml" if platform == "cuda" else f"test-npu-{level}.yml"
    steps = list(leaves(yaml.safe_load((ROOT / f".buildkite/{platform}/{name}").read_text())))
    keyed = {step.get("key"): step for step in steps}
    prefix = f"lingbot-{platform}-{level}"
    generate, evaluate = keyed[prefix + "-generation"], keyed[prefix + "-quality"]
    assert prefix + "-generation" in evaluate["depends_on"]
    assert evaluate["mirror_hardwares"] == "h100_1"
    for step in (generate, evaluate):
        assert not step.get("soft_fail")
        assert step["artifact_paths"]
        assert step["source_file_dependencies"] == "lingbot_world_regression"
    if platform == "npu":
        assert prefix + "-evaluator-image" in evaluate["depends_on"]
    if level == "weekly":
        assert prefix + "-long" in keyed


def test_selected_ci_skip_is_a_failed_pytest_job(tmp_path: Path):
    test = tmp_path / "test_missing_device.py"
    test.write_text('import pytest\ndef test_missing_device():\n    pytest.skip("missing device")\n')
    output = tmp_path / "artifacts"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tests.dfx.lingbot_world.conftest",
            str(test),
            "--rootdir",
            str(tmp_path),
            "--confcutdir",
            str(tmp_path),
            "--lingbot-output",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "BUILDKITE": "true", "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    reports = list(output.glob("*-failure.json"))
    assert reports and "missing device" in reports[0].read_text()


def test_activity_is_not_a_higher_is_better_quality_score():
    result = check_quality(
        [{"dynamic_degree": 0.1}] * 3,
        {"dynamic_degree": 0.9},
        tolerances={"dynamic_degree": 0.1},
        activity_dimensions=("dynamic_degree",),
    )
    assert not result["passed"]


def test_launcher_setup_failure_retains_nonzero_exit_and_junit(tmp_path: Path):
    output = tmp_path / "artifacts"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / ".buildkite/common/scripts/run-lingbot-ci.sh"),
            "generate",
            str(tmp_path / "missing-profile.json"),
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert ET.parse(output / "pytest.xml").find("testcase/failure") is not None
    assert not json.loads((output / "setup-failure.json").read_text())["passed"]


def test_profiles_pin_inputs_and_use_distinct_reference_starts():
    directory = ROOT / "tests/dfx/lingbot_world"
    lock = json.loads((directory / "assets.lock.json").read_text())
    for path in (directory / "configs").glob("*.json"):
        profile = json.loads(path.read_text())
        assert len(profile["reference_revision"]) == 40
        assert profile["model_revision"] == lock["model_revision"]
        assert profile["num_chunks"] > profile["warmup_chunks"] + 1
        assert len({case["id"] for case in profile["cases"]}) == len(profile["cases"])
        assert all(case["image"] in lock["images"] for case in profile["cases"])
        assert (ROOT / profile["deploy_config"]).is_file()
    assert len(lock["model_files"]) >= 20
