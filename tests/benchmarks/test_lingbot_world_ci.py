# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CI wiring and failure propagation, without substituting model inference."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from benchmarks.lingbot_world.regression import RegressionError, check_quality
from tests.dfx.lingbot_world.runner import server_environment

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("platform,count", [("cuda", 3), ("npu", 6)])
def test_weekly_shards_cover_every_original_case_once_without_changing_measurements(platform, count):
    from tests.dfx.lingbot_world.prepare import profile_for_case_shard

    original = json.loads((ROOT / f"tests/dfx/lingbot_world/configs/{platform}_weekly.json").read_text())
    shards = [profile_for_case_shard(original, f"{index}/{count}") for index in range(count)]
    ids = [case["id"] for shard in shards for case in shard["cases"]]
    assert len(ids) == len(set(ids)) == len(original["cases"]) == 18
    assert set(ids) == {case["id"] for case in original["cases"]}
    assert all(len(shard["cases"]) == 18 // count for shard in shards)
    for shard in shards:
        assert {k: v for k, v in shard.items() if k not in {"cases", "case_shard"}} == {
            k: v for k, v in original.items() if k != "cases"
        }


@pytest.mark.parametrize("selection", ["", "-1/3", "3/3", "0/0", "0/19", "0/3/4"])
def test_case_shards_reject_invalid_or_empty_populations(selection):
    from tests.dfx.lingbot_world.prepare import profile_for_case_shard

    with pytest.raises(RegressionError):
        profile_for_case_shard({"cases": [{"id": str(index)} for index in range(18)]}, selection)


@pytest.mark.parametrize("mask", [None, "7,6,5,4,3,2,1,0", ",".join(f"GPU-{index}" for index in range(8))])
def test_reserved_cuda_allocation_preserves_the_original_device_mapping(mask):
    from tests.dfx.lingbot_world.prepare import select_cuda_devices

    expected = "0,1,2,3" if mask is None else ",".join(mask.split(",")[:4])
    assert select_cuda_devices(8, mask, dies=4, reserved_cards=8) == expected


@pytest.mark.parametrize(
    "available,mask,dies,reserved",
    [(4, None, 4, 8), (16, None, 4, 8), (8, None, 9, 8), (8, "0,1,2,3", 4, 8), (8, "0," * 7 + "0", 4, 8)],
)
def test_reserved_cuda_allocation_rejects_incomplete_or_ambiguous_visibility(available, mask, dies, reserved):
    from tests.dfx.lingbot_world.prepare import select_cuda_devices

    with pytest.raises(RegressionError):
        select_cuda_devices(available, mask, dies=dies, reserved_cards=reserved)


def test_exclusive_allocation_preflight_records_all_reserved_devices(tmp_path, monkeypatch):
    import torch

    from tests.dfx.lingbot_world.prepare import prepare_cuda_allocation

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,6,5,4,3,2,1,0")
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "H100")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid=f"GPU-{index}"))
    output = tmp_path / "allocation.json"
    assert (
        prepare_cuda_allocation({"platform": "cuda", "cards": 4, "dies": 4, "reserved_cards": 8}, output) == "7,6,5,4"
    )
    evidence = json.loads(output.read_text())
    assert evidence["reserved_cards"] == 8 and evidence["visible_dies"] == 4
    assert [device["uuid"] for device in evidence["devices"]] == [f"GPU-{index}" for index in range(8)]


def test_exclusive_hardware_contract_requires_matching_allocation_evidence(tmp_path, monkeypatch):
    import vllm_omni.platforms
    from tests.dfx.lingbot_world.runner import hardware_contract

    monkeypatch.setattr(
        vllm_omni.platforms,
        "current_omni_platform",
        SimpleNamespace(
            device_type="cuda",
            get_device_count=lambda: 4,
            get_device_name=lambda index: "H100",
            get_device_total_memory=lambda index: 80 * 1024**3,
        ),
    )
    settings = {"platform": "cuda", "cards": 4, "dies": 4, "reserved_cards": 8}
    monkeypatch.delenv("LINGBOT_ALLOCATION_FILE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(RegressionError, match="allocation preflight"):
        hardware_contract(settings)
    output = tmp_path / "allocation.json"
    evidence = {
        "reserved_cards": 8,
        "visible_dies": 4,
        "visible_devices": "0,1,2,3",
        "devices": [{"uuid": f"GPU-{index}"} for index in range(8)],
    }
    output.write_text(json.dumps(evidence))
    monkeypatch.setenv("LINGBOT_ALLOCATION_FILE", str(output))
    assert hardware_contract(settings)["allocation"] == evidence
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    with pytest.raises(RegressionError, match="allocation evidence"):
        hardware_contract(settings)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    evidence["devices"] = [{"uuid": "GPU-duplicate"}] * 8
    output.write_text(json.dumps(evidence))
    with pytest.raises(RegressionError, match="allocation evidence"):
        hardware_contract(settings)


def test_reference_server_keeps_backend_import_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = tmp_path / "cann"
    backend.mkdir()
    (backend / "compiler_backend.py").write_text('origin = "backend"\n')
    for name in ("reference", "candidate"):
        package = tmp_path / name / "vllm_omni"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f'origin = "{name}"\n')
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(tmp_path / "candidate"), str(backend))))
    result = subprocess.run(
        [sys.executable, "-c", "import vllm_omni, compiler_backend; print(vllm_omni.origin, compiler_backend.origin)"],
        cwd=tmp_path,
        env={**os.environ, **server_environment(tmp_path / "reference")},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "reference backend"


def evaluator_repository(root: Path):
    directory = root / "tests/dfx/lingbot_world"
    directory.mkdir(parents=True)
    (directory / "evaluators.lock.json").write_text('{"repositories": {}, "files": {}}')
    (root / "requirements").mkdir()
    (root / "requirements/lingbot-quality.txt").write_text("example==1\n")


def test_explicit_model_provisioning_preserves_offline_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.dfx.lingbot_world import prepare

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"reference_revision": "a" * 40}))
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "model.json"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("LINGBOT_MODEL_PATH", raising=False)
    monkeypatch.setattr(prepare, "checkout", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "reference",
            "--profile",
            str(profile),
            "--root",
            str(tmp_path / "ref"),
            "--env-output",
            str(output),
        ],
    )
    calls = []

    def download(command, **kwargs):
        calls.append((command, kwargs))
        return str(model) + "\n"

    monkeypatch.setattr(prepare.subprocess, "check_output", download)
    prepare.main()
    command, options = calls[0]
    lock = json.loads((ROOT / "tests/dfx/lingbot_world/assets.lock.json").read_text())
    assert command == ["hf", "download", lock["model_id"], "--revision", lock["model_revision"], "--quiet"]
    assert options["env"]["HF_HUB_OFFLINE"] == "0"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert json.loads(output.read_text())["model_path"] == str(model)


def test_evaluator_cache_isolated_by_code_weights_and_dependencies(tmp_path: Path):
    from tests.dfx.lingbot_world.prepare import evaluator_cache_key

    evaluator_repository(tmp_path)
    initial = evaluator_cache_key(tmp_path)
    (tmp_path / "requirements/lingbot-quality.txt").write_text("example==2\n")
    assert evaluator_cache_key(tmp_path) != initial
    (tmp_path / "requirements/lingbot-quality.txt").write_text("example==1\n")
    (tmp_path / "tests/dfx/lingbot_world/evaluators.lock.json").write_text('{"revision": "different"}')
    assert evaluator_cache_key(tmp_path) != initial


def test_ready_evaluator_is_never_resynced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.dfx.lingbot_world import prepare

    repository = tmp_path / "repository"
    evaluator_repository(repository)
    monkeypatch.setattr(prepare, "ROOT", repository)
    monkeypatch.setattr(prepare, "verify_environment", lambda *args: {})
    calls = []

    def provision(command, **kwargs):
        calls.append(command)
        if command[:2] == ["uv", "venv"]:
            python = Path(command[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare.subprocess, "run", provision)
    root = tmp_path / "cache"
    prepare._prepare_evaluators(root)
    prepare._prepare_evaluators(root)
    assert sum(command[:3] == ["uv", "pip", "sync"] for command in calls) == 1
    (repository / "requirements/lingbot-quality.txt").write_text("example==2\n")
    with pytest.raises(RegressionError, match="fresh evaluator root"):
        prepare._prepare_evaluators(root)
    assert sum(command[:3] == ["uv", "pip", "sync"] for command in calls) == 1


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
    if platform == "cuda":
        assert generate["mirror_hardwares"] == "h100_4_exclusive"
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
        count = 3 if platform == "cuda" else 6
        assert generate["matrix"] == evaluate["matrix"] == {"setup": {"shard": [str(i) for i in range(count)]}}
        assert (
            generate["env"]["LINGBOT_CASE_SHARD"]
            == evaluate["env"]["LINGBOT_CASE_SHARD"]
            == ("{{matrix.shard}}/" + str(count))
        )
        if platform == "cuda":
            assert keyed[prefix + "-long"]["mirror_hardwares"] == "h100_4_exclusive"
        else:
            collection = keyed[prefix + "-long"]
            gate = keyed[prefix + "-long-gate"]
            quality = keyed[prefix + "-long-quality"]
            assert collection["matrix"] == {"setup": {"repeat": ["0", "1", "2"]}}
            assert " collect " in collection["commands"][0]
            assert prefix + "-long" in gate["depends_on"]
            assert gate["mirror_hardwares"] == "l4_1" and not gate.get("soft_fail")
            assert prefix + "-long-gate" in quality["depends_on"]
            assert quality["commands"][0].endswith(prefix + "-long-gate")


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


def test_every_configured_quality_dimension_rejects_a_collapse():
    limits = json.loads((ROOT / "tests/dfx/lingbot_world/quality.json").read_text())
    # Both pinned evaluators return these dimensions on a [0, 1] scale.
    # A tolerance accidentally expressed as percentage points must not make
    # a dimension incapable of blocking a regression.
    for suite in ("vbench", "wbench"):
        assert limits[suite].keys() == limits["max_reference_spreads"][suite].keys()
        for dimension, tolerance in limits[suite].items():
            result = check_quality(
                [{dimension: 0.9}] * 3,
                {dimension: 0.0},
                tolerances={dimension: tolerance},
                activity_dimensions=("dynamic_degree",),
                max_reference_spreads={dimension: limits["max_reference_spreads"][suite][dimension]},
            )
            assert not result["passed"], (suite, dimension)


@pytest.mark.parametrize("stage", ["generate", "unit"])
def test_launcher_setup_failure_retains_nonzero_exit_and_junit(tmp_path: Path, stage):
    output = tmp_path / "artifacts"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / ".buildkite/common/scripts/run-lingbot-ci.sh"),
            stage,
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


@pytest.mark.parametrize("cancel_signal", [signal.SIGTERM, signal.SIGKILL])
def test_launcher_cancellation_retains_failure_junit(tmp_path: Path, cancel_signal):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "if sys.argv[1:4] == ['-m', 'tests.dfx.lingbot_world.prepare', 'reference']:\n"
        "    Path(sys.argv[sys.argv.index('--env-output')+1]).write_text(json.dumps({'model_path':'unused'}))\n"
        "else:\n    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    pytest_cli = bin_dir / "pytest"
    ready = tmp_path / "ready"
    pytest_cli.write_text(
        f"#!{sys.executable}\nimport time\nfrom pathlib import Path\nPath({str(ready)!r}).touch()\ntime.sleep(120)\n"
    )
    python.chmod(0o755)
    pytest_cli.chmod(0o755)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"platform": "npu", "dies": 4}))
    output = tmp_path / "artifacts"
    process = subprocess.Popen(
        ["bash", str(ROOT / ".buildkite/common/scripts/run-lingbot-ci.sh"), "generate", str(profile), str(output)],
        cwd=ROOT,
        env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "Fake generation did not reach the interruption point"
        os.killpg(process.pid, cancel_signal)
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    assert process.returncode != 0
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
        if profile["platform"] == "cuda" and profile["cards"] == 4:
            assert profile["reserved_cards"] == 8 and profile["dies"] == 4
    assert len(lock["model_files"]) >= 20


def test_npu_profiles_offload_only_text_encoder_without_reducing_workload():
    from vllm_omni.diffusion.offloader.config import OffloadMode, parse_diffusion_offload_config

    for name in ("npu_nightly", "npu_weekly", "npu_long"):
        profile = json.loads((ROOT / f"tests/dfx/lingbot_world/configs/{name}.json").read_text())
        deployment = yaml.safe_load((ROOT / profile["deploy_config"]).read_text())
        stage = deployment["stages"][0]
        offload = parse_diffusion_offload_config(stage["diffusion_offload_config"])
        assert offload is not None and offload.mode == OffloadMode.MODULE
        assert offload.components == {"text_encoder"}
        assert stage["parallel_config"]["sequence_parallel_size"] == 4
        assert (profile["height"], profile["width"]) == (480, 832)
        assert not {"window_chunks", "sink_chunks"} & stage["model_config"]["ar_diffusion_kv_config"].keys()
