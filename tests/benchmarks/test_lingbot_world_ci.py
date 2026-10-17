# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CI wiring and failure propagation, without substituting model inference."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

from benchmarks.lingbot_world.regression import RegressionError, check_binary_activity, check_quality
from tests.dfx.lingbot_world.runner import server_arguments, server_environment

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


def test_lifecycle_playhead_blocks_second_admission_before_inference() -> None:
    from benchmarks.lingbot_world.workload import pixel_frames_in_chunk
    from tests.e2e.online_serving import test_lingbot_world_v2_stepwise as lifecycle
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, _StreamingPlayback

    state = _StreamingPlayback(
        buffer_seconds=lifecycle._paced_overrides()["streaming_pacing"]["max_lead_ms"] / 1000,
        next_media_duration=pixel_frames_in_chunk(0) / 16,
    )
    engine = object.__new__(DiffusionEngine)
    engine._cv = threading.Condition()
    engine._streaming_playback = {"paced": state}
    assert not engine._playback_blocked(), "the window must fit the initial chunk"

    state.generated_seconds = state.next_media_duration
    state.next_media_duration = pixel_frames_in_chunk(1) / 16
    assert engine._playback_blocked(), "held playback must block the second chunk before its inference starts"

    state.played_seconds = state.generated_seconds
    assert not engine._playback_blocked(), "reporting playback must release admission"


class _DeviceProperties(NamedTuple):
    uuid: str


class _MountInfo(NamedTuple):
    mountpoint: str
    fstype: str
    opts: str = ""


@pytest.mark.parametrize(
    "membership,relative,limit_name,peak_name",
    [
        ("0::/pod", "pod", "memory.max", "memory.peak"),
        ("5:memory:/pod", "memory/pod", "memory.limit_in_bytes", "memory.max_usage_in_bytes"),
    ],
)
def test_resource_snapshot_retains_cgroup_limits_and_peaks(
    tmp_path, monkeypatch, membership, relative, limit_name, peak_name
):
    from tests.dfx.lingbot_world import resources

    proc = tmp_path / "cgroup-membership"
    proc.write_text(membership + "\n")
    root = tmp_path / "cgroups"
    group = root / relative
    group.mkdir(parents=True)
    (group / limit_name).write_text("8589934592\n")
    (group / peak_name).write_text("7516192768\n")
    (group / "memory.pressure").write_text("some avg10=12.00 total=1234\n")
    cpu = (
        {
            "cpu.max": "3200000 100000",
            "cpu.stat": "nr_periods 100\nnr_throttled 12\nthrottled_usec 456",
            "cpu.pressure": "some avg10=8.00 total=5678",
        }
        if membership.startswith("0::")
        else {}
    )
    for name, value in cpu.items():
        (group / name).write_text(value + "\n")
    monkeypatch.setattr(resources, "_PROC_CGROUP", proc)
    monkeypatch.setattr(resources, "_CGROUP_ROOT", root)

    report = resources.collect_resources()
    assert report["cgroup_membership"] == membership
    kernel = report["kernel_memory"]
    assert isinstance(kernel, dict)
    assert kernel[str(group / limit_name)] == "8589934592"
    assert kernel[str(group / peak_name)] == "7516192768"
    assert "avg10=12.00" in kernel[str(group / "memory.pressure")]
    for name, value in cpu.items():
        assert kernel[str(group / name)] == value


@pytest.mark.parametrize("relative", ["", "pod/container"])
def test_resource_snapshot_retains_v1_cpu_quota_and_throttling(tmp_path: Path, monkeypatch, relative: str):
    from tests.dfx.lingbot_world import resources

    proc = tmp_path / "cgroup"
    proc.write_text("3:cpu,cpuacct:/pod/container\n2:cpuset:/pod/container\n")
    root = tmp_path / "controllers"
    cpu = root / "cpu,cpuacct" / relative
    cpuset = root / "cpuset" / relative
    cpu.mkdir(parents=True)
    cpuset.mkdir(parents=True)
    fields = {
        cpu / "cpu.cfs_quota_us": "3200000",
        cpu / "cpu.cfs_period_us": "100000",
        cpu / "cpu.stat": "nr_periods 100\nnr_throttled 81\nthrottled_time 123456789",
        cpu / "cpuacct.usage": "987654321",
        cpuset / "cpuset.cpus": "0-63",
    }
    for path, value in fields.items():
        path.write_text(value + "\n")
    monkeypatch.setattr(resources, "_PROC_CGROUP", proc)
    monkeypatch.setattr(resources, "_CGROUP_ROOT", root)
    kernel = resources.collect_resources()["kernel_memory"]
    assert isinstance(kernel, dict)
    for path, value in fields.items():
        assert kernel[str(path)] == value


def test_resource_snapshot_resolves_weight_symlinks_and_keeps_only_mount_metadata(tmp_path, monkeypatch):
    from tests.dfx.lingbot_world import resources

    cache = tmp_path / "cache"
    cache.mkdir()
    weight = cache / "blob"
    weight.write_bytes(b"checkpoint")
    link = tmp_path / "weight.safetensors"
    link.symlink_to(weight)
    monkeypatch.setattr(resources, "_PROC_CGROUP", tmp_path / "unavailable-proc")
    monkeypatch.setattr(resources, "_CGROUP_ROOT", tmp_path / "unavailable-cgroup")
    monkeypatch.setattr(
        resources.psutil,
        "disk_partitions",
        lambda all: [
            _MountInfo(str(tmp_path), "ext4"),
            _MountInfo(str(cache), "nfs4", "test-option-must-not-be-retained"),
        ],
    )
    report = resources.collect_resources(link)
    assert report["model_storage"] == {"weight_path": str(weight), "filesystem": "nfs4", "mountpoint": str(cache)}
    assert "test-option-must-not-be-retained" not in json.dumps(report)


def test_unavailable_resource_probes_retain_errors_without_raising(tmp_path, monkeypatch):
    from tests.dfx.lingbot_world import resources

    def unavailable():
        raise PermissionError("memory probe unavailable")

    monkeypatch.setattr(resources, "_PROC_CGROUP", tmp_path / "unavailable-proc")
    monkeypatch.setattr(resources, "_CGROUP_ROOT", tmp_path / "unavailable-cgroup")
    monkeypatch.setattr(resources.psutil, "virtual_memory", unavailable)
    report = resources.collect_resources()
    assert report["host_memory"] is None and report["kernel_memory"] == {}
    assert report["model_storage"] is None
    errors = report["errors"]
    assert isinstance(errors, dict)
    assert "PermissionError" in errors["host_memory"]


def test_failed_pytest_retains_original_error_and_resource_evidence(tmp_path):
    test = tmp_path / "test_model_failure.py"
    test.write_text('def test_model_failure():\n    raise RuntimeError("original model startup failure")\n')
    output = tmp_path / "failure-artifacts"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tests.dfx.lingbot_world.conftest",
            str(test),
            f"--lingbot-output={output}",
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "BUILDKITE": "true", "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    failure = json.loads((output / "test_model_failure-call-failure.json").read_text())
    assert failure["passed"] is False and "original model startup failure" in failure["error"]
    assert "kernel_memory" in failure["resources"]
    assert "excluded from performance measurements" in failure["resources"]["scope"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "selection,expected_names",
    [
        ("core_model and cpu", ("test_selected_text_encoder_is_released_before_following_video_work",)),
        (
            "full_model and cuda and H100",
            ("test_cuda_smoke", "test_cuda_served_regression", "test_wbench_vbench_and_video_precision"),
        ),
        ("full_model and npu and A2", ("test_collect_npu_measurement_pair", "test_npu_served_regression")),
        ("core_model and xpu and B60", ()),
    ],
)
def test_model_and_dfx_packages_collect_together_on_every_platform(reverse, selection, expected_names):
    # Intel collects both trees before applying its XPU expression. DFX must
    # not shadow the existing model package, including for deselected tests.
    paths = ["tests/diffusion/models/lingbot_world/test_offload_lingbot_world.py", "tests/dfx/lingbot_world"]
    if reverse:
        paths.reverse()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *paths, "-m", selection],
        cwd=ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "BUILDKITE_PARALLEL_JOB": "0",
            "BUILDKITE_PARALLEL_JOB_COUNT": "1",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == (0 if expected_names else 5), result.stdout + result.stderr
    assert "ERROR collecting" not in result.stdout
    assert all(name in result.stdout for name in expected_names)


@pytest.mark.parametrize("timeout", [37, 1200])
def test_server_initialization_and_stage_deadlines_use_the_explicit_profile_budget(timeout):
    settings = {"deploy_config": "tests/dfx/lingbot_world/configs/npu_usp4_offload.yaml", "startup_timeout_s": timeout}
    arguments = server_arguments(settings)
    assert dict(zip(arguments[::2], arguments[1::2])) == {
        "--deploy-config": str(ROOT / settings["deploy_config"]),
        "--init-timeout": str(timeout),
        "--stage-init-timeout": str(timeout),
    }


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
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: _DeviceProperties(uuid=f"GPU-{index}"))
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

    platform = vllm_omni.platforms.current_omni_platform
    monkeypatch.setattr(platform, "device_type", "cuda")
    monkeypatch.setattr(platform, "get_device_count", lambda: 4)
    monkeypatch.setattr(platform, "get_device_name", lambda index: "H100")
    monkeypatch.setattr(platform, "get_device_total_memory", lambda index: 80 * 1024**3)
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


@pytest.mark.parametrize("mask", [None, "7,6,5,4,3,2,1,0", "8,9,10,11,12,13,14,15"])
def test_npu_allocation_preserves_cann_runtime_mapping(mask):
    from tests.dfx.lingbot_world.prepare import select_npu_devices

    expected = "0,1,2,3" if mask is None else ",".join(mask.split(",")[:4])
    assert select_npu_devices(8, mask, dies=4, reserved_cards=8) == expected


@pytest.mark.parametrize(
    "available,mask", [(4, None), (16, None), (8, "0,1,2,3"), (8, "0," * 7 + "0"), (8, "-1,1,2,3,4,5,6,7")]
)
def test_npu_allocation_rejects_partial_or_ambiguous_reservations(available, mask):
    from tests.dfx.lingbot_world.prepare import select_npu_devices

    with pytest.raises(RegressionError):
        select_npu_devices(available, mask, dies=4, reserved_cards=8)


def test_npu_preflight_and_contract_verify_mask_and_actual_device_properties(tmp_path, monkeypatch):
    import vllm_omni.platforms
    from tests.dfx.lingbot_world import runner
    from tests.dfx.lingbot_world.prepare import prepare_npu_allocation

    platform = vllm_omni.platforms.current_omni_platform
    count = [8]
    monkeypatch.setattr(platform, "device_type", "npu")
    monkeypatch.setattr(platform, "get_device_count", lambda: count[0])
    monkeypatch.setattr(platform, "get_device_name", lambda index: "Ascend910B3")
    monkeypatch.setattr(platform, "get_device_total_memory", lambda index: 64 * 1024**3)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda package: "test-version")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7,6,5,4,3,2,1,0")
    settings = {"platform": "npu", "cards": 4, "dies": 4, "reserved_cards": 8}
    output = tmp_path / "allocation.json"
    assert prepare_npu_allocation(settings, output) == "7,6,5,4"
    evidence = json.loads(output.read_text())
    assert [device["runtime_id"] for device in evidence["devices"]] == list("76543210")
    count[0] = 4
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7,6,5,4")
    monkeypatch.delenv("LINGBOT_ALLOCATION_FILE", raising=False)
    with pytest.raises(RegressionError, match="allocation preflight"):
        runner.hardware_contract(settings)
    monkeypatch.setenv("LINGBOT_ALLOCATION_FILE", str(output))
    assert runner.hardware_contract(settings)["allocation"] == evidence
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(RegressionError, match="allocation evidence"):
        runner.hardware_contract(settings)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7,6,5,4")
    evidence["devices"][0]["total_bytes"] -= 1
    output.write_text(json.dumps(evidence))
    with pytest.raises(RegressionError, match="visible hardware"):
        runner.hardware_contract(settings)


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


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_engine_switches_override_inherited_environment_before_import(tmp_path, monkeypatch, role):
    monkeypatch.setenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER", "1")
    monkeypatch.setenv("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH", "999")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,runpy,sys; config=runpy.run_path(sys.argv[1]); "
            "print(config['contiguous_kv_gather_enabled']()); "
            "print(os.environ['AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH'])",
            str(ROOT / "vllm_omni/experimental/ar_diffusion/kv_cache/config.py"),
        ],
        env={**os.environ, **server_environment(tmp_path / role)},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.splitlines() == ["False", "0"]


def test_generation_records_fixed_engine_environment_before_capture_failure(tmp_path, monkeypatch):
    from tests.dfx.lingbot_world import runner

    output = tmp_path / "output"
    profile = tmp_path / "profile.json"
    config = json.loads((ROOT / "tests/dfx/lingbot_world/configs/npu_nightly.json").read_text())
    profile.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "checked_assets", lambda settings: (tmp_path / "model", {}))
    monkeypatch.setattr(runner, "validate_reference", lambda settings: tmp_path / "reference")
    monkeypatch.setattr(runner, "hardware_contract", lambda settings: {"platform": "npu"})

    def fail_capture(*args):
        raise RuntimeError("precision provisioning failed")

    monkeypatch.setattr(runner, "_capture", fail_capture)
    with pytest.raises(RuntimeError, match="precision provisioning failed"):
        runner.generate({"output": output, "profile_path": profile, "deploy_config": config["deploy_config"]})
    assert json.loads((output / "contract.json").read_text())["engine_environment"] == {
        "VLLM_OMNI_AR_DIFFUSION_KV_GATHER": "0",
        "AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH": "0",
    }


def test_fixed_worker_threads_apply_before_torch_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "19")
    monkeypatch.setenv("MKL_NUM_THREADS", "23")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "29")
    env = server_environment(tmp_path / "reference", cpu_threads=1)
    assert all(env[name] == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.get_num_threads())"],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "1"


@pytest.mark.parametrize("threads", [0, True])
def test_invalid_worker_thread_count_fails_before_launch(tmp_path: Path, threads):
    with pytest.raises(RegressionError, match="positive integer"):
        server_environment(tmp_path, cpu_threads=threads)


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_fixed_worker_threads_also_reach_precision_capture(tmp_path: Path, monkeypatch, role: str):
    from tests.dfx.lingbot_world import runner

    calls = []
    monkeypatch.setenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER", "1")
    monkeypatch.setenv("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH", "999")
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    source = tmp_path / role
    runner._capture(
        source,
        tmp_path / "model",
        tmp_path / "precision.npz",
        {
            "height": 480,
            "width": 832,
            "precision_timeout_s": 600,
            "cpu_threads_per_worker": 1,
        },
    )
    assert len(calls) == 1
    options = calls[0][1]
    assert options["cwd"] == source
    assert all(options["env"][name] == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    assert options["env"]["VLLM_OMNI_AR_DIFFUSION_KV_GATHER"] == "0"
    assert options["env"]["AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH"] == "0"


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


@pytest.mark.parametrize(
    "generation_passed,mutation", [(True, None), (False, None), (False, "unlisted"), (False, "corrupt")]
)
def test_quality_retains_scores_after_generation_failure_without_passing(
    tmp_path, monkeypatch, generation_passed, mutation
):
    import cv2
    import numpy as np

    from benchmarks.lingbot_world.regression import sha256, write_json
    from tests.dfx.lingbot_world import test_quality as gate

    source = tmp_path / "source"
    policy = source / "tests/dfx/lingbot_world"
    policy.mkdir(parents=True)
    limits = json.loads((ROOT / "tests/dfx/lingbot_world/quality.json").read_text())
    write_json(policy / "quality.json", limits)
    write_json(
        policy / "evaluators.lock.json",
        {"files": {}, "repositories": {suite: {"revision": "snapshot"} for suite in ("vbench", "wbench")}},
    )
    generation = tmp_path / "generation"
    artifacts = {}
    y, x = np.indices((24, 32))
    texture = ((x * 3 + y * 5) % 80 + 40).astype(np.uint8)
    frame = np.stack([texture, texture + 5, texture + 10], axis=-1)
    for role in ("reference", "candidate"):
        for repeat in range(3):
            path = generation / role / str(repeat) / "scene/rollout.mp4"
            path.parent.mkdir(parents=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16, (32, 24))
            assert writer.isOpened(), "CPU quality coordinator test requires an MP4 encoder"
            try:
                for _ in range(9):
                    writer.write(frame)
            finally:
                writer.release()
            artifacts[str(path.relative_to(generation))] = sha256(path)
    if mutation == "unlisted":
        artifacts.pop("candidate/2/scene/rollout.mp4")
    elif mutation == "corrupt":
        (generation / "candidate/2/scene/rollout.mp4").write_bytes(b"corrupt output")
    profile = tmp_path / "profile.json"
    write_json(profile, {"platform": "npu", "num_chunks": 1})
    write_json(
        generation / "generation.json",
        {
            "passed": generation_passed,
            "contract": {"profile": json.loads(profile.read_text())},
            "candidate_revision": "snapshot",
            "artifacts": artifacts,
        },
    )
    monkeypatch.setattr(gate, "ROOT", source)
    monkeypatch.setattr(gate, "git_revision", lambda path: "snapshot")
    monkeypatch.setattr(gate, "verify_environment", lambda *args: {})
    monkeypatch.setattr(gate, "hardware_contract", lambda settings: {"unit_test": True})
    monkeypatch.setattr(gate.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    monkeypatch.setenv("LINGBOT_EVALUATOR_ROOT", str(tmp_path / "evaluators"))
    monkeypatch.setenv("LINGBOT_EVALUATOR_PYTHON", sys.executable)
    for name in ("VBENCH_CACHE_DIR", "WBENCH_WEIGHTS_DIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    calls = []

    def score(videos, output, *, suite, dimensions, **kwargs):
        calls.append(suite)
        return {
            "status": "passed",
            "protocol": {"dynamic_degree_backend": "raft"},
            "scores": {label: {name: 1.0 for name in dimensions} for label in videos},
            "native_provenance": {
                label: {
                    "background_clip_inference": {
                        "batch_size": 32,
                        "encoded_images": 9,
                        "forward_batches": 1,
                        "max_forward_batch": 9,
                        "gradients": False,
                    }
                }
                for label in videos
            },
        }

    monkeypatch.setattr(gate, "evaluate_video_quality", score)
    output = tmp_path / "quality"
    settings = {
        "output": output,
        "generation": generation,
        "profile_path": profile,
        "platform": "npu",
        "num_chunks": 1,
        "width": 32,
        "height": 24,
        "mux_fps": 16,
        "cases": [{"id": "scene"}],
    }
    if mutation:
        with pytest.raises(RegressionError, match="checksum|Unhashed"):
            gate.test_wbench_vbench_and_video_precision(settings)
        assert calls == []
        return
    if generation_passed:
        gate.test_wbench_vbench_and_video_precision(settings)
    else:
        with pytest.raises(AssertionError, match="Quality regression"):
            gate.test_wbench_vbench_and_video_precision(settings)
    report = json.loads((output / "quality.json").read_text())
    assert calls == ["vbench", "wbench"]
    assert report["quality_passed"] is True
    assert report["generation_passed"] is generation_passed
    assert report["passed"] is generation_passed
    assert (output / "vbench-raw.json").is_file() and (output / "wbench-raw.json").is_file()


@pytest.mark.parametrize("platform", ["cuda", "npu"])
@pytest.mark.parametrize("level", ["nightly", "weekly"])
def test_generation_and_gpu_evaluation_are_blocking_and_connected(platform, level):
    name = f"test-{level}.yml" if platform == "cuda" else f"test-npu-{level}.yml"
    steps = list(leaves(yaml.safe_load((ROOT / f".buildkite/{platform}/{name}").read_text())))
    keyed = {step.get("key"): step for step in steps}
    prefix = f"lingbot-{platform}-{level}"
    generate, evaluate = keyed[prefix + "-generation"], keyed[prefix + "-quality"]
    assert len(evaluate["commands"][0].split()) == 6
    if platform == "cuda":
        assert generate["mirror_hardwares"] == "h100_4_exclusive"
    else:
        assert generate["mirror_hardwares"] == "a2b3_npu_8"
    assert {"step": prefix + "-generation", "allow_failure": True} in evaluate["depends_on"]
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
            assert collection["mirror_hardwares"] == "a2b3_npu_8"
            gate = keyed[prefix + "-long-gate"]
            quality = keyed[prefix + "-long-quality"]
            assert collection["matrix"] == {"setup": {"repeat": ["0", "1", "2"]}}
            assert " collect " in collection["commands"][0]
            assert prefix + "-long" in gate["depends_on"]
            assert gate["mirror_hardwares"] == "l4_1" and not gate.get("soft_fail")
            assert {"step": prefix + "-long-gate", "allow_failure": True} in quality["depends_on"]
            assert quality["commands"][0].endswith("lingbot-artifacts/npu-long")
            assert len(gate["commands"][0].split()) == len(quality["commands"][0].split()) == 6
        assert {"step": prefix + "-long" + ("-gate" if platform == "npu" else ""), "allow_failure": True} in keyed[
            prefix + "-long-quality"
        ]["depends_on"]


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


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_every_configured_quality_dimension_rejects_a_collapse(platform):
    limits = json.loads((ROOT / "tests/dfx/lingbot_world/quality.json").read_text())
    # Both pinned evaluators return these dimensions on a [0, 1] scale.
    # A tolerance accidentally expressed as percentage points must not make
    # a dimension incapable of blocking a regression.
    for suite in ("vbench", "wbench"):
        binary_dimensions = set(limits["binary_activity_dimensions"])
        assert binary_dimensions == {"dynamic_degree"}
        tolerances = limits["candidate_tolerances"][platform][suite]
        assert set(tolerances) - binary_dimensions == set(limits["max_reference_spreads"][platform][suite])
        for dimension, tolerance in tolerances.items():
            if dimension in binary_dimensions:
                assert not check_binary_activity([1, 1, 1], [0, 0, 0], tolerance=tolerance)["passed"]
                assert not check_binary_activity([0, 0, 0], [1, 1, 1], tolerance=tolerance)["passed"]
                continue
            result = check_quality(
                [{dimension: 0.9}] * 3,
                {dimension: 0.0},
                tolerances={dimension: tolerance},
                activity_dimensions=("dynamic_degree",),
                max_reference_spreads={dimension: limits["max_reference_spreads"][platform][suite][dimension]},
            )
            assert not result["passed"], (suite, dimension)


@pytest.mark.parametrize("stage", ["aggregate", "evaluate"])
def test_matrix_artifact_handoff_uses_unique_build_prefix(tmp_path, stage):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    generation = "generation/pairs" if stage == "aggregate" else "generation/shard-1"
    expected = (
        [f"{generation}/pair-{i}.json" for i in range(3)] if stage == "aggregate" else [f"{generation}/manifest.json"]
    )
    artifacts = (
        [[path] for path in expected] if stage == "aggregate" else [["generation/shard-0/manifest.json"], expected]
    )
    agent = bin_dir / "buildkite-agent"
    trace = tmp_path / "download.json"
    agent.write_text(
        f"#!{sys.executable}\nimport fnmatch,json,sys\nfrom pathlib import Path\n"
        f"jobs={artifacts!r}\n"
        "paths=jobs[0] if '--step' in sys.argv else [p for job in jobs for p in job]\n"
        "matches=[p for p in paths if fnmatch.fnmatch(p,sys.argv[3])]\n"
        f"Path({str(trace)!r}).write_text(json.dumps({{'args':sys.argv[1:],'paths':matches}}))\n"
        "if not matches: sys.exit(23)\n"
        "for value in matches:\n    p=Path(value); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}')\n"
    )
    consumed = tmp_path / "consumed"
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\nimport os,sys\nfrom pathlib import Path\n"
        "if sys.argv[1:3]==['-m','pytest'] or "
        "sys.argv[1:4]==['-m','tests.dfx.lingbot_world.prepare','evaluators']:\n"
        f"    if not all(Path(p).exists() for p in {expected!r}): sys.exit(29)\n"
        f"    Path({str(consumed)!r}).touch()\n    sys.exit(17)\n"
        "else:\n    os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n"
    )
    for path in (agent, python):
        path.chmod(0o755)
    env = {
        **os.environ,
        "BUILDKITE": "true",
        "PYTHONPATH": str(ROOT),
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
    }
    env.pop("LINGBOT_CASE_SHARD", None)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / ".buildkite/common/scripts/run-lingbot-ci.sh"),
            stage,
            str(ROOT / "tests/dfx/lingbot_world/configs/npu_long.json"),
            str(output),
            generation,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    assert consumed.exists()
    assert json.loads(trace.read_text())["paths"] == expected
    assert ET.parse(output / "pytest.xml").find("testcase/failure") is not None


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


@pytest.mark.parametrize("stage", ["generate", "collect"])
@pytest.mark.parametrize("preflight_status", [0, 19])
def test_npu_launcher_limits_visibility_before_model_preparation(tmp_path, stage, preflight_status):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    reached = tmp_path / "reference-environment.json"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "if sys.argv[1:4] == ['-m', 'tests.dfx.lingbot_world.prepare', 'devices']:\n"
        f"    if {preflight_status}: sys.exit({preflight_status})\n"
        "    root = Path(sys.argv[sys.argv.index('--root')+1])\n"
        "    (root/'allocation.json').write_text(json.dumps({'visible_devices': '7,6,5,4'}))\n"
        "    print('INFO backend initialized')\n"
        "    print('7,6,5,4')\n"
        "elif sys.argv[1:4] == ['-m', 'tests.dfx.lingbot_world.prepare', 'reference']:\n"
        f"    Path({str(reached)!r}).write_text(json.dumps({{key: os.environ.get(key) for key in "
        "['ASCEND_RT_VISIBLE_DEVICES', 'LINGBOT_ALLOCATION_FILE']}))\n"
        "    sys.exit(17)\n"
        "else:\n    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    output = tmp_path / "artifacts"
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
    env.pop("LINGBOT_CASE_SHARD", None)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / ".buildkite/common/scripts/run-lingbot-ci.sh"),
            stage,
            str(ROOT / "tests/dfx/lingbot_world/configs/npu_nightly.json"),
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == (preflight_status or 17), result.stdout + result.stderr
    assert reached.exists() is (preflight_status == 0)
    if reached.exists():
        environment = json.loads(reached.read_text())
        assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "7,6,5,4"
        assert Path(environment["LINGBOT_ALLOCATION_FILE"]) == output / "allocation.json"
    assert ET.parse(output / "pytest.xml").find("testcase/failure") is not None


@pytest.mark.parametrize("cancel_signal", [signal.SIGTERM, signal.SIGKILL])
def test_launcher_cancellation_retains_failure_junit(tmp_path: Path, cancel_signal):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "if sys.argv[1:4] == ['-m', 'tests.dfx.lingbot_world.prepare', 'devices']:\n"
        "    root = Path(sys.argv[sys.argv.index('--root')+1])\n"
        "    (root/'allocation.json').write_text(json.dumps({'visible_devices':'0,1,2,3'}))\n"
        "    print('0,1,2,3')\n"
        "elif sys.argv[1:4] == ['-m', 'tests.dfx.lingbot_world.prepare', 'reference']:\n"
        "    Path(sys.argv[sys.argv.index('--env-output')+1]).write_text(json.dumps({'model_path':'unused'}))\n"
        "else:\n    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    pytest_cli = bin_dir / "pytest"
    ready = tmp_path / "ready"
    pytest_cli.write_text(
        f"#!{sys.executable}\nimport os, time\nfrom pathlib import Path\n"
        "assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '0,1,2,3'\n"
        "assert Path(os.environ['LINGBOT_ALLOCATION_FILE']).is_file()\n"
        f"Path({str(ready)!r}).touch()\ntime.sleep(120)\n"
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
        if profile["cards"] == 4:
            assert profile["reserved_cards"] == 8 and profile["dies"] == 4
    assert len(lock["model_files"]) >= 20


def test_npu_profiles_offload_only_text_encoder_without_reducing_workload():
    from vllm_omni.config.pipeline_registry import resolve_pipeline_config
    from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
    from vllm_omni.diffusion.offloader.config import OffloadMode, parse_diffusion_offload_config

    for name in ("npu_nightly", "npu_weekly", "npu_long"):
        profile = json.loads((ROOT / f"tests/dfx/lingbot_world/configs/{name}.json").read_text())
        deployment = yaml.safe_load((ROOT / profile["deploy_config"]).read_text())
        stage = deployment["stages"][0]
        assert stage["enable_multithread_weight_load"] is False
        resolved = merge_pipeline_deploy(
            resolve_pipeline_config("lingbot_world"), load_deploy_config(ROOT / profile["deploy_config"])
        )
        assert resolved[0].yaml_engine_args["enable_multithread_weight_load"] is False
        offload = parse_diffusion_offload_config(stage["diffusion_offload_config"])
        assert offload is not None and offload.mode == OffloadMode.MODULE
        assert offload.components == {"text_encoder"}
        assert stage["parallel_config"]["sequence_parallel_size"] == 4
        assert (profile["height"], profile["width"]) == (480, 832)
        assert not {"window_chunks", "sink_chunks"} & stage["model_config"]["ar_diffusion_kv_config"].keys()
