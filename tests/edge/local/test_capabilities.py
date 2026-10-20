# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""What this process can run on, as opposed to what is plugged in."""

import pytest

from vllm_omni.edge.hardware_probe import ACCEL_GPU_DISCRETE
from vllm_omni.edge.local.capabilities import (
    FORMAT_DENSE,
    FORMAT_FP8,
    FORMAT_GPTQ,
    FORMAT_INT8,
    FORMAT_NVFP4,
    MASK_ENV,
    cuda_weight_formats,
    enumerate_devices,
    parse_mask,
    runs_exported_graphs,
)

from .conftest import blackwell_laptop, cpu_only_box

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _by_id(devices):
    return {d.device_id: d for d in devices}


# ---------------------------------------------------------- format gating
def test_blackwell_has_no_int8_path():
    """sm_120 is inside the SM >= 100 range where cutlass c3x has no int8
    scaled_mm. This is the single fact that turns a 40-second crash into a
    30-millisecond refusal, so it gets its own test."""
    formats = cuda_weight_formats((12, 0))
    assert FORMAT_INT8 not in formats
    assert FORMAT_FP8 in formats
    assert FORMAT_NVFP4 in formats


def test_ampere_keeps_int8_and_has_no_nvfp4():
    formats = cuda_weight_formats((8, 6))
    assert FORMAT_INT8 in formats
    assert FORMAT_NVFP4 not in formats
    assert FORMAT_FP8 not in formats


def test_dense_runs_on_every_cuda_generation():
    for capability in ((7, 5), (8, 0), (9, 0), (12, 0)):
        assert FORMAT_DENSE in cuda_weight_formats(capability)


# ------------------------------------------------------------ runnability
def test_a_cuda_wheel_cannot_drive_the_cpu_and_says_so():
    devices = _by_id(enumerate_devices(blackwell_laptop(), platform_device_type="cuda"))
    assert devices["cuda:0"].runnable
    assert not devices["cpu"].runnable
    assert "omni-cpu" in devices["cpu"].reason


def test_a_cpu_wheel_cannot_drive_the_card_and_says_so():
    devices = _by_id(enumerate_devices(blackwell_laptop(), platform_device_type="cpu"))
    assert devices["cpu"].runnable
    assert not devices["cuda:0"].runnable
    assert "omni-cuda" in devices["cuda:0"].reason


def test_the_igpu_and_npu_are_reported_not_dropped():
    """'The machine has an NPU' and 'a stage can run on it' are different
    claims; the plan has to be able to show which one it relied on.

    Both are now reachable (M3), through a worker in another interpreter, so
    the invariant is no longer "never runnable" -- it is that the record always
    says which of the two claims it is making, and never leaves a caller to
    guess.
    """
    devices = _by_id(enumerate_devices(blackwell_laptop(), platform_device_type="cuda"))
    assert "npu:amd" in devices
    assert "gpu_integrated:amd" in devices
    for key in ("npu:amd", "gpu_integrated:amd"):
        device = devices[key]
        assert device.evidence in ("B", "D")
        if device.runnable:
            assert device.backend, "a runnable device must name the backend that drives it"
            assert device.extra["worker_route"], "and the route it was resolved through"
        else:
            assert device.reason, "and an unrunnable one must say why"
            assert device.backend is None
        # Either way these execute exported graphs, never a checkpoint.
        assert runs_exported_graphs(device)


def test_the_amd_devices_refuse_with_the_missing_interpreter_named(monkeypatch):
    """A device that works and a venv that is not installed are different
    problems; the reason has to distinguish them well enough to act on."""
    for route in ("ORT_VITISAI", "ORT_DML", "TORCH_DML"):
        monkeypatch.setenv(f"VLLM_OMNI_EXTERNAL_PYTHON_{route}", "/nonexistent/python")
    devices = _by_id(enumerate_devices(blackwell_laptop(), platform_device_type="cuda"))
    for key in ("npu:amd", "gpu_integrated:amd"):
        device = devices[key]
        assert not device.runnable and device.backend is None
        assert "/nonexistent/python" in device.reason or "native-Windows" in device.reason


def test_the_amd_devices_get_no_memory_budget_of_their_own():
    """Section 6: the iGPU and the NPU allocate out of the CPU's RAM.

    Carrying the pool size on their rows would let a caller add three copies of
    the same memory together, so the number lives on the host row only and the
    stage planner reads it from there.
    """
    devices = _by_id(enumerate_devices(blackwell_laptop(), platform_device_type="cpu"))
    for key in ("npu:amd", "gpu_integrated:amd"):
        assert devices[key].memory_bytes == 0
        assert devices[key].memory_pool == "host_ram"
        assert devices[key].extra["host_pool_bytes"] > 0


def test_a_cpu_only_box_reports_no_cuda_row_at_all():
    """A masked GPU and an absent GPU are different situations and must not
    produce the same record."""
    devices = _by_id(enumerate_devices(cpu_only_box(), platform_device_type="cpu"))
    assert "cuda:0" not in devices
    assert devices["cpu"].runnable


# ------------------------------------------------------------------ masking
def test_masking_disables_without_hiding():
    devices = _by_id(enumerate_devices(
        blackwell_laptop(), mask=frozenset({ACCEL_GPU_DISCRETE}), platform_device_type="cuda"
    ))
    gpu = devices["cuda:0"]
    assert gpu.masked and not gpu.runnable
    # Still enumerated, with its real memory: a mask is a policy, not an
    # amnesia, and the record has to show the card was there.
    assert gpu.memory_bytes > 0
    assert "mask" in gpu.reason


def test_mask_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(MASK_ENV, "gpu_discrete, cpu")
    assert parse_mask() == frozenset({"gpu_discrete", "cpu"})
    monkeypatch.delenv(MASK_ENV)
    assert parse_mask() == frozenset()


def test_an_explicit_mask_beats_the_environment(monkeypatch):
    monkeypatch.setenv(MASK_ENV, "gpu_discrete")
    assert parse_mask("") == frozenset()


# ---------------------------------------------------------------- pooling
def test_cpu_and_accelerators_share_one_host_pool():
    """Section 6 forbids giving the iGPU and the NPU each their own RAM."""
    devices = enumerate_devices(blackwell_laptop(), platform_device_type="cpu")
    host_pool = [d for d in devices if d.memory_pool == "host_ram"]
    assert {d.device_id for d in host_pool} >= {"cpu", "npu:amd", "gpu_integrated:amd"}
    # Only the CPU row carries the bytes; the others are 0 so nothing can add
    # three copies of the same RAM together.
    assert sum(d.memory_bytes for d in host_pool) == next(
        d.memory_bytes for d in host_pool if d.device_id == "cpu"
    )


def test_the_discrete_gpu_has_its_own_pool():
    devices = _by_id(enumerate_devices(blackwell_laptop(), platform_device_type="cuda"))
    assert devices["cuda:0"].memory_pool == "vram"
    assert devices["cpu"].memory_pool == "host_ram"


def test_cpu_weight_formats_include_int8_on_x86():
    devices = _by_id(enumerate_devices(blackwell_laptop(), platform_device_type="cpu"))
    assert FORMAT_INT8 in devices["cpu"].weight_formats
    assert FORMAT_GPTQ in devices["cpu"].weight_formats
