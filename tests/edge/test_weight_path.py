# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The per-device weight/kernel selector."""

import pytest

from vllm_omni.edge.hardware_probe import HardwareProfile
from vllm_omni.edge.weight_path import has_amx, select_weight_path

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _x86(*, amx: bool = True, **kw) -> HardwareProfile:
    flags = ["avx2", "avx512f"] + (["amx_tile", "amx_bf16"] if amx else [])
    return HardwareProfile(
        arch="x86_64", cpu_flags=flags, n_physical=32, n_logical=64,
        ram_total_bytes=64 << 30, ram_available_bytes=48 << 30,
        supports_bf16=True, **kw,
    )


def test_amx_x86_takes_the_vllm_int4_kernel():
    path = select_weight_path(_x86(amx=True))
    assert path.name == "w4a8-amx"
    assert path.quantization == "gptq"
    assert path.runs_in_vllm and path.verified
    assert path.as_engine_kwargs() == {"quantization": "gptq"}


def test_x86_without_amx_falls_back_to_bf16_activations():
    """No AMX means no int4_scaled_mm_cpu, so tinygemm is the only 4-bit path."""
    path = select_weight_path(_x86(amx=False))
    assert path.name == "w4a16-tinygemm"
    assert path.quantization == "cpu_int4"


def test_fidelity_priority_gives_up_the_amx_kernel():
    """W4A8 quantizes activations to int8; asking for fidelity must not."""
    fast = select_weight_path(_x86(amx=True), priority="speed")
    faithful = select_weight_path(_x86(amx=True), priority="fidelity")
    assert fast.name == "w4a8-amx"
    assert faithful.name == "w4a16-tinygemm"


def test_phone_does_not_run_in_vllm_at_all():
    profile = HardwareProfile(arch="aarch64", accelerator="npu:qualcomm", n_physical=8)
    path = select_weight_path(profile)
    assert path.runs_in_vllm is False
    assert path.as_engine_kwargs() == {}
    assert path.build_with and "export" in path.build_with
    # The calibration lesson has to travel with the recommendation.
    assert any("calibrat" in n.lower() for n in path.notes)


def test_gpu_keeps_the_model_dtype():
    profile = HardwareProfile(
        arch="x86_64", accelerator="cuda_discrete", gpu_mem_bytes=80 << 30,
        supports_bf16=True, n_physical=32,
    )
    path = select_weight_path(profile)
    assert path.name == "bf16"
    assert path.quantization is None
    assert path.fused_cpu_norms is False


def test_arm_cpu_is_offered_but_flagged_unverified():
    """The 4-bit CPU GEMM has an ARM path, but a different kernel than the one
    measured here, so the selector must not claim it was."""
    profile = HardwareProfile(arch="aarch64", cpu_flags=["neon"], n_physical=8)
    path = select_weight_path(profile)
    assert path.quantization == "cpu_int4"
    assert path.verified is False


def test_has_amx_needs_x86_and_the_tile_flag():
    assert has_amx(_x86(amx=True))
    assert not has_amx(_x86(amx=False))
    assert not has_amx(HardwareProfile(arch="aarch64", cpu_flags=["amx_tile"]))


def test_every_path_states_why():
    for profile in (
        _x86(amx=True), _x86(amx=False),
        HardwareProfile(arch="aarch64", accelerator="npu:qualcomm"),
        HardwareProfile(arch="x86_64", accelerator="cuda_discrete"),
    ):
        path = select_weight_path(profile)
        assert len(path.rationale) > 40, path.name
