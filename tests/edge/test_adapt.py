"""Tests for vllm_omni.edge.adapt (CPU)."""

from __future__ import annotations

import pytest

from vllm_omni.edge import adapt
from vllm_omni.edge.hardware_probe import HardwareProfile

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GiB = 2**30


def _x86(ram=32 * GiB, flags=("avx2", "avx512", "avx512_bf16", "amx_bf16"), n=16):
    return HardwareProfile(
        arch="x86_64",
        cpu_flags=list(flags),
        n_logical=n,
        n_physical=n,
        clusters=[{"max_khz": 3000000, "cpus": list(range(n))}],
        ram_total_bytes=ram,
        ram_available_bytes=ram - 2 * GiB,
        supports_bf16="avx512_bf16" in flags,
    )


def test_dtype_rules():
    assert adapt.choose_dtype(_x86()) == "bfloat16"
    assert adapt.choose_dtype(_x86(flags=("avx2",))) == "float32"
    gpu = HardwareProfile(arch="x86_64", accelerator="cuda_discrete", gpu_mem_bytes=24 * GiB, supports_bf16=False)
    assert adapt.choose_dtype(gpu) == "float16"
    arm = HardwareProfile(arch="aarch64", cpu_flags=["neon", "fp16_arith"], supports_bf16=False)
    assert adapt.choose_dtype(arm) == "float32"


def test_cpu_class_binds_disjoint_ranges_and_absolute_kv_budget():
    prof = _x86(ram=16 * GiB, n=8)
    ov = adapt.derive_overrides(prof, weights_bytes=2 * GiB)
    assert ov["hardware_class"] == "x86_cpu"
    s0, s1 = ov["stages"][0], ov["stages"][1]
    assert s0["devices"] == "cpu" and s1["devices"] == "cpu"
    assert s0["env"]["VLLM_CPU_OMP_THREADS_BIND"] == "0-4"
    assert s1["env"]["VLLM_CPU_OMP_THREADS_BIND"] == "5-7"
    assert s0["enforce_eager"] is True and s1["enforce_eager"] is True
    kv0, kv1 = s0["engine_args"]["kv_cache_memory_bytes"], s1["engine_args"]["kv_cache_memory_bytes"]
    assert kv0 > 0 and kv1 > 0 and kv0 + kv1 == ov["memory_budget_bytes"]
    assert kv0 + kv1 <= prof.ram_available_bytes
    assert ov["connector_extra"]["codec_chunk_adaptive"] is True
    assert ov["timeouts"]["stage_init_timeout"] >= 300


def test_small_vram_gpu_forces_eager_and_tight_budgets():
    prof = HardwareProfile(
        arch="x86_64",
        accelerator="cuda_discrete",
        gpu_name="small",
        gpu_mem_bytes=6 * GiB,
        supports_bf16=True,
        ram_total_bytes=16 * GiB,
    )
    ov = adapt.derive_overrides(prof, weights_bytes=3 * GiB)
    assert ov["hardware_class"] == "cuda_discrete"
    assert ov["stages"][0]["enforce_eager"] is True and ov["stages"][1]["enforce_eager"] is True
    assert ov["stages"][0]["gpu_memory_utilization"] is None
    total_kv = sum(ov["stages"][i]["engine_args"]["kv_cache_memory_bytes"] for i in (0, 1))
    assert total_kv <= 6 * GiB - 3 * GiB
    assert "compilation_config" not in ov["stages"][0]


def test_jetson_unified_memory_uses_ram_and_graphs_off_stage1():
    prof = HardwareProfile(
        arch="aarch64", accelerator="cuda_unified", gpu_mem_bytes=32 * GiB, ram_total_bytes=32 * GiB, supports_bf16=True
    )
    ov = adapt.derive_overrides(prof, weights_bytes=4 * GiB)
    assert ov["hardware_class"] == "jetson"
    assert ov["stages"][1]["enforce_eager"] is True
    assert ov["stages"][0]["enforce_eager"] is False
    assert ov["stages"][0]["compilation_config"]["cudagraph_capture_sizes"] == [1, 2]
    assert ov["stages"][0]["max_num_seqs"] == 2
    assert ov["timeouts"]["init_timeout"] >= ov["timeouts"]["stage_init_timeout"]


def test_budget_never_exceeds_pool_and_seqs_shrink():
    prof = _x86(ram=3 * GiB, n=4)
    prof.ram_available_bytes = 3 * GiB
    ov = adapt.derive_overrides(prof, weights_bytes=1 * GiB)
    assert ov["memory_budget_bytes"] >= 0
    assert ov["stages"][0]["max_num_seqs"] >= 1


def test_split_cpu_ranges_and_cli_flatten():
    assert adapt.split_cpu_ranges([0, 1, 2, 3, 4, 5]) == ("0-3", "4-5")
    assert adapt.split_cpu_ranges([7]) == ("7", "7")
    assert adapt.split_cpu_ranges([0, 2, 3, 4]) == ("0,2-3", "4")
    assert adapt._ranges([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    ov = adapt.derive_overrides(_x86(), weights_bytes=GiB)
    flat = adapt.overrides_to_cli(ov)
    assert flat["async_chunk"] is True and "0" in flat["stage_overrides"] and "stage_init_timeout" in flat


def test_estimate_weights_bytes(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.safetensors").write_bytes(b"y" * 5)
    (tmp_path / "c.bin").write_bytes(b"z" * 100)
    assert adapt.estimate_weights_bytes(tmp_path) == 15
    assert adapt.estimate_weights_bytes(None) == 0
    assert adapt.estimate_weights_bytes(tmp_path / "missing") == 0
