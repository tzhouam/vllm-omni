"""Derive per-stage deploy overrides from a :class:`HardwareProfile`.

The output is a plain dict in the deploy-YAML shape (``{"stages": {idx: {...}},
"top": {...}, "connector_extra": {...}, "timeouts": {...}}``) so it can be
merged as CLI overrides or written as a variant YAML. Rules are deliberately
simple and documented inline; measured runs (``benchmarks/tts/hw_emulation.py``)
are the way to refine the constants in ``CLASS_TABLE``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_omni.edge.hardware_probe import (
    HW_CLASS_ARM64_CPU,
    HW_CLASS_CUDA_DISCRETE,
    HW_CLASS_JETSON,
    HW_CLASS_NPU_PHONE,
    HW_CLASS_X86_CPU,
    HardwareProfile,
    hardware_class,
)
from vllm_omni.edge.scheduling import KVGeometry, max_seqs_without_preemption

GiB = 2**30


@dataclass(frozen=True)
class ClassDefaults:
    max_num_seqs: int
    max_model_len_stage0: int
    max_model_len_stage1: int
    headroom_bytes: int  # left free for the OS / other apps
    kv_fraction_stage0: float  # share of the KV budget for the AR stage
    chunk_min_frames: int
    chunk_safety_margin_ms: float
    enforce_eager_stage0: bool
    enforce_eager_stage1: bool
    small_vram_bytes: int  # below this, graphs off and budgets tight
    read_bw_bytes_per_s: float  # for the load-time cost model


CLASS_TABLE: dict[str, ClassDefaults] = {
    HW_CLASS_CUDA_DISCRETE: ClassDefaults(4, 4096, 65536, 2 * GiB, 0.7, 2, 50.0, False, False, 8 * GiB, 2.0e9),
    HW_CLASS_JETSON: ClassDefaults(2, 2048, 32768, 3 * GiB, 0.7, 2, 80.0, False, True, 16 * GiB, 0.8e9),
    HW_CLASS_X86_CPU: ClassDefaults(2, 2048, 16384, 2 * GiB, 0.7, 2, 120.0, True, True, 0, 1.0e9),
    HW_CLASS_ARM64_CPU: ClassDefaults(1, 2048, 16384, 1 * GiB, 0.7, 3, 160.0, True, True, 0, 0.4e9),
    HW_CLASS_NPU_PHONE: ClassDefaults(1, 2048, 16384, 1 * GiB, 0.7, 4, 200.0, True, True, 0, 0.3e9),
}


def estimate_weights_bytes(model_dir: str | Path | None) -> int:
    """Sum of ``*.safetensors`` under ``model_dir`` (0 when unknown)."""
    if not model_dir:
        return 0
    root = Path(model_dir)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*.safetensors"))


def choose_dtype(profile: HardwareProfile) -> str:
    if profile.accelerator.startswith("cuda"):
        return "bfloat16" if profile.supports_bf16 else "float16"
    if profile.supports_bf16:
        return "bfloat16"
    # CPUs without bf16: fp16 matmuls are slow/unsupported on most CPUs -> fp32.
    return "float32"


def memory_budget_bytes(profile: HardwareProfile, weights_bytes: int, defaults: ClassDefaults) -> int:
    """Bytes available for KV caches after weights and headroom."""
    if profile.accelerator == "cuda_discrete" and profile.gpu_mem_bytes:
        pool = profile.gpu_mem_bytes
    elif profile.accelerator == "cuda_unified":
        pool = profile.gpu_mem_bytes or profile.ram_total_bytes
    else:
        pool = profile.ram_available_bytes or profile.ram_total_bytes
    # Activations + allocator fragmentation: 15 % of weights, at least 512 MiB.
    activations = max(int(0.15 * weights_bytes), 512 * 2**20)
    return max(0, pool - weights_bytes - activations - defaults.headroom_bytes)


def split_cpu_ranges(cpus: list[int], share_stage0: float = 2 / 3) -> tuple[str, str]:
    """Split a CPU list into two disjoint range strings for stage 0 / stage 1."""
    if not cpus:
        return "", ""
    n0 = max(1, int(round(len(cpus) * share_stage0)))
    if len(cpus) > 1:
        n0 = min(n0, len(cpus) - 1)
    a, b = cpus[:n0], cpus[n0:] or cpus[-1:]
    return _ranges(a), _ranges(b)


def _ranges(cpus: list[int]) -> str:
    cpus = sorted(set(cpus))
    parts: list[str] = []
    start = prev = cpus[0]
    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = c
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


def derive_overrides(
    profile: HardwareProfile,
    *,
    weights_bytes: int = 0,
    n_stages: int = 2,
    max_cpus: int | None = None,
    kv_geometry: KVGeometry | None = None,
) -> dict[str, Any]:
    """Derive deploy overrides for an AR-talker + generation-decoder pipeline."""
    cls = hardware_class(profile)
    d = CLASS_TABLE[cls]
    dtype = choose_dtype(profile)
    budget = memory_budget_bytes(profile, weights_bytes, d)
    kv0 = int(budget * d.kv_fraction_stage0)
    kv1 = int(budget - kv0)
    # No-preemption admission (WP8): the talker KV budget must hold
    # max_num_seqs x max_model_len tokens, otherwise vLLM preempts by recompute.
    # Exact bytes/token when the model's KV geometry is known (Qwen3-TTS 0.6B/1.7B:
    # 28 layers x 8 KV heads x 128 x 2 B x 2 = 112 KiB), else a 8 KiB/token floor
    # that keeps the historical behaviour for unknown models.
    bytes_per_token = kv_geometry.bytes_per_token(dtype) if kv_geometry is not None else 8 * 1024
    max_num_seqs = max(
        1, min(d.max_num_seqs, max_seqs_without_preemption(kv0, d.max_model_len_stage0, bytes_per_token))
    )

    small_vram = (
        profile.accelerator.startswith("cuda")
        and (profile.gpu_mem_bytes or profile.ram_total_bytes) < d.small_vram_bytes
    )
    eager0 = d.enforce_eager_stage0 or small_vram
    eager1 = d.enforce_eager_stage1 or small_vram

    stage0: dict[str, Any] = {
        "max_num_seqs": max_num_seqs,
        "max_model_len": d.max_model_len_stage0,
        "enforce_eager": eager0,
        "engine_args": {"dtype": dtype, "kv_cache_memory_bytes": kv0},
    }
    stage1: dict[str, Any] = {
        "max_num_seqs": max_num_seqs,
        "max_model_len": d.max_model_len_stage1,
        "enforce_eager": eager1,
        "engine_args": {"dtype": dtype, "kv_cache_memory_bytes": kv1},
    }
    if profile.accelerator.startswith("cuda"):
        stage0["gpu_memory_utilization"] = None  # absolute budget replaces the fraction
        stage1["gpu_memory_utilization"] = None
        if not eager0:
            stage0["compilation_config"] = {"cudagraph_capture_sizes": sorted({1, 2, max_num_seqs})}
    else:
        cpus = profile.big_cpus
        if max_cpus:
            cpus = cpus[:max_cpus]
        r0, r1 = split_cpu_ranges(cpus)
        stage0["devices"] = "cpu"
        stage1["devices"] = "cpu"
        stage0["env"] = {"VLLM_CPU_OMP_THREADS_BIND": r0}
        stage1["env"] = {"VLLM_CPU_OMP_THREADS_BIND": r1}

    connector_extra = {
        "codec_chunk_adaptive": True,
        "codec_chunk_min_frames": d.chunk_min_frames,
        "codec_chunk_safety_margin_ms": d.chunk_safety_margin_ms,
        "decode_batch_max_size": 1,
    }
    load_s = (weights_bytes / d.read_bw_bytes_per_s) if weights_bytes else 60.0
    compile_s = 0.0 if (eager0 and eager1) else 240.0
    stage_init = int(math.ceil((load_s + compile_s) * 2.0 + 60))
    timeouts = {"stage_init_timeout": max(300, stage_init), "init_timeout": max(600, stage_init * max(1, n_stages))}
    return {
        "hardware_class": cls,
        "dtype": dtype,
        "memory_budget_bytes": budget,
        "stages": {0: stage0, 1: stage1},
        "top": {"async_chunk": True},
        "connector_extra": connector_extra,
        "timeouts": timeouts,
        "notes": [
            f"class={cls} dtype={dtype} budget={budget / GiB:.2f}GiB weights={weights_bytes / GiB:.2f}GiB",
            f"small_vram={small_vram} eager=({eager0},{eager1}) max_num_seqs={max_num_seqs}",
        ],
    }


def overrides_to_cli(overrides: dict[str, Any]) -> dict[str, Any]:
    """Flatten for ``StageConfigFactory.create_from_model(cli_overrides=...)``-style consumers."""
    flat: dict[str, Any] = {}
    for k, v in overrides.get("top", {}).items():
        flat[k] = v
    flat["stage_overrides"] = {str(i): s for i, s in overrides.get("stages", {}).items()}
    flat["connector_extra"] = dict(overrides.get("connector_extra", {}))
    flat.update({k: v for k, v in overrides.get("timeouts", {}).items()})
    return flat
