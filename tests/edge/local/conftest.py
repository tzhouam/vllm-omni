# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixtures for the M0 local text mode.

Everything here is synthetic on purpose: a hardware profile built from
literals, and a checkpoint directory with a real ``config.json`` and
zero-filled weight files. The planner reads sizes and geometry, never tensors,
so this exercises the real code path without a GPU or 7.7 GB of weights. The
one test module that does need real weights says so and skips.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from vllm_omni.edge.hardware_probe import (
    ACCEL_GPU_DISCRETE,
    ACCEL_GPU_INTEGRATED,
    ACCEL_NPU,
    HardwareProfile,
)

# A 28-layer Spark-shaped decoder: 7 full-attention and 21 sliding@512, which
# is the 1.7B layout. Small hidden/vocab so nothing here is slow.
SPARK_SHAPED_CONFIG = {
    "architectures": ["Spark2_5ForCausalLM"],
    "model_type": "spark2_5",
    "dtype": "bfloat16",
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 28,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "hidden_act": "gelu",
    "vocab_size": 131072,
    "sliding_window": 512,
    "max_position_embeddings": 1048576,
    "tie_word_embeddings": True,
    "layer_types": [
        "full_attention" if (i + 1) % 4 == 0 else "sliding_attention"
        for i in range(28)
    ],
    "rope_parameters": {
        "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
    },
}

INT8_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "quantization_status": "compressed",
    "ignore": ["lm_head"],
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel"},
        }
    },
}

FP8_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "config_groups": {},
}


def write_checkpoint(
    root: Path,
    *,
    quantization: dict | None = None,
    weight_bytes: int = 4 * 1024 * 1024,
    shards: int = 1,
) -> Path:
    """A checkpoint directory the manifest and planner can read end to end."""
    root.mkdir(parents=True, exist_ok=True)
    config = dict(SPARK_SHAPED_CONFIG)
    if quantization is not None:
        config["quantization_config"] = quantization
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text('{"model_max_length": 4096}', encoding="utf-8")
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    per_shard = max(weight_bytes // shards, 1)
    for i in range(shards):
        name = "model.safetensors" if shards == 1 else f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        with (root / name).open("wb") as fh:
            fh.truncate(per_shard)
    return root


@pytest.fixture
def dense_checkpoint(tmp_path: Path) -> Path:
    return write_checkpoint(tmp_path / "dense", weight_bytes=8 * 1024 * 1024, shards=2)


@pytest.fixture
def int8_checkpoint(tmp_path: Path) -> Path:
    return write_checkpoint(tmp_path / "int8", quantization=INT8_QUANT_CONFIG)


def blackwell_laptop(ram_available_bytes: int = 28 * 2**30) -> HardwareProfile:
    """The reference machine: sm_120 discrete GPU, an iGPU and an NPU beside it.

    Modelled on the probe output recorded in the architecture proposal, so the
    tests exercise the same shape the real host produces.
    """
    return HardwareProfile(
        arch="x86_64",
        cpu_model="AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
        cpu_flags=["avx2", "avx512", "avx512_bf16", "avx512_vnni", "fma"],
        n_logical=24,
        n_physical=12,
        clusters=[{"max_khz": 5000000, "cpus": list(range(24))}],
        ram_total_bytes=int(30.91 * 2**30),
        ram_available_bytes=ram_available_bytes,
        accelerator="cuda_discrete",
        gpu_name="NVIDIA GeForce RTX 5090 Laptop GPU",
        gpu_mem_bytes=24463 * 2**20,
        gpu_sm_count=110,
        gpu_capability=[12, 0],
        supports_bf16=True,
        host_os="wsl2",
        accelerators=[
            {"kind": ACCEL_GPU_DISCRETE, "vendor": "nvidia",
             "name": "NVIDIA GeForce RTX 5090 Laptop GPU", "usable": True,
             "source": "torch.cuda", "mem_bytes": 24463 * 2**20},
            {"kind": ACCEL_NPU, "vendor": "amd", "name": "NPU Compute Accelerator Device",
             "usable": False, "source": "wsl-host", "reason": "no device node under WSL2"},
            {"kind": ACCEL_GPU_INTEGRATED, "vendor": "amd", "name": "AMD Radeon(TM) 890M Graphics",
             "usable": False, "source": "wsl-host", "reason": "no /dev/dri under WSL2"},
        ],
        source="test-fixture",
    )


def cpu_only_box() -> HardwareProfile:
    """The no-NVIDIA deployment class, as a machine rather than as a mask."""
    profile = blackwell_laptop()
    profile.accelerator = "none"
    profile.gpu_name = ""
    profile.gpu_mem_bytes = 0
    profile.gpu_capability = []
    profile.accelerators = [a for a in profile.accelerators if a["vendor"] != "nvidia"]
    return profile


@pytest.fixture
def laptop_profile() -> HardwareProfile:
    return blackwell_laptop()


# [edge-infer W8] The async tests in this directory (engine and serving e2e)
# talk to the engine over zmq.asyncio, which needs a selector event loop.
# Python on Windows defaults to the Proactor loop, and pytest-asyncio builds
# its loops from the current policy, so on win32 the e2e hung silently at the
# first async socket (measured: 30 min with no output, GPU idle) while the
# `accept` CLI -- which installs the selector policy itself -- passed on the
# same venv. Same policy here, set for the whole session.
if sys.platform == "win32":
    from vllm_omni.edge.local.cli import _use_selector_event_loop_on_windows

    _use_selector_event_loop_on_windows()


@pytest.fixture(scope="session")
def event_loop_policy():
    """pytest-asyncio builds every loop from this policy; make it the selector one on Windows."""
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()
