"""Explicit CPU/GPU llama.cpp split claims must match backend load evidence."""

from __future__ import annotations

import pytest

from vllm_omni.edge.agent.omni_backend import OmniLlamaConfig
from vllm_omni.edge.agent.native_app import _gpu_pool_refusal
from vllm_omni.engine.backends.llamacpp import (
    _hybrid_placement_evidence, _verify_vision_backend,
)
from vllm_omni.engine.resource_ledger import ResourceUnavailable


MIB = 1 << 20


def _config(**changes) -> OmniLlamaConfig:
    values = dict(
        route_id="split", model_file="model.gguf", model_sha256="a" * 64,
        server_bin="llama-server", server_sha256="b" * 64,
        log_file="load.log", placement="cpu+Vulkan0",
        capacities={"host_ram": 8_000 * MIB, "vram": 4_000 * MIB},
        demands={"host_ram": 6_000 * MIB, "vram": 3_000 * MIB},
        memory_overhead_bytes=1_000 * MIB,
        vram_overhead_bytes=1_000 * MIB,
        cpu_weight_budget_bytes=5_000 * MIB,
        gpu_weight_budget_bytes=2_000 * MIB,
        gpu_layers=10,
    )
    values.update(changes)
    return OmniLlamaConfig(**values)


def test_hybrid_contract_needs_explicit_dual_pool_budget_and_split() -> None:
    assert _config().placement == "cpu+Vulkan0"
    with pytest.raises(ValueError, match="split"):
        _config(gpu_layers=None)
    with pytest.raises(ValueError, match="host RAM claim"):
        _config(demands={"host_ram": 5_000 * MIB, "vram": 3_000 * MIB})
    with pytest.raises(ValueError, match="VRAM claim"):
        _config(demands={"host_ram": 6_000 * MIB, "vram": 2_000 * MIB})


def test_host_mapped_contract_is_a_separate_explicit_route() -> None:
    route = _config(
        placement="Vulkan_Host+Vulkan0", gpu_layers=None,
        host_mapped_expert_layers=40,
    )
    assert route.host_mapped_expert_layers == 40
    with pytest.raises(ValueError, match="exact host-mapped expert count"):
        _config(placement="Vulkan_Host+Vulkan0", gpu_layers=None)
    with pytest.raises(ValueError, match="no CPU or GPU layer split"):
        _config(placement="Vulkan_Host+Vulkan0", gpu_layers=None,
                host_mapped_expert_layers=40, cpu_moe_layers=40)
    with pytest.raises(ValueError, match="no CPU or GPU layer split"):
        _config(placement="Vulkan_Host+Vulkan0", host_mapped_expert_layers=40)
    with pytest.raises(ValueError, match="cannot claim Vulkan_Host"):
        _config(host_mapped_expert_layers=40)
    with pytest.raises(ValueError, match="host RAM claim"):
        _config(placement="Vulkan_Host+Vulkan0", gpu_layers=None,
                host_mapped_expert_layers=40,
                demands={"host_ram": 5_000 * MIB, "vram": 3_000 * MIB})


def test_integrated_gpu_and_cpu_use_one_physical_ram_pool() -> None:
    shared = _config(
        gpu_memory_pool="host_ram",
        capacities={"host_ram": 12_000 * MIB},
        demands={"host_ram": 9_000 * MIB},
    )
    assert shared.gpu_memory_pool == "host_ram"
    pure_igpu = _config(
        placement="Vulkan0", gpu_layers=None, cpu_weight_budget_bytes=None,
        gpu_weight_budget_bytes=None, vram_overhead_bytes=0,
        gpu_memory_pool="host_ram", capacities={"host_ram": 12_000 * MIB},
        demands={"host_ram": 9_000 * MIB},
    )
    assert pure_igpu.demands == {"host_ram": 9_000 * MIB}
    with pytest.raises(ValueError, match="combined CPU/iGPU"):
        _config(gpu_memory_pool="host_ram",
                capacities={"host_ram": 12_000 * MIB},
                demands={"host_ram": 8_000 * MIB})
    with pytest.raises(ValueError, match="separate VRAM"):
        _config(gpu_memory_pool="host_ram")


def test_native_vram_ceiling_requires_exact_measured_gpu() -> None:
    no_nvidia = {"gpu_name": None}
    igpu = {"placement": "cpu+Vulkan0", "gpu_memory_pool": "host_ram",
            "integrated_gpu": True, "expected_device_name": "AMD Radeon 890M"}
    assert _gpu_pool_refusal(igpu, no_nvidia) is None
    assert "NVML GPU 0" in _gpu_pool_refusal(
        {**igpu, "gpu_memory_pool": "vram"}, no_nvidia)
    mixed = {"gpu_name": "NVIDIA GeForce RTX 5090 Laptop GPU"}
    assert _gpu_pool_refusal(igpu, mixed) is None
    assert "NVML GPU 0" in _gpu_pool_refusal(
        {**igpu, "gpu_memory_pool": "vram"}, mixed)
    assert "integrated GPU" in _gpu_pool_refusal(
        {**igpu, "expected_device_name": mixed["gpu_name"]}, mixed)


def test_hybrid_layer_split_verifies_cpu_and_gpu_buffers() -> None:
    log = "\n".join((
        "llama_model_load: using device Vulkan0 (RTX 5090 Laptop)",
        "load_tensors: offloaded 10/41 layers to GPU",
        "load_tensors: layer   0 assigned to device CPU, is_swa = 0",
        "load_tensors: layer  31 assigned to device Vulkan0, is_swa = 0",
        "load_tensors: CPU_Mapped model buffer size = 500.00 MiB",
        "load_tensors: Vulkan0 model buffer size = 700.00 MiB",
    ))
    evidence = _hybrid_placement_evidence(
        log, gpu_device="Vulkan0", expected_device_name="RTX 5090 Laptop",
        gpu_layers=10, cpu_moe_layers=0,
        cpu_weight_budget_bytes=501 * MIB,
        gpu_weight_budget_bytes=701 * MIB,
    )
    assert evidence["offloaded_layers"] == (10, 41)
    assert evidence["model_buffer_bytes_by_pool"]["host_ram"] > 500 * MIB
    with pytest.raises(ResourceUnavailable, match="per-pool"):
        _hybrid_placement_evidence(
            log, gpu_device="Vulkan0", expected_device_name="RTX 5090 Laptop",
            gpu_layers=10, cpu_moe_layers=0,
            cpu_weight_budget_bytes=499 * MIB,
            gpu_weight_budget_bytes=701 * MIB,
        )
    with pytest.raises(RuntimeError, match="layer count"):
        _hybrid_placement_evidence(
            log, gpu_device="Vulkan0", expected_device_name="RTX 5090 Laptop",
            gpu_layers=12, cpu_moe_layers=0,
            cpu_weight_budget_bytes=501 * MIB,
            gpu_weight_budget_bytes=701 * MIB,
        )


def test_hybrid_cpu_expert_split_requires_override_evidence() -> None:
    log = "\n".join((
        "llama_model_load: using device Vulkan0 (RTX 5090 Laptop)",
        "load_tensors: offloaded 41/41 layers to GPU",
        "load_tensors: layer   0 assigned to device Vulkan0, is_swa = 0",
        "load_tensors: CPU model buffer size = 800.00 MiB",
        "load_tensors: Vulkan0 model buffer size = 700.00 MiB",
        *(f"tensor blk.0.ffn_{tensor}_exps.weight (400 MiB Q4) buffer type overridden to CPU"
          for tensor in ("up", "down", "gate")),
    ))
    evidence = _hybrid_placement_evidence(
        log, gpu_device="Vulkan0", expected_device_name="RTX 5090 Laptop",
        gpu_layers=None, cpu_moe_layers=1,
        cpu_weight_budget_bytes=801 * MIB,
        gpu_weight_budget_bytes=701 * MIB,
    )
    assert evidence["expert_cpu_tensors_verified"] == 3
    with pytest.raises(RuntimeError, match="exact CPU-expert"):
        _hybrid_placement_evidence(
            log.replace("tensor blk.0.ffn_gate_exps.weight (400 MiB Q4) buffer type overridden to CPU", ""),
            gpu_device="Vulkan0", expected_device_name="RTX 5090 Laptop",
            gpu_layers=None, cpu_moe_layers=1,
            cpu_weight_budget_bytes=801 * MIB,
            gpu_weight_budget_bytes=701 * MIB,
        )
    with pytest.raises(RuntimeError, match="exact CPU-expert"):
        _hybrid_placement_evidence(
            log.replace("blk.0.ffn_gate_exps", "blk.1.ffn_gate_exps"),
            gpu_device="Vulkan0", expected_device_name="RTX 5090 Laptop",
            gpu_layers=None, cpu_moe_layers=1,
            cpu_weight_budget_bytes=801 * MIB,
            gpu_weight_budget_bytes=701 * MIB,
        )
    with pytest.raises(RuntimeError, match="CPU-expert tensor override"):
        _hybrid_placement_evidence(
            log.replace("overridden to CPU", "overridden to Vulkan0"),
            gpu_device="Vulkan0", expected_device_name="RTX 5090 Laptop",
            gpu_layers=None, cpu_moe_layers=1,
            cpu_weight_budget_bytes=801 * MIB,
            gpu_weight_budget_bytes=701 * MIB,
        )


def _host_mapped_log() -> str:
    return "\n".join([
        "llama_model_load: using device Vulkan0 (RTX 5090 Laptop)",
        "load_tensors: offloaded 3/3 layers to GPU",
        *(f"load_tensors: layer {layer} assigned to device Vulkan0"
          for layer in range(3)),
        "load_tensors: CPU model buffer size = 800.00 MiB",
        "load_tensors: Vulkan_Host model buffer size = 100.00 MiB",
        "load_tensors: Vulkan0 model buffer size = 700.00 MiB",
        *(f"tensor blk.{layer}.ffn_{tensor}_exps.weight (redacted) "
          "buffer type overridden to Vulkan_Host"
          for layer in range(2) for tensor in ("up", "down", "gate")),
    ])


def _host_mapped_evidence(log: str, **changes):
    args = dict(
        gpu_device="Vulkan0", expected_device_name="RTX 5090 Laptop",
        gpu_layers=None, cpu_moe_layers=0, host_mapped_expert_layers=2,
        cpu_weight_budget_bytes=901 * MIB,
        gpu_weight_budget_bytes=701 * MIB,
    )
    args.update(changes)
    return _hybrid_placement_evidence(log, **args)


def test_host_mapped_experts_require_every_exact_override_and_gpu_layer() -> None:
    log = _host_mapped_log()
    evidence = _host_mapped_evidence(log)
    assert evidence["expert_host_mapped_tensors_selected"] == 6
    assert evidence["expert_cpu_tensors_verified"] == 0
    assert evidence["placement_evidence_level"] == "override_selection_only"
    assert evidence["expert_final_storage_verified"] is False
    assert evidence["expert_compute_verified"] is False
    assert evidence["model_buffer_bytes_by_pool"]["host_ram"] > 900 * MIB
    with pytest.raises(ResourceUnavailable, match="per-pool"):
        _host_mapped_evidence(log, cpu_weight_budget_bytes=899 * MIB)
    with pytest.raises(RuntimeError, match="exact Vulkan_Host"):
        _host_mapped_evidence(log.replace(
            "tensor blk.1.ffn_gate_exps.weight (redacted) buffer type overridden to Vulkan_Host", "",
        ))
    with pytest.raises(RuntimeError, match="exact Vulkan_Host"):
        _host_mapped_evidence(log.replace("blk.1.ffn_gate_exps", "blk.2.ffn_gate_exps"))
    with pytest.raises(RuntimeError, match="exact Vulkan_Host"):
        _host_mapped_evidence(log.replace("overridden to Vulkan_Host", "overridden to CPU", 1))
    with pytest.raises(RuntimeError, match="exact Vulkan_Host"):
        _host_mapped_evidence(log.replace("overridden to Vulkan_Host", "overridden to Vulkan0", 1))
    with pytest.raises(RuntimeError, match="exact Vulkan_Host"):
        _host_mapped_evidence(log.replace("layer 2 assigned to device Vulkan0", "layer 2 assigned to device CPU"))
    with pytest.raises(RuntimeError, match="GPU identity"):
        _host_mapped_evidence(log.replace("using device Vulkan0", "using device Vulkan1"))


def test_host_mapped_vision_backend_still_requires_requested_gpu() -> None:
    _verify_vision_backend(["Vulkan0"], expected_device="Vulkan0")
    for observed in ([], ["CPU"], ["Vulkan_Host"], ["Vulkan1"], ["Vulkan0", "CPU"]):
        with pytest.raises(RuntimeError, match="vision projector backend"):
            _verify_vision_backend(observed, expected_device="Vulkan0")
