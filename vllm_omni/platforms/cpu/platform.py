# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU OmniPlatform: AR / generation stages on vLLM's CPU backend.

Composes vLLM's ``CpuPlatform`` (device attributes, ``check_and_update_config``
forcing: no async scheduling, no CUDA graphs, ``CPU_ATTN``, thread binding,
LD_PRELOAD) with the Omni interface, exactly as ``CudaOmniPlatform`` composes
``CudaPlatformBase``. Diffusion stages are not supported on CPU.

Activate with ``VLLM_TARGET_DEVICE=cpu`` (or a ``+cpu`` vLLM wheel). Deploy
YAMLs can pin CPU values under ``platforms: cpu:``.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform

from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


def _numa_memory(device: torch.device | None = None) -> tuple[int, int]:
    """(available, total) bytes of the NUMA node this process is allowed on, else host memory."""
    try:
        from vllm.utils.cpu_resource_utils import get_allowed_cpu_list, get_memory_node_info

        node = get_allowed_cpu_list()[0].numa_node
        info = get_memory_node_info(node)
        return int(info.available_memory), int(info.total_memory)
    except Exception:
        try:
            import psutil

            vm = psutil.virtual_memory()
            return int(vm.available), int(vm.total)
        except Exception:
            return 0, 0


class _NoGraphWrapper:
    """Sentinel graph-wrapper type for the CPU platform (never instantiated)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - guard
        raise NotImplementedError("CUDA graphs are not available on the CPU OmniPlatform")


class CPUOmniPlatform(OmniPlatform, CpuPlatform):
    """CPU implementation of OmniPlatform (AR + generation stages only)."""

    _omni_enum = OmniPlatformEnum.CPU

    # ----------------------------------------------------------- workers
    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.platforms.cpu.worker.cpu_ar_worker.CPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.platforms.cpu.worker.cpu_generation_worker.CPUGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/deploy"

    # ---------------------------------------------------------- diffusion
    @classmethod
    def get_diffusion_worker_cls(cls) -> str:
        raise NotImplementedError("Diffusion stages are not supported on the CPU OmniPlatform")

    @classmethod
    def get_diffusion_model_runner_cls(cls) -> str:
        raise NotImplementedError("Diffusion stages are not supported on the CPU OmniPlatform")

    @classmethod
    def get_diffusion_attn_backend_cls(cls, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Diffusion attention backends are not supported on the CPU OmniPlatform")

    @classmethod
    def supports_diffusion_dense_flash_attention(cls) -> bool:
        return False

    @classmethod
    def has_flash_attn_package(cls) -> bool:
        return False

    # ------------------------------------------------------------ compile
    @classmethod
    def supports_torch_inductor(cls) -> bool:
        # The code predictor's torch.compile costs a long C++ codegen on first
        # use; keep it off unless explicitly requested (measured variant).
        return os.environ.get("VLLM_OMNI_CPU_INDUCTOR", "0") == "1"

    @classmethod
    def supports_talker_mtp_graph_capture(cls) -> bool:
        return False

    @classmethod
    def get_graph_wrapper_cls(cls) -> type:
        # Runners use this in ``isinstance`` checks on every decode step
        # (``_talker_mtp_forward``); on CPU nothing is ever wrapped, so return a
        # sentinel class that no object is an instance of.
        return _NoGraphWrapper

    # ------------------------------------------------------------- device
    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        return torch.device("cpu")

    @classmethod
    def get_device_count(cls) -> int:
        # One logical device: stage device locks / admission iterate range(count).
        return 1

    @classmethod
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    def synchronize(cls) -> None:
        return None

    @classmethod
    def record_device_event(cls):
        return None

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        return _numa_memory(device)[0]

    @classmethod
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
        return _numa_memory(device)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        try:
            return int(CpuPlatform.get_device_total_memory(device_id))
        except Exception:
            return _numa_memory()[1]

    @classmethod
    def supports_cpu_offload(cls) -> bool:
        return False

    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int] | None = None) -> bool:
        return True

    @classmethod
    def get_default_ir_op_priority(cls, *args: Any, **kwargs: Any):
        parent = getattr(CpuPlatform, "get_default_ir_op_priority", None)
        if parent is not None:
            return parent(*args, **kwargs)
        return super().get_default_ir_op_priority(*args, **kwargs)  # type: ignore[misc]
