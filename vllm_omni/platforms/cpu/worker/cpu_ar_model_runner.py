# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU model runners for Omni AR and generation stages.

MRO: ``CPUARModelRunner -> GPUARModelRunner -> OmniGPUModelRunner -> (omni
mixins) -> CPUModelRunner -> GPUModelRunner``. ``CPUModelRunner.__init__``
runs vLLM's CPU shims (torch.cuda placeholders, ``_postprocess_tensors``,
``_postprocess_triton``) around ``GPUModelRunner.__init__``; the Omni layers
then add their buffers on the CPU device. Nothing CUDA-graph related can be
reached because ``CpuPlatform.check_and_update_config`` forces
``cudagraph_capture_sizes=[]`` and ``async_scheduling=False``.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.worker.cpu_model_runner import CPUModelRunner

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

logger = init_logger(__name__)


class _CPUOmniRunnerMixin:
    """Overrides shared by the CPU AR and generation runners."""

    def _assert_cpu_invariants(self) -> None:
        if getattr(self, "use_async_scheduling", False):
            raise RuntimeError(
                "CPU Omni runners require async_scheduling=False (vLLM's CpuPlatform forces it); "
                "the async omni-output path uses CUDA streams."
            )

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        return None

    def capture_model(self) -> int:
        # No graphs on CPU; keep the call cheap and explicit.
        return 0

    def _capture_talker_mtp_graphs(self) -> None:
        return None


class CPUARModelRunner(_CPUOmniRunnerMixin, GPUARModelRunner, CPUModelRunner):
    """AR (talker / thinker) runner on CPU."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_cpu_invariants()


class CPUGenerationModelRunner(_CPUOmniRunnerMixin, GPUGenerationModelRunner, CPUModelRunner):
    """Generation (code2wav / decoder) runner on CPU."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_cpu_invariants()
