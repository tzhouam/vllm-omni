# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU workers for Omni AR and generation stages (XPU-worker pattern).

``CPUWorker`` owns the CPU specifics (NUMA binding, ``init_cpu_memory_env``,
gloo, NUMA-based ``determine_available_memory``, no cudagraph warm-up); the
Omni mixin only loads omni plugins in the worker process. Sleep/wake are not
supported on CPU and return an error ACK instead of raising in the RPC loop.
"""

from __future__ import annotations

import vllm.v1.worker.cpu.shm  # noqa: F401  # patches torch.Event/torch.cuda.Stream; must be first
from vllm.logger import init_logger
from vllm.v1.worker.cpu_worker import CPUWorker

from vllm_omni.diffusion.data import OmniACK
from vllm_omni.engine.init_timeline import timed_phase
from vllm_omni.platforms.cpu.worker.cpu_ar_model_runner import CPUARModelRunner, CPUGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


class _CPUOmniWorkerMixin:
    model_runner_cls: type

    @timed_phase("init_device")
    def init_device(self) -> None:
        super().init_device()  # type: ignore[misc]
        if getattr(self, "use_v2_model_runner", False):
            logger.warning("OMNI CPU worker forces v1 model runner for omni hooks.")
            self.use_v2_model_runner = False
        # CPUWorker.init_device built a plain CPUModelRunner; replace it with the omni runner.
        self.model_runner = self.model_runner_cls(self.vllm_config, self.device)

    @staticmethod
    def _task_id(task) -> str:
        if isinstance(task, dict):
            return str(task.get("task_id", ""))
        return str(getattr(task, "task_id", ""))

    def handle_sleep_task(self, task) -> OmniACK:
        return OmniACK(
            task_id=self._task_id(task), status="ERROR", error_msg="sleep mode is not supported on the CPU OmniPlatform"
        )

    def handle_wake_task(self, task) -> OmniACK:
        return OmniACK(
            task_id=self._task_id(task), status="ERROR", error_msg="wake-up is not supported on the CPU OmniPlatform"
        )


class CPUARWorker(_CPUOmniWorkerMixin, OmniWorkerMixin, CPUWorker):
    """CPU AR worker for thinker / talker stages."""

    model_runner_cls = CPUARModelRunner


class CPUGenerationWorker(_CPUOmniWorkerMixin, OmniWorkerMixin, CPUWorker):
    """CPU generation worker for decoder / code2wav stages."""

    model_runner_cls = CPUGenerationModelRunner
