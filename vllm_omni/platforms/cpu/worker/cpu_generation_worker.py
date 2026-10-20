# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Import-path shim: the CPU generation worker lives next to the AR worker."""

from vllm_omni.platforms.cpu.worker.cpu_ar_worker import CPUGenerationWorker

__all__ = ["CPUGenerationWorker"]
