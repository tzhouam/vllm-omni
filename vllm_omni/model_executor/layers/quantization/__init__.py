# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Out-of-tree quantization methods registered into vLLM."""

from vllm_omni.model_executor.layers.quantization.cpu_int4 import (  # noqa: F401
    CPUInt4Config,
)

__all__ = ["CPUInt4Config"]
