# SPDX-License-Identifier: Apache-2.0
"""One whole-session Qwen3.8 GGUF stage for bounded text and image requests."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig


QWEN3_8_GGUF_MULTIMODAL_PIPELINE = PipelineConfig(
    model_type="qwen3_8_gguf_multimodal",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="text_vision",
            execution_type=StageExecutionType.GRAPH,
            final_output=True,
            final_output_type="text",
        ),
    ),
)
