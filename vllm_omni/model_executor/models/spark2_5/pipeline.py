# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Spark-X2.5 as an Omni pipeline: one autoregressive text stage.

Spark's model definition has been in ``vllm_omni.model_executor.models`` for a
while, but only as a vLLM architecture -- there was no pipeline, so
``Omni(model=<a Spark checkpoint>)`` had nothing to resolve and the model was
reachable only through plain ``vllm.LLM``. That is the gap M0 closes: the text
mode has to run *through* the Omni stage runtime, or none of the stage
lifecycle, admission or cancellation work applies to it.

There is exactly one stage, and that is the design, not a simplification to be
undone later. The proposal's first decision is that an autoregressive session
stays on one backend by default: prefill and decode share the KV cache, and a
per-token boundary between devices buys nothing and costs a synchronization.
Spark's tokenizer, chat template and sampling are model-adapter concerns that
run in-process around this stage; they are not stages.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

SPARK2_5_PIPELINE = PipelineConfig(
    model_type="spark2_5",
    model_arch="Spark2_5ForCausalLM",
    hf_architectures=("Spark2_5ForCausalLM",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="spark2_5",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            engine_output_type="text",
            model_arch="Spark2_5ForCausalLM",
        ),
    ),
)
