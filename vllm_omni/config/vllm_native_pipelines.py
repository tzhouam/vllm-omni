# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pipelines for text models vLLM already implements and Omni does not own.

Every other pipeline in the registry sits next to an Omni model definition, in
``vllm_omni/model_executor/models/<name>/pipeline.py``. These have no such
directory, because there is no Omni model to put there: the model executor is
vLLM's, and all that is missing is the one thing vLLM has no concept of -- how
many stages the pipeline has and what they produce.

That gap is worth naming, because it is not obvious from either side. A model
being in vLLM's registry does not make it reachable through ``Omni(model=...)``:
the orchestrator resolves a pipeline by ``model_type`` first, and a
``model_type`` with no entry here fails before any weight is read. Spark-X2.5
hit exactly this in M0 -- its model definition had been registered for a while
and ``Omni(model=<checkpoint>)`` still could not resolve it.

**One stage, and the model_type is the key.** These are autoregressive causal
language models with a single stage that owns the tokenizer and emits text.
Because the registry is keyed on ``model_type`` rather than on a checkpoint,
registering ``llama`` covers every checkpoint whose config says ``llama`` -- not
only the one this was added for. That is the correct granularity (the pipeline
shape really is a property of the architecture, not of the weights), but it
should be read as what it is: a claim about the *shape* of the pipeline, not a
statement that any particular checkpoint has been run and verified. What has
actually been run is recorded in ``analysis/experiments/``.
"""

from __future__ import annotations

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)


def _single_stage_text(model_type: str, model_arch: str, hf_architectures: tuple[str, ...]) -> PipelineConfig:
    """One autoregressive stage that owns the tokenizer and emits text."""
    return PipelineConfig(
        model_type=model_type,
        model_arch=model_arch,
        hf_architectures=hf_architectures,
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage=model_type,
                execution_type=StageExecutionType.LLM_AR,
                input_sources=(),
                final_output=True,
                final_output_type="text",
                owns_tokenizer=True,
                engine_output_type="text",
                model_arch=model_arch,
            ),
        ),
    )


LLAMA_PIPELINE = _single_stage_text(
    model_type="llama",
    model_arch="LlamaForCausalLM",
    hf_architectures=("LlamaForCausalLM",),
)
"""The Llama architecture as a single-stage Omni text pipeline.

Added for MiniCPM5-2B, whose ``config.json`` declares ``model_type: "llama"``
and ``architectures: ["LlamaForCausalLM"]`` -- it is a Llama-shaped 2B model
(42 layers, 16 heads over 2 KV heads, head_dim 128, 130560 vocab, rope_theta
5e6), not a new architecture, so the support it needed was this entry plus the
runner no longer forcing omni-only kwargs on a forward that cannot take them
(see ``OmniGPUModelRunner._accepted_forward_kwargs``).
"""
