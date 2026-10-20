# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pipelines for text models vLLM implements and Omni does not own.

A model being in vLLM's registry does not make it reachable through
``Omni(model=...)``: the orchestrator resolves a pipeline by ``model_type``
before any weight is read, so a ``model_type`` with no entry fails early and
opaquely. Spark-X2.5 hit exactly that; these tests pin the general case.
"""

from __future__ import annotations

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.config.vllm_native_pipelines import LLAMA_PIPELINE


def test_llama_resolves_from_the_registry():
    """MiniCPM5-2B declares model_type "llama"; without this entry the
    orchestrator cannot resolve it at all."""
    assert OMNI_PIPELINES["llama"] is LLAMA_PIPELINE


def test_it_is_one_autoregressive_text_stage():
    assert LLAMA_PIPELINE.get_validation_errors() == []
    assert len(LLAMA_PIPELINE.stages) == 1
    stage = LLAMA_PIPELINE.stages[0]
    assert stage.execution_type is StageExecutionType.LLM_AR
    assert stage.input_sources == ()
    assert stage.final_output and stage.final_output_type == "text"
    assert stage.owns_tokenizer
    assert stage.model_arch == "LlamaForCausalLM"


def test_the_stage_declares_no_cross_chunk_state():
    """A causal LM's state is its KV cache, which the backend owns. Claiming
    otherwise would make the scheduler park requests to hold capacity."""
    assert not LLAMA_PIPELINE.stages[0].retains_state_across_chunks


def test_registering_llama_is_keyed_on_architecture_not_checkpoint():
    """Worth pinning because it is the surprising part: this entry covers every
    checkpoint whose config says "llama", not only the one it was added for.
    That is the right granularity — pipeline shape is a property of the
    architecture — but it is a claim about shape, not a verification of any
    particular weights."""
    assert LLAMA_PIPELINE.model_type == "llama"
    assert LLAMA_PIPELINE.hf_architectures == ("LlamaForCausalLM",)
