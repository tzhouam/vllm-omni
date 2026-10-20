"""Talker-only Qwen3-TTS pipeline registration (CPU)."""

from __future__ import annotations

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import _DEPLOY_DIR, StageExecutionType, load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_PIPELINE, QWEN3_TTS_TALKER_ONLY_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_registered_single_terminal_stage():
    assert OMNI_PIPELINES["qwen3_tts_talker_only"] is QWEN3_TTS_TALKER_ONLY_PIPELINE
    p = QWEN3_TTS_TALKER_ONLY_PIPELINE
    assert len(p.stages) == 1
    s = p.stages[0]
    assert s.execution_type is StageExecutionType.LLM_AR
    assert s.final_output is True and s.final_output_type == "latent" and s.engine_output_type == "latent"
    assert s.model_stage == QWEN3_TTS_PIPELINE.stages[0].model_stage == "qwen3_tts"
    assert s.custom_process_next_stage_input_func is None
    assert s.async_chunk_process_next_stage_input_func is None
    assert s.sampling_constraints["stop_token_ids"] == [2150]


def test_deploy_yaml_merges_to_one_stage_config():
    deploy_path = _DEPLOY_DIR / "qwen3_tts_talker_only.yaml"
    assert deploy_path.exists()
    deploy = load_deploy_config(deploy_path)
    assert len(deploy.stages) == 1 and deploy.stages[0].devices == "0"
    merged = merge_pipeline_deploy(QWEN3_TTS_TALKER_ONLY_PIPELINE, deploy)
    stage_configs = merged if isinstance(merged, list) else getattr(merged, "stages", merged)
    assert len(stage_configs) == 1
    cfg = stage_configs[0]
    engine_args = getattr(cfg, "engine_args", cfg)
    assert bool(getattr(engine_args, "async_chunk", False)) is False
    assert getattr(engine_args, "custom_process_next_stage_input_func", None) is None
