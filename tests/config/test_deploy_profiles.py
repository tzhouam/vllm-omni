"""Tests for named deploy profiles (``vllm_omni/deploy/<profile>/<model_type>.yaml``)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from vllm_omni.config.stage_config import _DEPLOY_DIR, load_deploy_config, resolve_deploy_yaml
from vllm_omni.entrypoints import utils as entry_utils
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import parse_chunk_ramp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EDGE_QWEN3_TTS = _DEPLOY_DIR / "edge" / "qwen3_tts.yaml"


def test_edge_profile_file_exists():
    assert EDGE_QWEN3_TTS.exists()
    assert "edge" in entry_utils.available_deploy_profiles()
    assert "edge" in entry_utils.available_deploy_profiles("qwen3_tts")


def test_edge_profile_resolves_two_stages_on_one_device():
    cfg = load_deploy_config(EDGE_QWEN3_TTS)
    assert len(cfg.stages) == 2
    assert all(stage.devices == "0" for stage in cfg.stages)
    assert cfg.stages[0].max_num_seqs == 4
    assert cfg.stages[1].max_num_seqs == 4
    assert cfg.async_chunk is True
    # Inherited from the base config: stage 0 keeps its sampling defaults.
    assert cfg.stages[0].default_sampling_params is not None


def test_edge_profile_ramp_parses_and_ends_at_steady_chunk():
    raw = resolve_deploy_yaml(EDGE_QWEN3_TTS)
    extra = raw["connectors"]["connector_of_shared_memory"]["extra"]
    steady = int(extra["codec_chunk_frames"])
    ramp = parse_chunk_ramp(extra, steady=steady)
    assert ramp == [2, 4, 8, 16, 25]
    assert ramp[-1] == steady
    assert extra["decode_batch_max_size"] == 1
    assert extra["codec_left_context_frames"] == 72


def test_deploy_path_for_profile_lookup():
    assert entry_utils.deploy_path_for_profile("qwen3_tts", "edge") == EDGE_QWEN3_TTS
    assert entry_utils.deploy_path_for_profile("qwen3_tts", "no_such_profile") is None
    assert entry_utils.deploy_path_for_profile("no_such_model", "edge") is None
    assert entry_utils.deploy_path_for_profile("qwen3_tts", "") is None


def test_resolve_deploy_profile_path_uses_default_stem():
    with mock.patch.object(entry_utils, "resolve_model_config_path", return_value=str(_DEPLOY_DIR / "qwen3_tts.yaml")):
        assert entry_utils.resolve_deploy_profile_path("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "edge") == str(
            EDGE_QWEN3_TTS
        )
        with pytest.raises(FileNotFoundError, match="no_such_profile"):
            entry_utils.resolve_deploy_profile_path("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "no_such_profile")


def test_apply_deploy_profile_translates_and_rejects_conflicts():
    with mock.patch.object(entry_utils, "resolve_model_config_path", return_value=str(_DEPLOY_DIR / "qwen3_tts.yaml")):
        args = {"deploy_profile": "edge", "other": 1}
        out = entry_utils.apply_deploy_profile("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", args)
        assert out["deploy_config"] == str(EDGE_QWEN3_TTS)
        assert "deploy_profile" not in out and out["other"] == 1
        with pytest.raises(ValueError, match="mutually exclusive"):
            entry_utils.apply_deploy_profile("m", {"deploy_profile": "edge", "deploy_config": "x.yaml"})
    # No profile: untouched.
    assert entry_utils.apply_deploy_profile("m", {"deploy_config": "x.yaml"}) == {"deploy_config": "x.yaml"}


def test_modify_stage_config_keeps_relative_base_config_resolvable():
    from tests.helpers.stage_config import modify_stage_config

    tmp_yaml = modify_stage_config(str(EDGE_QWEN3_TTS), updates={"stages": {0: {"max_num_seqs": 1}}})
    raw = resolve_deploy_yaml(Path(tmp_yaml))
    assert len(raw["stages"]) == 2
    cfg = load_deploy_config(Path(tmp_yaml))
    assert cfg.stages[0].max_num_seqs == 1
    assert cfg.stages[1].max_num_seqs == 4


@pytest.mark.core_model
@pytest.mark.cpu
def test_edge_profile_carries_parallel_stage_init():
    from vllm_omni.entrypoints.utils import apply_deploy_profile, profile_orchestrator_defaults

    args = apply_deploy_profile("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", {"deploy_profile": "edge"})
    assert args["parallel_stage_init"] is True
    assert profile_orchestrator_defaults(args["deploy_config"]) == {"parallel_stage_init": True}
    # explicit values win over the profile default
    args = apply_deploy_profile(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", {"deploy_profile": "edge", "parallel_stage_init": True}
    )
    assert args["parallel_stage_init"] is True
    # unknown block keys are ignored
    assert "init_timeout" not in args


@pytest.mark.core_model
@pytest.mark.cpu
def test_orchestrator_block_applies_however_the_config_was_loaded(tmp_path):
    """A deploy YAML must mean the same thing whichever flag loaded it.

    Applying the `orchestrator:` block only on the profile path made
    `--deploy-config deploy/edge/qwen3_tts.yaml` load the stages but drop
    `parallel_stage_init: true`, so an edge deployment started sequentially
    (63.8 s against 30.5 s on Qwen3-TTS) with nothing in the log to say why.
    """
    from vllm_omni.entrypoints import utils as entry_utils

    path = tmp_path / "profile.yaml"
    path.write_text(
        "pipeline: qwen3_tts\n"
        "orchestrator:\n"
        "  parallel_stage_init: true\n"
        "  stage_init_timeout: 900\n",
        encoding="utf-8",
    )
    out = entry_utils.apply_deploy_profile("m", {"deploy_config": str(path)})
    assert out["parallel_stage_init"] is True
    assert out["stage_init_timeout"] == 900


@pytest.mark.core_model
@pytest.mark.cpu
def test_explicit_arguments_still_beat_the_orchestrator_block(tmp_path):
    from vllm_omni.entrypoints import utils as entry_utils

    path = tmp_path / "profile.yaml"
    path.write_text("orchestrator:\n  stage_init_timeout: 900\n", encoding="utf-8")
    out = entry_utils.apply_deploy_profile(
        "m", {"deploy_config": str(path), "stage_init_timeout": 120})
    assert out["stage_init_timeout"] == 120


@pytest.mark.core_model
@pytest.mark.cpu
def test_unknown_orchestrator_keys_are_ignored(tmp_path):
    """The block is a narrow allow-list, not a way to inject arbitrary args."""
    from vllm_omni.entrypoints import utils as entry_utils

    path = tmp_path / "profile.yaml"
    path.write_text(
        "orchestrator:\n  parallel_stage_init: true\n  rm_rf_slash: true\n",
        encoding="utf-8")
    out = entry_utils.apply_deploy_profile("m", {"deploy_config": str(path)})
    assert out["parallel_stage_init"] is True
    assert "rm_rf_slash" not in out
