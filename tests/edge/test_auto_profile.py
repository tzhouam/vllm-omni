"""Tests for deploy_profile='auto' materialization (CPU, no engine)."""

from __future__ import annotations

from unittest import mock

import pytest
import yaml

from vllm_omni.config.stage_config import _DEPLOY_DIR, load_deploy_config
from vllm_omni.edge import auto_profile, calibrate
from vllm_omni.edge.hardware_probe import HardwareProfile
from vllm_omni.entrypoints import utils as entry_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GiB = 2**30
EDGE = _DEPLOY_DIR / "edge" / "qwen3_tts.yaml"


def _arm_profile():
    return HardwareProfile(
        arch="aarch64",
        cpu_model="RK3588",
        cpu_flags=["neon", "dotprod", "fp16_arith"],
        n_logical=8,
        n_physical=8,
        clusters=[{"max_khz": 2400000, "cpus": [4, 5, 6, 7]}, {"max_khz": 1800000, "cpus": [0, 1, 2, 3]}],
        ram_total_bytes=16 * GiB,
        ram_available_bytes=12 * GiB,
        supports_bf16=False,
    )


def test_materialize_arm64_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))
    path, report = auto_profile.materialize_auto_deploy(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", EDGE, profile=_arm_profile(), out_dir=tmp_path
    )
    assert report["hardware_class"] == "arm64_cpu"
    doc = yaml.safe_load(open(path))
    assert "base_config" not in doc
    s0, s1 = doc["stages"]
    # class overlay pins eager + max_num_seqs 1; derivation adds cpu binding on the big cluster only
    assert s0["enforce_eager"] is True and s1["enforce_eager"] is True
    assert s0["max_num_seqs"] == 1
    assert s0["devices"] == "cpu" and s1["devices"] == "cpu"
    assert s0["env"]["VLLM_CPU_OMP_THREADS_BIND"] == "4-6"
    assert s1["env"]["VLLM_CPU_OMP_THREADS_BIND"] == "7"
    assert s0["engine_args"]["dtype"] == "float32"
    assert s0["engine_args"]["kv_cache_memory_bytes"] > 0
    extra = doc["connectors"]["connector_of_shared_memory"]["extra"]
    assert extra["codec_chunk_adaptive"] is True and extra["codec_chunk_ramp"] == [2, 4, 8, 16, 25]
    # The result is a valid deploy config.
    cfg = load_deploy_config(path)
    assert len(cfg.stages) == 2 and cfg.stages[0].engine_extras["dtype"] == "float32"


def test_calibration_cache_is_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))
    prof = _arm_profile()
    fp = calibrate.hardware_fingerprint(prof, "m")
    calibrate.save_calibration(
        fp,
        {
            "codec_chunk_ramp": [3, 6, 12, 25],
            "codec_chunk_min_frames": 3,
            "codec_chunk_safety_margin_ms": 90.0,
            "codec_chunk_adaptive": True,
        },
    )
    path, report = auto_profile.materialize_auto_deploy("m", EDGE, profile=prof, out_dir=tmp_path)
    doc = yaml.safe_load(open(path))
    extra = doc["connectors"]["connector_of_shared_memory"]["extra"]
    assert extra["codec_chunk_ramp"] == [3, 6, 12, 25] and extra["codec_chunk_min_frames"] == 3
    assert report["calibration"]["codec_chunk_ramp"] == [3, 6, 12, 25]


def test_apply_deploy_profile_auto_uses_edge_base(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_HW_PROFILE", str(tmp_path / "hw.json"))
    (tmp_path / "hw.json").write_text(_arm_profile().to_json())
    with mock.patch.object(entry_utils, "resolve_model_config_path", return_value=str(_DEPLOY_DIR / "qwen3_tts.yaml")):
        args = entry_utils.apply_deploy_profile("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", {"deploy_profile": "auto"})
    assert args["deploy_config"].endswith(".yaml")
    doc = yaml.safe_load(open(args["deploy_config"]))
    assert doc["stages"][0]["devices"] == "cpu"
    assert doc["stages"][0]["max_num_batched_tokens"] == 512  # inherited from edge/qwen3_tts.yaml


def test_hardware_overlay_merge_semantics():
    doc = {
        "stages": [{"stage_id": 0, "a": 1, "d": {"x": 1}}, {"stage_id": 1}],
        "connectors": {"c": {"extra": {"k": 1}}},
    }
    out = auto_profile.apply_derived_overrides(
        doc,
        {
            "stages": {0: {"a": 2, "d": {"y": 2}, "gpu_memory_utilization": None}},
            "top": {"async_chunk": True},
            "connector_extra": {"z": 3},
        },
    )
    assert out["stages"][0] == {"stage_id": 0, "a": 2, "d": {"x": 1, "y": 2}}
    assert out["async_chunk"] is True and out["connectors"]["c"]["extra"] == {"k": 1, "z": 3}
    assert doc["stages"][0]["a"] == 1  # input untouched
