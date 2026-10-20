"""Tests for benchmarks/tts/init_profile.py helpers (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "tts"))
import init_profile as ip  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_flatten_edge_yaml_inherits_base_and_applies_updates(tmp_path):
    edge = ip.DEPLOY_DIR / "edge" / "qwen3_tts.yaml"
    flat = ip._flatten_deploy_yaml(edge)
    assert "base_config" not in flat
    assert [s["stage_id"] for s in flat["stages"]] == [0, 1]
    # inherited from base
    assert flat["stages"][0]["trust_remote_code"] is True
    # overridden by edge
    assert flat["stages"][0]["max_num_seqs"] == 4
    assert flat["connectors"]["connector_of_shared_memory"]["extra"]["codec_chunk_ramp"] == [2, 4, 8, 16, 25]

    out = ip.write_variant_yaml(
        edge, {"stages": {1: {"enforce_eager": True}}, "top": {"async_chunk": False}}, tmp_path, "v"
    )
    doc = yaml.safe_load(out.read_text())
    assert doc["stages"][1]["enforce_eager"] is True
    assert doc["stages"][0].get("enforce_eager") is None
    assert doc["async_chunk"] is False


def test_phase_durations_and_table():
    timeline = {
        "total_s": 100.0,
        "phases": [
            {"name": "load_weights", "dur_s": 10.0, "stage_id": 0},
            {"name": "load_weights", "dur_s": 5.0, "stage_id": 1},
            {"name": "predictor_setup_compile", "dur_s": 40.0, "stage_id": 0},
        ],
    }
    d = ip.phase_durations(timeline)
    assert d == {"load_weights": 15.0, "predictor_setup_compile": 40.0}
    by_stage = ip.phase_durations_by_stage(timeline)
    assert by_stage["0"]["load_weights"] == 10.0 and by_stage["1"]["load_weights"] == 5.0
    table = ip.render_table([{"config": "default", "init_s": 100.0, "ttfa_ms": 30.0, "gpu_mib": 5000, "phases": d}])
    assert "| default | 100.0 | 30.0 | 5000 |" in table
    assert "predictor_setup_compile" in table.splitlines()[0]
    assert ip.phase_durations(None) == {}


def test_config_registry_names():
    for name in ("default", "cold_cache", "eager_stage1", "eager_both", "parallel_stage_init", "edge"):
        assert name in ip.CONFIGS
    assert ip.CONFIGS["cold_cache"].fresh_caches is True
    assert ip.CONFIGS["parallel_stage_init"].extra_args == ["--parallel-stage-init"]
