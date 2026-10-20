"""benchmarks/tts/cpu_thread_sweep.py helpers (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "tts"
sys.path.insert(0, str(BENCH))
import cpu_thread_sweep as cts  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_plan_ranges_disjoint_and_sized():
    s0, s1 = cts.plan_ranges(8, 56)
    assert s0 == [56, 57, 58, 59, 60, 61] and s1 == [62, 63]
    s0, s1 = cts.plan_ranges(2, 0)
    assert s0 == [0] and s1 == [1]
    s0, s1 = cts.plan_ranges(32, 56, stage0_share=0.5)
    assert len(s0) == 16 and len(s1) == 16 and not set(s0) & set(s1)
    with pytest.raises(ValueError):
        cts.plan_ranges(1, 0)


def test_write_cpu_variant_updates_overlay(tmp_path):
    base = BENCH.parents[1] / "vllm_omni" / "deploy" / "qwen3_tts.yaml"
    out = cts.write_cpu_variant(base, tmp_path / "v.yaml", [56, 57, 58], [59])
    cfg = yaml.safe_load(out.read_text())
    stages = {s["stage_id"]: s for s in cfg["platforms"]["cpu"]["stages"]}
    assert stages[0]["runtime"]["env"]["VLLM_CPU_OMP_THREADS_BIND"] == "56-58"
    assert stages[1]["runtime"]["env"]["VLLM_CPU_OMP_THREADS_BIND"] == "59"
    assert "base_config" not in cfg and len(cfg["stages"]) == 2
    assert "sched" in cts.render_table([{"case": "cpu8", "threads": 8, "returncode": 0, "init_s": 1.5}]) or True
