"""Tests for benchmarks/tts/hw_emulation.py case construction (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "tts"))
import hw_emulation as hw  # noqa: E402

from vllm_omni.edge.hardware_probe import hardware_class  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cases_map_to_expected_classes():
    expected = {
        "x86_cpu_avx2": "x86_cpu",
        "x86_cpu_avx512": "x86_cpu",
        "x86_cpu_amx": "x86_cpu",
        "arm64_cpu_logic": "arm64_cpu",
        "jetson_orin_8g": "jetson",
        "jetson_orin_32g": "jetson",
        "cuda_discrete_8g": "cuda_discrete",
    }
    for name, cls in expected.items():
        prof = hw.build_profile(hw.CASES[name])
        assert hardware_class(prof) == cls, name
        assert prof.source == "emulation"


def test_avx2_case_disables_bf16_and_pins_cores():
    case = hw.CASES["x86_cpu_avx2"]
    prof = hw.build_profile(case)
    assert prof.supports_bf16 is False and prof.n_logical == 4
    assert case.env["ATEN_CPU_CAPABILITY"] == "avx2" and case.cpus == [0, 1, 2, 3]


def test_table_render_handles_missing_metrics():
    table = hw.render_table([{"case": "x", "returncode": -1, "note": "timeout"}])
    assert "| x | -1 |" in table and "timeout" in table
