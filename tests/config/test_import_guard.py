"""Import guard for AR/TTS-only deployments: heavy diffusion dependencies stay unloaded (CPU).

Runs in a subprocess so the assertion is not polluted by modules other tests
imported into this interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PROBE = r"""
import json, sys, time
t = time.perf_counter(); import vllm; t_vllm = time.perf_counter() - t
t = time.perf_counter(); import vllm_omni; t_omni = time.perf_counter() - t
from vllm_omni.config.config_factory import StageConfigFactory  # noqa: F401
from vllm_omni.entrypoints.async_omni import AsyncOmni  # noqa: F401
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
assert "qwen3_tts" in OMNI_PIPELINES
heavy = ["diffusers", "x_transformers", "whisper", "vllm_omni.diffusion.registry", "vllm_omni.diffusion.io_support"]
print(json.dumps({"t_vllm": t_vllm, "t_omni": t_omni, "loaded": [m for m in heavy if m in sys.modules]}))
"""


def test_tts_path_does_not_import_diffusion_stack():
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    info = json.loads(out.stdout.strip().splitlines()[-1])
    assert info["loaded"] == [], f"heavy modules imported on the AR/TTS path: {info['loaded']}"
    # Import overhead of vllm_omni over bare vllm is reported (not gated: it is host dependent).
    print(f"import vllm {info['t_vllm']:.1f}s, +vllm_omni {info['t_omni']:.1f}s")
