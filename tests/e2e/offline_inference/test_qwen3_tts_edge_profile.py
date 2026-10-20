# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E: Qwen3-TTS CustomVoice under the ``edge`` deploy profile.

The edge profile (``vllm_omni/deploy/edge/qwen3_tts.yaml``) runs both stages on
one device with small batches, a ramped codec chunk schedule and concurrent
stage init. This test drives the same profile YAML the ``deploy_profile="edge"``
selector resolves to (the runner fixture always passes ``deploy_config``) and
checks that audio is produced end to end. The chunk schedule itself is
measured by ``benchmarks/tts/stream_latency_bench.py`` (stall_at_ttfa == 0).
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.entrypoints.utils import profile_orchestrator_defaults, resolve_deploy_profile_path

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
EDGE_YAML = get_deploy_config_path("edge/qwen3_tts.yaml")

tts_edge_params = [
    pytest.param(
        (MODEL, EDGE_YAML, {"parallel_stage_init": True, "init_timeout": 1500, "stage_init_timeout": 900}),
        id="edge_profile",
    )
]


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_edge_params, indirect=True)
def test_edge_profile_text_to_audio(omni_runner, offline_client) -> None:
    # The selector resolves the same file the runner was started with and
    # carries the profile's orchestrator defaults.
    assert os.path.realpath(resolve_deploy_profile_path(MODEL, "edge")) == os.path.realpath(EDGE_YAML)
    assert profile_orchestrator_defaults(EDGE_YAML).get("parallel_stage_init") is True
    offline_client.send_audio_speech_request({"input": "Edge profile end to end check.", "voice": "vivian"})
