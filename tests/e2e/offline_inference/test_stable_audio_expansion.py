# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable Audio offline e2e: real weights, FP8 + TeaCache (single job to save GPU).

NOTE: This test instantiates Omni directly instead of using the omni_runner
fixture (introduced in PR #2711) because the fixture's parametrize interface
only accepts (model, stage_config_path) and does not support extra kwargs like
quantization, cache_backend, or cache_config.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import assert_audio_valid
from tests.helpers.mark import hardware_test
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

_SAMPLE_RATE = 44100
_CLIP_DURATION_S = 2.0
_MODEL_REPO = "stabilityai/stable-audio-open-1.0"


def _skip_if_gated_repo_inaccessible(repo_id: str) -> None:
    """Skip the test when the CI HF_TOKEN cannot read ``repo_id``.

    The Stable Audio checkpoint is a gated HuggingFace repository (see
    ``docs/contributing/ci/hf_credentials.md``). When the CI token is
    missing, expired, or hasn't accepted the license, ``from_pretrained``
    blows up deep inside the diffusion worker with a bare
    ``GatedRepoError`` that is hard to diagnose and masks real regressions.
    Detect that here and skip instead, so the CI signal only fires on
    actual code issues rather than credential drift.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import (
            GatedRepoError,
            RepositoryNotFoundError,
        )
    except Exception:
        return

    try:
        HfApi().model_info(repo_id)
    except GatedRepoError as exc:
        pytest.skip(
            f"Skipping: gated HF repo {repo_id!r} inaccessible to the current "
            f"HF_TOKEN ({exc}). See docs/contributing/ci/hf_credentials.md."
        )
    except RepositoryNotFoundError as exc:
        pytest.skip(
            f"Skipping: HF repo {repo_id!r} not visible to the current "
            f"HF_TOKEN ({exc}). See docs/contributing/ci/hf_credentials.md."
        )
    except Exception:
        return


def generate_stable_audio_short_clip(
    omni: Omni,
    *,
    audio_start_in_s: float = 0.0,
    audio_end_in_s: float = 2.0,
    num_inference_steps: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Run a minimal Stable Audio generation and return audio as (batch, channels, samples)."""
    outputs = omni.generate(
        prompts={
            "prompt": "The sound of a dog barking",
            "negative_prompt": "Low quality.",
        },
        sampling_params_list=OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            guidance_scale=7.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
            num_outputs_per_prompt=1,
            extra_args={
                "audio_start_in_s": audio_start_in_s,
                "audio_end_in_s": audio_end_in_s,
            },
        ),
    )

    assert outputs is not None
    first_output = outputs[0]
    # Outer OmniRequestOutput.final_output_type comes from get_stage_metadata.
    # The nested request_output is the worker OmniRequestOutput
    # (e.g. final_output_type="audio") and holds the multimodal payload.
    # Follow-up: add StableAudioPipeline stage YAML, and pass model into
    # _create_default_diffusion_stage_cfg so default diffusion metadata can set
    # final_output_type to "audio" for future audio pipelines without YAML.
    assert first_output.final_output_type == "image"
    assert hasattr(first_output, "request_output") and first_output.request_output

    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput)
    assert req_out.final_output_type == "audio"
    assert hasattr(req_out, "multimodal_output") and req_out.multimodal_output
    audio = req_out.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    return audio


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "xpu": "B60"})
def test_stable_audio_quantization_and_teacache() -> None:
    """Stable Audio Open on real Hub weights with FP8 + TeaCache (covers former L2 smoke + L4 features).

    CI should provide ``HF_TOKEN`` if the checkpoint is gated.
    """
    _skip_if_gated_repo_inaccessible(_MODEL_REPO)
    m = Omni(
        model=_MODEL_REPO,
        quantization="fp8",
        cache_backend="tea_cache",
        cache_config={"rel_l1_thresh": 0.2},
    )
    try:
        audio = generate_stable_audio_short_clip(m)
        assert_audio_valid(
            audio,
            sample_rate=_SAMPLE_RATE,
            channels=2,
            duration_s=_CLIP_DURATION_S,
        )
    finally:
        m.close()
