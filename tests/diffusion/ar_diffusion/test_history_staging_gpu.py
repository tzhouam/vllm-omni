# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA regressions for contiguous AR-Diffusion history staging."""

import pytest
import torch

from tests.diffusion.ar_diffusion.test_paged_attention import (
    BLOCK,
    HEAD_DIM,
    N_HEADS,
    POS,
    _commit_video_span,
    _gpu_flash_attn_usable,
    make_state,
)
from tests.helpers.mark import hardware_test
from vllm_omni.experimental.ar_diffusion.kv_cache import paged_write_attn
from vllm_omni.experimental.ar_diffusion.kv_cache.config import KV_GATHER_ENV

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.skipif(
    torch.version.hip is not None or not _gpu_flash_attn_usable(), reason="usable CUDA FlashAttention is required"
)
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("window_chunks", [2, 4])
@torch.inference_mode()
def test_staged_attention_updates_only_the_active_window_gpu(monkeypatch, window_chunks):
    """Full and reused gathers match fresh attention without touching spare capacity or stale history."""
    monkeypatch.setenv(KV_GATHER_ENV, "1")
    device, dtype = torch.device("cuda"), torch.bfloat16
    kv, st = make_state(device=device, dtype=dtype, window_chunks=window_chunks, reuse_history_staging=True)
    _commit_video_span(kv, st, kv_branch=POS, n_chunks=1, dtype=dtype, device=device)
    stage_key, stage_value = kv.history_staging[0]
    stage_key.fill_(-7)
    stage_value.fill_(-7)
    query = torch.randn(BLOCK, N_HEADS, HEAD_DIM, device=device, dtype=dtype)
    for step in range(2):
        ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0].forward_ctx
        ctx.max_video_tokens = 2 * BLOCK
        ctx.prepare(device, action_len=0, query_len=BLOCK)
        inputs = ctx.layer_inputs(0)
        assert inputs.reuse_history is (step > 0)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        actual = paged_write_attn(inputs, query, key, value, None, None, HEAD_DIM**-0.5)
        expected = paged_write_attn(
            inputs._replace(stage_key=None, stage_value=None, reuse_history=False),
            query,
            key,
            value,
            None,
            None,
            HEAD_DIM**-0.5,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        ids = inputs.block_table[0].long()
        for staged, cache in ((stage_key, kv.key_cache(0)), (stage_value, kv.value_cache(0))):
            torch.testing.assert_close(
                staged[: inputs.max_seq_len], cache.index_select(0, ids).flatten(0, 1), rtol=0, atol=0
            )
            assert (staged[inputs.max_seq_len :] == -7).all()
