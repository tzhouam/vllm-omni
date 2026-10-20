"""KV-cached incremental residual decoding of the Qwen3 code predictor equals the re-prefill forward (CPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorBaseModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny_config():
    return SimpleNamespace(
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=96,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_code_groups=16,
        rope_theta=10000.0,
        attention_bias=False,
    )


@torch.inference_mode()
def test_forward_cached_matches_full_prefix():
    torch.manual_seed(0)
    model = CodePredictorBaseModel(_tiny_config()).eval()
    for p in model.parameters():
        p.normal_(0, 0.2)
    bsz, max_seq, hidden = 2, 17, 64
    x = torch.randn(bsz, max_seq, hidden)
    pos = torch.arange(max_seq).unsqueeze(0).expand(bsz, -1)
    full = model(x, pos)  # [B, 17, H], causal over the whole prefix

    caches = model.new_kv_caches(bsz, max_seq, x.device, x.dtype)
    out_prefill = model.forward_cached(x[:, :2], pos[:, :2], caches, 0)
    assert torch.allclose(out_prefill, full[:, :2], atol=1e-4, rtol=1e-4)
    for step in range(2, max_seq):
        out_step = model.forward_cached(x[:, step : step + 1], pos[:, step : step + 1], caches, step)
        assert torch.allclose(out_step[:, 0], full[:, step], atol=1e-4, rtol=1e-4), step
    # caches hold exactly the prefix K/V that a fresh full pass would produce
    assert caches[0][0].shape == (bsz, 2, max_seq, 16)


@torch.inference_mode()
def test_forward_cached_rejects_multi_token_append():
    model = CodePredictorBaseModel(_tiny_config()).eval()
    x = torch.randn(1, 2, 64)
    pos = torch.arange(2).unsqueeze(0)
    caches = model.new_kv_caches(1, 17, x.device, x.dtype)
    model.forward_cached(x, pos, caches, 0)
    with pytest.raises(ValueError):
        model.forward_cached(x, pos + 2, caches, 2)


@torch.inference_mode()
def test_forward_static_step_matches_full_prefix():
    torch.manual_seed(1)
    model = CodePredictorBaseModel(_tiny_config()).eval()
    for p in model.parameters():
        p.normal_(0, 0.2)
    bsz, max_seq = 2, 17
    x = torch.randn(bsz, max_seq, 64)
    pos = torch.arange(max_seq).unsqueeze(0).expand(bsz, -1)
    full = model(x, pos)
    caches = model.new_kv_caches(bsz, max_seq, x.device, x.dtype)
    model.forward_cached(x[:, :2], pos[:, :2], caches, 0)  # eager 2-token prefill
    for step in range(2, max_seq):
        pos_t = torch.tensor([step])
        mask = (torch.arange(max_seq) <= step).view(1, 1, 1, max_seq).expand(bsz, 1, 1, max_seq)
        out = model.forward_static_step(x[:, step : step + 1], pos[:, step : step + 1], caches, pos_t, mask)
        assert torch.allclose(out[:, 0], full[:, step], atol=1e-4, rtol=1e-4), step


@torch.inference_mode()
def test_compiled_static_step_matches_eager():
    torch.manual_seed(2)
    model = CodePredictorBaseModel(_tiny_config()).eval()
    for p in model.parameters():
        p.normal_(0, 0.2)
    bsz, max_seq = 1, 17
    x = torch.randn(bsz, max_seq, 64)
    pos = torch.arange(max_seq).unsqueeze(0).expand(bsz, -1)
    full = model(x, pos)
    step_fn = torch.compile(model.forward_static_step, dynamic=False)
    caches = model.new_kv_caches(bsz, max_seq, x.device, x.dtype)
    model.forward_cached(x[:, :2], pos[:, :2], caches, 0)
    for step in range(2, max_seq):
        pos_t = torch.tensor([step])
        mask = (torch.arange(max_seq) <= step).view(1, 1, 1, max_seq).expand(bsz, 1, 1, max_seq)
        out = step_fn(x[:, step : step + 1], pos[:, step : step + 1], caches, pos_t, mask)
        assert torch.allclose(out[:, 0], full[:, step], atol=1e-4, rtol=1e-4), step


@torch.inference_mode()
def test_post_step_sample_matches_eager_gumbel():
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import _post_step_sample

    torch.manual_seed(3)
    bsz, hidden, vocab = 2, 64, 32
    h = torch.randn(bsz, hidden)
    head_w = torch.randn(vocab, hidden)
    embed_w = torch.randn(vocab, hidden)
    proj_w = torch.randn(hidden, hidden)
    proj_b = torch.randn(hidden)
    u = torch.rand(bsz, vocab).clamp_(1e-20, 1 - 1e-20)
    # eager reference (the loop body of the predictor): top-k mask + Gumbel-max
    logits = h @ head_w.T
    scaled = logits * (1 / 0.9)
    topk_vals, _ = scaled.topk(5, dim=-1)
    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
    ref_code = (scaled.float() - torch.log(-torch.log(u))).argmax(-1, keepdim=True)
    ref_proj = torch.nn.functional.linear(embed_w[ref_code.view(-1)], proj_w, proj_b)
    code, proj_row = _post_step_sample(h, head_w, embed_w, proj_w, proj_b, u, 1 / 0.9, 5, True)
    assert torch.equal(code, ref_code) and torch.allclose(proj_row, ref_proj, atol=1e-5)
    code_g, _ = _post_step_sample(h, head_w, embed_w, proj_w, proj_b, u, 0.0, 0, False)
    assert torch.equal(code_g.view(-1), logits.argmax(-1))
    compiled = torch.compile(_post_step_sample, dynamic=False)
    code_c, proj_c = compiled(h, head_w, embed_w, proj_w, proj_b, u, 1 / 0.9, 5, True)
    assert torch.equal(code_c, ref_code) and torch.allclose(proj_c, ref_proj, atol=1e-4)
