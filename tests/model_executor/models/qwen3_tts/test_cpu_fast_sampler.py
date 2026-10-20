"""Qwen3-TTS talker model-side CPU sampler: same contract as vLLM's Sampler, no compiled kernel (CPU)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

V, B = 3072, 2


def _metadata(temperature=0.9, top_k=50, rep=1.05, generators=None, max_num_logprobs=None):
    fields = {f.name: f for f in dataclasses.fields(SamplingMetadata)}
    kw = {}
    temp = torch.full((B,), float(temperature))
    for name, f in fields.items():
        if name == "temperature":
            kw[name] = temp
        elif name == "all_greedy":
            kw[name] = temperature == 0
        elif name == "all_random":
            kw[name] = temperature > 0
        elif name == "top_p":
            kw[name] = None
        elif name == "top_k":
            kw[name] = None if top_k <= 0 else torch.full((B,), top_k, dtype=torch.int64)
        elif name == "generators":
            kw[name] = generators or {}
        elif name == "max_num_logprobs":
            kw[name] = max_num_logprobs
        elif name == "no_penalties":
            kw[name] = rep == 1.0
        elif name == "prompt_token_ids":
            kw[name] = None if rep == 1.0 else torch.randint(0, 2048, (B, 8))
        elif name in ("frequency_penalties", "presence_penalties"):
            kw[name] = torch.zeros(B)
        elif name == "repetition_penalties":
            kw[name] = torch.full((B,), float(rep))
        elif name == "output_token_ids":
            kw[name] = [[1, 2, 3], [4, 5]]
        elif name == "allowed_token_ids_mask":
            kw[name] = None
        elif name == "bad_words_token_ids":
            kw[name] = {}
        elif name == "logitsprocs":
            from vllm.v1.sample.logits_processor import LogitsProcessors

            kw[name] = LogitsProcessors()
        elif name == "logprob_token_ids":
            kw[name] = None
        elif f.default is not dataclasses.MISSING:
            kw[name] = f.default
        elif getattr(f, "default_factory", dataclasses.MISSING) is not dataclasses.MISSING:
            kw[name] = f.default_factory()
        else:
            kw[name] = None
    return SamplingMetadata(**kw)


def _sample(logits, md):
    fake = SimpleNamespace()
    return Qwen3TTSTalkerForConditionalGeneration.sample(fake, logits, md)


def test_greedy_matches_argmax():
    logits = torch.randn(B, V)
    out = _sample(logits.clone(), _metadata(temperature=0.0, top_k=0, rep=1.0))
    assert out.sampled_token_ids.shape == (B, 1) and out.logprobs_tensors is None
    assert torch.equal(out.sampled_token_ids.view(-1), logits.argmax(-1))


def test_seeded_generators_are_deterministic_and_top_k_respected():
    logits = torch.randn(B, V)
    outs = []
    for _ in range(2):
        gens = {i: torch.Generator().manual_seed(1234 + i) for i in range(B)}
        outs.append(_sample(logits.clone(), _metadata(top_k=5, rep=1.0, generators=gens)).sampled_token_ids)
    assert torch.equal(outs[0], outs[1])
    top5 = logits.topk(5, dim=-1).indices
    for i in range(B):
        assert int(outs[0][i, 0]) in top5[i].tolist()


def test_falls_back_when_logprobs_requested_or_not_cpu():
    logits = torch.randn(B, V)
    assert _sample(logits, _metadata(max_num_logprobs=1)) is None
    assert _sample(None, _metadata()) is None
