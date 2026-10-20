# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni-only forward kwargs must not be forced on a model that cannot take them.

The AR runner hands every stage ``sampling_metadata``, ``logits_index`` and
``sampler``, which omni-native models absorb through ``**kwargs``. A model vLLM
owns does not: ``LlamaForCausalLM.forward`` is
``(input_ids, positions, intermediate_tensors, inputs_embeds)`` and nothing
else, so passing the extras is a ``TypeError`` on the very first forward.
"""

from __future__ import annotations

import torch

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


class _OmniNativeModel(torch.nn.Module):
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return input_ids


class _VLLMNativeModel(torch.nn.Module):
    """The shape of vLLM's own causal LMs: no ``**kwargs``."""

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return input_ids


class _Probe:
    """Just enough of a runner to exercise the helper."""

    def __init__(self, model):
        self.model = model

    _accepted_forward_kwargs = OmniGPUModelRunner._accepted_forward_kwargs


def test_a_model_taking_kwargs_is_left_alone():
    assert _Probe(_OmniNativeModel())._accepted_forward_kwargs() is None


def test_a_vllm_native_model_reports_exactly_its_parameters():
    accepted = _Probe(_VLLMNativeModel())._accepted_forward_kwargs()
    assert accepted == frozenset(
        {"input_ids", "positions", "intermediate_tensors", "inputs_embeds"}
    )
    for omni_only in ("sampling_metadata", "logits_index", "sampler"):
        assert omni_only not in accepted


def test_the_answer_is_cached():
    """``inspect.signature`` on every decode step would be absurd."""
    probe = _Probe(_VLLMNativeModel())
    first = probe._accepted_forward_kwargs()
    probe.model = _OmniNativeModel()  # would change the answer if recomputed
    assert probe._accepted_forward_kwargs() is first


def test_an_uninspectable_forward_falls_back_to_passing_everything():
    """Degrade to the previous behaviour rather than silently dropping
    arguments a model may actually need."""

    class Weird(torch.nn.Module):
        forward = staticmethod(print)  # no usable signature for our purposes

    accepted = _Probe(Weird())._accepted_forward_kwargs()
    assert accepted is None or "sampling_metadata" not in accepted
