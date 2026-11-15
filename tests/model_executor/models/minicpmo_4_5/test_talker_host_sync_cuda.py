# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MiniCPM-o 4.5 Talker's per-step path must not block the host on the GPU."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.minicpmo_4_5.test_talker_host_sync import _EOS, _infos, _make_talker, _states, _step
from vllm_omni.utils.device_copy import index_to_device

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)]


def _assert_device_states_match_scalar(talker, expected_states) -> None:
    # Read back only after the sync-debug scope: the host dictionary deliberately
    # retains its initialization values once a request owns a device slot.
    codec = talker._device_codec_state
    assert talker._request_audio_states.keys() == expected_states.keys()
    for rid, expected in expected_states.items():
        actual = talker._request_audio_states[rid]
        if "_gpu_slot" not in actual:
            assert actual == expected
            continue
        slot = actual["_gpu_slot"]
        assert codec.state[slot].tolist() == [
            expected["step"],
            int(expected["finished"]),
            expected["max_tokens"] - 1,
            expected.get("min_tokens", 0),
            int(expected.get("turn_end_drain", False)),
        ]
        history = ([talker._num_audio_tokens] * codec.window + expected.get("recent_codes", []))[-codec.window :]
        assert codec.history[slot].tolist() == history
        assert actual["recent_codes"].tolist() == history


def test_talker_step_does_not_synchronize_cuda(mocker) -> None:
    talker = _make_talker("cuda")
    states = _states()
    talker._request_audio_states = copy.deepcopy(states)
    infos = _infos(states)
    input_ids = torch.tensor([3, _EOS, _EOS, 6], dtype=torch.int32, device="cuda")
    hidden = torch.randn(4, 4, device="cuda")
    penalties = torch.tensor([1.05, 1.2, 1.05, 1.0], device="cuda")
    # Warm the pinned host allocator and CUDA context outside the check.
    index_to_device([1], "cuda")
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.inference_mode():
            *_, sampled = _step(talker, infos, hidden, mocker, batched=True, input_ids=input_ids, penalties=penalties)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert sampled.reshape(-1).tolist()[1:] == [_EOS, _EOS, _EOS]
    expected = copy.deepcopy(states)
    expected["req-live"].update(step=3, recent_codes=[1, 2, 3])
    expected["req-eos"]["finished"] = True
    expected["req-cap"].update(step=9, finished=True, recent_codes=(list(range(7)) * 3 + [6])[-16:])
    _assert_device_states_match_scalar(talker, expected)


def test_talker_condition_upload_does_not_synchronize_cuda() -> None:
    talker = _make_talker("cuda")
    talker.emb_text = nn.Embedding(16, 4).cuda()
    talker.projector_semantic = nn.Linear(6, 4).cuda()
    talker._normalize = True
    talker._text_eos_id = 14
    talker._tts_bos_id = 15
    token_ids = torch.tensor([2, 3, 4])
    hidden_states = torch.randn(3, 6)
    expected = talker._build_condition_embeddings(token_ids.cuda(), hidden_states.cuda())
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.inference_mode():
            condition = talker._build_condition_embeddings(token_ids, hidden_states)
            boundary = talker._build_condition_embeddings(token_ids[:0], hidden_states[:0])
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert torch.equal(condition, expected)
    assert boundary.shape == (2, 4)
