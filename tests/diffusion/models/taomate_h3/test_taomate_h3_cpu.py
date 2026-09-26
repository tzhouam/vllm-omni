# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the TaoMate-H3 streaming port."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm_omni.diffusion.models.taomate_h3 import geometry as geo
from vllm_omni.diffusion.models.taomate_h3.kv_cache import (
    AUDIO_TOKEN_TAG,
    TEXT_TOKEN_TAG,
    VIDEO_TOKEN_TAG,
    CleanAVKVCache,
    KVContract,
)
from vllm_omni.diffusion.models.taomate_h3.lora import TaoMateLoRAAdapter, is_taomate_lora_dir, taomate_lora_targets
from vllm_omni.diffusion.models.taomate_h3.packed import (
    prompt_rope_start,
    taomate_audio_only_frozen_prefix_packed_layout,
    taomate_audio_only_packed_layout,
    taomate_phase_packed_layout,
)
from vllm_omni.diffusion.models.taomate_h3.schedule import (
    euler_eta0_update_,
    student_sigmas,
    teacher_sigmas,
    time_shift_sigmas,
)

# ----------------------------------------------------------------------------
# geometry


def test_direct_plan_matches_the_official_request_geometry() -> None:
    plan = geo.direct_5s_plan()
    assert plan.native_frame_count == 124
    assert [p.frame_count for p in plan.phases] == [39, 34, 34, 17]
    assert [p.video_latent_count for p in plan.phases] == [12, 10, 10, 5]
    assert plan.video_latent_count == 37
    assert plan.audio_latent_count == 207
    assert [p.audio_latent_start for p in plan.phases] == [0, 65, 122, 178]


def test_continuation_plan_has_steady_geometry_and_rounded_audio() -> None:
    first = geo.request_plan(1)
    assert first.native_frame_count == 119
    assert [p.video_latent_count for p in first.phases] == [10, 10, 10, 5]
    assert first.audio_latent_count in (198, 199)
    # Audio boundaries are rounded on the global timeline: request k starts at
    # frame 124 + 119 (k - 1).
    for k in (1, 2, 5, 12):
        plan = geo.request_plan(k)
        start_frame = 124 + 119 * (k - 1)
        expected = geo.audio_latent_boundary(start_frame + 119) - geo.audio_latent_boundary(start_frame)
        assert plan.audio_latent_count == expected
        assert plan.phases[0].audio_latent_start == 0


def test_video_temporal_positions_follow_the_release_spacing() -> None:
    assert geo.video_temporal_position(0) == 0
    assert geo.video_temporal_position(1) == Fraction(5, 3)
    assert geo.video_temporal_position(2) == Fraction(5, 3) * 5
    # Five latents span 17 frames * 5/3 = 85/3 RoPE units.
    assert geo.video_temporal_position(5) == Fraction(85, 3)


def test_canvas_geometry_and_frame_budget() -> None:
    canvas = geo.CanvasGeometry(height=864, width=480)
    assert (canvas.latent_h, canvas.latent_w, canvas.frame_rows) == (54, 30, 405)
    with pytest.raises(ValueError):
        geo.CanvasGeometry(height=720, width=1280)
    assert geo.phases_for_frames(1) == 4
    assert geo.phases_for_frames(124) == 4
    assert geo.phases_for_frames(125) == 8
    assert geo.phases_for_frames(124 + 119 * 3) == 16


# ----------------------------------------------------------------------------
# schedules


def test_student_and_teacher_sigma_ladders() -> None:
    video, audio = student_sigmas()
    assert len(video) == len(audio) == 4
    assert video[0] == pytest.approx(1.0) and video[-1] == pytest.approx(0.0)
    full = time_shift_sigmas(num_steps=50, shift_scale=12.0)
    assert video == [full[0], full[16], full[33], full[49]]
    t_video, t_audio = teacher_sigmas()
    assert len(t_video) == len(t_audio) == 10
    assert t_video[3] > t_audio[3]  # video shift 12 keeps more noise than audio shift 3


def test_euler_eta0_update_matches_reference_formula() -> None:
    state = torch.randn(7, 4)
    velocity = torch.randn(7, 4)
    sigma, sigma_next = 0.7, 0.3
    expected_denoised = state + sigma * velocity
    expected = (sigma_next / sigma) * state + (1 - sigma_next / sigma) * expected_denoised
    out = euler_eta0_update_(state.clone(), velocity, sigma_curr=sigma, sigma_next=sigma_next)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------------
# packed layouts


def test_phase_layout_times_sit_on_the_global_timeline() -> None:
    plan = geo.request_plan(1)
    phase = plan.phases[2]
    text_len = 11
    origin = 17  # first request's text length
    video_offset, audio_offset = 37, 207
    packed = taomate_phase_packed_layout(
        text_len=text_len,
        phase=phase,
        latent_h=54,
        latent_w=30,
        media_time_origin=origin,
        video_latent_offset=video_offset,
        audio_latent_offset=audio_offset,
    )
    seq_len = int(packed["seq_len"])
    used = text_len + 2 * phase.audio_latent_count + phase.video_latent_count * 405
    assert seq_len % 64 == 0 and seq_len >= used
    assert int(packed["cu_seqlens"][1]) == used
    grid = packed["img_position_ids"]
    text_pos = packed["text_pos"].view(-1)
    start = prompt_rope_start(text_len=text_len, media_time_origin=origin, video_latent_offset=video_offset)
    assert grid[text_pos[0], 0].item() == pytest.approx(start)
    assert grid[text_pos[-1], 0].item() == pytest.approx(start + text_len - 1)
    first_video_row = packed["img_pos"].view(-1)[0]
    expected_video_time = float(origin + geo.video_temporal_position(video_offset + phase.video_latent_start))
    assert grid[first_video_row, 0].item() == pytest.approx(expected_video_time)
    audio_pos = packed["audio_pos"].view(-1).view(2, phase.audio_latent_count)
    assert grid[audio_pos[0, 0], 0].item() == pytest.approx(origin + audio_offset + phase.audio_latent_start)
    assert grid[audio_pos[1, -1], 0].item() == pytest.approx(origin + audio_offset + phase.audio_latent_stop - 1)
    tags = packed["token_tags"]
    assert (tags[text_pos] == TEXT_TOKEN_TAG).all()
    assert (tags[packed["audio_pos"].view(-1)] == AUDIO_TOKEN_TAG).all()
    assert (tags[packed["img_pos"].view(-1)] == VIDEO_TOKEN_TAG).all()
    assert (tags[used:] == -1).all()


def test_audio_only_layouts() -> None:
    plain = taomate_audio_only_packed_layout(text_len=9, audio_t=207, latent_h=54, latent_w=30)
    assert plain["img_pos"].numel() == 0 and plain["update_mask"].numel() == 0
    assert plain["audio_pos"].numel() == 2 * 207
    assert plain["audio_update_mask"].all()
    assert int(plain["cu_seqlens"][1]) == 9 + 414
    frozen = taomate_audio_only_frozen_prefix_packed_layout(
        text_len=9,
        ref_audio_t=40,
        audio_t=198,
        latent_h=54,
        latent_w=30,
        reference_time_start=9 + 207 - 40,
        target_time_start=9 + 207,
    )
    mask = frozen["audio_update_mask"]
    assert mask.shape[0] == 2 * (40 + 198)
    assert not mask[:80].any() and mask[80:].all()
    grid = frozen["img_position_ids"]
    audio_pos = frozen["audio_pos"].view(-1)
    assert grid[audio_pos[0], 0].item() == pytest.approx(9 + 207 - 40)
    assert grid[audio_pos[80], 0].item() == pytest.approx(9 + 207)


# ----------------------------------------------------------------------------
# KV cache


def _fake_kv(rows: int, contract: KVContract) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.randn(rows, contract.local_heads, contract.head_dim).to(contract.dtype)
    value = torch.randn(rows, contract.local_heads, contract.head_dim).to(contract.dtype)
    return key, value


def _stage_chunk(cache: CleanAVKVCache, contract: KVContract, *, video_rows: int, audio_rows: int, text_rows: int) -> None:
    seq = text_rows + audio_rows + video_rows + 3
    tags = torch.full((seq,), -1, dtype=torch.long)
    tags[:text_rows] = TEXT_TOKEN_TAG
    tags[text_rows : text_rows + audio_rows] = AUDIO_TOKEN_TAG
    tags[text_rows + audio_rows : text_rows + audio_rows + video_rows] = VIDEO_TOKEN_TAG
    mask = (tags == AUDIO_TOKEN_TAG) | (tags == VIDEO_TOKEN_TAG)
    cache.begin_clean_commit(cache.committed_blocks)
    for name in contract.layer_names:
        key, value = _fake_kv(seq, contract)
        cache.stage(name, key, value, tags, mask)
    cache.commit()


def test_kv_cache_retention_policy() -> None:
    contract = KVContract(num_layers=3, local_heads=2, head_dim=4, dtype=torch.float32)
    cache = CleanAVKVCache(contract)
    _stage_chunk(cache, contract, video_rows=12, audio_rows=6, text_rows=5)
    cache.retain_sink_and_recent_commits()
    assert cache.history_tokens == 18 and cache.committed_blocks == 1
    _stage_chunk(cache, contract, video_rows=10, audio_rows=4, text_rows=5)
    cache.retain_sink_and_recent_commits()
    assert cache.history_tokens == 18 + 14
    _stage_chunk(cache, contract, video_rows=10, audio_rows=4, text_rows=5)
    cache.retain_sink_and_recent_commits()
    # Chunk 0 aged to a video-only sink (12 rows), chunks 1 and 2 complete.
    assert cache.history_video_tokens == 12 + 10 + 10
    assert cache.history_audio_tokens == 4 + 4
    _stage_chunk(cache, contract, video_rows=5, audio_rows=2, text_rows=5)
    cache.retain_sink_and_recent_commits()
    assert cache.history_video_tokens == 12 + 10 + 5
    assert cache.history_audio_tokens == 4 + 2
    dropped = cache.drop_audio_history()
    assert dropped == 6 and cache.history_audio_tokens == 0 and cache.history_video_tokens == 27
    for name in contract.layer_names:
        assert cache.history(name).key.shape[0] == 27


def test_kv_cache_rollback_and_ordering() -> None:
    contract = KVContract(num_layers=2, local_heads=1, head_dim=2, dtype=torch.float32)
    cache = CleanAVKVCache(contract)
    cache.begin_clean_commit(0)
    with pytest.raises(RuntimeError):
        cache.commit()  # missing layers
    cache.rollback()
    assert not cache.clean_commit_active
    with pytest.raises(RuntimeError):
        cache.begin_clean_commit(1)


# ----------------------------------------------------------------------------
# LoRA adapter


class _TinyArch:
    num_layers = 1
    token_refiner_num_layers = 1
    num_attention_heads = 2
    attention_head_dim = 4
    hidden_size = 8
    ffn_hidden_size = 6


class _TupleLinear(nn.Linear):
    """A linear returning ``(out, bias)`` like vLLM's parallel linears."""

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        return super().forward(x), None


def _tiny_transformer() -> nn.Module:
    arch = _TinyArch()
    inner = arch.num_attention_heads * arch.attention_head_dim
    model = nn.Module()
    model.arch = arch
    for block in ("token_refiner.blocks.0", "blocks.0"):
        parent = model
        for part in block.split("."):
            child = getattr(parent, part, None)
            if child is None:
                child = nn.Module()
                parent.add_module(part, child)
            parent = child
        attn = nn.Module()
        attn.qkv_proj = _TupleLinear(arch.hidden_size, 3 * inner, bias=False)
        attn.out_proj = _TupleLinear(inner, arch.hidden_size, bias=False)
        mlp = nn.Module()
        mlp.fc1 = _TupleLinear(arch.hidden_size, 2 * arch.ffn_hidden_size, bias=False)
        mlp.fc2 = _TupleLinear(arch.ffn_hidden_size, arch.hidden_size, bias=False)
        parent.add_module("attn", attn)
        parent.add_module("mlp", mlp)
    return model


def _write_adapter(tmp_path, model: nn.Module, rank: int = 3, alpha: float = 6.0) -> str:
    tensors = {}
    modules = dict(model.named_modules())
    for target in taomate_lora_targets(num_blocks=1, num_refiner_blocks=1):
        weight = modules[target].weight
        out_features, in_features = weight.shape
        tensors[f"{target}.lora_a"] = torch.randn(rank, in_features)
        tensors[f"{target}.lora_b"] = torch.randn(out_features, rank)
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({"rank": rank, "alpha": alpha}))
    return str(tmp_path)


def _adapter_tensor(tmp_path, name: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(tmp_path / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def test_lora_adapter_hooks_add_scaled_delta_and_can_be_disabled(tmp_path) -> None:
    torch.manual_seed(0)
    model = _tiny_transformer()
    adapter_dir = _write_adapter(tmp_path, model)
    assert is_taomate_lora_dir(adapter_dir)
    adapter = TaoMateLoRAAdapter.load(adapter_dir, transformer=model, device=torch.device("cpu"), dtype=torch.float32)
    assert adapter.scale == pytest.approx(2.0)
    assert len(adapter.bound_targets) == 8
    x = torch.randn(5, 8)
    fc2_in = torch.randn(5, 6)
    with adapter.disabled():
        base_out, _ = model.get_submodule("blocks.0.mlp.fc2")(fc2_in)
    lora_out, _ = model.get_submodule("blocks.0.mlp.fc2")(fc2_in)
    a = adapter._lora_a["blocks.0.mlp.fc2"]
    b = adapter._lora_b["blocks.0.mlp.fc2"]
    expected = base_out + (fc2_in @ a.t()) @ b.t() * adapter.scale
    torch.testing.assert_close(lora_out, expected, rtol=1e-5, atol=1e-5)
    # The qkv B rows are consumed in the adapter's merged [Q; K; V] order.
    qkv_a = adapter._lora_a["blocks.0.attn.qkv_proj"]
    qkv_b = adapter._lora_b["blocks.0.attn.qkv_proj"]
    assert qkv_b.shape == (3 * 8, 3)
    torch.testing.assert_close(qkv_b, _adapter_tensor(tmp_path, "blocks.0.attn.qkv_proj.lora_b"))
    with adapter.disabled():
        base_qkv, _ = model.get_submodule("blocks.0.attn.qkv_proj")(x)
    lora_qkv, _ = model.get_submodule("blocks.0.attn.qkv_proj")(x)
    torch.testing.assert_close(lora_qkv - base_qkv, (x @ qkv_a.t()) @ qkv_b.t() * adapter.scale, rtol=1e-5, atol=1e-5)
    adapter.unbind()
    plain, _ = model.get_submodule("blocks.0.mlp.fc2")(fc2_in)
    torch.testing.assert_close(plain, base_out)


def test_lora_adapter_rejects_incomplete_inventory(tmp_path) -> None:
    model = _tiny_transformer()
    _write_adapter(tmp_path, model)
    tensors = {}
    from safetensors import safe_open

    with safe_open(str(tmp_path / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith("blocks.0.mlp.fc2"):
                tensors[key] = handle.get_tensor(key)
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    with pytest.raises(Exception, match="inventory"):
        TaoMateLoRAAdapter.load(str(tmp_path), transformer=model, device=torch.device("cpu"), dtype=torch.float32)


# ----------------------------------------------------------------------------
# streaming audio decoder tail handling


class _FakeAudioVAE:
    sample_rate = 32000

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        # 800 samples per latent, value = latent index of the window position.
        steps = int(latent.shape[-1])
        wave = latent[0, 0].repeat_interleave(800).view(1, 1, steps * 800).expand(1, 2, -1)
        return wave.float()


def test_audio_decoder_pads_the_rounded_tail_but_rejects_missing_audio() -> None:
    from vllm_omni.diffusion.models.taomate_h3.stream_decode import StreamingAudioDecoder

    decoder = StreamingAudioDecoder(_FakeAudioVAE(), device=torch.device("cpu"))
    latents = torch.zeros(2, 32, 603)
    latents[0, 0] = torch.arange(603, dtype=torch.float32)
    decoder.append(latents)
    # 362 frames at 24 fps need 482666 samples; 603 latents give 482400.
    samples = decoder.decode_range(453333, 482666)
    assert samples.shape == (482666 - 453333, 2)
    assert samples[-1, 0] == 0.0 and samples[-267, 0] == 602.0
    with pytest.raises(RuntimeError, match="do not cover"):
        decoder.decode_range(482400, 482400 + 2000)


# ----------------------------------------------------------------------------
# pinned packed lengths


def test_pinned_lengths_cover_every_phase_and_teacher_document() -> None:
    from vllm_omni.diffusion.models.taomate_h3 import pipeline as pipeline_module
    from vllm_omni.diffusion.models.taomate_h3.audio_teacher import ROLLOVER_LATENTS_PER_CHANNEL
    from vllm_omni.diffusion.models.taomate_h3.kv_cache import KVContract

    canvas = geo.CanvasGeometry(height=864, width=480)
    session = pipeline_module._Session(
        session_id="s",
        canvas=canvas,
        seed=1,
        contract=KVContract(num_layers=1, local_heads=1, head_dim=1),
        video_decoder=None,  # type: ignore[arg-type]
        audio_decoder=None,  # type: ignore[arg-type]
        audio_kv_reset_requests=12,
        pad_text_tokens=128,
    )
    seen: set[tuple[int, int]] = set()
    for request_index in range(6):
        plan = geo.request_plan(request_index)
        for phase in plan.phases:
            pinned = session.pinned_phase_seq_len(phase, text_len=40)
            assert pinned is not None and pinned % 64 == 0
            used = 128 + 2 * phase.audio_latent_count + phase.video_latent_count * canvas.frame_rows
            assert pinned >= used
            seen.add((phase.index, pinned))
    # Two shapes for phase 0 (12 latents once, then 10) and one per later phase.
    assert len(seen) == 5
    teacher_first = session.pinned_teacher_seq_len(40, with_reference=False)
    teacher_next = session.pinned_teacher_seq_len(40, with_reference=True)
    assert teacher_first == -(-(128 + 2 * 207) // 64) * 64
    assert teacher_next == -(-(128 + 2 * (207 + ROLLOVER_LATENTS_PER_CHANNEL)) // 64) * 64
    assert session.pinned_phase_seq_len(plan.phases[0], text_len=200) is None
