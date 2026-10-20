"""WP8 edge scheduling policy: playback slack, prefill throttling, no-preemption admission (CPU)."""

from __future__ import annotations

import pytest

from vllm_omni.edge import adapt
from vllm_omni.edge.hardware_probe import HardwareProfile
from vllm_omni.edge.scheduling import (
    ChunkArrival,
    KVGeometry,
    admission_allows_no_preemption,
    kv_bytes_for_seqs,
    kv_bytes_per_token,
    max_seqs_without_preemption,
    playback_slack_ms,
    should_throttle_prefills,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

QWEN3_TTS_TALKER = KVGeometry(num_layers=28, num_kv_heads=8, head_dim=128)


def test_playback_slack_tracks_buffered_audio_minus_elapsed():
    # measured default schedule: 1 frame (80 ms) at t=30, 25 frames (2000 ms) at t=190
    chunks = [ChunkArrival(30.0, 80.0), ChunkArrival(190.0, 2000.0)]
    assert playback_slack_ms(chunks, now_ms=30.0) == pytest.approx(80.0)
    assert playback_slack_ms(chunks, now_ms=110.0) == pytest.approx(0.0)  # underrun starts here
    assert playback_slack_ms(chunks, now_ms=150.0) == pytest.approx(-40.0)
    assert playback_slack_ms(chunks, now_ms=190.0) == pytest.approx(2080.0 - 160.0)
    # edge ramp: 2,4 frames at 42 ms / 200 ms never underruns
    ramp = [ChunkArrival(42.0, 160.0), ChunkArrival(200.0, 320.0)]
    assert playback_slack_ms(ramp, now_ms=199.0) > 0
    assert playback_slack_ms([], now_ms=5.0) == 0.0


def test_throttle_decision_uses_eta_and_margin():
    assert should_throttle_prefills(slack_ms=50.0, next_chunk_eta_ms=40.0, safety_margin_ms=20.0)
    assert not should_throttle_prefills(slack_ms=500.0, next_chunk_eta_ms=160.0, safety_margin_ms=20.0)
    assert should_throttle_prefills(slack_ms=-10.0, next_chunk_eta_ms=0.0, safety_margin_ms=0.0)


def test_kv_geometry_and_bytes():
    assert kv_bytes_per_token(28, 8, 128, "bfloat16") == 2 * 28 * 8 * 128 * 2 == 114688
    assert kv_bytes_per_token(28, 8, 128, "float32") == 229376
    assert QWEN3_TTS_TALKER.bytes_per_token("bfloat16") == 114688
    geo = KVGeometry.from_hf_config(
        {
            "talker_config": {
                "num_hidden_layers": 28,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "num_attention_heads": 16,
            }
        }
    )
    assert geo == QWEN3_TTS_TALKER
    assert KVGeometry.from_hf_config({"foo": 1}) is None
    # blocks round up to the block size
    assert kv_bytes_for_seqs(1, 17, 100, block_size=16) == 32 * 100


def test_admission_rule_for_qwen3_tts_edge_profile():
    bpt = QWEN3_TTS_TALKER.bytes_per_token("bfloat16")
    # 4 seqs x 2048 tokens x 112 KiB = 896 MiB needed
    need = kv_bytes_for_seqs(4, 2048, bpt)
    assert need == 4 * 2048 * 114688
    assert admission_allows_no_preemption(need, 4, 2048, bpt)
    assert not admission_allows_no_preemption(need - 1, 4, 2048, bpt)
    assert max_seqs_without_preemption(need - 1, 2048, bpt) == 3
    assert max_seqs_without_preemption(0, 2048, bpt) == 0


ADAPT_PROFILES = {
    "x86_cpu_8g": dict(
        arch="x86_64",
        accelerator="none",
        ram_total_bytes=8 << 30,
        ram_available_bytes=6 << 30,
        cpu_flags=["avx2"],
        n_physical=4,
        n_logical=4,
    ),
    "jetson_8g": dict(
        arch="aarch64",
        accelerator="cuda_unified",
        ram_total_bytes=8 << 30,
        ram_available_bytes=6 << 30,
        gpu_mem_bytes=8 << 30,
        cpu_flags=["neon"],
        n_physical=6,
        n_logical=6,
        supports_bf16=True,
    ),
    "l20x": dict(
        arch="x86_64",
        accelerator="cuda_discrete",
        ram_total_bytes=256 << 30,
        ram_available_bytes=200 << 30,
        gpu_mem_bytes=48 << 30,
        cpu_flags=["avx2", "avx512"],
        n_physical=32,
        n_logical=64,
        supports_bf16=True,
    ),
}


@pytest.mark.parametrize("profile_kwargs", list(ADAPT_PROFILES.values()), ids=list(ADAPT_PROFILES.keys()))
def test_derived_budgets_never_allow_preemption(profile_kwargs):
    """The derived stage-0 KV budget always holds max_num_seqs x max_model_len talker tokens."""
    profile = HardwareProfile(**profile_kwargs)
    ov = adapt.derive_overrides(profile, weights_bytes=2 << 30, kv_geometry=QWEN3_TTS_TALKER)
    s0 = ov["stages"][0]
    kv0 = s0["engine_args"]["kv_cache_memory_bytes"]
    dtype = s0["engine_args"]["dtype"]
    bpt = QWEN3_TTS_TALKER.bytes_per_token(dtype)
    assert s0["max_num_seqs"] >= 1
    assert admission_allows_no_preemption(kv0, s0["max_num_seqs"], s0["max_model_len"], bpt), (kv0, s0)
