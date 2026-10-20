# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Placing a checkpoint that does not fit in the VRAM."""

import json
import struct

import pytest

from vllm_omni.edge.placement import (
    GiB,
    ParamGroup,
    checkpoint_param_groups,
    group_key,
    is_cold,
    plan_placement,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LINK = 13.9e9   # measured pinned H2D on the reference laptop, PCIe 4.0 x8
VRAM = 472e9    # measured VRAM bandwidth on the same machine


def _plan(groups, avail_gib, **kw):
    return plan_placement(
        groups, device_bytes_available=int(avail_gib * GiB),
        link_bytes_per_s=LINK, vram_bytes_per_s=VRAM, **kw)


# ------------------------------------------------------------ classification
def test_embed_tokens_is_cold_and_lm_head_is_not():
    """Same shape, same dtype, opposite decision: the embedding is a gather
    and the head is a GEMM over the whole table."""
    assert is_cold("model.language_model.embed_tokens.weight")
    assert not is_cold("lm_head.weight")


def test_cold_matching_is_by_segment_not_substring():
    assert is_cold("model.visual.blocks.0.attn.qkv.weight")
    assert not is_cold("model.visualiser.weight")


def test_a_decoder_layer_is_never_folded_into_one_group():
    """A 3-segment prefix makes all 64 layers a single 24 GB group, and a
    planner that moves groups whole then has no choice between none and all."""
    assert group_key("model.language_model.layers.17.mlp.down_proj.weight") == (
        "model.language_model.layers.17")
    assert group_key("model.visual.blocks.3.attn.qkv.weight").startswith("model.visual")


# ------------------------------------------------------------------ planning
def test_a_checkpoint_that_fits_is_left_alone():
    groups = [ParamGroup("model.layers.0", 8 * GiB, False)]
    p = _plan(groups, 20)
    assert p.strategy == "resident"
    assert p.host_bytes == 0 and p.est_link_s_per_token == 0.0
    assert p.engine_kwargs == {}


def test_the_deficit_comes_out_of_the_cold_set_first():
    """4 GiB over, and there is 6 GiB of cold weight to spend: nothing that a
    decode step reads should end up on the host."""
    groups = [
        ParamGroup("model.layers.0", 18 * GiB, False),
        ParamGroup("model.embed_tokens", 6 * GiB, True),
    ]
    p = _plan(groups, 20)
    assert p.strategy == "cold_offload"
    assert p.host_bytes == 6 * GiB
    assert p.host_read_per_token == 0
    assert p.est_link_s_per_token == 0.0


def test_an_offloaded_embedding_still_pays_one_row_per_token():
    groups = [
        ParamGroup("model.layers.0", 18 * GiB, False),
        ParamGroup("model.embed_tokens", 6 * GiB, True),
    ]
    p = _plan(groups, 20, embedding_row_bytes=5120 * 2)
    assert p.host_read_per_token == 5120 * 2
    # 10 KiB over a 13.9 GB/s link is well under a microsecond.
    assert p.est_link_s_per_token < 1e-6


def test_hot_weight_is_only_spent_once_the_cold_set_runs_out():
    groups = [
        ParamGroup("model.embed_tokens", 2 * GiB, True),
        *[ParamGroup(f"model.layers.{i}", 1 * GiB, False) for i in range(20)],
    ]
    p = _plan(groups, 16)
    assert p.strategy == "hot_offload"
    # 6 GiB over; 2 GiB is free, so exactly 4 layers should cross.
    assert p.host_read_per_token == 4 * GiB
    assert p.host_bytes == 6 * GiB


def test_the_estimate_prices_host_bytes_at_the_link_and_the_rest_at_vram():
    groups = [
        ParamGroup("model.embed_tokens", 2 * GiB, True),
        *[ParamGroup(f"model.layers.{i}", 1 * GiB, False) for i in range(20)],
    ]
    p = _plan(groups, 16)
    assert p.est_link_s_per_token == pytest.approx(4 * GiB / LINK)
    assert p.est_vram_s_per_token == pytest.approx(16 * GiB / VRAM)
    # The link is 34x slower, so 4 GiB on the host outweighs 16 GiB resident.
    assert p.est_link_s_per_token > 8 * p.est_vram_s_per_token


def test_a_checkpoint_too_big_for_host_and_device_says_so():
    groups = [ParamGroup("model.layers.0", 40 * GiB, False)]
    p = _plan(groups, 8)
    # Everything hot can still move to the host, so this one is feasible but
    # miserable; infeasible needs the groups themselves to run out.
    assert p.strategy == "hot_offload"
    assert p.deficit_bytes == 32 * GiB


def test_engine_kwargs_name_only_segments_vllm_can_match():
    groups = [
        ParamGroup("model.embed_tokens", 6 * GiB, True),
        ParamGroup("model.layers.0", 18 * GiB, False),
    ]
    kw = _plan(groups, 20).engine_kwargs
    assert kw["cpu_offload_params"] == ["embed_tokens"]
    assert kw["cpu_offload_gb"] >= 6.0
    assert "offload_group_size" not in kw  # nothing hot was offloaded


# --------------------------------------------------------------- checkpoints
def _write_shard(path, tensors):
    """A minimal safetensors file: 8-byte header length, JSON header, data."""
    header, off = {}, 0
    for name, nbytes in tensors:
        header[name] = {"dtype": "BF16", "shape": [nbytes // 2],
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * off)


def test_param_groups_are_read_from_headers_without_loading_weights(tmp_path):
    _write_shard(tmp_path / "a.safetensors", [
        ("model.language_model.embed_tokens.weight", 4096),
        ("lm_head.weight", 4096),
        ("model.language_model.layers.0.mlp.down_proj.weight", 2048),
        ("model.language_model.layers.1.mlp.down_proj.weight", 2048),
    ])
    groups = {g.name: g for g in checkpoint_param_groups(tmp_path)}
    assert groups["model.language_model.embed_tokens"].cold
    assert not groups["lm_head.weight"].cold
    assert set(groups) >= {"model.language_model.layers.0",
                           "model.language_model.layers.1"}
    assert groups["model.language_model.layers.0"].bytes == 2048


def test_a_group_is_cold_only_if_every_tensor_in_it_is(tmp_path):
    _write_shard(tmp_path / "a.safetensors", [
        ("model.visual.blocks.0.attn.qkv.weight", 1024),
        ("model.visual.blocks.0.lm_head.weight", 1024),  # contrived, but hot
    ])
    (group,) = [g for g in checkpoint_param_groups(tmp_path)
                if g.name.startswith("model.visual")]
    assert not group.cold


def test_a_directory_with_no_shards_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="safetensors"):
        checkpoint_param_groups(tmp_path)


# ------------------------------------------- the cost model, against hardware
# Qwen3.8-27B-FP8, 4096-token context, batch 1, greedy, eager, idle machine.
# Four points from benchmarks/edge_harness; see docs/edge/laptop-heterogeneous.md.
MEASURED_S_PER_TOKEN = {10: 0.952, 12: 1.124, 14: 1.316, 16: 1.515}
QWEN38_FP8_GB = 30.87


def test_the_estimate_reproduces_the_measured_offload_curve():
    """The whole design rests on `resident/472 + offloaded/10.6`. It is worth
    one test that the arithmetic matches the hardware it claims to describe.

    It runs slightly *pessimistic* at every point, which is the right
    direction for a number used to decide whether a build will fit.
    """
    # 0.25 GB chunks, so the planner offloads close to exactly the target
    # rather than having to take one oversized group whole -- which is also
    # what the real checkpoint looks like at 0.38 GB per decoder layer.
    CHUNK = 0.25
    n = round(QWEN38_FP8_GB / CHUNK)
    for offloaded_gb, measured in MEASURED_S_PER_TOKEN.items():
        groups = [ParamGroup(f"layers.{i}", int(CHUNK * 1e9), False) for i in range(n)]
        p = plan_placement(
            groups,
            device_bytes_available=int((QWEN38_FP8_GB - offloaded_gb) * 1e9),
            link_bytes_per_s=10.6e9, vram_bytes_per_s=472e9,
        )
        modelled = p.est_link_s_per_token + p.est_vram_s_per_token
        assert modelled == pytest.approx(measured, rel=0.05), (
            f"{offloaded_gb} GB: modelled {modelled:.3f}s vs measured {measured:.3f}s")
        assert modelled >= measured, "the estimate must not flatter the build"


def test_the_link_term_dominates_at_every_measured_point():
    """At the tightest fit the 10 GB on the host is ~94% of the step. If this
    ever stops being true the advice in the design document changes."""
    groups = [
        ParamGroup("resident", int(20.87e9), False),
        ParamGroup("onhost", int(10e9), False),
    ]
    p = plan_placement(groups, device_bytes_available=int(20.87e9),
                       link_bytes_per_s=10.6e9, vram_bytes_per_s=472e9)
    share = p.est_link_s_per_token / (p.est_link_s_per_token + p.est_vram_s_per_token)
    assert share > 0.9
