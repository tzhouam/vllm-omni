# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shape arithmetic for the exported vision tower.

The tower itself needs a 29 GB checkpoint, so what is covered here is the part
that decides whether an export is even well-formed — and that is where a silent
mistake would land: a grid that does not divide by the merge size produces a
graph that exports, runs, and returns the wrong number of tokens to the LLM.
"""

from __future__ import annotations

import json

import pytest

from vllm_omni.edge.local.artifacts.vision_tower import tower_shape

VISION_CONFIG = {
    "depth": 27, "hidden_size": 1152, "in_channels": 3, "intermediate_size": 4304,
    "num_heads": 16, "num_position_embeddings": 2304, "out_hidden_size": 5120,
    "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
}


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": VISION_CONFIG}))
    return tmp_path


def test_448_gives_784_patches_and_196_tokens(checkpoint):
    shape = tower_shape(checkpoint, 448)
    assert (shape.grid_h, shape.grid_w) == (28, 28)
    assert shape.patches == 784
    # 3 channels x 2 temporal x 16 x 16
    assert shape.patch_dim == 1536
    # 784 patches merged 2x2
    assert shape.merged_tokens == 196
    assert shape.out_hidden_size == 5120


def test_image_size_must_divide_by_the_patch_size(checkpoint):
    with pytest.raises(ValueError, match="multiple of patch_size"):
        tower_shape(checkpoint, 450)


def test_grid_must_divide_by_the_spatial_merge(checkpoint):
    """A 3x3 grid of patches cannot be merged 2x2. Exporting it anyway would
    hand the language model a silently wrong number of image tokens."""
    with pytest.raises(ValueError, match="spatial_merge_size"):
        tower_shape(checkpoint, 48)


def test_a_checkpoint_without_a_tower_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "spark2_5"}))
    with pytest.raises(ValueError, match="no vision_config"):
        tower_shape(tmp_path, 448)


@pytest.mark.parametrize("size,patches,tokens", [(224, 196, 49), (448, 784, 196), (896, 3136, 784)])
def test_token_count_scales_with_area(checkpoint, size, patches, tokens):
    shape = tower_shape(checkpoint, size)
    assert (shape.patches, shape.merged_tokens) == (patches, tokens)
