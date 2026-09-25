# SPDX-License-Identifier: Apache-2.0
"""The complete-policy stage must reject malformed observations before dispatch."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from vllm_omni.edge.internvla_actions import InternVLAA2DActionCodec
from vllm_omni.engine.backends.internvla import InternVLAStageClient


def _prompt():
    return {
        **{f"image{i}": np.zeros((1, 2, 3, 224, 224), np.float32) for i in range(3)},
        **{f"mask{i}": np.ones((1,), np.bool_) for i in range(3)},
        "state": np.zeros((1, 32), np.float32),
        "noise": np.zeros((1, 50, 32), np.float32),
        "task": "Place the marker pen in its holder.",
        "observation_timestamp_ns": 1_000_000_000,
    }


def test_action_observation_round_trip():
    raw = InternVLAStageClient._encode_prompt("request-1", _prompt())
    assert len(raw) < 8 << 20
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        assert str(data["request_id"].item()) == "request-1"
        assert data["image0"].shape == (1, 2, 3, 224, 224)
        assert data["mask0"].dtype == np.bool_
        assert int(data["observation_timestamp_ns"].item()) == 1_000_000_000


@pytest.mark.parametrize("field,value", [
    ("image0", np.full((1, 2, 3, 224, 224), np.nan, np.float32)),
    ("image1", np.full((1, 2, 3, 224, 224), 1.1, np.float32)),
    ("mask2", np.ones((1,), np.uint8)),
    ("state", np.zeros((1, 31), np.float32)),
    ("noise", np.full((1, 50, 32), np.inf, np.float32)),
    ("task", "x" * 4097),
    ("observation_timestamp_ns", -1),
])
def test_invalid_observation_rejected(field, value):
    prompt = _prompt()
    prompt[field] = value
    with pytest.raises(ValueError):
        InternVLAStageClient._encode_prompt("request-1", prompt)


def test_extra_observation_field_rejected():
    prompt = _prompt()
    prompt["undocumented"] = 1
    with pytest.raises(ValueError):
        InternVLAStageClient._encode_prompt("request-1", prompt)


def test_a2d_physical_actions_match_open_loop_reconstruction(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "max_state_dim": 32, "max_action_dim": 32, "chunk_size": 50,
    }))
    (tmp_path / "train_config.json").write_text(json.dumps({"dataset": {"action_mode": "delta"}}))
    stats = {"a2d": {}}
    for name, count, mean, std in (
        ("observation.states.joint.position", 14, 2.0, 0.5),
        ("observation.states.effector.position", 2, -1.0, 2.0),
        ("actions.joint.position", 14, 0.25, 0.75),
        ("actions.effector.position", 2, -0.5, 1.5),
    ):
        stats["a2d"][name] = {"mean": [mean] * count, "std": [std] * count}
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    codec = InternVLAA2DActionCodec.from_checkpoint(tmp_path)
    raw = np.arange(16, dtype=np.float32)[None, :] / 4
    normalized = (raw - codec.state_mean) / codec.state_std
    state = np.pad(normalized, ((0, 0), (0, 16))).astype(np.float32)
    assert np.array_equal(codec.validate_state(state, raw), raw)

    padded = np.zeros((1, 50, 32), dtype=np.float32)
    padded[:, :, :16] = np.arange(16, dtype=np.float32)[None, None, :] / 8
    padded[:, :, 16:] = 999  # The checkpoint output has padded slots, not physical controls.
    physical = codec.decode(padded, raw)
    expected = padded[:, :, :16] * codec.action_std + codec.action_mean
    expected[:, :, :14] += raw[:, None, :14]
    assert physical.shape == (1, 50, 16)
    np.testing.assert_array_equal(physical, expected)

    prompt = _prompt()
    prompt["state"] = state
    prompt["state_raw"] = raw
    encoded = InternVLAStageClient._encode_prompt("request-1", prompt)
    with np.load(io.BytesIO(encoded), allow_pickle=False) as worker_input:
        assert "state_raw" not in worker_input.files  # Stays on the controlling side of the boundary.
    with pytest.raises(ValueError, match="does not match"):
        codec.validate_state(state, raw + 1)
