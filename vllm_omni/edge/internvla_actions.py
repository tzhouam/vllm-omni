# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-bound A2D action decoding for the InternVLA policy.

This reproduces the maintained A2D open-loop evaluator's 16-coordinate
unnormalization and joint-delta reconstruction. It does not assign physical
units, a controller joint order, or a safe action period.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_STATE_KEYS = ("observation.states.joint.position", "observation.states.effector.position")
_ACTION_KEYS = ("actions.joint.position", "actions.effector.position")


def _statistics(stats: dict, keys: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    means, standard_deviations = [], []
    for key in keys:
        item = stats[key]
        means.extend(item["mean"])
        standard_deviations.extend(item["std"])
    mean = np.asarray(means, dtype=np.float32)
    std = np.asarray(standard_deviations, dtype=np.float32)
    if (mean.shape != (16,) or std.shape != (16,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or (std < 0).any()):
        raise ValueError("InternVLA A2D joint/effector statistics are invalid")
    return mean, std


@dataclass(frozen=True)
class InternVLAA2DActionCodec:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @classmethod
    def from_checkpoint(cls, model_dir: Path) -> InternVLAA2DActionCodec:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        train = json.loads((model_dir / "train_config.json").read_text(encoding="utf-8"))
        if (config.get("max_state_dim") != 32 or config.get("max_action_dim") != 32
                or config.get("chunk_size") != 50
                or train.get("dataset", {}).get("action_mode") != "delta"):
            raise ValueError("InternVLA A2D checkpoint state/action contract changed")
        stats = json.loads((model_dir / "stats.json").read_text(encoding="utf-8"))["a2d"]
        state_mean, state_std = _statistics(stats, _STATE_KEYS)
        action_mean, action_std = _statistics(stats, _ACTION_KEYS)
        return cls(state_mean, state_std, action_mean, action_std)

    def validate_state(self, padded_normalized: np.ndarray, raw: np.ndarray) -> np.ndarray:
        state = np.asarray(padded_normalized)
        physical = np.asarray(raw)
        if (state.shape != (1, 32) or state.dtype != np.float32
                or physical.shape != (1, 16) or physical.dtype != np.float32
                or not np.isfinite(state).all() or not np.isfinite(physical).all()):
            raise ValueError("InternVLA A2D state must be finite float32 [1,32] and raw [1,16]")
        denominator = np.where(self.state_std == 0, np.float32(1), self.state_std)
        normalized = (physical - self.state_mean) / denominator
        if (not np.allclose(state[:, :16], normalized, rtol=1e-5, atol=1e-5)
                or not np.array_equal(state[:, 16:], np.zeros((1, 16), np.float32))):
            raise ValueError("InternVLA raw A2D state does not match the normalized policy input")
        return physical.copy()

    def decode(self, padded_actions: np.ndarray, raw_state: np.ndarray) -> np.ndarray:
        actions = np.asarray(padded_actions)
        state = np.asarray(raw_state)
        if (actions.shape != (1, 50, 32) or actions.dtype != np.float32
                or state.shape != (1, 16) or state.dtype != np.float32
                or not np.isfinite(actions).all() or not np.isfinite(state).all()):
            raise ValueError("InternVLA A2D decode requires finite padded actions and raw state")
        physical = actions[:, :, :16] * self.action_std + self.action_mean
        physical = np.ascontiguousarray(physical, dtype=np.float32)
        physical[:, :, :14] += state[:, None, :14]
        if not np.isfinite(physical).all():
            raise ValueError("InternVLA A2D physical actions are nonfinite")
        return physical
