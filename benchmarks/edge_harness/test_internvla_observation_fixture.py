"""The InternVLA profiler must reject observations that can misstate evidence."""

import json

import numpy as np
import pytest

from benchmarks.edge_harness.experiments.probe_omni_internvla_policy import (
    load_observation_fixture,
    sha256,
)


def test_fixture_pins_schema_and_content(tmp_path):
    fixture = tmp_path / "observation.npz"
    manifest_path = tmp_path / "observation.json"
    fields = {f"image{i}": np.zeros((1, 2, 3, 224, 224), dtype=np.float32)
              for i in range(3)}
    fields.update({f"mask{i}": np.ones((1,), dtype=np.bool_) for i in range(3)})
    fields.update(state=np.zeros((1, 32), dtype=np.float32),
                  noise=np.zeros((1, 50, 32), dtype=np.float32),
                  task=np.array("Place the marker pen in its holder."))
    np.savez(fixture, **fields)
    manifest = {
        "format": "internvla-a1-omni-observation-v1",
        "source": "local-test@revision",
        "state_fields": ["observation.states.joint.position",
                         "observation.states.effector.position"],
        "camera_fields": ["observation.images.head", "observation.images.hand_left",
                          "observation.images.hand_right"],
        "state_normalization": "checkpoint-stats-a2d-mean-std",
        "image_preprocessing": "resize-with-pad-224-float32-0-to-1",
        "checkpoint_model_sha256": "a" * 64,
        "checkpoint_stats_sha256": "b" * 64,
        "fixture_sha256": sha256(fixture),
        "task": str(fields["task"].item()),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    prompt, _ = load_observation_fixture(fixture, manifest_path)
    assert prompt["image0"].shape == (1, 2, 3, 224, 224)

    manifest["state_fields"] = ["states.left_joint.position", "states.right_joint.position"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible state_fields"):
        load_observation_fixture(fixture, manifest_path)

    manifest["state_fields"] = ["observation.states.joint.position",
                                "observation.states.effector.position"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    fields["state"][0, 0] = 1.0
    np.savez(fixture, **fields)
    with pytest.raises(ValueError, match="content hash differs"):
        load_observation_fixture(fixture, manifest_path)
