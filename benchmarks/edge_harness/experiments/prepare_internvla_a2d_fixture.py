#!/usr/bin/env python3
"""Prepare one real A2D observation for the bounded Omni InternVLA profiler.

This intentionally reuses the maintained offline example's Parquet/video reader,
state normalization and history alignment. It refuses similarly named Genie-1
datasets whose state fields do not match the Place_Markpen checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source", required=True,
                        help="Dataset identifier and immutable revision or archive digest")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-fixture", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_index < 0 or ("@" not in args.source and "sha256:" not in args.source):
        parser.error("sample index must be nonnegative and source must include a revision or SHA-256")
    if args.output_fixture.suffix != ".npz" or args.output_manifest.suffix != ".json":
        parser.error("fixture output must be .npz and manifest output must be .json")

    import numpy as np
    import pyarrow.parquet as pq

    model_dir = args.model_dir.resolve(strict=True)
    dataset_dir = args.dataset_dir.resolve(strict=True)
    required_state = ["observation.states.joint.position",
                      "observation.states.effector.position"]
    required_action = ["actions.joint.position", "actions.effector.position"]
    required_camera = ["observation.images.head", "observation.images.hand_left",
                       "observation.images.hand_right"]
    data_file = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
    available = set(pq.read_schema(data_file).names)
    missing_state = set(required_state) - available
    if missing_state:
        raise ValueError(f"dataset is not the checkpoint's A2D state schema: missing {sorted(missing_state)}")
    missing_action = set(required_action) - available
    if missing_action:
        raise ValueError(f"dataset is not the checkpoint's A2D action schema: missing {sorted(missing_action)}")
    info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    if any(key not in info.get("features", {}) for key in required_camera):
        raise ValueError("dataset is missing the checkpoint's A2D camera streams")

    example_dir = Path(__file__).resolve().parents[3] / "examples" / "offline_inference" / "internvla_a1"
    sys.path.insert(0, str(example_dir))
    from internvla_a1_common import A2DOpenLoopDataset, make_shared_noise
    from vllm_omni.diffusion.models.internvla_a1 import InternVLAA1Config
    from vllm_omni.diffusion.models.internvla_a1.model_internvla_a1 import pad_vector, resize_with_pad

    config = InternVLAA1Config.from_pretrained(model_dir)
    train_config = json.loads((model_dir / "train_config.json").read_text(encoding="utf-8"))
    if train_config.get("dataset", {}).get("action_mode") != "delta":
        raise ValueError("checkpoint A2D action mode is not delta")
    stats = json.loads((model_dir / "stats.json").read_text(encoding="utf-8"))["a2d"]
    dataset = A2DOpenLoopDataset(dataset_dir, config=config, train_stats=stats)
    if args.sample_index >= len(dataset.data_rows):
        raise ValueError("sample index exceeds the available Parquet rows")
    sample = dataset.get_sample(args.sample_index)
    fields = {}
    for index in range(3):
        image = sample.inputs[f"observation.images.image{index}"]
        fields[f"image{index}"] = resize_with_pad(image, (224, 224)).unsqueeze(0).numpy().astype(np.float32)
        fields[f"mask{index}"] = np.ones((1,), dtype=np.bool_)
    fields["state"] = pad_vector(sample.inputs["observation.state"], 32).unsqueeze(0).numpy().astype(np.float32)
    fields["state_raw"] = sample.state_raw.unsqueeze(0).numpy().astype(np.float32)
    fields["reference_actions"] = sample.action_raw.unsqueeze(0).numpy().astype(np.float32)
    fields["noise"] = make_shared_noise(args.seed, args.sample_index, (1, 50, 32), "cpu").numpy()
    fields["task"] = np.array(sample.task)

    args.output_fixture.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_fixture, **fields)
    manifest = {
        "format": "internvla-a1-omni-observation-v2",
        "source": args.source,
        "sample_index": args.sample_index,
        "episode_index": sample.episode_index,
        "seed": args.seed,
        "task": sample.task,
        "state_fields": required_state,
        "action_fields": required_action,
        "physical_action_dim": 16,
        "action_reconstruction": "checkpoint-stats-a2d-unnormalize-plus-joint-delta",
        "camera_fields": required_camera,
        "state_normalization": "checkpoint-stats-a2d-mean-std",
        "image_preprocessing": "resize-with-pad-224-float32-0-to-1",
        "image_offsets_frames": [-15, 0],
        "checkpoint_model_sha256": sha256(model_dir / "model.safetensors"),
        "checkpoint_stats_sha256": sha256(model_dir / "stats.json"),
        "checkpoint_train_config_sha256": sha256(model_dir / "train_config.json"),
        "dataset_info_sha256": sha256(dataset_dir / "meta" / "info.json"),
        "dataset_data_sha256": sha256(data_file),
        "fixture_sha256": sha256(args.output_fixture),
    }
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fixture": str(args.output_fixture),
                      "manifest": str(args.output_manifest),
                      "fixture_sha256": manifest["fixture_sha256"]}))


if __name__ == "__main__":
    main()
