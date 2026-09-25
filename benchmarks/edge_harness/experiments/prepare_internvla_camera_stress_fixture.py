#!/usr/bin/env python3
"""Build an explicitly out-of-domain camera stress fixture for InternVLA.

This is for numerical and execution robustness only. No recorded robot state or
reference action is imported, so the output cannot establish task quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frames(path: Path, indices: tuple[int, int]):
    import av
    import numpy as np

    selected = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in indices:
                selected[index] = frame.to_ndarray(format="rgb24")
            if len(selected) == 2:
                break
    if set(selected) != set(indices):
        raise ValueError(f"video has no frames at both requested indices: {path}")
    return np.stack([selected[index] for index in indices])


def resize_with_pad(frames):
    import torch
    import torch.nn.functional as F

    images = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255)
    height, width = images.shape[-2:]
    scale = min(224 / height, 224 / width)
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    resized = F.interpolate(images, size=(resized_h, resized_w),
                            mode="bilinear", align_corners=False)
    pad_h, pad_w = 224 - resized_h, 224 - resized_w
    return F.pad(resized, (pad_w // 2, pad_w - pad_w // 2,
                           pad_h // 2, pad_h - pad_h // 2)).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--head-video", type=Path, required=True)
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--source", required=True,
                        help="Dataset identifier with immutable revision or archive SHA-256")
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--frame-indices", type=int, nargs=2, default=(15, 30))
    parser.add_argument("--output-fixture", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    indices = tuple(args.frame_indices)
    if (indices[0] < 0 or indices[0] >= indices[1] or
            ("@" not in args.source and "sha256:" not in args.source)):
        parser.error("use increasing nonnegative frame indices and a pinned source")
    if args.output_fixture.suffix != ".npz" or args.output_manifest.suffix != ".json":
        parser.error("fixture output must be .npz and manifest output must be .json")

    import numpy as np

    model_dir = args.model_dir.resolve(strict=True)
    videos = [args.head_video.resolve(strict=True), args.left_video.resolve(strict=True),
              args.right_video.resolve(strict=True)]
    fields = {f"image{i}": resize_with_pad(read_frames(path, indices))[None].astype(np.float32)
              for i, path in enumerate(videos)}
    fields.update({f"mask{i}": np.ones((1,), dtype=np.bool_) for i in range(3)})
    fields.update(state=np.zeros((1, 32), dtype=np.float32),
                  noise=np.zeros((1, 50, 32), dtype=np.float32),
                  task=np.array("Place the marker pen in its holder."))
    args.output_fixture.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_fixture, **fields)
    manifest = {
        "format": "internvla-a1-omni-observation-v1",
        "source": args.source,
        "observation_kind": "recorded-camera-with-synthetic-checkpoint-mean-state",
        "source_task": args.source_task,
        "task": str(fields["task"].item()),
        "source_frame_indices": list(indices),
        "source_camera_fields": ["images.rgb.head", "images.rgb.hand_left",
                                 "images.rgb.hand_right"],
        "source_video_sha256": {key: sha256(path) for key, path in zip(
            ("head", "hand_left", "hand_right"), videos)},
        "state_source": "zero normalized vector: checkpoint training mean, not recorded robot state",
        "state_fields": ["observation.states.joint.position",
                         "observation.states.effector.position"],
        "camera_fields": ["observation.images.head", "observation.images.hand_left",
                          "observation.images.hand_right"],
        "state_normalization": "checkpoint-stats-a2d-mean-std",
        "image_preprocessing": "resize-with-pad-224-float32-0-to-1",
        "checkpoint_model_sha256": sha256(model_dir / "model.safetensors"),
        "checkpoint_stats_sha256": sha256(model_dir / "stats.json"),
        "fixture_sha256": sha256(args.output_fixture),
    }
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fixture_sha256": manifest["fixture_sha256"],
                      "video_sha256": manifest["source_video_sha256"]}))


if __name__ == "__main__":
    main()
