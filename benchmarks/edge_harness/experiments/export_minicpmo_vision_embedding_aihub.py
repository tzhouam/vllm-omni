#!/usr/bin/env python3
"""Export MiniCPM-o's first vision neural stage on one real-photo 448px bucket.

This is only patch/position embedding. It does not run the vision transformer,
audio encoder, thinker, speech head, or vocoder on the hosted target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from safetensors import safe_open


REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_l2(reference: np.ndarray, observed: np.ndarray) -> float:
    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


class FullBucketEmbedding(torch.nn.Module):
    """Equivalent to SiglipVisionEmbeddings for an unpadded 32×32 patch grid."""

    def __init__(self, embeddings: torch.nn.Module):
        super().__init__()
        self.patch_embedding = embeddings.patch_embedding
        self.position_embedding = embeddings.position_embedding
        side = embeddings.num_patches_per_side
        boundaries = torch.arange(1 / side, 1.0, 1 / side)
        fractional = torch.arange(0, 1 - 1e-6, 1 / 32)
        buckets = torch.bucketize(fractional, boundaries, right=True)
        position_ids = (buckets[:, None] * side + buckets).flatten()
        if position_ids.numel() != 1024:
            raise RuntimeError("32×32 position bucket changed")
        self.register_buffer("position_ids", position_ids, persistent=False)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        patch = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        positions = self.position_embedding(self.position_ids).unsqueeze(0)
        return patch + positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.threads <= 24:
        parser.error("threads must be 1..24")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(args.threads)
    model_dir = args.model_dir.resolve()
    image_path = args.image.resolve()
    config_path = model_dir / "config.json"
    shard = model_dir / "model-00004-of-00004.safetensors"
    source_path = model_dir / "modeling_navit_siglip.py"
    revisions = {
        (model_dir / ".cache/huggingface/download" / (path.name + ".metadata"))
        .read_text(encoding="utf-8").splitlines()[0]
        for path in (config_path, shard, source_path)
    }
    if revisions != {REVISION}:
        raise RuntimeError(f"vision source files do not share revision {REVISION}")
    sys.path.insert(0, str(model_dir))
    from modeling_navit_siglip import SiglipVisionConfig, SiglipVisionEmbeddings

    config = json.loads(config_path.read_text(encoding="utf-8"))
    vcfg = SiglipVisionConfig(**config["vision_config"])
    if vcfg.image_size != 980 or vcfg.patch_size != 14 or vcfg.hidden_size != 1152:
        raise RuntimeError("expected 980px position table and 448px, 32×32 image bucket")
    embeddings = SiglipVisionEmbeddings(vcfg).eval()
    with safe_open(str(shard), framework="pt", device="cpu") as weights:
        state = {key[len("vpm.embeddings."):]: weights.get_tensor(key)
                 for key in weights.keys() if key.startswith("vpm.embeddings.")}
    embeddings.load_state_dict(state, strict=True)
    photo = Image.open(image_path).convert("RGB").resize((448, 448))
    pixels = np.asarray(photo, dtype=np.float32) / 127.5 - 1.0
    pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
    tensor = torch.from_numpy(pixels)
    mask = torch.ones((1, 32, 32), dtype=torch.bool)
    sizes = torch.tensor([[32, 32]], dtype=torch.int32)
    with torch.inference_mode():
        fp32_ref = embeddings.float()(tensor, mask, sizes).numpy()
        bf16_ref = embeddings.to(torch.bfloat16)(tensor.to(torch.bfloat16), mask, sizes).float().numpy()
        wrapper = FullBucketEmbedding(embeddings.float()).eval()
        wrapped = wrapper(tensor).numpy()
    if fp32_ref.shape != (1, 1024, 1152) or not np.isfinite(fp32_ref).all():
        raise RuntimeError("vision embedding output contract failed")
    wrapper_error = relative_l2(fp32_ref, wrapped)
    if wrapper_error > 1e-7:
        raise RuntimeError(f"full-bucket rewrite changed embedding: {wrapper_error}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "vision_embedding_448.onnx"
    input_path = output_dir / "astronaut_pixels.npz"
    reference_path = output_dir / "reference_output.npz"
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (tensor,), str(onnx_path), input_names=["pixels"],
            output_names=["hidden"], opset_version=17, dynamo=False,
            do_constant_folding=True,
        )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    observed = session.run(["hidden"], {"pixels": pixels})[0]
    onnx_error = relative_l2(fp32_ref, observed)
    if observed.shape != fp32_ref.shape or onnx_error > 1e-5:
        raise RuntimeError(f"ONNX embedding parity failed: {onnx_error}")
    np.savez_compressed(input_path, pixels=pixels)
    np.savez_compressed(reference_path, fp32=fp32_ref, bf16=bf16_ref)
    report = {
        "scope": "first vision neural stage on one real astronaut photo; component only",
        "checkpoint": "openbmb/MiniCPM-o-4_5",
        "revision": REVISION,
        "image_sha256": sha256(image_path),
        "config_sha256": sha256(config_path),
        "source_sha256": sha256(source_path),
        "weight_shard_sha256": sha256(shard),
        "input_shape": list(pixels.shape),
        "output_shape": list(fp32_ref.shape),
        "precision": "BF16 checkpoint values promoted to FP32 for ONNX",
        "wrapper_vs_eager_fp32_relative_l2": wrapper_error,
        "onnx_vs_eager_fp32_relative_l2": onnx_error,
        "eager_fp32_vs_bf16_relative_l2": relative_l2(bf16_ref, fp32_ref),
        "artifacts": {path.name: sha256(path) for path in
                      (onnx_path, input_path, reference_path)},
        "batch_size": 1,
        "concurrency": 1,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"onnx_sha256": report["artifacts"][onnx_path.name],
                      "onnx_vs_eager_relative_l2": onnx_error,
                      "fp32_vs_bf16_relative_l2": report["eager_fp32_vs_bf16_relative_l2"]}))


if __name__ == "__main__":
    main()
