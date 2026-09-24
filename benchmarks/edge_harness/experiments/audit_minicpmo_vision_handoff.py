#!/usr/bin/env python3
"""Replay an actually measured MiniCPM-o NPU vision boundary through BF16 CPU layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path


REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
SHARD_SHA = "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31"
SOURCE_SHA = "bd3e7802a78a930c5cd3e8e1a9888ab73e3fa865d6f831e492e169cdd24540be"
CANDIDATE_SHA = "5e3d0ed188db01c643628411725614fd82927bed2a76f93a370085f49c5f2624"
SOURCE_SHA3 = "b68fe07f9ae74d2c4d3a953b43ce5b33e28f86a41f413f857d2e00edf67b9efb"
CANDIDATE_SHA3 = "bba44f82fff3964ff8c3d2470e45f2fb529bc4e049a098a0c9bfef75c0fae110"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(reference, actual) -> dict:
    import numpy as np

    a = reference.astype(np.float64).ravel()
    b = actual.astype(np.float64).ravel()
    error = float(np.linalg.norm(a - b))
    norm = max(float(np.linalg.norm(a)), 1e-12)
    return {
        "relative_l2": error / norm,
        "cosine": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
        "max_abs": float(np.max(np.abs(a - b))),
        "finite": bool(np.isfinite(actual).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--npu-report", type=Path, required=True)
    parser.add_argument("--npu-output", type=Path, required=True)
    parser.add_argument("--full-reference", type=Path)
    parser.add_argument("--boundary-layer", type=int, choices=(3, 4), default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source_sha = SOURCE_SHA3 if args.boundary_layer == 3 else SOURCE_SHA
    candidate_sha = CANDIDATE_SHA3 if args.boundary_layer == 3 else CANDIDATE_SHA
    if not 1 <= args.threads <= 24:
        parser.error("threads must be 1..24")

    import numpy as np
    import torch
    from safetensors import safe_open

    source_report = json.loads(args.npu_report.read_text(encoding="utf-8-sig"))
    if (source_report["source_sha256"] != source_sha
            or source_report["candidate_sha256"] != candidate_sha
            or source_report["fixture_sha256"] != sha256(args.fixture)
            or source_report["npu"]["output_npz_sha256"] != sha256(args.npu_output)
            or source_report["npu"]["node_counts"].get("vitisai", 0) < 1):
        raise ValueError("pinned NPU boundary or placement report changed")
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if sha256(shard) != SHARD_SHA or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION:
        raise ValueError("source vision checkpoint changed")
    with np.load(args.fixture, allow_pickle=False) as data:
        source_at_boundary = np.ascontiguousarray(data["torch_bf16"])
        source_fp32_at_boundary = np.ascontiguousarray(data["torch_fp32"])
    with np.load(args.npu_output, allow_pickle=False) as data:
        npu_at_boundary = np.ascontiguousarray(data["npu"])
        qdq_cpu_at_boundary = np.ascontiguousarray(data["cpu_qdq"])
    if any(value.shape != (1, 1024, 1152) or value.dtype != np.float32
           or not np.isfinite(value).all()
           for value in (source_at_boundary, source_fp32_at_boundary,
                         npu_at_boundary, qdq_cpu_at_boundary)):
        raise ValueError("vision boundary tensor contract changed")

    sys.path.insert(0, str(args.model_dir.resolve()))
    from modeling_navit_siglip import SiglipVisionConfig, SiglipVisionTransformer

    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    vision_config = SiglipVisionConfig(**config["vision_config"])
    vision_config._attn_implementation = "eager"
    torch.set_num_threads(args.threads)
    model = SiglipVisionTransformer(vision_config)
    with safe_open(shard, framework="pt", device="cpu") as source:
        state = {key[4:]: source.get_tensor(key)
                 for key in source.keys() if key.startswith("vpm.")}
    model.load_state_dict(state, strict=True)
    model.eval().to(torch.bfloat16)
    del state

    def suffix(boundary: np.ndarray):
        with torch.inference_mode():
            hidden = torch.from_numpy(boundary).to(torch.bfloat16)
            started = time.perf_counter()
            for layer in model.encoder.layers[args.boundary_layer:]:
                hidden = layer(hidden, None)[0]
            suffix_s = time.perf_counter() - started
            raw = hidden.float().numpy()
            normalized = model.post_layernorm(hidden).float().numpy()
        return raw, normalized, suffix_s

    source_raw, source_norm, source_s = suffix(source_at_boundary)
    fp32_raw, fp32_norm, fp32_s = suffix(source_fp32_at_boundary)
    npu_raw, npu_norm, npu_s = suffix(npu_at_boundary)
    cpu_raw, cpu_norm, cpu_s = suffix(qdq_cpu_at_boundary)
    if args.full_reference is not None:
        with np.load(args.full_reference, allow_pickle=False) as data:
            retained = data["torch_bf16"]
        if not np.array_equal(source_raw, retained):
            raise ValueError("full BF16 reference differs from retained export")
    report = {
        "scope": f"actual AMD NPU {args.boundary_layer}-layer boundary plus unchanged BF16 CPU vision layers {args.boundary_layer + 1}..27 and post-layernorm; component only",
        "model_revision": REVISION,
        "checkpoint_shard_sha256": SHARD_SHA,
        "source_onnx_sha256": source_sha,
        "candidate_sha256": candidate_sha,
        "fixture_sha256": sha256(args.fixture),
        "npu_output_sha256": sha256(args.npu_output),
        "npu_report_sha256": sha256(args.npu_report),
        "input_shape": [1, 1024, 1152],
        "boundary_npu_vs_bf16": compare(source_at_boundary, npu_at_boundary),
        "boundary_qdq_cpu_vs_bf16": compare(source_at_boundary, qdq_cpu_at_boundary),
        "boundary_npu_vs_fp32": compare(source_fp32_at_boundary, npu_at_boundary),
        "after27_npu_vs_bf16": compare(source_raw, npu_raw),
        "after27_fp32_boundary_vs_bf16": compare(source_raw, fp32_raw),
        "after27_qdq_cpu_vs_bf16": compare(source_raw, cpu_raw),
        "post_norm_npu_vs_bf16": compare(source_norm, npu_norm),
        "post_norm_fp32_boundary_vs_bf16": compare(source_norm, fp32_norm),
        "post_norm_qdq_cpu_vs_bf16": compare(source_norm, cpu_norm),
        "cpu_suffix_s": {"source": source_s, "fp32_boundary": fp32_s, "npu_boundary": npu_s,
                         "qdq_cpu_boundary": cpu_s},
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": args.threads,
        "status": "bf16_suffix_replay_measured",
    }
    if not all(report[key]["finite"] for key in
               ("after27_npu_vs_bf16", "post_norm_npu_vs_bf16")):
        raise RuntimeError("nonfinite downstream vision output")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
