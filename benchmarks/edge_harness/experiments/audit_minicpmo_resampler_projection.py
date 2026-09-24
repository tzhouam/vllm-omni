#!/usr/bin/env python3
"""Replay a measured MiniCPM-o NPU resampler postnorm through the BF16 CPU head."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
SHARD_SHA = "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31"
CASES = ("red", "blue", "third")
GATE = 0.01


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, observed: np.ndarray) -> float:
    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--npu-report", type=Path, required=True)
    parser.add_argument("--npu-output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--original-npu-output", type=Path, required=True)
    parser.add_argument("--original-reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    exported = json.loads(args.export_report.read_text(encoding="utf-8-sig"))
    measured = json.loads(args.npu_report.read_text(encoding="utf-8-sig"))
    if (exported.get("trace") is not True
            or exported["model_revision"] != REVISION
            or exported["shard_sha256"] != SHARD_SHA
            or exported["reference_sha256"] != sha256(args.reference)
            or exported["wrapped_sha256"] != measured["model_sha256"]
            or measured["reference_sha256"] != sha256(args.reference)
            or measured["output_npz_sha256"] != sha256(args.npu_output)
            or measured["node_counts"].get("vitisai") != len(CASES)):
        raise ValueError("traced BF16 resampler artifacts or NPU placement changed")
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if sha256(shard) != SHARD_SHA or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION:
        raise ValueError("MiniCPM-o checkpoint changed")
    with safe_open(shard, framework="pt", device="cpu") as source:
        projection = source.get_tensor("resampler.proj")
    if projection.shape != (4096, 4096) or projection.dtype != torch.bfloat16:
        raise ValueError("BF16 resampler projection contract changed")
    torch.set_num_threads(args.threads)

    with np.load(args.reference, allow_pickle=False) as source:
        reference = {key: np.ascontiguousarray(source[key]) for key in source.files}
    with np.load(args.npu_output, allow_pickle=False) as source:
        npu = {key: np.ascontiguousarray(source[key]) for key in source.files}
    with np.load(args.original_reference, allow_pickle=False) as source:
        original_ref = {key: np.ascontiguousarray(source[key]) for key in source.files}
    with np.load(args.original_npu_output, allow_pickle=False) as source:
        original_npu = {key: np.ascontiguousarray(source[key]) for key in source.files}

    report = {
        "scope": "offline exact BF16 CPU projection replay from measured HX370 NPU postnorm; component only",
        "model_revision": REVISION,
        "shard_sha256": SHARD_SHA,
        "trace_export_report_sha256": sha256(args.export_report),
        "trace_npu_report_sha256": sha256(args.npu_report),
        "trace_npu_output_sha256": sha256(args.npu_output),
        "trace_reference_sha256": sha256(args.reference),
        "original_npu_output_sha256": sha256(args.original_npu_output),
        "original_reference_sha256": sha256(args.original_reference),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cases": {},
    }
    for name in CASES:
        if (not np.array_equal(npu[name], npu[name + "_embedding"])
                or not np.array_equal(npu[name], original_npu[name])
                or not np.array_equal(reference[name], reference[name + "_embedding"])
                or not np.array_equal(reference[name], original_ref[name])):
            raise ValueError(f"{name} trace observation changed the final source or NPU output")
        with torch.inference_mode():
            source_cpu = (
                torch.from_numpy(reference[name + "_postnorm"]).to(torch.bfloat16)
                @ projection
            ).float().numpy()
            hybrid = (
                torch.from_numpy(npu[name + "_postnorm"]).to(torch.bfloat16)
                @ projection
            ).float().numpy()
        if not np.array_equal(source_cpu, reference[name]):
            raise ValueError(f"{name} source BF16 projection replay changed")
        report["cases"][name] = {
            "trace_final_bitwise_equal_to_uninstrumented": True,
            "source_cpu_projection_bitwise_equal": True,
            "npu_postnorm_relative_l2_vs_bf16": relative_l2(
                reference[name + "_postnorm"], npu[name + "_postnorm"]),
            "npu_postnorm_cpu_projection_relative_l2_vs_bf16": relative_l2(
                reference[name], hybrid),
            "npu_final_relative_l2_vs_bf16": relative_l2(reference[name], npu[name]),
            "npu_final_relative_l2_vs_same_postnorm_cpu_projection": relative_l2(
                hybrid, npu[name]),
        }
    train = ("red", "blue")
    observed = np.concatenate(
        [npu[name + "_postnorm"].reshape(-1, 4096) for name in train], axis=0
    ).astype(np.float64)
    expected = np.concatenate(
        [reference[name + "_postnorm"].reshape(-1, 4096) for name in train], axis=0
    ).astype(np.float64)
    observed_mean, expected_mean = observed.mean(axis=0), expected.mean(axis=0)
    scale = np.sum((observed - observed_mean) * (expected - expected_mean), axis=0)
    scale /= np.maximum(np.sum((observed - observed_mean) ** 2, axis=0), 1e-20)
    offset = expected_mean - scale * observed_mean
    report["postnorm_channel_affine_cpu_projection"] = {
        "train": list(train), "held_out": "third", "relative_l2_vs_bf16": {},
    }
    for name in CASES:
        corrected = (
            npu[name + "_postnorm"].astype(np.float64) * scale + offset
        ).astype(np.float32)
        with torch.inference_mode():
            embedding = (
                torch.from_numpy(corrected).to(torch.bfloat16) @ projection
            ).float().numpy()
        report["postnorm_channel_affine_cpu_projection"]["relative_l2_vs_bf16"][name] = (
            relative_l2(reference[name], embedding)
        )
    report["status"] = (
        "postnorm_split_numeric_pass_on_three_synthetic_images"
        if all(case["npu_postnorm_cpu_projection_relative_l2_vs_bf16"] < GATE
               for case in report["cases"].values())
        else "postnorm_split_numeric_failed"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
