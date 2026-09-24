#!/usr/bin/env python3
"""Test whether a cheap affine map repairs the measured late SigLIP NPU boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REFERENCE_SHA = "ea31d28f9c59e96a84aef12695cd14c4d4a552b5f1b1c019370cc42a5631a96f"
NPU_SHA = "bbe413e06f9ab38793cbaf328299ce44817339f05450e7b2d6444c6ab5452b8a"
SHARD_SHA = "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31"
REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
LAYER_INPUTS_SHA = "61a28d3cae38413fe2feb7161118fe7421f1a452bc2fa2b523c34a65007c339d"
CASES = ("red", "blue", "third")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, observed: np.ndarray) -> float:
    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def fit_affine(reference: np.ndarray, observed: np.ndarray, *, per_channel: bool):
    x = observed.reshape(-1, observed.shape[-1]).astype(np.float64)
    y = reference.reshape(-1, reference.shape[-1]).astype(np.float64)
    if not per_channel:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x_center = x - x_mean
    y_center = y - y_mean
    scale = np.sum(x_center * y_center, axis=0) / np.sum(x_center * x_center, axis=0)
    offset = y_mean - scale * x_mean
    return scale, offset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--npu-output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layer-inputs", type=Path, required=True)
    parser.add_argument("--fp32-export-report", type=Path)
    parser.add_argument("--fp32-reference", type=Path)
    parser.add_argument("--fp32-npu-report", type=Path)
    parser.add_argument("--fp32-npu-output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if (sha256(args.reference) != REFERENCE_SHA
            or sha256(args.npu_output) != NPU_SHA
            or sha256(args.layer_inputs) != LAYER_INPUTS_SHA):
        raise ValueError("late27 measured attention artifacts changed")
    with np.load(args.reference, allow_pickle=False) as source:
        refs = {name: np.ascontiguousarray(source[f"{name}_attention"]) for name in CASES}
    with np.load(args.npu_output, allow_pickle=False) as source:
        measured = {name: np.ascontiguousarray(source[f"{name}_attention"]) for name in CASES}
    with np.load(args.layer_inputs, allow_pickle=False) as source:
        layer_inputs = {name: np.ascontiguousarray(source[name]) for name in CASES}
    if any(refs[name].shape != (1, 1024, 1152)
           or refs[name].dtype != np.float32
           or measured[name].shape != refs[name].shape
           or layer_inputs[name].shape != refs[name].shape
           or not np.isfinite(refs[name]).all()
           or not np.isfinite(measured[name]).all()
           for name in CASES):
        raise ValueError("late27 measured attention shape or finiteness changed")

    import importlib
    import torch
    from safetensors import safe_open

    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if sha256(shard) != SHARD_SHA or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION:
        raise ValueError("MiniCPM-o checkpoint changed")
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
    sys.path.insert(0, str(args.model_dir.resolve().parent))
    Resampler = importlib.import_module(args.model_dir.name + ".modeling_minicpmo").Resampler
    resampler = Resampler(
        num_queries=config["query_num"], embed_dim=config["hidden_size"],
        num_heads=config["hidden_size"] // 128, kv_dim=vision_config.hidden_size,
        adaptive=True,
    )
    with safe_open(shard, framework="pt", device="cpu") as source:
        resampler_state = {key[len("resampler."):]: source.get_tensor(key)
                           for key in source.keys() if key.startswith("resampler.")}
    resampler.load_state_dict(resampler_state, strict=True)
    resampler.eval().to(torch.bfloat16)
    layer = model.encoder.layers[26]
    tgt_sizes = torch.tensor([[32, 32]], dtype=torch.long)

    def downstream(attention: np.ndarray):
        with torch.inference_mode():
            hidden = torch.from_numpy(attention).to(torch.bfloat16)
            full = hidden + layer.mlp(layer.layer_norm2(hidden))
            normalized = model.post_layernorm(full)
            embedding = resampler(normalized, tgt_sizes)
        return normalized.float().numpy(), embedding.float().numpy()

    source_embeddings = {}
    source_full_outputs = {}
    source_norms = {}
    for name in CASES:
        with torch.inference_mode():
            source_full = torch.from_numpy(refs[name].copy()).to(torch.bfloat16)
            source_full = source_full + layer.mlp(layer.layer_norm2(source_full))
            source_norm = model.post_layernorm(source_full)
            source_full_outputs[name] = source_full.float().numpy()
            source_norms[name] = source_norm.float().numpy()
            source_embeddings[name] = resampler(source_norm, tgt_sizes).float().numpy()

    # A coarser boundary would send the complete BF16 SigLIP output to the NPU
    # resampler. Check the unavoidable FP32-conversion error before compiling it.
    fp32_resampler = copy.deepcopy(resampler).float().eval()
    fp32_resampler_control = {}
    for name in CASES:
        with torch.inference_mode():
            embedding = fp32_resampler(
                torch.from_numpy(source_norms[name]).float(), tgt_sizes
            ).float().numpy()
        fp32_resampler_control[name] = {
            "resampler_relative_l2_vs_bf16": relative_l2(source_embeddings[name], embedding)
        }

    report = {
        "scope": "offline affine correction fit on measured NPU attention residuals; no new NPU execution",
        "reference_sha256": REFERENCE_SHA,
        "npu_output_sha256": NPU_SHA,
        "layer_inputs_sha256": LAYER_INPUTS_SHA,
        "shape": [1, 1024, 1152],
        "cases": {},
    }
    report["cases"]["fp32_resampler_cpu_control"] = fp32_resampler_control
    fp32_layer = copy.deepcopy(layer).float().eval()
    report["cases"]["fp32_last_layer_cpu_control"] = {}
    for name in CASES:
        with torch.inference_mode():
            fp32_full = fp32_layer(torch.from_numpy(layer_inputs[name]), None)[0]
            fp32_full_bf16 = fp32_full.to(torch.bfloat16)
            fp32_norm = model.post_layernorm(fp32_full_bf16)
            fp32_embedding = resampler(fp32_norm, tgt_sizes)
        report["cases"]["fp32_last_layer_cpu_control"][name] = {
            "full_relative_l2_vs_bf16": relative_l2(
                source_full_outputs[name], fp32_full.float().numpy()),
            "postnorm_relative_l2_vs_bf16": relative_l2(
                source_norms[name], fp32_norm.float().numpy()),
            "resampler_relative_l2_vs_bf16": relative_l2(
                source_embeddings[name], fp32_embedding.float().numpy()),
        }
    if args.fp32_npu_output is not None:
        required = (args.fp32_export_report, args.fp32_reference, args.fp32_npu_report)
        if any(path is None for path in required):
            raise ValueError("FP32 NPU output requires export, reference and native reports")
        exported = json.loads(args.fp32_export_report.read_text(encoding="utf-8-sig"))
        observed = json.loads(args.fp32_npu_report.read_text(encoding="utf-8-sig"))
        if (exported["compute_dtype"] != "fp32"
                or exported["layer_index"] != 26
                or exported["layer_inputs_sha256"] != LAYER_INPUTS_SHA
                or exported["wrapped_sha256"] != observed["model_sha256"]
                or exported["reference_sha256"] != sha256(args.fp32_reference)
                or observed["reference_sha256"] != sha256(args.fp32_reference)
                or observed["output_npz_sha256"] != sha256(args.fp32_npu_output)
                or observed["node_counts"].get("vitisai", 0) < len(CASES)):
            raise ValueError("FP32 NPU artifact, placement or reference changed")
        with np.load(args.fp32_reference, allow_pickle=False) as source:
            fp32_reference = {key: np.ascontiguousarray(source[key]) for key in source.files}
        with np.load(args.fp32_npu_output, allow_pickle=False) as source:
            fp32_npu = {key: np.ascontiguousarray(source[key]) for key in source.files}
        report["fp32_npu_provenance"] = {
            "export_report_sha256": sha256(args.fp32_export_report),
            "reference_sha256": sha256(args.fp32_reference),
            "npu_report_sha256": sha256(args.fp32_npu_report),
            "npu_output_sha256": sha256(args.fp32_npu_output),
            "vitisai_node_events": observed["node_counts"]["vitisai"],
        }
        report["cases"]["fp32_last_layer_npu"] = {}
        for name in CASES:
            attention = fp32_npu[name + "_attention"]
            full = fp32_npu[name + "_full"]
            if (attention.shape != (1, 1024, 1152) or full.shape != attention.shape
                    or not np.isfinite(attention).all() or not np.isfinite(full).all()):
                raise ValueError("FP32 NPU output contract changed")
            _, hybrid_embedding = downstream(attention)
            with torch.inference_mode():
                normalized = model.post_layernorm(torch.from_numpy(full).to(torch.bfloat16))
                full_embedding = resampler(normalized, tgt_sizes).float().numpy()
            report["cases"]["fp32_last_layer_npu"][name] = {
                "attention_relative_l2_vs_fp32_torch": relative_l2(
                    fp32_reference[name + "_attention"], attention),
                "full_relative_l2_vs_fp32_torch": relative_l2(
                    fp32_reference[name + "_full"], full),
                "attention_cpu_bf16_mlp_resampler_relative_l2_vs_bf16": relative_l2(
                    source_embeddings[name], hybrid_embedding),
                "full_resampler_relative_l2_vs_bf16": relative_l2(
                    source_embeddings[name], full_embedding),
            }
        fp32_scale, fp32_offset = fit_affine(
            np.concatenate([refs[name] for name in ("red", "blue")], axis=1),
            np.concatenate([fp32_npu[name + "_attention"] for name in ("red", "blue")], axis=1),
            per_channel=True,
        )
        report["cases"]["fp32_npu_channel_affine"] = {
            "train": ["red", "blue"],
            "held_out": "third",
            "relative_l2": {},
        }
        for name in CASES:
            corrected_attention = (
                fp32_npu[name + "_attention"].astype(np.float64) * fp32_scale + fp32_offset
            ).astype(np.float32)
            _, corrected_embedding = downstream(corrected_attention)
            report["cases"]["fp32_npu_channel_affine"]["relative_l2"][name] = {
                "attention_vs_bf16_torch": relative_l2(refs[name], corrected_attention),
                "resampler_vs_bf16_torch": relative_l2(
                    source_embeddings[name], corrected_embedding),
            }
    for per_channel in (False, True):
        mode = "channel_affine" if per_channel else "global_affine"
        train = ("red", "blue")
        scale, offset = fit_affine(
            np.concatenate([refs[name] for name in train], axis=1),
            np.concatenate([measured[name] for name in train], axis=1),
            per_channel=per_channel,
        )
        report["cases"][mode] = {
            "train": list(train),
            "held_out": "third",
            "scale_min_max": [float(np.min(scale)), float(np.max(scale))],
            "offset_min_max": [float(np.min(offset)), float(np.max(offset))],
            "relative_l2": {},
        }
        for name in CASES:
            corrected = (measured[name].astype(np.float64) * scale + offset).astype(np.float32)
            _, uncorrected_embedding = downstream(measured[name])
            corrected_norm, corrected_embedding = downstream(corrected)
            report["cases"][mode]["relative_l2"][name] = {
                "attention_uncorrected": relative_l2(refs[name], measured[name]),
                "attention_corrected": relative_l2(refs[name], corrected),
                "postnorm_corrected": relative_l2(source_norms[name], corrected_norm),
                "resampler_uncorrected": relative_l2(source_embeddings[name], uncorrected_embedding),
                "resampler_corrected": relative_l2(source_embeddings[name], corrected_embedding),
            }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["cases"], indent=2))


if __name__ == "__main__":
    main()
