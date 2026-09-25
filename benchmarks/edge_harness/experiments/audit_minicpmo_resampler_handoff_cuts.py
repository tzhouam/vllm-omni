#!/usr/bin/env python3
"""Replay measured MiniCPM-o HX370 NPU resampler cuts through exact BF16 CPU suffixes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path

REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
SHARD_SHA = "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31"
KV_CANDIDATE_SHA = "330bbbd0d18caaf6903aa41f836dbeaef57643a8afbaba333df68e3c1e722aeb"
CASES = ("red", "blue", "third")
CUTS = ("projected", "keys", "attended", "postnorm", "embedding")
GATE = 0.01


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference, observed) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--npu-report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--npu-output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import numpy as np
    import torch
    from safetensors import safe_open

    exported = json.loads(args.export_report.read_text(encoding="utf-8-sig"))
    measured = json.loads(args.npu_report.read_text(encoding="utf-8-sig"))
    if (not exported.get("trace") or exported.get("outputs") != list(CUTS)
            or exported["model_revision"] != REVISION
            or exported["shard_sha256"] != SHARD_SHA
            or exported["reference_sha256"] != sha256(args.reference)
            or exported["inputs_sha256"] != sha256(args.inputs)
            or exported["wrapped_sha256"] != measured["model_sha256"]
            or measured["reference_sha256"] != sha256(args.reference)
            or measured["output_npz_sha256"] != sha256(args.npu_output)
            or measured["node_counts"].get("vitisai") != len(CASES)):
        raise ValueError("resampler trace identity or NPU placement changed")
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if sha256(shard) != SHARD_SHA or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION:
        raise ValueError("MiniCPM-o checkpoint changed")
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(args.model_dir.resolve().parent))
    Resampler = importlib.import_module(args.model_dir.name + ".modeling_minicpmo").Resampler
    resampler = Resampler(
        num_queries=config["query_num"],
        embed_dim=config["hidden_size"],
        num_heads=config["hidden_size"] // 128,
        kv_dim=config["vision_config"]["hidden_size"],
        adaptive=True,
    )
    with safe_open(shard, framework="pt", device="cpu") as source:
        state = {key[len("resampler."):]: source.get_tensor(key)
                 for key in source.keys() if key.startswith("resampler.")}
    resampler.load_state_dict(state, strict=True)
    resampler.eval().to(torch.bfloat16)
    torch.set_num_threads(args.threads)
    with np.load(args.reference, allow_pickle=False) as source:
        reference = {key: np.ascontiguousarray(source[key]) for key in source.files}
    with np.load(args.npu_output, allow_pickle=False) as source:
        npu = {key: np.ascontiguousarray(source[key]) for key in source.files}
    with np.load(args.inputs, allow_pickle=False) as source:
        inputs = {key: np.ascontiguousarray(source[key]) for key in source.files}
    if bool(args.candidate_report) != bool(args.candidate_output):
        raise ValueError("candidate report and output must be supplied together")
    candidate = None
    candidate_outputs = None
    candidate_kind = None
    if args.candidate_report:
        candidate = json.loads(args.candidate_report.read_text(encoding="utf-8-sig"))
        candidate_kind = ("npu" if "output_npz_sha256" in candidate else "cpu")
        output_hash_key = "output_npz_sha256" if candidate_kind == "npu" else "candidate_cpu_output_sha256"
        if (candidate[output_hash_key] != sha256(args.candidate_output)
                or candidate["inputs_sha256"] != sha256(args.inputs)
                or candidate["reference_sha256"] != sha256(args.reference)
                or candidate["candidate_sha256"] != KV_CANDIDATE_SHA):
            raise ValueError("A16W8 candidate output changed")
        if candidate_kind == "npu" and candidate["node_counts"].get("vitisai") != len(CASES):
            raise ValueError("A16W8 candidate did not place all calls on NPU")
        with np.load(args.candidate_output, allow_pickle=False) as source:
            candidate_outputs = {name: np.ascontiguousarray(source[name]) for name in CASES}

    query = resampler.ln_q(resampler.query).unsqueeze(1)
    position = resampler.pos_embed[:32, :32, :].reshape(1024, -1).to(torch.bfloat16).unsqueeze(1)
    mask = torch.zeros((1, 1024), dtype=torch.bool)

    def suffix(cut: str, value):
        tensor = torch.from_numpy(value).to(torch.bfloat16)
        if cut == "embedding":
            return value
        if cut == "projected":
            keys = resampler.ln_kv(tensor).permute(1, 0, 2)
        elif cut == "keys":
            keys = tensor.permute(1, 0, 2)
        else:
            keys = None
        if keys is not None:
            attended = resampler.attn(query, keys + position, keys, key_padding_mask=mask)[0]
            tensor = attended.permute(1, 0, 2)
        if cut in ("projected", "keys", "attended"):
            tensor = resampler.ln_post(tensor)
        return (tensor @ resampler.proj).float().numpy()

    report = {
        "scope": "offline exact BF16 CPU suffix replay from measured HX370 NPU resampler cuts; component only",
        "model_revision": REVISION,
        "shard_sha256": SHARD_SHA,
        "export_report_sha256": sha256(args.export_report),
        "npu_report_sha256": sha256(args.npu_report),
        "reference_sha256": sha256(args.reference),
        "npu_output_sha256": sha256(args.npu_output),
        "inputs_sha256": sha256(args.inputs),
        "candidate_report_sha256": sha256(args.candidate_report) if args.candidate_report else None,
        "candidate_output_sha256": sha256(args.candidate_output) if args.candidate_output else None,
        "candidate_kind": candidate_kind,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cases": {},
    }
    with torch.inference_mode():
        for name in CASES:
            target = reference[name]
            if (not np.array_equal(target, reference[name + "_embedding"])
                    or not np.array_equal(npu[name], npu[name + "_embedding"])):
                raise ValueError(f"{name} traced final output identity changed")
            cuts = {}
            for cut in CUTS:
                source_value = reference[name + "_" + cut]
                measured_value = npu[name + "_" + cut]
                if source_value.shape != measured_value.shape or source_value.dtype != np.float32:
                    raise ValueError(f"{name} {cut} tensor contract changed")
                source_replay = suffix(cut, source_value)
                measured_replay = suffix(cut, measured_value)
                if not np.array_equal(source_replay, target):
                    raise ValueError(f"{name} {cut} source BF16 suffix replay changed")
                cuts[cut] = {
                    "boundary_relative_l2": relative_l2(source_value, measured_value),
                    "cpu_suffix_embedding_relative_l2": relative_l2(target, measured_replay),
                    "finite": bool(np.isfinite(measured_replay).all()),
                    "source_replay_bitwise_equal": True,
                }
            report["cases"][name] = cuts
            hidden = torch.from_numpy(inputs[name])
            if hidden.shape != (1, 1024, 1152) or hidden.dtype != torch.float32:
                raise ValueError(f"{name} resampler input contract changed")
            source_projected = resampler.kv_proj(hidden.to(torch.bfloat16)).float().numpy()
            if not np.array_equal(source_projected, reference[name + "_projected"]):
                raise ValueError(f"{name} source KV projection input is not exact")
            for dtype in (torch.float32, torch.float16):
                projected = torch.nn.functional.linear(
                    hidden.to(dtype), resampler.kv_proj.weight.to(dtype),
                    resampler.kv_proj.bias.to(dtype) if resampler.kv_proj.bias is not None else None,
                ).float().numpy()
                report["cases"][name]["cpu_" + str(dtype).replace("torch.", "") + "_kv_projection"] = {
                    "projected_relative_l2": relative_l2(source_projected, projected),
                    "cpu_suffix_embedding_relative_l2": relative_l2(target, suffix("projected", projected)),
                }
            if candidate_outputs is not None:
                projected = candidate_outputs[name]
                if projected.shape != (1, 1024, 4096) or projected.dtype != np.float32:
                    raise ValueError(f"{name} A16W8 projected output contract changed")
                report["cases"][name]["a16w8_" + candidate_kind + "_kv_projection"] = {
                    "projected_relative_l2": relative_l2(source_projected, projected),
                    "cpu_suffix_embedding_relative_l2": relative_l2(target, suffix("projected", projected)),
                }
    report["qualifying_cuts_on_three_synthetic_images"] = [
        cut for cut in CUTS
        if all(report["cases"][name][cut]["finite"]
               and report["cases"][name][cut]["cpu_suffix_embedding_relative_l2"] < GATE
               for name in CASES)
    ]
    report["status"] = ("at_least_one_cut_numeric_pass" if report["qualifying_cuts_on_three_synthetic_images"]
                        else "no_tested_cut_numeric_pass")
    if candidate_outputs is not None:
        key = "a16w8_" + candidate_kind + "_kv_projection"
        report["candidate_kv_final_embedding_gate_pass"] = all(
            report["cases"][name][key]["cpu_suffix_embedding_relative_l2"] < GATE
            for name in CASES
        )
        if candidate_kind == "npu" and report["candidate_kv_final_embedding_gate_pass"]:
            report["status"] = "npu_kv_cpu_suffix_numeric_pass_on_three_synthetic_images"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
