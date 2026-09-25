#!/usr/bin/env python3
"""Profile MiniCPM-o's NPU KV projection and BF16 CPU resampler suffix through Omni.

This fixed-shape component handoff is not a complete image or multimodal request.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

from vllm_omni.edge.hardware_probe import load_profile
from vllm_omni.edge.local.capabilities import FORMAT_ONNX_A16W8, enumerate_devices
from vllm_omni.edge.local.external.stage import ExternalStage, PlacementRefused, plan_external_stage
from vllm_omni.edge.local.manifest import build_graph_artifact

REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
SHARD_SHA = "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31"
GRAPH_SHA = "330bbbd0d18caaf6903aa41f836dbeaef57643a8afbaba333df68e3c1e722aeb"
INPUTS_SHA = "ab71cc3a6b8461c99cdf9e458b2e8b99cc092dd6adc3d49f92d643ec4b8693de"
REFERENCE_SHA = "b155827adf49819355193ca743ae03b5dee2c05c857e70bef21589094ce6d368"
CASES = ("red", "blue", "third")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def nearest(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def relative_l2(reference, observed):
    a, b = reference.astype(np.float64), observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--order", choices=("split-first", "cpu-first"), default="split-first")
    parser.add_argument("--worker-peak-rss-hint-bytes", type=int)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    checkout = Path(__file__).resolve().parents[3]
    stage_source = Path(inspect.getfile(plan_external_stage)).resolve()
    if stage_source != checkout / "vllm_omni/edge/local/external/stage.py":
        raise RuntimeError("Omni external stage is not loaded from this checkout")
    if (sha256(args.graph) != GRAPH_SHA or sha256(args.inputs) != INPUTS_SHA
            or sha256(args.reference) != REFERENCE_SHA):
        raise ValueError("pinned KV graph, input or reference changed")
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if sha256(shard) != SHARD_SHA or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION:
        raise ValueError("MiniCPM-o checkpoint changed")
    with np.load(args.inputs, allow_pickle=False) as data:
        hidden = {name: np.ascontiguousarray(data[name]) for name in CASES}
    with np.load(args.reference, allow_pickle=False) as data:
        target = {name: np.ascontiguousarray(data[name]) for name in CASES}
    if any(hidden[name].shape != (1, 1024, 1152) or target[name].shape != (1, 64, 4096)
           for name in CASES):
        raise ValueError("resampler input/output contract changed")

    import torch
    from safetensors import safe_open

    torch.set_num_threads(8)
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(args.model_dir.resolve().parent))
    Resampler = importlib.import_module(args.model_dir.name + ".modeling_minicpmo").Resampler
    resampler = Resampler(
        num_queries=config["query_num"], embed_dim=config["hidden_size"],
        num_heads=config["hidden_size"] // 128,
        kv_dim=config["vision_config"]["hidden_size"], adaptive=True,
    )
    with safe_open(shard, framework="pt", device="cpu") as source:
        state = {key[len("resampler."):]: source.get_tensor(key)
                 for key in source.keys() if key.startswith("resampler.")}
    resampler.load_state_dict(state, strict=True)
    resampler.eval().to(torch.bfloat16)
    query = resampler.ln_q(resampler.query).unsqueeze(1)
    position = resampler.pos_embed[:32, :32, :].reshape(1024, -1).to(torch.bfloat16).unsqueeze(1)
    mask = torch.zeros((1, 1024), dtype=torch.bool)

    def cpu_suffix(projected):
        projected_bf16 = torch.from_numpy(projected).to(torch.bfloat16)
        keys = resampler.ln_kv(projected_bf16).permute(1, 0, 2)
        attended = resampler.attn(query, keys + position, keys, key_padding_mask=mask)[0]
        postnorm = resampler.ln_post(attended.permute(1, 0, 2))
        return (postnorm @ resampler.proj).float().numpy()

    tgt_sizes = torch.tensor([[32, 32]], dtype=torch.long)
    with torch.inference_mode():
        for name in CASES:
            source_output = resampler(torch.from_numpy(hidden[name]).to(torch.bfloat16), tgt_sizes)
            if not np.array_equal(source_output.float().numpy(), target[name]):
                raise ValueError(f"{name} full BF16 CPU resampler changed")

    artifact = build_graph_artifact(
        args.graph, fmt=FORMAT_ONNX_A16W8, opset=21,
        source_model="openbmb/MiniCPM-o-4_5", source_revision=REVISION,
        component="resampler_kv_projection_32x32",
        exporter="probe_minicpmo_resampler_kv_npu.py quantize",
        calibration={"inputs_sha256": INPUTS_SHA, "calibration_cases": ["red", "blue"]},
        parity={"npu_cpu_suffix_three_image_relative_l2_max": 0.00925311117731196,
                "npu_report": "resampler_kv_a16w8_npu_report.json"},
    )
    plan = plan_external_stage(
        artifact, enumerate_devices(load_profile(use_torch=False)),
        require="npu:amd", min_fraction_on_target=1 / 3,
        worker_peak_rss_hint_bytes=args.worker_peak_rss_hint_bytes,
    )
    if not plan.admitted:
        raise RuntimeError(plan.summary())
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    cpu_full_times = []
    with ExternalStage(plan) as stage:
        try:
            placement = stage.open({"hidden": hidden["red"]}, profile_dir=args.profile_dir)
        except PlacementRefused as exc:
            report = {
                "scope": "MiniCPM-o KV Omni worker admission refusal; no measured requests",
                "graph_sha256": GRAPH_SHA, "inputs_sha256": INPUTS_SHA,
                "platform": platform.platform(),
                "planner_budget_bytes": plan.budget_bytes,
                "worker_peak_rss_hint_bytes": args.worker_peak_rss_hint_bytes,
                "refusal": exc.refusal.to_dict(),
                "load_report": exc.report.to_dict() if exc.report else None,
                "status": "refused",
            }
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            raise
        if placement.target_nodes < 1:
            raise RuntimeError("NPU placement not verified")
        with torch.inference_mode():
            for index in range(args.repeats):
                name = CASES[index % len(CASES)]
                if args.order == "cpu-first":
                    started = time.perf_counter()
                    cpu_observed = resampler(torch.from_numpy(hidden[name]).to(torch.bfloat16), tgt_sizes)
                    cpu_full_times.append(time.perf_counter() - started)
                    if not np.array_equal(cpu_observed.float().numpy(), target[name]):
                        raise ValueError("BF16 CPU baseline changed during timing")
                output, timing = stage.run({"hidden": hidden[name]})
                projected = output["projected"]
                suffix_started = time.perf_counter()
                observed = cpu_suffix(projected)
                suffix_s = time.perf_counter() - suffix_started
                samples.append({
                    "case": name, **timing.to_dict(), "cpu_suffix_s": suffix_s,
                    "split_component_s": timing.round_trip_s + suffix_s,
                    "embedding_relative_l2": relative_l2(target[name], observed),
                    "finite": bool(np.isfinite(observed).all()),
                })
                if args.order == "split-first":
                    started = time.perf_counter()
                    cpu_observed = resampler(torch.from_numpy(hidden[name]).to(torch.bfloat16), tgt_sizes)
                    cpu_full_times.append(time.perf_counter() - started)
                    if not np.array_equal(cpu_observed.float().numpy(), target[name]):
                        raise ValueError("BF16 CPU baseline changed during timing")
        worker_stats = stage.stats()

    metrics = {
        key: {"p50": nearest([sample[key] for sample in samples], 0.50),
              "p95": nearest([sample[key] for sample in samples], 0.95)}
        for key in ("worker_s", "transport_s", "round_trip_s", "cpu_suffix_s", "split_component_s")
    }
    metrics["cpu_full_resampler_s"] = {"p50": nearest(cpu_full_times, 0.50),
                                        "p95": nearest(cpu_full_times, 0.95)}
    report = {
        "scope": "fixed-shape MiniCPM-o KV NPU+BF16 CPU resampler component handoff through Omni; not full image request",
        "graph_sha256": GRAPH_SHA, "inputs_sha256": INPUTS_SHA,
        "reference_sha256": REFERENCE_SHA, "model_revision": REVISION,
        "platform": platform.platform(), "torch": torch.__version__,
        "omni_stage_source": str(stage_source),
        "concurrency": 1,
        "measurement_order": f"{args.order}, interleaved per case while NPU worker resident",
        "warmup": "one profiled stage-open inference and three exact BF16 CPU controls",
        "repeats": args.repeats, "placement": placement.to_dict(),
        "planner_budget_bytes": plan.budget_bytes,
        "worker_peak_rss_hint_bytes": args.worker_peak_rss_hint_bytes,
        "worker_peak_rss_bytes": worker_stats.get("peak_rss_bytes"),
        "worker_stats": worker_stats, "samples": samples,
        "cpu_full_resampler_call_s": cpu_full_times, "metrics": metrics,
        "gate": "provisional final vision-embedding relative L2 <1% on three synthetic images",
        "status": ("component_numeric_pass" if all(
            sample["finite"] and sample["embedding_relative_l2"] < 0.01 for sample in samples)
                   else "component_numeric_fail"),
        "power_thermal": "not measured",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics,
                      "max_embedding_relative_l2": max(s["embedding_relative_l2"] for s in samples),
                      "worker_peak_rss_bytes": report["worker_peak_rss_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
