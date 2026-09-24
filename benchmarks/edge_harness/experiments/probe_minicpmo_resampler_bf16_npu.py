#!/usr/bin/env python3
"""Probe the complete BF16 MiniCPM-o resampler on a real HX370 NPU.

The fixed 32x32-patch inputs are post-layernorm tensors from the unchanged
BF16 SigLIP path. FP32 external casts work around the Windows ORT Python
binding's BF16 NumPy limitation; the graph itself keeps BF16 computation.
This is a component experiment, not an Omni end-to-end request.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path


REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
SHARD_SHA = "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31"
LATE27_REFERENCE_SHA = "ea31d28f9c59e96a84aef12695cd14c4d4a552b5f1b1c019370cc42a5631a96f"
CASES = ("red", "blue", "third")
THRESHOLD = 0.01


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference, observed) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def export(args: argparse.Namespace) -> None:
    import numpy as np
    import onnx
    import torch
    from onnx import TensorProto, helper
    from safetensors import safe_open

    if sha256(args.late27_reference) != LATE27_REFERENCE_SHA:
        raise ValueError("BF16 SigLIP last-layer reference changed")
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if sha256(shard) != SHARD_SHA or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION:
        raise ValueError("MiniCPM-o checkpoint changed")
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.model_dir.resolve()))
    from modeling_navit_siglip import SiglipVisionConfig, SiglipVisionTransformer

    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    vision_config = SiglipVisionConfig(**config["vision_config"])
    vision_config._attn_implementation = "eager"
    vision = SiglipVisionTransformer(vision_config)
    with safe_open(shard, framework="pt", device="cpu") as source:
        postnorm_state = {
            key[len("vpm.post_layernorm."):]: source.get_tensor(key)
            for key in source.keys() if key.startswith("vpm.post_layernorm.")
        }
    vision.post_layernorm.load_state_dict(postnorm_state, strict=True)
    vision.post_layernorm.eval().to(torch.bfloat16)
    postnorm = vision.post_layernorm
    del vision

    sys.path.insert(0, str(args.model_dir.resolve().parent))
    Resampler = importlib.import_module(args.model_dir.name + ".modeling_minicpmo").Resampler
    resampler = Resampler(
        num_queries=config["query_num"],
        embed_dim=config["hidden_size"],
        num_heads=config["hidden_size"] // 128,
        kv_dim=vision_config.hidden_size,
        adaptive=True,
    )
    with safe_open(shard, framework="pt", device="cpu") as source:
        state = {
            key[len("resampler."):]: source.get_tensor(key)
            for key in source.keys() if key.startswith("resampler.")
        }
    resampler.load_state_dict(state, strict=True)
    resampler.eval().to(torch.bfloat16)
    del state

    class FixedResampler(torch.nn.Module):
        def __init__(self, module, trace):
            super().__init__()
            self.module = module
            self.trace = trace
            self.register_buffer("key_padding_mask", torch.zeros((1, 1024), dtype=torch.bool))

        def forward(self, hidden):
            # Equivalent to the source for batch=1, tgt_sizes=[[32,32]].
            # Its dynamic pad_sequence has no ONNX symbolic despite no padding
            # being needed at this fixed 1024-patch bucket.
            module = self.module
            if not self.trace:
                # Keep the original export path byte-for-byte reproducible.
                keys = module.ln_kv(module.kv_proj(hidden)).permute(1, 0, 2)
                query = module.ln_q(module.query).unsqueeze(1)
                position = module.pos_embed[:32, :32, :].reshape(1024, -1)
                position = position.to(hidden.dtype).unsqueeze(1)
                attended = module.attn(
                    query, keys + position, keys,
                    key_padding_mask=self.key_padding_mask,
                )[0]
                return module.ln_post(attended.permute(1, 0, 2)) @ module.proj
            projected = module.kv_proj(hidden)
            keys = module.ln_kv(projected).permute(1, 0, 2)
            query = module.ln_q(module.query).unsqueeze(1)
            position = module.pos_embed[:32, :32, :].reshape(1024, -1)
            position = position.to(hidden.dtype).unsqueeze(1)
            attended = module.attn(
                query, keys + position, keys,
                key_padding_mask=self.key_padding_mask,
            )[0]
            attended = attended.permute(1, 0, 2)
            postnorm = module.ln_post(attended)
            embedding = postnorm @ module.proj
            if self.trace:
                return projected, keys.permute(1, 0, 2), attended, postnorm, embedding
            return embedding

    fixed = FixedResampler(resampler, args.trace).eval()
    output_names = ("projected", "keys", "attended", "postnorm", "embedding") if args.trace else ("embedding",)
    normalized = {}
    references = {}
    with np.load(args.late27_reference, allow_pickle=False) as source:
        with torch.inference_mode():
            for name in CASES:
                full = np.ascontiguousarray(source[name + "_full"])
                if full.shape != (1, 1024, 1152) or full.dtype != np.float32:
                    raise ValueError(f"{name} source layer shape or dtype changed")
                # post_layernorm is loaded independently from the pinned shard.
                hidden = postnorm(torch.from_numpy(full).to(torch.bfloat16))
                output = fixed(hidden)
                source_output = resampler(hidden, torch.tensor([[32, 32]], dtype=torch.long))
                final_output = output[-1] if args.trace else output
                if not torch.equal(final_output, source_output):
                    raise ValueError(f"fixed-shape wrapper changed {name} BF16 resampler output")
                normalized[name] = np.ascontiguousarray(hidden.float().numpy())
                references[name] = np.ascontiguousarray(final_output.float().numpy())
                if args.trace:
                    for stage, tensor in zip(output_names, output):
                        references[name + "_" + stage] = np.ascontiguousarray(tensor.float().numpy())
    if any(references[name].shape != (1, 64, config["hidden_size"])
           or not np.isfinite(references[name]).all() for name in CASES):
        raise ValueError("BF16 resampler reference contract failed")

    args.inputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.inputs, **normalized)
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.reference, **references)
    args.direct_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            fixed, (torch.from_numpy(normalized["red"]).to(torch.bfloat16),),
            str(args.direct_output), input_names=["hidden_bf16"],
            output_names=[stage + "_bf16" for stage in output_names], opset_version=17,
            dynamo=False, do_constant_folding=True,
        )
    export_s = time.perf_counter() - started
    graph = onnx.load(args.direct_output)
    onnx.checker.check_model(graph)
    if (graph.graph.input[0].type.tensor_type.elem_type != TensorProto.BFLOAT16
            or len(graph.graph.output) != len(output_names)
            or any(output.type.tensor_type.elem_type != TensorProto.BFLOAT16
                   for output in graph.graph.output)):
        raise ValueError("export did not retain BF16 boundaries")
    graph.graph.input[0].name = "hidden_fp32"
    graph.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
    graph.graph.node.insert(0, helper.make_node(
        "Cast", ["hidden_fp32"], ["hidden_bf16"],
        to=TensorProto.BFLOAT16, name="input_to_bf16"))
    for output in graph.graph.output:
        original = output.name
        output.name = original.replace("_bf16", "_fp32")
        output.type.tensor_type.elem_type = TensorProto.FLOAT
        graph.graph.node.append(helper.make_node(
            "Cast", [original], [output.name],
            to=TensorProto.FLOAT,
            name=original + "_to_fp32" if args.trace else "output_to_fp32"))
    onnx.checker.check_model(graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, args.output)
    report = {
        "scope": "pinned BF16 MiniCPM-o postnorm-to-resampler graph; component only",
        "model_revision": REVISION,
        "shard_sha256": SHARD_SHA,
        "late27_reference_sha256": LATE27_REFERENCE_SHA,
        "inputs_sha256": sha256(args.inputs),
        "reference_sha256": sha256(args.reference),
        "direct_sha256": sha256(args.direct_output),
        "wrapped_sha256": sha256(args.output),
        "input_shape": [1, 1024, 1152],
        "output_shape": [1, 64, config["hidden_size"]],
        "internal_precision": "BF16",
        "external_precision": "FP32 casts",
        "outputs": list(output_names),
        "trace": bool(args.trace),
        "export_s": export_s,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort

    exported = json.loads(args.export_report.read_text(encoding="utf-8-sig"))
    if (sha256(args.model) != exported["wrapped_sha256"]
            or sha256(args.inputs) != exported["inputs_sha256"]
            or sha256(args.reference) != exported["reference_sha256"]):
        raise ValueError("resampler graph, inputs or reference changed")
    with np.load(args.inputs, allow_pickle=False) as data:
        inputs = {name: np.ascontiguousarray(data[name]) for name in CASES}
    with np.load(args.reference, allow_pickle=False) as data:
        reference = {name: np.ascontiguousarray(data[name]) for name in data.files}
    if any(inputs[name].shape != (1, 1024, 1152)
           or inputs[name].dtype != np.float32
           or reference[name].shape != (1, 64, 4096)
           for name in CASES):
        raise ValueError("resampler input or output shape changed")

    ep_dir = args.ep_dir.resolve(strict=True)
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library(
        "vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
    devices = [device for device in ort.get_ep_devices()
               if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
    if not devices:
        raise RuntimeError("VitisAI NPU unavailable")
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    options.enable_profiling = True
    options.profile_file_prefix = str(args.profile_prefix)
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options)
    load_s = time.perf_counter() - started
    outputs = {}
    cases = {}
    for name in CASES:
        started = time.perf_counter()
        result = session.run(None, {"hidden_fp32": inputs[name]})
        call_s = time.perf_counter() - started
        if len(result) != len(exported.get("outputs", ["embedding"])):
            raise ValueError("instrumented resampler output count changed")
        for stage, observed in zip(exported.get("outputs", ["embedding"]), result):
            outputs[name + "_" + stage] = observed
        observed = result[-1]
        outputs[name] = observed
        cases[name] = {
            "one_call_s": call_s,
            "relative_l2_vs_bf16_torch": relative_l2(reference[name], observed),
            "finite": bool(np.isfinite(observed).all()),
        }
        if exported.get("trace"):
            cases[name]["stage_relative_l2_vs_bf16_torch"] = {
                stage: relative_l2(reference[name + "_" + stage], value)
                for stage, value in zip(exported["outputs"], result)
            }
    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text(encoding="utf-8-sig"))
    counts = {}
    for event in events:
        if event.get("cat") == "Node":
            provider = (event.get("args") or {}).get("provider")
            counts[provider] = counts.get(provider, 0) + 1
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **outputs)
    report = {
        "scope": "pinned BF16 MiniCPM-o resampler on HX370 NPU; component only",
        "model_sha256": sha256(args.model),
        "inputs_sha256": sha256(args.inputs),
        "reference_sha256": sha256(args.reference),
        "output_npz_sha256": sha256(args.output_npz),
        "profile_sha256": sha256(profile),
        "profile_path": str(profile),
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "providers": session.get_providers(),
        "node_counts": counts,
        "load_s": load_s,
        "cases": cases,
        "status": (
            "component_numeric_pass_on_three_synthetic_images"
            if counts.get("vitisai", 0) >= len(CASES)
            and all(case["finite"] and case["relative_l2_vs_bf16_torch"] < THRESHOLD
                    for case in cases.values())
            else "component_unqualified"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def cpu_check(args: argparse.Namespace) -> None:
    """Check the exact BF16 ONNX graph without VitisAI on the same ORT build."""
    import numpy as np
    import onnxruntime as ort

    exported = json.loads(args.export_report.read_text(encoding="utf-8-sig"))
    if (sha256(args.model) != exported["wrapped_sha256"]
            or sha256(args.inputs) != exported["inputs_sha256"]
            or sha256(args.reference) != exported["reference_sha256"]):
        raise ValueError("resampler CPU-control artifacts changed")
    report = {
        "scope": "exact BF16 ONNX resampler on ORT CPU; provider control only",
        "model_sha256": sha256(args.model),
        "inputs_sha256": sha256(args.inputs),
        "reference_sha256": sha256(args.reference),
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
    }
    try:
        session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
        with np.load(args.inputs, allow_pickle=False) as data:
            inputs = {name: np.ascontiguousarray(data[name]) for name in CASES}
        with np.load(args.reference, allow_pickle=False) as data:
            reference = {name: np.ascontiguousarray(data[name]) for name in CASES}
        report["cases"] = {}
        for name in CASES:
            observed = session.run(None, {"hidden_fp32": inputs[name]})[0]
            report["cases"][name] = {
                "relative_l2_vs_bf16_torch": relative_l2(reference[name], observed),
                "finite": bool(np.isfinite(observed).all()),
            }
        report["status"] = "cpu_graph_executed"
    except Exception as exc:
        report["status"] = "cpu_graph_unavailable"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "run", "cpu_check"):
        p = sub.add_parser(command)
        if command == "export":
            p.add_argument("--model-dir", type=Path, required=True)
            p.add_argument("--late27-reference", type=Path, required=True)
            p.add_argument("--direct-output", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--inputs", type=Path, required=True)
            p.add_argument("--reference", type=Path, required=True)
            p.add_argument("--report", type=Path, required=True)
            p.add_argument("--threads", type=int, default=8)
            p.add_argument("--trace", action="store_true")
        else:
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--inputs", type=Path, required=True)
            p.add_argument("--reference", type=Path, required=True)
            p.add_argument("--export-report", type=Path, required=True)
            if command == "run":
                p.add_argument("--ep-dir", type=Path, required=True)
                p.add_argument("--profile-prefix", type=Path, required=True)
                p.add_argument("--output-npz", type=Path, required=True)
            p.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
