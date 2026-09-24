#!/usr/bin/env python3
"""Locate HX370 BF16 SigLIP layer error at its attention boundary.

The intermediate output changes graph observability and may change provider
compilation. Compare its full-layer output with the earlier one-output graph
before transferring any conclusion to that graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path


REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
SHARD_SHA = "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31"
FIXTURE_SHA = {
    "red": "cd25c58727814ee306e4c4b88ed78d89a2fc2de1cc2d5caca90782264bd02fc4",
    "blue": "37fa24a0f99ca5b9b8105211f6b9dc4d28a8aece43ef3d68384c9c52ea32fe78",
    "third": "ded8164dde9cbefd8fd53c859d7cff1693072e8cac95607ed5b161293350b348",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_l2(reference, observed) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def fixtures(args):
    import numpy as np

    result = {}
    for name in FIXTURE_SHA:
        path = getattr(args, f"{name}_fixture")
        if sha256(path) != FIXTURE_SHA[name]:
            raise ValueError(f"{name} fixture changed")
        with np.load(path, allow_pickle=False) as data:
            hidden = np.ascontiguousarray(data["hidden"])
        if hidden.shape != (1, 1024, 1152) or hidden.dtype != np.float32:
            raise ValueError(f"{name} input contract changed")
        result[name] = hidden
    return result


def export(args) -> None:
    import numpy as np
    import onnx
    import torch
    from onnx import TensorProto, helper
    from safetensors import safe_open

    inputs = fixtures(args)
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if (sha256(shard) != SHARD_SHA
            or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION):
        raise ValueError("checkpoint revision or shard changed")
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
    if not 0 <= args.layer_index < len(model.encoder.layers):
        raise ValueError("layer index exceeds the checkpoint's vision encoder")
    layer = model.encoder.layers[args.layer_index]
    if args.layer_index:
        if args.layer_inputs is None:
            raise ValueError("--layer-inputs is required for a later layer")
        with torch.inference_mode():
            for name, hidden in inputs.items():
                state = torch.from_numpy(hidden).to(torch.bfloat16)
                for preceding in model.encoder.layers[:args.layer_index]:
                    state = preceding(state, None)[0]
                inputs[name] = np.ascontiguousarray(state.float().numpy())
        args.layer_inputs.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.layer_inputs, **inputs)

    class AttentionBoundary(torch.nn.Module):
        def __init__(self, source_layer):
            super().__init__()
            self.source_layer = source_layer

        def forward(self, hidden):
            attention, _ = self.source_layer.self_attn(
                self.source_layer.layer_norm1(hidden), None)
            after_attention = hidden + attention
            after_layer = after_attention + self.source_layer.mlp(
                self.source_layer.layer_norm2(after_attention))
            return after_attention, after_layer

    boundary = AttentionBoundary(layer).eval()
    references = {}
    with torch.inference_mode():
        for name, hidden in inputs.items():
            after_attention, after_layer = boundary(
                torch.from_numpy(hidden).to(torch.bfloat16))
            references[name + "_attention"] = after_attention.float().numpy()
            references[name + "_full"] = after_layer.float().numpy()
        args.direct_output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        torch.onnx.export(
            boundary, (torch.from_numpy(inputs["red"]).to(torch.bfloat16),),
            str(args.direct_output), input_names=["hidden"],
            output_names=["after_attention", "after_layer"],
            opset_version=17, dynamo=False, do_constant_folding=True,
        )
        export_s = time.perf_counter() - started
    for name in FIXTURE_SHA:
        if not np.array_equal(references[name + "_full"],
                              layer(torch.from_numpy(inputs[name]).to(torch.bfloat16), None)[0].float().detach().numpy()):
            raise ValueError("instrumented layer changed Torch BF16 output")
    graph = onnx.load(args.direct_output)
    onnx.checker.check_model(graph)
    if (graph.graph.input[0].type.tensor_type.elem_type != TensorProto.BFLOAT16
            or any(output.type.tensor_type.elem_type != TensorProto.BFLOAT16
                   for output in graph.graph.output)):
        raise ValueError("instrumented graph did not retain BF16 tensors")
    graph.graph.input[0].name = "hidden_fp32"
    graph.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
    graph.graph.node.insert(0, helper.make_node(
        "Cast", ["hidden_fp32"], ["hidden"],
        to=TensorProto.BFLOAT16, name="input_to_bf16"))
    for output in graph.graph.output:
        internal = output.name
        output.name = internal + "_fp32"
        output.type.tensor_type.elem_type = TensorProto.FLOAT
        graph.graph.node.append(helper.make_node(
            "Cast", [internal], [output.name], to=TensorProto.FLOAT,
            name=internal + "_to_fp32"))
    onnx.checker.check_model(graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, args.output)
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.reference, **references)
    report = {
        "scope": f"real-weight BF16 SigLIP layer {args.layer_index + 1} with attention residual and full output; component only",
        "model_revision": REVISION,
        "shard_sha256": SHARD_SHA,
        "fixture_sha256": FIXTURE_SHA,
        "layer_index": args.layer_index,
        "layer_inputs_sha256": sha256(args.layer_inputs) if args.layer_index else None,
        "direct_sha256": sha256(args.direct_output),
        "wrapped_sha256": sha256(args.output),
        "reference_sha256": sha256(args.reference),
        "export_s": export_s,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "status": "bf16_attention_boundary_exported",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def run(args) -> None:
    import numpy as np
    import onnxruntime as ort

    inputs = fixtures(args)
    exported = json.loads(args.export_report.read_text(encoding="utf-8-sig"))
    layer_index = exported.get("layer_index", 0)
    if layer_index:
        if args.layer_inputs is None or sha256(args.layer_inputs) != exported["layer_inputs_sha256"]:
            raise ValueError("later-layer activation artifact changed")
        with np.load(args.layer_inputs, allow_pickle=False) as data:
            inputs = {name: np.ascontiguousarray(data[name]) for name in FIXTURE_SHA}
        if any(value.shape != (1, 1024, 1152) or value.dtype != np.float32
               for value in inputs.values()):
            raise ValueError("later-layer activation contract changed")
    if sha256(args.model) != exported["wrapped_sha256"]:
        raise ValueError("instrumented ONNX artifact changed")
    if sha256(args.reference) != exported["reference_sha256"]:
        raise ValueError("instrumented Torch reference changed")
    with np.load(args.reference, allow_pickle=False) as data:
        reference = {key: np.ascontiguousarray(data[key]) for key in data.files}
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
    for name, hidden in inputs.items():
        started = time.perf_counter()
        after_attention, after_layer = session.run(None, {"hidden_fp32": hidden})
        call_s = time.perf_counter() - started
        outputs[name + "_attention"] = after_attention
        outputs[name + "_full"] = after_layer
        cases[name] = {
            "one_call_s": call_s,
            "attention_relative_l2": relative_l2(
                reference[name + "_attention"], after_attention),
            "full_relative_l2": relative_l2(
                reference[name + "_full"], after_layer),
            "finite": bool(np.isfinite(after_attention).all()
                           and np.isfinite(after_layer).all()),
        }
    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text(encoding="utf-8-sig"))
    counts: dict[str, int] = {}
    for event in events:
        if event.get("cat") == "Node":
            provider = (event.get("args") or {}).get("provider")
            counts[provider] = counts.get(provider, 0) + 1
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **outputs)
    report = {
        "scope": f"real-weight BF16 SigLIP layer {layer_index + 1} attention boundary on HX370 NPU; component only",
        "model_sha256": sha256(args.model),
        "layer_index": layer_index,
        "layer_inputs_sha256": exported.get("layer_inputs_sha256"),
        "reference_sha256": sha256(args.reference),
        "fixture_sha256": FIXTURE_SHA,
        "output_npz_sha256": sha256(args.output_npz),
        "profile_sha256": sha256(profile),
        "profile_path": str(profile),
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "providers": session.get_providers(),
        "node_counts": counts,
        "load_s": load_s,
        "cases": cases,
        "status": ("attention_boundary_npu_measured"
                   if counts.get("vitisai", 0) > 0
                   and all(case["finite"] for case in cases.values())
                   else "attention_boundary_unqualified"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def audit(args) -> None:
    import numpy as np
    import torch
    from safetensors import safe_open

    exported = json.loads(args.export_report.read_text(encoding="utf-8-sig"))
    measured = json.loads(args.npu_report.read_text(encoding="utf-8-sig"))
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if (exported["model_revision"] != REVISION
            or exported["shard_sha256"] != SHARD_SHA
            or exported["wrapped_sha256"] != measured["model_sha256"]
            or exported["reference_sha256"] != sha256(args.reference)
            or measured["reference_sha256"] != sha256(args.reference)
            or measured["output_npz_sha256"] != sha256(args.npu_output)
            or measured["node_counts"].get("vitisai", 0) < 1
            or exported.get("layer_index", 0) != measured.get("layer_index", 0)
            or sha256(shard) != SHARD_SHA
            or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION):
        raise ValueError("checkpoint or measured attention boundary changed")
    with np.load(args.reference, allow_pickle=False) as data:
        reference = {key: np.ascontiguousarray(data[key]) for key in data.files}
    with np.load(args.npu_output, allow_pickle=False) as data:
        npu = {key: np.ascontiguousarray(data[key]) for key in data.files}
    prior = None
    if args.prior_output is not None:
        with np.load(args.prior_output, allow_pickle=False) as data:
            prior = {key: np.ascontiguousarray(data[key]) for key in data.files}

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
    layer_index = exported.get("layer_index", 0)
    if layer_index == 0 and args.prior_output is None:
        raise ValueError("--prior-output is required for the layer-one audit")
    layer = model.encoder.layers[layer_index]
    resampler = None
    if args.resampler:
        import importlib

        sys.path.insert(0, str(args.model_dir.resolve().parent))
        Resampler = importlib.import_module(
            args.model_dir.name + ".modeling_minicpmo").Resampler
        full_config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
        embed_dim = full_config["hidden_size"]
        resampler = Resampler(
            num_queries=full_config["query_num"], embed_dim=embed_dim,
            num_heads=embed_dim // 128, kv_dim=vision_config.hidden_size,
            adaptive=True,
        )
        with safe_open(shard, framework="pt", device="cpu") as source:
            resampler_state = {
                key[len("resampler."):]: source.get_tensor(key)
                for key in source.keys() if key.startswith("resampler.")
            }
        resampler.load_state_dict(resampler_state, strict=True)
        resampler.eval().to(torch.bfloat16)

    def suffix(boundary):
        with torch.inference_mode():
            hidden = torch.from_numpy(boundary).to(torch.bfloat16)
            for remaining in model.encoder.layers[layer_index + 1:]:
                hidden = remaining(hidden, None)[0]
            raw = hidden.float().numpy()
            normalized = model.post_layernorm(hidden).float().numpy()
        return raw, normalized

    cases = {}
    for name in FIXTURE_SHA:
        attn = npu[name + "_attention"]
        full = npu[name + "_full"]
        source_attn = reference[name + "_attention"]
        source_full = reference[name + "_full"]
        if (attn.shape != (1, 1024, 1152)
                or full.shape != attn.shape
                or (prior is not None and not np.array_equal(full, prior[name]))):
            raise ValueError(f"{name} instrumented NPU full output changed")
        with torch.inference_mode():
            hidden = torch.from_numpy(attn).to(torch.bfloat16)
            hybrid = (hidden + layer.mlp(layer.layer_norm2(hidden))).float().numpy()
        source_raw, source_norm = suffix(source_full)
        hybrid_raw, hybrid_norm = suffix(hybrid)
        npu_full_raw, npu_full_norm = suffix(full)
        resampler_errors = {}
        if resampler is not None:
            tgt_sizes = torch.tensor([[32, 32]], dtype=torch.long)
            with torch.inference_mode():
                def resample(normalized):
                    return resampler(
                        torch.from_numpy(normalized).to(torch.bfloat16),
                        tgt_sizes).float().numpy()

                source_resampled = resample(source_norm)
                hybrid_resampled = resample(hybrid_norm)
                npu_full_resampled = resample(npu_full_norm)
            resampler_errors = {
                "hybrid_resampler_relative_l2": relative_l2(
                    source_resampled, hybrid_resampled),
                "npu_full_resampler_relative_l2": relative_l2(
                    source_resampled, npu_full_resampled),
                "resampler_output_shape": list(source_resampled.shape),
                "resampler_finite": bool(np.isfinite(source_resampled).all()
                                         and np.isfinite(hybrid_resampled).all()
                                         and np.isfinite(npu_full_resampled).all()),
            }
        cases[name] = {
            "instrumented_full_matches_original_npu_bitwise": (
                True if prior is not None else None),
            "npu_attention_vs_torch_relative_l2": relative_l2(source_attn, attn),
            "npu_full_vs_torch_relative_l2": relative_l2(source_full, full),
            "cpu_bf16_mlp_on_npu_attention_vs_torch_full_relative_l2":
                relative_l2(source_full, hybrid),
            "npu_full_vs_cpu_bf16_mlp_on_npu_attention_relative_l2":
                relative_l2(hybrid, full),
            "hybrid_after27_relative_l2": relative_l2(source_raw, hybrid_raw),
            "hybrid_post_norm_relative_l2": relative_l2(source_norm, hybrid_norm),
            "npu_full_after27_relative_l2": relative_l2(source_raw, npu_full_raw),
            "npu_full_post_norm_relative_l2": relative_l2(source_norm, npu_full_norm),
            "finite": bool(np.isfinite(hybrid_norm).all()
                           and np.isfinite(npu_full_norm).all()),
            **resampler_errors,
        }
    component_pass = all(
        case["finite"]
        and case["npu_attention_vs_torch_relative_l2"] <= .01
        and case["cpu_bf16_mlp_on_npu_attention_vs_torch_full_relative_l2"] <= .01
        for case in cases.values())
    suffix_pass = all(case["hybrid_post_norm_relative_l2"] <= .01
                      and (resampler is None or (
                          case["resampler_finite"]
                          and case["hybrid_resampler_relative_l2"] <= .01))
                      for case in cases.values())
    report = {
        "scope": f"measured HX370 BF16 layer {layer_index + 1} attention residual plus Torch BF16 MLP and vision suffix; component-only offline replay",
        "model_revision": REVISION,
        "shard_sha256": SHARD_SHA,
        "export_report_sha256": sha256(args.export_report),
        "npu_report_sha256": sha256(args.npu_report),
        "reference_sha256": sha256(args.reference),
        "npu_output_sha256": sha256(args.npu_output),
        "prior_one_output_sha256": sha256(args.prior_output) if args.prior_output else None,
        "layer_index": layer_index,
        "resampler_replayed": resampler is not None,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "threads": args.threads,
        "cases": cases,
        "status": (
            "attention_hybrid_postnorm_numeric_pass_on_three_fixed_images"
            if component_pass and suffix_pass else
            "attention_component_pass_on_three_fixed_images; vision_suffix_unqualified"
            if component_pass else "attention_component_or_suffix_requires_review"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "run"):
        sub = commands.add_parser(command)
        for name in FIXTURE_SHA:
            sub.add_argument(f"--{name}-fixture", type=Path, required=True)
        sub.add_argument("--reference", type=Path, required=True)
        sub.add_argument("--report", type=Path, required=True)
        if command == "export":
            sub.add_argument("--model-dir", type=Path, required=True)
            sub.add_argument("--direct-output", type=Path, required=True)
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--layer-index", type=int, default=0)
            sub.add_argument("--layer-inputs", type=Path)
            sub.add_argument("--threads", type=int, default=8)
        else:
            sub.add_argument("--model", type=Path, required=True)
            sub.add_argument("--export-report", type=Path, required=True)
            sub.add_argument("--layer-inputs", type=Path)
            sub.add_argument("--ep-dir", type=Path, required=True)
            sub.add_argument("--profile-prefix", type=Path, required=True)
            sub.add_argument("--output-npz", type=Path, required=True)
    auditor = commands.add_parser("audit")
    auditor.add_argument("--model-dir", type=Path, required=True)
    auditor.add_argument("--reference", type=Path, required=True)
    auditor.add_argument("--npu-output", type=Path, required=True)
    auditor.add_argument("--prior-output", type=Path)
    auditor.add_argument("--export-report", type=Path, required=True)
    auditor.add_argument("--npu-report", type=Path, required=True)
    auditor.add_argument("--report", type=Path, required=True)
    auditor.add_argument("--threads", type=int, default=8)
    auditor.add_argument("--resampler", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        if not 1 <= args.threads <= 24:
            parser.error("threads must be 1..24")
        export(args)
    elif args.command == "run":
        run(args)
    else:
        if not 1 <= args.threads <= 24:
            parser.error("threads must be 1..24")
        audit(args)


if __name__ == "__main__":
    main()
