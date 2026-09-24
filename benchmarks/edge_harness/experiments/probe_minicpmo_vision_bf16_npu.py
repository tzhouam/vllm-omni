#!/usr/bin/env python3
"""Probe a real MiniCPM-o SigLIP BF16 layer on the HX370 NPU.

The graph keeps BF16 inside the layer. FP32 I/O casts only work around the
Windows ORT Python binding's inability to accept a BF16 NumPy input.
This is a fixed-image component experiment, not a complete model route.
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
FIXTURE_SHA = "cd25c58727814ee306e4c4b88ed78d89a2fc2de1cc2d5caca90782264bd02fc4"
BLUE_FIXTURE_SHA = "37fa24a0f99ca5b9b8105211f6b9dc4d28a8aece43ef3d68384c9c52ea32fe78"
THIRD_FIXTURE_SHA = "ded8164dde9cbefd8fd53c859d7cff1693072e8cac95607ed5b161293350b348"
WRAPPED_SHA = "6fe2bfa6ce6af5857e88066534b006935b3716f7206c278c35d0460ae58e7ae5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_l2(reference, actual) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = actual.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def export(args: argparse.Namespace) -> None:
    import numpy as np
    import onnx
    import torch
    from onnx import TensorProto, helper
    from safetensors import safe_open

    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if (sha256(shard) != SHARD_SHA
            or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION
            or sha256(args.fixture) != (args.fixture_sha256 or FIXTURE_SHA)):
        raise ValueError("source checkpoint or fixed image fixture changed")
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

    class Prefix(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = torch.nn.ModuleList(layers)

        def forward(self, hidden):
            for layer in self.layers:
                hidden = layer(hidden, None)[0]
            return hidden

    prefix = Prefix(list(model.encoder.layers[:1])).eval()
    with np.load(args.fixture, allow_pickle=False) as data:
        input_fp32 = np.ascontiguousarray(data["hidden"])
    if input_fp32.shape != (1, 1024, 1152) or input_fp32.dtype != np.float32:
        raise ValueError("vision input contract changed")
    hidden = torch.from_numpy(input_fp32).to(torch.bfloat16)
    with torch.inference_mode():
        reference = prefix(hidden).float().numpy()
        args.direct_output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        torch.onnx.export(
            prefix, (hidden,), str(args.direct_output),
            input_names=["hidden"], output_names=["hidden_after_prefix"],
            opset_version=17, dynamo=False, do_constant_folding=True,
        )
        export_s = time.perf_counter() - started
    graph = onnx.load(args.direct_output)
    onnx.checker.check_model(graph)
    old_input = graph.graph.input[0].name
    old_output = graph.graph.output[0].name
    if (graph.graph.input[0].type.tensor_type.elem_type != TensorProto.BFLOAT16
            or graph.graph.output[0].type.tensor_type.elem_type != TensorProto.BFLOAT16):
        raise ValueError("export did not retain BF16 I/O")
    graph.graph.input[0].name = old_input + "_fp32"
    graph.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
    graph.graph.output[0].name = old_output + "_fp32"
    graph.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    graph.graph.node.insert(0, helper.make_node(
        "Cast", [old_input + "_fp32"], [old_input],
        to=TensorProto.BFLOAT16, name="input_to_bf16"))
    graph.graph.node.append(helper.make_node(
        "Cast", [old_output], [old_output + "_fp32"],
        to=TensorProto.FLOAT, name="output_to_fp32"))
    onnx.checker.check_model(graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, args.output)
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.reference, source_bf16=reference)
    report = {
        "scope": f"real-weight MiniCPM-o SigLIP BF16 layer 1, fixed {args.fixture.stem} vision activation; component only",
        "model_revision": REVISION,
        "shard_sha256": SHARD_SHA,
        "fixture_sha256": sha256(args.fixture),
        "direct_sha256": sha256(args.direct_output),
        "wrapped_sha256": sha256(args.output),
        "reference_sha256": sha256(args.reference),
        "input_output_shape": [1, 1024, 1152],
        "internal_precision": "BF16",
        "external_precision": "FP32 casts",
        "export_s": export_s,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "status": "bf16_onnx_exported; NPU execution separate",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort

    if sha256(args.model) != WRAPPED_SHA or sha256(args.fixture) != FIXTURE_SHA:
        raise ValueError("wrapped graph or source fixture changed")
    with np.load(args.fixture, allow_pickle=False) as data:
        hidden = np.ascontiguousarray(data["hidden"])
    if hidden.shape != (1, 1024, 1152) or hidden.dtype != np.float32:
        raise ValueError("vision input contract changed")
    ep_dir = args.ep_dir.resolve(strict=True)
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library(
        "vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
    devices = [device for device in ort.get_ep_devices()
               if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
    if not devices:
        raise RuntimeError("VitisAI NPU device unavailable")
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    options.enable_profiling = True
    options.profile_file_prefix = str(args.profile_prefix)
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options)
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    observed = session.run(None, {"hidden_fp32": hidden})[0]
    run_s = time.perf_counter() - started
    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text(encoding="utf-8-sig"))
    counts: dict[str, int] = {}
    for event in events:
        if event.get("cat") == "Node":
            name = (event.get("args") or {}).get("provider")
            counts[name] = counts.get(name, 0) + 1
    with np.load(args.reference, allow_pickle=False) as data:
        reference = data["source_bf16"]
    if observed.shape != reference.shape or observed.dtype != np.float32:
        raise ValueError("vision output contract changed")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, output=observed)
    error = relative_l2(reference, observed)
    report = {
        "scope": "real-weight MiniCPM-o SigLIP BF16 layer 1 on HX370; component only",
        "model_sha256": WRAPPED_SHA,
        "fixture_sha256": FIXTURE_SHA,
        "reference_sha256": sha256(args.reference),
        "npu_output_sha256": sha256(args.output_npz),
        "profile_sha256": sha256(profile),
        "profile_path": str(profile),
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "ep_directory": str(ep_dir),
        "providers": session.get_providers(),
        "node_counts": counts,
        "load_s": load_s,
        "one_call_run_s": run_s,
        "relative_l2_vs_bf16_torch": error,
        "finite": bool(np.isfinite(observed).all()),
        "status": ("component_numeric_pass" if error <= .01
                   and counts.get("vitisai", 0) > 0 else "component_numeric_unqualified"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_multi(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort

    if sha256(args.model) != WRAPPED_SHA:
        raise ValueError("wrapped BF16 graph changed")
    cases = {
        "red": (args.red_fixture, args.red_reference, FIXTURE_SHA),
        "blue": (args.blue_fixture, args.blue_reference, BLUE_FIXTURE_SHA),
        "third": (args.third_fixture, args.third_reference, THIRD_FIXTURE_SHA),
    }
    inputs = {}
    references = {}
    for name, (fixture, reference, expected_sha) in cases.items():
        if sha256(fixture) != expected_sha:
            raise ValueError(f"{name} fixed fixture changed")
        with np.load(fixture, allow_pickle=False) as data:
            inputs[name] = np.ascontiguousarray(data["hidden"])
        with np.load(reference, allow_pickle=False) as data:
            references[name] = np.ascontiguousarray(data["source_bf16"])
        if (inputs[name].shape != (1, 1024, 1152)
                or inputs[name].dtype != np.float32
                or references[name].shape != inputs[name].shape
                or references[name].dtype != np.float32):
            raise ValueError(f"{name} vision tensor contract changed")

    ep_dir = args.ep_dir.resolve(strict=True)
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library(
        "vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
    devices = [device for device in ort.get_ep_devices()
               if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
    if not devices:
        raise RuntimeError("VitisAI NPU device unavailable")
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    options.enable_profiling = True
    options.profile_file_prefix = str(args.profile_prefix)
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options)
    load_s = time.perf_counter() - started
    outputs = {}
    cases_report = {}
    for name in cases:
        started = time.perf_counter()
        observed = session.run(None, {"hidden_fp32": inputs[name]})[0]
        call_s = time.perf_counter() - started
        if observed.shape != inputs[name].shape or observed.dtype != np.float32:
            raise ValueError(f"{name} NPU output contract changed")
        outputs[name] = observed
        cases_report[name] = {
            "fixture_sha256": sha256(cases[name][0]),
            "reference_sha256": sha256(cases[name][1]),
            "one_call_s": call_s,
            "relative_l2_vs_bf16_torch": relative_l2(references[name], observed),
            "finite": bool(np.isfinite(observed).all()),
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
        "scope": "one real-weight BF16 MiniCPM-o SigLIP layer on three fixed images; component only",
        "model_sha256": WRAPPED_SHA,
        "npu_output_sha256": sha256(args.output_npz),
        "profile_sha256": sha256(profile),
        "profile_path": str(profile),
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "providers": session.get_providers(),
        "node_counts": counts,
        "load_s": load_s,
        "cases": cases_report,
        "status": ("component_numeric_pass_three_images"
                   if counts.get("vitisai", 0) == len(cases)
                   and all(case["finite"] and case["relative_l2_vs_bf16_torch"] <= .01
                           for case in cases_report.values())
                   else "component_numeric_unqualified"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def audit(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from safetensors import safe_open

    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    exported = json.loads(args.export_report.read_text(encoding="utf-8-sig"))
    measured = json.loads(args.npu_report.read_text(encoding="utf-8-sig"))
    output_sha = measured.get("npu_output_sha256", measured.get("output_npz_sha256"))
    expected_fixture_sha = {
        None: FIXTURE_SHA, "red": FIXTURE_SHA,
        "blue": BLUE_FIXTURE_SHA, "third": THIRD_FIXTURE_SHA,
    }[args.case]
    if args.case is not None:
        case = measured["cases"][args.case]
        if (case["fixture_sha256"] != exported["fixture_sha256"]
                or case["reference_sha256"] != exported["reference_sha256"]):
            raise ValueError("multi-image case does not match export report")
    if (sha256(shard) != SHARD_SHA
            or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION
            or exported["model_revision"] != REVISION
            or exported["fixture_sha256"] != expected_fixture_sha
            or exported["wrapped_sha256"] != measured["model_sha256"]
            or sha256(args.reference) != exported["reference_sha256"]
            or sha256(args.npu_output) != output_sha
            or measured["node_counts"].get("vitisai", 0) < 1):
        raise ValueError("BF16 source or measured NPU boundary changed")
    with np.load(args.reference, allow_pickle=False) as data:
        source_boundary = np.ascontiguousarray(data["source_bf16"])
    with np.load(args.npu_output, allow_pickle=False) as data:
        npu_boundary = np.ascontiguousarray(data[args.case or "output"])
    if (source_boundary.shape != (1, 1024, 1152)
            or npu_boundary.shape != source_boundary.shape
            or source_boundary.dtype != np.float32
            or npu_boundary.dtype != np.float32):
        raise ValueError("vision boundary contract changed")
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

    def suffix(boundary):
        with torch.inference_mode():
            hidden = torch.from_numpy(boundary).to(torch.bfloat16)
            for layer in model.encoder.layers[1:]:
                hidden = layer(hidden, None)[0]
            raw = hidden.float().numpy()
            normalized = model.post_layernorm(hidden).float().numpy()
        return raw, normalized

    source_raw, source_norm = suffix(source_boundary)
    npu_raw, npu_norm = suffix(npu_boundary)
    report = {
        "scope": "measured HX370 BF16 NPU layer 1 plus unchanged BF16 Torch layers 2..27 and post-layernorm; component only",
        "case": args.case or "single_red",
        "model_revision": REVISION,
        "shard_sha256": SHARD_SHA,
        "reference_sha256": sha256(args.reference),
        "npu_output_sha256": output_sha,
        "export_report_sha256": sha256(args.export_report),
        "npu_report_sha256": sha256(args.npu_report),
        "boundary_relative_l2": relative_l2(source_boundary, npu_boundary),
        "after27_relative_l2": relative_l2(source_raw, npu_raw),
        "post_norm_relative_l2": relative_l2(source_norm, npu_norm),
        "finite": bool(np.isfinite(npu_norm).all()),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": args.threads,
        "status": "bf16_suffix_replay_measured_not_e2e",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--model-dir", type=Path, required=True)
    exporter.add_argument("--fixture", type=Path, required=True)
    exporter.add_argument("--fixture-sha256")
    exporter.add_argument("--direct-output", type=Path, required=True)
    exporter.add_argument("--output", type=Path, required=True)
    exporter.add_argument("--reference", type=Path, required=True)
    exporter.add_argument("--report", type=Path, required=True)
    exporter.add_argument("--threads", type=int, default=8)
    runner = commands.add_parser("run")
    runner.add_argument("--model", type=Path, required=True)
    runner.add_argument("--fixture", type=Path, required=True)
    runner.add_argument("--reference", type=Path, required=True)
    runner.add_argument("--ep-dir", type=Path, required=True)
    runner.add_argument("--profile-prefix", type=Path, required=True)
    runner.add_argument("--output-npz", type=Path, required=True)
    runner.add_argument("--report", type=Path, required=True)
    multi = commands.add_parser("run-multi")
    multi.add_argument("--model", type=Path, required=True)
    for name in ("red", "blue", "third"):
        multi.add_argument(f"--{name}-fixture", type=Path, required=True)
        multi.add_argument(f"--{name}-reference", type=Path, required=True)
    multi.add_argument("--ep-dir", type=Path, required=True)
    multi.add_argument("--profile-prefix", type=Path, required=True)
    multi.add_argument("--output-npz", type=Path, required=True)
    multi.add_argument("--report", type=Path, required=True)
    auditor = commands.add_parser("audit")
    auditor.add_argument("--model-dir", type=Path, required=True)
    auditor.add_argument("--reference", type=Path, required=True)
    auditor.add_argument("--npu-output", type=Path, required=True)
    auditor.add_argument("--export-report", type=Path, required=True)
    auditor.add_argument("--npu-report", type=Path, required=True)
    auditor.add_argument("--case", choices=("red", "blue", "third"))
    auditor.add_argument("--report", type=Path, required=True)
    auditor.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.command == "export":
        if not 1 <= args.threads <= 24:
            parser.error("threads must be 1..24")
        export(args)
    elif args.command == "run":
        run(args)
    elif args.command == "run-multi":
        run_multi(args)
    else:
        if not 1 <= args.threads <= 24:
            parser.error("threads must be 1..24")
        audit(args)


if __name__ == "__main__":
    main()
