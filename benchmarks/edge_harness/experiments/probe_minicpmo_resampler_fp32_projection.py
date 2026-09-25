#!/usr/bin/env python3
"""Test real-weight MiniCPM-o projection precision and HX370 provider placement.

This fixed-shape component probe never claims a complete image request. The
reference NPZ comes from the exact BF16 resampler trace on three images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
SHARD_SHA = "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31"
REFERENCE_SHA = "b155827adf49819355193ca743ae03b5dee2c05c857e70bef21589094ce6d368"
CASES = ("red", "blue", "third")
SHAPE = (1, 64, 4096)
GATE = 0.01


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference, observed) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def load_reference(path: Path):
    import numpy as np

    if sha256(path) != REFERENCE_SHA:
        raise ValueError("BF16 resampler trace reference changed")
    with np.load(path, allow_pickle=False) as source:
        values = {
            name: (np.ascontiguousarray(source[name + "_postnorm"]),
                   np.ascontiguousarray(source[name + "_embedding"]))
            for name in CASES
        }
    for name, (postnorm, embedding) in values.items():
        if (postnorm.shape != SHAPE or embedding.shape != SHAPE
                or postnorm.dtype != np.float32 or embedding.dtype != np.float32
                or not np.isfinite(postnorm).all() or not np.isfinite(embedding).all()):
            raise ValueError(f"{name} trace contract changed")
    return values


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def export(args: argparse.Namespace) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import TensorProto, helper, numpy_helper
    from safetensors import safe_open

    reference = load_reference(args.reference)
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if sha256(shard) != SHARD_SHA or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION:
        raise ValueError("MiniCPM-o checkpoint changed")
    with safe_open(shard, framework="pt", device="cpu") as source:
        weight = source.get_tensor("resampler.proj")
    if weight.shape != (4096, 4096) or weight.dtype != torch.bfloat16:
        raise ValueError("projection weight contract changed")
    weight_fp32 = np.ascontiguousarray(weight.float().numpy())
    if args.fp16:
        nodes = [
            helper.make_node("Cast", ["postnorm"], ["postnorm_fp16"], to=TensorProto.FLOAT16),
            helper.make_node("MatMul", ["postnorm_fp16", "weight"], ["embedding_fp16"]),
            helper.make_node("Cast", ["embedding_fp16"], ["embedding"], to=TensorProto.FLOAT),
        ]
        initial_weight = weight_fp32.astype(np.float16)
    else:
        nodes = [helper.make_node("MatMul", ["postnorm", "weight"], ["embedding"])]
        initial_weight = weight_fp32
    graph = helper.make_graph(
        nodes,
        "minicpmo_resampler_projection",
        [helper.make_tensor_value_info("postnorm", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, SHAPE)],
        [numpy_helper.from_array(initial_weight, name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.model)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    cases = {}
    for name, (postnorm, bf16_embedding) in reference.items():
        if args.fp16:
            torch_compute = (torch.from_numpy(postnorm).half()
                             @ torch.from_numpy(initial_weight)).float().numpy()
        else:
            torch_compute = (torch.from_numpy(postnorm) @ torch.from_numpy(weight_fp32)).numpy()
        ort_cpu = session.run(None, {"postnorm": postnorm})[0]
        cases[name] = {
            "torch_compute_vs_source_bf16_relative_l2": relative_l2(bf16_embedding, torch_compute),
            "ort_cpu_vs_torch_compute_relative_l2": relative_l2(torch_compute, ort_cpu),
            "ort_cpu_vs_source_bf16_relative_l2": relative_l2(bf16_embedding, ort_cpu),
        }
    report = {
        "scope": "real-weight postnorm projection component on three synthetic images",
        "revision": REVISION,
        "shard_sha256": SHARD_SHA,
        "reference_sha256": REFERENCE_SHA,
        "graph_sha256": sha256(args.model),
        "graph_bytes": args.model.stat().st_size,
        "shape": list(SHAPE),
        "weight_dtype_source": "BF16",
        "compute_dtype": "FP16" if args.fp16 else "FP32",
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "cases": cases,
    }
    write_report(args.report, report)


def quantize(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
        quantize_static,
    )

    exported = json.loads(args.export_report.read_text(encoding="utf-8-sig"))
    if sha256(args.model) != exported["graph_sha256"]:
        raise ValueError("projection graph changed")
    reference = load_reference(args.reference)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter(({"postnorm": reference[name][0]} for name in ("red", "blue")))

        def get_next(self):
            return next(self.items, None)

    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(args.model), str(args.output), Reader(), quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt16, weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"], calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )
    quantization_s = time.perf_counter() - started
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    cases = {}
    for name, (postnorm, bf16_embedding) in reference.items():
        output = session.run(None, {"postnorm": postnorm})[0]
        cases[name] = {
            "qdq_cpu_vs_source_bf16_relative_l2": relative_l2(bf16_embedding, output),
            "finite": bool(np.isfinite(output).all()),
        }
    report = {
        "scope": "red/blue calibrated A16W8 real-weight projection candidate; third image held out",
        "source_graph_sha256": exported["graph_sha256"],
        "reference_sha256": REFERENCE_SHA,
        "candidate_sha256": sha256(args.output),
        "candidate_bytes": args.output.stat().st_size,
        "onnxruntime": ort.__version__,
        "quantization_s": quantization_s,
        "calibration": "red and blue postnorm, MinMax per-channel A16W8 QDQ",
        "cases": cases,
        "status": ("cpu_candidate_numeric_pass" if all(
            case["finite"] and case["qdq_cpu_vs_source_bf16_relative_l2"] < GATE
            for case in cases.values()) else "cpu_candidate_numeric_fail"),
    }
    write_report(args.report, report)


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort

    exported = json.loads(args.export_report.read_text(encoding="utf-8-sig"))
    if args.warmup < 0 or args.samples < 1:
        raise ValueError("warmup must be nonnegative and samples must be positive")
    if args.candidate_report:
        candidate = json.loads(args.candidate_report.read_text(encoding="utf-8-sig"))
        if (sha256(args.model) != candidate["candidate_sha256"]
                or candidate["source_graph_sha256"] != exported["graph_sha256"]
                or candidate["reference_sha256"] != REFERENCE_SHA):
            raise ValueError("quantized projection candidate changed")
    elif sha256(args.model) != exported["graph_sha256"]:
        raise ValueError("projection graph changed")
    reference = load_reference(args.reference)
    ep_dir = args.ep_dir.resolve(strict=True)
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library("vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
    devices = [device for device in ort.get_ep_devices()
               if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
    if not devices:
        raise RuntimeError("VitisAI NPU unavailable")
    args.profile_prefix.parent.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    options.enable_profiling = True
    options.profile_file_prefix = str(args.profile_prefix)
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options)
    load_s = time.perf_counter() - started
    cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    outputs = {}
    cases = {}
    for name, (postnorm, bf16_embedding) in reference.items():
        cpu_result = cpu.run(None, {"postnorm": postnorm})[0]
        for _ in range(args.warmup):
            session.run(None, {"postnorm": postnorm})
            cpu.run(None, {"postnorm": postnorm})
        npu_times = []
        cpu_times = []
        for _ in range(args.samples):
            t = time.perf_counter()
            observed = session.run(None, {"postnorm": postnorm})[0]
            npu_times.append(time.perf_counter() - t)
            t = time.perf_counter()
            cpu.run(None, {"postnorm": postnorm})
            cpu_times.append(time.perf_counter() - t)
        outputs[name] = observed
        npu_sorted, cpu_sorted = sorted(npu_times), sorted(cpu_times)
        cases[name] = {
            "requested_vs_source_bf16_relative_l2": relative_l2(bf16_embedding, observed),
            "requested_vs_same_graph_ort_cpu_relative_l2": relative_l2(cpu_result, observed),
            "same_graph_ort_cpu_vs_source_bf16_relative_l2": relative_l2(bf16_embedding, cpu_result),
            "finite": bool(np.isfinite(observed).all()),
            "requested_call_s": npu_times,
            "cpu_call_s": cpu_times,
            "requested_p50_s": npu_sorted[(len(npu_sorted) - 1) // 2],
            "requested_p95_s": npu_sorted[(95 * len(npu_sorted) + 99) // 100 - 1],
            "cpu_p50_s": cpu_sorted[(len(cpu_sorted) - 1) // 2],
            "cpu_p95_s": cpu_sorted[(95 * len(cpu_sorted) + 99) // 100 - 1],
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
        "scope": "VitisAI-requested real-weight MiniCPM-o projection placement probe; not E2E",
        "graph_sha256": sha256(args.model),
        "candidate_report_sha256": sha256(args.candidate_report) if args.candidate_report else None,
        "reference_sha256": REFERENCE_SHA,
        "output_sha256": sha256(args.output_npz),
        "profile_sha256": sha256(profile),
        "profile_path": str(profile),
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "providers": session.get_providers(),
        "node_counts": counts,
        "session_creation_s": load_s,
        "warmup_per_case": args.warmup,
        "measured_calls_per_case": args.samples,
        "cases": cases,
        "status": ("component_numeric_pass_on_three_synthetic_images"
                   if counts.get("vitisai", 0) >= len(CASES) * (args.warmup + args.samples)
                   and all(item["finite"] and item["requested_vs_source_bf16_relative_l2"] < GATE
                           for item in cases.values()) else "component_unqualified"),
    }
    write_report(args.report, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--model-dir", type=Path, required=True)
    export_parser.add_argument("--reference", type=Path, required=True)
    export_parser.add_argument("--model", type=Path, required=True)
    export_parser.add_argument("--report", type=Path, required=True)
    export_parser.add_argument("--fp16", action="store_true")
    quantize_parser = sub.add_parser("quantize")
    quantize_parser.add_argument("--model", type=Path, required=True)
    quantize_parser.add_argument("--reference", type=Path, required=True)
    quantize_parser.add_argument("--export-report", type=Path, required=True)
    quantize_parser.add_argument("--output", type=Path, required=True)
    quantize_parser.add_argument("--report", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--model", type=Path, required=True)
    run_parser.add_argument("--reference", type=Path, required=True)
    run_parser.add_argument("--export-report", type=Path, required=True)
    run_parser.add_argument("--candidate-report", type=Path)
    run_parser.add_argument("--ep-dir", type=Path, required=True)
    run_parser.add_argument("--profile-prefix", type=Path, required=True)
    run_parser.add_argument("--output-npz", type=Path, required=True)
    run_parser.add_argument("--report", type=Path, required=True)
    run_parser.add_argument("--warmup", type=int, default=2)
    run_parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.command == "export":
        export(args)
    elif args.command == "quantize":
        quantize(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
