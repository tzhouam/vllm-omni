#!/usr/bin/env python3
"""Export and test MiniCPM-o's real KV projection as an optional HX370 NPU stage."""

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
INPUTS_SHA = "ab71cc3a6b8461c99cdf9e458b2e8b99cc092dd6adc3d49f92d643ec4b8693de"
REFERENCE_SHA = "b155827adf49819355193ca743ae03b5dee2c05c857e70bef21589094ce6d368"
CASES = ("red", "blue", "third")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference, observed) -> float:
    import numpy as np

    a, b = reference.astype(np.float64), observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def load_arrays(inputs: Path, reference: Path):
    import numpy as np

    if sha256(inputs) != INPUTS_SHA or sha256(reference) != REFERENCE_SHA:
        raise ValueError("resampler input or BF16 reference changed")
    with np.load(inputs, allow_pickle=False) as data:
        hidden = {name: np.ascontiguousarray(data[name]) for name in CASES}
    with np.load(reference, allow_pickle=False) as data:
        projected = {name: np.ascontiguousarray(data[name + "_projected"]) for name in CASES}
    for name in CASES:
        if (hidden[name].shape != (1, 1024, 1152) or hidden[name].dtype != np.float32
                or projected[name].shape != (1, 1024, 4096) or projected[name].dtype != np.float32):
            raise ValueError(f"{name} KV projection contract changed")
    return hidden, projected


def write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def export(args) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import TensorProto, helper, numpy_helper
    from safetensors import safe_open

    hidden, projected = load_arrays(args.inputs, args.reference)
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if sha256(shard) != SHARD_SHA or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION:
        raise ValueError("MiniCPM-o checkpoint changed")
    with safe_open(shard, framework="pt", device="cpu") as source:
        weight = source.get_tensor("resampler.kv_proj.weight")
    if weight.shape != (4096, 1152) or weight.dtype != torch.bfloat16:
        raise ValueError("KV projection weight contract changed")
    weight_fp32 = np.ascontiguousarray(weight.float().T.numpy())
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["hidden", "weight"], ["projected"])],
        "minicpmo_kv_projection",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 1024, 1152])],
        [helper.make_tensor_value_info("projected", TensorProto.FLOAT, [1, 1024, 4096])],
        [numpy_helper.from_array(weight_fp32, name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.model)
    cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    cases = {}
    for name in CASES:
        output = cpu.run(None, {"hidden": hidden[name]})[0]
        cases[name] = {
            "fp32_cpu_vs_source_bf16_projected_relative_l2": relative_l2(projected[name], output),
            "finite": bool(np.isfinite(output).all()),
        }
    write(args.report, {
        "scope": "real-weight FP32 MiniCPM-o KV projection export; component only",
        "revision": REVISION, "shard_sha256": SHARD_SHA,
        "inputs_sha256": INPUTS_SHA, "reference_sha256": REFERENCE_SHA,
        "graph_sha256": sha256(args.model), "graph_bytes": args.model.stat().st_size,
        "onnxruntime": ort.__version__, "torch": torch.__version__,
        "platform": platform.platform(), "cases": cases,
    })


def quantize(args) -> None:
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
        quantize_static,
    )

    exported = json.loads(args.export_report.read_text(encoding="utf-8-sig"))
    if sha256(args.model) != exported["graph_sha256"]:
        raise ValueError("KV projection source changed")
    hidden, projected = load_arrays(args.inputs, args.reference)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.iterator = iter(({"hidden": hidden[name]} for name in ("red", "blue")))

        def get_next(self):
            return next(self.iterator, None)

    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(args.model), str(args.output), Reader(), quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt16, weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"], calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )
    quantization_s = time.perf_counter() - started
    cpu = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    outputs = {name: cpu.run(None, {"hidden": hidden[name]})[0] for name in CASES}
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **outputs)
    cases = {name: {
        "qdq_cpu_vs_source_bf16_projected_relative_l2": relative_l2(projected[name], outputs[name]),
        "finite": bool(np.isfinite(outputs[name]).all()),
    } for name in CASES}
    write(args.report, {
        "scope": "red/blue calibrated A16W8 KV projection; third image held out",
        "source_graph_sha256": exported["graph_sha256"],
        "candidate_sha256": sha256(args.output),
        "candidate_bytes": args.output.stat().st_size,
        "candidate_cpu_output_sha256": sha256(args.output_npz),
        "inputs_sha256": INPUTS_SHA, "reference_sha256": REFERENCE_SHA,
        "onnxruntime": ort.__version__, "quantization_s": quantization_s,
        "cases": cases,
    })


def run(args) -> None:
    import numpy as np
    import onnxruntime as ort

    candidate = json.loads(args.candidate_report.read_text(encoding="utf-8-sig"))
    if sha256(args.model) != candidate["candidate_sha256"]:
        raise ValueError("A16W8 KV candidate changed")
    if args.warmup < 0 or args.samples < 1:
        raise ValueError("warmup must be nonnegative and samples positive")
    hidden, projected = load_arrays(args.inputs, args.reference)
    ep_dir = args.ep_dir.resolve(strict=True)
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library("vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
    devices = [device for device in ort.get_ep_devices()
               if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
    if not devices:
        raise RuntimeError("HX370 NPU unavailable")
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    options.enable_profiling = True
    options.profile_file_prefix = str(args.profile_prefix)
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options)
    creation_s = time.perf_counter() - started
    cpu_fp32 = None
    if args.fp32_model:
        if sha256(args.fp32_model) != candidate["source_graph_sha256"]:
            raise ValueError("FP32 CPU baseline graph changed")
        cpu_fp32 = ort.InferenceSession(str(args.fp32_model), providers=["CPUExecutionProvider"])
    outputs, cases = {}, {}
    for name in CASES:
        feed = {"hidden": hidden[name]}
        for _ in range(args.warmup):
            session.run(None, feed)
            if cpu_fp32:
                cpu_fp32.run(None, feed)
        requested_times, cpu_times = [], []
        for _ in range(args.samples):
            started = time.perf_counter()
            observed = session.run(None, feed)[0]
            requested_times.append(time.perf_counter() - started)
            if cpu_fp32:
                started = time.perf_counter()
                cpu_fp32.run(None, feed)
                cpu_times.append(time.perf_counter() - started)
        outputs[name] = observed
        cases[name] = {
            "requested_vs_source_bf16_projected_relative_l2": relative_l2(projected[name], observed),
            "finite": bool(np.isfinite(observed).all()),
            "requested_call_s": requested_times,
            "requested_p50_s": sorted(requested_times)[(len(requested_times) - 1) // 2],
            "requested_p95_s": sorted(requested_times)[(95 * len(requested_times) + 99) // 100 - 1],
        }
        if cpu_fp32:
            baseline = cpu_fp32.run(None, feed)[0]
            cases[name]["fp32_cpu_call_s"] = cpu_times
            cases[name]["fp32_cpu_p50_s"] = sorted(cpu_times)[(len(cpu_times) - 1) // 2]
            cases[name]["fp32_cpu_p95_s"] = sorted(cpu_times)[(95 * len(cpu_times) + 99) // 100 - 1]
            cases[name]["requested_vs_fp32_cpu_relative_l2"] = relative_l2(baseline, observed)
    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text(encoding="utf-8-sig"))
    counts = {}
    for event in events:
        if event.get("cat") == "Node":
            provider = (event.get("args") or {}).get("provider")
            counts[provider] = counts.get(provider, 0) + 1
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **outputs)
    write(args.report, {
        "scope": "VitisAI-requested A16W8 KV projection on HX370; component only",
        "candidate_sha256": candidate["candidate_sha256"],
        "inputs_sha256": INPUTS_SHA, "reference_sha256": REFERENCE_SHA,
        "output_npz_sha256": sha256(args.output_npz),
        "profile_sha256": sha256(profile), "profile_path": str(profile),
        "onnxruntime": ort.__version__, "platform": platform.platform(),
        "providers": session.get_providers(), "node_counts": counts,
        "session_creation_s": creation_s, "cases": cases,
        "warmup_per_case": args.warmup,
        "measured_calls_per_case": args.samples,
        "fp32_cpu_graph_sha256": sha256(args.fp32_model) if args.fp32_model else None,
        "status": ("npu_placed" if counts.get("vitisai") == len(CASES) * (args.warmup + args.samples)
                   else "npu_not_placed"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "quantize", "run"):
        p = sub.add_parser(command)
        p.add_argument("--inputs", type=Path, required=True)
        p.add_argument("--reference", type=Path, required=True)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--report", type=Path, required=True)
        if command == "export":
            p.add_argument("--model-dir", type=Path, required=True)
        elif command == "quantize":
            p.add_argument("--export-report", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--output-npz", type=Path, required=True)
        else:
            p.add_argument("--candidate-report", type=Path, required=True)
            p.add_argument("--ep-dir", type=Path, required=True)
            p.add_argument("--profile-prefix", type=Path, required=True)
            p.add_argument("--output-npz", type=Path, required=True)
            p.add_argument("--fp32-model", type=Path)
            p.add_argument("--warmup", type=int, default=0)
            p.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
