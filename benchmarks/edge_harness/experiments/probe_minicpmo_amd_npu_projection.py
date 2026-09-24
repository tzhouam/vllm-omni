#!/usr/bin/env python3
"""Extract and check a MiniCPM-o speech-token projection at a state-safe boundary.

The CPU speech head retains the 20-layer state and all new K/V tensors. Only
its final normalized activation crosses to the optional AMD NPU projection.
This experiment does not implement or qualify a complete MiniCPM-o request.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path


SOURCE_SHA = "78cf64804f11ad269ee4180da61588ccc5eefe2942773227345499756e4cc229"
FIXTURE_SHA = "65919ea1f351feec33c1415a3e3db76d60bc9c2beb3d49a72af75dcb418e83b4"
REFERENCE_SHA = "902adcceadbe761d938ef2ade51cf953c8917ea1abfaee33a10c84e300c1bd96"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def relative_l2(reference, actual) -> float:
    import numpy as np

    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return float(np.linalg.norm(difference) / max(np.linalg.norm(reference.astype(np.float64)), 1e-12))


def export(args: argparse.Namespace) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    hashes = {name: sha256(getattr(args, name)) for name in ("source", "fixture", "reference")}
    if hashes != {"source": SOURCE_SHA, "fixture": FIXTURE_SHA, "reference": REFERENCE_SHA}:
        raise ValueError("source graph or retained fixture/reference changed")
    with np.load(args.fixture, allow_pickle=False) as file:
        inputs = {name: np.ascontiguousarray(file[name]) for name in file.files}
    with np.load(args.reference, allow_pickle=False) as file:
        retained = {name: file[name] for name in file.files}
    if len(inputs) != 44 or len(retained) != 41:
        raise ValueError("source fixture schema changed")

    model = onnx.load(args.source)
    nodes = {node.name: node for node in model.graph.node}
    linear = nodes["node_linear_140"]
    select = nodes["node_select"]
    if (linear.op_type != "MatMul" or list(linear.input) != ["mul_201", "val_1711"]
            or list(linear.output) != ["linear_140"] or select.op_type != "Gather"
            or list(select.input) != ["linear_140", "val_27"]
            or list(select.output) != ["logits"]):
        raise ValueError("final speech-token projection boundary changed")
    initializers = {item.name: item for item in model.graph.initializer}
    if list(initializers["val_1711"].dims) != [768, 6562]:
        raise ValueError("speech-token projection weight shape changed")
    model.graph.output.extend([helper.make_tensor_value_info("mul_201", TensorProto.FLOAT, [1, 1, 768])])
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    outputs = [item.name for item in session.get_outputs()]
    if set(outputs) != set(retained) | {"mul_201"} or set(item.name for item in session.get_inputs()) != set(inputs):
        raise ValueError("captured source graph I/O changed")
    started = time.perf_counter()
    result = dict(zip(outputs, session.run(None, inputs), strict=True))
    full_cpu_s = time.perf_counter() - started
    if any(not np.array_equal(result[name], retained[name]) for name in retained):
        raise ValueError("augmented CPU speech head changed a retained output")
    activation = np.ascontiguousarray(result["mul_201"])
    if activation.shape != (1, 1, 768) or not np.isfinite(activation).all():
        raise ValueError("projection input contract failed")

    graph = helper.make_graph(
        [linear, select], "minicpmo_speech_token_projection",
        [helper.make_tensor_value_info("mul_201", TensorProto.FLOAT, [1, 1, 768])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 6562])],
        [initializers["val_1711"], initializers["val_27"]],
    )
    projection = helper.make_model(graph, opset_imports=model.opset_import)
    projection.ir_version = model.ir_version
    onnx.checker.check_model(projection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(projection, args.output)
    small = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    projected = small.run(None, {"mul_201": activation})[0]
    if not np.array_equal(projected, result["logits"]):
        raise ValueError("isolated projection changed CPU logits")

    cache_outputs = [copy.deepcopy(item) for item in model.graph.output if item.name != "logits" and item.name != "mul_201"]
    if len(cache_outputs) != 40 or list(model.graph.node[-2:]) != [linear, select]:
        raise ValueError("speech-head prefix boundary changed")
    del model.graph.node[-2:]
    del model.graph.output[:]
    model.graph.output.extend(cache_outputs)
    model.graph.output.extend([helper.make_tensor_value_info("mul_201", TensorProto.FLOAT, [1, 1, 768])])
    weights = [item for item in model.graph.initializer if item.name not in ("val_1711", "val_27")]
    del model.graph.initializer[:]
    model.graph.initializer.extend(weights)
    onnx.checker.check_model(model)
    args.prefix_output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.prefix_output)
    prefix = ort.InferenceSession(str(args.prefix_output), providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    prefix_outputs = dict(zip([item.name for item in prefix.get_outputs()], prefix.run(None, inputs), strict=True))
    prefix_cpu_s = time.perf_counter() - started
    if set(prefix_outputs) != set(retained) - {"logits"} | {"mul_201"}:
        raise ValueError("CPU prefix output schema changed")
    if (not np.array_equal(prefix_outputs["mul_201"], activation)
            or any(not np.array_equal(prefix_outputs[name], retained[name]) for name in retained if name != "logits")):
        raise ValueError("CPU prefix state or activation differs from source")
    if args.activation.exists():
        with np.load(args.activation, allow_pickle=False) as file:
            if not np.array_equal(file["mul_201"], activation) or not np.array_equal(file["logits"], result["logits"]):
                raise ValueError("existing captured activation changed")
    else:
        np.savez(args.activation, mul_201=activation, logits=result["logits"])
    report = {
        "scope": "one real-weight MiniCPM-o speech-token projection after CPU speech-head state update; component only",
        "source_sha256": SOURCE_SHA, "fixture_sha256": FIXTURE_SHA,
        "reference_sha256": REFERENCE_SHA,
        "projection_sha256": sha256(args.output), "projection_bytes": args.output.stat().st_size,
        "cpu_prefix_sha256": sha256(args.prefix_output),
        "cpu_prefix_bytes": args.prefix_output.stat().st_size,
        "activation_sha256": sha256(args.activation),
        "onnxruntime": ort.__version__, "full_cpu_step_s": full_cpu_s,
        "cpu_prefix_step_s": prefix_cpu_s,
        "cpu_prefix_cache_and_activation_bitwise_parity": True,
        "cpu_projection_bitwise_parity": True,
        "cpu_top1": int(np.argmax(projected)), "status": "cpu_projection_extracted",
    }
    write(args.report, report)


def quantize(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    source = json.loads(args.export_report.read_text(encoding="utf-8"))
    if source.get("projection_sha256") != sha256(args.model) or source.get("activation_sha256") != sha256(args.activation):
        raise ValueError("projection or calibration activation changed")
    with np.load(args.activation, allow_pickle=False) as file:
        activation = np.ascontiguousarray(file["mul_201"])
        reference = file["logits"]

    class Reader(CalibrationDataReader):
        sent = False

        def get_next(self):
            if self.sent:
                return None
            self.sent = True
            return {"mul_201": activation}

    started = time.perf_counter()
    quantize_static(
        str(args.model), str(args.output), Reader(), quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt16, weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"], calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )
    quantization_s = time.perf_counter() - started
    cpu = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    projected = cpu.run(None, {"mul_201": activation})[0]
    report = {
        "scope": "single-fixture A16W8 speech-token projection candidate; not representative calibration or E2E",
        "source_projection_sha256": source["projection_sha256"],
        "activation_sha256": source["activation_sha256"],
        "candidate_sha256": sha256(args.output), "candidate_bytes": args.output.stat().st_size,
        "onnxruntime": ort.__version__, "quantization_s": quantization_s,
        "calibration": "one captured normalized real-model activation, MinMax per-channel A16W8 QDQ",
        "cpu_relative_l2_from_fp32": relative_l2(reference, projected),
        "cpu_top1": int(np.argmax(projected)),
    }
    report["status"] = "cpu_candidate_numeric_pass" if report["cpu_top1"] == source["cpu_top1"] and report["cpu_relative_l2_from_fp32"] <= 0.01 else "cpu_candidate_numeric_fail"
    write(args.report, report)


def probe(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort

    candidate = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    source = json.loads(args.export_report.read_text(encoding="utf-8"))
    if (candidate["candidate_sha256"] != sha256(args.model)
            or source["projection_sha256"] != sha256(args.fp32_model)
            or source["activation_sha256"] != sha256(args.activation)):
        raise ValueError("candidate or activation hash changed")
    if candidate["status"] != "cpu_candidate_numeric_pass":
        raise ValueError("candidate failed its CPU numerical gate")
    with np.load(args.activation, allow_pickle=False) as file:
        activation = np.ascontiguousarray(file["mul_201"])
        reference = file["logits"]
    fp32_cpu = ort.InferenceSession(str(args.fp32_model), providers=["CPUExecutionProvider"])
    fp32_samples = []
    for _ in range(args.samples):
        started = time.perf_counter()
        full_precision = fp32_cpu.run(None, {"mul_201": activation})[0]
        fp32_samples.append((time.perf_counter() - started) * 1000)
        if not np.array_equal(full_precision, reference):
            raise ValueError("FP32 CPU projection differs from captured source logits")
    cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    cpu_logits = cpu.run(None, {"mul_201": activation})[0]
    cpu_samples = []
    for _ in range(args.samples):
        started = time.perf_counter()
        repeated = cpu.run(None, {"mul_201": activation})[0]
        cpu_samples.append((time.perf_counter() - started) * 1000)
        if not np.array_equal(repeated, cpu_logits):
            raise ValueError("CPU projection output changed across identical calls")
    ep_dir = args.ep_dir.resolve(strict=True)
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library("vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
    devices = [device for device in ort.get_ep_devices()
               if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
    if not devices:
        raise RuntimeError("VitisAI NPU device unavailable")
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    options.enable_profiling = True
    options.profile_file_prefix = str(args.profile_prefix)
    report = {
        "scope": "one real MiniCPM-o speech-token projection on HX370 AMD NPU; CPU retains speech state; not E2E",
        "os": platform.platform(), "onnxruntime": ort.__version__,
        "candidate_sha256": candidate["candidate_sha256"],
        "activation_sha256": source["activation_sha256"],
        "status": "npu_session_creation_started",
    }
    write(args.report, report)
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options)
    report["npu_session_create_s"] = time.perf_counter() - started
    report["status"] = "npu_session_created"
    write(args.report, report)
    for _ in range(args.warmup):
        session.run(None, {"mul_201": activation})
    samples = []
    candidate_errors = []
    fp32_errors = []
    top_tokens = []
    for _ in range(args.samples):
        started = time.perf_counter()
        actual = session.run(None, {"mul_201": activation})[0]
        samples.append((time.perf_counter() - started) * 1000)
        if not np.isfinite(actual).all():
            raise ValueError("NPU projection produced non-finite logits")
        candidate_errors.append(relative_l2(cpu_logits, actual))
        fp32_errors.append(relative_l2(reference, actual))
        top_tokens.append(int(np.argmax(actual)))
    path = Path(session.end_profiling())
    events = json.loads(path.read_text(encoding="utf-8"))
    providers = Counter((event.get("args") or {}).get("provider") for event in events if event.get("cat") == "Node")
    samples.sort()
    cpu_samples.sort()
    fp32_samples.sort()
    report.update({
        "profile_file": str(path), "node_providers": dict(providers),
        "warmup": args.warmup, "samples_ms": samples,
        "fp32_cpu_samples_ms": fp32_samples,
        "fp32_cpu_step_p50_ms": fp32_samples[(len(fp32_samples) - 1) // 2],
        "fp32_cpu_step_p95_ms": fp32_samples[max(0, (95 * len(fp32_samples) + 99) // 100 - 1)],
        "cpu_samples_ms": cpu_samples,
        "cpu_step_p50_ms": cpu_samples[(len(cpu_samples) - 1) // 2],
        "cpu_step_p95_ms": cpu_samples[max(0, (95 * len(cpu_samples) + 99) // 100 - 1)],
        "npu_step_p50_ms": samples[(len(samples) - 1) // 2],
        "npu_step_p95_ms": samples[max(0, (95 * len(samples) + 99) // 100 - 1)],
        "npu_vs_candidate_cpu_relative_l2_max": max(candidate_errors),
        "npu_vs_fp32_relative_l2_max": max(fp32_errors),
        "npu_top1_set": sorted(set(top_tokens)), "cpu_top1": int(np.argmax(cpu_logits)),
    })
    report["status"] = ("component_npu_numeric_pass" if providers.get("vitisai", 0) > 0
                        and report["npu_top1_set"] == [report["cpu_top1"]] == [source["cpu_top1"]]
                        and report["npu_vs_candidate_cpu_relative_l2_max"] <= 0.01
                        and report["npu_vs_fp32_relative_l2_max"] <= 0.01 else "component_not_qualified")
    write(args.report, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    extracting = sub.add_parser("export")
    for name in ("source", "fixture", "reference", "output", "prefix-output", "activation", "report"):
        extracting.add_argument(f"--{name}", type=Path, required=True)
    quantizing = sub.add_parser("quantize")
    for name in ("model", "activation", "export-report", "output", "report"):
        quantizing.add_argument(f"--{name}", type=Path, required=True)
    probing = sub.add_parser("probe")
    for name in ("model", "fp32-model", "activation", "export-report", "candidate-report", "ep-dir", "report", "profile-prefix"):
        probing.add_argument(f"--{name}", type=Path, required=True)
    probing.add_argument("--warmup", type=int, default=1)
    probing.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.mode == "export":
        export(args)
    elif args.mode == "quantize":
        quantize(args)
    else:
        if args.warmup < 0 or args.samples < 1:
            raise ValueError("warmup must be nonnegative and samples positive")
        probe(args)


if __name__ == "__main__":
    main()
