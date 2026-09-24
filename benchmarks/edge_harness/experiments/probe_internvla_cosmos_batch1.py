#!/usr/bin/env python3
"""Check whether batch-one execution repairs the Cosmos Conv13 VitisAI output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path


SOURCE_SHA = "ff2ebad2c59a472b561536fa49cdd8bc43cf8e7bcf000a8ff4420f8e3a4766c8"
FIXTURE_SHA = "7bca8b4dcafa0ab0d45387e95a4d3e844f04f296f9253a4d4ac54ab5682d068c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def relative_l2(reference, actual) -> float:
    import numpy as np

    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return float(np.linalg.norm(delta) / max(np.linalg.norm(reference.astype(np.float64)), 1e-12))


def nearest_rank(values: list[float], percent: int) -> float:
    ordered = sorted(values)
    return ordered[(percent * len(ordered) + 99) // 100 - 1]


def fixtures(path: Path):
    import numpy as np

    if sha256(path) != FIXTURE_SHA:
        raise ValueError("Cosmos boundary activation fixture changed")
    with np.load(path, allow_pickle=False) as file:
        result = {name: np.ascontiguousarray(file[f"{name}_group_norm"]) for name in ("ramp", "pattern")}
    if any(value.shape != (6, 128, 64, 64) or value.dtype != np.float32 for value in result.values()):
        raise ValueError("Cosmos boundary activation contract changed")
    return result


def prepare(args: argparse.Namespace) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    if sha256(args.source) != SOURCE_SHA:
        raise ValueError("real Cosmos Sigmoid+Mul+Conv boundary changed")
    cases = fixtures(args.fixtures)
    model = onnx.load(args.source)
    if [node.op_type for node in model.graph.node] != ["Sigmoid", "Mul", "Conv"]:
        raise ValueError("Cosmos boundary operations changed")
    changed = []
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        dimensions = value.type.tensor_type.shape.dim
        if dimensions and dimensions[0].dim_value == 6:
            dimensions[0].dim_value = 1
            changed.append(value.name)
    if "group_norm" not in changed or "conv2d_13" not in changed:
        raise ValueError("batch dimension was not changed at both boundary tensors")
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    full_cpu = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    single_cpu = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    errors = {}
    for name, data in cases.items():
        full = full_cpu.run(None, {"group_norm": data})[0]
        assembled = np.concatenate([single_cpu.run(None, {"group_norm": data[i:i + 1]})[0] for i in range(6)], axis=0)
        errors[name] = [relative_l2(full[i], assembled[i]) for i in range(6)]
    report = {
        "scope": "real Cosmos Sigmoid+Mul+Conv13 rebatch from six to one; CPU parity only",
        "source_sha256": SOURCE_SHA, "boundary_fixtures_sha256": FIXTURE_SHA,
        "candidate_sha256": sha256(args.output), "candidate_bytes": args.output.stat().st_size,
        "batch_dim_changed_for": changed, "onnxruntime": ort.__version__,
        "cpu_batch1_vs_batch6_relative_l2": errors,
        "status": "cpu_rebatch_pass" if max(max(value) for value in errors.values()) <= 1e-5 else "cpu_rebatch_fail",
    }
    write(args.report, report)
    if report["status"] != "cpu_rebatch_pass":
        raise ValueError("batch-one FP32 graph differs from batch-six source")


def quantize(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    prepared = json.loads(args.preparation_report.read_text(encoding="utf-8"))
    if prepared["candidate_sha256"] != sha256(args.model):
        raise ValueError("batch-one FP32 source changed")
    cases = fixtures(args.fixtures)
    calibration = [("ramp", i) for i in range(6)] + [("pattern", 0)]

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iter = iter({"group_norm": cases[name][i:i + 1]} for name, i in calibration)

        def get_next(self):
            return next(self._iter, None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    quantize_static(
        str(args.model), str(args.output), Reader(), quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt16, weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"], nodes_to_quantize=["node_conv2d_13"],
        calibrate_method=CalibrationMethod.MinMax, per_channel=True,
    )
    quantization_s = time.perf_counter() - start
    fp32 = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    qdq = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    errors = {}
    for name, data in cases.items():
        errors[name] = []
        for i in range(6):
            value = {"group_norm": data[i:i + 1]}
            expected = fp32.run(None, value)[0]
            actual = qdq.run(None, value)[0]
            errors[name].append(relative_l2(expected, actual))
    report = {
        "scope": "real Cosmos Conv13 batch-one A16W8 QDQ on seven synthetic calibration frames; component only",
        "source_sha256": prepared["candidate_sha256"],
        "boundary_fixtures_sha256": FIXTURE_SHA,
        "candidate_sha256": sha256(args.output), "candidate_bytes": args.output.stat().st_size,
        "onnxruntime": ort.__version__, "quantization_s": quantization_s,
        "calibration_frames": [f"{name}:{i}" for name, i in calibration],
        "cpu_qdq_vs_fp32_relative_l2": errors,
        "status": "cpu_candidate_numeric_pass" if max(max(value) for value in errors.values()) <= 0.01 else "cpu_candidate_numeric_fail",
    }
    write(args.report, report)


def probe(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort

    prepared = json.loads(args.preparation_report.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    if (prepared["status"] != "cpu_rebatch_pass" or candidate["status"] != "cpu_candidate_numeric_pass"
            or candidate["candidate_sha256"] != sha256(args.model)
            or prepared["candidate_sha256"] != sha256(args.fp32_model)
            or sha256(args.source) != SOURCE_SHA):
        raise ValueError("batch-one artifact or prior CPU gate changed")
    cases = fixtures(args.fixtures)
    full_cpu = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    qdq_cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
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
        "scope": "real Cosmos Conv13 six-frame boundary through six batch-one AMD NPU calls; not full encoder or policy",
        "os": platform.platform(), "onnxruntime": ort.__version__,
        "source_sha256": SOURCE_SHA, "batch1_fp32_sha256": prepared["candidate_sha256"],
        "candidate_sha256": candidate["candidate_sha256"],
        "boundary_fixtures_sha256": FIXTURE_SHA,
        "status": "npu_session_creation_started",
    }
    write(args.report, report)
    start = time.perf_counter()
    npu = ort.InferenceSession(str(args.model), sess_options=options)
    report["npu_session_create_s"] = time.perf_counter() - start
    report["status"] = "npu_session_created"
    write(args.report, report)
    npu.run(None, {"group_norm": cases["pattern"][:1]})
    errors_qdq = {}
    errors_fp32 = {}
    case_wall_ms = {}
    assembled_outputs = {}
    reference_outputs = {}
    times_ms = []
    for name, data in cases.items():
        fp32_full = full_cpu.run(None, {"group_norm": data})[0]
        npu_parts = []
        errors_qdq[name] = []
        errors_fp32[name] = []
        case_wall_ms[name] = []
        for i in range(6):
            value = {"group_norm": data[i:i + 1]}
            cpu_qdq = qdq_cpu.run(None, value)[0]
            start = time.perf_counter()
            actual = npu.run(None, value)[0]
            elapsed_ms = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed_ms)
            case_wall_ms[name].append(elapsed_ms)
            if actual.shape != (1, 256, 64, 64) or not np.isfinite(actual).all():
                raise ValueError("batch-one NPU output contract failed")
            errors_qdq[name].append(relative_l2(cpu_qdq, actual))
            errors_fp32[name].append(relative_l2(fp32_full[i:i + 1], actual))
            npu_parts.append(actual)
        assembled = np.concatenate(npu_parts, axis=0)
        if assembled.shape != fp32_full.shape:
            raise ValueError("assembled six-frame shape changed")
        assembled_outputs[name] = assembled
        reference_outputs[name] = fp32_full
    repeated_cpu_ms = []
    repeated_npu_ms = []
    pattern = cases["pattern"]
    pattern_reference = full_cpu.run(None, {"group_norm": pattern})[0]
    for trial in range(args.profile_requests):
        if trial % 2 == 0:
            start = time.perf_counter()
            cpu_result = full_cpu.run(None, {"group_norm": pattern})[0]
            repeated_cpu_ms.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            npu_result = np.concatenate([npu.run(None, {"group_norm": pattern[i:i + 1]})[0]
                                         for i in range(6)], axis=0)
            repeated_npu_ms.append((time.perf_counter() - start) * 1000)
        else:
            start = time.perf_counter()
            npu_result = np.concatenate([npu.run(None, {"group_norm": pattern[i:i + 1]})[0]
                                         for i in range(6)], axis=0)
            repeated_npu_ms.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            cpu_result = full_cpu.run(None, {"group_norm": pattern})[0]
            repeated_cpu_ms.append((time.perf_counter() - start) * 1000)
        if not np.array_equal(cpu_result, pattern_reference) or relative_l2(pattern_reference, npu_result) > 0.01:
            raise ValueError("repeated six-frame boundary output changed")
    profile = Path(npu.end_profiling())
    providers = Counter((event.get("args") or {}).get("provider") for event in json.loads(profile.read_text(encoding="utf-8")) if event.get("cat") == "Node")
    times_ms.sort()
    report.update({
        "profile_file": str(profile), "node_providers": dict(providers),
        "warmup": 1, "measured_frames": len(times_ms), "frame_wall_ms": times_ms,
        "frame_wall_p50_ms": times_ms[(len(times_ms) + 1) // 2 - 1],
        "frame_wall_p95_ms": times_ms[(95 * len(times_ms) + 99) // 100 - 1],
        "six_frame_sequential_npu_sum_ms": {name: sum(values) for name, values in case_wall_ms.items()},
        "repeated_pattern_requests": args.profile_requests,
        "repeated_pattern_cpu_batch6_wall_ms": repeated_cpu_ms,
        "repeated_pattern_npu_six_batch1_wall_ms": repeated_npu_ms,
        "repeated_pattern_cpu_p50_p95_ms": [nearest_rank(repeated_cpu_ms, 50), nearest_rank(repeated_cpu_ms, 95)],
        "repeated_pattern_npu_p50_p95_ms": [nearest_rank(repeated_npu_ms, 50), nearest_rank(repeated_npu_ms, 95)],
        "npu_vs_qdq_cpu_relative_l2": errors_qdq,
        "npu_vs_fp32_batch6_relative_l2": errors_fp32,
        "max_npu_vs_qdq_cpu_relative_l2": max(max(value) for value in errors_qdq.values()),
        "max_npu_vs_fp32_batch6_relative_l2": max(max(value) for value in errors_fp32.values()),
    })
    if args.save_assembled is not None:
        args.save_assembled.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_assembled, **{
            f"{name}_{kind}": value for name in cases
            for kind, value in (("npu", assembled_outputs[name]), ("cpu", reference_outputs[name]))
        })
        report["assembled_outputs_sha256"] = sha256(args.save_assembled)
        report["assembled_outputs_bytes"] = args.save_assembled.stat().st_size
    report["status"] = ("component_npu_numeric_pass" if providers.get("vitisai", 0) >= 13 + 6 * args.profile_requests
                        and report["max_npu_vs_qdq_cpu_relative_l2"] <= 0.01
                        and report["max_npu_vs_fp32_batch6_relative_l2"] <= 0.01
                        else "component_not_qualified")
    write(args.report, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("prepare")
    for name in ("source", "fixtures", "output", "report"):
        a.add_argument(f"--{name}", type=Path, required=True)
    b = sub.add_parser("quantize")
    for name in ("model", "fixtures", "preparation-report", "output", "report"):
        b.add_argument(f"--{name}", type=Path, required=True)
    c = sub.add_parser("probe")
    for name in ("source", "model", "fp32-model", "fixtures", "preparation-report", "candidate-report", "ep-dir", "profile-prefix", "report"):
        c.add_argument(f"--{name}", type=Path, required=True)
    c.add_argument("--profile-requests", type=int, default=20)
    c.add_argument("--save-assembled", type=Path)
    args = parser.parse_args()
    if args.mode == "probe" and args.profile_requests < 1:
        raise ValueError("profile-requests must be positive")
    {"prepare": prepare, "quantize": quantize, "probe": probe}[args.mode](args)


if __name__ == "__main__":
    main()
