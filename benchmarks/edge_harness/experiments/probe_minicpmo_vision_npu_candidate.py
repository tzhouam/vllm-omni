#!/usr/bin/env python3
"""Quantize and probe the pinned MiniCPM-o vision prefix on native AMD NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path


SOURCE_SHA = "bd3e7802a78a930c5cd3e8e1a9888ab73e3fa865d6f831e492e169cdd24540be"
FIXTURE_SHA = "9e58e815ccf46039b28d70f1b952f79ccc27b97a57ab44eaed50bd5218f6bcd2"
SOURCE_SHA3 = "b68fe07f9ae74d2c4d3a953b43ce5b33e28f86a41f413f857d2e00edf67b9efb"
FIXTURE_SHA3 = "cd25c58727814ee306e4c4b88ed78d89a2fc2de1cc2d5caca90782264bd02fc4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def relative_l2(reference, actual) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = actual.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-prefix", type=Path)
    parser.add_argument("--ep-dir", type=Path)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--layers", type=int, choices=(3, 4), default=4)
    parser.add_argument("--exclude-last-layer", action="store_true")
    parser.add_argument("--fixture-sha256")
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--fixture-role", choices=("held_out", "calibration_extra"), default="held_out")
    parser.add_argument("--extra-calibration-fixture", type=Path)
    parser.add_argument("--extra-calibration-sha256")
    args = parser.parse_args()
    source_sha = SOURCE_SHA3 if args.layers == 3 else SOURCE_SHA
    calibration_sha = FIXTURE_SHA3 if args.layers == 3 else FIXTURE_SHA
    if args.samples < 1 or bool(args.ep_dir) != bool(args.profile_prefix):
        parser.error("samples must be positive; ep-dir and profile-prefix must be supplied together")
    if args.layers == 3 and args.exclude_last_layer:
        parser.error("three-layer graph has no fourth layer to exclude")
    if sha256(args.source) != source_sha or sha256(args.fixture) != (args.fixture_sha256 or calibration_sha):
        raise ValueError("vision source or calibration fixture changed")
    if args.fixture_sha256 and not args.candidate_sha256:
        parser.error("held-out input requires a pinned pre-existing candidate")
    if bool(args.extra_calibration_fixture) != bool(args.extra_calibration_sha256):
        parser.error("extra calibration fixture and SHA-256 must be supplied together")
    if args.extra_calibration_fixture and (
        args.fixture_sha256 or sha256(args.extra_calibration_fixture) != args.extra_calibration_sha256
    ):
        raise ValueError("extra calibration fixture changed or input is held out")
    if args.candidate_sha256 and (not args.candidate.is_file()
                                  or sha256(args.candidate) != args.candidate_sha256):
        raise ValueError("held-out candidate hash differs from pinned artifact")

    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    with np.load(args.fixture, allow_pickle=False) as fixture:
        hidden = np.ascontiguousarray(fixture["hidden"])
        fp32 = np.ascontiguousarray(fixture["ort_fp32"])
        bf16 = np.ascontiguousarray(fixture["torch_bf16"])
    if (hidden.shape != (1, 1024, 1152) or fp32.shape != hidden.shape
            or bf16.shape != hidden.shape or hidden.dtype != np.float32):
        raise ValueError("vision fixture contract changed")
    calibration_inputs = [hidden]
    if args.extra_calibration_fixture:
        with np.load(args.extra_calibration_fixture, allow_pickle=False) as extra:
            extra_hidden = np.ascontiguousarray(extra["hidden"])
        if extra_hidden.shape != hidden.shape or extra_hidden.dtype != hidden.dtype:
            raise ValueError("extra calibration activation contract changed")
        calibration_inputs.append(extra_hidden)

    class Reader(CalibrationDataReader):
        index = 0

        def get_next(self):
            if self.index == len(calibration_inputs):
                return None
            value = calibration_inputs[self.index]
            self.index += 1
            return {"hidden": value}

    if not args.candidate.exists():
        excluded = []
        if args.exclude_last_layer:
            import onnx

            graph = onnx.load(str(args.source), load_external_data=False)
            excluded = [node.name for node in graph.graph.node
                        if node.name.startswith("/layers.3/")
                        and node.op_type in {"MatMul", "Gemm"}]
            if len(excluded) != 8:
                raise ValueError("last vision layer quantization boundary changed")
        args.candidate.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        quantize_static(
            str(args.source), str(args.candidate), Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt16,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            calibrate_method=CalibrationMethod.MinMax,
            per_channel=True,
            nodes_to_exclude=excluded,
        )
        quantize_s = time.perf_counter() - started
    else:
        quantize_s = None

    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    cpu = ort.InferenceSession(str(args.candidate), sess_options=options,
                               providers=["CPUExecutionProvider"])
    if [item.name for item in cpu.get_inputs()] != ["hidden"]:
        raise ValueError("quantized vision input contract changed")
    cpu_samples = []
    cpu_output = None
    for _ in range(args.samples):
        started = time.perf_counter()
        output = cpu.run(None, {"hidden": hidden})[0]
        cpu_samples.append((time.perf_counter() - started) * 1000)
        if cpu_output is not None and not np.array_equal(output, cpu_output):
            raise RuntimeError("quantized CPU vision output is not repeatable")
        cpu_output = output
    cpu_error_fp32 = relative_l2(fp32, cpu_output)
    cpu_error_bf16 = relative_l2(bf16, cpu_output)
    source_opsets = [(item.domain, item.version) for item in onnx.load(
        str(args.source), load_external_data=False).opset_import]
    candidate_opsets = [(item.domain, item.version) for item in onnx.load(
        str(args.candidate), load_external_data=False).opset_import]

    report = {
        "scope": f"{args.layers} real-weight MiniCPM-o vision layers on one fixed synthetic image; component only",
        "source_sha256": source_sha,
        "fixture_sha256": sha256(args.fixture),
        "input_role": args.fixture_role if args.fixture_sha256 else "calibration",
        "calibration_fixture_sha256": calibration_sha,
        "candidate_sha256": sha256(args.candidate),
        "candidate_bytes": args.candidate.stat().st_size,
        "source_opsets": source_opsets,
        "candidate_opsets": candidate_opsets,
        "layout": "[batch, patches, hidden] = [1, 1024, 1152] FP32 input/output",
        "quantization": "A16W8 per-channel QDQ MatMul/Gemm, MinMax calibrated on pinned image activations",
        "extra_calibration_fixture_sha256": args.extra_calibration_sha256,
        "last_layer_quantization_excluded": args.exclude_last_layer,
        "quantization_s": quantize_s,
        "candidate_cpu_vs_fp32_relative_l2": cpu_error_fp32,
        "candidate_cpu_vs_source_bf16_relative_l2": cpu_error_bf16,
        "candidate_cpu_p50_ms": percentile(cpu_samples, .5),
        "candidate_cpu_p95_ms": percentile(cpu_samples, .95),
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "samples": args.samples,
        "npu": None,
    }
    if args.ep_dir is not None:
        ep_dir = args.ep_dir.resolve(strict=True)
        os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(ep_dir))
        ort.register_execution_provider_library(
            "vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll")
        )
        devices = [device for device in ort.get_ep_devices()
                   if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
        if not devices:
            raise RuntimeError("VitisAI NPU unavailable")
        profile_options = ort.SessionOptions()
        profile_options.add_provider_for_devices(devices, {})
        profile_options.enable_profiling = True
        profile_options.profile_file_prefix = str(args.profile_prefix)
        started = time.perf_counter()
        npu = ort.InferenceSession(str(args.candidate), sess_options=profile_options)
        load_s = time.perf_counter() - started
        npu_samples = []
        npu_error_cpu = []
        npu_error_fp32 = []
        first_npu_output = None
        for _ in range(args.samples):
            started = time.perf_counter()
            output = npu.run(None, {"hidden": hidden})[0]
            npu_samples.append((time.perf_counter() - started) * 1000)
            if first_npu_output is None:
                first_npu_output = output.copy()
            elif not np.array_equal(output, first_npu_output):
                raise RuntimeError("NPU vision output changed across identical calls")
            npu_error_cpu.append(relative_l2(cpu_output, output))
            npu_error_fp32.append(relative_l2(fp32, output))
        profile = Path(npu.end_profiling())
        events = json.loads(profile.read_text(encoding="utf-8-sig"))
        counts = {}
        for event in events:
            if event.get("cat") == "Node":
                name = (event.get("args") or {}).get("provider")
                counts[name] = counts.get(name, 0) + 1
        report["npu"] = {
            "device_type": str(devices[0].device.type),
            "providers": npu.get_providers(),
            "load_s": load_s,
            "profile_path": str(profile),
            "profile_sha256": sha256(profile),
            "node_counts": counts,
            "p50_ms": percentile(npu_samples, .5),
            "p95_ms": percentile(npu_samples, .95),
            "max_relative_l2_vs_qdq_cpu": max(npu_error_cpu),
            "max_relative_l2_vs_source_fp32": max(npu_error_fp32),
        }
        if args.output_npz is not None:
            args.output_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output_npz, npu=first_npu_output,
                                cpu_qdq=cpu_output)
            report["npu"]["output_npz"] = str(args.output_npz)
            report["npu"]["output_npz_sha256"] = sha256(args.output_npz)
    report["status"] = (
        "npu_numeric_pass_on_fixed_fixture"
        if report["npu"] and report["npu"]["node_counts"].get("vitisai", 0) > 0
        and report["npu"]["max_relative_l2_vs_qdq_cpu"] <= .01
        and report["npu"]["max_relative_l2_vs_source_fp32"] <= .01
        else "cpu_candidate_only_or_npu_unqualified"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
