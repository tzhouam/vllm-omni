#!/usr/bin/env python3
"""Numerically gate one Workbench QDQ vocoder export on local ONNX Runtime CPU.

The Workbench quantize job and any compile-time quantization are separate
artifacts. This probe measures only the explicitly downloaded QDQ ONNX model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from audit_qwen_tts_vocoder_qualcomm import (
    CPU_OUTPUT_SHA256,
    FIXTURE_SHA256,
    SOURCE_SHA256,
    compare,
    read_output,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model", "archive", "source-onnx", "fixture", "cpu-reference",
        "quantize-report", "output", "report",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--other-output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")

    quantized = json.loads(args.quantize_report.read_text(encoding="utf-8"))
    if (quantized.get("status") != "SUCCESS"
            or quantized.get("source_model_id") != "mqyer339n"
            or quantized.get("calibration_dataset_id") != "d7m80jl32"
            or not quantized.get("target_model_id")):
        raise ValueError("quantize job differs from pinned vocoder source and fixture")
    if [sha256(path) for path in (args.source_onnx, args.fixture, args.cpu_reference)] != [
        SOURCE_SHA256, FIXTURE_SHA256, CPU_OUTPUT_SHA256,
    ]:
        raise ValueError("source, fixture or CPU reference differs from pinned data")
    with np.load(args.fixture, allow_pickle=False) as fixture:
        if fixture.files != ["quantized"]:
            raise ValueError("unexpected vocoder input name")
        x = np.asarray(fixture["quantized"])
    if x.shape != (1, 512, 97) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("invalid vocoder fixture shape, dtype or values")

    optimizer_error = None
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    try:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        default_session = ort.InferenceSession(
            str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        del default_session
    except Exception as exc:
        optimizer_error = str(exc)

    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    load_s = time.perf_counter() - started
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("model did not stay on ONNX Runtime CPU")
    started = time.perf_counter()
    y = np.asarray(session.run(["wav"], {"quantized": x})[0])
    inference_s = time.perf_counter() - started
    if y.shape != (1, 48000) or y.dtype != np.float32 or not np.isfinite(y).all():
        raise ValueError("invalid QDQ vocoder output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, wav=y)

    report = {
        "scope": "one Workbench QDQ vocoder export on local ONNX Runtime CPU; no device-local or complete TTS stream",
        "status": "qdq_cpu_numeric_measured",
        "quantize_job_id": quantized["job_id"],
        "target_model_id": quantized["target_model_id"],
        "calibration_dataset_id": quantized["calibration_dataset_id"],
        "source_sha256": SOURCE_SHA256,
        "fixture_sha256": FIXTURE_SHA256,
        "cpu_reference_sha256": CPU_OUTPUT_SHA256,
        "target_archive_sha256": sha256(args.archive),
        "target_onnx_sha256": sha256(args.model),
        "target_external_data_sha256": sha256(args.model.parent / "model.data"),
        "cpu_qdq_output_sha256": sha256(args.output),
        "ort_version": ort.__version__,
        "provider": session.get_providers(),
        "intra_op_threads": args.threads,
        "graph_optimization_level": "ORT_ENABLE_BASIC",
        "default_ORT_ENABLE_ALL_error": optimizer_error,
        "load_s_not_request_timing": load_s,
        "one_inference_s_not_profile": inference_s,
        "source_onnx_cpu_vs_qdq_cpu": compare(
            read_output(args.cpu_reference, "wav"), y
        ),
        "limits": [
            "One synthetic calibration fixture cannot establish audio quality or generalization.",
            "The explicit QDQ quantize job and compile-time INT8 conversion are distinct artifacts.",
            "This local CPU run is not an embedded-device profile or complete TTS stream.",
        ],
    }
    if args.other_output is not None:
        report["qdq_cpu_vs_distinct_hosted_output"] = compare(
            y, read_output(args.other_output, "output_0__0")
        )
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["source_onnx_cpu_vs_qdq_cpu"], indent=2))


if __name__ == "__main__":
    main()
