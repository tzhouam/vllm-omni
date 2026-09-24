#!/usr/bin/env python3
"""Attempt an analytical BF16 ONNX resampler control and reject it if divergent."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np


CASES = ("red", "blue", "third")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, observed: np.ndarray) -> float:
    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--npu-output", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--npu-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()

    exported = json.loads(args.export_report.read_text(encoding="utf-8-sig"))
    measured = json.loads(args.npu_report.read_text(encoding="utf-8-sig"))
    if (sha256(args.model) != exported["wrapped_sha256"]
            or sha256(args.inputs) != exported["inputs_sha256"]
            or sha256(args.reference) != exported["reference_sha256"]
            or sha256(args.npu_output) != measured["output_npz_sha256"]
            or measured["model_sha256"] != exported["wrapped_sha256"]
            or measured["node_counts"].get("vitisai", 0) < len(CASES)):
        raise ValueError("pinned BF16 resampler export or measured NPU output changed")
    with np.load(args.inputs, allow_pickle=False) as data:
        inputs = {name: np.ascontiguousarray(data[name]) for name in args.cases}
    with np.load(args.reference, allow_pickle=False) as data:
        references = {name: np.ascontiguousarray(data[name]) for name in args.cases}
    with np.load(args.npu_output, allow_pickle=False) as data:
        npu_outputs = {name: np.ascontiguousarray(data[name]) for name in args.cases}
    for name in args.cases:
        if (inputs[name].shape != (1, 1024, 1152)
                or inputs[name].dtype != np.float32
                or references[name].shape != (1, 64, 4096)
                or npu_outputs[name].shape != references[name].shape):
            raise ValueError(f"{name} resampler tensor contract changed")

    import onnx
    import ml_dtypes
    from onnx.reference import ReferenceEvaluator
    from onnx.reference.op_run import OpRun

    class MatMul(OpRun):
        op_domain = ""

        def _run(self, a, b):
            # NumPy's large BF16 matmul returns FP32 on this build. ONNX
            # declares BF16 output, so retain FP32 accumulation and round at
            # the operator boundary. This is an analytical CPU control, not
            # a bitwise implementation of the VitisAI or Torch kernel.
            result = np.matmul(a.astype(np.float32), b.astype(np.float32))
            if a.dtype == ml_dtypes.bfloat16 and b.dtype == ml_dtypes.bfloat16:
                result = result.astype(ml_dtypes.bfloat16)
            return (result,)

    started = time.perf_counter()
    evaluator = ReferenceEvaluator(onnx.load_model(args.model), new_ops=[MatMul])
    load_s = time.perf_counter() - started
    report = {
        "scope": "analytical ONNX ReferenceEvaluator attempt for the BF16 resampler; not an execution-equivalent backend",
        "model_sha256": sha256(args.model),
        "inputs_sha256": sha256(args.inputs),
        "torch_reference_sha256": sha256(args.reference),
        "npu_output_sha256": sha256(args.npu_output),
        "onnx": onnx.__version__,
        "matmul_reference_override": "FP32 accumulation rounded to declared BF16 output",
        "platform": platform.platform(),
        "evaluator_load_s": load_s,
        "cases": {},
    }
    for name in args.cases:
        started = time.perf_counter()
        output = evaluator.run(None, {"hidden_fp32": inputs[name]})[0]
        call_s = time.perf_counter() - started
        if output.shape != (1, 64, 4096) or not np.isfinite(output).all():
            raise ValueError(f"{name} ONNX reference output contract failed")
        report["cases"][name] = {
            "reference_evaluator_s": call_s,
            "onnx_vs_torch_bf16_relative_l2": relative_l2(references[name], output),
            "npu_vs_onnx_relative_l2": relative_l2(output, npu_outputs[name]),
            "npu_vs_torch_bf16_relative_l2": relative_l2(references[name], npu_outputs[name]),
        }
    report["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["status"] = (
        "reference_evaluator_numeric_pass_on_selected_cases"
        if all(case["onnx_vs_torch_bf16_relative_l2"] < 0.01
               for case in report["cases"].values())
        else "reference_evaluator_unqualified"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
