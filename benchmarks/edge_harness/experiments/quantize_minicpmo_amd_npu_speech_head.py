#!/usr/bin/env python3
"""Probe A16W8 QDQ calibration quality for a real MiniCPM-o speech-head step."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


MODEL_SHA = "78cf64804f11ad269ee4180da61588ccc5eefe2942773227345499756e4cc229"
FIXTURE_SHA = "65919ea1f351feec33c1415a3e3db76d60bc9c2beb3d49a72af75dcb418e83b4"
REFERENCE_SHA = "902adcceadbe761d938ef2ade51cf953c8917ea1abfaee33a10c84e300c1bd96"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--node", action="append", help="Quantize only the named MatMul node; repeat for several")
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    if sha256(args.model) != MODEL_SHA or sha256(args.fixture) != FIXTURE_SHA or sha256(args.reference) != REFERENCE_SHA:
        raise ValueError("MiniCPM-o model, fixture or reference changed")
    with np.load(args.fixture, allow_pickle=False) as file:
        inputs = {name: np.ascontiguousarray(file[name]) for name in file.files}
    with np.load(args.reference, allow_pickle=False) as file:
        reference = {name: np.ascontiguousarray(file[name]) for name in file.files}
    if len(inputs) != 44 or len(reference) != 41:
        raise ValueError("speech-head fixture schema changed")

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._sent = False

        def get_next(self):
            if self._sent:
                return None
            self._sent = True
            return inputs

    report = {
        "scope": "real-weight 20-layer MiniCPM-o speech-head one-step A16W8 MatMul QDQ candidate; no full model claim",
        "model_sha256": MODEL_SHA, "fixture_sha256": FIXTURE_SHA,
        "reference_sha256": REFERENCE_SHA,
        "onnxruntime": ort.__version__,
        "calibration": "one pinned synthetic hidden/cache fixture, MinMax; not representative task calibration",
        "nodes_to_quantize": args.node or "all MatMul",
        "status": "quantization_started",
    }
    write(args.report, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    quantize_static(
        str(args.model), str(args.output), Reader(),
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt16,
        weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
        nodes_to_quantize=args.node,
        calibrate_method=CalibrationMethod.MinMax, per_channel=True,
    )
    report["quantization_s"] = time.perf_counter() - start
    report["candidate_sha256"] = sha256(args.output)
    report["candidate_bytes"] = args.output.stat().st_size
    report["status"] = "candidate_created"
    write(args.report, report)
    start = time.perf_counter()
    cpu = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    report["cpu_session_create_s"] = time.perf_counter() - start
    outputs = [item.name for item in cpu.get_outputs()]
    if set(outputs) != set(reference) or set(item.name for item in cpu.get_inputs()) != set(inputs):
        raise ValueError("quantized speech-head input/output schema changed")
    start = time.perf_counter()
    actual = dict(zip(outputs, cpu.run(None, inputs), strict=True))
    report["cpu_step_s"] = time.perf_counter() - start
    comparison = {}
    for name in outputs:
        if actual[name].shape != reference[name].shape or not np.isfinite(actual[name]).all():
            raise ValueError(f"quantized output contract failed: {name}")
        delta = actual[name].astype(np.float64) - reference[name].astype(np.float64)
        comparison[name] = float(np.linalg.norm(delta) / max(np.linalg.norm(reference[name].astype(np.float64)), 1e-12))
    report["cpu_top1"] = int(np.argmax(actual["logits"]))
    report["logits_relative_l2"] = comparison["logits"]
    report["max_cache_relative_l2"] = max(value for name, value in comparison.items() if name != "logits")
    report["output_relative_l2"] = comparison
    report["status"] = "cpu_candidate_numeric_pass" if report["cpu_top1"] == 1867 and report["logits_relative_l2"] <= 0.01 and report["max_cache_relative_l2"] <= 0.01 else "cpu_candidate_numeric_fail"
    write(args.report, report)


if __name__ == "__main__":
    main()
