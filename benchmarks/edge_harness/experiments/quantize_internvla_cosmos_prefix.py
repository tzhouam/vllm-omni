#!/usr/bin/env python3
"""Calibrate a real-weight Cosmos encoder prefix as a bounded A8W8 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


FIXTURES_SHA = "dcbaf1391ca914d4bbccc1cb384daf05666e771d9cdae90c4f069ec8fcf05368"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--node", action="append", help="Quantize only the named node; repeat for several")
    parser.add_argument("--activation-bits", type=int, choices=(8, 16), default=8)
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    if sha256(args.source) != args.source_sha256 or sha256(args.fixtures) != FIXTURES_SHA:
        raise ValueError("Cosmos source prefix or fixtures changed")
    with np.load(args.fixtures, allow_pickle=False) as data:
        cases = [(name, np.ascontiguousarray(data[f"{name}_pixels"])) for name in ("ramp", "pattern")]
    if any(pixels.shape != (6, 3, 256, 256) or pixels.dtype != np.float32 for _, pixels in cases):
        raise ValueError("Cosmos pixel fixture contract changed")

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iter = iter({"pixels": pixels} for _, pixels in cases)

        def get_next(self):
            return next(self._iter, None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    quantize_static(
        str(args.source), str(args.output), Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8 if args.activation_bits == 8 else QuantType.QUInt16,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul"],
        nodes_to_quantize=args.node,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )
    report = {
        "scope": f"real-weight Cosmos prefix A{args.activation_bits}W8 QDQ CPU candidate; no NPU or policy claim",
        "source_sha256": args.source_sha256,
        "fixtures_sha256": FIXTURES_SHA,
        "candidate_sha256": sha256(args.output),
        "candidate_bytes": args.output.stat().st_size,
        "quantize_s": time.perf_counter() - start,
        "quantization": f"QDQ Conv+MatMul A{args.activation_bits}W8, per-channel weights, MinMax on pinned ramp and pattern",
        "nodes_to_quantize": args.node or "all Conv+MatMul",
    }
    source_cpu = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    candidate_cpu = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    parity = {}
    for name, pixels in cases:
        expected = source_cpu.run(None, {"pixels": pixels})[0]
        actual = candidate_cpu.run(None, {"pixels": pixels})[0]
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        parity[name] = {
            "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12)),
            "max_abs": float(np.max(np.abs(difference))),
        }
    report["cpu_parity_vs_source"] = parity
    report["status"] = "cpu_candidate_numeric_pass" if all(item["relative_l2"] <= 0.01 for item in parity.values()) else "cpu_candidate_numeric_fail"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
