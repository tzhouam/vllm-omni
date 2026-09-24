#!/usr/bin/env python3
"""Quantize an isolated real-weight Cosmos activation-to-convolution stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--boundary-fixtures", type=Path, required=True)
    parser.add_argument("--activation-bits", type=int, choices=(8, 16), required=True)
    parser.add_argument("--per-tensor", action="store_true", help="Use one weight scale instead of per-channel scales")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    preparation = json.loads(args.preparation_report.read_text(encoding="utf-8"))
    if sha256(args.source) != preparation["candidate_sha256"] or sha256(args.boundary_fixtures) != preparation["boundary_fixtures_sha256"]:
        raise ValueError("boundary graph or real activation fixtures changed")
    with np.load(args.boundary_fixtures, allow_pickle=False) as data:
        cases = [(name, np.ascontiguousarray(data[f"{name}_group_norm"])) for name in ("ramp", "pattern")]
    if any(value.shape != (6, 128, 64, 64) or value.dtype != np.float32 for _, value in cases):
        raise ValueError("boundary activation contract changed")

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iter = iter({"group_norm": value} for _, value in cases)

        def get_next(self):
            return next(self._iter, None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(args.source), str(args.output), Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8 if args.activation_bits == 8 else QuantType.QUInt16,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"],
        nodes_to_quantize=["node_conv2d_13"],
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=not args.per_tensor,
    )
    source_cpu = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    candidate_cpu = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    parity = {}
    for name, activation in cases:
        expected = source_cpu.run(None, {"group_norm": activation})[0]
        actual = candidate_cpu.run(None, {"group_norm": activation})[0]
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        parity[name] = float(np.linalg.norm(difference) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12))
    report = {
        "scope": "real-weight Cosmos activation-to-convolution subgraph QDQ candidate; no full encoder or policy claim",
        "source_sha256": preparation["candidate_sha256"],
        "boundary_fixtures_sha256": preparation["boundary_fixtures_sha256"],
        "candidate_sha256": sha256(args.output),
        "candidate_bytes": args.output.stat().st_size,
        "quantization": f"A{args.activation_bits}W8 Conv13 QDQ {'per-tensor' if args.per_tensor else 'per-channel'}, two pinned real activations MinMax",
        "relative_l2_vs_fp32_cpu": parity,
        "status": "cpu_candidate_numeric_pass" if all(value <= 0.01 for value in parity.values()) else "cpu_candidate_numeric_fail",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
