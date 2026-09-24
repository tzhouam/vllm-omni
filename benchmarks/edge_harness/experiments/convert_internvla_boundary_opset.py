#!/usr/bin/env python3
"""Try an older ONNX opset for the isolated Cosmos stage."""

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
    parser.add_argument("--quantization-report", type=Path)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--boundary-fixtures", type=Path, required=True)
    parser.add_argument("--target-opset", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort

    quant = json.loads(args.quantization_report.read_text(encoding="utf-8")) if args.quantization_report else None
    prep = json.loads(args.preparation_report.read_text(encoding="utf-8"))
    expected_source_hash = quant["candidate_sha256"] if quant else prep["candidate_sha256"]
    if sha256(args.source) != expected_source_hash or sha256(args.boundary_fixtures) != prep["boundary_fixtures_sha256"]:
        raise ValueError("source or boundary fixture changed")
    original = onnx.load(str(args.source), load_external_data=True)
    converted = onnx.version_converter.convert_version(original, args.target_opset)
    onnx.checker.check_model(converted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, str(args.output))
    original_cpu = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    converted_cpu = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    comparison = {}
    with np.load(args.boundary_fixtures, allow_pickle=False) as data:
        for name in ("ramp", "pattern"):
            activation = np.ascontiguousarray(data[f"{name}_group_norm"])
            expected = original_cpu.run(None, {"group_norm": activation})[0]
            actual = converted_cpu.run(None, {"group_norm": activation})[0]
            delta = actual.astype(np.float64) - expected.astype(np.float64)
            comparison[name] = float(np.linalg.norm(delta) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12))
    report = {
        "scope": "isolated real-weight Cosmos stage opset compatibility candidate; no full encoder or policy claim",
        "source_candidate_sha256": expected_source_hash,
        "source_sha256": prep["candidate_sha256"],
        "boundary_fixtures_sha256": prep["boundary_fixtures_sha256"],
        "candidate_sha256": sha256(args.output),
        "candidate_bytes": args.output.stat().st_size,
        "source_opset": [(item.domain, item.version) for item in original.opset_import],
        "target_opset": [(item.domain, item.version) for item in converted.opset_import],
        "quantization": (quant["quantization"] if quant else "FP32") + f", converted to opset {args.target_opset}",
        "relative_l2_vs_original_cpu": comparison,
        "relative_l2_vs_fp32_cpu": quant["relative_l2_vs_fp32_cpu"] if quant else comparison,
        "status": "cpu_conversion_pass" if all(value <= 1e-5 for value in comparison.values()) else "cpu_conversion_numeric_fail",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
