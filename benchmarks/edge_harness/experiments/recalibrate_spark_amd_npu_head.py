# SPDX-License-Identifier: Apache-2.0
"""Recalibrate the pinned Spark A16W8 output head with live activations."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


SOURCE_SHA = "1a426841ebb0e1255268b7c796bfcbe44072cf65613e47cfdf0c9d807746518b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class SparkCalibrationReader(CalibrationDataReader):
    def __init__(self, samples: np.ndarray) -> None:
        self.samples = samples
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.samples):
            return None
        sample = {"x": self.samples[self.index]}
        self.index += 1
        return sample

    def rewind(self) -> None:
        self.index = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-channel", action="store_true")
    args = parser.parse_args()

    if sha256(args.source) != SOURCE_SHA:
        raise ValueError("pinned pre-quantization Spark graph changed")
    manifest = json.loads(args.calibration_manifest.read_text())
    calibration_sha = sha256(args.calibration)
    if manifest["calibration_sha256"] != calibration_sha:
        raise ValueError("calibration inputs changed")
    with np.load(args.calibration, allow_pickle=False) as archive:
        samples = archive["x"].copy()
    if (samples.dtype != np.float32 or samples.ndim != 4
        or samples.shape[1:] != (1, 1, 2048)):
        raise ValueError("expected float32 [N,1,1,2048] pre-final-norm activations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(args.source), str(args.output), SparkCalibrationReader(samples),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        op_types_to_quantize=["MatMul"],
        calibrate_method=CalibrationMethod.MinMax,
    )
    report = {
        "scope": "recalibration only; numerical and NPU placement checks pending",
        "source_sha256": SOURCE_SHA,
        "calibration_sha256": calibration_sha,
        "calibration_samples": len(samples),
        "calibration_min": float(samples.min()),
        "calibration_max": float(samples.max()),
        "weight_quantization_per_channel": args.per_channel,
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
