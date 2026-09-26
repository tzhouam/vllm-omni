"""Profile a fixed ONNX component on native CPU with explicit ORT placement."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output-npz", type=Path)
    args = parser.parse_args()
    session = ort.InferenceSession(str(args.graph), providers=["CPUExecutionProvider"])
    with np.load(args.inputs) as loaded:
        inputs = {key: loaded[key] for key in loaded.files}
    if args.output_npz:
        np.savez(args.output_npz, output=session.run(None, inputs)[0])
    for _ in range(3):
        session.run(None, inputs)
    samples = []
    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(None, inputs)
        samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    report = {
        "graph": str(args.graph),
        "inputs": str(args.inputs),
        "onnxruntime": ort.__version__,
        "providers": session.get_providers(),
        "warmups": 3,
        "runs": args.runs,
        "samples_ms": samples,
        "p50_ms": statistics.median(samples),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "samples_ms"}, indent=2))


if __name__ == "__main__":
    main()
