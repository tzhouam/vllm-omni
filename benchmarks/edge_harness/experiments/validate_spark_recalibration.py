# SPDX-License-Identifier: Apache-2.0
"""Check Spark output-head recalibrations on captured live activations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--old-graph", type=Path, required=True)
    parser.add_argument("--new-graph", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--reference-worker", type=Path)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()

    capture_report = json.loads(args.capture_report.read_text())
    capture_sha = sha256(args.capture)
    recorded_capture_sha = capture_report.get("capture_sha256") or capture_report["captured_activations"]["sha256"]
    if recorded_capture_sha != capture_sha:
        raise ValueError("live activation capture changed")
    with np.load(args.capture, allow_pickle=False) as archive:
        activations = archive["x"].copy()
    calibration = json.loads(args.calibration_manifest.read_text())
    if (calibration.get("live_capture_sha256") or calibration.get("capture_sha256")) != capture_sha:
        raise ValueError("calibration and validation captures differ")
    calibrated_indices = set(calibration["selected_capture_indices"])
    comparisons = (
        {row["call_index"]: row
         for row in json.loads(args.reference_worker.read_text())["reference_comparisons"]}
        if args.reference_worker else {}
    )
    if not all(0 <= i < len(activations) and (not comparisons or i in comparisons)
               for i in args.indices):
        raise ValueError("validation index is outside capture or BF16 reference")
    input_name = calibration.get("input_name", "x")

    sessions = {
        name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for name, path in (("source", args.source), ("old", args.old_graph), ("new", args.new_graph))
    }
    rows = []
    for index in args.indices:
        x = activations[index]
        outputs = {name: session.run(None, {input_name: x})[0].reshape(-1)
                   for name, session in sessions.items()}
        reference = comparisons.get(index)
        source = outputs["source"]
        rows.append({
            "index": index,
            "calibration_member": index in calibrated_indices,
            "activation_min": float(x.min()),
            "activation_max": float(x.max()),
            "bf16_top1_on_live_trajectory": reference["cpu_top1"] if reference else None,
            "source_fp32_top1": int(source.argmax()),
            **{
                name + "_top1": int(outputs[name].argmax())
                for name in ("old", "new")
            },
            **{
                name + "_vs_source_relative_l2": float(
                    np.linalg.norm(outputs[name] - source) / np.linalg.norm(source)
                )
                for name in ("old", "new")
            },
        })
    report = {
        "source_sha256": sha256(args.source),
        "old_graph_sha256": sha256(args.old_graph),
        "new_graph_sha256": sha256(args.new_graph),
        "capture_sha256": capture_sha,
        "calibration_manifest_sha256": sha256(args.calibration_manifest),
        "onnxruntime": ort.__version__,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
