#!/usr/bin/env python3
"""Compare a measured NPU-prefix waveform with the retained long CPU decode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def relative_l2(reference, actual) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = actual.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("real-report", "long-reference", "handoff-report", "waveforms", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.start_frame <= 23:
        parser.error("two-frame segment must fit the retained 25-frame window")

    import numpy as np

    provenance = json.loads(args.real_report.read_text(encoding="utf-8-sig"))
    handoff = json.loads(args.handoff_report.read_text(encoding="utf-8-sig"))
    if (provenance["model_revision"] != "85e237c12c027371202489a0ec509ded67b5e4b5"
            or sha256(args.long_reference) != provenance["files"]["ort"]["sha256"]
            or sha256(args.waveforms) != handoff["waveforms_sha256"]
            or handoff["npu_node_events"] < 1):
        raise ValueError("long reference, NPU placement or replay waveform changed")
    with np.load(args.long_reference, allow_pickle=False) as source:
        full = np.ascontiguousarray(source["wav"])
    with np.load(args.waveforms, allow_pickle=False) as source:
        short_cpu = np.ascontiguousarray(source["cpu_full"])
        injected = np.ascontiguousarray(source["npu_prefix_cpu_suffix"])
        zero = np.ascontiguousarray(source["zero_boundary_cpu_suffix"])
    reference = full[:, args.start_frame * 1920:(args.start_frame + 2) * 1920]
    if (full.shape != (1, 48000) or reference.shape != (1, 3840)
            or short_cpu.shape != reference.shape
            or injected.shape != reference.shape
            or zero.shape != reference.shape
            or not np.isfinite(injected).all()):
        raise ValueError("waveform shape or finiteness changed")
    report = {
        "scope": "one generated-code two-frame NPU-prefix plus CPU-suffix waveform against long CPU decode",
        "model_revision": provenance["model_revision"],
        "source_25_frame_onnx_sha256": provenance["onnx_sha256"],
        "long_reference_sha256": sha256(args.long_reference),
        "handoff_report_sha256": sha256(args.handoff_report),
        "waveforms_sha256": sha256(args.waveforms),
        "start_frame": args.start_frame,
        "cpu_short_vs_long_relative_l2": relative_l2(reference, short_cpu),
        "npu_prefix_cpu_suffix_vs_long_relative_l2": relative_l2(reference, injected),
        "zero_boundary_cpu_suffix_vs_long_relative_l2": relative_l2(reference, zero),
        "status": "one_window_numerical_audit",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
