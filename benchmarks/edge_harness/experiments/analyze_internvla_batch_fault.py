#!/usr/bin/env python3
"""Report per-frame numerical failures in two pinned Cosmos NPU probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ORIGINAL_SHA = "c108accf75ac895dccfc873e626c6de14c93a8f45855949e9be5ef67be6e89f7"
FIXTURE_SHA = "7bca8b4dcafa0ab0d45387e95a4d3e844f04f296f9253a4d4ac54ab5682d068c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(path: Path, expected_sha: str) -> dict:
    import numpy as np

    if sha256(path) != expected_sha:
        raise ValueError("paired Cosmos output changed")
    with np.load(path, allow_pickle=False) as file:
        cpu = file["cpu"]
        npu = file["vitisai"]
    if cpu.shape != npu.shape or cpu.shape != (6, 256, 64, 64):
        raise ValueError("paired Cosmos output shape changed")
    relative = []
    cosine = []
    npu_std = []
    for i in range(6):
        a = cpu[i].astype(np.float64)
        b = npu[i].astype(np.float64)
        relative.append(float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12)))
        cosine.append(float(np.sum(a * b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)))
        npu_std.append(float(b.std()))
    return {
        "paired_sha256": expected_sha,
        "frame_relative_l2": relative,
        "frame_cosine": cosine,
        "npu_frame_std": npu_std,
        "cpu_frames_equal_to_first": [bool(np.array_equal(cpu[0], cpu[i])) for i in range(6)],
        "npu_frames_1_to_5_identical": all(np.array_equal(npu[1], npu[i]) for i in range(2, 6)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-paired", type=Path, required=True)
    parser.add_argument("--bias-outside-paired", type=Path, required=True)
    parser.add_argument("--bias-outside-report", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    probe = json.loads(args.bias_outside_report.read_text(encoding="utf-8-sig"))
    outside_sha = probe["paired_sha256"]
    if sha256(args.fixture) != FIXTURE_SHA:
        raise ValueError("Cosmos activation fixture changed")
    with np.load(args.fixture, allow_pickle=False) as file:
        pattern = file["pattern_group_norm"]
        ramp = file["ramp_group_norm"]
    report = {
        "scope": "batch-index diagnosis on pinned six-frame real-weight Cosmos Conv13 candidate outputs; not full encoder",
        "fixture_sha256": FIXTURE_SHA,
        "pattern_input_frames_equal_to_first": [bool(np.array_equal(pattern[0], pattern[i])) for i in range(6)],
        "ramp_input_frames_equal_to_first": [bool(np.array_equal(ramp[0], ramp[i])) for i in range(6)],
        "original_bias_inside_qdq": summarize(args.original_paired, ORIGINAL_SHA),
        "rewritten_bias_outside_qdq": summarize(args.bias_outside_paired, outside_sha),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
