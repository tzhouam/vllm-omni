#!/usr/bin/env python3
"""Check whether a failed Cosmos NPU boundary output is a simple layout permutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PAIRED_SHA = "c108accf75ac895dccfc873e626c6de14c93a8f45855949e9be5ef67be6e89f7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    if sha256(args.paired) != PAIRED_SHA:
        raise ValueError("paired CPU/NPU tensor changed")
    with np.load(args.paired, allow_pickle=False) as data:
        cpu = np.asarray(data["cpu"])
        npu = np.asarray(data["vitisai"])
    if cpu.shape != (6, 256, 64, 64) or npu.shape != cpu.shape:
        raise ValueError("paired tensor shape changed")

    def metrics(candidate) -> dict:
        reference = cpu.astype(np.float64)
        tested = np.asarray(candidate, dtype=np.float64)
        difference = tested - reference
        return {
            "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
            "cosine": float(np.sum(tested * reference) / max(np.linalg.norm(tested) * np.linalg.norm(reference), 1e-12)),
        }

    candidates = {
        "identity": npu,
        "nhwc_memory_interpretation": npu.reshape(6, 64, 64, 256).transpose(0, 3, 1, 2),
        "height_width_swap": npu.transpose(0, 1, 3, 2),
        "reverse_channels": npu[:, ::-1],
        "channel_block16_transpose": npu.reshape(6, 16, 16, 64, 64).transpose(0, 2, 1, 3, 4).reshape(npu.shape),
        "channel_block32_transpose": npu.reshape(6, 8, 32, 64, 64).transpose(0, 2, 1, 3, 4).reshape(npu.shape),
    }
    report = {
        "scope": "single paired real-weight convolution output; simple layout hypotheses only",
        "paired_sha256": PAIRED_SHA,
        "candidate_metrics": {name: metrics(value) for name, value in candidates.items()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
