#!/usr/bin/env python3
"""Compare pinned joint InternVLA actions with separately measured placements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("joint", "cpu", "npu", "radeon"):
        parser.add_argument(f"--{name}-actions", type=Path, required=True)
        parser.add_argument(f"--{name}-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    actions = {}
    files = {}
    for name in ("joint", "cpu", "npu", "radeon"):
        path = getattr(args, f"{name}_actions")
        report_path = getattr(args, f"{name}_report")
        source = json.loads(report_path.read_text(encoding="utf-8-sig"))
        value = np.load(path, allow_pickle=False)
        if (value.shape != (1, 50, 32) or value.dtype != np.float32
                or not np.isfinite(value).all()
                or hashlib.sha256(value.tobytes()).hexdigest()
                not in source["unique_action_hashes"]):
            raise ValueError(f"{name} action shape, finiteness or report hash differs")
        actions[name] = value.astype(np.float64)
        files[name] = {"actions_sha256": sha256(path), "report_sha256": sha256(report_path),
                       "action_bytes_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    joint = actions["joint"]
    comparisons = {}
    for name in ("cpu", "npu", "radeon"):
        reference = actions[name]
        difference = joint - reference
        comparisons[f"joint_vs_{name}"] = {
            "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
            "max_abs": float(np.max(np.abs(difference))),
            "cosine": float(np.dot(joint.ravel(), reference.ravel())
                            / (np.linalg.norm(joint) * np.linalg.norm(reference))),
        }
    result = {
        "scope": "separately measured one-pattern synthetic InternVLA action outputs; no task-quality tolerance or paired speedup claim",
        "files": files,
        "comparisons": comparisons,
        "status": "action_sensitivity_measured",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
