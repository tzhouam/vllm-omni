#!/usr/bin/env python3
"""Summarize fixed time windows of raw Qwen3-TTS sustained request streams."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


WINDOWS_MIN = ((0, 25), (25, 27), (27, 30))


def nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def required_prebuffer_ms(row: dict) -> float:
    chunks = [chunk for chunk in row["chunks_all"] if chunk["samples"]]
    if not chunks:
        raise ValueError(f"{row['request_id']}: no audio chunks")
    first_ms = chunks[0]["t_ms"]
    played_ms = 0.0
    required_ms = 0.0
    for chunk in chunks:
        required_ms = max(required_ms, chunk["t_ms"] - first_ms - played_ms)
        played_ms += 1000 * chunk["samples"] / chunk["sr"]
    return required_ms


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    sustained = [row for row in rows if row["phase"] == "sustained"]
    if not sustained:
        raise ValueError(f"{path}: no sustained requests")
    start = sustained[0]["submitted_unix"]
    result = {"source": str(path), "sustained_count": len(sustained), "windows": []}
    for lo, hi in WINDOWS_MIN:
        selected = [row for row in sustained
                    if lo * 60 <= row["submitted_unix"] - start < hi * 60]
        rtf = [row["rtf_total"] for row in selected]
        wall = [row["total_wall_s"] for row in selected]
        buffers = [required_prebuffer_ms(row) for row in selected]
        result["windows"].append({
            "minutes": [lo, hi],
            "requests": len(selected),
            "rtf_p50": nearest_rank(rtf, 0.5),
            "rtf_p95": nearest_rank(rtf, 0.95),
            "wall_s_p50": nearest_rank(wall, 0.5),
            "wall_s_p95": nearest_rank(wall, 0.95),
            "requests_with_underrun_at_ttfa": sum(bool(row["underruns"]) for row in selected),
            "required_prebuffer_ms_p50": nearest_rank(buffers, 0.5),
            "required_prebuffer_ms_p95": nearest_rank(buffers, 0.95),
            "required_prebuffer_ms_max": max(buffers) if buffers else None,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {
        "method": "Non-overlapping windows from each run's first sustained submission; nearest-rank percentiles; "
                  "prebuffer is minimum added delay after first audio arrival to avoid all simulated underruns",
        "runs": [summarize(path) for path in args.paths],
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
