#!/usr/bin/env python3
"""Read torch.profiler folded-stack files and turn them into flamegraphs.

Consumes the ``stacks_cpu_rankN.txt`` / ``stacks_cuda_rankN.txt`` files emitted by
OmniTorchProfilerWrapper via ``torch.profiler.export_stacks()`` (only written when
``torch_profiler_with_stack=True``). Each line is Brendan-Gregg "folded" format:

    frameA;frameB;...;leaf <value>

where ``<value>`` is the metric (``self_cpu_time_total`` or ``self_cuda_time_total``,
in microseconds).

Default (no deps): print the hottest stack paths, hottest leaves (self time), and
hottest frames (inclusive time) — a textual "understanding" of the flamegraph.

Optional renders:
  --speedscope out.json  : dependency-free interactive flamegraph (open at speedscope.app)
  --svg out.svg          : static SVG via flamegraph.pl or inferno-flamegraph if on PATH

torch's export_stacks output is not always directly accepted by flamegraph.pl
(pytorch#73556). We normalize every line to ``stack<space>integer`` before rendering.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def parse_folded(path: Path) -> list[tuple[list[str], int]]:
    """Return a list of (frames, value). Tolerant of frames containing spaces."""
    samples: list[tuple[list[str], int]] = []
    skipped = 0
    with path.open("rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            stack, sep, val = line.rpartition(" ")
            if not sep:
                skipped += 1
                continue
            try:
                value = int(float(val))
            except ValueError:
                skipped += 1
                continue
            if value <= 0:
                continue
            frames = [fr for fr in stack.split(";") if fr]
            if not frames:
                continue
            samples.append((frames, value))
    if skipped:
        print(f"  (skipped {skipped} unparseable line(s))", file=sys.stderr)
    return samples


def summarize(samples: list[tuple[list[str], int]], top: int) -> None:
    total = sum(v for _, v in samples)
    by_leaf: dict[str, int] = {}          # self time: value attributed to the leaf frame
    by_frame_incl: dict[str, int] = {}    # inclusive: value of every stack containing the frame
    by_stack: dict[str, int] = {}
    for frames, value in samples:
        by_leaf[frames[-1]] = by_leaf.get(frames[-1], 0) + value
        by_stack[";".join(frames)] = by_stack.get(";".join(frames), 0) + value
        for fr in set(frames):
            by_frame_incl[fr] = by_frame_incl.get(fr, 0) + value

    def pct(v: int) -> float:
        return v / total * 100 if total else 0.0

    print(f"\nstacks={len(samples)} unique_stacks={len(by_stack)} total={total / 1e3:.3f} ms")

    print("\nTop leaves by SELF time:")
    for name, v in sorted(by_leaf.items(), key=lambda kv: kv[1], reverse=True)[:top]:
        print(f"  {v / 1e3:9.3f} ms  {pct(v):5.1f}%  {name[:170]}")

    print("\nTop frames by INCLUSIVE time:")
    for name, v in sorted(by_frame_incl.items(), key=lambda kv: kv[1], reverse=True)[:top]:
        print(f"  {v / 1e3:9.3f} ms  {pct(v):5.1f}%  {name[:170]}")

    print("\nHottest full stack paths:")
    for stack, v in sorted(by_stack.items(), key=lambda kv: kv[1], reverse=True)[:top]:
        frames = stack.split(";")
        tail = " ; ".join(frames[-4:])
        print(f"  {v / 1e3:9.3f} ms  {pct(v):5.1f}%  (depth {len(frames)}) ...{tail[:200]}")


def write_speedscope(samples: list[tuple[list[str], int]], out: Path, name: str) -> None:
    frame_index: dict[str, int] = {}
    frames_list: list[dict[str, str]] = []

    def idx(fr: str) -> int:
        if fr not in frame_index:
            frame_index[fr] = len(frames_list)
            frames_list.append({"name": fr})
        return frame_index[fr]

    out_samples: list[list[int]] = []
    weights: list[int] = []
    for frames, value in samples:
        out_samples.append([idx(fr) for fr in frames])
        weights.append(value)

    doc: dict[str, Any] = {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "shared": {"frames": frames_list},
        "profiles": [
            {
                "type": "sampled",
                "name": name,
                "unit": "microseconds",
                "startValue": 0,
                "endValue": sum(weights),
                "samples": out_samples,
                "weights": weights,
            }
        ],
    }
    out.write_text(json.dumps(doc))
    print(f"\nWrote speedscope profile: {out}  (open at https://www.speedscope.app)")


def write_normalized(samples: list[tuple[list[str], int]]) -> str:
    lines = [f"{';'.join(frames)} {value}" for frames, value in samples]
    tmp = tempfile.NamedTemporaryFile("wt", suffix=".folded", delete=False)
    tmp.write("\n".join(lines))
    tmp.close()
    return tmp.name


def render_svg(samples: list[tuple[list[str], int]], out: Path, title: str) -> None:
    folded = write_normalized(samples)
    flamegraph_pl = shutil.which("flamegraph.pl")
    inferno = shutil.which("inferno-flamegraph")
    try:
        if flamegraph_pl:
            with out.open("wb") as svg:
                subprocess.run([flamegraph_pl, "--title", title, folded], check=True, stdout=svg)
            print(f"\nWrote SVG via flamegraph.pl: {out}")
        elif inferno:
            with out.open("wb") as svg:
                subprocess.run([inferno, "--title", title, folded], check=True, stdout=svg)
            print(f"\nWrote SVG via inferno-flamegraph: {out}")
        else:
            print(
                "\nNo flamegraph.pl or inferno-flamegraph on PATH — skipped SVG.\n"
                f"  Normalized folded stacks left at: {folded}\n"
                "  Install Brendan Gregg's FlameGraph or `cargo install inferno`, or use --speedscope instead."
            )
            return
    except subprocess.CalledProcessError as e:
        print(f"\nSVG render failed ({e}); normalized folded stacks at: {folded}")
        return
    Path(folded).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stacks", type=Path, help="stacks_cpu_rankN.txt or stacks_cuda_rankN.txt")
    parser.add_argument("--top", type=int, default=30, help="number of hot entries to print")
    parser.add_argument("--speedscope", type=Path, default=None, help="write speedscope JSON to this path")
    parser.add_argument("--svg", type=Path, default=None, help="render an SVG flamegraph to this path")
    args = parser.parse_args()

    if not args.stacks.exists():
        sys.exit(f"No such file: {args.stacks}  (stacks are only written when torch_profiler_with_stack=True)")

    print(f"== {args.stacks}")
    samples = parse_folded(args.stacks)
    if not samples:
        sys.exit("No usable folded-stack samples found.")

    summarize(samples, args.top)
    if args.speedscope:
        write_speedscope(samples, args.speedscope, args.stacks.name)
    if args.svg:
        render_svg(samples, args.svg, args.stacks.name)


if __name__ == "__main__":
    main()
