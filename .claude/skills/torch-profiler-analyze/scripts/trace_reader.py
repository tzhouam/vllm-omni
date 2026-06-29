#!/usr/bin/env python3
"""Read and summarize torch.profiler Chrome traces for any vLLM Omni model.

Model-agnostic superset of diffusion-perf-opt/scripts/trace_analyzer.py. Adds:
  * multi-rank aggregation (per-rank idle_pct + cross-rank straggler summary)
  * named-region attribution (roll up record_function / user_annotation by name)

Inputs are the ``trace_rankN.json[.gz]`` files emitted by OmniTorchProfilerWrapper
(``vllm_omni/profiler/omni_torch_profiler.py``). Output is timing-only: it does not
parse tensor shapes (use ops_rankN.xlsx) or provide final latency claims (use a
non-profiler baseline).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path
from typing import Any

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
CPU_CATS = {"python_function", "user_annotation", "cpu_op", "cuda_runtime", "cuda_driver"}
REGION_CATS = {"user_annotation", "python_function"}

_RANK_RE = re.compile(r"rank(\d+)")


def open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def rank_of(path: Path) -> str:
    """Infer the rank label from a ``trace_rankN.json[.gz]`` filename."""
    m = _RANK_RE.search(path.name)
    return m.group(1) if m else "?"


def load_events(path: Path) -> list[dict[str, Any]]:
    with open_trace(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("traceEvents", [])


def summarize_trace(path: Path, min_gap_us: float, topn: int, per_region: bool) -> dict[str, Any]:
    events = load_events(path)
    gpu: list[tuple[float, float, float, str, str, Any, Any]] = []
    cpu: list[tuple[float, float, float, str, str, Any, Any]] = []
    by_gpu_name: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0, 0.0])
    by_region: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0, 0.0])
    nccl: dict[tuple[str, str], list[float]] = collections.defaultdict(lambda: [0, 0.0, 0.0])

    for event in events:
        dur = event.get("dur")
        ts = event.get("ts")
        if dur is None or ts is None or dur <= 0:
            continue
        cat = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        row = (float(ts), float(ts + dur), float(dur), name, cat, event.get("pid"), event.get("tid"))
        if cat in GPU_CATS:
            gpu.append(row)
            gstat = by_gpu_name[name]
            gstat[0] += 1
            gstat[1] += dur
            gstat[2] = max(gstat[2], dur)
        elif cat in CPU_CATS:
            cpu.append(row)
        if per_region and cat in REGION_CATS:
            rstat = by_region[name]
            rstat[0] += 1
            rstat[1] += dur
            rstat[2] = max(rstat[2], dur)
        if "nccl" in name.lower():
            nstat = nccl[(cat, name)]
            nstat[0] += 1
            nstat[1] += dur
            nstat[2] = max(nstat[2], dur)

    rank = rank_of(path)
    print(f"\n== {path}  (rank {rank})")
    print(f"events={len(events)} gpu_events={len(gpu)} cpu_events={len(cpu)}")
    if not gpu:
        print("No GPU events found.")
        return {"rank": rank, "path": str(path), "idle_pct": None, "span_s": 0.0}

    gpu.sort()
    merged: list[list[Any]] = []
    for start, end, dur, name, cat, pid, tid in gpu:
        if not merged or start > merged[-1][1]:
            merged.append([start, end, [(start, end, dur, name, cat, pid, tid)]])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].append((start, end, dur, name, cat, pid, tid))

    span = max(end for _, end, *_ in gpu) - min(start for start, *_ in gpu)
    busy = sum(end - start for start, end, _ in merged)
    idle = span - busy
    idle_pct = idle / span * 100 if span else 0.0
    print(
        f"gpu_span_s={span / 1e6:.3f} "
        f"busy_union_s={busy / 1e6:.3f} "
        f"idle_union_s={idle / 1e6:.3f} "
        f"idle_pct={idle_pct:.2f}"
    )

    interesting_cpu = [
        r
        for r in cpu
        if r[2] >= 1000
        and (
            r[4] in {"python_function", "user_annotation"}
            or "cudaStreamSynchronize" in r[3]
            or "cudaDeviceSynchronize" in r[3]
            or "cudaLaunch" in r[3]
            or "cudaMemcpy" in r[3]
        )
    ]

    gaps = []
    for idx in range(1, len(merged)):
        gap_start = merged[idx - 1][1]
        gap_end = merged[idx][0]
        gap_dur = gap_end - gap_start
        if gap_dur >= min_gap_us:
            prev_event = max(merged[idx - 1][2], key=lambda x: x[1])
            next_event = min(merged[idx][2], key=lambda x: x[0])
            mid = (gap_start + gap_end) / 2
            containers = [r for r in interesting_cpu if r[0] <= mid <= r[1]]
            containers = sorted(containers, key=lambda x: x[2])[:8]
            gaps.append((gap_dur, gap_start, gap_end, prev_event, next_event, containers))

    print(f"gaps_ge_{min_gap_us / 1000:.3f}ms count={len(gaps)} sum_s={sum(g[0] for g in gaps) / 1e6:.3f}")
    for gap_dur, gap_start, gap_end, prev_event, next_event, containers in sorted(gaps, reverse=True)[:topn]:
        print(f"\nGAP {gap_dur / 1000:.3f} ms ts={gap_start:.0f}->{gap_end:.0f}")
        print(f"  prev {prev_event[4]} {prev_event[2] / 1000:.3f} ms {prev_event[3][:160]}")
        print(f"  next {next_event[4]} {next_event[2] / 1000:.3f} ms {next_event[3][:160]}")
        for r in containers:
            print(f"  in   {r[4]} {r[2] / 1000:.3f} ms {r[3][:180]}")

    print("\nTop GPU/operator events by total duration:")
    for name, (count, total, max_dur) in sorted(by_gpu_name.items(), key=lambda kv: kv[1][1], reverse=True)[:topn]:
        print(f"  {int(count):8d} total={total / 1e6:9.3f}s max={max_dur / 1000:9.3f}ms {name[:180]}")

    print("\nTop NCCL-like events by category:")
    for (cat, name), (count, total, max_dur) in sorted(nccl.items(), key=lambda kv: kv[1][1], reverse=True)[:topn]:
        print(f"  {int(count):8d} total={total / 1e6:9.3f}s max={max_dur / 1000:9.3f}ms cat={cat} {name[:160]}")

    if per_region and by_region:
        print("\nTop named regions (record_function / user_annotation) by total duration:")
        print("  NOTE: nested annotations overcount; treat as relative stage weight, not exclusive time.")
        for name, (count, total, max_dur) in sorted(by_region.items(), key=lambda kv: kv[1][1], reverse=True)[:topn]:
            mean = total / count if count else 0.0
            print(f"  {int(count):8d} total={total / 1e6:9.3f}s mean={mean / 1000:9.3f}ms max={max_dur / 1000:9.3f}ms {name[:160]}")

    return {"rank": rank, "path": str(path), "idle_pct": idle_pct, "span_s": span / 1e6}


def print_cross_rank(summaries: list[dict[str, Any]]) -> None:
    ranked = [s for s in summaries if s.get("idle_pct") is not None]
    if len(ranked) < 2:
        return
    print("\n=== Cross-rank summary ===")
    print(f"  {'rank':>6}  {'idle_pct':>9}  {'span_s':>9}")
    for s in sorted(ranked, key=lambda x: x["rank"]):
        print(f"  {s['rank']:>6}  {s['idle_pct']:>9.2f}  {s['span_s']:>9.3f}")
    hi = max(ranked, key=lambda x: x["idle_pct"])
    lo = min(ranked, key=lambda x: x["idle_pct"])
    skew = hi["idle_pct"] - lo["idle_pct"]
    print(
        f"  idle skew = {skew:.2f} pp  (max rank {hi['rank']} @ {hi['idle_pct']:.2f}% , "
        f"min rank {lo['rank']} @ {lo['idle_pct']:.2f}%)"
    )
    if skew >= 10.0:
        print(f"  HINT: rank {hi['rank']} idles notably more — possible straggler / load imbalance; inspect its GAPs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", type=Path, help="trace_rankN.json or .json.gz files")
    parser.add_argument("--min-gap-ms", type=float, default=5.0, help="minimum GPU idle gap to print")
    parser.add_argument("--topn", type=int, default=20, help="number of gaps/hotspots/regions to print")
    parser.add_argument("--per-region", action="store_true", help="roll up time by record_function/annotation name")
    parser.add_argument("--ranks", action="store_true", help="print a cross-rank idle summary (auto when >1 trace)")
    args = parser.parse_args()

    summaries = [
        summarize_trace(trace, args.min_gap_ms * 1000.0, args.topn, args.per_region) for trace in args.traces
    ]
    if args.ranks or len(args.traces) > 1:
        print_cross_rank(summaries)


if __name__ == "__main__":
    main()
