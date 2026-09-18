# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cumulative results table for served LingBot-World arms (bench.json + overlay spans).

Usage: python eval_table.py <root> [arm ...]

Per arm: the benchmark's own steady-state numbers (bench.json), the single-chunk
end-to-end latency from the span timeline (chunk_latency logic), and the
per-stage medians on the driver rank (decompose logic, warmup 12 chunks).
Rows whose spans exist only on some arms (profiler.*) are listed when present.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decompose  # noqa: E402

WARMUP = 12
ARMS = ["A_base", "B_pr7648", "C_async_meta", "D_u8", "E_cond", "F_bf16", "G_shard", "H_shard_prof", "A_base_rep"]


def bench(root, arm):
    p = os.path.join(root, arm, "bench.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    m = d["metrics"]
    if "interval_steady" not in m:
        # Multi-session run: pool the steady chunk intervals of every session.
        ivals, stalls, ttfc, rtf = [], [], [], []
        for sess in d["sessions"]:
            ivals += [c["inter_arrival_ms"] for c in sess["chunks"] if c["index"] > m.get("warmup_chunks", 6)]
            sm = sess["metrics"]
            stalls.append(sm["playback"]["stall_ratio"])
            ttfc.append(sm["ttfc_ms"])
            rtf.append(sm["steady_video_rtf"])
        ivals.sort()
        n = len(ivals)
        mean = sum(ivals) / n
        std = (sum((x - mean) ** 2 for x in ivals) / (n - 1)) ** 0.5
        return {
            "steady median": statistics.median(ivals),
            "steady mean": mean,
            "steady std": std,
            "steady p99": ivals[min(n - 1, int(round(0.99 * (n - 1))))],
            "steady n": n,
            "TTFC": sum(ttfc) / len(ttfc),
            "RTF@16fps": sum(rtf) / len(rtf),
            "RTF@12fps": sum(rtf) / len(rtf) * 12 / 16,
            "stall ratio %": 100 * sum(stalls) / len(stalls),
            "underruns": m["underrun_count"],
        }
    s = m["interval_steady"]
    return {
        "steady median": s["median_ms"],
        "steady mean": s["mean_ms"],
        "steady std": s["std_ms"],
        "steady p99": s["p99_ms"],
        "steady n": s["count"],
        "TTFC": m["ttfc_ms"],
        "RTF@16fps": m["steady_video_rtf"],
        "RTF@12fps": m["steady_video_rtf"] * 12 / 16,
        "stall ratio %": 100 * m["playback"]["stall_ratio"],
        "underruns": m["playback"]["underrun_count"],
    }


def latency(root, arm):
    d = os.path.join(root, arm, "spans")
    if not os.path.isdir(d):
        return None
    spans = decompose.load(d)
    wp = {s["pid"] for s in spans if s["name"] == "pipe.post_decode"}
    if not wp:
        return None
    driver = min(wp)
    posts = sorted([s for s in spans if s["pid"] == driver and s["name"] == "pipe.post_decode"], key=lambda s: s["t0"])
    ex = sorted(
        [s for s in spans if s["pid"] == driver and s["name"] == "runner.execute_stepwise"], key=lambda s: s["t0"]
    )
    mux = sorted([s for s in spans if s["name"] == "api.mux_encode"], key=lambda s: s["t0"])
    if len(posts) <= WARMUP + 2 or not mux:
        return None
    blocks, cur, pi = [], [], 0
    pt = [(p["t0"], p["t1"]) for p in posts]
    for c in ex:
        cur.append(c)
        while pi < len(pt) and pt[pi][1] < c["t0"]:
            pi += 1
        if pi < len(pt) and c["t0"] <= pt[pi][0] and pt[pi][1] <= c["t1"]:
            blocks.append(cur)
            cur = []
            pi += 1
    lat, compute, tail = [], [], []
    for b in blocks:
        if b[0]["t0"] < posts[WARMUP]["t0"]:
            continue
        start, end_compute = b[0]["t0"], b[-1]["t1"]
        m = next((x for x in mux if x["t0"] >= end_compute - 0.010), None)
        if m is None:
            continue
        lat.append((m["t1"] - start) * 1000)
        compute.append((end_compute - start) * 1000)
        tail.append((m["t1"] - end_compute) * 1000)
    if not lat:
        return None
    return {
        "e2e latency": statistics.median(lat),
        "  compute": statistics.median(compute),
        "  deliver": statistics.median(tail),
        "latency n": len(lat),
    }


STAGE_ROWS = [
    ("denoise_step x4", "denoise_step x4"),
    ("post_decode", "pipe.post_decode"),
    ("  decode_to_pixels", "pipe.decode_to_pixels"),
    ("    vae_streaming_decode", "pipe.vae_streaming_decode"),
    ("  prepare_next_chunk", "pipe.prepare_next_chunk"),
    ("    vae_condition_encode", "pipe.vae_condition_encode"),
    ("  commit_block_kv (derived)", "  commit_block_kv (derived)"),
    ("gpu tail in runner (derived)", "gpu tail in runner (derived)"),
    ("outside runner (derived)", "outside runner (derived)"),
    ("= span chunk interval", "= chunk interval"),
    ("api.mux_encode (overlapped)", "api.mux_encode"),
    ("api.coerce_frames (overlapped)", "api.coerce_frames"),
]


def main(root, *arms):
    arms = list(arms) or [a for a in ARMS if os.path.isdir(os.path.join(root, a))]
    cols = {}
    for a in arms:
        b = bench(root, a)
        if b is None:
            continue
        row = dict(b)
        lat = latency(root, a)
        if lat:
            row.update(lat)
        st = decompose.arm_stats(root, a, WARMUP)
        if st:
            row.update({label: st.get(key) for label, key in STAGE_ROWS})
            for k, v in st.items():
                if k.startswith("profiler."):
                    row[k] = v
        cols[a] = row
    names = list(cols)
    keys = []
    for n in names:
        for k in cols[n]:
            if k not in keys:
                keys.append(k)
    print("| metric (ms unless noted) | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    for k in keys:
        vals = []
        for n in names:
            v = cols[n].get(k)
            if v is None:
                vals.append("")
            elif isinstance(v, float) and abs(v) < 10 and "RTF" in k:
                vals.append(f"{v:.3f}")
            else:
                vals.append(f"{v:.0f}" if isinstance(v, float) else str(v))
        print(f"| {k} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
