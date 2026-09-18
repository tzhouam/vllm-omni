# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage-by-stage decomposition of the served chunk across every arm.

Each arm's spans are reduced to per-chunk medians on the driver rank, so the
end-to-end delta between arms can be attributed to the stages that actually
moved rather than inferred from the interval alone.

Derived rows:
  commit_block_kv   post_decode - decode_to_pixels - prepare_next_chunk
                    (the clean-KV fifth forward, which has no span of its own)
  gpu tail          sum(execute_stepwise) - (4 x denoise_step + post_decode)
                    (work enqueued by the pipeline that drains at the chunk
                    barrier; pipeline spans are wall times without a sync, so
                    this is real GPU time the spans above under-charge)
  outside runner    interval - sum(execute_stepwise)
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict

ARMS = [
    ("baseline", 6),
    ("u8", 6),
    ("u8_cond", 6),
    ("u8_cond_shard", 6),
    ("long_baseline", 12),
    ("long_all", 12),
]
ROWS = [
    ("denoise_step x4", None),
    ("pipe.post_decode", "pipe.post_decode"),
    ("  pipe.decode_to_pixels", "pipe.decode_to_pixels"),
    ("    pipe.vae_streaming_decode", "pipe.vae_streaming_decode"),
    ("  pipe.prepare_next_chunk", "pipe.prepare_next_chunk"),
    ("    pipe.vae_condition_encode", "pipe.vae_condition_encode"),
    ("  commit_block_kv (derived)", None),
    ("gpu tail in runner (derived)", None),
    ("outside runner (derived)", None),
    ("= chunk interval", None),
    ("api.mux_encode (overlapped)", "api.mux_encode"),
    ("api.coerce_frames (overlapped)", "api.coerce_frames"),
]


def load(d):
    spans = []
    for f in sorted(os.listdir(d)):
        if f.startswith("spans_"):
            with open(os.path.join(d, f)) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        spans.append(json.loads(line))
    return spans


def arm_stats(root, name, warmup):
    d = os.path.join(root, name, "spans")
    if not os.path.isdir(d):
        return None
    spans = load(d)
    if not spans:
        return None
    worker_pids = {s["pid"] for s in spans if s["name"] == "pipe.post_decode"}
    if not worker_pids:
        return None
    driver = min(worker_pids)
    posts = sorted([s for s in spans if s["pid"] == driver and s["name"] == "pipe.post_decode"], key=lambda s: s["t0"])
    if len(posts) <= warmup + 2:
        return None
    t0 = posts[warmup]["t0"]

    med = {}
    grouped = defaultdict(list)
    for s in spans:
        if s["t0"] >= t0 and (s["pid"] == driver or s["name"].startswith("api.")):
            grouped[s["name"]].append(s["ms"])
    for k, v in grouped.items():
        med[k] = statistics.median(v)

    steady_posts = [p for p in posts if p["t0"] >= t0]
    interval = statistics.median([(b["t1"] - a["t1"]) * 1000 for a, b in zip(steady_posts, steady_posts[1:])])
    ex_all = sorted(
        [s for s in spans if s["pid"] == driver and s["name"] == "runner.execute_stepwise"], key=lambda s: s["t0"]
    )
    post_times = [(p["t0"], p["t1"]) for p in posts]
    blocks, current, pi = [], [], 0
    for call in ex_all:
        current.append(call)
        while pi < len(post_times) and post_times[pi][1] < call["t0"]:
            pi += 1
        if pi < len(post_times) and call["t0"] <= post_times[pi][0] and post_times[pi][1] <= call["t1"]:
            blocks.append(current)
            current = []
            pi += 1
    blocks = [b for b in blocks if b and b[0]["t0"] >= t0]
    if not blocks:
        return None
    ex_total = statistics.median([sum(c["ms"] for c in b) for b in blocks])
    interval = statistics.median([(b[-1]["t1"] - a[-1]["t1"]) * 1000 for a, b in zip(blocks, blocks[1:])])

    denoise4 = med.get("pipe.denoise_step", 0) * 4
    post = med.get("pipe.post_decode", 0)
    commit = post - med.get("pipe.decode_to_pixels", 0) - med.get("pipe.prepare_next_chunk", 0)
    tail = ex_total - (denoise4 + post)
    outside = interval - ex_total

    med["denoise_step x4"] = denoise4
    med["  commit_block_kv (derived)"] = commit
    med["gpu tail in runner (derived)"] = tail
    med["outside runner (derived)"] = outside
    med["= chunk interval"] = interval
    return med


def main(root: str) -> int:
    stats = {}
    for name, warmup in ARMS:
        st = arm_stats(root, name, warmup)
        if st:
            stats[name] = st
    names = list(stats)
    w = 15
    print(f"{'stage (median ms/chunk)':<32}" + "".join(f"{n[:w]:>{w}}" for n in names))
    print("-" * (32 + w * len(names)))
    for label, key in ROWS:
        k = key or label
        row = f"{label:<32}"
        for n in names:
            v = stats[n].get(k)
            row += f"{v:>{w}.1f}" if v is not None else " " * w
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
