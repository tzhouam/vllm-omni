#!/usr/bin/env python3
"""Render the committed edge results (stream_latency / init_profile / hw_emulation / cpu_sweep /
codec_stream) as markdown tables.

  python benchmarks/tts/edge_results/summarize.py [--root benchmarks/tts/edge_results] [--out summary.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GiB = 1 << 30


def _load(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _f(v: Any, nd: int = 1) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def stream_latency_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for f in sorted((root / "stream_latency").glob("*/*/*.json")):
        d = _load(f)
        if not d or "summary" not in d:
            continue
        s = d["summary"]
        meta = d.get("meta", {})
        res = d.get("resources", {}) or {}
        tag = f.stem.split("_", 1)[1] if "_" in f.stem else ""
        st = d.get("step_stats") or {}
        rows.append(
            {
                "model": f.parents[1].name.replace("Qwen3-TTS-12Hz-", "").replace("-CustomVoice", ""),
                "deploy": f.parent.name,
                "tag": tag,
                "run": f.stem.split("_", 1)[0],
                "n": s.get("n_requests"),
                "init_s": d.get("init", {}).get("init_s"),
                "ttfa_p50": s["ttfa_ms"]["p50"],
                "ttfa_p90": s["ttfa_ms"]["p90"],
                "playback_p50": s["playback_start_ms"]["p50"],
                "stall_p50": s["stall_at_ttfa_ms"]["p50"],
                "rtf_stream_p50": s["rtf_streamed"]["p50"],
                "rtf_total_p50": s["rtf_total"]["p50"],
                "rss_gib": (res.get("peak_rss_bytes") or 0) / GiB,
                "gpu_mib": res.get("peak_gpu_mem_mib"),
                "sha": (meta.get("git_sha") or "")[:8],
                "step_stats": st.get("attribution", [])[:6] if st else None,
                "path": str(f.relative_to(root)),
            }
        )
    return rows


def table(rows: list[dict[str, Any]], cols: list[str], nd: int = 1) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_f(r.get(c), nd) for c in cols) + " |")
    return "\n".join(out)


def render(root: Path) -> str:
    parts = ["# Edge results summary\n"]
    rows = stream_latency_rows(root)
    parts.append("## Streaming latency (stream_latency_bench.py)\n")
    parts.append(
        table(
            rows,
            [
                "model",
                "deploy",
                "tag",
                "run",
                "n",
                "init_s",
                "ttfa_p50",
                "ttfa_p90",
                "playback_p50",
                "stall_p50",
                "rtf_stream_p50",
                "rtf_total_p50",
                "rss_gib",
                "gpu_mib",
                "sha",
            ],
        )
    )
    for r in rows:
        if r.get("step_stats"):
            parts.append(f"\n### Step attribution: {r['model']} {r['deploy']} {r['tag']} ({r['run']})\n")
            parts.append(
                table(r["step_stats"], ["role", "counter", "count", "mean_ms", "p95_ms", "share_of_frame"], nd=3)
            )
    for name, cols in (
        ("init_profile", ["config", "init_s", "ttfa_ms", "gpu_mib"]),
        (
            "hw_emulation",
            [
                "case",
                "hardware_class",
                "returncode",
                "init_s",
                "ttfa_p50",
                "playback_start_p50",
                "stall_p50",
                "rtf_total_p50",
                "peak_rss_gib",
                "peak_gpu_mib",
            ],
        ),
        (
            "cpu_sweep",
            [
                "case",
                "threads",
                "inductor",
                "returncode",
                "init_s",
                "ttfa_p50",
                "playback_start_p50",
                "rtf_total_p50",
                "peak_rss_gib",
            ],
        ),
    ):
        for f in sorted((root / name).glob("*/summary.json")):
            d = _load(f)
            if not d:
                continue
            parts.append(f"\n## {name} {f.parent.name}\n")
            rows_n = d.get("rows") or d.get("configs") or []
            if name == "init_profile" and isinstance(rows_n, dict):
                rows_n = [{"config": k, **v} for k, v in rows_n.items()]
            parts.append(table(rows_n, cols))
            if name == "cpu_sweep":
                for r in rows_n:
                    if r.get("step_stats"):
                        parts.append(f"\n### Step attribution: {r['case']}\n")
                        parts.append(
                            table(
                                r["step_stats"],
                                ["role", "counter", "count", "mean_ms", "p95_ms", "share_of_frame"],
                                nd=3,
                            )
                        )
    for f in sorted((root / "codec_stream").glob("**/*.json")):
        d = _load(f)
        if d and ("snr_db" in d or "parity" in d):
            parts.append(f"\n## codec_stream {f.relative_to(root)}\n")
            parts.append("```json\n" + json.dumps(d, indent=1)[:3000] + "\n```")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(HERE))
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    text = render(Path(args.root))
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
