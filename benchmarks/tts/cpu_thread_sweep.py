#!/usr/bin/env python3
"""CPU-only Qwen3-TTS thread sweep (WP3): 8 / 16 / 32 threads (+ inductor variant).

For each thread count N the two stages get disjoint OpenMP thread ranges
(stage 0 = talker gets ``--stage0-share`` of N, stage 1 = code2wav the rest)
written into a variant of ``vllm_omni/deploy/qwen3_tts.yaml``'s ``platforms:
cpu:`` overlay, and ``stream_latency_bench.py`` runs pinned to those cores with
``--step-stats`` so the per-step attribution (scheduler / step / chunk hop) is
captured next to TTFA, playback-start latency, RTF and RSS.

  VLLM_TARGET_DEVICE=cpu python benchmarks/tts/cpu_thread_sweep.py \
      --python /path/to/omni-cpu/bin/python --threads 8,16,32 --inductor-at 16 --cpu-base 56

No GPU is used. Runs one configuration at a time.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[1]
sys.path.insert(0, str(BENCH_DIR))
from init_profile import _flatten_deploy_yaml  # noqa: E402

GiB = 1 << 30


def plan_ranges(n_threads: int, cpu_base: int, stage0_share: float = 0.75) -> tuple[list[int], list[int]]:
    """Disjoint core lists for stage 0 and stage 1 (stage 0 gets the larger share, >= 1 core each)."""
    if n_threads < 2:
        raise ValueError("need at least 2 threads (one per stage)")
    n0 = max(1, min(n_threads - 1, round(n_threads * stage0_share)))
    cpus = list(range(cpu_base, cpu_base + n_threads))
    return cpus[:n0], cpus[n0:]


def _range_str(cpus: list[int]) -> str:
    return f"{cpus[0]}-{cpus[-1]}" if len(cpus) > 1 else str(cpus[0])


def write_cpu_variant(base_yaml: Path, out_path: Path, stage0: list[int], stage1: list[int]) -> Path:
    """Copy ``base_yaml`` (flattened) with the CPU overlay's thread binding replaced."""
    cfg = copy.deepcopy(_flatten_deploy_yaml(base_yaml))
    cpu = cfg.setdefault("platforms", {}).setdefault("cpu", {})
    stages = {int(s.get("stage_id", i)): s for i, s in enumerate(cpu.setdefault("stages", []))}
    for sid, cpus in ((0, stage0), (1, stage1)):
        st = stages.get(sid)
        if st is None:
            st = {"stage_id": sid}
            cpu["stages"].append(st)
        st.setdefault("runtime", {}).setdefault("env", {})["VLLM_CPU_OMP_THREADS_BIND"] = _range_str(cpus)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def run_one(name: str, n: int, inductor: bool, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    s0, s1 = plan_ranges(n, args.cpu_base, args.stage0_share)
    yaml_path = write_cpu_variant(Path(args.base_yaml), case_dir / f"{name}.yaml", s0, s1)
    env = dict(os.environ)
    env["VLLM_TARGET_DEVICE"] = "cpu"
    env["VLLM_OMNI_INIT_TIMELINE"] = str(case_dir / "timeline.json")
    env["VLLM_OMNI_STEP_STATS_DIR"] = str(case_dir / "step_stats")
    if inductor:
        env["VLLM_OMNI_CPU_INDUCTOR"] = "1"
    cmd: list[str] = []
    if shutil.which("taskset"):
        cmd += ["taskset", "-c", f"{s0[0]}-{s1[-1]}"]
    cmd += [
        args.python,
        str(BENCH_DIR / "stream_latency_bench.py"),
        "--model",
        args.model,
        "--deploy-config",
        str(yaml_path),
        "--query-type",
        "CustomVoice",
        "--txt-prompts",
        args.prompts,
        "--warmup",
        "1",
        "--repeat",
        str(args.repeat),
        "--output-dir",
        str(case_dir / "bench"),
        "--tag",
        name,
        "--step-stats",
        "--init-timeout",
        "1500",
        "--stage-init-timeout",
        "1200",
    ]
    print(
        f"[cpu_sweep] {name}: threads={n} stage0={_range_str(s0)} stage1={_range_str(s1)} inductor={inductor}",
        flush=True,
    )
    with open(case_dir / "stdout.log", "w") as log:
        try:
            rc = subprocess.run(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_s, cwd=str(REPO_ROOT)
            ).returncode
        except subprocess.TimeoutExpired:
            rc = -1
    row: dict[str, Any] = {
        "case": name,
        "threads": n,
        "stage0_cpus": _range_str(s0),
        "stage1_cpus": _range_str(s1),
        "inductor": inductor,
        "returncode": rc,
        "cmd": cmd,
    }
    files = sorted((case_dir / "bench").rglob("*.json"))
    if files:
        doc = json.loads(files[-1].read_text())
        s = doc["summary"]
        row.update(
            init_s=doc["init"]["init_s"],
            ttfa_p50=s["ttfa_ms"]["p50"],
            playback_start_p50=s["playback_start_ms"]["p50"],
            rtf_total_p50=s["rtf_total"]["p50"],
            peak_rss_gib=doc["resources"]["peak_rss_bytes"] / GiB,
            step_stats=(doc.get("step_stats") or {}).get("attribution", [])[:8],
        )
    print(f"[cpu_sweep] {name}: rc={rc} ttfa_p50={row.get('ttfa_p50')} rtf={row.get('rtf_total_p50')}", flush=True)
    return row


def render_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        "case",
        "threads",
        "inductor",
        "rc",
        "init_s",
        "ttfa_p50",
        "playback_start_p50",
        "rtf_total_p50",
        "peak_rss_gib",
    ]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get("returncode") if c == "rc" else r.get(c)
            vals.append(f"{v:.2f}" if isinstance(v, float) else ("" if v is None else str(v)))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", default="8,16,32")
    p.add_argument("--inductor-at", type=int, default=16, help="thread count for the inductor variant (0 = skip)")
    p.add_argument("--cpu-base", type=int, default=0, help="first core of the contiguous range to pin to")
    p.add_argument("--stage0-share", type=float, default=0.75)
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    p.add_argument("--base-yaml", default=str(REPO_ROOT / "vllm_omni" / "deploy" / "qwen3_tts.yaml"))
    p.add_argument("--prompts", default=str(BENCH_DIR / "prompts_12.txt"))
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--timeout-s", type=int, default=2400)
    p.add_argument("--output-dir", default=str(BENCH_DIR / "edge_results" / "cpu_sweep"))
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir) / _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in [int(x) for x in args.threads.split(",") if x.strip()]:
        rows.append(run_one(f"cpu{n}", n, False, args, out_dir))
        if args.inductor_at and n == args.inductor_at:
            rows.append(run_one(f"cpu{n}_inductor", n, True, args, out_dir))
    summary = {"model": args.model, "timestamp": out_dir.name, "host_cpu_base": args.cpu_base, "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    table = render_table(rows)
    (out_dir / "summary.md").write_text(table + "\n")
    print(table)
    print(f"[cpu_sweep] wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
