#!/usr/bin/env python3
"""Initialization (cold-start) profiler for multi-stage TTS pipelines.

Runs ``stream_latency_bench.py --init-only`` once per configuration as a
subprocess with ``VLLM_OMNI_INIT_TIMELINE`` set, then tabulates the phase
durations reported by the init timeline (engine + worker processes).

Configurations are config-only variants of one deploy YAML (flattened through
``resolve_deploy_yaml`` so ``base_config`` inheritance is preserved) plus
environment changes for cache warm/cold experiments. Run sequentially on ONE
GPU through the scheduler, e.g.:

  gpu run --gpus 1 --timeout 40m --note "init_profile warm set" -- env HF_HUB_OFFLINE=1 \\
    python benchmarks/tts/init_profile.py --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
    --configs default,eager_stage1,eager_both,parallel_stage_init,edge
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
DEPLOY_DIR = REPO_ROOT / "vllm_omni" / "deploy"
PHASE_COLUMNS = [
    "stages_init",
    "g2_spawn_and_ready",
    "engine_core_init",
    "init_device",
    "load_model",
    "load_weights",
    "predictor_setup_compile",
    "predictor_warmup",
    "cudagraph_capture",
    "talker_mtp_capture",
    "code2wav_cudagraph",
]


@dataclass
class Config:
    name: str
    deploy_updates: dict[str, Any] = field(default_factory=dict)  # {"stages": {idx: {...}}, "top": {...}}
    env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    deploy_config: str | None = None  # explicit YAML instead of the model default
    fresh_caches: bool = False


CONFIGS: dict[str, Config] = {
    "default": Config("default"),
    "cold_cache": Config("cold_cache", fresh_caches=True),
    "eager_stage1": Config("eager_stage1", deploy_updates={"stages": {1: {"enforce_eager": True}}}),
    "eager_both": Config(
        "eager_both", deploy_updates={"stages": {0: {"enforce_eager": True}, 1: {"enforce_eager": True}}}
    ),
    "parallel_stage_init": Config("parallel_stage_init", extra_args=["--parallel-stage-init"]),
    "edge": Config("edge", deploy_config=str(DEPLOY_DIR / "edge" / "qwen3_tts.yaml")),
    "edge_eager_stage1": Config(
        "edge_eager_stage1",
        deploy_config=str(DEPLOY_DIR / "edge" / "qwen3_tts.yaml"),
        deploy_updates={"stages": {1: {"enforce_eager": True}}},
    ),
    "edge_parallel": Config(
        "edge_parallel", deploy_config=str(DEPLOY_DIR / "edge" / "qwen3_tts.yaml"), extra_args=["--parallel-stage-init"]
    ),
}


def _flatten_deploy_yaml(path: Path) -> dict[str, Any]:
    """Resolve ``base_config`` inheritance without importing torch-heavy modules."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base_config", None)
    if not base:
        return raw
    base_path = Path(base) if os.path.isabs(base) else (path.parent / base)
    merged = _flatten_deploy_yaml(base_path.resolve())
    for key, value in raw.items():
        if key == "stages":
            by_id = {int(s.get("stage_id", i)): dict(s) for i, s in enumerate(merged.get("stages", []))}
            for s in value or []:
                sid = int(s.get("stage_id", 0))
                by_id.setdefault(sid, {"stage_id": sid}).update(s)
            merged["stages"] = [by_id[k] for k in sorted(by_id)]
        elif key == "platforms" and isinstance(value, dict) and isinstance(merged.get("platforms"), dict):
            for plat, block in value.items():
                merged["platforms"].setdefault(plat, {})
                merged["platforms"][plat].update(block or {})
        else:
            merged[key] = value
    return merged


def write_variant_yaml(base_path: Path, updates: dict[str, Any], out_dir: Path, name: str) -> Path:
    cfg = copy.deepcopy(_flatten_deploy_yaml(base_path))
    for key, value in (updates.get("top") or {}).items():
        cfg[key] = value
    for idx, stage_updates in (updates.get("stages") or {}).items():
        for stage in cfg.get("stages", []):
            if int(stage.get("stage_id", -1)) == int(idx):
                stage.update(stage_updates)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def phase_durations(timeline: dict[str, Any] | None) -> dict[str, float]:
    """Sum durations per phase name (a phase may appear per stage/worker)."""
    out: dict[str, float] = {}
    if not timeline:
        return out
    for p in timeline.get("phases", []):
        out[p["name"]] = out.get(p["name"], 0.0) + float(p.get("dur_s", 0.0))
    return out


def phase_durations_by_stage(timeline: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not timeline:
        return out
    for p in timeline.get("phases", []):
        key = "-" if p.get("stage_id") is None else str(p["stage_id"])
        out.setdefault(key, {})
        out[key][p["name"]] = out[key].get(p["name"], 0.0) + float(p.get("dur_s", 0.0))
    return out


def render_table(rows: list[dict[str, Any]]) -> str:
    cols = ["config", "init_s", "ttfa_ms", "gpu_mib", *PHASE_COLUMNS]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        vals = [
            r["config"],
            f"{r.get('init_s', float('nan')):.1f}",
            f"{(r.get('ttfa_ms') or float('nan')):.1f}",
            str(r.get("gpu_mib", "")),
        ]
        for c in PHASE_COLUMNS:
            v = r.get("phases", {}).get(c)
            vals.append("" if v is None else f"{v:.1f}")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run_config(
    cfg: Config, model: str, out_dir: Path, timeout_s: int, bench_passthrough: list[str], python: str
) -> dict[str, Any]:
    run_dir = out_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(cfg.env)
    env["VLLM_OMNI_INIT_TIMELINE"] = str(run_dir / "timeline.json")
    if cfg.fresh_caches:
        for var in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            env[var] = tempfile.mkdtemp(prefix=f"omni_{var.lower()}_")
    deploy = cfg.deploy_config or _default_deploy_for(model)
    if cfg.deploy_updates:
        deploy = str(write_variant_yaml(Path(deploy), cfg.deploy_updates, run_dir, cfg.name))
    cmd = [
        python,
        str(BENCH_DIR / "stream_latency_bench.py"),
        "--init-only",
        "--warmup",
        "1",
        "--model",
        model,
        "--deploy-config",
        deploy,
        "--output-dir",
        str(run_dir / "bench"),
        "--tag",
        cfg.name,
        *cfg.extra_args,
        *bench_passthrough,
    ]
    t0 = time.perf_counter()
    with open(run_dir / "stdout.log", "w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s, cwd=str(REPO_ROOT))
    wall = time.perf_counter() - t0
    result: dict[str, Any] = {
        "config": cfg.name,
        "returncode": proc.returncode,
        "wall_s": wall,
        "cmd": cmd,
        "deploy_config": deploy,
    }
    timeline = None
    tl_path = run_dir / "timeline.json"
    if tl_path.exists():
        timeline = json.loads(tl_path.read_text())
    bench_files = sorted((run_dir / "bench").rglob("*.json"))
    if bench_files:
        doc = json.loads(bench_files[-1].read_text())
        result["init_s"] = doc["init"]["init_s"]
        reqs = doc.get("requests") or []
        result["ttfa_ms"] = reqs[0].get("ttfa_ms") if reqs else None
        result["gpu_mib"] = doc.get("resources", {}).get("peak_gpu_mem_mib")
        timeline = timeline or doc["init"].get("init_timeline")
    result["phases"] = phase_durations(timeline)
    result["phases_by_stage"] = phase_durations_by_stage(timeline)
    result["total_s"] = timeline.get("total_s") if timeline else None
    return result


def _default_deploy_for(model: str) -> str:
    # Avoid importing vllm_omni here (heavy); mirror the deploy naming rule for Qwen3-TTS.
    if "qwen3-tts" in model.lower() or "qwen3_tts" in model.lower():
        return str(DEPLOY_DIR / "qwen3_tts.yaml")
    raise SystemExit(f"Pass --deploy-config-base for model {model!r} (no default known here)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    p.add_argument("--configs", default="default,eager_stage1,eager_both,parallel_stage_init,edge")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--timeout-s", type=int, default=1500)
    p.add_argument("--output-dir", default=str(BENCH_DIR / "edge_results" / "init_profile"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--list", action="store_true")
    args, passthrough = p.parse_known_args(argv)
    if args.list:
        for name, cfg in CONFIGS.items():
            print(
                f"{name}: updates={cfg.deploy_updates} env={cfg.env} args={cfg.extra_args} "
                f"deploy={cfg.deploy_config} fresh={cfg.fresh_caches}"
            )
        return 0
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs {unknown}; known: {sorted(CONFIGS)}")
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    # Default passthrough: generous timeouts so cold compiles finish; callers may override.
    if "--init-timeout" not in passthrough:
        passthrough += ["--init-timeout", "1500"]
    if "--stage-init-timeout" not in passthrough:
        passthrough += ["--stage-init-timeout", "1200"]
    for name in names:
        for rep in range(args.repeat):
            cfg = CONFIGS[name]
            if args.repeat > 1:
                cfg = Config(
                    f"{name}_r{rep}", cfg.deploy_updates, cfg.env, cfg.extra_args, cfg.deploy_config, cfg.fresh_caches
                )
            print(f"[init_profile] running {cfg.name} ...", flush=True)
            try:
                row = run_config(cfg, args.model, out_dir, args.timeout_s, passthrough, args.python)
            except subprocess.TimeoutExpired:
                row = {"config": cfg.name, "returncode": -1, "error": f"timeout after {args.timeout_s}s", "phases": {}}
            rows.append(row)
            print(
                f"[init_profile] {cfg.name}: rc={row.get('returncode')} init_s={row.get('init_s')} "
                f"phases={row.get('phases')}",
                flush=True,
            )
    summary = {"model": args.model, "timestamp": ts, "rows": rows, "table": render_table(rows)}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(render_table(rows))
    print(f"[init_profile] wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
