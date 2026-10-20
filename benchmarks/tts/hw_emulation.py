#!/usr/bin/env python3
"""Emulate edge hardware classes on a development host and benchmark each.

Each case sets ``VLLM_OMNI_HW_PROFILE`` to a synthetic hardware profile (so
``deploy_profile="auto"`` derives budgets/threads/dtype for that class), plus
host-level levers that really constrain execution:

* CPU cases: ``ATEN_CPU_CAPABILITY`` (torch ISA dispatch: ``avx2`` /
  ``avx512`` / ``default``) and ``taskset`` core pinning; run in a CPU vLLM
  environment with ``VLLM_TARGET_DEVICE=cpu``.
* CUDA cases (Jetson-like / small discrete GPU): the profile overrides GPU
  memory so the derived KV budgets and eager switches apply; the card's real
  memory is not capped (documented in the result).

Writes one ``stream_latency_bench.py`` JSON per case plus a summary table under
``benchmarks/tts/edge_results/hw_emulation/<timestamp>/``.

  # CPU cases (no GPU):
  VLLM_TARGET_DEVICE=cpu python benchmarks/tts/hw_emulation.py \\
    --cases x86_cpu_avx2,x86_cpu_avx512,x86_cpu_amx,arm64_cpu_logic
  # GPU cases, one GPU through the scheduler:
  gpu run --gpus 1 --timeout 40m --note "hw emulation gpu" -- env HF_HUB_OFFLINE=1 \\
    python benchmarks/tts/hw_emulation.py --cases jetson_orin_8g,jetson_orin_32g,cuda_discrete_8g
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.edge.hardware_probe import HardwareProfile, probe  # noqa: E402

GiB = 2**30


@dataclass
class Case:
    name: str
    profile: dict[str, Any]  # fields overriding the live probe
    env: dict[str, str] = field(default_factory=dict)
    cpus: list[int] | None = None  # taskset pinning
    gpu: bool = False
    note: str = ""


def _cpu_profile(flags: list[str], n: int, ram_gib: int, arch: str = "x86_64", model: str = "") -> dict[str, Any]:
    return {
        "arch": arch,
        "cpu_model": model or f"emulated {arch}",
        "cpu_flags": flags,
        "n_logical": n,
        "n_physical": n,
        "clusters": [{"max_khz": 3000000, "cpus": list(range(n))}],
        "ram_total_bytes": ram_gib * GiB,
        "ram_available_bytes": int(ram_gib * 0.75) * GiB,
        "accelerator": "none",
        "supports_bf16": "avx512_bf16" in flags or "bf16" in flags,
        "supports_fp16": True,
        "source": "emulation",
    }


def _gpu_profile(name: str, vram_gib: int, unified: bool, ram_gib: int, arch: str) -> dict[str, Any]:
    return {
        "arch": arch,
        "cpu_model": f"emulated {name} host",
        "cpu_flags": ["neon", "dotprod", "fp16_arith"] if arch == "aarch64" else ["avx2", "avx512"],
        "n_logical": 8,
        "n_physical": 8,
        "clusters": [{"max_khz": 2200000, "cpus": list(range(8))}],
        "ram_total_bytes": ram_gib * GiB,
        "ram_available_bytes": int(ram_gib * 0.75) * GiB,
        "accelerator": "cuda_unified" if unified else "cuda_discrete",
        "gpu_name": name,
        "gpu_mem_bytes": vram_gib * GiB,
        "gpu_sm_count": 16,
        "gpu_capability": [8, 7] if unified else [8, 6],
        "supports_bf16": True,
        "supports_fp16": True,
        "source": "emulation",
    }


CASES: dict[str, Case] = {
    "x86_cpu_avx2": Case(
        "x86_cpu_avx2",
        _cpu_profile(["avx2", "fma", "f16c"], 4, 8),
        env={"ATEN_CPU_CAPABILITY": "avx2"},
        cpus=[0, 1, 2, 3],
        note="4 cores, AVX2 dispatch, 8 GiB profile -> float32",
    ),
    "x86_cpu_avx512": Case(
        "x86_cpu_avx512",
        _cpu_profile(["avx2", "avx512", "avx512_bf16"], 8, 16),
        env={"ATEN_CPU_CAPABILITY": "avx512"},
        cpus=list(range(8)),
        note="8 cores, AVX-512 (+bf16) dispatch, 16 GiB",
    ),
    "x86_cpu_amx": Case(
        "x86_cpu_amx",
        _cpu_profile(["avx2", "avx512", "avx512_bf16", "amx_bf16", "amx_tile"], 16, 32),
        cpus=list(range(16)),
        note="16 cores, AMX bf16, 32 GiB",
    ),
    "arm64_cpu_logic": Case(
        "arm64_cpu_logic",
        _cpu_profile(["neon", "dotprod", "fp16_arith", "i8mm"], 4, 8, arch="aarch64", model="emulated RK3588"),
        cpus=[0, 1, 2, 3],
        note="derivation/binding logic only: ISA is x86 underneath (labelled logic-only)",
    ),
    "jetson_orin_8g": Case(
        "jetson_orin_8g",
        _gpu_profile("Jetson Orin 8GB (emulated)", 8, True, 8, "aarch64"),
        gpu=True,
        note="unified 8 GiB -> tight budgets, eager",
    ),
    "jetson_orin_32g": Case(
        "jetson_orin_32g",
        _gpu_profile("Jetson Orin 32GB (emulated)", 32, True, 32, "aarch64"),
        gpu=True,
        note="unified 32 GiB",
    ),
    "cuda_discrete_8g": Case(
        "cuda_discrete_8g",
        _gpu_profile("small discrete GPU (emulated)", 8, False, 16, "x86_64"),
        gpu=True,
        note="8 GiB VRAM -> eager + small KV",
    ),
}


def build_profile(case: Case) -> HardwareProfile:
    base = probe(use_torch=case.gpu).to_dict()
    base.update(case.profile)
    return HardwareProfile.from_dict(base)


def run_case(case: Case, args: argparse.Namespace, out_dir: Path, passthrough: list[str]) -> dict[str, Any]:
    case_dir = out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    prof = build_profile(case)
    prof_path = case_dir / "hw_profile.json"
    prof_path.write_text(prof.to_json())
    env = dict(os.environ)
    env.update(case.env)
    env["VLLM_OMNI_HW_PROFILE"] = str(prof_path)
    env["VLLM_OMNI_INIT_TIMELINE"] = str(case_dir / "timeline.json")
    if not case.gpu:
        env.setdefault("VLLM_TARGET_DEVICE", "cpu")
    cmd: list[str] = []
    if case.cpus and shutil.which("taskset"):
        cmd += ["taskset", "-c", ",".join(str(c) for c in case.cpus)]
    cmd += [
        args.python,
        str(BENCH_DIR / "stream_latency_bench.py"),
        "--model",
        args.model,
        "--deploy-profile",
        "auto",
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
        case.name,
        *passthrough,
    ]
    with open(case_dir / "stdout.log", "w") as log:
        try:
            proc = subprocess.run(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_s, cwd=str(REPO_ROOT)
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -1
    row: dict[str, Any] = {
        "case": case.name,
        "note": case.note,
        "returncode": rc,
        "cmd": cmd,
        "hardware_class": parse_hardware_class(case_dir / "stdout.log"),
    }
    files = sorted((case_dir / "bench").rglob("*.json"))
    if files:
        doc = json.loads(files[-1].read_text())
        s = doc["summary"]
        row.update(
            init_s=doc["init"]["init_s"],
            ttfa_p50=s["ttfa_ms"]["p50"],
            playback_start_p50=s["playback_start_ms"]["p50"],
            stall_p50=s["stall_at_ttfa_ms"]["p50"],
            rtf_total_p50=s["rtf_total"]["p50"],
            peak_rss_gib=doc["resources"]["peak_rss_bytes"] / GiB,
            peak_gpu_mib=doc["resources"]["peak_gpu_mem_mib"],
        )
    return row


_AUTO_PROFILE_RE = re.compile(r"\[auto_profile\] (\S+) -> ")


def parse_hardware_class(log_path: Path) -> str | None:
    """Hardware class chosen by ``deploy_profile=auto`` as logged by auto_profile."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    m = _AUTO_PROFILE_RE.search(text)
    return m.group(1) if m else None


def render_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        "case",
        "rc",
        "init_s",
        "ttfa_p50",
        "playback_start_p50",
        "stall_p50",
        "rtf_total_p50",
        "peak_rss_gib",
        "peak_gpu_mib",
        "note",
    ]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        vals = [r["case"], str(r.get("returncode"))]
        for c in ("init_s", "ttfa_p50", "playback_start_p50", "stall_p50", "rtf_total_p50", "peak_rss_gib"):
            v = r.get(c)
            vals.append("" if v is None else (f"{v:.3f}" if c == "rtf_total_p50" else f"{v:.1f}"))
        vals.append(str(r.get("peak_gpu_mib", "")))
        vals.append(r.get("note", ""))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", default="x86_cpu_avx2,x86_cpu_avx512,x86_cpu_amx,arm64_cpu_logic")
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    p.add_argument("--prompts", default=str(BENCH_DIR / "prompts_12.txt"))
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--timeout-s", type=int, default=2400)
    p.add_argument("--output-dir", default=str(BENCH_DIR / "edge_results" / "hw_emulation"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--list", action="store_true")
    args, passthrough = p.parse_known_args(argv)
    if args.list:
        for name, c in CASES.items():
            print(f"{name}: gpu={c.gpu} env={c.env} cpus={c.cpus} {c.note}")
        return 0
    names = [n.strip() for n in args.cases.split(",") if n.strip()]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit(f"Unknown cases {unknown}; known: {sorted(CASES)}")
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    if "--init-timeout" not in passthrough:
        passthrough += ["--init-timeout", "1500"]
    if "--stage-init-timeout" not in passthrough:
        passthrough += ["--stage-init-timeout", "1200"]
    rows = []
    for name in names:
        print(f"[hw_emulation] {name} ...", flush=True)
        row = run_case(CASES[name], args, out_dir, passthrough)
        rows.append(row)
        print(
            f"[hw_emulation] {name}: rc={row['returncode']} "
            f"ttfa_p50={row.get('ttfa_p50')} rtf={row.get('rtf_total_p50')}",
            flush=True,
        )
    summary = {"model": args.model, "timestamp": ts, "rows": rows, "table": render_table(rows)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(render_table(rows))
    print(f"[hw_emulation] wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
