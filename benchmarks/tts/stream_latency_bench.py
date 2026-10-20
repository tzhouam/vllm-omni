#!/usr/bin/env python3
"""Offline streaming-latency benchmark for chunk-streamed TTS pipelines (AsyncOmni).

Reuses the Qwen3-TTS example's input builder so prompts/templates are identical to
``examples/offline_inference/text_to_speech/qwen3_tts/end2end.py``; every flag that
example accepts is passed through (``--query-type``, ``--txt-prompts``,
``--deploy-config``, ``--init-timeout``, ``--stage-init-timeout``, ...).

Bench-only flags:
  --model            override the model id chosen by the example (e.g. the 0.6B variants)
  --warmup N         warm-up requests (excluded from statistics), default 1
  --repeat K         repeat the prompt list K times, default 1
  --init-only        initialize the engine, run the warm-ups, write JSON, exit
  --tag              free-form tag appended to the result file name
  --output-dir       results root (default benchmarks/tts/edge_results/stream_latency)
  --save-wav         also write the streamed audio per request
  --gpu-poll-interval-s  resource poller interval (default 0.2)

Metrics per request: TTFA, chunk arrivals, uninterrupted-playback start, stall at
TTFA, RTF (streamed and streamed+tail), see ``stream_latency_metrics.py``.

Example (one GPU through the scheduler):
  gpu run --gpus 1 --timeout 40m --note "wp0 baseline" -- env HF_HUB_OFFLINE=1 \\
    python benchmarks/tts/stream_latency_bench.py --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \\
    --query-type CustomVoice --txt-prompts benchmarks/tts/prompts_12.txt --warmup 1 --repeat 3
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
EXAMPLE_DIR = REPO_ROOT / "examples" / "offline_inference" / "text_to_speech" / "qwen3_tts"
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(EXAMPLE_DIR))

import stream_latency_metrics as metrics  # noqa: E402

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- args
def parse_bench_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", default=None)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--init-only", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--output-dir", default=str(BENCH_DIR / "edge_results" / "stream_latency"))
    p.add_argument("--save-wav", action="store_true")
    p.add_argument("--gpu-poll-interval-s", type=float, default=0.2)
    p.add_argument("--parallel-stage-init", action="store_true", help="Pass parallel_stage_init=True to the engine")
    p.add_argument("--deploy-profile", default=None, help="Named deploy profile (e.g. edge); forwarded to Omni")
    p.add_argument(
        "--step-stats",
        action="store_true",
        help="Collect per-step overhead attribution (VLLM_OMNI_STEP_STATS_DIR) and embed it in the JSON",
    )
    p.add_argument("--bench-help", action="store_true")
    bench, rest = p.parse_known_args(argv)
    if bench.bench_help:
        p.print_help()
        sys.exit(0)
    return bench, rest


def build_inputs(example_args: argparse.Namespace, model_override: str | None) -> tuple[str, list[dict[str, Any]]]:
    import end2end  # example module, imported lazily so --bench-help works without deps

    model_name, inputs = end2end._build_inputs(example_args)
    if model_override and model_override != model_name:
        model_name = model_override
        rebuilt = []
        for item in inputs:
            info = item["additional_information"]
            rebuilt.append(
                {
                    "prompt_token_ids": [0] * end2end._estimate_prompt_len(info, model_name),
                    "additional_information": info,
                }
            )
        inputs = rebuilt
    return model_name, inputs


# ----------------------------------------------------------------- resources
class ResourcePoller(threading.Thread):
    """Samples RSS of this process tree and GPU memory of its processes."""

    def __init__(self, interval_s: float = 0.2):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop_evt = threading.Event()
        self.peak_rss_bytes = 0
        self.peak_gpu_mem_mib = 0
        self.gpu_mem_samples = 0
        self.gpu_mem_mode = "none"
        try:
            import psutil

            self._proc = psutil.Process()
        except Exception:  # pragma: no cover - psutil is a vllm dependency
            self._proc = None

    def _pids(self) -> set[int]:
        if self._proc is None:
            return {os.getpid()}
        try:
            return {self._proc.pid, *(c.pid for c in self._proc.children(recursive=True))}
        except Exception:
            return {os.getpid()}

    def _sample_rss(self, pids: set[int]) -> int:
        if self._proc is None:
            return 0
        import psutil

        total = 0
        for pid in pids:
            try:
                total += psutil.Process(pid).memory_info().rss
            except Exception:
                continue
        return total

    def _sample_gpu(self, pids: set[int]) -> int | None:
        if os.environ.get("VLLM_TARGET_DEVICE", "").lower() == "cpu":
            return None  # CPU platform: no GPU belongs to this run
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        if out.returncode != 0:
            return None
        total = 0
        matched = False
        for line in out.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                pid, mem = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if pid in pids:
                total += mem
                matched = True
        if matched:
            self.gpu_mem_mode = "per-process"
            return total
        # fallback: whole-GPU usage of the first visible device
        idx = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0").strip()
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", idx],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                self.gpu_mem_mode = "whole-gpu"
                return int(out.stdout.strip().splitlines()[0])
        except Exception:
            return None
        return None

    def run(self) -> None:
        while not self._stop_evt.is_set():
            pids = self._pids()
            self.peak_rss_bytes = max(self.peak_rss_bytes, self._sample_rss(pids))
            gpu = self._sample_gpu(pids)
            if gpu is not None:
                self.gpu_mem_samples += 1
                self.peak_gpu_mem_mib = max(self.peak_gpu_mem_mib, gpu)
            self._stop_evt.wait(self.interval_s)

    def stop(self) -> dict[str, Any]:
        self._stop_evt.set()
        self.join(timeout=5)
        return {
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_gpu_mem_mib": self.peak_gpu_mem_mib,
            "gpu_mem_samples": self.gpu_mem_samples,
            "gpu_mem_mode": self.gpu_mem_mode,
        }


def _git_sha(path: Path) -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def _gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


# --------------------------------------------------------------------- audio
def _cat_audio(audio: Any) -> torch.Tensor | None:
    if audio is None:
        return None
    if isinstance(audio, list):
        parts = [a for a in audio if a is not None]
        if not parts:
            return None
        audio = torch.cat([torch.as_tensor(a).reshape(-1) for a in parts], dim=-1)
    return torch.as_tensor(audio).reshape(-1).float().cpu()


def _sample_rate(mm: dict[str, Any] | None, default: int = 24000) -> int:
    if not mm or mm.get("sr") is None:
        return default
    s = mm["sr"]
    s = s[-1] if isinstance(s, list) and s else s
    return int(s.item() if hasattr(s, "item") else s)


# ----------------------------------------------------------------------- run
async def run_request(omni: Any, prompt: dict[str, Any], request_id: str) -> tuple[dict[str, Any], torch.Tensor | None]:
    t_start = time.perf_counter()
    t_prev = t_start
    chunks: list[dict[str, Any]] = []
    streamed: list[torch.Tensor] = []
    tail: torch.Tensor | None = None
    sr = 24000
    t_end = t_start
    async for so in omni.generate(prompt, request_id=request_id):
        now = time.perf_counter()
        mm = so.outputs[0].multimodal_output if so.outputs else None
        sr = _sample_rate(mm, sr)
        audio = _cat_audio(mm.get("audio") if mm else None)
        if not so.finished:
            n = 0 if audio is None else int(audio.numel())
            chunks.append(
                {"idx": len(chunks), "t_ms": (now - t_start) * 1e3, "gap_ms": (now - t_prev) * 1e3, "samples": n}
            )
            if audio is not None and n:
                streamed.append(audio)
            t_prev = now
        else:
            t_end = now
            tail = audio
    total_wall_s = t_end - t_start
    tail_s = (tail.numel() / sr) if tail is not None else 0.0
    rec = {
        "request_id": request_id,
        "sr": sr,
        "chunks": chunks,
        "t_final_ms": total_wall_s * 1e3,
        **metrics.request_metrics(chunks, sr, total_wall_s, tail_audio_s=tail_s),
    }
    text = prompt.get("additional_information", {}).get("text")
    rec["text"] = text[0] if isinstance(text, list) and text else text
    audio_out = torch.cat(streamed) if streamed else None
    return rec, audio_out


async def main() -> int:
    bench, rest = parse_bench_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]
    import end2end

    example_args = end2end.parse_args()
    model_name, inputs = build_inputs(example_args, bench.model)

    if bench.step_stats and not os.environ.get("VLLM_OMNI_STEP_STATS_DIR"):
        # Must be exported before the stage processes spawn (they inherit it).
        stats_dir = Path(bench.output_dir) / "step_stats" / _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        stats_dir.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_OMNI_STEP_STATS_DIR"] = str(stats_dir)

    omni_kwargs = vars(example_args).copy()
    omni_kwargs["model"] = model_name
    if bench.parallel_stage_init:
        omni_kwargs["parallel_stage_init"] = True
    if bench.deploy_profile:
        omni_kwargs["deploy_profile"] = bench.deploy_profile
    from vllm_omni import AsyncOmni

    poller = ResourcePoller(bench.gpu_poll_interval_s)
    poller.start()
    t0 = time.perf_counter()
    omni = AsyncOmni(**omni_kwargs)
    init_s = time.perf_counter() - t0

    results: list[dict[str, Any]] = []
    wavs: dict[str, torch.Tensor] = {}
    plan: list[tuple[str, dict[str, Any]]] = []
    for i in range(bench.warmup):
        plan.append((f"w{i}", inputs[i % len(inputs)]))
    if not bench.init_only:
        for r in range(bench.repeat):
            for i, prompt in enumerate(inputs):
                plan.append((f"r{r}_{i}", prompt))
    for rid, prompt in plan:
        rec, audio = await run_request(omni, prompt, rid)
        rec["warmup"] = rid.startswith("w")
        results.append(rec)
        if audio is not None:
            wavs[rid] = audio
        print(
            f"[bench] {rid}: TTFA={rec['ttfa_ms']} ms playback_start={rec['playback_start_ms']} ms "
            f"stall@ttfa={rec['stall_at_ttfa_ms']} ms chunks={rec['n_chunks']} audio={rec['streamed_audio_s']:.2f}s "
            f"wall={rec['total_wall_s']:.3f}s rtf_total={rec['rtf_total']}",
            flush=True,
        )
    resources = poller.stop()
    measured = [r for r in results if not r["warmup"]]
    summary = metrics.summarize(measured)

    step_stats = None
    if bench.step_stats:
        from vllm_omni.edge.step_stats import StepStats, format_table, merge_dir

        StepStats.get().dump()  # orchestrator-side counters live in this process
        step_stats = merge_dir(os.environ["VLLM_OMNI_STEP_STATS_DIR"])
        print("[bench] step overhead attribution (share of the 80 ms frame budget):", flush=True)
        print(format_table(step_stats["attribution"]), flush=True)

    timeline = None
    tl_path = os.environ.get("VLLM_OMNI_INIT_TIMELINE")
    if tl_path and os.path.exists(tl_path):
        try:
            with open(tl_path) as f:
                timeline = json.load(f)
        except Exception:
            timeline = None

    deploy_cfg = getattr(example_args, "deploy_config", None)
    model_tag = model_name.split("/")[-1]
    deploy_tag = (
        Path(deploy_cfg).stem
        if deploy_cfg
        else (f"profile_{bench.deploy_profile}" if bench.deploy_profile else "default")
    )
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(bench.output_dir) / model_tag / deploy_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"{ts}_{bench.tag}.json" if bench.tag else f"{ts}.json")
    doc = {
        "schema": SCHEMA_VERSION,
        "meta": {
            "model": model_name,
            "deploy_config": deploy_cfg,
            "query_type": getattr(example_args, "query_type", None),
            "txt_prompts": getattr(example_args, "txt_prompts", None),
            "git_sha": _git_sha(REPO_ROOT),
            "gpu_name": _gpu_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "timestamp": ts,
            "argv": sys.argv[1:],
            "bench_args": vars(bench),
            "warmup": bench.warmup,
            "repeat": bench.repeat,
            "init_only": bench.init_only,
        },
        "init": {"init_s": init_s, "init_timeline": timeline},
        "step_stats": step_stats,
        "resources": resources,
        "requests": results,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1)
    if bench.save_wav:
        import soundfile as sf

        for rid, audio in wavs.items():
            sf.write(str(out_dir / f"{ts}_{rid}.wav"), audio.numpy(), results[0]["sr"], format="WAV")
    print(f"[bench] init_s={init_s:.1f} resources={resources}", flush=True)
    print(f"[bench] SUMMARY {json.dumps(summary)}", flush=True)
    print(f"[bench] wrote {out_path}", flush=True)
    shutdown = getattr(omni, "shutdown", None)
    if callable(shutdown):
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
