# SPDX-License-Identifier: Apache-2.0
"""Timed DreamZero rollout for AR-Diffusion-vs-main comparison.

Runs one session of (prefill + N chunk) forwards, timing each `omni.generate`
(the per-forward E2E), saving the decoded latents for a precision check, and
dumping a timing JSON. Works on both the AR-Diffusion branch (with AR_DIFFUSION_KV_ENABLE=1) and a
clean main checkout (baseline) — it only uses helpers from export_prediction_video,
which exist on both.

    AR_DIFFUSION_KV_ENABLE=1 CUDA_VISIBLE_DEVICES=0 HF_HOME=/models \\
      python examples/offline_inference/dreamzero/ar_diffusion_perf_compare.py \\
      --num-chunks 12 --tag bde \\
      --latents outputs/bde_parity/perf_bde.pt --timing outputs/bde_parity/perf_bde.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_prediction_video as E  # noqa: E402


def _stats(times: list, warmup: int) -> dict:
    """prefill / steady (post-warmup) / total stats for a per-request time series."""
    if not times:
        return {"prefill_s": None, "steady_mean_s": None, "steady_min_s": None,
                "steady_max_s": None, "steady_n": 0, "total_s": None}
    prefill = times[0]
    steady = times[warmup:] if len(times) > warmup else (times[1:] or times)
    return {
        "prefill_s": prefill,
        "steady_mean_s": sum(steady) / len(steady),
        "steady_min_s": min(steady),
        "steady_max_s": max(steady),
        "steady_n": len(steady),
        "total_s": sum(times),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GEAR-Dreams/DreamZero-DROID")
    ap.add_argument("--deploy-config", type=Path, default=Path("vllm_omni/deploy/dreamzero.yaml"))
    # Canonical full-length (419-frame) set shared with the upstream test_client_AR.py
    # so both stacks read byte-identical frames; the legacy /models/dreamzero-assets is
    # only 24 frames (degenerate for multi-chunk rollouts). Prompt matches upstream too.
    ap.add_argument("--video-dir", type=Path, default=Path("/models/dreamzero-assets-full"))
    ap.add_argument(
        "--prompt",
        default="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
    )
    ap.add_argument("--session-id", default="perf-cmp")
    ap.add_argument("--num-chunks", type=int, default=12)
    ap.add_argument("--tag", default="run")
    ap.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA graph (enforce_eager=False); default runs eager.",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Forwards to drop for steady-state stats (default 2 = prefill + 1 warmup)",
    )
    ap.add_argument("--latents", type=Path, default=None)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--timing", type=Path, default=None)
    ap.add_argument(
        "--repeat-chunk-observations",
        action="store_true",
        help="Pad short asset dirs by repeating the last chunk to reach --num-chunks. "
        "Off by default so the schedule matches upstream's stop-early behavior; only "
        "needed for the degenerate 24-frame asset set.",
    )
    args = ap.parse_args()

    _, observations = E._build_observations(
        args.video_dir,
        prompt=args.prompt,
        session_id=args.session_id,
        num_chunks=args.num_chunks,
        repeat_chunk_observations=args.repeat_chunk_observations,
    )
    n = len(observations)
    print(
        f"[perf:{args.tag}] {n} forwards (1 prefill + {args.num_chunks} chunk) "
        f"AR_DIFFUSION_KV_ENABLE={os.environ.get('AR_DIFFUSION_KV_ENABLE', '0')}"
    )

    t0 = time.perf_counter()
    omni = E.Omni(
        model=args.model,
        deploy_config=str(args.deploy_config),
        enforce_eager=not args.cuda_graph,
        worker_extension_cls=E.WORKER_EXTENSION,
    )
    load_s = time.perf_counter() - t0
    print(f"[perf:{args.tag}] engine load: {load_s:.2f}s")

    # Drain any startup/warmup forward timings (e.g. the engine's dummy forward at
    # construction) so server_e2e_s aligns 1:1 with the rollout's per_forward wall times.
    try:
        engine0 = getattr(omni.engine.stage_clients[0], "_engine", None)
        if engine0 is not None:
            engine0.executor.collective_rpc("ar_diffusion_perf_stats", unique_reply_rank=0, exec_all_ranks=True)
    except Exception as e:  # noqa: BLE001 — profiling extra must never fail the run
        print(f"[perf:{args.tag}] WARN: could not drain warmup timings: {e}")

    outputs = []
    per_forward = []
    for index, obs in enumerate(observations):
        sp = E.OmniDiffusionSamplingParams(
            extra_args={"reset": index == 0, "session_id": obs["session_id"], "robot_obs": obs}
        )
        s = time.perf_counter()
        result = omni.generate(obs["prompt"], sampling_params_list=[sp])  # blocking -> E2E
        dt = time.perf_counter() - s
        if not result:
            raise RuntimeError(f"No output for forward {index}")
        outputs.append(result[0])
        per_forward.append(dt)
        print(f"[perf:{args.tag}] forward {index:2d} ({'prefill' if index == 0 else 'chunk'}): {dt * 1000:8.1f} ms")

    # Worker-side per-request forward E2E (true compute time, excludes the
    # engine<->worker IPC the omni.generate wall time includes) + peak GPU memory,
    # both queried from the worker process where the model + KV pools live.
    server_e2e: list = []
    peak_reserved_gib = None
    peak_allocated_gib = None
    try:
        engine = getattr(omni.engine.stage_clients[0], "_engine", None)
        if engine is not None:
            mem = engine.executor.collective_rpc("gpu_mem_stats", unique_reply_rank=0, exec_all_ranks=True)
            mem = mem[0] if isinstance(mem, (list, tuple)) else mem
            peak_reserved_gib = mem["peak_reserved_gib"]
            peak_allocated_gib = mem["peak_allocated_gib"]
            perf = engine.executor.collective_rpc("ar_diffusion_perf_stats", unique_reply_rank=0, exec_all_ranks=True)
            perf = perf[0] if isinstance(perf, (list, tuple)) else perf
            server_e2e = list(perf.get("server_e2e_s") or [])
    except Exception as e:  # noqa: BLE001 — profiling extra must never fail the run
        print(f"[perf:{args.tag}] WARN: could not read worker stats: {e}")

    # Headline stats use the worker-side E2E when available (true compute time),
    # else fall back to the omni.generate wall time.
    basis = "server_e2e" if server_e2e else "wall"
    primary = server_e2e if server_e2e else per_forward
    e2e = _stats(primary, args.warmup)
    wall = _stats(per_forward, args.warmup)

    summary = {
        "tag": args.tag,
        "bde_kv_enable": os.environ.get("AR_DIFFUSION_KV_ENABLE", "0"),
        "bde_kv_no_memo": os.environ.get("AR_DIFFUSION_KV_NO_MEMO", "0"),
        "num_forwards": n,
        "warmup_dropped": args.warmup,
        "timing_basis": basis,
        "load_s": load_s,
        # headline (worker-side E2E when available)
        "prefill_s": e2e["prefill_s"],
        "steady_mean_s": e2e["steady_mean_s"],
        "steady_min_s": e2e["steady_min_s"],
        "steady_max_s": e2e["steady_max_s"],
        "total_gen_s": e2e["total_s"],
        "peak_reserved_gib": peak_reserved_gib,
        "peak_allocated_gib": peak_allocated_gib,
        # per-request series (full) + the post-warmup steady slice of the E2E series
        "per_forward_s": primary,
        "server_e2e_s": server_e2e or None,
        "server_e2e_steady_s": primary[args.warmup:] if len(primary) > args.warmup else None,
        "wall_per_request_s": per_forward,
        # wall-time stats kept separately (include engine<->worker IPC)
        "wall_prefill_s": wall["prefill_s"],
        "wall_steady_mean_s": wall["steady_mean_s"],
        "wall_total_s": wall["total_s"],
    }
    mem_str = f"  peak_reserved={peak_reserved_gib:.2f}GiB" if peak_reserved_gib is not None else ""
    print(
        f"[perf:{args.tag}] basis={basis}  prefill={(e2e['prefill_s'] or 0) * 1000:.1f}ms  "
        f"steady_mean={(e2e['steady_mean_s'] or 0) * 1000:.1f}ms (n={e2e['steady_n']})  "
        f"total={(e2e['total_s'] or 0):.2f}s{mem_str}"
    )
    if server_e2e:
        print(
            f"[perf:{args.tag}] wall steady_mean={(wall['steady_mean_s'] or 0) * 1000:.1f}ms "
            f"(incl. engine<->worker IPC)"
        )

    # Decode once (after timing) so latents + the final video are both saved.
    latents = torch.cat([E._extract_latents(o) for o in outputs], dim=2)
    if args.latents is not None:
        args.latents.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents.detach().cpu(), args.latents)
        print(f"[perf:{args.tag}] SAVED_LATENTS={args.latents} shape={tuple(latents.shape)}")
    if args.video is not None:
        frames = E._decode_with_worker(omni, latents)
        args.video.parent.mkdir(parents=True, exist_ok=True)
        E._write_mp4(args.video, frames, fps=args.fps)
        print(f"[perf:{args.tag}] SAVED_MP4={args.video} frames={frames.shape[0]}")
    if args.timing is not None:
        args.timing.parent.mkdir(parents=True, exist_ok=True)
        args.timing.write_text(json.dumps(summary, indent=2))
        print(f"[perf:{args.tag}] SAVED_TIMING={args.timing}")


if __name__ == "__main__":
    main()
