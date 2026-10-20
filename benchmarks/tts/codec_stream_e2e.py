#!/usr/bin/env python3
"""End-to-end codec-token streaming check on one device.

1. Launch a talker-only server (``qwen3_tts_talker_only.yaml``), run the
   edge decoder client for each prompt (decoder on ``--client-device``),
   record TTFA / playback-start / RTF / decode time and save WAVs.
2. Stop it, launch the full two-stage server, request the same prompts with
   the same ``tts_local_seed`` through ``/v1/audio/speech`` and compute the
   SNR between server-decoded and client-decoded audio.

Both servers run one after another on the same device, so this fits a single
``gpu run --gpus 1`` window (or a CPU host with ``VLLM_TARGET_DEVICE=cpu``).
Results: ``benchmarks/tts/edge_results/codec_stream/<timestamp>/summary.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
CLIENT_DIR = REPO_ROOT / "examples" / "online_serving" / "text_to_speech" / "qwen3_tts"
sys.path.insert(0, str(CLIENT_DIR))
sys.path.insert(0, str(BENCH_DIR))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Server:
    def __init__(
        self, model: str, deploy_config: str | None, port: int, log_path: Path, extra: list[str], env: dict[str, str]
    ):
        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            model,
            "--omni",
            "--port",
            str(port),
            "--trust-remote-code",
        ]
        if deploy_config:
            cmd += ["--deploy-config", deploy_config]
        cmd += extra
        self.cmd, self.port, self.log_path = cmd, port, log_path
        self.env = {**os.environ, **env}
        self.proc: subprocess.Popen | None = None

    def start(self, timeout_s: int) -> float:
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=self.env,
            cwd=str(REPO_ROOT),
            start_new_session=True,
        )
        t0 = time.perf_counter()
        import urllib.request

        while time.perf_counter() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early (rc={self.proc.returncode}); see {self.log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return time.perf_counter() - t0
            except Exception:
                time.sleep(1.0)
        raise TimeoutError(f"server on port {self.port} not healthy after {timeout_s}s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        if getattr(self, "log", None):
            self.log.close()


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(ref.size, test.size)
    if n == 0:
        return float("nan")
    ref, test = ref[:n].astype(np.float64), test[:n].astype(np.float64)
    return 10.0 * np.log10((np.sum(ref**2) + 1e-12) / (np.sum((ref - test) ** 2) + 1e-12))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    p.add_argument("--prompts", default=str(BENCH_DIR / "prompts_12.txt"))
    p.add_argument("--n-prompts", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--voice", default="vivian")
    p.add_argument("--client-device", default="cpu")
    p.add_argument("--talker-deploy", default=str(REPO_ROOT / "vllm_omni" / "deploy" / "qwen3_tts_talker_only.yaml"))
    p.add_argument("--full-deploy", default=None, help="deploy YAML for the 2-stage server (default: model default)")
    p.add_argument("--server-timeout-s", type=int, default=1500)
    p.add_argument("--output-dir", default=str(BENCH_DIR / "edge_results" / "codec_stream"))
    p.add_argument("--skip-parity", action="store_true")
    args = p.parse_args(argv)

    import edge_decoder_client as client

    prompts = [line.strip() for line in open(args.prompts) if line.strip()][: args.n_prompts]
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    stage0_eager = ["--stage-overrides", json.dumps({"0": {"enforce_eager": True}})]
    summary: dict[str, Any] = {
        "model": args.model,
        "timestamp": ts,
        "prompts": prompts,
        "client_device": args.client_device,
        "rows": [],
    }

    # --- talker-only server + client decoding
    port = _free_port()
    talker = Server(args.model, args.talker_deploy, port, out_dir / "talker_server.log", stage0_eager, {})
    try:
        summary["talker_server_start_s"] = talker.start(args.server_timeout_s)
        for i, text in enumerate(prompts):
            cargs = argparse.Namespace(
                host="127.0.0.1",
                port=port,
                model=args.model,
                text=text,
                voice=args.voice,
                task_type="CustomVoice",
                language=None,
                instructions=None,
                max_new_tokens=None,
                device=args.client_device,
                left_context=72,
                ramp=None,
                out=str(out_dir / f"client_{i}.wav"),
                json=str(out_dir / f"client_{i}.json"),
                seed=args.seed,
            )
            res = asyncio.run(client.run(cargs))
            summary["rows"].append(
                {
                    "i": i,
                    "text": text,
                    "frames": res["total_frames"],
                    "client_ttfa_ms": res["ttfa_ms"],
                    "client_playback_start_ms": res["playback_start_ms"],
                    "client_rtf": res["rtf"],
                    "client_decode_ms": res["decode_ms_total"],
                    "client_audio_s": res["audio_s"],
                }
            )
            print(
                f"[codec_e2e] talker-only prompt {i}: frames={res['total_frames']} "
                f"ttfa={res['ttfa_ms']} rtf={res['rtf']}",
                flush=True,
            )
    finally:
        talker.stop()

    # --- full server for parity
    if not args.skip_parity:
        import soundfile as sf

        port = _free_port()
        full = Server(args.model, args.full_deploy, port, out_dir / "full_server.log", stage0_eager, {})
        try:
            summary["full_server_start_s"] = full.start(args.server_timeout_s)
            import urllib.request

            for row in summary["rows"]:
                body = json.dumps(
                    {
                        "model": args.model,
                        "input": row["text"],
                        "voice": args.voice,
                        "task_type": "CustomVoice",
                        "response_format": "wav",
                        "seed": args.seed,
                    }
                ).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/audio/speech", data=body, headers={"Content-Type": "application/json"}
                )
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=600) as r:
                    wav = r.read()
                row["full_server_s"] = time.perf_counter() - t0
                ref, sr = sf.read(io.BytesIO(wav), dtype="float32")
                ref = ref.reshape(-1)
                sf.write(str(out_dir / f"full_{row['i']}.wav"), ref, sr, format="WAV")
                test, _ = sf.read(str(out_dir / f"client_{row['i']}.wav"), dtype="float32")
                row["full_audio_s"] = ref.size / sr
                row["snr_db"] = snr_db(ref, test.reshape(-1))
                row["length_ratio"] = (test.size / ref.size) if ref.size else None
                print(
                    f"[codec_e2e] parity prompt {row['i']}: snr_db={row['snr_db']:.1f} len_ratio={row['length_ratio']}",
                    flush=True,
                )
        finally:
            full.stop()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    for row in summary["rows"]:
        print(json.dumps(row))
    print(f"[codec_e2e] wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
