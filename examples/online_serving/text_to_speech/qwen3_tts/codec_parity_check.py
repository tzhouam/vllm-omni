#!/usr/bin/env python3
"""Parity check: server-decoded PCM vs client-decoded PCM from codec tokens.

Run the full 2-stage server and the talker-only server one after another (or
on two ports) with ``enforce_eager`` on stage 0 so per-request seeds are
reproducible, then:

  python codec_parity_check.py --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \\
      --full-port 8000 --talker-port 8001 --seed 1234 --text "..."

Reports the length-trimmed SNR (dB) between the two waveforms and the codec
frame count. Both requests use the same ``seed``.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(ref.size, test.size)
    if n == 0:
        return float("nan")
    ref, test = ref[:n].astype(np.float64), test[:n].astype(np.float64)
    noise = ref - test
    p_sig = float(np.sum(ref**2)) + 1e-12
    p_noise = float(np.sum(noise**2)) + 1e-12
    return 10.0 * np.log10(p_sig / p_noise)


def fetch_full_pcm(host: str, port: int, model: str, text: str, voice: str, seed: int) -> np.ndarray:
    import requests

    body = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "seed": seed,
    }
    r = requests.post(f"http://{host}:{port}/v1/audio/speech", json=body, timeout=600)
    r.raise_for_status()
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return data.reshape(-1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--full-port", type=int, default=8000)
    p.add_argument("--talker-port", type=int, default=8001)
    p.add_argument("--voice", default="vivian")
    p.add_argument("--text", default="Parity check between server decoding and on-device decoding.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cpu")
    p.add_argument("--full-wav", default=None, help="use a previously saved full-server WAV instead of requesting one")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    import edge_decoder_client as client

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    if args.full_wav:
        import soundfile as sf

        ref, _ = sf.read(args.full_wav, dtype="float32")
        ref = ref.reshape(-1)
    else:
        ref = fetch_full_pcm(args.host, args.full_port, args.model, args.text, args.voice, args.seed)
    client_args = argparse.Namespace(
        host=args.host,
        port=args.talker_port,
        model=args.model,
        text=args.text,
        voice=args.voice,
        task_type="CustomVoice",
        language=None,
        instructions=None,
        max_new_tokens=None,
        device=args.device,
        left_context=72,
        ramp=None,
        seed=args.seed,
        out=str(out_dir / "client.wav") if out_dir else None,
        json=str(out_dir / "client.json") if out_dir else None,
    )
    result = asyncio.run(client.run(client_args))
    import soundfile as sf

    test, _ = sf.read(client_args.out, dtype="float32") if client_args.out else (np.zeros(0, dtype=np.float32), 0)
    report = {
        "ref_samples": int(ref.size),
        "client_samples": int(test.size),
        "snr_db": snr_db(ref, test),
        "length_ratio": (test.size / ref.size) if ref.size else None,
        "client_ttfa_ms": result.get("ttfa_ms"),
        "client_rtf": result.get("rtf"),
        "frames": result.get("total_frames"),
    }
    print(json.dumps(report, indent=1))
    if out_dir:
        (out_dir / "parity.json").write_text(json.dumps(report, indent=1))
        sf.write(str(out_dir / "full.wav"), ref, 24000, format="WAV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
