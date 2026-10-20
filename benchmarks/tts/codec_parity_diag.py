#!/usr/bin/env python3
"""Offline parity diagnosis for the codec-token stream (CPU only).

Given a codec_stream_e2e.py output directory (client_<i>.codes.npy, client_<i>.wav,
full_<i>.wav) decode the *received* codes with the exact (non-chunked) decoder and
compare (a) against the client's chunked decode (tests the client's chunking) and
(b) against the full server's PCM (tests token equality + the server's decode).
Reports lag-aligned SNR and correlation.

  python benchmarks/tts/codec_parity_diag.py --dir benchmarks/tts/edge_results/codec_stream/<ts>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def aligned_snr(ref: np.ndarray, test: np.ndarray, max_lag: int = 4000) -> tuple[float, float, int]:
    n = min(ref.size, test.size)
    ref, test = ref[:n].astype(np.float64), test[:n].astype(np.float64)
    if n < 3 * max_lag:
        lag = 0
    else:
        c = np.correlate(ref[max_lag:-max_lag], test, mode="valid")
        lag = int(np.argmax(c)) - max_lag
    if lag >= 0:
        a, b = test[lag:], ref[: n - lag]
    else:
        a, b = test[: n + lag], ref[-lag:]
    m = min(a.size, b.size)
    a, b = a[:m], b[:m]
    snr = 10 * np.log10((b**2).sum() / (((b - a) ** 2).sum() + 1e-12))
    corr = float(np.corrcoef(a, b)[0, 1]) if m > 1 else 0.0
    return float(snr), corr, lag


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    args = p.parse_args(argv)
    from vllm_omni.edge.decoder_export import load_code2wav_decoder

    dec = load_code2wav_decoder(args.model)
    d = Path(args.dir)
    report = []
    for codes_path in sorted(d.glob("client_*.codes.npy")):
        i = codes_path.name.split("_")[1].split(".")[0]
        codes = np.load(codes_path)
        if codes.size == 0:
            continue
        with torch.inference_mode():
            exact = dec._forward_exact(torch.as_tensor(codes.T[None], dtype=torch.long)).reshape(-1).numpy()
        client, _ = sf.read(d / f"client_{i}.wav", dtype="float32")
        full, _ = sf.read(d / f"full_{i}.wav", dtype="float32")
        s1, c1, l1 = aligned_snr(exact, client.reshape(-1))
        s2, c2, l2 = aligned_snr(full.reshape(-1), exact)
        row = {
            "i": int(i),
            "frames": int(codes.shape[0]),
            "exact_vs_client_chunked": {"snr_db": s1, "corr": c1, "lag": l1},
            "full_server_vs_exact": {"snr_db": s2, "corr": c2, "lag": l2},
        }
        report.append(row)
        print(json.dumps(row))
    (d / "parity_diag.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
