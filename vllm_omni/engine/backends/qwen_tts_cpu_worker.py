#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated local HTTP worker for the official Qwen3-TTS CPU wrapper.

This file is launched as a script by Omni's parent process. Its interpreter
loads a pinned Transformers 4.57 overlay; the Omni interpreter keeps its own
Transformers version. The worker owns all model and vocoder state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 0 < args.port < 65536 or not 0 < args.threads <= 24:
        parser.error("invalid port or CPU thread bound")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU TTS worker requires CUDA_VISIBLE_DEVICES empty")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("CPU TTS worker requires offline model loading")

    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from qwen_tts import Qwen3TTSModel

    overlay = os.path.normcase(os.path.realpath(os.environ["QWEN_TTS_OVERLAY_DIR"]))
    loaded_transformers = os.path.normcase(os.path.realpath(transformers.__file__))
    if transformers.__version__ != "4.57.3" or not loaded_transformers.startswith(overlay + os.sep):
        raise RuntimeError("Qwen TTS worker loaded the wrong Transformers dependency")
    if importlib.metadata.version("qwen-tts") != "0.1.1":
        raise RuntimeError("Qwen TTS worker loaded the wrong model wrapper")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(args.model_dir.resolve(strict=True)),
        device_map="cpu", dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True,
    )
    if {parameter.device.type for parameter in model.model.parameters()} != {"cpu"}:
        raise RuntimeError("Qwen TTS worker loaded a non-CPU parameter")
    if "ryan" not in model.get_supported_speakers():
        raise RuntimeError("pinned CustomVoice checkpoint does not provide Ryan")
    props = {
        "model_dir": str(args.model_dir.resolve()),
        "placement": "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "qwen_tts": "0.1.1",
        "dtype": "bfloat16",
        "attention": "sdpa",
        "threads": args.threads,
        "sample_rate": 24000,
        "speaker": "Ryan",
        "max_new_tokens": 64,
        "load_s": time.perf_counter() - started,
    }

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/props":
                self._json(200, props)
            else:
                self._json(404, {"error": "unknown endpoint"})

        def do_POST(self) -> None:
            if self.path != "/v1/audio/speech":
                self._json(404, {"error": "unknown endpoint"})
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                self._json(413, {"error": "request exceeds worker bound"})
                return
            try:
                payload = json.loads(self.rfile.read(size))
                text = payload["input"]
                if not isinstance(text, str) or not 0 < len(text.encode()) <= 4096:
                    raise ValueError("invalid bounded TTS input")
                if payload.get("response_format") != "wav" or payload.get("seed") != 42:
                    raise ValueError("worker requires WAV and pinned seed 42")
                torch.manual_seed(42)
                started = time.perf_counter()
                print("tts request-start text_sha256="
                      f"{hashlib.sha256(text.encode()).hexdigest()}", flush=True)
                wavs, sample_rate = model.generate_custom_voice(
                    text=text, language="English", speaker="Ryan",
                    max_new_tokens=64, do_sample=True, non_streaming_mode=True,
                )
                if len(wavs) != 1 or sample_rate != 24000:
                    raise RuntimeError("model did not return one 24 kHz waveform")
                audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
                if not audio.size or not np.isfinite(audio).all():
                    raise RuntimeError("model returned empty or non-finite audio")
                with io.BytesIO() as buffer:
                    sf.write(buffer, audio, sample_rate, subtype="PCM_16", format="WAV")
                    data = buffer.getvalue()
                if len(data) > (4 << 20):
                    raise RuntimeError("model returned audio above worker bound")
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                print(f"tts request duration_s={len(audio) / sample_rate:.3f} wall_s={time.perf_counter() - started:.3f}", flush=True)
            except Exception as exc:
                traceback.print_exc()
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"qwen-tts-cpu-worker ready port={args.port} props={json.dumps(props)}", flush=True)
        server.serve_forever(poll_interval=0.1)


if __name__ == "__main__":
    main()
