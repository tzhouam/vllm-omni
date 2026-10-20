#!/usr/bin/env python3
"""Codec-token streaming client with on-device decoding (Qwen3-TTS).

Connects to ``/v1/audio/speech/stream`` of a **talker-only** deployment
(``--deploy-config qwen3_tts_talker_only.yaml``), requests
``output_mode="codec_tokens"``, receives 16-codebook frames at 12.5 Hz and
decodes them locally with ``Qwen3TTSTokenizerV2Decoder`` (from the model's
``speech_tokenizer/`` folder) using the same chunk schedule / 72-frame left
context as the server-side Code2Wav stage. Measures TTFA (first PCM out),
per-chunk decode time, playback-start latency and RTF, and writes a WAV.

Example:
  python edge_decoder_client.py --host localhost --port 8000 \\
      --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --voice vivian \\
      --text "Hello from the edge." --device cpu --out out.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "tts"))

from vllm_omni.entrypoints.openai.codec_stream import FLAG_EOS, FLAG_RESET, unpack_codec_frame  # noqa: E402
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import parse_chunk_ramp  # noqa: E402

DEFAULT_RAMP = [2, 4, 8, 16, 25]


# ------------------------------------------------------------------ decoder
class LocalDecoder:
    """Stateful chunked Qwen3-TTS decoder mirroring the Code2Wav stage."""

    def __init__(
        self,
        model: str,
        device: str = "cpu",
        left_context: int = 72,
        chunk_frames: int = 25,
        dtype: str = "float32",
        ramp: list[int] | None = None,
    ):
        from safetensors.torch import load_file

        from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Config,
        )
        from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Decoder,
        )

        model_dir = self._resolve_dir(model)
        cfg = Qwen3TTSTokenizerV2Config.from_pretrained(str(model_dir), subfolder="speech_tokenizer")
        self.decoder = Qwen3TTSTokenizerV2Decoder._from_config(cfg.decoder_config)
        state = load_file(str(model_dir / "speech_tokenizer" / "model.safetensors"))
        dec_state = {k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")}
        missing, unexpected = self.decoder.load_state_dict(dec_state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected decoder keys: {unexpected[:5]}")
        if missing:
            print(f"[decoder] warning: {len(missing)} missing keys (first: {missing[:3]})", file=sys.stderr)
        self.decoder.eval().to(device=device, dtype=getattr(torch, dtype))
        if hasattr(self.decoder, "precompute_snake_caches"):
            self.decoder.precompute_snake_caches()
        # Mirror the Code2Wav stage's chunk-schedule attributes so the decoder's
        # prefix/suffix transitions match the server's (qwen3_tts_code2wav.py).
        ramp = list(ramp or [])
        self.decoder._initial_codec_chunk_frames = int(ramp[0]) if ramp else 1
        self.decoder._incremental_chunk_frames = chunk_frames
        self.decoder._incremental_chunk_ramp = ramp
        self.device = device
        self.num_quantizers = int(cfg.decoder_config.num_quantizers)
        self.sample_rate = int(cfg.output_sample_rate)
        self.left_context = left_context
        self.chunk_frames = chunk_frames
        self.reset()

    @staticmethod
    def _resolve_dir(model: str) -> Path:
        p = Path(model)
        if p.is_dir():
            return p
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model, local_files_only=True))

    def reset(self) -> None:
        self.state: dict[str, Any] = {"prefix_frames": 0}
        self.frames_decoded = 0

    @torch.inference_mode()
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode ``[n_frames, Q]`` new frames, carrying state across calls; returns float32 PCM."""
        if codes.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        qf = torch.as_tensor(codes.T, dtype=torch.long, device=self.device)  # [Q, F]
        wavs = self.decoder.batched_chunked_decode(
            qf.unsqueeze(0),  # [B=1, Q, F]; the decoder slices rows itself
            [int(qf.shape[-1])],
            caches=[self.state],
            chunk_size=self.chunk_frames,
            left_context_size=self.left_context,
            max_batch_size=1,
        )
        wav = wavs[0]
        if wav.dim() == 2 and wav.shape[0] == 1:
            wav = wav[0]
        self.frames_decoded += int(codes.shape[0])
        return wav.float().cpu().numpy().reshape(-1)


class ExecutorchWindowedDecoder:
    """Decoder backend for phone/NPU targets: exported windowed programs (see vllm_omni.edge.decoder_export).

    Keeps the last ``context_frames`` decoded frames as explicit state and runs
    the ``.pte`` program matching the chunk size (falls back to the closest
    larger program with zero-padded context when a size is missing). Untested
    in this checkout (executorch not installed); mirrors ``LocalDecoder``'s API.
    """

    def __init__(self, manifest_path: str):
        import json as _json

        from executorch.runtime import Runtime  # type: ignore[import-not-found]

        self.manifest = _json.loads(Path(manifest_path).read_text())
        self.num_quantizers = int(self.manifest["num_quantizers"])
        self.sample_rate = int(self.manifest["sample_rate"])
        self.hop = int(self.manifest["hop"])
        self.context = int(self.manifest["context_frames"])
        rt = Runtime.get()
        self.methods = {}
        for chunk, entry in self.manifest["programs"].items():
            program = rt.load_program(entry["path"])
            self.methods[int(chunk)] = program.load_method("forward")
        self.reset()

    def reset(self) -> None:
        self.history = np.zeros((0, self.num_quantizers), dtype=np.int64)
        self.frames_decoded = 0

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if codes.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        n = int(codes.shape[0])
        sizes = sorted(self.methods)
        size = next((s for s in sizes if s >= n), sizes[-1])
        window = np.concatenate([self.history, codes], axis=0)[-(self.context + size) :]
        pad = self.context + size - window.shape[0]
        if pad > 0:
            window = np.concatenate([np.zeros((pad, self.num_quantizers), dtype=np.int64), window], axis=0)
        inp = torch.as_tensor(window.T[None], dtype=torch.long)  # [1, Q, ctx+size]
        out = self.methods[size].execute([inp])[0]
        wav = out.reshape(-1).float().numpy()[-size * self.hop :]
        self.history = np.concatenate([self.history, codes], axis=0)[-self.context :]
        self.frames_decoded += n
        return wav[-n * self.hop :] if n < size else wav


def build_decoder(args: argparse.Namespace, steady: int, schedule: list[int] | None = None):
    backend = getattr(args, "decoder_backend", "torch")
    if backend == "executorch":
        if not getattr(args, "pte_manifest", None):
            raise SystemExit("--decoder-backend executorch needs --pte-manifest <export dir>/manifest.json")
        return ExecutorchWindowedDecoder(args.pte_manifest)
    return LocalDecoder(
        args.model, device=args.device, left_context=args.left_context, chunk_frames=steady, ramp=list(schedule)
    )


# ------------------------------------------------------------------- client
async def run(args: argparse.Namespace) -> dict[str, Any]:
    import websockets

    ramp = args.ramp or DEFAULT_RAMP
    schedule = parse_chunk_ramp({"codec_chunk_ramp": ramp}, steady=ramp[-1]) or ramp
    steady = schedule[-1]
    decoder = build_decoder(args, steady, list(schedule))
    url = f"ws://{args.host}:{args.port}/v1/audio/speech/stream"
    config: dict[str, Any] = {
        "type": "session.config",
        "model": args.model,
        "output_mode": "codec_tokens",
        "stream_audio": True,
        "response_format": "pcm",
    }
    for key in ("voice", "task_type", "language", "instructions", "max_new_tokens", "seed"):
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val
    pcm_parts: list[np.ndarray] = []
    codes_all: list[np.ndarray] = []
    chunk_log: list[dict[str, Any]] = []
    pending = np.zeros((0, decoder.num_quantizers), dtype=np.int64)
    chunk_idx = 0
    t0 = None
    t_first_pcm = None
    codec_start = None
    total_frames = 0
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps(config))
        await ws.send(json.dumps({"type": "input.text", "text": args.text}))
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "input.done"}))
        while True:
            raw = await ws.recv()
            now = time.perf_counter()
            if isinstance(raw, (bytes, bytearray)):
                seq, n, flags, arr = unpack_codec_frame(bytes(raw), decoder.num_quantizers)
                if flags & FLAG_RESET:
                    decoder.reset()
                    pending = pending[:0]
                total_frames += n
                if n:
                    pending = np.concatenate([pending, arr.astype(np.int64)], axis=0)
                    codes_all.append(arr.astype(np.int64))
                eos = bool(flags & FLAG_EOS)
                target = schedule[min(chunk_idx, len(schedule) - 1)]
                while pending.shape[0] >= target or (eos and pending.shape[0] > 0):
                    take = pending[: min(target, steady)] if not eos else pending[:steady]
                    pending = pending[take.shape[0] :]
                    td = time.perf_counter()
                    pcm = decoder.decode(take)
                    t_dec = time.perf_counter() - td
                    if t_first_pcm is None and pcm.size:
                        t_first_pcm = time.perf_counter()
                    pcm_parts.append(pcm)
                    chunk_log.append(
                        {
                            "idx": chunk_idx,
                            "t_ms": (time.perf_counter() - t0) * 1e3,
                            "arrival_ms": (now - t0) * 1e3,
                            "frames": int(take.shape[0]),
                            "samples": int(pcm.size),
                            "decode_ms": t_dec * 1e3,
                        }
                    )
                    chunk_idx += 1
                    target = schedule[min(chunk_idx, len(schedule) - 1)]
                if eos:
                    continue
            else:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "codec.start":
                    codec_start = msg
                elif mtype in ("codec.done", "audio.done"):
                    pass
                elif mtype == "session.done":
                    break
                elif mtype == "error":
                    raise RuntimeError(f"server error: {msg.get('message')}")
        await ws.send(json.dumps({"type": "session.close"}))
    import stream_latency_metrics as metrics

    sr = decoder.sample_rate
    audio = np.concatenate(pcm_parts) if pcm_parts else np.zeros(0, dtype=np.float32)
    total_s = (time.perf_counter() - t0) if t0 else 0.0
    result = {
        "codec_start": codec_start,
        "schedule": schedule,
        "total_frames": total_frames,
        "ttfa_ms": ((t_first_pcm - t0) * 1e3) if t_first_pcm else None,
        "playback_start_ms": metrics.playback_start_ms(chunk_log, sr),
        "stall_at_ttfa_ms": metrics.stall_at_ttfa_ms(chunk_log, sr),
        "audio_s": audio.size / sr,
        "total_s": total_s,
        "rtf": (total_s / (audio.size / sr)) if audio.size else None,
        "decode_ms_total": sum(c["decode_ms"] for c in chunk_log),
        "chunks": chunk_log,
        "device": args.device,
    }
    if args.out:
        import soundfile as sf

        sf.write(args.out, audio, sr, format="WAV")
        result["wav"] = args.out
        # Raw received codec rows (frames x codebooks) for offline parity checks.
        codes_path = str(Path(args.out).with_suffix(".codes.npy"))
        np.save(codes_path, np.concatenate(codes_all, axis=0) if codes_all else np.zeros((0, 0), dtype=np.int64))
        result["codes"] = codes_path
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "chunks"}, indent=1))
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--model", required=True, help="HF id or local dir (decoder weights are read from speech_tokenizer/)"
    )
    p.add_argument("--text", default="Hello, this is a codec token streaming test from the edge decoder client.")
    p.add_argument("--voice", default="vivian")
    p.add_argument("--task-type", dest="task_type", default="CustomVoice")
    p.add_argument("--language", default=None)
    p.add_argument("--instructions", default=None)
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--decoder-backend", dest="decoder_backend", choices=["torch", "executorch"], default="torch")
    p.add_argument("--pte-manifest", dest="pte_manifest", default=None, help="manifest.json from decoder_export")
    p.add_argument("--left-context", type=int, default=72)
    p.add_argument("--ramp", type=int, nargs="*", default=None, help="chunk schedule in frames, default 2 4 8 16 25")
    p.add_argument("--out", default=None)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
