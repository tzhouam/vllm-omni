#!/usr/bin/env python3
"""Capture one generated CustomVoice code stream and audit a 25-frame export.

Set PYTHONPATH to include the pinned qwen_tts source checkout when invoking.
The hosted device is not used here; this checks a real-code local window and
its truncation against the full FP32 decoder before a device quality attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)
from safetensors.torch import load_file


def load_decoder(model: Path) -> Qwen3TTSTokenizerV2Decoder:
    config = Qwen3TTSTokenizerV2Config.from_pretrained(str(model), subfolder="speech_tokenizer")
    decoder = Qwen3TTSTokenizerV2Decoder._from_config(config.decoder_config)
    state = load_file(str(model / "speech_tokenizer" / "model.safetensors"))
    decoder_state = {key[len("decoder."):]: value for key, value in state.items()
                     if key.startswith("decoder.")}
    missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"decoder state mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
    decoder.eval().float()
    if hasattr(decoder, "precompute_snake_caches"):
        decoder.precompute_snake_caches()
    return decoder


def decode_quantized(decoder: Qwen3TTSTokenizerV2Decoder, quantized: torch.Tensor) -> torch.Tensor:
    hidden = decoder.pre_conv(quantized).transpose(1, 2)
    hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
    hidden = hidden.permute(0, 2, 1)
    for blocks in decoder.upsample:
        for block in blocks:
            hidden = block(hidden)
    for block in decoder.decoder:
        hidden = block(hidden)
    return hidden.clamp(-1, 1).reshape(1, -1)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(reference: np.ndarray, observed: np.ndarray) -> dict:
    a = reference.astype(np.float64).ravel()
    b = observed.astype(np.float64).ravel()
    error = float(np.linalg.norm(a - b))
    norm = float(np.linalg.norm(a))
    return {
        "relative_l2": error / norm,
        "snr_db": 20 * math.log10(norm / error) if error else None,
        "max_abs": float(np.max(np.abs(a - b))),
        "reference_rms": float(np.sqrt(np.mean(a * a))),
        "observed_rms": float(np.sqrt(np.mean(b * b))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text", default=(
        "The small red square is centered on the white card. I can see its "
        "straight edges clearly, and the surrounding background is bright "
        "and uncluttered."
    ))
    parser.add_argument("--max-new-tokens", type=int, default=144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-generated-frames", type=int, default=97)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if (args.max_new_tokens < 97 or args.min_generated_frames < 97
            or args.threads <= 0):
        parser.error("at least 97 max new tokens/frames and positive threads required")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(args.threads)

    wrapper = Qwen3TTSModel.from_pretrained(
        str(args.model.resolve()), device_map="cpu", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    captured: list[torch.Tensor] = []
    original_decode = wrapper.model.speech_tokenizer.decode

    def capture(items: list[dict], *extra: object, **kwargs: object):
        captured.extend(item["audio_codes"].detach().cpu().clone() for item in items)
        return original_decode(items, *extra, **kwargs)

    wrapper.model.speech_tokenizer.decode = capture
    torch.manual_seed(args.seed)
    wavs, sample_rate = wrapper.generate_custom_voice(
        text=args.text, language="English", speaker="Ryan",
        max_new_tokens=args.max_new_tokens, do_sample=True,
        non_streaming_mode=True,
    )
    if len(captured) != 1 or len(wavs) != 1 or sample_rate != 24000:
        raise RuntimeError("expected one generated code stream and 24 kHz waveform")
    codes = captured[0]
    if (codes.ndim != 2 or codes.shape[0] < args.min_generated_frames
            or codes.shape[1] != 16):
        raise RuntimeError(
            f"expected >={args.min_generated_frames} frames of 16-code speech, "
            f"got {tuple(codes.shape)}")
    generated = np.asarray(wavs[0], dtype=np.float32).ravel()
    if len(generated) != codes.shape[0] * 1920 or not np.isfinite(generated).all():
        raise RuntimeError("complete request has invalid audio length or samples")

    del wrapper
    decoder = load_decoder(args.model)
    window_codes = codes[:97].T.unsqueeze(0).contiguous().long()
    full_codes = codes.T.unsqueeze(0).contiguous().long()
    with torch.inference_mode():
        quantized = decoder.quantizer.decode(window_codes).detach().float().numpy()
        eager = decode_quantized(decoder, torch.from_numpy(quantized))[:, -25 * 1920:].detach().numpy()
        full = decoder(full_codes).reshape(1, -1).detach().numpy()
    if quantized.shape != (1, 512, 97) or eager.shape != (1, 48000):
        raise RuntimeError("real-code window violates fixed export shape")
    full_segment = full[:, 72 * 1920 : 97 * 1920]
    if full_segment.shape != eager.shape:
        raise RuntimeError("full decoder output does not align with window")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    observed = np.asarray(session.run(["wav"], {"quantized": quantized})[0])
    if observed.shape != eager.shape or not np.isfinite(observed).all():
        raise RuntimeError("ONNX output violates waveform contract")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "codes": ("codes.npz", {"codes": codes.numpy()}),
        "fixture": ("fixture.npz", {"quantized": quantized}),
        "eager": ("eager_output.npz", {"wav": eager}),
        "ort": ("ort_output.npz", {"wav": observed}),
        "full_reference": ("full_reference_segment.npz", {"wav": full_segment}),
    }
    for name, (filename, arrays) in files.items():
        np.savez_compressed(args.output_dir / filename, **arrays)
    report = {
        "scope": "one real generated-code 25-frame Code2Wav window on local CPU; no device execution",
        "model_revision": args.model.resolve().name,
        "model_weight_sha256": sha256(args.model / "model.safetensors"),
        "decoder_weight_sha256": sha256(args.model / "speech_tokenizer" / "model.safetensors"),
        "onnx_sha256": sha256(args.onnx),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "qwen_tts_sources": {
            "wrapper": {"path": inspect.getfile(Qwen3TTSModel),
                        "sha256": sha256(Path(inspect.getfile(Qwen3TTSModel)))},
            "decoder": {"path": inspect.getfile(Qwen3TTSTokenizerV2Decoder),
                        "sha256": sha256(Path(inspect.getfile(Qwen3TTSTokenizerV2Decoder)))},
        },
        "text": args.text,
        "speaker": "Ryan",
        "language": "English",
        "seed": args.seed,
        "generator_dtype": "bfloat16",
        "decoder_dtype": "float32",
        "threads": args.threads,
        "generated_frames": int(codes.shape[0]),
        "generated_audio_samples": len(generated),
        "window": {"start_frame": 0, "history_frames": 72, "new_frames": 25},
        "files": {name: {"filename": filename, "sha256": sha256(args.output_dir / filename)}
                  for name, (filename, _) in files.items()},
        "ort_vs_window_eager": compare(eager, observed),
        "window_eager_vs_full_fp32_decoder_segment": compare(full_segment, eager),
        "limits": [
            "One generated stream and one window do not establish speech quality.",
            "The model generates codes on local CPU; this does not test a mobile talker or handoff.",
            "The FP32 full decoder reference is separate from the BF16 generator's waveform.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"frames": report["generated_frames"],
                      "ort_vs_eager": report["ort_vs_window_eager"],
                      "window_vs_full": report["window_eager_vs_full_fp32_decoder_segment"]}, indent=2))


if __name__ == "__main__":
    main()
