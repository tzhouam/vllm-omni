#!/usr/bin/env python3
"""Check the existing incremental Code2Wav cache on retained generated codes.

This is a CPU decoder-state and chunk-timing experiment. The talker, Omni
stage, accelerator handoff and audio playback are not part of this run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import time
from pathlib import Path


REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def relative_l2(reference, observed) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "source-report", "codes", "report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")

    import numpy as np
    import torch

    from vllm_omni.edge.decoder_export import load_code2wav_decoder

    provenance = json.loads(args.source_report.read_text(encoding="utf-8-sig"))
    if (args.model.resolve().name != REVISION
            or provenance["model_revision"] != REVISION
            or sha256(args.codes) != provenance["files"]["codes"]["sha256"]
            or sha256(args.model / "speech_tokenizer" / "model.safetensors")
            != provenance["decoder_weight_sha256"]):
        raise ValueError("generated codes or decoder checkpoint changed")
    with np.load(args.codes, allow_pickle=False) as source:
        codes = np.ascontiguousarray(source["codes"])
    if codes.shape != (117, 16) or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("generated-code fixture shape or dtype changed")

    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    decoder = load_code2wav_decoder(str(args.model)).eval()
    load_s = time.perf_counter() - started
    tensor = torch.from_numpy(codes.T.copy()).unsqueeze(0).long()
    hop = int(decoder.total_upsample)
    if hop != 1920:
        raise ValueError("unexpected Code2Wav hop")
    with torch.inference_mode():
        started = time.perf_counter()
        reference = decoder(tensor).reshape(1, -1).detach().numpy()
        full_decode_s = time.perf_counter() - started
    if reference.shape != (1, 117 * hop) or not np.isfinite(reference).all():
        raise ValueError("full decoder output contract failed")

    outputs = {"full_reference": reference}
    schedules = {}
    decoder_module = importlib.import_module(type(decoder).__module__)
    original_downstream_context = int(decoder_module._DOWNSTREAM_CONTEXT_FRAME)
    cases = ((original_downstream_context, 2),
             (original_downstream_context, 25),
             (25, 25), (50, 25), (95, 25), (117, 25))
    for downstream_context, chunk_frames in cases:
        decoder_module._DOWNSTREAM_CONTEXT_FRAME = downstream_context
        cache = {"prefix_frames": 0}
        segments = []
        rows = []
        with torch.inference_mode():
            for start in range(0, codes.shape[0], chunk_frames):
                end = min(start + chunk_frames, codes.shape[0])
                started = time.perf_counter()
                wave = decoder(tensor[:, :, start:end], cache).reshape(1, -1).detach().numpy()
                elapsed = time.perf_counter() - started
                expected = reference[:, start * hop:end * hop]
                if wave.shape != expected.shape or not np.isfinite(wave).all():
                    raise ValueError(f"chunk {start}:{end} output contract failed")
                segments.append(wave)
                rows.append({
                    "start_frame": start,
                    "end_exclusive_frame": end,
                    "relative_l2_vs_full": relative_l2(expected, wave),
                    "max_abs_vs_full": float(np.max(np.abs(expected - wave))),
                    "wall_s": elapsed,
                    "suffix_frames_retained": int(cache["suffix_quantized"].shape[-1]),
                    "suffix_frames_seen": int(cache["suffix_frames"]),
                })
        combined = np.concatenate(segments, axis=-1)
        if combined.shape != reference.shape:
            raise ValueError("joined chunk length differs from full decoder")
        key = f"context_{downstream_context}_chunk_{chunk_frames}"
        outputs[key] = combined
        timings = [row["wall_s"] for row in rows[1:]]
        schedules[key] = {
            "downstream_context_frames": downstream_context,
            "chunk_frames": chunk_frames,
            "chunk_count": len(rows),
            "joined_relative_l2_vs_full": relative_l2(reference, combined),
            "joined_max_abs_vs_full": float(np.max(np.abs(reference - combined))),
            "max_chunk_relative_l2_vs_full": max(row["relative_l2_vs_full"] for row in rows),
            "post_first_chunk_wall_p50_s": percentile(timings, .50),
            "post_first_chunk_wall_p95_s": percentile(timings, .95),
            "rows": rows,
        }
    decoder_module._DOWNSTREAM_CONTEXT_FRAME = original_downstream_context

    from transformers.cache_utils import DynamicCache

    for downstream_context, chunk_frames in ((12, 25), (25, 25), (72, 25), (12, 2)):
        transformer_cache = DynamicCache(config=decoder.config)
        previous_quantized = None
        previous_transformer_hidden = None
        segments = []
        rows = []
        with torch.inference_mode():
            for start in range(0, codes.shape[0], chunk_frames):
                end = min(start + chunk_frames, codes.shape[0])
                started = time.perf_counter()
                quantized = decoder.quantizer.decode(tensor[:, :, start:end])
                if previous_quantized is None:
                    conv = decoder.pre_conv(quantized).transpose(1, 2)
                else:
                    conv = decoder.pre_conv(torch.cat(
                        [previous_quantized[:, :, -2:], quantized], dim=-1))
                    conv = conv[:, :, -(end - start):].transpose(1, 2)
                transformer_hidden = decoder.pre_transformer(
                    inputs_embeds=conv, past_key_values=transformer_cache,
                    use_cache=True).last_hidden_state
                hidden = (transformer_hidden if previous_transformer_hidden is None
                          else torch.cat([previous_transformer_hidden, transformer_hidden], dim=1))
                previous_transformer_hidden = hidden[:, -downstream_context:, :]
                previous_quantized = quantized
                hidden = hidden.permute(0, 2, 1)
                for blocks in decoder.upsample:
                    for block in blocks:
                        hidden = block(hidden)
                wave = hidden
                for block in decoder.decoder:
                    wave = block(wave)
                wave = wave.clamp(-1, 1).reshape(1, -1)[:, -(end - start) * hop:]
                observed = wave.detach().numpy()
                elapsed = time.perf_counter() - started
                expected = reference[:, start * hop:end * hop]
                if observed.shape != expected.shape or not np.isfinite(observed).all():
                    raise ValueError(f"KV-append chunk {start}:{end} output contract failed")
                segments.append(observed)
                rows.append({
                    "start_frame": start,
                    "end_exclusive_frame": end,
                    "relative_l2_vs_full": relative_l2(expected, observed),
                    "max_abs_vs_full": float(np.max(np.abs(expected - observed))),
                    "wall_s": elapsed,
                    "transformer_cache_seq_length": int(transformer_cache.get_seq_length()),
                    "downstream_hidden_retained": int(previous_transformer_hidden.shape[1]),
                })
        combined = np.concatenate(segments, axis=-1)
        key = f"kv_append_context_{downstream_context}_chunk_{chunk_frames}"
        outputs[key] = combined
        timings = [row["wall_s"] for row in rows[1:]]
        schedules[key] = {
            "downstream_context_frames": downstream_context,
            "chunk_frames": chunk_frames,
            "chunk_count": len(rows),
            "joined_relative_l2_vs_full": relative_l2(reference, combined),
            "joined_max_abs_vs_full": float(np.max(np.abs(reference - combined))),
            "max_chunk_relative_l2_vs_full": max(row["relative_l2_vs_full"] for row in rows),
            "post_first_chunk_wall_p50_s": percentile(timings, .50),
            "post_first_chunk_wall_p95_s": percentile(timings, .95),
            "rows": rows,
        }

    for chunk_frames in (25, 2):
        cache = {"prefix_frames": 0, "exact_xvec_kv": True}
        segments = []
        rows = []
        with torch.inference_mode():
            for start in range(0, codes.shape[0], chunk_frames):
                end = min(start + chunk_frames, codes.shape[0])
                started = time.perf_counter()
                [wave] = decoder.batched_chunked_decode(
                    tensor[:, :, start:end], [end - start], caches=[cache],
                    chunk_size=300, left_context_size=25,
                )
                observed = wave.reshape(1, -1).detach().numpy()
                elapsed = time.perf_counter() - started
                expected = reference[:, start * hop:end * hop]
                if observed.shape != expected.shape or not np.isfinite(observed).all():
                    raise ValueError(f"integrated KV chunk {start}:{end} output contract failed")
                segments.append(observed)
                kv = cache["exact_xvec_transformer_cache"]
                rows.append({
                    "start_frame": start,
                    "end_exclusive_frame": end,
                    "relative_l2_vs_full": relative_l2(expected, observed),
                    "max_abs_vs_full": float(np.max(np.abs(expected - observed))),
                    "wall_s": elapsed,
                    "transformer_cache_total_seen": int(kv.get_seq_length()),
                    "transformer_cache_physical_key_frames": int(kv.layers[0].keys.shape[-2]),
                    "downstream_hidden_retained": int(cache["exact_xvec_hidden_tail"].shape[1]),
                })
        combined = np.concatenate(segments, axis=-1)
        key = f"integrated_exact_kv_chunk_{chunk_frames}"
        outputs[key] = combined
        timings = [row["wall_s"] for row in rows[1:]]
        schedules[key] = {
            "chunk_frames": chunk_frames,
            "chunk_count": len(rows),
            "joined_relative_l2_vs_full": relative_l2(reference, combined),
            "joined_max_abs_vs_full": float(np.max(np.abs(reference - combined))),
            "max_chunk_relative_l2_vs_full": max(row["relative_l2_vs_full"] for row in rows),
            "post_first_chunk_wall_p50_s": percentile(timings, .50),
            "post_first_chunk_wall_p95_s": percentile(timings, .95),
            "rows": rows,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **outputs)
    report = {
        "scope": "one generated 117-frame Qwen3-TTS Code2Wav stream, CPU incremental-cache numerical and one-pass chunk timing; no talker or device handoff",
        "checkpoint_revision": REVISION,
        "decoder_weight_sha256": provenance["decoder_weight_sha256"],
        "codes_sha256": sha256(args.codes),
        "decoder_source": "vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2",
        "frames": int(codes.shape[0]),
        "hop_samples": hop,
        "decoder_sliding_window_frames": int(decoder.config.sliding_window),
        "source_downstream_context_frames": original_downstream_context,
        "threads": args.threads,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "model_load_s": load_s,
        "full_decode_s": full_decode_s,
        "output_sha256": sha256(args.output),
        "schedules": schedules,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"schedules": {
        key: {name: value for name, value in schedule.items() if name != "rows"}
        for key, schedule in schedules.items()}}, indent=2))


if __name__ == "__main__":
    main()
