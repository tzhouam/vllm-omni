#!/usr/bin/env python3
"""Compare real-weight CUDA Code2Wav chunk state with a full decoder call."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.edge.decoder_export import load_code2wav_decoder


REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, observed: np.ndarray) -> float:
    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    decoder = load_code2wav_decoder(str(args.model)).eval().to("cuda")
    if hasattr(decoder, "precompute_snake_caches"):
        decoder.precompute_snake_caches()
    tensor = torch.from_numpy(codes.T.copy()).unsqueeze(0).long().to("cuda")
    hop = int(decoder.total_upsample)
    torch.cuda.synchronize()
    with torch.inference_mode():
        started = time.perf_counter()
        reference_gpu = decoder(tensor).reshape(1, -1)
        torch.cuda.synchronize()
        full_wall_s = time.perf_counter() - started
        reference = reference_gpu.cpu().numpy()
        if reference.shape != (1, 117 * hop) or not np.isfinite(reference).all():
            raise ValueError("full decoder output contract failed")
        schedules = {}
        for exact in (False, True):
            for chunk_frames in (25, 2):
                cache = {"prefix_frames": 0}
                if exact:
                    cache["exact_xvec_kv"] = True
                waves = []
                rows = []
                torch.cuda.reset_peak_memory_stats()
                for start in range(0, 117, chunk_frames):
                    end = min(start + chunk_frames, 117)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    [wave] = decoder.batched_chunked_decode(
                        tensor[:, :, start:end], [end - start], caches=[cache],
                        chunk_size=300, left_context_size=25,
                    )
                    torch.cuda.synchronize()
                    wall_s = time.perf_counter() - started
                    observed = wave.reshape(1, -1).cpu().numpy()
                    target = reference[:, start * hop:end * hop]
                    if observed.shape != target.shape or not np.isfinite(observed).all():
                        raise ValueError(f"invalid CUDA chunk {start}:{end}")
                    waves.append(observed)
                    rows.append({
                        "start_frame": start,
                        "end_exclusive_frame": end,
                        "relative_l2_vs_full": relative_l2(target, observed),
                        "wall_s": wall_s,
                    })
                combined = np.concatenate(waves, axis=-1)
                timings = [row["wall_s"] for row in rows[1:]]
                key = f"{'exact' if exact else 'legacy'}_chunk_{chunk_frames}"
                schedules[key] = {
                    "joined_relative_l2_vs_full": relative_l2(reference, combined),
                    "joined_max_abs_vs_full": float(np.max(np.abs(reference - combined))),
                    "chunk_count": len(rows),
                    "post_first_chunk_wall_p50_s": percentile(timings, .50),
                    "post_first_chunk_wall_p95_s": percentile(timings, .95),
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "physical_kv_frames_at_end": (
                        int(cache["exact_xvec_transformer_cache"].layers[0].keys.shape[-2])
                        if exact else None
                    ),
                    "rows": rows,
                }
        # Exercise the production segmented wrapper's stateful dispatcher
        # without capturing graphs. Exact requests must reach its eager
        # fallback, which previously reconstructed the truncated suffix.
        from vllm_omni.model_executor.models.qwen3_tts.segmented_graph_wrapper import (
            CUDAGraphDecoderWrapper,
        )

        wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
        wrapper.decoder = decoder
        wrapper.async_chunk = True
        wrapper.prefix_length = int(decoder.config.sliding_window)
        wrapper.initial_chunk_frames = 1
        wrapper.codec_chunk_frames = 25
        wrapper._icl_previous_frames_by_target = {}
        wrapper._xvec_previous_frames_by_target = {}
        wrapper._decode_icl_prefix_batch = lambda *_args: None
        wrapper._decode_xvec_prefix_batch = lambda *_args: None
        wrapper._decode_suffix_batch = lambda *_args: None
        wrapper._suppress_stats = False
        cache = {"prefix_frames": 0, "exact_xvec_kv": True}
        wrapper_waves = []
        for start in range(0, 117, 25):
            end = min(start + 25, 117)
            [wave] = wrapper.batched_chunked_decode_with_cudagraph(
                tensor[:, :, start:end], [end - start], caches=[cache],
            )
            wrapper_waves.append(wave.reshape(1, -1).cpu().numpy())
        wrapper_joined = np.concatenate(wrapper_waves, axis=-1)
        schedules["segmented_wrapper_eager_exact_chunk_25"] = {
            "joined_relative_l2_vs_full": relative_l2(reference, wrapper_joined),
            "joined_max_abs_vs_full": float(np.max(np.abs(reference - wrapper_joined))),
            "chunk_count": len(wrapper_waves),
            "physical_kv_frames_at_end": int(
                cache["exact_xvec_transformer_cache"].layers[0].keys.shape[-2]
            ),
            "captured_graph": False,
        }
    report = {
        "scope": "one pinned generated-code Code2Wav stream; decoder only, no talker/Omni/playback",
        "checkpoint_revision": REVISION,
        "decoder_weight_sha256": provenance["decoder_weight_sha256"],
        "codes_sha256": sha256(args.codes),
        "device": torch.cuda.get_device_name(),
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "nvidia_driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": args.threads,
        "tf32": False,
        "precision": "FP32 decoder weights and activations",
        "backend": "PyTorch CUDA eager, segmented wrapper stateful eager fallback; no graph capture",
        "concurrency": 1,
        "warmup": "no explicit warmup; first chunk excluded from reported chunk percentiles",
        "power_condition": "uncontrolled laptop power and ambient load; no sustained power samples",
        "frames": 117,
        "code_shape": list(tensor.shape),
        "hop_samples": hop,
        "full_decode_wall_s": full_wall_s,
        "schedules": schedules,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {name: value for name, value in item.items() if name != "rows"}
                      for key, item in schedules.items()}, indent=2))


if __name__ == "__main__":
    main()
