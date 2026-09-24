#!/usr/bin/env python3
"""Measure later two-frame Code2Wav output against retained long-history audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"
FIXTURE_SHA = "4220eec4ee00e5fbb0df6f195d883c6d4549accf399853c3a8e2f83821724452"
SOURCE_ONNX_SHA = "01ee1cd02e5cd264200e5c50767a1805a6180378b051ff9b7ec0d8b6c75d7989"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def relative_l2(reference, observed) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "real-report", "fixture", "long-reference", "report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=23)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.start_frame <= 23 or args.threads < 1:
        parser.error("start-frame must be 0..23 and threads must be positive")

    import numpy as np
    import torch

    from benchmarks.edge_harness.experiments.export_qwen_tts_vocoder_short_chunk import (
        QuantizedWindowDecoder,
    )
    from vllm_omni.edge.decoder_export import load_code2wav_decoder

    provenance = json.loads(args.real_report.read_text(encoding="utf-8-sig"))
    if (args.model.resolve().name != REVISION
            or provenance["model_revision"] != REVISION
            or provenance["onnx_sha256"] != SOURCE_ONNX_SHA
            or provenance["files"]["fixture"]["sha256"] != FIXTURE_SHA
            or provenance["files"]["ort"]["sha256"] != sha256(args.long_reference)
            or sha256(args.fixture) != FIXTURE_SHA
            or provenance["decoder_weight_sha256"]
            != sha256(args.model / "speech_tokenizer" / "model.safetensors")):
        raise ValueError("checkpoint, source graph, generated codes or reference changed")
    with np.load(args.fixture, allow_pickle=False) as source:
        quantized = np.ascontiguousarray(source["quantized"])
    with np.load(args.long_reference, allow_pickle=False) as source:
        long_wave = np.ascontiguousarray(source["wav"])
    if (quantized.shape != (1, 512, 97) or quantized.dtype != np.float32
            or long_wave.shape != (1, 48000)
            or not np.isfinite(quantized).all()
            or not np.isfinite(long_wave).all()):
        raise ValueError("retained generated-code fixture contract changed")

    torch.set_num_threads(args.threads)
    model = QuantizedWindowDecoder(load_code2wav_decoder(str(args.model)), 2).eval()
    hop = model.hop
    assert hop == 1920
    reference = long_wave[:, args.start_frame * hop:(args.start_frame + 2) * hop]
    histories = [72, 80, 88, 95] if args.start_frame == 23 else [72]
    rows = []
    outputs = {"long_reference": reference}
    for history in histories:
        start = 72 + args.start_frame - history
        end = 72 + args.start_frame + 2
        window = np.ascontiguousarray(quantized[:, :, start:end])
        started = time.perf_counter()
        with torch.inference_mode():
            actual = model(torch.from_numpy(window)).detach().numpy()
        elapsed = time.perf_counter() - started
        if actual.shape != reference.shape or not np.isfinite(actual).all():
            raise ValueError("two-frame output contract failed")
        key = f"history_{history}"
        outputs[key] = actual
        rows.append({
            "history_frames": history,
            "window_start_frame": start,
            "window_end_exclusive_frame": end,
            "input_shape": list(window.shape),
            "waveform_relative_l2_vs_long_25_frame_cpu": relative_l2(reference, actual),
            "waveform_max_abs_vs_long_25_frame_cpu": float(np.max(np.abs(reference - actual))),
            "one_eager_forward_s_not_profile": elapsed,
        })
    if args.start_frame == 23 and rows[-1]["waveform_relative_l2_vs_long_25_frame_cpu"] > 1e-5:
        raise ValueError("full 95-frame history did not recover the long CPU result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **outputs)
    report = {
        "scope": "one generated-code later two-frame CPU history sweep; no NPU, stream or performance profile",
        "checkpoint_revision": REVISION,
        "decoder_weight_sha256": provenance["decoder_weight_sha256"],
        "source_25_frame_onnx_sha256": SOURCE_ONNX_SHA,
        "fixture_sha256": FIXTURE_SHA,
        "long_reference_sha256": sha256(args.long_reference),
        "start_frame": args.start_frame,
        "threads": args.threads,
        "torch_version": torch.__version__,
        "outputs_sha256": sha256(args.output),
        "rows": rows,
        "status": "cpu_history_sweep_pass",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
