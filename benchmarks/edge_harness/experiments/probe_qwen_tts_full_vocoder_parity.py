#!/usr/bin/env python3
"""Compare a pinned eager 25-frame Code2Wav decoder with retained ONNX CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.edge_harness.experiments.export_qwen_tts_vocoder_short_chunk import (
    QuantizedWindowDecoder,
    compare,
    sha256,
)
from vllm_omni.edge.decoder_export import load_code2wav_decoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "source-onnx", "fixture", "ort-output", "manifest", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for key, path in (("vocoder_source_onnx", args.source_onnx),
                      ("vocoder_fixture", args.fixture),
                      ("vocoder_ort_cpu_output", args.ort_output)):
        if sha256(path) != manifest["files"][key]["sha256"]:
            raise ValueError(f"{key} differs from the retained historical artifact")
    with np.load(args.fixture, allow_pickle=False) as data:
        if data.files != ["quantized"]:
            raise ValueError("unexpected fixture schema")
        quantized = np.asarray(data["quantized"])
    with np.load(args.ort_output, allow_pickle=False) as data:
        if data.files != ["wav"]:
            raise ValueError("unexpected output schema")
        reference = np.asarray(data["wav"])
    if (quantized.shape != (1, 512, 97) or quantized.dtype != np.float32
            or reference.shape != (1, 48000) or reference.dtype != np.float32):
        raise ValueError("historical tensor contract changed")
    torch.set_num_threads(4)
    decoder = QuantizedWindowDecoder(load_code2wav_decoder(str(args.checkpoint)), 25).eval()
    with torch.inference_mode():
        eager = decoder(torch.from_numpy(quantized)).detach().numpy()
    if eager.shape != reference.shape or not np.isfinite(eager).all():
        raise ValueError("eager output violates finite 48,000-sample contract")
    report = {
        "scope": "one synthetic 25-frame Code2Wav waveform parity check; no device stream",
        "pinned_checkpoint_revision": args.checkpoint.resolve().name,
        "pinned_decoder_weight_sha256": sha256(args.checkpoint / "speech_tokenizer" / "model.safetensors"),
        "source_onnx_sha256": sha256(args.source_onnx),
        "fixture_sha256": sha256(args.fixture),
        "ort_output_sha256": sha256(args.ort_output),
        "eager_output_bytes_sha256": hashlib.sha256(eager.tobytes()).hexdigest(),
        "torch_version": torch.__version__,
        "eager_vs_historical_onnx_cpu": compare(reference, eager),
        "limits": [
            "Numerical agreement on one fixture cannot prove the historical ONNX export's exact checkpoint revision.",
            "One synthetic latent window cannot establish listening quality or multi-chunk history.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["eager_vs_historical_onnx_cpu"], indent=2))


if __name__ == "__main__":
    main()
