#!/usr/bin/env python3
"""Check whether shorter Code2Wav history preserves a pinned two-frame output.

This is a one-fixture numerical screen for smaller target graphs, not an audio
quality or device-performance result.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-fixture", type=Path, required=True)
    parser.add_argument("--long-reference", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    export = json.loads(args.export_report.read_text())
    if (sha256(args.source_fixture) != export["source_fixture_sha256"]
            or sha256(args.long_reference) != export["long_cpu_reference_sha256"]
            or sha256(args.model / "speech_tokenizer" / "model.safetensors")
            != export["model_weight_sha256"]):
        raise ValueError("fixture, reference or checkpoint differs from the export")
    with np.load(args.source_fixture, allow_pickle=False) as data:
        if data.files != ["quantized"]:
            raise ValueError("unexpected fixture schema")
        full_input = np.asarray(data["quantized"])
    with np.load(args.long_reference, allow_pickle=False) as data:
        if data.files != ["wav"]:
            raise ValueError("unexpected reference schema")
        long_output = np.asarray(data["wav"])
    if (full_input.shape != (1, 512, 97) or full_input.dtype != np.float32
            or long_output.shape != (1, 48000)):
        raise ValueError("reference tensor shape changed")

    torch.set_num_threads(4)
    model = QuantizedWindowDecoder(load_code2wav_decoder(str(args.model)), 2).eval()
    reference = long_output[:, :3840]
    rows = []
    baseline = None
    for history in (72, 64, 48, 32, 24, 16, 8):
        window = np.ascontiguousarray(full_input[:, :, 72 - history:74])
        with torch.inference_mode():
            output = model(torch.from_numpy(window)).detach().numpy()
        if output.shape != (1, 3840) or not np.isfinite(output).all():
            raise ValueError(f"{history}-frame history violated waveform contract")
        if baseline is None:
            baseline = output
        rows.append({
            "history_frames": history,
            "input_shape": list(window.shape),
            "vs_72_history": compare(baseline, output),
            "vs_25_frame_reference_prefix": compare(reference, output),
        })
    report = {
        "scope": "one synthetic two-frame Code2Wav window; local CPU numerical screen only",
        "checkpoint_revision": export["model_snapshot_revision"],
        "checkpoint_weight_sha256": export["model_weight_sha256"],
        "source_fixture_sha256": export["source_fixture_sha256"],
        "long_reference_sha256": export["long_cpu_reference_sha256"],
        "torch_version": torch.__version__,
        "rows": rows,
        "limits": [
            "One synthetic first window cannot establish multi-chunk history or speech quality.",
            "No smaller-history graph was exported, compiled or executed on a device.",
            "CPU forward timing was not measured as a performance profile.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{"history_frames": x["history_frames"],
                       "relative_l2": x["vs_72_history"]["relative_l2"]}
                      for x in rows], indent=2))


if __name__ == "__main__":
    main()
