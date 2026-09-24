#!/usr/bin/env python3
"""Export and numerically gate a fixed Qwen3-TTS Code2Wav window.

The retained reference graph takes 72 context frames plus 25 new frames.
This experiment keeps the same decoder while selecting 1-25 new frames and
their explicit history, before any device compilation or performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from vllm_omni.edge.decoder_export import load_code2wav_decoder


class QuantizedWindowDecoder(nn.Module):
    """The existing decoder after RVQ lookup, with an explicit context window."""

    def __init__(self, decoder: nn.Module, chunk_frames: int) -> None:
        super().__init__()
        self.decoder = decoder
        self.chunk_frames = chunk_frames
        self.hop = int(decoder.total_upsample)

    def forward(self, quantized: torch.Tensor) -> torch.Tensor:
        decoder = self.decoder
        hidden = decoder.pre_conv(quantized).transpose(1, 2)
        hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        for block in decoder.decoder:
            hidden = block(hidden)
        return hidden.clamp(-1, 1).reshape(1, -1)[:, -self.chunk_frames * self.hop :]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(reference: np.ndarray, output: np.ndarray) -> dict:
    a = reference.astype(np.float64).ravel()
    b = output.astype(np.float64).ravel()
    error = float(np.linalg.norm(a - b))
    norm = float(np.linalg.norm(a))
    return {
        "relative_l2": error / norm,
        "snr_db": 20 * math.log10(norm / error) if error else None,
        "max_abs": float(np.max(np.abs(a - b))),
        "reference_rms": float(np.sqrt(np.mean(a * a))),
        "output_rms": float(np.sqrt(np.mean(b * b))),
    }


def strip_export_guards(path: Path) -> None:
    graph = onnx.load(str(path))
    nodes = graph.graph.node
    producers = {out: node for node in nodes for out in node.output}
    rename: dict[str, str] = {}
    for node in list(nodes):
        if node.op_type != "Where" or node.input[0] not in producers:
            continue
        guard = producers[node.input[0]]
        if guard.op_type != "IsNaN":
            continue
        source = guard.input[0]
        if source not in node.input[1:]:
            continue
        rename[node.output[0]] = source
        nodes.remove(node)
        nodes.remove(guard)
    for node in nodes:
        for index, value in enumerate(node.input):
            if value in rename:
                node.input[index] = rename[value]
        if node.op_type == "Reshape":
            for attribute in node.attribute:
                if attribute.name == "allowzero":
                    attribute.i = 0
    for value in graph.graph.output:
        if value.name in rename:
            value.name = rename[value.name]
    onnx.save(graph, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "fixture", "long-reference", "onnx-output", "fixture-output",
                 "eager-output", "ort-output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=2)
    parser.add_argument("--context-frames", type=int, default=72)
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Offset within the retained 25 generated frames")
    args = parser.parse_args()
    if (not 1 <= args.chunk_frames <= 25
            or not 0 <= args.start_frame <= 25 - args.chunk_frames
            or not 1 <= args.context_frames <= 72 + args.start_frame):
        raise ValueError("history/new-frame window exceeds the retained 72+25 frames")
    with np.load(args.fixture, allow_pickle=False) as source:
        if source.files != ["quantized"]:
            raise ValueError("unexpected retained fixture")
        full_input = np.asarray(source["quantized"])
    if full_input.shape != (1, 512, 97) or full_input.dtype != np.float32:
        raise ValueError("retained vocoder fixture differs from expected shape/dtype")
    window_end = 72 + args.start_frame + args.chunk_frames
    window_start = 72 + args.start_frame - args.context_frames
    short_input = np.ascontiguousarray(full_input[:, :, window_start:window_end])
    if not np.isfinite(short_input).all():
        raise ValueError("fixture contains nonfinite values")

    decoder = load_code2wav_decoder(str(args.model))
    model = QuantizedWindowDecoder(decoder, args.chunk_frames).eval()
    with torch.inference_mode():
        eager = model(torch.from_numpy(short_input)).detach().numpy()
    expected_shape = (1, args.chunk_frames * model.hop)
    if eager.shape != expected_shape or not np.isfinite(eager).all():
        raise ValueError("eager short-window output violates contract")
    args.onnx_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    program = torch.onnx.export(
        model, (torch.from_numpy(short_input),),
        input_names=["quantized"], output_names=["wav"],
        opset_version=17, dynamo=True, optimize=True,
    )
    program.save(str(args.onnx_output))
    strip_export_guards(args.onnx_output)
    export_s = time.perf_counter() - started
    exported_model = onnx.load(str(args.onnx_output), load_external_data=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.onnx_output), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    observed = np.asarray(session.run(["wav"], {"quantized": short_input})[0])
    if observed.shape != expected_shape or not np.isfinite(observed).all():
        raise ValueError("ORT short-window output violates contract")
    with np.load(args.long_reference, allow_pickle=False) as reference:
        if reference.files != ["wav"] or reference["wav"].shape != (1, 48000):
            raise ValueError("long-window reference differs from retained CPU output")
        long_segment = np.asarray(reference["wav"][
            :, args.start_frame * model.hop:
            args.start_frame * model.hop + expected_shape[1]
        ])
    for path, key, value in (
        (args.fixture_output, "quantized", short_input),
        (args.eager_output, "wav", eager),
        (args.ort_output, "wav", observed),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{key: value})
    report = {
        "scope": "one fixed short vocoder window; no device execution or complete TTS stream",
        "model_snapshot": str(args.model.resolve()),
        "model_snapshot_revision": args.model.resolve().name,
        "model_weight_sha256": sha256(args.model / "speech_tokenizer" / "model.safetensors"),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "actual_opsets": {entry.domain or "ai.onnx": entry.version
                          for entry in exported_model.opset_import},
        "source_fixture_sha256": sha256(args.fixture),
        "long_cpu_reference_sha256": sha256(args.long_reference),
        "context_frames": args.context_frames,
        "chunk_frames": args.chunk_frames,
        "start_frame": args.start_frame,
        "window_start_frame": window_start,
        "window_end_exclusive_frame": window_end,
        "input_shape": list(short_input.shape),
        "output_shape": list(expected_shape),
        "export_s_not_inference": export_s,
        "onnx_sha256": sha256(args.onnx_output),
        "short_fixture_sha256": sha256(args.fixture_output),
        "eager_output_sha256": sha256(args.eager_output),
        "ort_output_sha256": sha256(args.ort_output),
        "ort_vs_eager": compare(eager, observed),
        "short_eager_vs_25_frame_segment": compare(long_segment, eager),
        "limits": [
            "The short-window graph still requires its prior quantized frames from a model adapter.",
            "Segment agreement is tested on one retained fixture, not a speech-quality set.",
            "CPU export parity does not establish target compilation, placement or memory admission.",
        ],
    }
    if args.start_frame == 0:
        report["short_eager_vs_25_frame_prefix"] = report["short_eager_vs_25_frame_segment"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ort_vs_eager": report["ort_vs_eager"],
                      "short_vs_long": report["short_eager_vs_25_frame_segment"]}, indent=2))


if __name__ == "__main__":
    main()
