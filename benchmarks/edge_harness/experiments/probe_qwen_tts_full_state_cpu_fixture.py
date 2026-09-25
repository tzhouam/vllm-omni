#!/usr/bin/env python3
"""Check the pinned eight-layer rolling-KV ONNX step on independent generated codes."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from probe_qwen_tts_stateful_npu import run_steps
from replay_qwen_tts_layer0_npu_suffix import (
    REVISION,
    decode_tail,
    new_cache,
    relative_l2,
    sha256,
    state_tensors,
    step,
)
from vllm_omni.edge.decoder_export import load_code2wav_decoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "source-report", "codes", "onnx", "fixture", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-onnx-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--steps", type=int, default=11)
    args = parser.parse_args()
    if not 1 <= args.steps <= 11:
        parser.error("steps must be 1..11 for the 95-frame state bucket")

    import onnxruntime as ort

    source = json.loads(args.source_report.read_text(encoding="utf-8-sig"))
    if (args.model.resolve().name != REVISION
            or source["model_revision"] != REVISION
            or sha256(args.codes) != source["files"]["codes"]["sha256"]
            or sha256(args.model / "speech_tokenizer" / "model.safetensors")
            != source["decoder_weight_sha256"]
            or sha256(args.onnx) != args.expected_onnx_sha256
            or sha256(args.fixture) != args.expected_fixture_sha256):
        raise ValueError("checkpoint, independent codes, ONNX or fixture hash changed")
    with np.load(args.codes, allow_pickle=False) as archive:
        code_array = np.ascontiguousarray(archive["codes"])
    with np.load(args.fixture, allow_pickle=False) as archive:
        fixture = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    if (code_array.shape[0] < 95 + 2 * args.steps or code_array.shape[1] != 16
            or not np.issubdtype(code_array.dtype, np.integer)):
        raise ValueError("generated-code shape or dtype changed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    observed = run_steps(session, fixture, layer0=False, step_count=args.steps)

    torch.set_num_threads(4)
    decoder = load_code2wav_decoder(str(args.model)).eval()
    config = decoder.config
    if config.num_hidden_layers != 8 or config.sliding_window != 72:
        raise ValueError("source decoder state contract changed")
    codes = torch.from_numpy(code_array.T.copy()).unsqueeze(0).long()
    cache = new_cache(config, fixture)
    rows = []
    with torch.no_grad():
        source_cache = {"prefix_frames": 0}
        decoder.decode_xvec_exact(codes[:, :, :95], source_cache)
        prefill_error = max(relative_l2(a, b) for a, b in zip(
            state_tensors(source_cache["exact_xvec_transformer_cache"]),
            state_tensors(cache)))
        if prefill_error > 1e-5:
            raise ValueError(f"source prefill differs from fixture: {prefill_error}")
        history = source_cache["exact_xvec_hidden_tail"].detach().clone()
        for index, item in enumerate(observed):
            start = 95 + 2 * index
            conv_name = "conv" if index == 0 else "conv_next" if index == 1 else f"conv_step{index}"
            hidden, state, _ = step(decoder.pre_transformer, config, cache,
                                    torch.from_numpy(fixture[conv_name]), start)
            outputs = [torch.from_numpy(value) for value in item["outputs"]]
            if len(outputs) != 17:
                raise ValueError("eight-layer graph output contract changed")
            source_wave = decoder.decode_xvec_exact(
                codes[:, :, start:start + 2], source_cache)
            graph_wave, history = decode_tail(decoder, history, outputs[0])
            rows.append({
                "start_frame": start,
                "hidden_relative_l2": relative_l2(hidden, outputs[0]),
                "max_cache_relative_l2": max(relative_l2(a, b)
                                             for a, b in zip(state, outputs[1:])),
                "waveform_relative_l2": relative_l2(source_wave, graph_wave),
                "one_ort_cpu_step_s_not_profile": item["elapsed_s"],
                "finite": bool(all(torch.isfinite(value).all() for value in outputs)
                               and torch.isfinite(graph_wave).all()),
            })
    report = {
        "scope": "real eight-layer rolling-KV ONNX CPU replay on a generated utterance; not NPU or complete TTS",
        "checkpoint_revision": REVISION,
        "source_report_sha256": sha256(args.source_report),
        "codes_sha256": sha256(args.codes),
        "decoder_weight_sha256": source["decoder_weight_sha256"],
        "onnx_sha256": args.expected_onnx_sha256,
        "fixture_sha256": args.expected_fixture_sha256,
        "os": platform.platform(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "threads": 4,
        "prefill_state_max_relative_l2": prefill_error,
        "rows": rows,
        "status": ("cpu_state_and_waveform_parity_pass" if all(
            row["finite"] and row["hidden_relative_l2"] <= 1e-4
            and row["max_cache_relative_l2"] <= 1e-4
            and row["waveform_relative_l2"] <= 1e-4 for row in rows)
            else "cpu_state_or_waveform_parity_failed"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "steps": len(rows),
                      "worst_waveform_relative_l2": max(row["waveform_relative_l2"]
                                                        for row in rows)}))


if __name__ == "__main__":
    main()
