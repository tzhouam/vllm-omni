#!/usr/bin/env python3
"""Replay captured eight-layer AMD NPU decode outputs through the real CPU vocoder tail."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

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
    for name in ("model", "codes", "fixture", "capture", "probe-report", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-codes-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--expected-onnx-sha256", required=True)
    parser.add_argument("--expected-capture-sha256", required=True)
    args = parser.parse_args()

    probe = json.loads(args.probe_report.read_text(encoding="utf-8-sig"))
    if (args.model.resolve().name != REVISION
            or sha256(args.codes) != args.expected_codes_sha256
            or sha256(args.fixture) != args.expected_fixture_sha256
            or sha256(args.capture) != args.expected_capture_sha256
            or probe.get("capture", {}).get("sha256") != args.expected_capture_sha256
            or probe.get("artifact_sha256", {}).get("fixture") != args.expected_fixture_sha256
            or probe.get("artifact_sha256", {}).get("model") != args.expected_onnx_sha256
            or probe.get("npu_state_source") != "self"
            or probe.get("node_providers", {}).get("vitisai", 0) < 1):
        raise ValueError("full-state checkpoint, native placement, fixture or capture changed")
    with np.load(args.fixture, allow_pickle=False) as archive:
        fixture = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    with np.load(args.codes, allow_pickle=False) as archive:
        code_array = np.ascontiguousarray(archive["codes"])
    with np.load(args.capture, allow_pickle=False) as archive:
        captured = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    step_count = len([name for name in captured if name.startswith("npu_step")
                      and name.endswith("_out0")])
    if (not 1 <= step_count <= 11 or code_array.ndim != 2
            or code_array.shape[0] < 95 + 2 * step_count
            or code_array.shape[1] != 16):
        raise ValueError("generated codes or full-state step count changed")

    torch.set_num_threads(4)
    decoder = load_code2wav_decoder(str(args.model)).eval()
    config = decoder.config
    if config.num_hidden_layers != 8 or config.sliding_window != 72:
        raise ValueError("source decoder contract changed")
    codes = torch.from_numpy(code_array.T.copy()).unsqueeze(0).long()
    control_cache = new_cache(config, fixture)
    rows = []
    source_waves = []
    npu_waves = []
    with torch.no_grad():
        source_decode_caches = {"prefix_frames": 0}
        decoder.decode_xvec_exact(codes[:, :, :95], source_decode_caches)
        prefill_error = max(relative_l2(a, b) for a, b in zip(
            state_tensors(source_decode_caches["exact_xvec_transformer_cache"]),
            state_tensors(control_cache)))
        if prefill_error > 1e-5:
            raise ValueError(f"decoder prefill differs from export fixture: {prefill_error}")
        control_history = source_decode_caches["exact_xvec_hidden_tail"].detach().clone()
        npu_history = control_history.clone()
        for index in range(step_count):
            start = 95 + 2 * index
            conv_name = "conv" if index == 0 else "conv_next" if index == 1 else f"conv_step{index}"
            conv = torch.from_numpy(fixture[conv_name])
            source_hidden, source_state, _ = step(
                decoder.pre_transformer, config, control_cache, conv, start)
            cpu = [torch.from_numpy(captured[f"cpu_step{index}_out{j}"])
                   for j in range(17)]
            npu = [torch.from_numpy(captured[f"npu_step{index}_out{j}"])
                   for j in range(17)]
            control_errors = [relative_l2(source_hidden, cpu[0])]
            control_errors.extend(relative_l2(a, b)
                                  for a, b in zip(source_state, cpu[1:]))
            source_wave = decoder.decode_xvec_exact(
                codes[:, :, start:start + 2], source_decode_caches)
            control_wave, control_history = decode_tail(
                decoder, control_history, source_hidden)
            npu_wave, npu_history = decode_tail(decoder, npu_history, npu[0])
            if max(control_errors) > 1e-4 or relative_l2(source_wave, control_wave) > 1e-4:
                raise ValueError(f"CPU full-state control diverged at frame {start}")
            source_waves.append(source_wave.detach().clone())
            npu_waves.append(npu_wave.detach().clone())
            error = relative_l2(source_wave, npu_wave)
            rms = float(torch.sqrt(torch.mean(source_wave.double().square())))
            rmse = float(torch.sqrt(torch.mean((npu_wave.double()-source_wave.double()).square())))
            rows.append({
                "start_frame": start,
                "cpu_capture_max_relative_l2_vs_source": max(control_errors),
                "npu_hidden_relative_l2_vs_source": relative_l2(source_hidden, npu[0]),
                "npu_max_cache_relative_l2_vs_source": max(
                    relative_l2(a, b) for a, b in zip(source_state, npu[1:])),
                "npu_full_state_cpu_tail_waveform_relative_l2": error,
                "source_wave_rms": rms,
                "npu_wave_snr_db": 20.0 * math.log10(max(rms, 1e-12)/max(rmse, 1e-12)),
                "finite": bool(torch.isfinite(npu_wave).all() and
                               all(torch.isfinite(value).all() for value in npu)),
            })
    joined_error = relative_l2(torch.cat(source_waves, dim=-1),
                               torch.cat(npu_waves, dim=-1))
    report = {
        "scope": "captured native AMD NPU eight-layer rolling-state output through real CPU vocoder tail; offline component, not complete TTS",
        "checkpoint_revision": REVISION,
        "decoder_weight_sha256": sha256(args.model / "speech_tokenizer" / "model.safetensors"),
        "codes_sha256": sha256(args.codes),
        "onnx_sha256": args.expected_onnx_sha256,
        "fixture_sha256": sha256(args.fixture),
        "capture_sha256": sha256(args.capture),
        "probe_report_sha256": sha256(args.probe_report),
        "prefill_state_max_relative_l2": prefill_error,
        "torch": torch.__version__,
        "rows": rows,
        "joined_waveform_relative_l2": joined_error,
        "waveform_gate": {"relative_l2_max": 0.01,
                          "passed_chunks": sum(row["finite"] and
                                               row["npu_full_state_cpu_tail_waveform_relative_l2"] <= 0.01
                                               for row in rows),
                          "total_chunks": len(rows)},
        "status": ("waveform_gate_pass_on_fixture" if all(
            row["finite"] and row["npu_full_state_cpu_tail_waveform_relative_l2"] <= 0.01
            for row in rows) else "waveform_gate_failed_on_fixture"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "waveform_gate": report["waveform_gate"]}))


if __name__ == "__main__":
    main()
