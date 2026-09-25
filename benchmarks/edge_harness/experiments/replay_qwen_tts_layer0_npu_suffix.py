#!/usr/bin/env python3
"""Replay a captured AMD NPU first-layer output through the real CPU suffix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from vllm_omni.edge.decoder_export import load_code2wav_decoder


REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: torch.Tensor, actual: torch.Tensor) -> float:
    a, b = reference.double(), actual.double()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-12))


def new_cache(config, fixture):
    cache = DynamicCache(config=config)
    for index, layer in enumerate(cache.layers):
        layer.update(torch.from_numpy(fixture[f"key_{index}"]).clone(),
                     torch.from_numpy(fixture[f"value_{index}"]).clone())
    return cache


def state_tensors(cache):
    return tuple(tensor.detach().clone() for layer in cache.layers
                 for tensor in (layer.keys, layer.values))


def decode_tail(decoder, history, new_hidden):
    joined = torch.cat([history, new_hidden], dim=1)
    hidden = joined.permute(0, 2, 1)
    for blocks in decoder.upsample:
        for block in blocks:
            hidden = block(hidden)
    wave = hidden
    for block in decoder.decoder:
        wave = block(wave)
    return wave[..., -new_hidden.shape[1] * decoder.total_upsample:].clamp(-1, 1), joined[:, -12:, :]


def step(transformer, config, cache, conv, start, injected=None):
    positions = torch.tensor([[start, start + 1]], dtype=torch.long)
    key_positions = positions[:, :1] - (config.sliding_window - 1) + torch.arange(
        config.sliding_window + 1)
    allowed = ((key_positions[:, None, None, :] <= positions[:, None, :, None])
               & (key_positions[:, None, None, :] >
                  positions[:, None, :, None] - config.sliding_window))
    mask = torch.where(allowed, 0.0, torch.finfo(conv.dtype).min).to(conv.dtype)
    layer0 = []

    def layer0_hook(_module, _inputs, output):
        layer0.append(output.detach().clone())
        return torch.from_numpy(injected[0]).to(output) if injected is not None else None

    hook = transformer.layers[0].register_forward_hook(layer0_hook)
    try:
        result = transformer(
            inputs_embeds=conv,
            attention_mask={"sliding_attention": mask},
            position_ids=positions,
            cache_position=positions[0],
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state.detach().clone()
    finally:
        hook.remove()
    if injected is not None:
        cache.layers[0].keys = torch.from_numpy(injected[1]).clone()
        cache.layers[0].values = torch.from_numpy(injected[2]).clone()
    return result, state_tensors(cache), layer0[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "codes", "fixture", "capture", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-codes-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--expected-capture-sha256", required=True)
    args = parser.parse_args()

    if (args.model.resolve().name != REVISION
            or sha256(args.codes) != args.expected_codes_sha256
            or sha256(args.fixture) != args.expected_fixture_sha256
            or sha256(args.capture) != args.expected_capture_sha256):
        raise ValueError("checkpoint revision or fixture/capture hash changed")
    with np.load(args.fixture, allow_pickle=False) as archive:
        fixture = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    with np.load(args.codes, allow_pickle=False) as archive:
        code_array = np.ascontiguousarray(archive["codes"])
    if code_array.shape != (117, 16):
        raise ValueError("generated-code fixture shape changed")
    codes = torch.from_numpy(code_array.T.copy()).unsqueeze(0).long()
    with np.load(args.capture, allow_pickle=False) as archive:
        captured = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    step_count = len([name for name in captured if name.startswith("npu_step")
                      and name.endswith("_out0")])
    if not 1 <= step_count <= 11:
        raise ValueError(f"expected 1..11 captured steps, got {step_count}")
    torch.set_num_threads(4)
    decoder = load_code2wav_decoder(str(args.model)).eval()
    config = decoder.config
    if config.num_hidden_layers != 8 or config.sliding_window != 72:
        raise ValueError("the retained decoder state contract changed")
    control_cache = new_cache(config, fixture)
    hybrid_cache = new_cache(config, fixture)
    rows = []
    with torch.no_grad():
        source_decode_caches = {"prefix_frames": 0}
        decoder.decode_xvec_exact(codes[:, :, :95], source_decode_caches)
        prefill_state_error = max(relative_l2(a, b) for a, b in zip(
            state_tensors(source_decode_caches["exact_xvec_transformer_cache"]),
            state_tensors(control_cache)))
        if prefill_state_error > 1e-5:
            raise ValueError(f"95-frame decoder prefill differs from export fixture: {prefill_state_error}")
        control_history = source_decode_caches["exact_xvec_hidden_tail"].detach().clone()
        hybrid_history = control_history.clone()
        for index in range(step_count):
            start = 95 + 2 * index
            conv_name = "conv" if index == 0 else "conv_next" if index == 1 else f"conv_step{index}"
            conv = torch.from_numpy(fixture[conv_name])
            source_hidden, source_state, source_layer0 = step(
                decoder.pre_transformer, config, control_cache, conv, start)
            npu_tuple = tuple(captured[f"npu_step{index}_out{output_index}"]
                              for output_index in range(3))
            cpu_tuple = tuple(captured[f"cpu_step{index}_out{output_index}"]
                              for output_index in range(3))
            control_errors = [relative_l2(source_layer0, torch.from_numpy(cpu_tuple[0]))]
            control_errors.extend(relative_l2(source_state[output_index],
                                              torch.from_numpy(cpu_tuple[output_index + 1]))
                                  for output_index in range(2))
            hybrid_hidden, hybrid_state, _ = step(
                decoder.pre_transformer, config, hybrid_cache, conv, start,
                injected=npu_tuple)
            source_wave = decoder.decode_xvec_exact(
                codes[:, :, start:start + 2], source_decode_caches)
            control_wave, control_history = decode_tail(
                decoder, control_history, source_hidden)
            hybrid_wave, hybrid_history = decode_tail(
                decoder, hybrid_history, hybrid_hidden)
            if (max(control_errors) > 1e-4
                    or relative_l2(source_wave, control_wave) > 1e-4):
                raise ValueError(f"CPU suffix control diverged at frame {start}")
            rows.append({
                "start_frame": start,
                "cpu_capture_max_relative_l2_vs_source": max(control_errors),
                "injected_layer0_hidden_relative_l2": relative_l2(
                    source_layer0, torch.from_numpy(npu_tuple[0])),
                "final_hidden_relative_l2": relative_l2(source_hidden, hybrid_hidden),
                "max_all_layer_state_relative_l2": max(
                    relative_l2(a, b) for a, b in zip(source_state, hybrid_state)),
                "cpu_suffix_wave_relative_l2_vs_source_decode": relative_l2(
                    source_wave, control_wave),
                "npu_layer0_cpu_suffix_wave_relative_l2": relative_l2(
                    source_wave, hybrid_wave),
                "finite": bool(torch.isfinite(hybrid_hidden).all()
                               and torch.isfinite(hybrid_wave).all()
                               and all(torch.isfinite(value).all() for value in hybrid_state)),
            })
    report = {
        "scope": "captured native AMD NPU first-layer output replayed through seven real CPU transformer layers; not a complete TTS request",
        "checkpoint_revision": REVISION,
        "decoder_weight_sha256": sha256(args.model / "speech_tokenizer" / "model.safetensors"),
        "codes_sha256": sha256(args.codes),
        "fixture_sha256": sha256(args.fixture),
        "capture_sha256": sha256(args.capture),
        "prefill_state_max_relative_l2": prefill_state_error,
        "torch": torch.__version__,
        "rows": rows,
        "waveform_gate": {
            "relative_l2_max": 0.01,
            "passed_chunks": sum(row["finite"] and
                                 row["npu_layer0_cpu_suffix_wave_relative_l2"] <= 0.01
                                 for row in rows),
            "total_chunks": len(rows),
        },
        "status": ("waveform_gate_pass_on_fixture"
                   if all(row["finite"] and
                          row["npu_layer0_cpu_suffix_wave_relative_l2"] <= 0.01
                          for row in rows)
                   else "waveform_gate_failed_on_fixture"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
