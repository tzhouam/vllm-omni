#!/usr/bin/env python3
"""Capture an independent real-code rollout for the pinned Qwen3-TTS MLP cut."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers.cache_utils import DynamicCache

from vllm_omni.edge.decoder_export import load_code2wav_decoder


REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "codes", "prefix", "attention", "mlp", "fixture",
                 "replay-fixture", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("codes", "prefix", "attention", "mlp"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    args = parser.parse_args()
    if args.model.resolve().name != REVISION:
        raise ValueError("unexpected checkpoint revision")
    for name in ("codes", "prefix", "attention", "mlp"):
        if sha256(getattr(args, name)) != getattr(args, f"expected_{name}_sha256"):
            raise ValueError(f"{name} hash changed")
    with np.load(args.codes, allow_pickle=False) as archive:
        codes = np.ascontiguousarray(archive["codes"])
    if codes.ndim != 2 or codes.shape[0] < 117 or codes.shape[1] != 16:
        raise ValueError("need at least 117 frames of 16-code generated speech")

    torch.set_num_threads(4)
    decoder = load_code2wav_decoder(str(args.model)).eval()
    config = decoder.config
    if config.sliding_window != 72 or config.num_hidden_layers != 8:
        raise ValueError("rolling-state contract changed")
    prefix = ort.InferenceSession(str(args.prefix), providers=["CPUExecutionProvider"])
    attention = ort.InferenceSession(str(args.attention), providers=["CPUExecutionProvider"])
    mlp = ort.InferenceSession(str(args.mlp), providers=["CPUExecutionProvider"])
    if ([item.name for item in prefix.get_inputs()] != ["conv"]
            or [item.name for item in attention.get_inputs()]
            != ["projected", "positions", "key_0", "value_0"]
            or [item.name for item in mlp.get_inputs()] != ["residual"]):
        raise ValueError("extracted ONNX contract changed")

    source_codes = torch.from_numpy(codes.T.copy()).unsqueeze(0).long()

    def conv_frames(start: int, end: int) -> torch.Tensor:
        first = max(0, start - 2)
        quantized = decoder.quantizer.decode(source_codes[:, :, first:end])
        return decoder.pre_conv(quantized)[:, :, -(end - start):].transpose(1, 2)

    with torch.no_grad():
        cache = DynamicCache(config=config)
        decoder.pre_transformer(inputs_embeds=conv_frames(0, 95),
                                past_key_values=cache, use_cache=True)
        if any(layer.keys.shape[-2] != 71 for layer in cache.layers):
            raise ValueError("95-frame prefill did not produce the expected sliding KV")
        replay = {name: tensor.detach().numpy().copy()
                  for index, layer in enumerate(cache.layers)
                  for name, tensor in ((f"key_{index}", layer.keys),
                                       (f"value_{index}", layer.values))}
        fixture = {}
        rows = []
        for index in range(11):
            start = 95 + 2 * index
            conv = conv_frames(start, start + 2)
            replay["conv" if index == 0 else "conv_next" if index == 1
                   else f"conv_step{index}"] = conv.detach().numpy().copy()
            key = cache.layers[0].keys.detach().numpy().copy()
            value = cache.layers[0].values.detach().numpy().copy()
            positions = np.array([[start, start + 1]], np.int64)
            projected = prefix.run(None, {"conv": conv.numpy()})[0]
            residual, next_key, next_value = attention.run(None, {
                "projected": projected, "positions": positions,
                "key_0": key, "value_0": value,
            })
            cpu_hidden = mlp.run(None, {"residual": residual})[0]
            current = []

            def capture(_module, _inputs, output):
                current.append(output.detach().numpy().copy())

            hook = decoder.pre_transformer.layers[0].register_forward_hook(capture)
            try:
                position_tensor = torch.from_numpy(positions)
                key_positions = position_tensor[:, :1] - 71 + torch.arange(73)
                allowed = ((key_positions[:, None, None, :] <= position_tensor[:, None, :, None])
                           & (key_positions[:, None, None, :] >
                              position_tensor[:, None, :, None] - 72))
                mask = torch.where(allowed, 0.0, torch.finfo(conv.dtype).min).to(conv.dtype)
                decoder.pre_transformer(
                    inputs_embeds=conv, attention_mask={"sliding_attention": mask},
                    position_ids=position_tensor, cache_position=position_tensor[0],
                    past_key_values=cache, use_cache=True,
                )
            finally:
                hook.remove()
            errors = [relative_l2(current[0], cpu_hidden),
                      relative_l2(cache.layers[0].keys.numpy(), next_key),
                      relative_l2(cache.layers[0].values.numpy(), next_value)]
            if max(errors) > 1e-4:
                raise ValueError(f"CPU ONNX layer diverged at frame {start}: {errors}")
            fixture[f"residual_step{index}"] = residual
            for output_index, output in enumerate((cpu_hidden, next_key, next_value)):
                fixture[f"cpu_step{index}_out{output_index}"] = output
            rows.append({"start_frame": start, "max_cpu_layer_relative_l2": max(errors)})

    for path in (args.fixture, args.replay_fixture, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fixture, **fixture)
    np.savez_compressed(args.replay_fixture, **replay)
    report = {
        "scope": "independent generated-code eleven-step CPU MLP/rolling-KV fixture; no NPU execution",
        "checkpoint_revision": REVISION,
        "decoder_weight_sha256": sha256(args.model / "speech_tokenizer" / "model.safetensors"),
        "codes_sha256": sha256(args.codes),
        "artifact_sha256": {name: sha256(getattr(args, name))
                            for name in ("prefix", "attention", "mlp")},
        "fixture_sha256": sha256(args.fixture),
        "replay_fixture_sha256": sha256(args.replay_fixture),
        "onnxruntime": ort.__version__, "torch": torch.__version__,
        "rows": rows, "status": "cpu_layer_parity_pass",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "max_error":
                      max(row["max_cpu_layer_relative_l2"] for row in rows)}))


if __name__ == "__main__":
    main()
