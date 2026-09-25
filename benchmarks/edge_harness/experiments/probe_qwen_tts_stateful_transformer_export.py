#!/usr/bin/env python3
"""Probe an explicit rolling-KV ONNX boundary for the real Qwen3-TTS decoder.

The wrapper calls the checkpoint's own pre-transformer. It makes its already
validated sliding KV state explicit instead of replaying a truncated window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path


REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def relative_l2(reference, actual) -> float:
    import torch

    a, b = reference.double(), actual.double()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "source-report", "codes", "report", "onnx", "fixture", "layer0-onnx"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import numpy as np
    import torch
    from torch import nn
    from transformers.cache_utils import DynamicCache

    from vllm_omni.edge.decoder_export import load_code2wav_decoder

    source = json.loads(args.source_report.read_text(encoding="utf-8-sig"))
    if (args.model.resolve().name != REVISION
            or source["model_revision"] != REVISION
            or source["files"]["codes"]["sha256"] != sha256(args.codes)
            or source["decoder_weight_sha256"] != sha256(
                args.model / "speech_tokenizer" / "model.safetensors")):
        raise ValueError("generated codes or decoder revision changed")
    with np.load(args.codes, allow_pickle=False) as archive:
        codes = np.ascontiguousarray(archive["codes"])
    if codes.shape != (117, 16) or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("retained code stream shape changed")

    torch.set_num_threads(args.threads)
    decoder = load_code2wav_decoder(str(args.model)).eval()
    config = decoder.config
    if (config.sliding_window != 72 or config.num_hidden_layers != 8
            or config.num_key_value_heads != 16 or config.head_dim != 64):
        raise ValueError("this fixed rolling-state bucket needs a new contract")

    class StatefulTransformerStep(nn.Module):
        def __init__(self, transformer, state_config):
            super().__init__()
            self.transformer = transformer
            self.state_config = state_config
            self.layers = int(state_config.num_hidden_layers)
            self.window = int(state_config.sliding_window)

        def forward(self, conv, positions, *past):
            cache = DynamicCache(config=self.state_config)
            for index, layer in enumerate(cache.layers):
                layer.update(past[index * 2], past[index * 2 + 1])
            # Cached keys represent positions [first_new - 71, first_new).
            key_positions = positions[:, :1] - (self.window - 1) + torch.arange(
                self.window + 1, device=positions.device)
            allowed = ((key_positions[:, None, None, :] <= positions[:, None, :, None])
                       & (key_positions[:, None, None, :] >
                          positions[:, None, :, None] - self.window))
            mask = torch.where(allowed, 0.0, torch.finfo(conv.dtype).min).to(conv.dtype)
            result = self.transformer(
                inputs_embeds=conv,
                attention_mask={"sliding_attention": mask},
                position_ids=positions,
                cache_position=positions[0],
                past_key_values=cache,
                use_cache=True,
            )
            updated = tuple(tensor for layer in cache.layers for tensor in (layer.keys, layer.values))
            return (result.last_hidden_state, *updated)

    wrapper = StatefulTransformerStep(decoder.pre_transformer, config).eval()
    tensor = torch.from_numpy(codes.T.copy()).unsqueeze(0).long()

    def conv_frames(start: int, end: int):
        # The source decoder's two-frame causal convolution tail is reused.
        first = max(0, start - 2)
        quantized = decoder.quantizer.decode(tensor[:, :, first:end])
        return decoder.pre_conv(quantized)[:, :, -(end - start):].transpose(1, 2)

    with torch.no_grad():
        cache = DynamicCache(config=config)
        decoder.pre_transformer(
            inputs_embeds=conv_frames(0, 95), past_key_values=cache, use_cache=True)
        if (cache.get_seq_length() != 95
                or any(layer.keys.shape[-2] != 71 for layer in cache.layers)):
            raise ValueError("native rolling cache is not the expected 95/71 state")
        past = tuple(tensor.detach().clone()
                     for layer in cache.layers for tensor in (layer.keys, layer.values))
        rows = []
        for start in (95, 97):
            conv = conv_frames(start, start + 2)
            positions = torch.arange(start, start + 2).unsqueeze(0)
            reference = decoder.pre_transformer(
                inputs_embeds=conv, past_key_values=cache, use_cache=True).last_hidden_state
            observed = wrapper(conv, positions, *past)
            error = relative_l2(reference, observed[0])
            next_past = observed[1:]
            cache_error = max(relative_l2(native, explicit) for native, explicit in zip(
                (tensor for layer in cache.layers for tensor in (layer.keys, layer.values)),
                next_past))
            rows.append({"start_frame": start, "hidden_relative_l2": error,
                         "max_cache_relative_l2": cache_error,
                         "state_tensor_bytes": sum(int(t.numel() * t.element_size()) for t in next_past)})
            if error > 1e-5 or cache_error > 1e-5:
                raise ValueError(f"explicit rolling KV differs at frame {start}: {rows[-1]}")
            past = tuple(item.detach().clone() for item in next_past)

        # Export at the first later-state bucket; position IDs remain tensor inputs.
        # The prefill cache was consumed above, so recover its 95-frame state.
        fresh = DynamicCache(config=config)
        decoder.pre_transformer(inputs_embeds=conv_frames(0, 95),
                                past_key_values=fresh, use_cache=True)
        export_inputs = (conv_frames(95, 97), torch.arange(95, 97).unsqueeze(0), *(
            tensor.detach().clone() for layer in fresh.layers
            for tensor in (layer.keys, layer.values)))

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    export_error = None
    try:
        torch.onnx.export(
            wrapper, export_inputs, str(args.onnx), opset_version=20,
            input_names=["conv", "positions"] + [f"{kind}_{index}" for index in range(8)
                                                  for kind in ("key", "value")],
            output_names=["hidden"] + [f"next_{kind}_{index}" for index in range(8)
                                               for kind in ("key", "value")],
            dynamo=False,
        )
    except Exception as error:
        export_error = f"{type(error).__name__}: {error}"
    ort_error = None
    ort_rows = []
    layer0_error = None
    layer0_comparison = None
    if export_error is None:
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = args.threads
            session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            input_names = [item.name for item in session.get_inputs()]
            past_tensors = list(export_inputs[2:])
            args.fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture_values = {name: tensor.detach().numpy() for name, tensor in zip(
                input_names, export_inputs)}
            fixture_values["conv_next"] = conv_frames(97, 99).detach().numpy()
            np.savez_compressed(args.fixture, **fixture_values)
            for start in (95, 97):
                conv = conv_frames(start, start + 2)
                positions = torch.arange(start, start + 2).unsqueeze(0)
                feeds = {name: tensor.detach().numpy() for name, tensor in zip(
                    input_names, (conv, positions, *past_tensors))}
                started = time.perf_counter()
                observed = session.run(None, feeds)
                elapsed = time.perf_counter() - started
                with torch.no_grad():
                    reference = wrapper(conv, positions, *past_tensors)
                errors = [relative_l2(expected, torch.from_numpy(actual))
                          for expected, actual in zip(reference, observed)]
                ort_rows.append({"start_frame": start, "hidden_relative_l2": errors[0],
                                 "max_cache_relative_l2": max(errors[1:]),
                                 "one_ort_cpu_forward_s_not_profile": elapsed})
                if errors[0] > 1e-4 or max(errors[1:]) > 1e-4:
                    raise ValueError(f"ORT CPU rolling state differs: {ort_rows[-1]}")
                past_tensors = [torch.from_numpy(value) for value in observed[1:]]
        except Exception as error:
            ort_error = f"{type(error).__name__}: {error}"
    if export_error is None and ort_error is None:
        try:
            import onnx
            import onnxruntime as ort

            args.layer0_onnx.parent.mkdir(parents=True, exist_ok=True)
            onnx.utils.extract_model(
                str(args.onnx), str(args.layer0_onnx),
                ["conv", "positions", "key_0", "value_0"],
                ["/transformer/layers.0/Add_1_output_0", "next_key_0", "next_value_0"],
            )
            captured = []
            hook = decoder.pre_transformer.layers[0].register_forward_hook(
                lambda _module, _inputs, output: captured.append(output.detach().clone()))
            try:
                with torch.no_grad():
                    full = wrapper(*export_inputs)
            finally:
                hook.remove()
            if len(captured) != 1:
                raise ValueError("first transformer layer hook did not fire once")
            layer0 = ort.InferenceSession(str(args.layer0_onnx),
                                          providers=["CPUExecutionProvider"])
            subset = {name: tensor.detach().numpy() for name, tensor in zip(
                ("conv", "positions", "key_0", "value_0"), export_inputs[:4])}
            outputs = layer0.run(None, subset)
            expected = (captured[0], full[1], full[2])
            errors = [relative_l2(reference, torch.from_numpy(actual))
                      for reference, actual in zip(expected, outputs)]
            layer0_comparison = {
                "node_count": len(onnx.load(str(args.layer0_onnx), load_external_data=False).graph.node),
                "hidden_relative_l2": errors[0],
                "key_relative_l2": errors[1],
                "value_relative_l2": errors[2],
            }
            if max(errors) > 1e-4:
                raise ValueError(f"first-layer ORT output differs from source: {layer0_comparison}")
        except Exception as error:
            layer0_error = f"{type(error).__name__}: {error}"
    report = {
        "scope": "real-weight Qwen3-TTS rolling-KV pre-transformer; CPU export feasibility, not live NPU/TTS",
        "checkpoint_revision": REVISION,
        "decoder_weight_sha256": source["decoder_weight_sha256"],
        "codes_sha256": sha256(args.codes),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "threads": args.threads,
        "rolling_rows": rows,
        "onnx_export_s": time.perf_counter() - started,
        "onnx_export_error": export_error,
        "onnx_sha256": sha256(args.onnx) if args.onnx.exists() and export_error is None else None,
        "fixture_sha256": sha256(args.fixture) if args.fixture.exists() and ort_error is None else None,
        "ort_cpu_rows": ort_rows,
        "ort_cpu_error": ort_error,
        "layer0_onnx_sha256": sha256(args.layer0_onnx) if args.layer0_onnx.exists() and layer0_error is None else None,
        "layer0_cpu_comparison": layer0_comparison,
        "layer0_error": layer0_error,
        "status": (
            "cpu_and_ort_state_parity_pass" if export_error is None and ort_error is None and layer0_error is None
            else "cpu_state_parity_pass_onnx_export_failed" if export_error is not None
            else "cpu_state_parity_pass_ort_cpu_failed" if ort_error is not None
            else "cpu_state_parity_pass_layer0_failed"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": rows,
                      "ort_cpu_rows": ort_rows, "export_error": export_error,
                      "ort_error": ort_error, "layer0_error": layer0_error}))


if __name__ == "__main__":
    main()
