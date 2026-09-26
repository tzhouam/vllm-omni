# SPDX-License-Identifier: Apache-2.0
"""Compare the real Spark decode-step export boundary with HF cached decode.

This is a CPU numerical probe, not a device-local generation benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from vllm_omni.edge.spark_export import (
    FULL,
    SLIDING,
    SparkDecodeCache,
    SparkDecodeStep,
    config_from_spark,
    load_spark_weights,
)
from vllm_omni.edge import spark_export


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_l2(reference: torch.Tensor, actual: torch.Tensor) -> float:
    a, b = reference.double(), actual.double()
    return float(torch.linalg.vector_norm(a - b)
                 / torch.linalg.vector_norm(a).clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prompt", default="What is the capital of France? Answer in one word.")
    parser.add_argument("--prefill-tokens", type=int, default=0)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--context-capacity", type=int, default=0,
                        help="fixed full-attention cache capacity; default fits the run")
    parser.add_argument("--full-bucket-width", type=int, default=0,
                        help="grow the full-attention cache in this many fixed-shape slots")
    parser.add_argument("--reference-dtype", choices=("bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--export-arithmetic",
                        choices=("fp32_export", "hf_bf16_reference"),
                        default="fp32_export")
    parser.add_argument("--attention-accumulation",
                        choices=("bf16", "fp32_score", "fp32_value", "fp32_both"),
                        default="bf16",
                        help="CPU diagnostic for BF16 attention matmul rounding")
    parser.add_argument("--softmax-accumulation", choices=("source_fp32", "fp64"),
                        default="source_fp32",
                        help="CPU diagnostic for padded attention normalization")
    parser.add_argument("--cache-layout", choices=("auto", "roll"), default="auto")
    parser.add_argument("--compact-inputs", action="store_true",
                        help="diagnose static padding by using only filled cache slots")
    parser.add_argument("--compact-layer-type",
                        choices=("all", "sliding", "full"), default="all")
    parser.add_argument("--ordered-sliding", action="store_true",
                        help="read the physical ring in chronological order")
    parser.add_argument("--right-align-full", action="store_true",
                        help="place filled full-cache entries next to the new token in a padded CPU diagnostic")
    parser.add_argument("--trace-position", type=int, default=-1,
                        help="capture source/export hidden-state parity after each layer at this decode position")
    args = parser.parse_args()
    if args.decode_steps < 1:
        raise ValueError("decode-steps must be positive")
    if args.full_bucket_width < 0:
        raise ValueError("full-bucket-width cannot be negative")
    if args.ordered_sliding and args.cache_layout == "roll":
        raise ValueError("ordered-sliding applies only to the ring layout")
    if args.right_align_full and args.compact_inputs and args.compact_layer_type in ("all", "full"):
        raise ValueError("right-aligned full cache conflicts with compact full inputs")
    if args.softmax_accumulation != "source_fp32" and args.export_arithmetic != "hf_bf16_reference":
        raise ValueError("promoted softmax needs the BF16 source reference mode")

    torch.set_num_threads(8)
    hf_config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model_class = get_class_from_dynamic_module(
        hf_config.auto_map["AutoModelForCausalLM"], str(args.model)
    )
    # This revision targets Transformers 4.57. The installed 5.14 mask API
    # renamed input_embeds and removed the unused cache_position argument.
    # Keep the compatibility shim local to this reference probe.
    def mask_compat(mask_function):
        def call(**kwargs):
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            kwargs.pop("cache_position", None)
            return mask_function(**kwargs)

        return call

    model_module = sys.modules[model_class.__module__]
    model_module.create_causal_mask = mask_compat(create_causal_mask)
    model_module.create_sliding_window_causal_mask = mask_compat(
        create_sliding_window_causal_mask
    )
    if isinstance(getattr(model_class, "_tied_weights_keys", None), list):
        model_class._tied_weights_keys = {
            "lm_head.weight": "model.embedding.weight"
        }
    model = model_class.from_pretrained(
        args.model, dtype=getattr(torch, args.reference_dtype), local_files_only=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )["input_ids"]
    if args.prefill_tokens:
        if args.prefill_tokens < prompt_ids.shape[1]:
            raise ValueError("prefill-tokens must not shorten the base prompt")
        filler = tokenizer(
            " Repeat the exact request and give a concise answer.",
            add_special_tokens=False,
        )["input_ids"]
        extra = args.prefill_tokens - prompt_ids.shape[1]
        prompt_ids = torch.cat(
            (prompt_ids, torch.tensor([filler * (extra // len(filler) + 1)])[:, :extra]),
            dim=1,
        )

    with torch.inference_mode():
        prefill = model(prompt_ids, use_cache=True, logits_to_keep=1)
        current_token = int(prefill.logits[0, -1].argmax())
        reference_cache = prefill.past_key_values
        cache_shapes = [
            [list(layer.keys.shape), list(layer.values.shape)]
            for layer in reference_cache.layers
        ]
        cached_tensors = [
            (layer.keys.detach().float().clone(), layer.values.detach().float().clone())
            for layer in reference_cache.layers
        ]
        raw_config = json.loads((args.model / "config.json").read_text())
        cfg = config_from_spark(
            raw_config, arithmetic_mode=args.export_arithmetic,
            cache_layout=args.cache_layout,
            attention_accumulation=args.attention_accumulation,
        )
        step = SparkDecodeStep(cfg).eval()
        copied = load_spark_weights(step, args.model)
        position = int(prompt_ids.shape[1])
        required_capacity = position + args.decode_steps
        context_capacity = (args.context_capacity or
                            max(1024, required_capacity + 1))
        if args.full_bucket_width and not args.context_capacity:
            context_capacity = (math.ceil(context_capacity / args.full_bucket_width)
                                * args.full_bucket_width)
        if context_capacity < position + args.decode_steps:
            raise ValueError("context-capacity is too small for this rollout")
        initial_capacity = (context_capacity if not args.full_bucket_width else
                            math.ceil((position + 1) / args.full_bucket_width)
                            * args.full_bucket_width)
        if initial_capacity > context_capacity:
            raise ValueError("initial full-cache bucket exceeds admitted context")
        state = SparkDecodeCache(
            cfg, initial_capacity,
            dtype=(torch.bfloat16 if args.export_arithmetic == "hf_bf16_reference"
                   else torch.float32),
        )
        state.seed(cached_tensors, position)
        steps: list[dict] = []
        bucket_transitions: list[dict] = []
        traced_layer_hidden: list[dict] = []
        traced_attention_matmuls: list[dict] = []
        for _ in range(args.decode_steps):
            decode_position = state.position
            if args.full_bucket_width and decode_position >= state.max_context:
                next_capacity = min(context_capacity,
                                    state.max_context + args.full_bucket_width)
                state.grow_full_capacity(next_capacity)
                bucket_transitions.append({"position": decode_position,
                                           "full_capacity": next_capacity})
            x = model.model.embedding(torch.tensor([[current_token]])).float()
            step_inputs = list(state.step_inputs(x))
            if (args.ordered_sliding and cfg.layout_for(SLIDING) == "ring"
                    and decode_position >= cfg.sliding_window - 1):
                capacity = cfg.sliding_window - 1
                order = (torch.arange(capacity) + decode_position - capacity) % capacity
                for i, layer_type in enumerate(cfg.layer_types):
                    if layer_type == SLIDING:
                        step_inputs[7 + 2 * i] = state.buffers[2 * i].index_select(2, order)
                        step_inputs[8 + 2 * i] = state.buffers[2 * i + 1].index_select(2, order)
            if args.right_align_full:
                capacity = state.max_context
                step_inputs[6] = torch.full_like(step_inputs[6], float("-inf"))
                step_inputs[6][:, :, :, capacity - decode_position:] = 0
                for i, layer_type in enumerate(cfg.layer_types):
                    if layer_type == FULL:
                        for kv in (0, 1):
                            source = state.buffers[2 * i + kv]
                            packed = torch.zeros_like(source)
                            packed[:, :, capacity - decode_position:] = source[:, :, :decode_position]
                            step_inputs[7 + 2 * i + kv] = packed
            if args.compact_inputs:
                compact_sw = args.compact_layer_type in ("all", "sliding")
                compact_full = args.compact_layer_type in ("all", "full")
                if compact_sw and cfg.layout_for(SLIDING) == "roll":
                    raise ValueError("compact sliding diagnostic requires a ring cache")
                if compact_sw and decode_position >= cfg.sliding_window - 1:
                    raise ValueError("compact sliding diagnostic requires an unfilled ring")
                if compact_sw:
                    step_inputs[5] = torch.zeros_like(
                        step_inputs[5][:, :, :, :decode_position + 1]
                    )
                if compact_full:
                    step_inputs[6] = torch.zeros_like(
                        step_inputs[6][:, :, :, :decode_position + 1]
                    )
                for i, layer_type in enumerate(cfg.layer_types):
                    if ((layer_type == SLIDING and compact_sw)
                            or (layer_type == FULL and compact_full)):
                        step_inputs[7 + 2 * i] = state.buffers[2 * i][:, :, :decode_position]
                        step_inputs[8 + 2 * i] = state.buffers[2 * i + 1][:, :, :decode_position]
            hooks = []
            export_hidden: dict[int, torch.Tensor] = {}
            source_hidden: dict[int, torch.Tensor] = {}
            if decode_position == args.trace_position:
                for i, layer in enumerate(step.layers):
                    hooks.append(layer.register_forward_hook(
                        lambda _module, _inputs, output, index=i:
                        export_hidden.__setitem__(index, output[0].detach().clone())
                    ))
                for i, layer in enumerate(model.model.layers):
                    hooks.append(layer.register_forward_hook(
                        lambda _module, _inputs, output, index=i:
                        source_hidden.__setitem__(index, output.detach().clone())
                    ))
            export_matmuls: list[dict[str, torch.Tensor]] = []
            source_matmuls: list[dict[str, torch.Tensor]] = []
            def trace_matmuls(call, captured):
                original = torch.matmul
                def recording_matmul(left, right, *matmul_args, **matmul_kwargs):
                    result = original(left, right, *matmul_args, **matmul_kwargs)
                    captured.append({
                        "left": left.detach().clone(),
                        "right": right.detach().clone(),
                        "output": result.detach().clone(),
                    })
                    return result
                torch.matmul = recording_matmul
                try:
                    return call()
                finally:
                    torch.matmul = original
            def run_step():
                if args.softmax_accumulation == "source_fp32":
                    return step(*step_inputs)
                original = torch.softmax
                def promoted_softmax(scores, dim, *softmax_args, **softmax_kwargs):
                    return original(scores.double(), dim, *softmax_args,
                                    **softmax_kwargs).float()
                torch.softmax = promoted_softmax
                try:
                    return step(*step_inputs)
                finally:
                    torch.softmax = original
            try:
                if hooks:
                    out = trace_matmuls(run_step, export_matmuls)
                    ref = trace_matmuls(lambda: model(
                        torch.tensor([[current_token]]),
                        past_key_values=reference_cache,
                        cache_position=torch.tensor([decode_position]),
                        use_cache=True,
                        logits_to_keep=1,
                    ), source_matmuls)
                else:
                    out = run_step()
                    ref = model(
                        torch.tensor([[current_token]]),
                        past_key_values=reference_cache,
                        cache_position=torch.tensor([decode_position]),
                        use_cache=True,
                        logits_to_keep=1,
                    )
            finally:
                for hook in hooks:
                    hook.remove()
            if hooks:
                if len(export_hidden) != cfg.num_layers or len(source_hidden) != cfg.num_layers:
                    raise ValueError("trace did not capture every source/export layer")
                traced_layer_hidden = [
                    {"layer": i,
                     "relative_l2": relative_l2(source_hidden[i], export_hidden[i]),
                     "bitwise_equal": torch.equal(source_hidden[i], export_hidden[i]),
                     "max_abs_error": float((source_hidden[i].double()
                                             - export_hidden[i].double()).abs().max())}
                    for i in range(cfg.num_layers)
                ]
                if len(export_matmuls) != 2 * cfg.num_layers or len(source_matmuls) != 2 * cfg.num_layers:
                    raise ValueError("trace did not capture two attention matmuls per layer")
                for i, layer_type in enumerate(cfg.layer_types):
                    source_score = source_matmuls[2 * i]["output"]
                    export_score = export_matmuls[2 * i]["output"]
                    if layer_type == FULL and not args.right_align_full:
                        export_score = torch.cat(
                            (export_score[..., :decode_position], export_score[..., -1:]),
                            dim=-1,
                        )
                    elif layer_type == FULL and args.right_align_full:
                        export_score = export_score[..., -decode_position - 1:]
                    if export_score.shape != source_score.shape:
                        raise ValueError(f"trace score shape mismatch at layer {i}")
                    source_prob = source_matmuls[2 * i + 1]["left"]
                    export_prob = export_matmuls[2 * i + 1]["left"]
                    source_values = source_matmuls[2 * i + 1]["right"]
                    export_values = export_matmuls[2 * i + 1]["right"]
                    if layer_type == FULL and not args.right_align_full:
                        export_prob = torch.cat(
                            (export_prob[..., :decode_position], export_prob[..., -1:]),
                            dim=-1,
                        )
                        export_values = torch.cat(
                            (export_values[..., :decode_position, :],
                             export_values[..., -1:, :]),
                            dim=-2,
                        )
                    elif layer_type == FULL and args.right_align_full:
                        export_prob = export_prob[..., -decode_position - 1:]
                        export_values = export_values[..., -decode_position - 1:, :]
                    if (export_prob.shape != source_prob.shape
                            or export_values.shape != source_values.shape):
                        raise ValueError(f"trace probability/value shape mismatch at layer {i}")
                    source_value = source_matmuls[2 * i + 1]["output"]
                    export_value = export_matmuls[2 * i + 1]["output"]
                    traced_attention_matmuls.append({
                        "layer": i,
                        "layer_type": layer_type,
                        "score_relative_l2": relative_l2(source_score, export_score),
                        "score_bitwise_equal": torch.equal(source_score, export_score),
                        "probability_relative_l2": relative_l2(source_prob, export_prob),
                        "probability_bitwise_equal": torch.equal(source_prob, export_prob),
                        "value_input_relative_l2": relative_l2(source_values, export_values),
                        "value_input_bitwise_equal": torch.equal(source_values, export_values),
                        "value_relative_l2": relative_l2(source_value, export_value),
                        "value_bitwise_equal": torch.equal(source_value, export_value),
                    })
            ref_logits = ref.logits[0, -1].float()
            got_logits = out[0][0].float()
            if not torch.isfinite(ref_logits).all() or not torch.isfinite(got_logits).all():
                raise ValueError("non-finite decode logits")
            delta = got_logits - ref_logits
            next_ref = int(ref_logits.argmax())
            next_export = int(got_logits.argmax())
            cache_mismatch_layers = []
            max_new_kv_relative_l2 = 0.0
            worst_new_kv = None
            if args.export_arithmetic == "hf_bf16_reference":
                for i, layer in enumerate(reference_cache.layers):
                    if (not torch.equal(out[1 + 2 * i][:, :, -1:], layer.keys[:, :, -1:])
                            or not torch.equal(out[2 + 2 * i][:, :, -1:], layer.values[:, :, -1:])):
                        cache_mismatch_layers.append(i)
            for i, layer in enumerate(reference_cache.layers):
                for kind, reference, actual in (
                    ("key", layer.keys[:, :, -1:], out[1 + 2 * i][:, :, -1:]),
                    ("value", layer.values[:, :, -1:], out[2 + 2 * i][:, :, -1:]),
                ):
                    error = relative_l2(reference, actual)
                    if error > max_new_kv_relative_l2:
                        max_new_kv_relative_l2 = error
                        worst_new_kv = {
                            "layer": i,
                            "kind": kind,
                            "reference_l2_norm": float(torch.linalg.vector_norm(
                                reference.double())),
                            "candidate_l2_norm": float(torch.linalg.vector_norm(
                                actual.double())),
                            "max_abs_error": float((reference.double()
                                                    - actual.double()).abs().max()),
                        }
            steps.append({
                "position": decode_position,
                "input_token": current_token,
                "hf_next_token": next_ref,
                "export_next_token": next_export,
                "top1_match": next_ref == next_export,
                "logits_relative_l2": float(delta.norm() / ref_logits.norm()),
                "logits_max_abs": float(delta.abs().max()),
                "cache_mismatch_layers": cache_mismatch_layers,
                "max_new_kv_relative_l2": max_new_kv_relative_l2,
                "worst_new_kv": worst_new_kv,
            })
            state.commit(out, expected_position=decode_position)
            current_token = next_ref

    mismatch_positions = [s["position"] for s in steps if not s["top1_match"]]
    report = {
        "model": str(args.model),
        "model_index_sha256": sha256(args.model / "model.safetensors.index.json"),
        "model_config_sha256": sha256(args.model / "config.json"),
        "prompt_ids_sha256": hashlib.sha256(prompt_ids.numpy().tobytes()).hexdigest(),
        "probe_source_sha256": sha256(Path(__file__)),
        "export_source_sha256": sha256(Path(spark_export.__file__)),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cpu_threads": torch.get_num_threads(),
        "prompt_tokens": int(position),
        "decode_steps": args.decode_steps,
        "top1_matches": args.decode_steps - len(mismatch_positions),
        "mismatch_positions": mismatch_positions,
        "max_logits_relative_l2": max(s["logits_relative_l2"] for s in steps),
        "max_logits_abs": max(s["logits_max_abs"] for s in steps),
        "max_new_kv_relative_l2": max(s["max_new_kv_relative_l2"] for s in steps),
        "final_position": state.position,
        "full_cache_capacity": state.max_context,
        "initial_full_cache_capacity": initial_capacity,
        "admitted_full_cache_capacity": context_capacity,
        "full_bucket_width": args.full_bucket_width,
        "bucket_transitions": bucket_transitions,
        "sliding_cache_capacity": state.buffers[0].shape[2],
        "cache_shapes": cache_shapes,
        "weights_copied": copied,
        "dtype": (f"HF {args.reference_dtype}; step {args.export_arithmetic}; "
                  "same checkpoint weights"),
        "export_arithmetic": args.export_arithmetic,
        "attention_accumulation": args.attention_accumulation,
        "softmax_accumulation": args.softmax_accumulation,
        "cache_layout": args.cache_layout,
        "compact_inputs": args.compact_inputs,
        "compact_layer_type": args.compact_layer_type,
        "ordered_sliding": args.ordered_sliding,
        "right_align_full": args.right_align_full,
        "trace_position": args.trace_position,
        "traced_layer_hidden": traced_layer_hidden,
        "traced_attention_matmuls": traced_attention_matmuls,
        "steps": steps,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cache_shapes", "steps")}, indent=2))


if __name__ == "__main__":
    main()
