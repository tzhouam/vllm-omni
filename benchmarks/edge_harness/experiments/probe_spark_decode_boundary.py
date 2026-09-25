# SPDX-License-Identifier: Apache-2.0
"""Compare the real Spark decode-step export boundary with HF cached decode.

This is a CPU numerical probe, not a device-local generation benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prompt", default="What is the capital of France? Answer in one word.")
    parser.add_argument("--prefill-tokens", type=int, default=0)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--reference-dtype", choices=("bfloat16", "float32"),
                        default="bfloat16")
    args = parser.parse_args()
    if args.decode_steps < 1:
        raise ValueError("decode-steps must be positive")

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
        cfg = config_from_spark(raw_config)
        step = SparkDecodeStep(cfg).eval()
        copied = load_spark_weights(step, args.model)
        position = int(prompt_ids.shape[1])
        state = SparkDecodeCache(
            cfg, max(1024, position + args.decode_steps + 1)
        )
        state.seed(cached_tensors, position)
        steps: list[dict] = []
        for _ in range(args.decode_steps):
            decode_position = state.position
            x = model.model.embedding(torch.tensor([[current_token]])).float()
            out = step(*state.step_inputs(x))
            ref = model(
                torch.tensor([[current_token]]),
                past_key_values=reference_cache,
                cache_position=torch.tensor([decode_position]),
                use_cache=True,
                logits_to_keep=1,
            )
            ref_logits = ref.logits[0, -1].float()
            got_logits = out[0][0].float()
            if not torch.isfinite(ref_logits).all() or not torch.isfinite(got_logits).all():
                raise ValueError("non-finite decode logits")
            delta = got_logits - ref_logits
            next_ref = int(ref_logits.argmax())
            next_export = int(got_logits.argmax())
            steps.append({
                "position": decode_position,
                "input_token": current_token,
                "hf_next_token": next_ref,
                "export_next_token": next_export,
                "top1_match": next_ref == next_export,
                "logits_relative_l2": float(delta.norm() / ref_logits.norm()),
                "logits_max_abs": float(delta.abs().max()),
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
        "final_position": state.position,
        "full_cache_capacity": state.max_context,
        "sliding_cache_capacity": cfg.sliding_window - 1,
        "cache_shapes": cache_shapes,
        "weights_copied": copied,
        "dtype": f"HF {args.reference_dtype}; export FP32 from the same checkpoint weights",
        "steps": steps,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cache_shapes", "steps")}, indent=2))


if __name__ == "__main__":
    main()
