#!/usr/bin/env python3
"""Capture pinned HF BF16 block-boundary outputs for Spark Hub comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--token-input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    hf_config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    )
    model_class = get_class_from_dynamic_module(
        hf_config.auto_map["AutoModelForCausalLM"], str(args.model),
    )

    def mask_compat(mask_function):
        def call(**kwargs):
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            kwargs.pop("cache_position", None)
            return mask_function(**kwargs)
        return call

    source_module = sys.modules[model_class.__module__]
    source_module.create_causal_mask = mask_compat(create_causal_mask)
    source_module.create_sliding_window_causal_mask = mask_compat(
        create_sliding_window_causal_mask
    )
    if isinstance(getattr(model_class, "_tied_weights_keys", None), list):
        model_class._tied_weights_keys = {
            "lm_head.weight": "model.embedding.weight",
        }
    model = model_class.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True,
    ).eval()
    with np.load(args.token_input, allow_pickle=False) as archive:
        token_id = int(archive["token_id"].item())
    captured: dict[int, np.ndarray] = {}
    hooks = []
    for block, end_layer in enumerate((1, 5, 9, 13, 17, 21, 25)):
        hooks.append(model.model.layers[end_layer].register_forward_hook(
            lambda _module, _inputs, output, index=block:
            captured.__setitem__(index, output.detach().float().numpy().copy())
        ))
    try:
        with torch.inference_mode():
            output = model(torch.tensor([[token_id]]), use_cache=False, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
    if len(captured) != 7:
        raise ValueError("did not capture seven hidden block boundaries")
    tensors = {f"block{index}": value for index, value in captured.items()}
    tensors["logits"] = output.logits[0, -1].float().numpy()[None, :]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **tensors)
    metadata = {
        "checkpoint": str(args.model),
        "checkpoint_revision": "14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8",
        "config_sha256": sha256(args.model / "config.json"),
        "index_sha256": sha256(args.model / "model.safetensors.index.json"),
        "token_id": token_id,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "dtype": "BF16 model; saved FP32 boundary tensors",
        "output_file_sha256": sha256(args.out),
        "output_shapes": {name: list(value.shape) for name, value in tensors.items()},
        "reference_top1": int(np.argmax(tensors["logits"])),
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"reference_top1": metadata["reference_top1"],
                      "output_sha256": metadata["output_file_sha256"]}))


if __name__ == "__main__":
    main()
