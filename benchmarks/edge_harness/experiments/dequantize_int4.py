"""Expand a `cpu_int4` checkpoint back to a plain bf16 one.

Used as a control: if the engine is applying the quantized weights faithfully,
the expanded checkpoint must score exactly the same perplexity as the
quantized one, and any difference is the runtime's, not the quantizer's.
"""
import argparse, json, shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from vllm_omni.model_executor.layers.quantization.cpu_int4 import unpack_nibbles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    group = cfg["quantization_config"]["group_size"]
    cfg.pop("quantization_config", None)

    state = {}
    for shard in sorted(src.glob("*.safetensors")):
        state.update(load_file(str(shard)))

    out = {}
    for name, t in state.items():
        if name.endswith(".weight_packed"):
            base = name[: -len(".weight_packed")]
            codes = unpack_nibbles(t).float()
            n, k = codes.shape
            s = state[base + ".weight_scale"].float()
            z = state[base + ".weight_zero"].float()
            deq = ((codes.reshape(n, k // group, group) - 8.0) * s.unsqueeze(-1)
                   + z.unsqueeze(-1)).reshape(n, k)
            out[base + ".weight"] = deq.to(torch.bfloat16)
        elif name.endswith((".weight_scale", ".weight_zero")):
            continue
        else:
            out[name] = t

    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "vocab.json", "merges.txt", "generation_config.json", "chat_template.jinja"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    print(json.dumps({"tensors": len(out), "group": group, "dst": str(dst)}))


if __name__ == "__main__":
    main()
