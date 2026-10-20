"""Write a compressed-tensors W8A8-int8 checkpoint for Spark-X2.5.

Done directly from the safetensors rather than through llmcompressor: that
package pulls transformers 5.x, on which Spark's remote modelling code does
not import, and symmetric per-channel round-to-nearest needs no calibration
data anyway.

The head-wise gate (g_proj) is deliberately left in full precision. It is a
2048x8 projection whose eight outputs each scale a whole attention head, so
quantisation error there is multiplied across 256 dims -- and it is 0.001% of
the weights, so leaving it out costs nothing.
"""
import argparse, json, shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

QUANT_SUFFIXES = (
    "self_attn.q_k_v_proj.weight",
    "self_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def quantize_channelwise(w: torch.Tensor):
    """Symmetric per-output-channel int8."""
    w = w.to(torch.float32)
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.clamp(scale, min=1e-12)
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    state = {}
    for shard in sorted(src.glob("*.safetensors")):
        state.update(load_file(str(shard)))

    out, n_q, err = {}, 0, []
    for name, t in state.items():
        if name.endswith(QUANT_SUFFIXES):
            q, s = quantize_channelwise(t)
            out[name] = q
            out[name + "_scale"] = s
            deq = q.to(torch.float32) * s
            ref = t.to(torch.float32)
            err.append(float(
                20 * torch.log10(ref.norm() / (ref - deq).norm().clamp(min=1e-12))
            ))
            n_q += 1
        else:
            out[name] = t

    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})

    for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt",
              "generation_config.json", "chat_template.jinja"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    cfg = json.loads((dst / "config.json").read_text())
    cfg.pop("auto_map", None)  # the vendored vLLM config/model handle this arch
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head", "re:.*g_proj.*"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8, "type": "int", "symmetric": True,
                    "strategy": "channel", "dynamic": False,
                    "observer": "minmax",
                },
                "input_activations": {
                    "num_bits": 8, "type": "int", "symmetric": True,
                    "strategy": "token", "dynamic": True, "observer": None,
                },
            }
        },
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({
        "quantized_tensors": n_q,
        "weight_snr_db_min": round(min(err), 2),
        "weight_snr_db_mean": round(sum(err) / len(err), 2),
        "dst": str(dst),
    }, indent=2))


main()
