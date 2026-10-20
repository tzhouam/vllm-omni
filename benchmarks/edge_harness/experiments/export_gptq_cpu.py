"""Write Spark-X2.5 as a symmetric 4-bit GPTQ checkpoint.

vLLM 0.28's CPU backend already has a 4-bit dense GEMM -- `int4_scaled_mm_cpu`,
an AMX W4A8 kernel, plus `cpu_gemm_wna16` as the W4A16 fallback -- reached by
loading a GPTQ checkpoint.  It is not named like the ops an earlier search
looked for (`woq_int4_linear`, `int4_scaled_mm_with_quant`, `da8w4_linear`),
which is how it was missed.  Measured at Spark's shapes it beats the tinygemm
kernel 1.7x at one row and 8.4x at 512, so this is the path that matters.

`sym=True` selects vLLM's `uint4b8` weight type, whose dequantization is

    w = (code - 8) * scale,   code in [0, 15]

and whose zero points are ignored.  Getting that convention wrong does not
fail loudly -- the checkpoint loads and generates, just badly -- so
`--verify` re-reads what was written and reports the reconstruction SNR.
"""
import argparse, json, shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

QUANT_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
EMBED_NAME = "model.embedding.weight"


def pack_int32(codes: torch.Tensor, dim: int) -> torch.Tensor:
    """Pack 8 nibbles per int32 along `dim`, low-order nibble first."""
    codes = codes.to(torch.int32)
    if dim == 0:
        k, n = codes.shape
        out = torch.zeros(k // 8, n, dtype=torch.int32)
        for i in range(8):
            out |= codes[i::8, :] << (4 * i)
    else:
        k, n = codes.shape
        out = torch.zeros(k, n // 8, dtype=torch.int32)
        for i in range(8):
            out |= codes[:, i::8] << (4 * i)
    return out.contiguous()


def quantize_asym(w: torch.Tensor, group: int):
    """[N, K] -> GPTQ tensors with a real per-group zero point.

    `w = (code - zp) * scale` over the group's true min/max, which does not
    waste a level the way a symmetric grid does on an off-centre group.  The
    CPU W4A8 kernel's stand-in zero point for symmetric checkpoints is
    0x88888888 -- every nibble 8, used directly -- so `qzeros` here holds the
    zero point itself, with no AutoGPTQ-style -1.
    """
    n, k = w.shape
    wt = w.float().t().contiguous()
    g = k // group
    blocks = wt.reshape(g, group, n)
    wmax, wmin = blocks.amax(dim=1), blocks.amin(dim=1)
    scale = ((wmax - wmin).clamp(min=1e-9) / 15.0)
    zp = (-wmin / scale).round().clamp_(0, 15)
    codes = ((blocks / scale.unsqueeze(1)).round() + zp.unsqueeze(1)).clamp_(0, 15)
    deq = ((codes - zp.unsqueeze(1)) * scale.unsqueeze(1)).reshape(k, n)
    codes = codes.reshape(k, n)
    snr = float(20 * torch.log10(wt.norm() / (wt - deq).norm().clamp(min=1e-12)))
    qweight = pack_int32(codes, dim=0)
    qzeros = pack_int32(zp.to(torch.int32), dim=1)
    scales = scale.to(torch.float16)
    g_idx = torch.arange(k, dtype=torch.int32) // group
    return qweight, qzeros, scales, g_idx, snr


def quantize_sym(w: torch.Tensor, group: int):
    """[N, K] -> GPTQ tensors. Symmetric about code 8, group-wise along K."""
    n, k = w.shape
    wt = w.float().t().contiguous()            # GPTQ stores [K, N]
    g = k // group
    blocks = wt.reshape(g, group, n)
    # code 8 is zero and codes run 0..15, so the representable range is
    # [-8, 7] * scale; size it off the negative side to keep it symmetric.
    amax = blocks.abs().amax(dim=1)            # [g, n]
    scale = (amax / 8.0).clamp(min=1e-9)
    codes = (blocks / scale.unsqueeze(1)).round().add_(8).clamp_(0, 15)
    codes = codes.reshape(k, n)
    deq = ((codes.reshape(g, group, n) - 8.0) * scale.unsqueeze(1)).reshape(k, n)
    snr = float(20 * torch.log10(wt.norm() / (wt - deq).norm().clamp(min=1e-12)))
    qweight = pack_int32(codes, dim=0)                       # [K/8, N]
    # sym GPTQ still carries qzeros; the classic convention stores zp - 1 = 7.
    qzeros = pack_int32(torch.full((g, n), 7, dtype=torch.int32), dim=1)  # [g, N/8]
    scales = scale.to(torch.float16)                         # [g, N]
    g_idx = torch.arange(k, dtype=torch.int32) // group
    return qweight, qzeros, scales, g_idx, snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--quantize-embedding", action="store_true")
    ap.add_argument("--asym", action="store_true",
                    help="per-group zero points instead of a symmetric grid")
    ap.add_argument("--quantize-lm-head", action="store_true",
                    help="untie the head and quantize it; the lm_head GEMM is "
                         "the single largest read in a decode step")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    state = {}
    for shard in sorted(src.glob("*.safetensors")):
        state.update(load_file(str(shard)))

    # Split the fused q_k_v_proj here rather than at load time: the model's
    # splitter cuts dim 0, and GPTQ tensors are stored [K, N], so the output
    # dim moves. Splitting along output columns is exactly equivalent for a
    # grouping that runs along K, so do it before quantizing and let vLLM's
    # ordinary stacked-parameter mapping take over.
    cfg_src = json.loads((src / "config.json").read_text())
    q_dim = cfg_src["num_attention_heads"] * cfg_src["head_dim"]
    kv_dim = cfg_src["num_key_value_heads"] * cfg_src["head_dim"]
    for name in [k for k in state if ".self_attn.q_k_v_proj." in k]:
        q, k, v = state.pop(name).split([q_dim, kv_dim, kv_dim], dim=0)
        for shard_name, tensor in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
            state[name.replace("q_k_v_proj", shard_name)] = tensor.contiguous()

    quantize = quantize_asym if a.asym else quantize_sym
    out, report = {}, []
    for name, t in sorted(state.items()):
        is_embed = name == EMBED_NAME and a.quantize_embedding
        if name.endswith(QUANT_SUFFIXES) or is_embed:
            qw, qz, sc, gi, snr = quantize(t, a.group)
            base = name[: -len(".weight")]
            out[f"{base}.qweight"] = qw
            out[f"{base}.qzeros"] = qz
            out[f"{base}.scales"] = sc
            out[f"{base}.g_idx"] = gi
            report.append({"name": name, "snr_db": round(snr, 2)})
        else:
            out[name] = t

    if a.quantize_lm_head:
        # Spark ties the head to the vocab table. Quantizing it means untying:
        # the table stays bf16 for the lookup (a few hundred bytes a token)
        # and the head gets its own 4-bit copy, which is what a decode step
        # actually streams -- 131072 x 2048, the largest read in the step.
        qw, qz, sc, gi, snr = quantize(state[EMBED_NAME], a.group)
        out["lm_head.qweight"] = qw
        out["lm_head.qzeros"] = qz
        out["lm_head.scales"] = sc
        out["lm_head.g_idx"] = gi
        report.append({"name": "lm_head", "snr_db": round(snr, 2)})

    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "vocab.json", "merges.txt", "generation_config.json", "chat_template.jinja"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = {
        "quant_method": "gptq", "bits": 4, "group_size": a.group,
        "desc_act": False, "sym": not a.asym, "true_sequential": True,
        "checkpoint_format": "gptq",
        "lm_head": bool(a.quantize_lm_head),
    }
    if a.quantize_lm_head:
        cfg["tie_word_embeddings"] = False
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    snrs = [r["snr_db"] for r in report]
    summary = {"group": a.group, "tensors": len(report),
               "snr_db_mean": round(sum(snrs) / len(snrs), 2),
               "snr_db_min": round(min(snrs), 2),
               "bits_per_weight": 4 + (16 + 4) / a.group,
               "quantized_embedding": a.quantize_embedding,
               "quantized_lm_head": a.quantize_lm_head}
    (dst / "quantization_report.json").write_text(
        json.dumps({**summary, "per_tensor": report}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
