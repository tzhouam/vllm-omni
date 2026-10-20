"""Write a GPTQ-format W4A8 checkpoint for Spark-X2.5.

vLLM 0.28's CPU backend *does* have a 4-bit kernel -- it is just not reachable
under the names a quantization search turns up.  `CPUWNA16LinearKernel`
(`vllm/model_executor/kernels/linear/mixed_precision/cpu.py`) dispatches GPTQ
and AWQ checkpoints to `int4_scaled_mm_cpu`, an AMX W4A8 GEMM, whenever the
activations are bf16, the ISA has AMX tiles and the checkpoint has no
activation reordering.  Four-bit weights times eight-bit activations is
exactly the arithmetic llama.cpp's Q4_K_M does (Q4_K weights, Q8_K
activations), so this is the honest precision match.

The kernel pins the zero point at 8 regardless of what the checkpoint says
(`_process_gptq_weights_w4a8` builds its own), so the grid has to be
symmetric: w = (code - 8) * scale.  `--search` sweeps the scale per group and
keeps the lowest squared error, which recovers most of what symmetry costs
against an asymmetric min/max grid.

Emitted per quantized linear, in the layout `AutoGPTQLinearMethod` creates:

    <name>.qweight  int32 [K // 8, N]   4-bit codes packed along the input dim
    <name>.qzeros   int32 [K // g, N // 8]
    <name>.scales   bf16  [K // g, N]
    <name>.g_idx    int32 [K]

Tied embeddings are untied here: the head is quantized (it is the single
largest read in a decode step) while the vocab table stays bf16, since an
embedding lookup reads one row and gains nothing from being 4-bit.
"""
import argparse, json, shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

from vllm_omni.model_executor.layers.quantization.gptq_cpu_export import gptq_tensors

# Split the fused projection here rather than in the model: the GPTQ layout
# transposes N to dim 1, and the loader's stacked mapping already knows how to
# put q/k/v back together.
FUSED_QKV = "self_attn.q_k_v_proj.weight"
QUANT_SUFFIXES = (
    "self_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
EMBED_NAME = "model.embedding.weight"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--no-lm-head", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    state = {}
    for shard in sorted(src.glob("*.safetensors")):
        state.update(load_file(str(shard)))

    cfg = json.loads((src / "config.json").read_text())
    q_dim = cfg["num_attention_heads"] * cfg["head_dim"]
    kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]

    def emit(out, base, w, report):
        qw, qz, sc, gi, snr = gptq_tensors(w, a.group, not a.no_search)
        out[f"{base}.qweight"] = qw
        out[f"{base}.qzeros"] = qz
        out[f"{base}.scales"] = sc
        out[f"{base}.g_idx"] = gi
        report.append({"name": base, "shape": list(w.shape), "snr_db": round(snr, 2)})

    out, report = {}, []
    for name, t in sorted(state.items()):
        if name.endswith(FUSED_QKV):
            base = name[: -len(".weight")].replace("q_k_v_proj", "")
            for shard, part in zip(("q_proj", "k_proj", "v_proj"),
                                   t.split([q_dim, kv_dim, kv_dim], dim=0)):
                emit(out, base + shard, part, report)
        elif name.endswith(QUANT_SUFFIXES):
            emit(out, name[: -len(".weight")], t, report)
        elif name == EMBED_NAME:
            out[name] = t
            if not a.no_lm_head:
                emit(out, "lm_head", t, report)
        else:
            out[name] = t

    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})

    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "vocab.json", "merges.txt", "generation_config.json",
              "chat_template.jinja"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    # The head is quantized separately from the table it used to be tied to,
    # so the tie has to come off or vLLM would skip loading lm_head.*
    cfg["tie_word_embeddings"] = a.no_lm_head
    cfg["quantization_config"] = {
        "quant_method": "gptq", "bits": 4, "group_size": a.group,
        "desc_act": False, "sym": True, "lm_head": not a.no_lm_head,
        "checkpoint_format": "gptq",
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    snrs = [r["snr_db"] for r in report]
    summary = {
        "group": a.group, "search": not a.no_search,
        "tensors_quantized": len(report),
        "snr_db_min": round(min(snrs), 2), "snr_db_max": round(max(snrs), 2),
        "snr_db_mean": round(sum(snrs) / len(snrs), 2),
        "bits_per_weight": round(4 + 16 / a.group + 4 / a.group, 3),
        "per_tensor": report,
    }
    (dst / "quantization_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_tensor"}, indent=2))


if __name__ == "__main__":
    main()
