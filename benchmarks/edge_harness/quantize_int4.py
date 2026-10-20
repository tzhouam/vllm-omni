"""Write a `cpu_int4` W4A16 checkpoint for Spark-X2.5.

Group-wise 4-bit weights at the same bits-per-weight budget as llama.cpp's
Q4_K_M (4.5 at group 64), so the CPU decode comparison is precision-matched
rather than a quantization contest.

Round-to-nearest over the group's min/max is the obvious grid, but it is not
the best one: an outlier stretches the range and costs every other weight in
the group.  `--search` sweeps a shrink factor per group and keeps the one with
the lowest squared error, which is the same idea as llama.cpp's
`make_qkx2_quants` and buys about a dB.

The head-wise gate (g_proj) stays in full precision: eight outputs that each
scale a whole attention head, 0.001% of the weights.
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
EMBED_NAME = "model.embedding.weight"
SHRINKS = (1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.85)


def _pack(q: torch.Tensor) -> torch.Tensor:
    q = q.to(torch.uint8)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()


def quantize(w: torch.Tensor, group: int, search: bool):
    n, k = w.shape
    wf = w.float().reshape(n, k // group, group)
    wmin, wmax = wf.amin(dim=-1), wf.amax(dim=-1)
    span = (wmax - wmin).clamp(min=1e-9)

    best_err = None
    best = None
    for f in (SHRINKS if search else (1.0,)):
        # Shrink the grid symmetrically about the group mid-point: the
        # extremes lose a little, everything between them gains.
        scale = span * f / 15.0
        lo = wmin + span * (1.0 - f) * 0.5
        q = ((wf - lo.unsqueeze(-1)) / scale.unsqueeze(-1)).round().clamp_(0, 15)
        err = ((q * scale.unsqueeze(-1) + lo.unsqueeze(-1)) - wf).pow_(2).sum(-1)
        if best_err is None:
            best_err, best = err, (q, scale, lo)
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            t = take.unsqueeze(-1)
            best = (
                torch.where(t, q, best[0]),
                torch.where(take, scale, best[1]),
                torch.where(take, lo, best[2]),
            )
    q, scale, lo = best
    zero = lo + 8.0 * scale
    deq = (q - 8.0) * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    ref = wf
    snr = float(20 * torch.log10(ref.norm() / (ref - deq).norm().clamp(min=1e-12)))
    return _pack(q.reshape(n, k)), scale.to(torch.bfloat16), zero.to(torch.bfloat16), snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--keep-embedding", action="store_true",
                    help="leave the tied vocab table in bf16")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    state = {}
    for shard in sorted(src.glob("*.safetensors")):
        state.update(load_file(str(shard)))

    targets = list(QUANT_SUFFIXES)
    out, report = {}, []
    for name, t in sorted(state.items()):
        is_embed = name == EMBED_NAME and not a.keep_embedding
        if name.endswith(tuple(targets)) or is_embed:
            packed, scale, zero, snr = quantize(t, a.group, not a.no_search)
            base = name[: -len(".weight")]
            out[f"{base}.weight_packed"] = packed
            out[f"{base}.weight_scale"] = scale
            out[f"{base}.weight_zero"] = zero
            report.append({"name": name, "shape": list(t.shape), "snr_db": round(snr, 2)})
        else:
            out[name] = t

    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})

    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "vocab.json", "merges.txt", "generation_config.json",
              "chat_template.jinja"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = {
        "quant_method": "cpu_int4",
        "bits": 4,
        "group_size": a.group,
        "quantize_embedding": not a.keep_embedding,
        "prefill_dequant": True,
        # Only single-row steps may take the tinygemm kernel; see the AMX
        # tile-state note in vllm_omni/.../cpu_int4.py.
        "dequant_threshold": 2,
        "ignore": [],
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    snrs = [r["snr_db"] for r in report]
    summary = {
        "group": a.group, "search": not a.no_search,
        "tensors_quantized": len(report),
        "snr_db_min": round(min(snrs), 2), "snr_db_max": round(max(snrs), 2),
        "snr_db_mean": round(sum(snrs) / len(snrs), 2),
        "bits_per_weight": 4 + 2 * 16 / a.group,
        "per_tensor": report,
    }
    (dst / "quantization_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_tensor"}, indent=2))


if __name__ == "__main__":
    main()
