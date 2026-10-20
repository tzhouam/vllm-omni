"""A fidelity metric that does not saturate.

Greedy divergence over 12 prompts -- "did the quantized build reproduce the
bf16 output exactly?" -- answers 2/12 for W4A16 and 1/12 for W4A8 and cannot
say whether the gap between them is small or enormous. Everything past the
first divergent token is thrown away, and one unlucky token near position 3
scores the same as a build that is wrong everywhere.

This measures the distribution instead, teacher-forced over a corpus, so every
position contributes:

* **top-1 agreement** -- how often the quantized model's argmax matches the
  reference's. This is what actually decides whether greedy decoding diverges.
* **KL(reference || candidate)** -- the same quantity ``llama-perplexity
  --kl-divergence`` reports, so llama.cpp's Q4_K_M stays comparable against
  our builds rather than being compared on a different metric.
* **Delta log-prob of the reference token**, which is the cross-entropy
  difference and is what perplexity is built from.

One honest limitation, stated because it changes how the number should be
read: vLLM returns the top-k log-probs per position, not the full
distribution, so KL is computed over the union of the two top-k sets with the
remaining mass as a single tail bucket. ``coverage`` reports the reference
probability mass that fell inside the top-k, and a run whose coverage is low
is reporting a lower bound, not a KL. llama.cpp stores full logits and has no
such limitation; raise ``--topk`` (and vLLM's ``--max-logprobs``) to close the
gap when the two are compared directly.
"""

from __future__ import annotations

import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_harness import provenance  # noqa: E402


def chunk_tokens(token_ids: list[int], size: int, limit: int | None) -> list[list[int]]:
    chunks = [token_ids[i:i + size] for i in range(0, len(token_ids), size)]
    chunks = [c for c in chunks if len(c) >= 16]
    return chunks[:limit] if limit else chunks


def collect(llm, chunks, topk: int):
    """Teacher-forced top-k log-probs at every position of every chunk."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=topk)
    outs = llm.generate([{"prompt_token_ids": c} for c in chunks], sp,
                        use_tqdm=False)
    positions = []
    for out, chunk in zip(outs, chunks):
        for i, entry in enumerate(out.prompt_logprobs or []):
            if entry is None:          # first token has no prediction
                continue
            positions.append({
                "target": chunk[i],
                "logprobs": {int(tid): float(lp.logprob) for tid, lp in entry.items()},
            })
    return positions


def _top1(d: dict[int, float]) -> int:
    return max(d.items(), key=lambda kv: kv[1])[0]


def compare(ref: list[dict], cand: list[dict]) -> dict:
    if len(ref) != len(cand):
        raise SystemExit(f"position count differs: {len(ref)} vs {len(cand)}; "
                         f"the two runs did not see the same corpus")
    n = len(ref)
    agree = 0
    kls, dlps, coverages = [], [], []
    for r, c in zip(ref, cand):
        rl, cl = r["logprobs"], c["logprobs"]
        if _top1(rl) == _top1(cl):
            agree += 1

        # Reference distribution over its own top-k, plus a tail bucket.
        rp = {t: math.exp(lp) for t, lp in rl.items()}
        covered = sum(rp.values())
        coverages.append(min(covered, 1.0))
        tail_r = max(0.0, 1.0 - covered)

        # Candidate probability for the same tokens. A token outside the
        # candidate's top-k is bounded by its smallest returned probability,
        # which overestimates q and therefore underestimates KL -- the reason
        # this is reported as a lower bound when coverage is poor.
        floor = math.exp(min(cl.values())) if cl else 1e-12
        cq = {t: math.exp(cl[t]) if t in cl else floor for t in rp}
        covered_c = sum(cq.values())
        tail_c = max(1e-12, 1.0 - min(covered_c, 1.0 - 1e-12))

        kl = sum(p * math.log(p / max(cq[t], 1e-12)) for t, p in rp.items() if p > 0)
        if tail_r > 0:
            kl += tail_r * math.log(tail_r / tail_c)
        kls.append(max(kl, 0.0))

        target = r["target"]
        if target in rl and target in cl:
            dlps.append(cl[target] - rl[target])

    kls.sort()
    return {
        "positions": n,
        "top1_agreement": round(agree / n, 4) if n else None,
        "kl_mean": round(sum(kls) / len(kls), 6) if kls else None,
        "kl_median": round(kls[len(kls) // 2], 6) if kls else None,
        "kl_p99": round(kls[int(len(kls) * 0.99)], 6) if kls else None,
        "delta_logprob_mean": round(sum(dlps) / len(dlps), 6) if dlps else None,
        "delta_logprob_n": len(dlps),
        "coverage_mean": round(sum(coverages) / len(coverages), 4) if coverages else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--corpus", required=True, help="UTF-8 text file")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer to chunk with; defaults to --model")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--save-ref", default=None,
                    help="write this run's distributions as the reference")
    ap.add_argument("--ref", default=None,
                    help="reference file to compare against")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.save_ref and not a.ref:
        ap.error("give --save-ref (to record a reference) or --ref (to compare)")

    import vllm_omni  # noqa: F401
    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model,
                                        trust_remote_code=False)
    text = open(a.corpus, encoding="utf-8").read()
    chunks = chunk_tokens(tok(text).input_ids, a.chunk, a.max_chunks)
    if not chunks:
        raise SystemExit("corpus too short for one chunk")

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              max_num_seqs=1, quantization=a.quantization,
              enable_prefix_caching=False, trust_remote_code=False,
              disable_log_stats=True, max_logprobs=a.topk)
    positions = collect(llm, chunks, a.topk)

    meta = {"label": a.label, "model": a.model, "quantization": a.quantization,
            "corpus": os.path.abspath(a.corpus), "chunk": a.chunk,
            "n_chunks": len(chunks), "topk": a.topk,
            "provenance": provenance({"metric": "fidelity_kl"})}

    if a.save_ref:
        with open(a.save_ref, "w") as f:
            json.dump({**meta, "positions": positions}, f)
        print(f"wrote reference: {len(positions)} positions -> {a.save_ref}")
        result = {**meta, "role": "reference", "n_positions": len(positions)}
    else:
        ref = json.loads(open(a.ref).read())
        if ref["chunk"] != a.chunk or ref["n_chunks"] != len(chunks):
            raise SystemExit("reference was taken on a different corpus slicing")
        result = {**meta, "role": "candidate",
                  "reference_model": ref["model"],
                  **compare(ref["positions"], positions)}
        print(json.dumps({k: v for k, v in result.items()
                          if k != "provenance"}, indent=2))

    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
