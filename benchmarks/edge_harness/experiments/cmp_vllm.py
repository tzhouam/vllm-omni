"""Compare the vLLM-Omni Spark2_5 implementation against ref.json."""
import argparse, json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="ref.json")
    ap.add_argument("--ref-logits", default="ref_logits.npy")
    ap.add_argument("--model", default="XHToken/Spark-X2.5-1.7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out", default="parity.json")
    a = ap.parse_args()

    ref = json.load(open(a.ref))
    ref_logits = np.load(a.ref_logits)

    import vllm_omni  # noqa: F401  registers Spark2_5ForCausalLM
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype=a.dtype,
              max_model_len=a.max_model_len, enforce_eager=True,
              max_num_seqs=1, trust_remote_code=False)
    n_new = len(ref["new_ids"])
    sp = SamplingParams(temperature=0.0, max_tokens=n_new, logprobs=20)
    out = llm.generate([{"prompt_token_ids": ref["prompt_ids"]}], sp)[0]
    v_ids = list(out.outputs[0].token_ids)

    n = min(len(ref["new_ids"]), len(v_ids))
    match = sum(int(x == y) for x, y in zip(ref["new_ids"][:n], v_ids[:n]))
    # first-step logprob agreement on the reference top-20
    lp0 = out.outputs[0].logprobs[0]
    ref_top = np.argsort(-ref_logits)[:20]
    ref_lse = np.log(np.exp(ref_logits - ref_logits.max()).sum()) + ref_logits.max()
    pairs = [(int(t), float(ref_logits[t] - ref_lse), lp0[t].logprob)
             for t in ref_top if t in lp0]
    d = [abs(r - v) for _, r, v in pairs]
    rep = {
        "ref_text": ref["text"], "vllm_text": out.outputs[0].text,
        "greedy_token_match": f"{match}/{n}",
        "exact_match": ref["new_ids"][:n] == v_ids[:n],
        "ref_top1": ref["top1"], "vllm_first_token": v_ids[0] if v_ids else None,
        "first_token_agree": bool(v_ids and v_ids[0] == ref["top1"]),
        "top20_logprob_overlap": f"{len(pairs)}/20",
        "top20_logprob_max_abs_diff": max(d) if d else None,
        "top20_logprob_mean_abs_diff": float(np.mean(d)) if d else None,
        "dtype": a.dtype,
    }
    json.dump(rep, open(a.out, "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
