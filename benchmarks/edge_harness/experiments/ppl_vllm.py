"""Teacher-forced perplexity of a text file under a vLLM model.

Quantization is scored against the same engine's own bf16 run, not against
llama.cpp's: each engine tokenizes and chunks its own way, and comparing the
*increase* over an in-engine baseline cancels that, where comparing absolute
perplexity across engines would not.
"""
import argparse, json, math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="ppl.jsonl")
    a = ap.parse_args()

    import vllm_omni  # noqa: F401
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.window + 8,
              max_num_seqs=1, enable_prefix_caching=False,
              trust_remote_code=False, disable_log_stats=True)
    tok = llm.get_tokenizer()
    ids = tok(open(a.corpus).read(), add_special_tokens=False)["input_ids"]

    chunks = [ids[i:i + a.window] for i in range(0, len(ids) - a.window + 1, a.window)]
    if a.max_chunks:
        chunks = chunks[: a.max_chunks]

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0,
                        ignore_eos=True, detokenize=False)
    outs = llm.generate([{"prompt_token_ids": c} for c in chunks], sp, use_tqdm=False)

    total_nll, n = 0.0, 0
    for o, c in zip(outs, chunks):
        pl = o.prompt_logprobs
        # position 0 has no prediction behind it
        for pos in range(1, len(c)):
            entry = pl[pos]
            lp = entry[c[pos]].logprob
            total_nll -= lp
            n += 1
    ppl = math.exp(total_nll / n)
    rec = {"tag": a.tag or a.model, "model": a.model, "window": a.window,
           "chunks": len(chunks), "scored_tokens": n,
           "nll": total_nll / n, "ppl": ppl}
    print(json.dumps(rec, indent=2))
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
