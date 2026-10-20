"""Aggregate decode throughput across concurrent sequences (vLLM, CPU).

Mirrors `llama-batched-bench`: B sequences each with a short prompt generate
N tokens; the figure is total generated tokens per second.
"""
import argparse, json, os, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="XHToken/Spark-X2.5-1.7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--tag", default="vllm")
    ap.add_argument("--out", default="batch_bench.jsonl")
    a = ap.parse_args()

    import vllm_omni  # noqa: F401
    from vllm import LLM, SamplingParams
    batches = [int(b) for b in a.batches.split(",")]
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=1024,
              max_num_seqs=max(batches), enable_prefix_caching=False,
              trust_remote_code=False, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)
    res = {}
    for b in batches:
        prompts = [{"prompt_token_ids": list(range(10 + i * 7, 10 + i * 7 + a.prompt))}
                   for i in range(b)]
        llm.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=1,
                                                 ignore_eos=True), use_tqdm=False)
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        res[b] = round(b * a.gen / dt, 2)
        print(f"batch {b:>3}: {res[b]:>8.2f} gen tok/s total", flush=True)
    row = {"tag": a.tag, "threads": os.environ.get("OMP_NUM_THREADS", ""),
           "gen": a.gen, "prompt": a.prompt, "tok_s_by_batch": res}
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
