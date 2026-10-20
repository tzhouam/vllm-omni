"""pp512 / tg128 for Spark-X2.5 on one GPU, for vLLM or SGLang.

Same protocol as the CPU grid and llama-bench: a 512-token prefill measured on
its own, then 128 generated tokens with the one-token prefill subtracted.
Prefix caching is disabled on both engines so a replayed prompt is actually
recomputed.
"""
import argparse, json, os, time


def bench_vllm(a):
    import vllm_omni  # noqa: F401  registers Spark2_5ForCausalLM
    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.max_model_len,
              max_num_seqs=1, enable_prefix_caching=False,
              gpu_memory_utilization=0.85, trust_remote_code=False,
              disable_log_stats=True)
    init = time.perf_counter() - t0
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    spg = SamplingParams(temperature=0.0, max_tokens=a.tg, ignore_eos=True)
    pp_prompt = {"prompt_token_ids": list(range(10, 10 + a.pp))}
    tg_prompt = {"prompt_token_ids": [10]}

    def run(p, sp):
        s = time.perf_counter()
        llm.generate([p], sp, use_tqdm=False)
        return time.perf_counter() - s

    run(pp_prompt, sp1)
    pp = min(run(pp_prompt, sp1) for _ in range(a.reps))
    tg = min(run(tg_prompt, spg) for _ in range(a.reps))
    tg1 = min(run(tg_prompt, sp1) for _ in range(a.reps))
    return init, pp, tg, tg1


def bench_sglang(a):
    import sglang as sgl
    t0 = time.perf_counter()
    llm = sgl.Engine(model_path=a.model, dtype=a.dtype,
                     context_length=a.max_model_len,
                     disable_radix_cache=True, mem_fraction_static=0.85,
                     disable_cuda_graph=False, log_level="error")
    init = time.perf_counter() - t0
    pp_ids = list(range(10, 10 + a.pp))

    def run(ids, n):
        s = time.perf_counter()
        llm.generate(input_ids=[ids],
                     sampling_params={"temperature": 0.0, "max_new_tokens": n,
                                      "ignore_eos": True})
        return time.perf_counter() - s

    run(pp_ids, 1)
    pp = min(run(pp_ids, 1) for _ in range(a.reps))
    tg = min(run([10], a.tg) for _ in range(a.reps))
    tg1 = min(run([10], 1) for _ in range(a.reps))
    llm.shutdown()
    return init, pp, tg, tg1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--model", default="XHToken/Spark-X2.5-1.7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out", default="gpu_bench.jsonl")
    a = ap.parse_args()

    init, pp, tg, tg1 = (bench_vllm if a.engine == "vllm" else bench_sglang)(a)
    res = {
        "engine": a.engine, "dtype": a.dtype, "gpu": os.environ.get("GPU_NAME", ""),
        "init_s": round(init, 2),
        "pp_tokens": a.pp, "pp_s": round(pp, 4), "pp_tps": round(a.pp / pp, 2),
        "tg_tokens": a.tg, "tg_s": round(tg - tg1, 4),
        "tg_tps": round((a.tg - 1) / max(tg - tg1, 1e-9), 2),
    }
    print(json.dumps(res, indent=2))
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
