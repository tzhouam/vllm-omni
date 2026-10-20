"""pp512 / tg128 for vLLM on CPU, mirroring `llama-bench` semantics.

pp512: time to process a 512-token prompt (prefill throughput).
tg128: time to generate 128 tokens (decode throughput), prefill subtracted.
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_harness import provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="XHToken/Spark-X2.5-1.7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--tag", default="vllm")
    ap.add_argument("--no-detokenize", action="store_true",
                    help="skip incremental detokenization, as llama-bench does")
    ap.add_argument("--compilation-config", default=None,
                    help="JSON passed straight to LLM(compilation_config=...)")
    ap.add_argument("--out", default="vllm_cpu_bench.json")
    a = ap.parse_args()

    import vllm_omni  # noqa: F401
    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    # Prefix caching must be off: the pp measurement replays one prompt, and a
    # cache hit would report a prefill that never happened. llama-bench has no
    # such cache, so leaving it on would not be a comparison.
    extra = {}
    if a.compilation_config:
        extra["compilation_config"] = json.loads(a.compilation_config)
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.max_model_len,
              enforce_eager=a.enforce_eager, max_num_seqs=1,
              quantization=a.quantization, enable_prefix_caching=False,
              trust_remote_code=False, disable_log_stats=True, **extra)
    init_s = time.perf_counter() - t0

    def rss_mb(field):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field):
                    return round(int(line.split()[1]) / 1024, 1)
        return None

    prompt_pp = {"prompt_token_ids": list(range(10, 10 + a.pp))}
    prompt_tg = {"prompt_token_ids": [10]}
    det = not a.no_detokenize
    sp_pre = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True,
                            detokenize=det)
    sp_gen = SamplingParams(temperature=0.0, max_tokens=a.tg, ignore_eos=True,
                            detokenize=det)

    def timed(prompt, sp, reps):
        best = []
        for _ in range(reps):
            s = time.perf_counter()
            llm.generate([prompt], sp, use_tqdm=False)
            best.append(time.perf_counter() - s)
        return best

    timed(prompt_pp, sp_pre, 1)  # warmup

    pp_t = timed(prompt_pp, sp_pre, a.reps)
    tg_t = timed(prompt_tg, sp_gen, a.reps)
    tg1_t = timed(prompt_tg, sp_pre, a.reps)  # 1-token prefill overhead

    pp_best, tg_best, tg1_best = min(pp_t), min(tg_t), min(tg1_t)
    res = {
        "tag": a.tag, "model": a.model, "dtype": a.dtype,
        "compilation_config": a.compilation_config,
        "detokenize": not a.no_detokenize,
        "quantization": a.quantization,
        "enforce_eager": a.enforce_eager, "init_s": round(init_s, 2),
        # Which build, which threads, which commits -- a throughput number
        # without these cannot be invalidated later, only doubted.
        "provenance": provenance({
            "model_path": os.path.realpath(a.model) if os.path.exists(a.model)
            else a.model,
            "pp": a.pp, "tg": a.tg, "reps": a.reps,
            "max_model_len": a.max_model_len,
        }),
        "rss_mb": rss_mb("VmRSS:"), "rss_peak_mb": rss_mb("VmHWM:"),
        "threads_bind": os.environ.get("VLLM_CPU_OMP_THREADS_BIND", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "pp_tokens": a.pp, "pp_s": round(pp_best, 4),
        "pp_tps": round(a.pp / pp_best, 2),
        "tg_tokens": a.tg,
        "tg_s": round(tg_best - tg1_best, 4),
        "tg_tps": round((a.tg - 1) / max(tg_best - tg1_best, 1e-9), 2),
        "raw": {"pp": pp_t, "tg": tg_t, "tg1": tg1_t},
    }
    print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=2))
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
