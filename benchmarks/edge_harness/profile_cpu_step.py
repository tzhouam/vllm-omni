"""Where does a single-token CPU decode step go?

Runs the real engine and splits decode self-CPU time into linear GEMMs,
attention, elementwise and everything else.

Two things this deliberately does *not* do any more, because both produced
published numbers that meant something other than what they said:

* **It does not divide a whole ``generate`` by ``tg``.** That call contains the
  prefill step and the sampling around it, so the quotient is not a decode
  step. Decode is measured the way the wall clock measures it -- profile a
  prefill-only run and a full run, subtract, divide by ``tg - 1`` -- so the
  profile and the wall clock are finally the same quantity.
* **It does not profile a one-token prompt by default and call the result
  representative.** Spark's sliding layers have a 512-token window; at a
  1-token prompt nothing in the KV cache is being read, so attention cost here
  says nothing about attention cost at context. ``--prompt-len`` sets it, and
  the length is recorded in the artifact.

Every result carries provenance (model path, context, build SHAs, thread set),
because two saved profiles in this tree are indistinguishable from each other
and one set describes a checkpoint later found to be wrong.
"""
import argparse, json, os, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_harness import provenance  # noqa: E402

GEMM = ("addmm", "mm", "linear", "matmul", "bmm", "packed_linear",
        "int8_scaled_mm", "int4pack", "int8pack", "onednn", "dnnl")
ATTN = ("attention", "paged", "flash", "softmax", "reshape_and_cache")


def bucket_of(key: str) -> str:
    n = key.lower()
    if any(g in n for g in GEMM):
        return "gemm"
    if any(s in n for s in ATTN):
        return "attn"
    if n.startswith("aten::") or n.startswith("_c::"):
        return "other_op"
    return "framework"


def profile_once(llm, prompt, sp):
    """Self-CPU microseconds and call counts, per op key, for one generate."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        llm.generate([prompt], sp, use_tqdm=False)
    us, calls = defaultdict(float), defaultdict(int)
    for e in prof.key_averages():
        us[e.key] += e.self_cpu_time_total
        calls[e.key] += e.count
    return us, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="XHToken/Spark-X2.5-1.7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--tg", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=1,
                    help="Prompt tokens. Use >=512 to fill Spark's sliding "
                         "window; the default of 1 measures an empty cache.")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="cpu_step_profile.json")
    a = ap.parse_args()
    if a.tg < 2:
        ap.error("--tg must be at least 2: decode is measured by difference")

    import vllm_omni  # noqa: F401
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.max_model_len,
              enforce_eager=a.enforce_eager, max_num_seqs=1,
              quantization=a.quantization, enable_prefix_caching=False,
              trust_remote_code=False, disable_log_stats=True)

    prompt = {"prompt_token_ids": [10] * a.prompt_len}
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    spN = SamplingParams(temperature=0.0, max_tokens=a.tg, ignore_eos=True)
    decode_steps = a.tg - 1

    llm.generate([prompt], spN, use_tqdm=False)  # warm

    def wall(sp):
        best = float("inf")
        for _ in range(a.reps):
            t = time.perf_counter()
            llm.generate([prompt], sp, use_tqdm=False)
            best = min(best, time.perf_counter() - t)
        return best

    t1, tN = wall(sp1), wall(spN)
    step_ms = (tN - t1) / decode_steps * 1e3

    # Same subtraction under the profiler, so the two agree by construction.
    us1, calls1 = profile_once(llm, prompt, sp1)
    usN, callsN = profile_once(llm, prompt, spN)

    rows, buckets = [], defaultdict(float)
    for key in set(usN) | set(us1):
        delta_us = usN.get(key, 0.0) - us1.get(key, 0.0)
        delta_calls = callsN.get(key, 0) - calls1.get(key, 0)
        if delta_us <= 0 and delta_calls <= 0:
            continue  # prefill-only work; not part of a decode step
        b = bucket_of(key)
        buckets[b] += delta_us
        rows.append((key, b, delta_us / 1e3, delta_calls))
    rows.sort(key=lambda r: -r[2])

    prefill_total_ms = sum(us1.values()) / 1e3
    res = {
        "phase": "decode",
        "tag": f"{a.dtype}/{a.quantization}/eager={a.enforce_eager}",
        "wall_step_ms": round(step_ms, 3),
        "tps": round(1e3 / step_ms, 2),
        "decode_steps": decode_steps,
        "profiled_total_ms_per_step": round(
            sum(buckets.values()) / 1e3 / decode_steps, 3),
        "buckets_ms_per_step": {
            k: round(v / 1e3 / decode_steps, 3)
            for k, v in sorted(buckets.items(), key=lambda x: -x[1])},
        "top": [{"op": r[0], "bucket": r[1],
                 "ms_per_step": round(r[2] / decode_steps, 4),
                 "calls_per_step": round(r[3] / decode_steps, 1)}
                for r in rows[:35]],
        "prefill": {
            "wall_ms": round(t1 * 1e3, 3),
            "profiled_self_ms": round(prefill_total_ms, 3),
            "prompt_tokens": a.prompt_len,
        },
        "provenance": provenance({
            "model": a.model,
            "model_path": os.path.realpath(a.model) if os.path.exists(a.model)
            else a.model,
            "quantization": a.quantization,
            "dtype": a.dtype,
            "enforce_eager": a.enforce_eager,
            "prompt_len": a.prompt_len,
            "tg": a.tg,
            "context_at_end": a.prompt_len + a.tg,
            "max_model_len": a.max_model_len,
            "torch_threads": torch.get_num_threads(),
        }),
    }
    print(json.dumps(res, indent=2))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
