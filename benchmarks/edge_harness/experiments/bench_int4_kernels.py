"""Microbenchmark of candidate CPU 4-bit weight kernels at Spark-X2.5 shapes.

Decode is one token, so every linear is a GEMV and the only thing that matters
is how fast the weight bytes can be streamed. llama.cpp's Q4_K_M moves ~4.5
bits per weight; anything we pick has to get close to that at a comparable
fraction of peak bandwidth.
"""
import argparse, json, time
import torch

SHAPES = [
    ("qkv", 3072, 2048),
    ("o", 2048, 2048),
    ("gate_up", 13312, 2048),
    ("down", 2048, 6656),
    ("lm_head", 131072, 2048),
]


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--out", default="int4_kernels.json")
    a = ap.parse_args()
    torch.set_num_threads(torch.get_num_threads())
    print("threads", torch.get_num_threads())

    rows = []
    for name, N, K in SHAPES:
        x16 = torch.randn(a.m, K, dtype=torch.bfloat16)
        x32 = x16.float()
        w16 = torch.randn(N, K, dtype=torch.bfloat16)
        rec = {"name": name, "N": N, "K": K, "M": a.m}

        # bf16 reference (oneDNN through aten)
        t = timeit(lambda: torch.nn.functional.linear(x16, w16), a.iters)
        rec["bf16_ms"] = t * 1e3
        rec["bf16_GBps"] = N * K * 2 / t / 1e9

        # int8 weight-only (aten _weight_int8pack_mm)
        try:
            w8 = torch.randint(-127, 127, (N, K), dtype=torch.int8)
            s8 = torch.rand(N, dtype=torch.bfloat16) * 0.01
            t = timeit(lambda: torch.ops.aten._weight_int8pack_mm(x16, w8, s8), a.iters)
            rec["int8_ms"] = t * 1e3
            rec["int8_GBps"] = N * K / t / 1e9
        except Exception as e:
            rec["int8_err"] = f"{type(e).__name__}: {e}"

        # int4 group-wise (aten _weight_int4pack_mm_for_cpu)
        for dt, xx in (("bf16", x16), ("fp32", x32)):
            try:
                wint = torch.randint(0, 15, (N, K), dtype=torch.int32)
                packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(wint, 1)
                sz = torch.rand(K // a.group, N, 2, dtype=xx.dtype)
                f = lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(xx, packed, a.group, sz)
                out = f()
                t = timeit(f, a.iters)
                rec[f"int4_{dt}_ms"] = t * 1e3
                rec[f"int4_{dt}_GBps"] = N * K * 0.5 / t / 1e9
                rec[f"int4_{dt}_out"] = str(tuple(out.shape)) + " " + str(out.dtype)
            except Exception as e:
                rec[f"int4_{dt}_err"] = f"{type(e).__name__}: {e}"

        rows.append(rec)
        print(json.dumps(rec))

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
