"""Is vLLM's own CPU 4-bit kernel faster than the tinygemm one?

vLLM 0.28's CPU wheel does ship a 4-bit dense GEMM -- `_C::int4_scaled_mm_cpu`,
an AMX W4A8 kernel reached by loading a GPTQ checkpoint with
`VLLM_CPU_INT4_W4A8=1`.  It is not named like the ops an earlier search looked
for, which is why it was missed.  This measures it against
`aten::_weight_int4pack_mm_for_cpu` at Spark-X2.5's shapes.
"""
import json, time

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types

from vllm_omni.model_executor.layers.quantization.cpu_int4 import (
    _to_tinygemm, quantize_groupwise,
)

SHAPES = [("qkv", 3072, 2048), ("o", 2048, 2048), ("gate_up", 13312, 2048),
          ("down", 2048, 6656), ("lm_head", 131072, 2048)]
GROUP = 128


def timeit(fn, iters=20):
    for _ in range(5):
        fn()
    best = float("inf")
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def build_w4a8(w, group):
    """[N, K] float -> the blocked tensors `int4_scaled_mm_cpu` wants."""
    n, k = w.shape
    gnum = k // group
    wf = w.float().reshape(n, gnum, group)
    wmax, wmin = wf.amax(-1), wf.amin(-1)
    scale = ((wmax - wmin).clamp(min=1e-9) / 15.0)
    zp = (-wmin / scale).round().clamp(0, 15)
    q = ((wf - wmin.unsqueeze(-1)) / scale.unsqueeze(-1)).round().clamp_(0, 15)
    codes = q.reshape(n, k).to(torch.int32).t().contiguous()      # [K, N]
    zpi = zp.to(torch.int32).t().contiguous()                     # [G, N]

    # AWQ interleave the kernel's repack expects
    codes = codes.view(k, n // 8, 8)[:, :, (0, 2, 4, 6, 1, 3, 5, 7)].reshape(k, n)
    zpi = zpi.view(gnum, n // 8, 8)[:, :, (0, 2, 4, 6, 1, 3, 5, 7)].reshape(gnum, n)
    packed_w = pack_quantized_values_into_int32(codes, scalar_types.uint4, 1).contiguous()
    packed_zp = pack_quantized_values_into_int32(zpi, scalar_types.uint4, 1).contiguous()
    scales = scale.t().contiguous().to(torch.bfloat16)            # [G, N]
    return ops.convert_weight_packed_scale_zp(
        packed_w, packed_zp, scales, ops.CPUQuantAlgo.AWQ)


def main():
    rows = []
    for name, N, K in SHAPES:
        w = torch.randn(N, K)
        packed, s, z = quantize_groupwise(w, GROUP)
        inner, sz = _to_tinygemm(packed, s, z)
        try:
            bw, bzp, bs = build_w4a8(w, GROUP)
        except Exception as e:
            print(json.dumps({"name": name, "w4a8_build_error": f"{type(e).__name__}: {e}"}))
            continue
        rec = {"name": name, "N": N, "K": K, "group": GROUP}
        for M in (1, 8, 512):
            x = torch.randn(M, K, dtype=torch.bfloat16)
            t_tg = timeit(lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(x, inner, GROUP, sz))
            try:
                out = ops.int4_scaled_mm_cpu(x, bw, bzp, bs, None)
                t_w4 = timeit(lambda: ops.int4_scaled_mm_cpu(x, bw, bzp, bs, None))
                ref = torch.ops.aten._weight_int4pack_mm_for_cpu(x, inner, GROUP, sz)
                rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                rec[f"m{M}"] = {"tinygemm_ms": round(t_tg * 1e3, 4),
                                "w4a8_ms": round(t_w4 * 1e3, 4),
                                "speedup": round(t_tg / t_w4, 2),
                                "rel_diff_vs_tinygemm": round(rel, 4)}
            except Exception as e:
                rec[f"m{M}"] = {"tinygemm_ms": round(t_tg * 1e3, 4),
                                "w4a8_error": f"{type(e).__name__}: {str(e)[:90]}"}
        rows.append(rec)
        print(json.dumps(rec))
    with open("w4a8_vs_tinygemm.json", "w") as f:
        json.dump(rows, f, indent=2)
    for M in (1, 8, 512):
        tg = sum(r[f"m{M}"].get("tinygemm_ms", 0) for r in rows[:4])
        w4 = sum(r[f"m{M}"].get("w4a8_ms", 0) for r in rows[:4])
        if w4:
            head_tg = rows[4][f"m{M}"].get("tinygemm_ms", 0)
            head_w4 = rows[4][f"m{M}"].get("w4a8_ms", 0)
            print(f"M={M:4d}  28 layers + head: tinygemm {tg*28+head_tg:8.3f} ms   "
                  f"w4a8 {w4*28+head_w4:8.3f} ms")


if __name__ == "__main__":
    main()
