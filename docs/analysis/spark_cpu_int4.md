# Closing the CPU decode gap: 4-bit weights for Spark-X2.5 on vLLM

`analysis/spark_edge_support.md` ended with one open item. At matched bf16,
vLLM beat llama.cpp on Spark-X2.5-1.7B everywhere -- 8.7-15.4x on prefill,
1.19-2.65x on decode -- but llama.cpp's **Q4_K_M** build won single-stream
decode outright. Decode at batch 1 reads every weight to emit one token, so
bits per weight, not flops, set the token rate, and vLLM had no 4-bit CPU
path in use. This note closes that and re-runs the comparison at matched
precision.

Two corrections to earlier findings come first, because both changed the
answer.

Code: `analysis/experiments/spark_edge/{export_gptq_cpu,quantize_int4,ppl_vllm,profile_cpu_step,bench_vllm_w4a8,cmp_llamacpp,dequantize_int4}.py`,
`vllm_omni/model_executor/layers/quantization/cpu_int4.py` (+ tests).

## 0a. Correction: the llama.cpp numbers were measured cold

The Q4_K_M column in `spark_edge_support.md` was taken on a cold page cache.
The GGUF lives on a FUSE network filesystem and llama.cpp mmaps it, so the
first grid was partly timing storage. Warm, same pinned cores:

| llama.cpp Q4_K_M, 32 threads | pp512 | tg128 |
|---|---|---|
| as previously published (cold) | 1199 | 82.81 |
| warm | **1356** | **98.02** |

The bar this work has to clear is 98 tok/s, not 82.8.

## 0b. Correction: vLLM's CPU backend *does* ship a 4-bit kernel

`spark_edge_support.md` reported that vLLM 0.28's CPU build has no int4 dense
kernel, on the evidence that `woq_int4_linear`, `int4_scaled_mm_with_quant`
and `da8w4_linear` are all absent. They are -- but the kernel exists under a
different name. `torch.ops._C.int4_scaled_mm_cpu` is present in the released
CPU wheel: an **AMX W4A8** GEMM, reached by loading a GPTQ checkpoint with
`VLLM_CPU_INT4_W4A8=1`, with `cpu_gemm_wna16` as the W4A16 fallback. The
search missed it because it searched for names, not for the capability.

This matters, because it is the fastest option by a wide margin. Measured at
Spark's shapes against PyTorch's `aten::_weight_int4pack_mm_for_cpu`
(`bench_vllm_w4a8.py`, 28 layers + head, group 128):

| batch | tinygemm | vLLM W4A8 (AMX) | speedup |
|---|---|---|---|
| 1 row (decode) | 4.45 ms | **2.66 ms** | 1.67x |
| 8 rows | 14.25 ms | **6.76 ms** | 2.11x |
| 512 rows (prefill) | 878 ms | **104 ms** | 8.44x |

## 1. What was actually slow

`profile_cpu_step.py` attributes one decode step. At bf16, 32 threads:

| bucket | ms/step | share |
|---|---|---|
| `_C::onednn_mm` (113 GEMMs) | 21.24 | 76% |
| inductor-generated kernels (norms, rope, GEGLU) | 2.47 | 9% |
| attention + KV cache ops (4 x 28) | 1.63 | 6% |
| head-gate `aten::mm` (28) | 1.08 | 4% |
| sampling (`argmax` over 131072) | 0.24 | 1% |
| ~900 small view/slice ops | 0.55 | 2% |
| **total** | **27.96** | 35.8 tok/s |

3.42 GB of bf16 weights at ~161 GB/s effective. Only fewer bits move it.

## 2. Results

Xeon Platinum 8480C, one socket, 32 physical cores pinned with `taskset`,
warm page cache, engines run strictly one at a time, and every run records the
busy fraction of its own cores before it starts (`cpu_busy.py`) -- a
shared-core benchmark measures the scheduler, not the engine. Perplexity is
teacher-forced over 512-token windows, 126 217 scored tokens, each engine
scored against **its own** bf16 run so tokenizer and chunking differences
cancel.

| build | bits/weight | pp512 | tg128 | ppl vs own bf16 |
|---|---|---|---|---|
| vLLM bf16 | 16 | 4455 | 36.35 | -- |
| vLLM GPTQ W4A16 (sym, g128) | 4.16 | **4659** | 37.53 | +25.5% |
| **vLLM GPTQ W4A8 (sym, g128)** | 4.16 | **4355** | **88.7** | +27.2% |
| vLLM `cpu_int4` (asym, g128) | 4.25 | 3589 | 85.5 | +18.2% |
| vLLM `cpu_int4` (asym, g64) | 4.50 | 3689 | 81.1 | +12.5% |
| llama.cpp Q4_K_M | 5.15 | 1356 | **98.02** | **+7.6%** |

Against llama.cpp at 4-bit: **prefill 3.21x faster, decode 0.90x.** Decode
went 36.35 -> 88.7, a 2.44x gain, which turns a 2.70x deficit into a 1.11x
one without quite closing it.

Quantizing the head matters more than anything else in that table: Spark ties
its 131072 x 2048 vocab table to the output head, and that one GEMM is the
largest read in a decode step. Leaving it in bf16 costs 12 tok/s (76.0 vs
88.7), so `export_gptq_cpu.py --quantize-lm-head` unties it, keeping the
table in bf16 for the lookup (a few hundred bytes a token) and giving the
head its own 4-bit copy.

Note the W4A16 row: best prefill of anything measured, worst decode of the
4-bit builds. It dequantizes to bf16 per call, which amortizes over 512 rows
and is pure overhead on one.

### Thread scaling (cores 56-87, both engines, default OpenMP settings)

Measured with `cpu_int4`, before the W4A8 path was found, but the shape of it
is the point:

| threads | vLLM int4 pp512 | llama.cpp pp512 | vLLM int4 tg128 | llama.cpp tg128 |
|---|---|---|---|---|
| 4 | **800.1** | 210.2 | **27.08** | 24.20 |
| 8 | **1476.2** | 409.8 | 44.02 | 44.00 |
| 16 | **2599.1** | 799.2 | 66.72 | **73.19** |
| 32 | **3524.9** | 1348.0 | 80.20 | **99.55** |

Decode wins at 4 threads, ties at 8, loses from 16 up. That is the signature
of a fixed per-step cost that does not shrink when the GEMMs do.

### What else moved the needle, and what didn't

| change | tg128 | |
|---|---|---|
| `cpu_int4` g64 baseline | 81.10 | |
| `KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=ACTIVE KMP_LIBRARY=turnaround` | 83.61 | ~250 OpenMP parallel regions per step; how threads park between them is not free |
| group 128 instead of 64 | 85.48 | 5.5% fewer weight bytes |
| `detokenize=False` (what `llama-bench` measures) | 82.13 | |

Tried and rejected: `compile_sizes=[1]`, to give inductor static shapes for
the decode step (81.77, no change); `enforce_eager` (58.88, much worse);
inductor `cpp_wrapper` (fails outright -- vLLM's own ops take a `str`
argument `custom_op_wrapper` cannot box).

## 3. The remaining decode gap is framework, not kernel

Re-profiling the `cpu_int4` step at 12.50 ms:

| bucket | ms/step |
|---|---|
| int4 GEMMs (113) | 5.49 |
| inductor-generated kernels | 2.29 |
| attention + KV cache ops | 1.41 |
| head-gate `aten::mm` (28) | 0.92 |
| custom-op dispatch (113) | 0.58 |
| sampling + small ops | 0.80 |
| **total** | **12.50** |

Those GEMMs move 0.95 GB in 5.49 ms, **173 GB/s**; llama.cpp does its whole
10.2 ms step against 1.1 GB, ~108 GB/s all-in. The 4-bit kernel is not what
is losing. **7.0 ms of the 12.5 ms step is per-step framework cost**, and
llama.cpp has almost none of it. With the faster W4A8 kernel the GEMM share
falls further and the framework share rises, which is why the gap closes to
0.90x and then stops.

Closing the rest is vLLM CPU engine work -- the attention wrappers, the ~900
small view ops, inductor launch overhead -- and it is out of reach here
because vLLM itself is off-limits to modification. The one piece inside reach
is the head gate: 28 `aten::mm` calls, 0.92 ms, almost all thread barrier for
a 2048x8 matmul. Rewriting it as a fused broadcast-and-reduce was tried and is
much worse (44.5 tok/s; inductor does not vectorize the reduction). Folding
those eight rows into the fused QKV GEMM would remove it, and is the obvious
next step.

## 4. Quality: the grid, not the runtime

Q4_K_M spends **5.15 bits per weight** (1.1 GB for 1.7 B parameters) against
4.16-4.50 here, and still loses less: +7.6% perplexity against +12.5% for the
best of ours. Two separate things cause that.

**vLLM's GPTQ path cannot express an asymmetric grid at 4 bits.** It rejects
`sym=False` outright ("Unsupported quantization config: bits=4, sym=False"),
so the fast W4A8 kernel can only be fed a symmetric grid -- which wastes a
level on every group whose weights are off-centre. That is most of why the
W4A8 rows cost +27% while `cpu_int4`, which is asymmetric, costs +12.5% at a
comparable bit budget. The AMX kernel itself takes real zero points
(`convert_weight_packed_scale_zp(..., CPUQuantAlgo.AWQ)`); it is the GPTQ
*config* that refuses. Exporting AWQ format instead should get an asymmetric
grid through the same kernel, and is the highest-value thing left undone
here.

**W4A8 quantizes activations to int8, and that is nearly free.** W4A16 on the
same checkpoint scores +25.5% against W4A8's +27.2% -- a 1.7 point difference
for a 2.4x decode speedup. That is a much better outcome than int8
activations have had elsewhere in this repo (Qwen3-TTS, MiniCPM-o, Spark on
the NPU), and worth recording as the counter-example.

**Even asymmetric, our grid is coarser than K-quants.** `cpu_int4` does
round-to-nearest on each group's min/max with a small shrink search; the
search is worth about what it costs (at g64: 48.0 perplexity with it, 59.7
without, on the smaller corpus) but it optimizes per-group squared error,
which is the wrong objective. Q4_K_M uses an importance-weighted scale search
per 32-weight sub-block and spends more bits on the tensors that need them.

This is a quantizer gap, not a runtime gap, and the control proves it:
expanding a quantized checkpoint back to plain bf16 offline
(`dequantize_int4.py`) and scoring that reproduces the quantized run to
**within 0.2%** (48.08 vs 48.01 at g64, 56.80 vs 56.89 at g32). Both GEMM
paths, the nibble packing and the quantized tied embedding are all faithful.

A caution on measuring this: on a smaller, harder corpus (34 k words of this
repo's own notes, bf16 perplexity 43.7) the ordering is not even stable --
group 32 lands worse than group 128 there, and reverses on the larger corpus.
Perplexity differences between 4-bit variants of a 1.7 B model need a corpus
big enough to be in a sane regime before they mean anything.

## 5. An AMX hazard worth recording

Running PyTorch's tinygemm kernel on a multi-row batch and then vLLM's
`cpu_attention_with_kv_cache` in the same forward kills the process with
**SIGILL inside the attention kernel** -- the signature of an AMX instruction
issued with no valid tile configuration loaded. The identical prompt is fine
in bf16, and fine at any batch size once every multi-row GEMM takes a
different path, so the two kernels disagree about who owns the AMX tile state.
`cpu_int4` therefore routes only single-row steps to tinygemm
(`dequant_threshold` defaults to 2). vLLM's own W4A8 path does not have the
problem.

## 6. Where this leaves it

| | before | after |
|---|---|---|
| decode vs llama.cpp Q4_K_M | 0.37x (36.35 vs 98.02) | **0.90x** (88.7) |
| prefill vs llama.cpp Q4_K_M | no 4-bit path in use | **3.21x** (4355 vs 1356) |
| quality cost of 4-bit | -- | +27.2% ppl (W4A8) / +12.5% (`cpu_int4` g64) vs Q4_K_M's +7.6% |
| model on disk | 3.4 GB | 0.9-1.4 GB |

Honest summary: 4-bit weights turn a 2.7x single-stream decode deficit into a
1.11x one and put prefill 3.2x ahead, but vLLM still does not beat llama.cpp
at single-stream decode on 32 cores, and our 4-bit grids are measurably
coarser than K-quants. Three specific, measured things stand between here and
a win, in order of expected value:

1. **An AWQ-format export**, to get an asymmetric grid through the AMX W4A8
   kernel -- worth roughly the +27.2% -> +12.5% perplexity difference at no
   speed cost.
2. **Folding the head gate into the QKV GEMM** -- 0.92 ms of a ~11 ms step.
3. **vLLM CPU per-step overhead** -- 7.0 ms at batch 1, the whole remaining
   gap, and not addressable without changing vLLM.

`cpu_int4` keeps a narrower role than intended: it is the more accurate of the
two (asymmetric grid, +12.5% against +27.2%) and it needs no AMX, but on an
AMX host vLLM's own W4A8 path is faster on both axes. Its docstring says so.
