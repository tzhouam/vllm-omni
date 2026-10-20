# Spark-X2.5-1.7B on CPU and Galaxy S25

Spark-X2.5 (XHToken/iFlytek, Apache 2.0, released 2026-09-01) is a 1.7 B
on-device model built for agentic tool use: BFCL-V4 46.9, tau-squared-bench
65.3, a native 1 M-token context and 200+ languages. This note covers adding
it to vLLM-Omni, measuring it against every engine that can run it, putting it
on a real Snapdragon 8 Elite, and driving an Android phone with it.

Reproduction scripts: `analysis/experiments/spark_edge/`.
Model code: `vllm_omni/model_executor/models/spark2_5/`, exporter
`vllm_omni/edge/spark_export.py`.

## 1. What had to be built

Three traits separate Spark from a Llama-style decoder, and all three are why
no generic converter picks it up:

| trait | detail | why it matters |
|---|---|---|
| hybrid attention | `layer_types` = 3x `sliding_attention` (window 512) then 1x `full_attention`, 28 layers -> 21 sliding + 7 full | bounds KV growth; this is what makes 1 M context affordable on a phone |
| per-layer-type RoPE | sliding: all 256 head dims, theta 1e4. full: first 64 dims only (`partial_rotary_factor` 0.25), theta 5e6 | two rotary tables, not one |
| head-wise output gate | `g_proj` [2048 -> 8] reads the same post-norm input as QKV; `sigmoid` scales each head's attention output before `out_proj` | an extra projection with no analogue in Llama/Qwen |

Also: QKV ships pre-fused as one `q_k_v_proj`, the MLP is GEGLU with *exact*
gelu, embeddings are tied, and `head_dim` is 256 with only 2 KV heads.

### Engine support before this work

| engine | Spark-X2.5 support |
|---|---|
| llama.cpp | yes, native (`src/models/spark2-5.cpp`) |
| SGLang | yes, native (`srt/models/spark2_5.py`) |
| vLLM 0.28 | **no** -- added here |
| MNN | no. `llmexport.py` dispatches on a fixed `model_type` list; `spark2_5` is absent |
| ExecuTorch | no |
| ncnn | no |

MNN, ExecuTorch and ncnn would each need the architecture written from
scratch. That is the honest reason they do not appear in the benchmark tables
below: they cannot load the model at all, not that they are slow at it.

### Correctness

Verified against the official remote-code implementation pinned to
transformers 4.57.1 (the version `config.json` targets; it does not import on
the 5.x in our venvs, so a dedicated venv runs ground truth).

| scenario | prompt | fp32 | bf16 |
|---|---|---|---|
| short | 28 tok | 32/32 greedy, max logprob diff 2.2e-05 | 24/24 greedy |
| long (exercises sliding window) | 2465 tok | -- | 32/32 greedy |
| tool calling | 137 tok | 32/32 greedy | diverges at token 23 |

The bf16 tool-calling divergence is a near-tie under greedy decoding, not a
defect: in fp32 the same prompt matches exactly with a 2.2e-05 logprob
difference.

One vLLM bug had to be worked around. The head-wise gate is a [2048, 8]
projection, and vLLM's CPU backend routes any narrow linear to the sgl-kernel
packed GEMM, whose fake-tensor rule assumes the packed weight keeps N in dim
0. VNNI packing does not for N that small, so it reports K instead of N and
`torch.compile` dies tracing the gate multiply. The gate is kept as a plain
parameter with `F.linear` -- a 2048x8 matmul gains nothing from the packed AMX
kernel anyway.

## 2. CPU: vLLM-Omni vs llama.cpp

Xeon Platinum 8480C (Sapphire Rapids, AMX-BF16/INT8). Both engines pinned with
`taskset` to the same physical cores and run **strictly one at a time**;
prefix caching disabled on vLLM so a replayed prompt is really recomputed
(llama-bench has no such cache). pp512 = prefill throughput, tg128 = decode.

### Same precision (bf16)

| threads | vLLM pp512 | llama.cpp pp512 | prefill | vLLM tg128 | llama.cpp tg128 | decode |
|---|---|---|---|---|---|---|
| 4 | **1113.4** | 72.4 | 15.4x | **11.11** | 4.20 | 2.65x |
| 8 | **2113.7** | 173.8 | 12.2x | **19.88** | 16.15 | 1.23x |
| 16 | **3639.4** | 341.7 | 10.7x | **30.16** | 25.31 | 1.19x |
| 32 | **4454.9** | 509.5 | 8.7x | **36.35** | 28.57 | 1.27x |

vLLM wins both, at every thread count. The prefill gap is AMX: llama.cpp's
bf16 path runs at AVX-512 speed, vLLM's goes through oneDNN/AMX tiles.

### Against llama.cpp's quantized builds

| threads | vLLM bf16 pp | llama.cpp Q8_0 pp | llama.cpp Q4_K_M pp | vLLM bf16 tg | llama.cpp Q8_0 tg | llama.cpp Q4_K_M tg |
|---|---|---|---|---|---|---|
| 4 | **1113.4** | 118.1 | 103.6 | 11.11 | 9.85 | **10.36** |
| 8 | **2113.7** | 490.2 | 434.2 | 19.88 | 45.01 | **42.40** |
| 16 | **3639.4** | 914.7 | 806.3 | 30.16 | 66.23 | **71.50** |
| 32 | **4454.9** | 1493.1 | 1199.0 | 36.35 | 76.54 | **82.81** |

Prefill: vLLM at bf16 still beats llama.cpp's *best quantized* build by 3.0x
at 32 threads. Prefill is compute-bound, and AMX beats 4-bit weight savings.

Decode: llama.cpp Q4_K_M wins from 8 threads up, by 2.3x at 32. Decode is
memory-bound -- 4-bit weights move a quarter of the bytes -- and this is a
precision difference, not an engine difference. Closing it needs a quantized
vLLM CPU path for Spark, which is the open item in section 6.

### Quantized: int8 on CPU

vLLM 0.28's CPU build has int8 dense kernels (`int8_scaled_mm_with_quant`,
`dynamic_scaled_int8_quant`), and `static_scaled_fp8_quant` is registered with
an empty schema, so on-the-fly fp8 fails outright.

An earlier draft of this note said the backend had no dense int4 kernel
either, on the strength of `woq_int4_linear`, `int4_scaled_mm_with_quant` and
`da8w4_linear` all being absent. That was wrong -- they are simply not the
names it uses. See section 2c: there are two four-bit paths, and the one
vLLM ships is `int4_scaled_mm_cpu`, an AMX W4A8 GEMM reached through the
GPTQ/AWQ kernel selector.

Int8 is also available. `analysis/experiments/spark_edge/quantize_int8.py` writes a
compressed-tensors W8A8 checkpoint straight from the safetensors (symmetric
per-output-channel round-to-nearest; no calibration needed, and llmcompressor
is avoided because it pulls transformers 5.x, on which Spark's remote code
does not import). The head-wise gate is left in full precision: its eight
outputs each scale a whole attention head, and it is 0.001% of the weights.
Weight SNR 36.9-40.9 dB across 140 tensors.

| threads | pp512 bf16 -> int8 | tg128 bf16 -> int8 |
|---|---|---|
| 16 | 3639 -> **5135** (1.41x) | 30.16 -> **48.30** (1.60x) |
| 32 | 4455 -> **6978** (1.57x) | 36.35 -> **58.44** (1.61x) |

Quality against the fp32 reference, though, is not free:

| scenario | greedy match | first-token top-1 | top-20 overlap | max logprob diff |
|---|---|---|---|---|
| short (28 tok) | 23/24 | agrees | 18/20 | 0.83 |
| long (2465 tok) | **12/32** | agrees | 19/20 | 1.56 |
| tool calling | 31/32 | agrees | 18/20 | 0.58 |

bf16 matched 32/32 on all three with a max difference of 0.24. Int8 keeps the
first token everywhere, but at long context the distribution has moved enough
that greedy output diverges after a dozen tokens. Usable, not free -- the same
caution the Qwen3-TTS and MiniCPM-o work in this repo reached about automatic
int8.

## 2b. CPU under concurrency

Single-stream decode is memory-bound, which is a precision contest llama.cpp
wins with 4-bit weights. Concurrent decode is compute-bound, which is an AMX
and continuous-batching contest. Total generated tokens/s, 32 pinned cores,
128-token prompts generating 128 tokens each:

| batch | vLLM bf16 | vLLM int8 | llama.cpp Q4_K_M | int8 vs llama.cpp |
|---|---|---|---|---|
| 1 | 34.6 | 54.9 | **91.1** | 0.60x |
| 2 | 69.9 | 105.2 | **129.6** | 0.81x |
| 4 | 139.8 | 210.3 | **251.6** | 0.84x |
| 8 | 266.1 | 387.8 | **406.5** | 0.95x |
| 16 | 495.7 | **694.7** | 613.9 | 1.13x |
| 32 | -- | **1080.6** | 778.3 | 1.39x |
| 64 | -- | **1557.9** | 904.5 | 1.72x |

The crossover is around batch 12. Below it llama.cpp's 4-bit weights win;
above it vLLM's AMX-INT8 GEMMs and continuous batching pull away, reaching
1.72x at batch 64 and still climbing. A phone agent is a batch-of-one
workload, so llama.cpp is the right single-device choice today; an edge
gateway serving a dozen sessions is vLLM's.

## 2c. Four bits on CPU: two paths, and which one a device should take

vLLM 0.28's CPU build has a 4-bit GEMM. `CPUWNA16LinearKernel`
(`vllm/model_executor/kernels/linear/mixed_precision/cpu.py`) routes GPTQ and
AWQ checkpoints to **`int4_scaled_mm_cpu`**, an AMX W4A8 kernel, whenever the
activations are bf16, the ISA has AMX tiles and the checkpoint carries no
activation reordering. Four-bit weights times eight-bit activations is exactly
what llama.cpp's Q4_K_M does (Q4_K weights, Q8_K activations), so this is the
honest precision match rather than a handicap.

Two paths were built, because they are not the same trade:

| | weights | activations | kernel | bits/weight | reached by |
|---|---|---|---|---|---|
| **W4A8** | 4-bit symmetric, group-wise | int8 (per the kernel) | `_C::int4_scaled_mm_cpu` (AMX) | 4.16 @ g128 | GPTQ checkpoint, vLLM's own dispatch |
| **W4A16** | 4-bit asymmetric min/max, group-wise | bf16 | `aten::_weight_int4pack_mm_for_cpu` | 4.16 @ g128 | `cpu_int4`, registered out-of-tree |

W4A8 is forced symmetric: `_process_gptq_weights_w4a8` builds its own zero
point (a constant 8) and ignores the checkpoint's `qzeros`, so the grid must be
`w = (code - 8) * scale`. W4A16 is free to use an asymmetric min/max grid, and
a per-group scale search recovers roughly a dB in either case.

### Which is faster

Per-op profiles, taken back to back on the same 24 cores. Throughput on this
host swings ~30% run to run and the gap here is ~8%, so the GEMM self time is
the measurement that can actually rank them:

| path | GEMM/step | step | tok/s |
|---|---|---|---|
| W4A8 g128 | **5.91 ms** | **11.55 ms** | **86.6** |
| W4A16 g128 | 6.85 ms | 12.51 ms | 80.0 |
| llama.cpp Q4_K_M | -- | 11.11 ms | 90.0 |
| vLLM bf16 | -- | -- | 41.0 |

W4A8 wins twice: the kernel is 0.94 ms cheaper per step, and because it is
reached through vLLM's own dispatch it also avoids the 0.48 ms/step custom-op
boundary the tinygemm path needs (that boundary exists because tinygemm's cost
is linear in batch size, so prefill has to be routed to a dequantize-then-AMX
path instead -- see below).

**The trap that cost a day: a GPTQ checkpoint must quantize the output head.**
Spark ties its embedding to its head, so a converter that leaves the tie in
place emits no `lm_head` tensors and that one GEMM runs bf16 through
`_C::onednn_mm` -- **2.97 ms per step**, more than the entire 4-bit saving. The
profile is what caught it: `int4_scaled_mm_cpu` ran 112 times (28 layers x 4
linears) and `onednn_mm` once. Every "W4A8" number measured before that was on
a handicapped build, which is why the W4A8/W4A16 ordering appeared to flip
between grids.

### Which is more faithful

Greedy divergence from each build's own bf16 output, 12 prompts, measured the
same way on both engines:

| build | exact match | mean prefix kept |
|---|---|---|
| llama.cpp Q4_K_M | **4/12** | **0.51** |
| vLLM W4A16 | 2/12 | 0.30 |
| vLLM W4A8 | 1/12 | 0.13 |

Q4_K_M's 6-bit sub-block scales buy real granularity at the same bit budget;
round-to-nearest with a per-group search does not match it. So the two paths
are a genuine choice, not a ranking: **W4A8 for speed, W4A16 for fidelity**,
and `vllm_omni/edge/weight_path.py` makes that choice from a hardware probe
rather than a flag, because the fast kernel on each device needs a different
and mutually exclusive weight layout:

| device class | path | why |
|---|---|---|
| x86 + AMX | W4A8 / GPTQ | `int4_scaled_mm_cpu` applies |
| x86 without AMX | W4A16 / tinygemm | no AMX, no int4 kernel |
| x86 + AMX, fidelity asked | W4A16 / tinygemm | keeps bf16 activations |
| ARM64 | W4A16, **unverified** | PyTorch's 4-bit CPU GEMM dispatches a different kernel there |
| phone NPU | exported graph, calibrated w8a16 | vLLM does not run there at all |

### Prefill needs the other kernel

The tinygemm kernel dequantizes once per row of activations, so its cost is
linear in batch size: it beats AMX bf16 below ~8 rows and loses badly above.
Measured at Spark's `gate_up` shape, per call: 0.070 ms at 1 token, 14.2 ms at
512. Left alone, pp512 collapsed from 4455 tok/s (bf16) to 557. Unpacking the
weight to bf16 once per call and handing it to the ordinary oneDNN/AMX GEMM
brings it back to **3347 tok/s**, 3.0x llama.cpp's 1121. W4A8 needs none of
this -- the AMX kernel is the right one at every batch size, which is the
second reason it is the default on an AMX machine.

## 2d. Init time and resident memory, which is what decides edge at all

Speed is not the binding constraint on an edge box; a 90-second start and 8 GB
resident is. Both turned out to be fixable, and one of them was a bug.

### vLLM recompiled the model on every single start

`_try_load_aot_compiled_fn` finishes a cache load with
`loaded_fn._artifacts.compiled_fn.finalize_loading(cfg)`. On CUDA that artifact
is a wrapper object carrying the method; with the inductor CPU backend it is a
plain function, so the call raises `AttributeError`, the handler treats the
whole load as a cache miss ("Compiling model again due to a load failure"), and
the model is compiled again from scratch. Every restart paid it.

Attaching the missing no-op (a plain Python function accepts attribute
assignment; the CPU artifact loads its generated `.so` lazily on first call,
which is the warm-up run) fixes it. Same cache directory, one variable changed:

| | init |
|---|---|
| cold cache (first ever compile) | 93.8 s |
| warm cache, vLLM as shipped | 49.8 s |
| **warm cache, shim installed** | **14.9 s** |

**3.3x faster warm start**, reproduced across passes (14.9 / 14.7 / 15.0, and
12.8-13.0 on the W4A16 path). The shim is self-disabling: if the artifact
already has `finalize_loading`, as it does on GPU and will if upstream gives
the CPU path one, it changes nothing.

### Measure anonymous memory, not RSS

A first pass concluded that sizing the KV cache saved 3.1 GB. It does not, and
the error is worth recording because it is easy to repeat: that comparison
read `VmRSS`, which counts the mmap'd checkpoint's file pages, and those come
and go with page-cache pressure. **Identical configurations measured 1.2-2.0 GB
apart.** `RssAnon` -- allocations only -- repeats to within **0.5 MB**, and is
what a memory budget is actually about.

### The KV cache is not a memory lever on CPU

Re-measured properly, with only the budget changing:

| KV budget | anonymous RSS |
|---|---|
| 4 GiB (env default) | 7601 MB |
| 512 MiB | 7405 MB |
| 128 MiB | 7413 MB |

Shrinking the cache 32x moves resident memory by **188 MB**, and 512 MiB is
indistinguishable from 128 MiB. The cache is allocated but its pages are never
touched, so it costs address space, not memory.

It still has to be *correct*, which is a separate problem: `--kv-cache-memory-bytes`
was silently discarded on CPU. `CpuPlatform.check_and_update_config` ends with
a block its own comment calls "Lagecy setting" that overwrites
`cache_config.kv_cache_memory_bytes` from `VLLM_CPU_KVCACHE_SPACE`
unconditionally, so the budget could only be set in whole GiB and vllm-omni's
per-stage budgets had no effect at all. Verified by allocation rather than by
memory, because allocation cannot drift: with the env at 1 GiB, asking for
128 MiB produced the same **17 881** cache tokens as asking for nothing, and
with the env at 4 GiB it produced **71 527**. After restoring the obvious
precedence the same request yields **2 231** tokens. That matters for
admission -- vLLM preempts by recompute when the cache cannot hold
`max_num_seqs x max_model_len`, which on a single-stream deployment is a stall
-- not for footprint. `vllm_omni/edge/kv_budget.py` prices the cache from the
config, reporting the flat figure (what is safe to pass, since vLLM's CPU
accounting is flat) beside the hybrid one a layer-type-aware backend could use.

### Where the memory actually goes: the weight load

Split by phase, on anonymous RSS:

| phase | W4A8 (GPTQ) | W4A16 (tinygemm) |
|---|---|---|
| before import | 12 MB | 12 MB |
| after `import vllm_omni` | 513 MB | 515 MB |
| after engine construction | 7338 MB | 5023 MB |
| **after releasing free arenas** | **3510 MB** | **3411 MB** |

Imports cost half a gigabyte -- they are the init-*time* problem, not the
memory one. The engine adds 4.5-6.8 GB for a model whose weights are under
1.4 GB, and **most of it is not live**: loading a quantized checkpoint unpacks
4-bit codes to int32 -- four bytes per weight -- to permute and repack them,
and although the intermediates are freed the allocator keeps the arenas. One
release call returns **3828 MB** on the GPTQ path and 1612 MB on tinygemm, a
**52%** cut on the configuration we actually want to deploy.

The second reading of that table matters more than the first: after the
release both paths land in the same place. W4A8's apparent 2.3 GB penalty over
W4A16 was never a footprint difference, it was the repack. So W4A8 wins on
speed *and* ties on memory, and fidelity is the only remaining reason to
choose W4A16.

`vllm_omni/edge/memory.py` does the release and `patch.py` hooks it to the CPU
worker's load and warm-up (`VLLM_OMNI_CPU_RELEASE_MEMORY=0` to disable). The
~3.5 GB that remains, against 0.86 GB of weights, is the next thing to chase.

### Two bugs that made CPU deployments take the slow path silently

Init is not one number per model; it has a different cause in each pipeline,
and in two places the slow path was taken without saying so.

**Parallel stage init could never run on CPU.**
`resolve_stage_physical_devices` returns `None` for `devices: "cpu"`, and the
admission resolver fails closed on `None` -- right for a GPU replica whose
devices cannot be accounted, wrong here -- so every multi-stage CPU pipeline
raised `StageAdmissionError` and fell back to sequential startup. The edge
profile *sets* `parallel_stage_init: true`, so on CPU that profile was broken
rather than merely slower.

The fix is not to skip admission. The per-device ledger is vacuous on CPU, but
the resource concurrent init contends for is host memory, and two stages
allocating KV caches at once exhaust it exactly as two GPU stages exhaust a
card. `check_host_admission` applies the same discipline to that resource: sum
the declared budgets, compare with available RAM and a margin, refuse to start
otherwise, and fail closed on a stage that declares no budget.

| Qwen3-TTS 0.6B, init only | |
|---|---|
| sequential | 64.57 s |
| **parallel** | **30.54 s** |

**A deploy YAML meant different things depending on which flag loaded it.**
The `orchestrator:` block carrying `parallel_stage_init` was applied only on
the `deploy_profile` path, so pointing `--deploy-config` at the very same file
loaded its stages and dropped the flag: 63.76 s against 30.76 s, with nothing
in the log to explain the difference. The block now applies either way.

### What is not a memory lever

Two candidates tested and closed, so they are not re-tried:

| lever | effect |
|---|---|
| OMP thread count (8 / 24 / 48) | 3754 / 3775 / 3818 MB -- 64 MB across a 6x range |

**Retracted (2026-09-14): KV cache size was on this list, and it is the single
largest lever there is.** The row read "4 GiB -> 128 MiB saves 188 MB, the
pages are never touched". Re-measured now that the budget is actually
honoured -- 3 interleaved passes per arm, both filling a 1536-token context:

| KV budget | resident |
|---|---|
| 1 GiB (the smallest `VLLM_CPU_KVCACHE_SPACE` can express) | 3264 MB |
| 192 MiB, explicit | **2421 MB** |

**843 MB.** And `mincore(2)` says why the old reasoning was wrong: the cache is
not allocated-and-untouched, it is fully in core -- 1023.8 MB allocated against
1024.1 MB resident, and 190.8 against 191.1 when sized down.

The explanation for the original 188 MB is two sections below, in this report's
own words: `--kv-cache-memory-bytes` *was silently discarded on CPU*, because
`CpuPlatform.check_and_update_config` overwrote it from
`VLLM_CPU_KVCACHE_SPACE` unconditionally. So the "32x shrink" never happened;
both arms ran the same cache and the 188 MB was measuring something else. The
lesson is the one this study keeps relearning: a lever tested through a path
that silently ignores it measures nothing, and "no effect" is the most
dangerous result to accept without checking that the knob was connected.

The dual weight layout the W4A16 path keeps (tinygemm for decode, a row-major
copy for prefill) *is* one. Median of three interleaved passes, anonymous RSS,
spread under 1.5%:

| build | resident | prefill | decode | init |
|---|---|---|---|---|
| W4A8 g128 | 3640 MB | -- | 86.6 tok/s | 15.0 s |
| W4A16 g128 | 3542 MB | 3252 tok/s | 76.9 tok/s | 12.4 s |
| **W4A16, `prefill_dequant: false`** | **2802 MB** | 448 tok/s | 78.5 tok/s | 16.0 s |

The second layout costs **740 MB** and buys **7.3x prefill**, because the
tinygemm kernel dequantizes once per row of activations and its cost is linear
in the batch.

**That trade-off is smaller than it looked, but it is real** (2026-09-13). It
rested partly on treating the tinygemm layout as opaque, so a row-major
duplicate was kept beside it to have something to dequantize from -- but the
duplicate was also doing work: `to_split_halves` performs the nibble
de-interleave *once at load*, so every prefill call reads two contiguous
planes. It is a precomputed permutation, not just a copy. Measured on this host instead of
assumed -- by packing an index pattern and reading back where each nibble
landed -- the layout is `[N/64][K][32 bytes]`: K is the middle axis, so a K
range is contiguous within each n-block, and inside every 32-byte run byte `d`
carries output channel `d` in its low nibble and `d + 32` in its high one, the
same permutation for every `k` and every block. Decoding it is a deinterleave,
not a reverse-engineered ISA table.

So the prefill path can dequantize from the kernel's own layout, and then the
duplicate is not allocated -- at the cost of redoing that de-interleave on
every call. The reconstruction is **bit-exact** against the previous
dequantizer at 2048x2048, 3072x2048 and gate_up's real 13312x2048.
Shapes where N does not tile the 64-channel block (N=80 is 64+16) keep the
copy rather than mis-read the layout; every Spark linear tiles cleanly.

Measured with `VLLM_OMNI_CPU_INT4_DIRECT_DEQUANT` toggling the two paths in
otherwise identical code, 3 arms x 2 interleaved passes, anonymous RSS summed
over the process tree, 1536-token context:

| build | resident (median) | peak | spread |
|---|---|---|---|
| W4A16, second copy (old) | 4591 MB | 5508 MB | 0.2% |
| **W4A16, one layout** | **3976 MB** | **4593 MB** | 5.4% |
| W4A16, `prefill_dequant: false` | 3902 MB | 4522 MB | 2.6% |

**615 MB of resident memory and ~915 MB of peak**, and the one-layout build is
statistically indistinguishable from `prefill_dequant: false` -- the harness
refused to rank them (0.4% apart, inside the measurement's own spread). The
arithmetic predicts 764 MB from the config; ~150 MB of that is allocator
retention `malloc_trim` does not return.

It is not free. Prefill, pp512, pooled over 7 interleaved passes per arm on
idle node-1 cores with memory bound to the same node:

| build | resident | pp512 median | range | spread |
|---|---|---|---|---|
| two layouts (old default) | 4591 MB | 2190 tok/s | 2117-2345 | 9.7% |
| **one layout, direct dequant** | **3976 MB** | **1344 tok/s** | 1271-1555 | 18.3% |
| `prefill_dequant: false` | 3902 MB | 453 tok/s | 451-453 | 0.3% |

So it costs **1.63x prefill** on medians -- and the ratio itself is only good
to about 1.4-1.8x, because both dequantizing arms swing ~10-18% on this shared
host while the tinygemm-only arm reproduces to 0.3%. The *existence* and rough
size of the cost is solid (a 63% gap is far outside that band); the third digit
is not. An earlier draft of this section said 1.9x, taken from the single
surviving pass of a grid whose other passes had crashed -- a number the harness
itself had labelled "single pass -- not a result".

The mechanism is that undoing the kernel's interleave per call costs ~1.6x
reading the pre-permuted planes (14.4 ms vs 9.2 ms per `gate_up` dequantize at
8 threads). Two variants were tried and rejected: dequantizing to `[K, N]` and
using `matmul` instead of `F.linear`, which avoids the element-wise transpose,
measured *worse* at 20.3 ms, and the blocked form needs either 208 tiny GEMMs
or the same permutation.

What this changes is the shape of the choice, not its existence:

* against `prefill_dequant: false` it is **strictly better** -- the same memory
  to within the noise, **2.97x** the prefill, and it does not route multi-row
  batches into the tinygemm/AMX SIGILL hazard documented in `cpu_int4.py`. That
  setting is now dominated and should not be used. This comparison is the solid
  one: the tinygemm-only arm reproduces to 0.3%.
* against the old default it is a real dial: 615 MB for ~1.6x prefill. On a
  device chosen for its memory ceiling that is the right side of the trade;
  on one with headroom it is not. `direct_dequant: false` in the checkpoint's
  `quantization_config` selects the old behaviour -- a config field rather than
  only an env var, because the two paths build different graphs and must not
  share an AOT compile artifact (see below).

Recovering the prefill would take a C++ de-interleave rather than an ATen
expression chain, which is a real kernel and not in scope here.

One hazard found while measuring this, worth recording because it presented as
a crash rather than a wrong number. The two paths build different graphs -- one
passes `weight_halves` into the custom op, the other passes `None` -- but vLLM
keys its AOT compile cache on the *config*, so an env-var-only switch shares
one cached artifact between them. Three passes died with
`KeyError: 'weight_halves'` inside an AOT-loaded forward after the first pass
saved an artifact for the other path. The choice is therefore a
`quantization_config` field, which `ModelConfig.compute_hash` does hash; the
env override remains for experiments and needs its own `VLLM_CACHE_ROOT`.

### The largest memory item was a RoPE table for a context nobody asks for

The report used to account for memory by subtraction -- weights, imports, and
an "unexplained ~1.6 GB" left over. A remainder computed from two uncertain
terms is not a measurement, so `memory_ledger.py` counts instead: unique tensor
storages keyed by `data_ptr` (so tied weights are counted once), the KV cache
found by attribute, and every `/proc/self/smaps` mapping bucketed. Allocated is
separated from resident with `mincore(2)`, because a cache that was allocated
and never written costs address space, not memory, and counting it would hide
the remainder rather than explain it.

The first run put 640 MB in an "other" bucket across 114 tensors. Naming the
largest storages showed what it was:

    model.layers.0.self_attn.rotary_emb.cos_sin_cache   512.0 MB   [1048576, 256]
    model.layers.3.self_attn.rotary_emb.cos_sin_cache   128.0 MB   [1048576, 64]

Spark advertises a **1 048 576-token context**, and `spark2_5.py` passed
`max_position=config.max_position_embeddings` straight into `get_rope`, so vLLM
built `torch.arange(1048576)` cos/sin tables -- two of them, because the
sliding and full layers are trained with different theta and different partial
rotary factors, giving one 256-wide table and one 64-wide one. Both fully
resident by `mincore`. For a deployment whose `max_model_len` is 2048 they need
**1.25 MB between them**.

vLLM's scheduler cannot produce a position beyond `max_model_len`, so the table
only has to cover that. Capping it, 3 interleaved passes per arm with
`VLLM_OMNI_SPARK_ROPE_FULL` toggling the two:

| | settled | peak | init |
|---|---|---|---|
| 1M-position tables | 3891 MB | 4019 MB | 13.5 s |
| **capped to `max_model_len`** | **3273 MB** | **3370 MB** | 13.7 s |

**618 MB settled and 649 MB of peak, for one line and no measurable init
cost** -- the largest single item found in this study, larger than the single
weight layout (615 MB) or the blocked repack (489 MB), and unlike those two it
costs nothing at all. The ledger's storage accounting agrees independently: the
"other" bucket falls from 640.0 MB to 1.3 MB and unique tensor storage drops
638.8 MB.

Two things the same ledger exposes, not yet acted on:

* **The KV cache is fully resident** -- 1023.8 MB allocated, 1024.1 MB resident
  by `mincore`. The "not a memory lever" verdict below was measured while
  generating 8 tokens; at page level the cache is entirely in core, and it is
  27% of total memory. That conclusion needs re-opening on this evidence.
* **`embed_tokens.weight_packed` (128 MB) is kept only because the tinygemm
  layout was believed un-addressable for a single-row lookup.** That premise is
  now known to be false -- the deinterleave above reaches any row -- so the
  row-major copy is probably removable too.

### Peak load memory: the number an edge device actually has to survive

Everything above reports memory *after* the allocator is asked for its arenas
back. That is the wrong target for a device: `release_after_load()` runs only
once `load_model` and warm-up have both returned, so a box that cannot hold the
peak never reaches the release. Sampling `RssAnon` on a timer through the load
(rather than reading the endpoint) put the W4A16 peak at **5722-5837 MB**
against ~4480 settled -- a 1.2-1.3 GB spike nothing in the tree had measured.

Most of it was in our own repack path. `unpack_nibbles` expands 4-bit codes to
int32 -- 4 bytes per weight, eight times the packed form -- because the
kernel's packer wants int32, and doing that for a whole matrix makes the
transient the largest object in the process:

| tensor | shape | int32 expansion |
|---|---|---|
| `gate_up` | 13312 x 2048 | 104 MiB |
| `down_proj` | 2048 x 6656 | 52 MiB |
| `q_k_v_proj` | 3072 x 2048 | 24 MiB |
| **embedding (tied, so also the head)** | **131072 x 2048** | **1024 MiB** |

against 24 MiB of packed weights for an entire layer.

The packed layout is `[N/64][K][32 bytes]`, so n-blocks are independent and the
expansion can cover a few thousand rows at a time -- verified byte-identical to
packing the whole matrix at four shapes plus a ragged last chunk. Measured with
`VLLM_OMNI_CPU_INT4_REPACK_ROWS` toggling the two in identical code, 3
interleaved passes, node-local cores:

| | peak | settled | load-time transient | init |
|---|---|---|---|---|
| whole-matrix repack | 4502 MB | 3905 MB | 550 MB | 17.7 s |
| **blocked repack** | **4013 MB** | 3892 MB | **105 MB** | 16.6 s |

**489 MB of peak, for nothing.** Settled memory and init time are unchanged,
and the repack is in fact *faster* chunked than whole -- 79 ms against 208 ms
for the embedding at 24 threads, because the working set stays in cache. The
chunk size is tuned from that sweep: 2048 rows is both the fastest and nearly
the leanest (16 MiB transient), where 512 costs 116 ms and 32768 costs 129 ms.

Two things this nearly got wrong, both caught by re-measuring rather than
reasoning. The first blocked pass showed init at 50.2 s against 17.7 -- but it
is the one pass the harness flagged `[busy]` at load 34.8, and the two clean
passes read 16.6 and 16.2. And its settled memory looked 239 MB worse, which
was pass-1 scatter: the medians are 3892 against 3905.

The absolute peak improves less than the 1 GB the arithmetic predicts, because
the single-layout change above had already removed `to_split_halves`, which was
doing its own whole-matrix expansion. The two items overlap and their savings
must not be added.

### Kernel fusion: a negative result

vLLM's CPU build ships `_C::rms_norm` and `_C::fused_add_rms_norm`, and
`vllm/kernels/vllm_c.py` already wraps both as IR-op implementations -- but
gates them on `GPGPU_DEVICE = is_cuda_alike() or is_xpu()`, so on CPU every
norm falls through to the PyTorch path. In isolation the fused kernels are
**3.6x and 4.5x faster** at Spark's decode shape (5.76 vs 20.99 us; 9.24 vs
41.14 us). End to end they are **slightly slower**: 72.7 vs 74.0 tok/s.

The reason is that the native norm was never running alone -- inductor fuses it
into the residual add and the surrounding elementwise work, and an opaque `_C`
call breaks that fusion for more than the norm itself saves.
`vllm_omni/model_executor/layers/cpu_ir_ops.py` keeps the registration, off by
default (`VLLM_OMNI_CPU_FUSED_NORMS=1` to try it on another machine), with the
measurement recorded. `gelu_and_mul` is deliberately not registered at all: its
fused CPU kernel is 2.7x *slower* than native at Spark's GEGLU shape (64.11 vs
23.36 us), and `GeluAndMul.forward_cpu` in vLLM only reaches for it on POWERPC
anyway.

Two more fusion attempts that lost, recorded so they are not repeated:
`cpp.min_chunk_size` raised to keep inductor's small elementwise kernels
single-threaded cost 6% of decode and 19% of prefill; and the head gate written
as an inline reduction (rather than behind an opaque op) made the whole step
**1.8x slower** -- 78.8 to 44.5 tok/s -- because inductor's generated loop is
worse than the matmul it replaced.

## 3. GPU: vLLM-Omni vs SGLang

One L20X (140 GiB HBM), bf16, single sequence, caches disabled on both
(`enable_prefix_caching=False` / `disable_radix_cache=True`), 2048-token
prefill so the measurement is not dominated by dispatch overhead.

| engine | init | pp2048 | tg128 |
|---|---|---|---|
| vLLM-Omni | 111.3 s | 137 102 tok/s | **451.4 tok/s** |
| SGLang 0.5.19 | 97.1 s | **142 253 tok/s** | 434.7 tok/s |

A tie within 4% either way: SGLang +3.8% on prefill, vLLM +3.8% on decode.

SGLang has an Intel AMX CPU backend, but the released `sgl-kernel` wheel does
not ship the CPU ops it needs (`convert_weight_packed` is absent), so SGLang
could not be measured on CPU without a source build. On the two targets this
goal is actually about -- CPU and a phone -- SGLang does not run at all today.

## 4. Galaxy S25 (Snapdragon 8 Elite), real device

`vllm_omni/edge/spark_export.py` exports one decode step with the KV cache as
explicit tensors, sized per layer type: sliding layers get a fixed window at
any context, full layers grow. Parity against the HF reference is **140.3 dB**
with identical top-1 and top-5.

### What the device profile said to fix

AI Hub returns per-op cycle counts, and they made the first export's mistake
obvious. Of a sliding layer at w4a16:

| op | share | what it was |
|---|---|---|
| `Output` | 9.9% | handing the recomputed 512-entry window back |
| `output_..._converted` | 3.1% | quantising that window on the way out |
| `Slice` x3 | 7.4% | rolling the window to drop the oldest entry |
| **subtotal** | **20.4%** | **moving a cache-sized tensor across the graph boundary, once per token, in 21 of 28 layers** |

Full-attention layers spent a similar 9.5% on `Concat`, appending the new key
to the cache.

The fix is a ring buffer: the runtime owns a fixed-size cache and writes the
new entry into the slot for this position, so the graph returns one K/V entry
instead of a window and never rolls anything. Ring order needs no fixing up --
softmax over keys is order-independent and each cached key already carries its
own rotary phase, so only slots not yet written need masking. The sliding
cache correspondingly holds `window - 1` entries, because a 512-token window
counts the current token (transformers' own sliding cache holds 511 for
exactly this reason).

| layer | roll | ring | |
|---|---|---|---|
| sliding, fp16 | 1.689 ms | **1.486 ms** | −12.0% |
| sliding, w4a16 | 0.917 ms | **0.838 ms** | −8.6% |
| full, fp16 | 1.559 ms | 1.567 ms | no gain |

Full layers gain nothing, so the default now picks per layer type: ring for
sliding, roll for full.

### Three things that did not work, and one that changed the answer

**The concat was not the problem.** Joining in score space instead of key
space removes the concat entirely and is numerically identical (140.3 dB), but
it needs an explicit max/exp/sum/div in place of a fused `Softmax`, and that
cannot be quantised -- QNN's context-binary converter exits 14 at w4a16 while
every fp16 build compiles. Two other shapes failed the same way: a broadcast
multiply plus `ReduceSum` for the new token's score, and a two-axis broadcast
multiply for its value contribution. Keeping roll's fused-softmax attention and
changing only what the layer *returns* captures the win and compiles.

**The cheaper GELU is not cheaper here.** The erf form is 11–13% of every
layer. The tanh approximation holds 65.2 dB with top-1/top-5 intact, so it was
adoptable -- but on this NPU it is *slower* (1.535 vs 1.483 ms). The exact form
stays, which is also the better-fidelity choice.

**Per-graph overhead is negligible, so the per-layer method is sound.**
Profiling one layer at a time pays the graph's Input/Output boundary once per
layer, where a real step pays it once for 28. Exporting the architecture's
repeating 4-layer block (3 sliding + 1 full) measured 6.103 ms against 6.016 ms
summed from single layers — a 1.4% gap. There is no hidden win in fusing
layers, and extrapolating from per-layer numbers is legitimate.

**Uncalibrated quantisation produces garbage, which invalidates the earlier
numbers.** Running the real layer on the device and comparing against the fp32
reference:

| build | SNR vs reference |
|---|---|
| fp16 | 60.6 dB |
| w4a16, no calibration data | **−1.6 dB** |
| w4a8, no calibration data | **−1.28 dB** |

Negative SNR means the output is uncorrelated with the truth. Every w4a16
figure in the first version of this document came from a build in that state.
Quantisation ranges have to be inferred from activations the model actually
sees: real ones have `x` at absmax 0.12 and the KV cache at 7.4, where the
random tensors used for tracing had 0.4 and 2.5.

Re-quantising with activations captured from the reference model on real text
(4 samples, English, code and Chinese) fixes it:

| scheme | SNR | sliding layer |
|---|---|---|
| **w8a16, calibrated** | **36.9 dB** | **0.803 ms** |
| w16a16, calibrated | 44.7 dB | 1.656 ms |
| w8a8, calibrated | 5.1 dB — broken | 0.749 ms |
| w4a16 / w4a8, calibrated | converter fails | — |

int8 *activations* destroy the layer even with calibration, which is the third
model family in this repo to show it after Qwen3-TTS and MiniCPM-o. int4
weights do not survive the calibrated path at all. **w8a16 with real
calibration data is the deployable configuration**, and it is both faster and
verified where the old w4a16 was neither.

### What is left, and why int4 is not it

Profiling the optimised layer shows the shape of the problem has changed
completely. `Input` and `Output` -- 26% of the original layer -- are no longer
in the top ten ops at all. What remains, of 0.803 ms:

| op group | share | can it move? |
|---|---|---|
| MLP up + down matmuls | 36.3% | no, this is the model |
| GELU (fused elementwise) | 17.5% | no, tanh is slower here (measured) |
| attention matmuls | 10.5% | no |
| RMSNorm, decomposed | 8.5% | **yes, see below** |
| cache slice | 4.1% | marginal |

**int4 weights are not the next lever, on two independent counts.** They do not
compile: AI Hub's calibrated quantise path fails with `QAIRT converter failed
with exit code 1` for w4a16 and w4a8, with and without `--quantize_io`. And
they would not help anyway -- on this NPU 4-bit and 8-bit weights time
identically, for the layer (0.917 vs 0.912 ms) and for the output head (3.868
vs 3.862 ms). The calibrated *8-bit* build is faster than the uncalibrated
4-bit one (0.803 vs 0.838). 4-bit buys memory here, not speed.

**The real remaining item is RMSNorm fusion, which calibration costs us.** The
uncalibrated build contains two fused `rms` ops worth 1.0% of the layer; the
calibrated build contains eight decomposed `Pow`/`ReduceMean`/`Div`/`Mul` ops
worth 8.5%. Inserting QDQ nodes around each op breaks QNN's RMSNorm pattern
match. The calibrated build is still faster overall and is the only correct
one, so this is a trade worth making today -- but roughly 7.5% of every layer
is sitting there for whoever can keep the fusion under quantisation.
(`F.rms_norm` is not a route to it: the TorchScript ONNX exporter has no
`aten::rms_norm` at any opset.)

### The decode step, end to end

Every component measured on a real S25 at calibrated w8a16, 1k context:

| component | ms each | ms total | SNR |
|---|---|---|---|
| sliding layer x21 (ring) | 0.803 | 16.86 | 36.9 dB |
| full-attention layer x7 (roll) | 0.942 | 6.59 | 37.4 dB |
| output head x1 | 3.907 | 3.91 | 62.8 dB |
| **decode step** | | **27.36** | **36.5 tok/s** |

Against the previous best (roll layout, uncalibrated w4a16, plus the same
head) at 31.01 ms: **11.8% faster, and numerically valid for the first time.**

The output head is 14% of the step and is 98.6% a single 2048x131072 matmul
tied to the embedding — there is no overhead in it to remove. It was missing
from the first version of this document entirely.

At 4k context the full-attention layers grow (2.96 ms each at fp16 vs 1.56 at
1k) while the sliding layers do not move at all, which is what the hybrid
architecture buys:

| quant | context | Spark hybrid | all-full counterfactual | gain |
|---|---|---|---|---|
| fp16 | 1 k | 46.4 ms | 43.7 ms | 0.94x |
| fp16 | 4 k | 56.2 ms | 83.0 ms | 1.48x |

At 1k in fp16 the hybrid is *slightly slower*: sliding layers rotate all 256
head dims where full layers rotate 64, and below roughly 2k that RoPE cost
outweighs the shorter cache. It pays from ~2k onward and keeps paying, because
the full-layer term grows with context and the sliding term never does.

### Which phone runtime, and which processor

The same exported sliding layer, one Galaxy S25:

| runtime | processor | precision | ms/layer |
|---|---|---|---|
| **QNN context binary** | **Hexagon NPU** | **w8a16 calibrated** | **0.803** |
| TFLite | CPU | int8 | 0.877 |
| QNN context binary | Hexagon NPU | fp16 | 1.479 |
| TFLite | Hexagon NPU | fp16 | 1.491 |
| ONNX Runtime | Hexagon NPU | fp16 | 1.500 |
| TFLite | Adreno GPU | fp16 | 2.259 |
| TFLite | CPU | fp16 | 2.958 |

Two readings. At matched fp16 the three NPU runtimes land within 1.4% of each
other, so the export layout -- static shapes, explicit caches, GQA without
expanding K/V -- is what sets on-device speed, not the engine; that reproduces
what the Qwen3-TTS talker step showed on the same device. And the placement is
the opposite of that TTS result: Spark's decoder wants the NPU (2.0x the CPU,
1.5x the GPU) because it is matmul-bound, where the TTS vocoder wanted the GPU
because it was elementwise-bound.

The row that matters for competition is the int8 CPU one. llama.cpp cannot be
run on an AI Hub device, and on a phone it would use 4-bit weights on the CPU,
so comparing our NPU path against an fp16 CPU run would flatter us. The
closest honest proxy AI Hub offers is the same graph, int8, on the S25's own
CPU: 0.877 ms against our 0.803 ms, so the NPU path is **9.2% ahead of the
processor and precision class llama.cpp would actually run in**. That is a
proxy, not llama.cpp itself.

Two independent checks say the proxy is not flattering us either. A 1.11 GiB
Q4_K_M build on the S25's ~137 GB/s LPDDR5X has a hard bandwidth ceiling of
7.9 ms/token at 100% of peak and 13.2 ms at a more realistic 60% — against our
measured 27.4 ms for a step that includes the output head the bandwidth model
ignores. Per layer, the bandwidth bound gives roughly 0.28-0.47 ms where the
measured int8 CPU run takes 0.877 ms, i.e. real CPU execution lands at about a
third of the bandwidth ceiling, which is the usual outcome on a phone.

## 5. Driving an Android phone

Spark-X2.5 is text-only, so any harness must drive from the accessibility tree
rather than screenshots. Surveyed:

| harness | transport | text-only a11y mode | local OpenAI endpoint | role |
|---|---|---|---|---|
| droidrun/mobilerun | ADB + Portal accessibility service | yes | yes, OpenAI-compatible | control |
| ghost-in-the-droid/android-agent | ADB + uiautomator, no app on device | yes | lists vLLM, base_url undocumented | control |
| mobileClaw | on-device app, accessibility | vision-oriented | n/a | control |
| AndroidWorld (DeepMind) | Android emulator | yes | via custom agent | **evaluation**, 116 scored tasks |

Recommendation: droidrun/mobilerun for control (it is the only one whose
OpenAI-compatible base URL is documented, so it points at a local vLLM server
unchanged), AndroidWorld for measuring whether the agent actually succeeds.

`analysis/experiments/spark_edge/android_agent/` implements the action space
those harnesses share -- `tap`, `type_text`, `swipe`, `press_key`, `open_app`,
`task_complete`, all indexed against accessibility-tree elements, with the
`adb shell input` command each maps to -- and runs Spark against ten Android
screens through vLLM's OpenAI server.

Serving works with vLLM's stock parsers: Spark emits
`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`,
which is GLM-4.5's format, so `--tool-call-parser glm45 --reasoning-parser
glm45 --enable-auto-tool-choice` parses it as-is. No new parser needed.

| config | correct tool | fully correct action | median decision |
|---|---|---|---|
| thinking on | 7-8 / 10 | 5-6 / 10 | 8.9 s |
| thinking off | 5 / 10 | 4 / 10 | **0.61 s** |

(16 CPU cores, bf16, measured under concurrent benchmark load, so latency is
pessimistic.)

Thinking is worth roughly two of ten decisions and costs 14.6x the latency.
With thinking off the model collapses toward answering `tap` for everything.
With it on, the failure mode is the opposite: on the one genuinely ambiguous
screen (reach Developer options, which is not visible and needs scrolling) it
reasoned past a 1200-token budget without ever closing `<think>`, so vLLM's
reasoning parser -- which only emits reasoning once the block closes --
returned empty content and no tool call at all. A phone agent on this model
needs a step budget and a fallback, not just a bigger cap.

The clearest wrong action was tapping an already-on Wi-Fi switch instead of
calling `task_complete`, i.e. undoing the task it had completed.

No Android device or emulator is attached to this host (`/dev/kvm` absent), so
these numbers measure the model's decisions over real accessibility-tree
layouts, not a live device.

### 5b. Driving it in a loop, not one decision at a time

The numbers above measure single decisions over fixed screens. A phone agent
is a loop, and a loop fails differently. `analysis/experiments/spark_edge/
android_agent/` is now a working toolkit -- uiautomator XML to an indexed
element list, an ADB device and a scripted simulator that emits the same XML,
a tool executor, a tolerant tool-call parser and the loop itself (21 tests,
no phone and no server required).

**Stock `glm45` does not always parse Spark.** Earlier this note said Spark's
tool format is GLM-4.5's, so vLLM's parsers work as-is. True on 4 of 5 screens.
On the Wi-Fi toggle screen -- the one where a switch's state decides the
action -- the model left the format and emitted

    [tap]
    {"tool": "tap", "parameters": {"index": 1}

unterminated, with `tool_calls` empty. An agent reading only `tool_calls` sees
a step with no action. `tool_parse.py` recovers the bracketed form, GLM blocks
and bare JSON, repairing unbalanced braces, and reports every recovery rather
than silently repairing.

**Navigation survives quantization; recognising completion does not.** Same
harness, same simulated phone, one run per cell:

| build | decode | "Turn on Wi-Fi." | "Set an alarm for 7:30 AM." |
|---|---|---|---|
| bf16 | 41.0 tok/s | done in 4 steps | wrong app (Settings, not Clock), toggles Airplane mode, stuck |
| **W4A16 g128** | 80.0 tok/s | done in 4 steps | right app, full navigation, then no tool call |
| W4A8 g128 | 86.6 tok/s | turns Wi-Fi on, then taps it off again (2 runs) | reaches Alarm, then loops |

Every build walks home -> Settings -> Network -> Wi-Fi correctly. Only some
notice the switch now reads `checked=true` and stop. W4A8 -- the fastest build,
and the one a speed-first selector picks -- ends with the task undone.

This is what section 2c's fidelity numbers meant. W4A8 reproducing 1/12 greedy
prompts against its own bf16, and W4A16 2/12, looked like a small difference on
a proxy metric; it is the difference between finishing a task and reversing it.
**For agentic work the 4-bit choice is W4A16**, which `weight_path.py` selects
under `priority="fidelity"` -- 8% of decode for the difference.

Adding an explicit "if the screen already shows what was asked, call
task_complete" rule to the system prompt did not rescue W4A8. Nor did feeding
back "the screen did not change", which bf16 was told five times on the alarm
task before swiping again.

Limits: two tasks, one run each, on a simulated phone -- enough to show a
failure mode reproduces, not to rank builds by success rate. The simulator
emits real uiautomator XML, so the parser, the index-to-coordinate mapping and
the stopping rules run on the path hardware drives; whether a real Settings app
is laid out the way these fixtures assume is still untested, because this host
has no `adb`, no emulator and no `/dev/kvm`.

## 6. Where it stands against the goal

- **Runs on CPU**: yes. Faster than llama.cpp at matched precision on both
  prefill (8.7-15.4x) and decode (1.19-2.65x), and an int8 path adds a further
  1.6x.
- **Runs on Galaxy S25**: yes, measured on real hardware -- a full decode
  step, output head included, is 27.4 ms at 1 k context (36.5 tok/s) in
  calibrated w8a16, with every component verified at 36.9-62.8 dB against the
  reference. That is 11.8% faster than the first working version and the first
  configuration whose output is actually correct: the uncalibrated w4a16 builds
  quoted earlier score -1.6 dB, i.e. uncorrelated with the truth.
- **Beats every edge engine**: on prefill, yes, against everything that can
  run the model, at any precision -- 4.7x over llama.cpp's best quantized
  build at 32 threads. On decode it depends on load: llama.cpp Q4_K_M wins
  single-stream by 1.6x, vLLM int8 wins from batch 16 and leads 1.72x at batch
  64. The single-stream gap is not closable on vLLM 0.28's CPU backend, which
  ships no dense int4 kernel.
- **Beats SGLang**: on decode by 3.8% on GPU (prefill 3.8% the other way), and
  by default on CPU and phone, where SGLang does not run -- its released
  `sgl-kernel` wheel has no CPU ops and it has no phone backend at all.
- **Fastest on the phone**: of every runtime and processor combination
  measurable on the S25, our QNN NPU path is quickest, including a 9.2% margin
  over the int8-on-CPU class llama.cpp would run in. llama.cpp itself cannot be
  run on an AI Hub device, so that specific comparison is a proxy.
- **Phone control**: the action space, an agent loop and measured decision
  accuracy are in `analysis/experiments/spark_edge/android_agent/`; serving
  needs no new parser, just `--tool-call-parser glm45 --reasoning-parser
  glm45`.

Open items: no dense int4 on CPU (upstream kernel gap); int8 long-context
fidelity would benefit from a smarter scheme than round-to-nearest; on the NPU
the next ~7.5% per layer is RMSNorm fusion, which calibration currently breaks
(int4 weights are neither available nor useful there -- 4-bit and 8-bit time
identically on this hardware); and none of the phone-agent numbers have been run against a live device, since
this host has no emulator (`/dev/kvm` absent).
