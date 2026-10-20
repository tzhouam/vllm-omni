# Optimization plan: inference and harness

Written 2026-09-12 against [spark_edge_support.md](spark_edge_support.md), then
revised after an independent review by `gpt-6-astra` through the Codex hook
(read-only against these checkouts; the sandbox could not start because user
namespaces are disabled on this host, so it ran unsandboxed under explicit
read-only instructions and the checkouts were verified clean afterwards — the
same procedure recorded in [README.md](README.md) for the 2026-09-08 study).

The review rejected the first draft's ordering and several of its expected
gains. Most of that was correct and is applied below; §"Review record" lists
what I accepted, what I corrected with new measurements, and why.

Every item starts from a measurement in this tree, states the expected gain,
and says how it would be verified — because on this host a claim without a
verification method is how three earlier conclusions turned out to be wrong.
**A fourth is now on the list**: the first draft of this plan mis-described the
740 MB it proposed to remove.

## Where the time actually goes

W4A16 g128 decode, 24 threads, per-op profile (reproducible to ~5%, where
throughput swings ~30%). The first draft built this table from two different
builds — 5.91 ms is the **W4A8** GEMM, 6.85 ms the W4A16 one
([support:193](spark_edge_support.md)) — so the step it described (11.8 ms) was
neither configuration. The W4A16 step is **12.51 ms**:

| | ms | share of 12.51 |
|---|---|---|
| int4 GEMM (tinygemm) | 6.85 | 55% |
| inductor elementwise (norms, rope, GEGLU, residuals) | 2.20 | 18% |
| vLLM attention ops (4 per layer × 28) | 1.31 | 10% |
| custom-op dispatch boundary (W4A16 only) | 0.55 | 4% |
| unattributed remainder (sampling, view/slice churn) | 1.60 | 13% |

Two corrections to how this table was read. The GEMM runs near this socket's
achievable bandwidth for *its own* shape, which says the GEMM is memory-bound;
it does **not** say the other 45% is framework overhead — attention, softmax,
norms and sampling contain real arithmetic. And the per-step savings of W4A8
over W4A16 do not add: 0.94 ms (GEMM) + 0.48 ms (dispatch) = 1.42 ms predicted
against 12.51 − 11.55 = **0.96 ms** measured. Something offsets them, and no
item below may assume its saving reaches the step time unchallenged.

## Where the memory actually goes

Anonymous RSS by phase ([support:324](spark_edge_support.md)):

| phase | W4A8 (GPTQ) | W4A16 (tinygemm) |
|---|---|---|
| before import | 12 MB | 12 MB |
| after `import vllm_omni` | 513 MB | 515 MB |
| after engine construction | **7338 MB** | 5023 MB |
| after releasing free arenas | 3510 MB | 3411 MB |

The first draft treated the post-release figure as the number to reduce. That
is the wrong target for an edge device: **7338 MB is the peak that has to fit**,
and `release_after_load()` runs only after `load_model` and warm-up have both
returned ([patch.py:730](../vllm-omni/vllm_omni/patch.py)). A device that cannot
survive the load never reaches the release. This reordered the plan.

The draft's weight breakdown ("0.86 GB quantized + 0.54 GB bf16 embedding") is
also wrong for the checkpoint actually measured: `spark2_5-int4-g128/config.json`
sets `quantize_embedding: true` with `tie_word_embeddings: true`, so there is no
bf16 embedding, and the persistent prefill duplicate was left out of the
breakdown entirely. There is therefore no trustworthy ledger to subtract from —
which is why the "unexplained 1.6 GB" item is now a prerequisite, not a
long-shot.

---

## Inference

### I0. Peak load-time memory — *now first*

7338 MB peak vs 3510 MB steady is the gap that decides whether a device can run
this at all. The peak comes from holding the checkpoint, the dequantized
intermediate and the repacked layouts alive at once, per layer, with freed
arena pages not yet returned.

**Do**: (a) release arenas *during* the load, per layer or per N-layer block,
rather than only after `load_model` returns; (b) repack blockwise so no
whole-tensor bf16 intermediate exists; (c) as the durable fix, write the packed
layout to a deployment artifact at build time so the device loads what it runs.

**Expect**: peak toward the steady 3.4–3.5 GB. Unquantified deliberately — the
mechanism is understood, the size is not.

**Verify**: sample `RssAnon` on a timer through load (0.2% reproducible), report
the maximum, not the endpoint. The harness measures endpoints today and would
not have caught this.

### I1. One packed weight layout, not two — *replaces the draft's I2*

The W4A16 build keeps a second copy costing **740 MB** (3542 → 2802 MB) and
buying **7.3× prefill** (3252 → 448 tok/s). The draft described this as "prefill
unpacks to bf16" and proposed a tiled dequant kernel to avoid the bf16 copy.
That was wrong about the artefact. `_finalize_int4` keeps `weight_halves` — a
**second packed 4-bit copy** — plus cloned `weight_group_scale` and
`weight_group_zero`; the bf16 matrix is already a per-call temporary and
`weight_packed` is deleted
([cpu_int4.py:344,375](../vllm-omni/vllm_omni/model_executor/layers/quantization/cpu_int4.py)).

So the cheap fix is not a new GEMM. It is to **dequantize directly from the
tinygemm layout** and delete `weight_halves` and the duplicated scale/zero,
keeping the existing single `F.linear`. The only reason that was not the
original design is that the tinygemm layout was believed opaque. It is not —
measured on this host (`torch._convert_weight_to_int4pack_for_cpu`, AVX-512):

- The packed buffer is `[N/64][K][32 bytes]`: **K is the middle axis**, so a
  K-range is contiguous *within each n-block of 64*.
- Reassembling a K-tile from those per-n-block segments is **byte-identical** to
  packing that K-slice directly (verified exactly at N=2048, K=2048 for tile
  widths 128/256/512), and split-K through tinygemm reconstructs the single-call
  result at **51.5 dB**.
- The lane order inside a 32-byte run is a **fixed 64-entry interleave** — low
  nibble = output channel *d*, high nibble = *d+32* — identical across every k
  and every n-block. So decoding is a deinterleave, not a gather or a
  reverse-engineered ISA table.

**Do**: point the existing dequantizer at the tinygemm layout via that
deinterleave; drop `weight_halves`, `weight_group_scale`, `weight_group_zero`.

**Expect**: most of the 740 MB back **at unchanged prefill**, because the GEMM
shape and reduction order do not change — only where the nibbles are read from.
The largest single temporary stays one dequantized `gate_up` (N=13312, K=2048,
bf16 = **52 MiB**), which already exists today.

**Verify**: `RssAnon` (0.2%) plus pp512, three interleaved passes, and an SNR
assertion against the current dequant path — the check that caught the original
layout bug.

**Risk**: tails. N not divisible by 64 or K odd need the scalar path; the
deinterleave rule above is verified only for full blocks.

**Not doing first**: the draft's K-tiled dequant-and-accumulate. Even granting
the layout, calling oneDNN once per K-tile means the M×N accumulator is re-read
and re-written per tile — for `gate_up` at M=512 that is ~26 MiB of fp32
accumulator × 15 extra tiles ≈ 780 MiB per layer per prefill, which is not the
"few MB of scratch" the draft claimed. Avoiding that traffic means keeping the
accumulator panel in registers across the K loop, i.e. writing a blocked GEMM
microkernel rather than calling a library per tile. If temporaries still need
cutting after the item above, **N-blocking is the better experiment** — disjoint
output columns need no cross-tile accumulation at all.

### I2. Measure the native non-AMX 4-bit path before building anything

`weight_path.py:14` calls tinygemm "the only option" without AMX. That is false:
when `layer.use_w4a8` is off, vLLM 0.28 falls through to `ops.cpu_gemm_wna16`
([cpu.py:202](../vllm/vllm/model_executor/kernels/linear/mixed_precision/cpu.py)),
a vector-ISA W4A16 kernel that has never been measured here. Both the ARM story
and the "fidelity" weight path rest on an untested assertion.

**Do**: benchmark `cpu_gemm_wna16` against tinygemm at Spark's shapes, decode
and prefill. **Verify**: `bench_int4_kernels.py`. **Then** fix `weight_path.py`.

This is a day's work that could invalidate a kernel project, so it comes first
among the kernel items.

### I3. Asymmetric W4A8 — a fidelity experiment already supported by the kernel

The report treats W4A8's zero point pinned at 8 as a kernel limitation. It is
not: `_process_gptq_weights_w4a8` uses the checkpoint's zero points whenever
`config.zero_points` is set, and synthesizes the constant only otherwise
([cpu.py:98](../vllm/vllm/model_executor/kernels/linear/mixed_precision/cpu.py)).
Our exporter chooses `"sym": true`
([quantize_gptq_int4.py:102](experiments/spark_edge/quantize_gptq_int4.py)).
W4A8 is the fastest path and the worst on task completion (1/12 greedy; the
agent navigates correctly then fails to stop). An asymmetric export is a
config change, not a kernel project.

**Do**: re-export with `sym: false` and zero points, re-run fidelity and the
agent suite. **Expect**: unknown, but it is the cheapest fidelity experiment
available and it tests the *stated* cause of W4A8's error.

### I4. A better quantizer — only after I3, and on an honest bit budget

Ours reproduces 2/12 greedy prompts; llama.cpp Q4_K_M manages 4/12. The draft
called this "the same bit budget". It is not. Ours is **4.25 bits/weight**
(4 + two bf16 per 128-weight group; `weight_path.py` says 4.16, which is wrong);
plain Q4_K is **4.5 bits** over groups of 32, and Q4_K_M additionally promotes
some tensors to higher-bit types. Group size, per-tensor type selection and
rounding are all confounded, so "6-bit sub-block scales cause the gap" is an
attribution the experiment never isolated.

**Do**: first close the budget gap cheaply (g64 and/or fp16 scale+zero, which is
a config change) and re-measure; only then implement error-compensated rounding
(GPTQ-style, calibration data exists) or AWQ-style scaling.

**Verify**: greedy divergence *and* the scored agent suite — but only after H1
below, because today's suite cannot rank builds.

**Risk**: may produce nothing. Timebox it; a null result is worth recording.

### I5. Attention op-chain fusion — worthwhile, but not "highest confidence"

Four ops per layer, 112 dispatches, **1.3085 ms/step** measured
([profile_w4a16_g128_p2.json](experiments/spark_edge/profile_w4a16_g128_p2.json)):
`unified_attention_with_output` 0.4379, `unified_kv_cache_update` 0.3948,
`cpu_attention_with_kv_cache` 0.3471, `cpu_attn_reshape_and_cache` 0.1287.

These are **two wrapper/kernel pairs**, not four independent kernels. Wrapper
self time is 0.833 ms — 64% of the total — so overhead does dominate at this
shape: at batch 1 the cache-update OpenMP loop runs over tokens × KV heads, i.e.
**two iterations**, while both paths independently rebuild cache views and the
wrapper allocates a dummy tensor for compile ordering.

But the draft's supporting arithmetic was about a workload it never profiled.
"28 MB moved" assumes every layer attends 512 tokens; 7 of 28 layers are full
attention, and `profile_cpu_step.py` profiles a short prompt generating 64
tokens — nowhere near a filled window.

**Expect**: revised down. Eliminating *all* wrapper time saves 0.833 ms, and
some wrapper is required, so the draft's 0.6–0.9 ms demanded savings from the
C++ bodies too. Target **0.3–0.5 ms** (2–4% of step) from consolidating context
and cache-view handling through the existing `forward_includes_kv_cache_update`
interface, plus a serial path for the two-iteration cache update.

**Verify**: per-op profile **and** unprofiled step latency — one outer call that
still launches two threaded kernels removes neither thread launch, and fewer
profiler events alone make this look better than it is.

### I6. Import cost — an upper bound, not a saving

`import vllm_omni` costs 19.6 s against `import vllm`'s 5.2 s; auto
image-processing (4.6 s) and the `openai` client (1.3 s) are identifiable inside
it. The draft promised 4–6 s per process. That 5.9 s is an **upper bound on
removable import work**, not a demonstrated reduction in time-to-ready: deferred
imports often just move the cost later. The init timer starts *after* imports
([bench_init_memory.py:44](experiments/spark_edge/bench_init_memory.py)), so the
current harness cannot measure this at all. And `patch.py` uses `ModelConfig` at
import time to install a descriptor patch
([patch.py:54](../vllm-omni/vllm_omni/patch.py)) — deferring it means changing
how that patch installs, not moving a statement.

**Do**: first make the harness measure **process start → ready**; then defer.
**Verify**: that end-to-end number, with `-X importtime` as the diagnostic.

### I7. A real memory ledger

Prerequisite for any claim about a remainder. The draft's tools do not do the
job: `smaps_rollup` is an aggregate (use `smaps` per mapping), and `tracemalloc`
does not attribute native tensor storage or allocator arenas.

**Do**: count unique tensor storages (both embedding layouts included), then
`smaps` by mapping for the rest. **Expect**: a decision about whether the
remainder is reducible — the draft's "unexplained 1.6 GB" was a subtraction from
a breakdown now known to be wrong.

### I8. Resolve the `prefill_dequant: false` correctness hazard

`cpu_int4.py:55` records that running tinygemm on a multi-row batch and then
`cpu_attention_with_kv_cache` in the same forward dies with SIGILL — an AMX tile
configuration bug — and sets `dequant_threshold: 2` to avoid it. But
`prefill_dequant: false`, the 2802 MB build the report calls the smallest
resident configuration, routes multi-row batches to exactly that path. Either
the 448 tok/s measurement and the crash report disagree about build or shape, or
that configuration is not deployable. **Do**: reconcile before quoting 2802 MB
as a build anyone can use.

### I10. Megakernel / persistent thread pool -- the biggest measured headroom

Raised after the review. On GPU a megakernel (Hazy Research's low-latency
Llama, Mirage's MPK) fuses a whole forward pass into one launch, removing
per-kernel launch cost and the pipeline bubble where the next layer's weights
cannot start loading. There are no kernel launches on CPU, so the analogues
are **OpenMP fork/join per op**, **ATen dispatch**, and the same cross-op
prefetch bubble.

This is not a hypothetical for us: **llama.cpp already is one.**
`ggml_threadpool` starts its threads once and they walk the entire graph
together, synchronising at `ggml_barrier` through an atomic spin
(`ggml-cpu.c:490,576`). vLLM enters a separate `omp parallel` region inside
each kernel. So the comparison this whole study is built on is, in part, a
comparison of execution models.

Measured here, so the prize is sized before anything is built:

* **Parallel-region cost: ~1.2 us** at 24 threads (a just-above-grain-size op,
  24 threads vs 1). Spin-waiting does not help -- `OMP_WAIT_POLICY=ACTIVE` plus
  `GOMP_SPINCOUNT=infinite` moved it 1.23 -> 1.18 us, inside noise -- so this
  is barrier and scheduling cost, not thread wake-up, and the one-env-var
  approximation of a persistent pool buys nothing.
* **Regions per decode step: a few hundred.** 113 quantized GEMMs and 112
  attention ops certainly parallelise; most elementwise tensors at batch 1 are
  below PyTorch's grain size and run serially, so the 764 candidate calls in
  the profile overstate it. At ~250-350 regions this is **0.3-0.4 ms, 2-3%** --
  real, but the same money as I5, which is far cheaper.
* **The bubble is the prize, not the barriers.** The GEMM sustains 60% of the
  measured read ceiling and the whole step 33%. A persistent kernel that keeps
  the weight stream running across op boundaries is the mechanism that could
  close part of a **2.76 ms** gap.

One of our own decisions works against this: `cpu_int4_linear` is deliberately
opaque to dynamo so the batch-size branch never becomes a guard on vLLM's
traced graph. That also stops inductor fusing anything across it. The guard
stability was worth it, but it is a cost worth naming.

**Do**: *not* build a megakernel next. The honest read is that the fork/join
half is small and already covered by I5, and the streaming half needs the
execution model replaced -- a persistent pool plus cross-op pipelining inside
vLLM's CPU backend, fighting its attention backends, KV cache management and
torch.compile integration. That is weeks, and it reimplements ggml.

What *is* worth doing, in order, because each one tests the same thesis for a
fraction of the cost:

1. **Find out why the GEMM is at 60%.** It is not obviously bandwidth. The
   tinygemm kernel dequantizes once per activation row, so part of that time is
   arithmetic, and each of the 113 calls streams only a few MB -- too little for
   the prefetcher to reach steady state. Attribute it before optimising it.
2. **Measure `cpu_gemm_wna16`** (I2) and W4A8's 69% against the same ceiling.
   If one kernel is materially closer, the answer is a better kernel, not a
   different execution model.
3. **I5**, which is the cheap slice of the fork/join half.

**Verify**: per-op profile plus the streaming-bandwidth figure above, which is
now the denominator any kernel claim should be quoted against.

### I9. Explicitly not planned

- **More kernel fusion.** Four attempts, four losses: vLLM's fused CPU RMSNorm
  is 3.6–4.5× faster in isolation and slower in situ (inductor already fuses the
  native form with the residual add); fused `gelu_and_mul` 2.7× slower at
  Spark's shape; `cpp.min_chunk_size` −6% decode/−19% prefill; an inline gate
  reduction −44%. I5 is op-*count* reduction, a different mechanism.
- **MiniCPM-o CPU overlay.** Dropped. The 8 B thinker at bf16 is ~19 GB, so it
  buys no edge deployability; and the draft's "three stages should beat the
  two-stage 2.1×" is not how overlap works — the bound is ΣT/max(T), so stages
  of 90/5/5 s give 1.11×. (That the measured two-stage ratio is 64.57/30.54 =
  2.114× when the bound for two tasks is lower means something else changed too,
  which the init work should explain before anyone quotes it.)
- **Thread count and warm-up batch size as memory levers** — 64 MB and 2 MB.
- **KV-cache sizing is *not* on this list any more.** The 188 MB that retired it
  was measured generating 8 tokens. At 32 k context `kv_budget.py`'s own formula
  gives 1792 MiB flat vs 469 MiB hybrid — a 1323 MiB difference. It was measured
  at a context where no cache is touched, which is not evidence that it never
  matters. (`CpuPlatform.support_hybrid_kv_cache()` returns `True` in vLLM 0.28,
  so the report's flat accounting needs a configuration explanation too.)
- **Speculative decoding** for tg128: it would inflate a fixed-prompt
  `ignore_eos` benchmark without representing anything. Measure it on the agent
  suite or not at all.

---

## Harness

The review's sharpest finding is that the harness cannot currently support the
comparisons the inference work needs. Correctness first, statistics second.

### H1. Fix the scored suite before running it more

- **The alarm oracle is wrong.** `add_alarm` passes if any typed string contains
  `"7"` ([tasks.py:101](experiments/spark_edge/android_agent/tasks.py)) — not
  7:30, not AM, not that an alarm was saved. Typing `7` anywhere scores a pass.
- **The toggle task scores final state only**, so turning a setting off and back
  on counts as success.
- **The scored suite is simulator-only**: task builders and predicates are typed
  to `SimulatedDevice`. The real-device *episode* path exists; real-device
  scoring and reset do not.
- **Repeats at temperature 0 over fixed fixtures measure repeatability, not
  generalization** ([agent.py:48](experiments/spark_edge/android_agent/agent.py)).

**Do**: correct the oracles; add paired task/layout/state variations so N runs
are N trials; implement real-device scoring and reset.

This supersedes the draft's "N≥5 runs per task" — more repetitions of a wrong
oracle produce a confident wrong answer, which is the failure mode this project
already has three instances of.

### H2. Fix measurement semantics

- `run_grid`'s docstring says it rotates arms; it iterates `for pass: for arm in
  arms`, i.e. the **same order every pass**
  ([bench_harness.py:160](experiments/spark_edge/bench_harness.py)). It
  interleaves but does not rotate, so position bias survives.
- `summarize` reports `best = max(values)` for every metric
  ([bench_harness.py:123](experiments/spark_edge/bench_harness.py)) — wrong for
  memory and init, where lower is better. Published "best" memory figures are
  the worst pass.
- `profile_cpu_step.py` wraps all of `generate` (prefill + sampling) and divides
  by `tg`, while the wall-clock path subtracts independent best-of-three
  measurements. These are different quantities presented as one.
- Profile JSONs record no model path, context length or build; saved W4A8 and
  W4A16 tags both read `bfloat16/None/eager=False`, and some saved W4A8 profiles
  still contain the retracted bf16-head build with nothing marking them invalid.

**Do**: rotate arm order per pass; take `best` by metric direction; separate
prefill from decode in the profiler; stamp provenance (model path, config hash,
commit, thread set) into every artifact and mark the retracted ones.

### H3. Measure what the inference items need

- **Process start → ready**, not engine-construction time (required by I6).
- **Peak `RssAnon` through load**, sampled, not the endpoint (required by I0).
- **Memory at realistic occupied context**, not 8 generated tokens (required to
  re-open KV sizing).

### H4. Contention guard

Two sessions on the same cores wrecked several measurements and one grid's
control drifted 30% mid-run. A cooperative lock file only excludes cooperating
benchmarks, and load average > 8 identifies neither *which* cores are busy nor
whether the load is ours.

**Do**: lock file plus a pre-flight check on per-core utilization for the
*target* core set, recorded in the artifact.

### H5. Regression gates on measured bands

The draft proposed an 8% throughput band. That 8% is a configured constant in
the harness, not an estimated interval, and it sits against acknowledged ~30%
run-to-run variation — it would fire constantly or never.

**Do**: a checked-in expectations file whose bands come from the observed
distribution per metric (`RssAnon` ~0.2%, per-op self time ~5%, end-to-end
throughput wide enough to be meaningless, which is itself the argument for
gating on per-op time and memory instead).

### H6. A denser fidelity metric

Greedy divergence over 12 prompts saturates and cannot separate "slightly worse"
from "much worse". Top-1 agreement and KL against the bf16 reference over a few
thousand teacher-forced tokens, symmetric with `llama-perplexity
--kl-divergence` so llama.cpp stays comparable. Needs `llama-perplexity` built.

Bring this forward: I3 and I4 are fidelity experiments and cannot be judged on a
saturated metric.

### H7. Make profiling the default for small differences

Per-op self time resolves ~5%; throughput on this host ~30%. Have
`bench_harness.py` say so when an arm difference falls inside the noise band,
and point at `profile_cpu_step.py`.

### H8. Purge retracted numbers from executable source

Retractions are still being shipped as documentation:
`kv_budget.py:9` advertises the withdrawn **8329 → 5254 MB**;
`weight_path.py` reports W4A16 as **4.16 bits** (it is 4.25), sets
`fused_cpu_norms=True` in three profiles against the recorded negative result
and the `"0"` default, and calls tinygemm "the only option" (see I2).

---

## Order

1. **H1, H2, H8** — the suite and harness cannot currently rank builds, and two
   modules still assert retracted numbers. Everything downstream depends on this.
2. **H3** — add the three measurements the inference items are defined in terms
   of (start→ready, peak RSS, memory at context). Then **H4**, **H5**, **H6**.
3. **I0** (peak load memory) — the actual edge blocker.
4. **I7** (memory ledger), **I6** (imports) — now measurable.
5. **I2** (measure `cpu_gemm_wna16`) — may invalidate a kernel project; cheap.
6. **I1** (single packed layout) — largest confirmed memory win at unchanged
   prefill, and the layout facts it needs are now measured.
7. **I8** (SIGILL reconciliation), then **I3** (asymmetric W4A8) and **I4**
   (quantizer) on a fixed fidelity metric.
8. **I5** (attention wrappers) — bounded overhead experiment, revised target.

The draft ordered by "whose verification method already exists". The revision
orders by **whether the verification method is correct**, which is not the same
question and is the one this project keeps getting wrong.

---

## Progress

Started 2026-09-13, in the plan's own order. Items are struck through here only
when measured, not when coded.

**Step 1 -- H1, H2, H8: done.** Commits `6424c69` (oracles and measurement
semantics), `27b94007` (retracted numbers out of `vllm-omni` source), `28aa63c`,
`b713d6e`, `4285f92`. Agent tests 27 -> 38, harness tests 9 -> 34, plus 7 for
the new fidelity metric.

**Step 2 -- H3 to H7: done.** And it found a fifth measurement problem, of the
same kind as the first four:

> **The memory benchmark was reading the wrong process.** It took
> `/proc/self/status`, which is the whole story only when the model is
> in-process. That happens when `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps the
> executor at `uni`; `vllm/platforms/cpu.py:273` promotes it to `mp` otherwise.
> Every earlier run in this study set that variable, so **every published
> memory figure is an in-process figure**. At vLLM's defaults the weights load
> into a worker the benchmark never looked at: a smoke run reported 796 MB
> resident for a model whose weights alone are 1.4 GB. Memory is now summed
> over the process tree.

First numbers from the rebuilt benchmark (W4A16 g128, 24 threads, 2048 ctx,
`VLLM_CPU_KVCACHE_SPACE=1`, 2 interleaved passes):

| | in-process (`uni`) | default (`mp`) |
|---|---|---|
| peak RssAnon, tree | 5722-5837 MB | 6068-6078 MB |
| after arena release | ~4480 MB | ~5544 MB |
| processes | 1 | 4 |
| `ready_s` (process start to serving) | 78.5-111 s | 88-153 s |
| of which imports | ~29-30 s | ~30 s |

Three things follow. **The peak is 1.2-1.3 GB above the settled figure**, which
is I0's premise and is now measured rather than asserted. **vLLM's default
process layout costs ~240 MB of peak and ~1.1 GB of settled memory** over the
in-process one, which no published number reflected. And **`ready_s` is far
above `init_s`** -- imports are ~30 s, not the 19.6 s the plan quotes, so I6 is
worth more than it claimed and was previously unmeasurable.

These are not comparable to the report's 3411 MB: that row was taken
in-process, with a sized KV budget, generating 8 tokens. Which is the argument
for the provenance stamping added in the same step.

**Step 3 -- I0 (peak load memory): done.** The driver was in our own repack
path, not in vLLM: `unpack_nibbles` expands 4-bit codes to int32, and doing
that a whole matrix at a time makes the 131072-row tied embedding a **1024 MiB
transient**. Repacking in n-blocks (the layout makes them independent) is
byte-identical and cuts the load-time transient from 550 to 105 MB, **489 MB
off the peak**, with settled memory and init time unchanged -- chunked
repacking is in fact faster than whole-matrix (79 ms vs 208 ms for the
embedding). Chunk size tuned from a sweep rather than guessed.

**Step 5 -- I2 (measure `cpu_gemm_wna16`): done, and it did not invalidate
anything.** The fallback is real, so `weight_path.py`'s "only option" was
false, but at 32.3 tok/s it is **2.2x slower than tinygemm** (71.6) and 2.3x
slower than W4A8 (74.4), all on the same cores with only the env var
differing. The routing was right; it now cites a measurement. Caveat recorded:
`_get_isa_hint` returns `"amx"` whenever AMX is present, so this measured the
kernel's best path on this host, not the `"vec"` path an AMX-less machine takes.

**Step 6 -- I1 (single packed layout): done.** The layout
facts the item needed are now established on this host rather than assumed
(see I1 above), and the reconstruction is **bit-exact** against the existing
dequantizer at 2048x2048, 3072x2048 and 256x512 -- including `gate_up`'s real
13312x2048. `weight_halves` and the duplicated scale/zero are no longer
allocated where N tiles the 64-channel block, which every Spark linear does
(2048, 3072, 13312, 131072); shapes like N=80 keep the copy rather than
mis-read the layout. `VLLM_OMNI_CPU_INT4_DIRECT_DEQUANT=0` restores the old
path, which is how the two are being A/B'd with identical code.

*Process note, since this study keeps being bitten by measurement: I edited
`cpu_int4.py` while an unrelated grid was running, so that grid's later passes
used different code from its earlier ones. Those rows were discarded and the
comparison re-run under the escape hatch instead.*

---

## Review record

**Accepted and applied** (each verified in source before applying): the step
table conflated W4A8 and W4A16 (W4A16 is 12.51 ms, and the remainder is 1.60 ms
not ~1.9); bandwidth saturation of the GEMM does not make the rest overhead;
per-step savings were added when the measured total contradicts the sum; the
740 MB is a second *packed* copy plus duplicated scale/zero, not a bf16 matrix,
so the cheap fix is direct dequant rather than a tiled kernel; per-tile oneDNN
calls incur accumulator traffic the draft ignored; 4.25 ≠ 4.16 bits and Q4_K_M
is 4.5 bits with mixed types, so the fidelity comparison was never
budget-matched; the import figure is an upper bound and the init timer starts
after imports; the MiniCPM-o stage-count extrapolation is invalid and the item
does not serve edge deployability; peak load memory, not steady memory, is the
edge blocker; KV sizing was retired on an 8-token measurement; W4A8 supports
asymmetric zero points, so the report mistook our export choice for a kernel
limitation; `cpu_gemm_wna16` is a real non-AMX path and `weight_path.py`'s "only
option" is false; `smaps_rollup`/`tracemalloc` cannot deliver the attribution
I7 needs; `prefill_dequant: false` bypasses the documented SIGILL guard; the
harness does not rotate arm order, takes `max` as "best" for every metric, mixes
prefill into per-step timing and stamps no provenance; the alarm oracle accepts
a bare `"7"`, the toggle task scores final state only, and the scored suite is
simulator-only.

**Corrected with new measurements.** The review held that the tinygemm layout
requires implementing "the ISA-specific lane mapping and tail handling" and that
a K interval "is not a plain column slice", concluding a tiled decoder is
expensive to get right. The first half is understated and the second is true but
not the obstacle it implies. Measured here: a K-tile is contiguous *within each
n-block of 64* and reassembling those segments is byte-identical to packing the
K-slice (N=2048, K=2048, tile widths 128/256/512), split-K reconstructs the
single call at 51.5 dB, and the lane order inside each 32-byte run is a fixed
interleave (low nibble → channel *d*, high → *d+32*) that is constant across
every k and n-block. So decoding needs a deinterleave, not a reverse-engineered
table — which is what makes I1's direct-dequant fix small enough to be the first
thing tried. The review's conclusion (do direct dequant, not the tiled kernel)
is unchanged and adopted; its reasoning about why is superseded.

The review's accumulator-traffic objection assumes K-outer with a full M×N
accumulator, which a register-blocked GEMM would not pay — but since the draft
proposed calling a library per tile, the objection lands against the draft as
written, and I restructured rather than defended it.
