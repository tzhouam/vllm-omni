# Roadmap

Ordered by whether the verification method is *correct*, not by expected size.
That ordering is deliberate: this project retracted five conclusions, and in
every case the fix was to the measurement rather than the code.

Each item states the measurement it starts from, what it should gain, and how
that gain would be verified. An item without a verification method is not on
this list.

## Done

| | item | result |
|---|---|---|
| H1 | Task oracles judged by device state, with variants | alarm oracle needed a *saved* alarm, not a typed `7`; a Wi-Fi round trip now fails the already-satisfied task |
| H2 | Harness semantics | arm order really rotates; `best` respects metric direction; provenance stamped; failures recorded |
| H3 | The three missing measurements | `ready_s`, peak `RssAnon`, memory at a filled context — all previously unmeasurable |
| H4 | Contention guard | aggregate stolen capacity + straggler rule, on the target cores rather than host load average |
| H5 | Regression gates | bands from observed spread; throughput explicitly ungateable at ~30% swing |
| H6 | Fidelity metric that ranks | top-1 agreement and KL, replacing a saturating 12-prompt greedy check |
| H8 | Retracted numbers purged from source | two modules were still shipping withdrawn figures as documentation |
| I0 | Blocked weight repack | **489 MB** of load-time peak, no cost |
| I1 | One packed weight layout | **615 MB** resident, at ~1.63× prefill |
| I2 | Measure `cpu_gemm_wna16` | 2.2× slower than tinygemm; routing was right, now for a measured reason |
| I7 | Memory ledger, by attribution | found a **618 MB** RoPE table and an **843 MB** KV cache, both inside an "unexplained 1.6 GB" |

## Next

**I6 — import cost.** `import vllm_omni` is ~30 s against `import vllm`'s 5.2 s,
with transformers' auto image-processing (4.6 s) and the `openai` client (1.3 s)
identifiable inside it, for a text model that needs neither. Every stage
subprocess pays it again.
*Expect:* an upper bound of ~6 s per process; deferred imports often move cost
rather than remove it. *Verify:* `ready_s`, which now exists — the old timer
started after imports, so the benchmark meant to prove this could not see it.

**Drop the row-major embedding copy.** `embed_tokens.weight_packed` is 128 MB,
kept solely because the tinygemm layout was believed un-addressable for a
single-row lookup. That premise is now known false. *Verify:* `RssAnon` plus an
embedding-output SNR assertion.

**I10 first step — why is the GEMM at 60%?** The decode GEMM sustains 132 GB/s
against a measured 222 GB/s ceiling. Part is arithmetic (tinygemm dequantizes
per activation row) and part may be that each of 113 calls streams only a few MB,
too little for the prefetcher to reach steady state. Attribute before optimising.
*Expect:* unknown — a diagnosis, whose output is whether the 2.76 ms implied by
the gap is reachable at all.

**I3 — asymmetric W4A8.** The symmetric grid is our exporter's choice, not the
kernel's limit; `models/spark2_5-gptq-asym-g128` already exists. The cheapest
untried fidelity experiment, and it tests the *stated* cause of W4A8's error.
*Verify:* `fidelity_kl.py` and the scored agent suite.

**I5 — attention wrapper consolidation.** Four ops per layer, 1.31 ms/step, of
which 0.83 ms is wrapper self time; at batch 1 the cache-update loop runs two
iterations while the wrapper rebuilds context and allocates a dummy tensor for
compile ordering. *Expect:* 0.3–0.5 ms, revised down from an earlier 0.6–0.9 ms
that assumed a workload never profiled. *Verify:* per-op profile **and**
unprofiled step latency — one outer call that still launches two threaded
kernels removes neither thread launch.

**I4 — a better quantizer.** Ours reproduces 2/12 greedy prompts against its own
bf16; Q4_K_M manages 4/12 — but at 4.5 bits against our 4.25, so close the
budget gap first (g64, or fp16 scale+zero, both config changes) before
implementing error-compensated rounding. Timebox it; a null result is worth
recording.

## Considered and rejected

**A CPU megakernel.** On GPU this fuses a forward pass into one launch to remove
launch overhead and the cross-kernel prefetch bubble. On CPU the analogues are
OpenMP fork/join and ATen dispatch — and llama.cpp already is one:
`ggml_threadpool` starts threads once and they walk the whole graph,
synchronising at `ggml_barrier` through an atomic spin, where vLLM enters a
separate `omp parallel` region per kernel.

Measured before deciding: a parallel region costs ~1.2 µs at 24 threads, and
spin-waiting does not help (`OMP_WAIT_POLICY=ACTIVE` moved it 1.23 → 1.18 µs), so
it is barrier cost rather than thread wake-up. At a few hundred regions per
decode step that is 0.3–0.4 ms, 2–3% — the same money as I5, far more cheaply.
The larger prize is the streaming bubble, but capturing it means replacing the
execution model inside vLLM's CPU backend, which reimplements ggml. Not next.

**More kernel fusion.** Four attempts, four losses: vLLM's fused CPU RMSNorm is
3.6–4.5× faster in isolation and slower in situ (inductor already fuses the
native form with the residual add); fused `gelu_and_mul` 2.7× slower at Spark's
shape; `cpp.min_chunk_size` −6% decode; an inline gate reduction −44%.

**MiniCPM-o CPU overlay.** The 8 B thinker at bf16 is ~19 GB, so it buys no edge
deployability. The argument for it — that three-stage parallel init should beat
the two-stage 2.1× — is also not how overlap works: the bound is ΣT/max(T).

**Speculative decoding for tg128.** It would inflate a fixed-prompt `ignore_eos`
benchmark without representing anything. Measure it on the agent suite or not at
all.

**Thread count and warm-up batch size as memory levers.** 64 MB and 2 MB.

## Harness work still open

- **A real device in the agent loop.** The toolkit has 38 tests and has never
  touched a phone — this host has no `adb`, no emulator, no `/dev/kvm`. The
  real-device path and per-task resets exist; what is untested is whether a real
  Settings app matches the fixtures.
- **Success rates, not single runs.** The build comparison that produced
  "navigation survives quantization, completion does not" is one run per cell.
  Variants now exist so repeats are trials; they need running.
