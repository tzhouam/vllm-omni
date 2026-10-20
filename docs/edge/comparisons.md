# Baselines and cross-engine comparison

Where this fork's numbers sit against other engines, and what each comparison
does and does not establish.

## The baseline engine: llama.cpp

llama.cpp is the reference for CPU inference of a quantized model, so it is what
"good" is measured against. Both engines are pinned to the same physical cores
with `taskset` and run **strictly one at a time**; vLLM's prefix caching is
disabled so a replayed prompt is really recomputed, because `llama-bench` has no
such cache and leaving it on would not be a comparison.

Spark-X2.5-1.7B, Xeon Platinum 8480C:

| | decode (tg128) | prefill (pp512) |
|---|---|---|
| llama.cpp Q4_K_M | **90.0 tok/s** | 1121 tok/s |
| this fork, W4A8 AMX | 74.4 tok/s | — |
| this fork, W4A16 tinygemm | 71.6 tok/s | **2190 tok/s** |

**Decode is llama.cpp's.** Prefill is ours by ~2×, which is the shape one expects:
vLLM's prefill goes through oneDNN/AMX bf16 GEMMs at large batch, where
llama.cpp's advantage in per-step overhead does not apply.

Not a matched-precision comparison in the strict sense: Q4_K_M stores **4.5
bits/weight** over groups of 32 with higher-bit types for some tensors, against
our **4.25** over groups of 128. It buys fidelity with those bits (below).

### Why llama.cpp wins decode

Both engines are far from the hardware limit. A node-local streaming read on the
same 24 cores sustains 222 GB/s:

| | GB/s | of ceiling |
|---|---|---|
| llama.cpp whole step | 86 | 39% |
| this fork, whole step | 73 | 33% |

So the gap is not bandwidth, and it is not primarily kernel quality — it is the
execution model. `ggml_threadpool` starts threads once and they walk the entire
graph together, synchronising at `ggml_barrier` through an atomic spin. vLLM
enters a separate `omp parallel` region inside each kernel, and a region costs
~1.2 µs at 24 threads. See [ROADMAP.md](../../ROADMAP.md) item I10 for the
measurements that size this and the reason it is not the next thing to build.

### Fidelity

Against each build's own bf16 output, 12 greedy prompts, exact reproductions:

| | | bits/weight |
|---|---|---|
| llama.cpp Q4_K_M | 4/12 | 4.5 |
| W4A16 | 2/12 | 4.25 |
| W4A8 | 1/12 | 4.25 |

Some of that gap is simply the extra quarter-bit and the finer grouping. How
much is unknown, which is why closing the budget gap (g64, or fp16 scale+zero —
both config changes) comes before implementing a better quantizer.

This metric also saturates: it discards everything after the first divergent
token. `benchmarks/edge_harness/fidelity_kl.py` replaces it with top-1 agreement
and KL over a corpus, deliberately symmetric with `llama-perplexity
--kl-divergence` so llama.cpp stays comparable on our metric rather than being
judged on a different one.

## Other engines considered

Full analyses in [docs/engines/](../engines/). In brief, for this workload:

| engine | verdict |
|---|---|
| **llama.cpp** | the CPU baseline; best decode, mature quantization, no multimodal pipeline |
| **ExecuTorch** | the phone path — exported artifacts, XNNPACK on CPU, QNN for NPU; not a server engine |
| **MNN** | strong mobile CPU/GPU, own model format, another export toolchain |
| **ncnn** | mobile-first, no LLM serving story of the kind needed here |
| **vLLM / vLLM-Omni** | the only one with a multi-stage multimodal pipeline, continuous batching and an OpenAI-compatible server — which is why the edge work is a fork of it rather than a port away from it |

The decision recorded in [docs/analysis/comparison.md](../engines/comparison.md)
is that a phone NPU cannot run vLLM at all: there the deliverable is an exported
graph, and the quantization scheme is decided by the export toolchain. On device
(Snapdragon via QNN) only calibrated **w8a16** was measured correct — 36.9 dB;
int8 activations break it at 5.1 dB, and uncalibrated w4a16 scores −1.6 dB, i.e.
uncorrelated with the reference.

## What the comparisons do not establish

- **ARM.** Every number here is x86 with AMX. `cpu_gemm_wna16` was measured on
  its AMX path because `_get_isa_hint` returns `"amx"` whenever AMX is present;
  the `"vec"` path a genuinely AMX-less machine takes is untested.
- **Task-level quality.** One run per cell on the agent suite showed that
  navigation survives quantization and recognising completion does not. That is
  enough to show a failure reproduces, not to rank builds.
- **Other models.** Qwen3-TTS and MiniCPM-o have edge profiles and init work in
  this tree, but the 4-bit CPU results above are Spark-X2.5 only.
