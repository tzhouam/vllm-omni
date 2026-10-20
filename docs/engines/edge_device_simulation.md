# Simulating edge devices on this host, and why the CPU path is slow

Companion to [edge_engine_comparison.md](edge_engine_comparison.md). Measured 2026-09-09; raw data under [experiments/edge_engine_compare/results/gpu_emu](experiments/edge_engine_compare/results/gpu_emu/) and [results/cpu_attrib](experiments/edge_engine_compare/results/cpu_attrib/), driver scripts `gpu_emu_chain2.sh` (MPS-capped runs) and the branch's `hw_emulation.py`.

## 1. What this GPU is, and what can be emulated without root

`nvidia-smi` reports the eight cards as "NVIDIA L20X, Ada Lovelace, 143,771 MiB", but the measured dense bf16 matmul throughput is **804 TFLOPS** (8192³, 20 iterations) with 132 SMs at 1980 MHz and 140 GiB, which is H200-class silicon, not an Ada L20 (≈120 TFLOPS). The comparison and the branch's results should therefore be read as **H200-class** numbers; the label in earlier documents is the driver's.

| Edge property | Emulable here? | How | Not covered |
|---|---|---|---|
| Fewer SMs (Orin AGX 16 SMs, Orin NX 8, Thor 20 vs 132) | **yes** | user-level MPS daemon + `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` (verified: 12 % cap → 100.6 TFLOPS, ratio 0.125). Two traps: the Unix-socket path of `CUDA_MPS_PIPE_DIRECTORY` must be < 108 bytes, and clients inside the MPS namespace must use `CUDA_VISIBLE_DEVICES=0` even when `gpu run` assigned another index | lower SM clocks (Orin 1.3 GHz vs 1.98 GHz) |
| Small VRAM / unified memory budgets | yes | branch `deploy_profile="auto"` with `VLLM_OMNI_HW_PROFILE` overrides: absolute KV budgets, eager mode, small batches (`hw_emulation.py` jetson_orin_8g / 32g / cuda_discrete_8g) | unified-memory bandwidth sharing with the CPU |
| Weak host CPU (Orin: 8–12 Cortex-A78AE at 2.2 GHz) | partly | `taskset` to 4–6 Xeon cores for the whole engine; x86 cores are still ≈2× faster per core | ARM ISA, big.LITTLE |
| Memory bandwidth (HBM 4.8 TB/s vs LPDDR5 102–273 GB/s) | **no** | analytic projection only (§4) | – |
| Clock / power caps, MIG partitions | no | `nvidia-smi -lgc/-pl` and MIG need root; MIG is disabled on these cards | – |
| Thermal throttling, CPU/GPU memory contention | no | – | – |

## 2. GPU: sensitivity to SM count (MPS caps, H200, 0.6B CustomVoice, edge profile)

Same 12 prompts, one utterance at a time, `deploy_profile="edge"`, 0.6B CustomVoice; MPS `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` = share of the 132 SMs available to every process of the deployment (both stages). llama.cpp rows: `llama-tts` CUDA build, all layers offloaded, same cap. The "Jetson-like" row combines a 12 % SM cap (16 SMs ≈ Orin AGX), 6 host cores for the whole engine, and the branch's `jetson_orin_8g` hardware profile (eager mode, absolute 8 GB budget).

| SM share (≈ SMs) | vLLM-Omni talker step ms/frame | vLLM-Omni TTFA p50 / p90 | vLLM-Omni RTF | llama.cpp talker ms/frame | llama.cpp vocoder ms/frame | llama.cpp RTF |
|---|---|---|---|---|---|---|
| 100 % (132) | 6.0 | 41–46 / 49 ms | 0.10 | 17.4 | 3.8 | 0.28 |
| 25 % (33) | 6.5–7.0 | 43 / 49–53 ms | 0.08–0.09 | 23.1 | 2.1 | 0.33 |
| 12 % (16, Orin AGX class) | 9.1 | 52 / 58 ms | 0.12 | 33.7 | 4.7 | 0.50 |
| 6 % (8, Orin NX class) | 9.3 | 58 / 66 ms | 0.13 | 48.0 | 4.1 | 0.68 |
| Jetson-like combined: 12 % + 6 host cores + 8 GB profile (eager) | ≈13 (from RTF) | 160 / – ms | 0.50 | – | – | – |
| (reference) 8 GB profile alone, uncapped, from §8.5 | – | 181 ms | 0.57 | – | – | – |

Reading: with CUDA graphs and one utterance in flight, the talker step is latency-bound: cutting the SMs to 6 % of an H200 costs vLLM-Omni only +55 % per frame (6.0 → 9.3 ms) and +15 ms of TTFA, and RTF stays ≈0.13. llama.cpp's kernels are launch-bound per layer and lose 2.8× at the same cap (17 → 48 ms per frame). The combined Jetson-like case shows where the real cost sits: forcing eager mode (the 8 GB profile disables CUDA graphs so the small budget is not consumed by graph pools) raises the frame time to ≈13 ms and TTFA to 160 ms regardless of the SM cap, i.e. **on a small-memory Jetson the branch should keep graph capture for the talker/predictor and spend the memory elsewhere**, which is the next tuning item for `deploy/edge/hardware/jetson.yaml`. What SM caps cannot show is the memory-bandwidth term, projected in §4.

## 3. Where a GPU frame goes (synchronized attribution, uncapped)

`VLLM_OMNI_STEP_STATS_SYNC=1` synchronizes the device around the model-level timers (adds ≈3 ms per step of sync overhead, so `core.step` reads 9.6 ms here instead of the 6.0 ms measured without sync):

| per talker step (= one 80 ms frame) | ms |
|---|---|
| model forward (28-layer talker, CUDA graph) incl. sampling hook | 1.8 |
| sampler | 0.6 |
| talker-MTP graph (15 code-predictor sub-steps + embeddings, one CUDA graph) | inside the forward's 1.8 ms (graph replay; the separate `model.mtp_graph_ms` timer measures 0.04–0.43 ms of launch time without sync) |
| scheduler (`schedule` + `update_from_output`) | 0.2 |
| remainder: input preparation, D2H of codes/hidden, Python bookkeeping | ≈3–4 |

The GPU compute for a frame is a couple of milliseconds; more than half of the 6 ms step is host-side runner work, which is what a slow ARM host CPU would stretch first.

## 4. Analytic projection to Jetson-class devices

The talker step is memory-bound at batch 1: every frame streams the talker weights once (0.6B bf16: 1.2 GB) and the code-predictor weights fifteen times (5 layers × 1024 hidden ≈ 160 MB each pass, 2.4 GB per frame). Using the measured H200 split and datasheet bandwidths (weights from the HF config; bandwidths: H200 4.8 TB/s, Orin AGX 204.8 GB/s, Orin NX 102.4 GB/s, Thor 273 GB/s):

| device | weight streaming per frame (talker + 15 predictor passes) | host-side runner work (≈4 ms on Xeon, scaled by single-core speed) | projected ms/frame | projected RTF (80 ms frame) |
|---|---|---|---|---|
| H200 (measured) | 0.75 ms | ≈4 ms | 6.0 (measured) | 0.10 (measured) |
| Jetson AGX Orin 64 GB | 17.6 ms | ≈8 ms | ≈26 | ≈0.33 |
| Jetson Orin NX 16 GB | 35 ms | ≈8 ms | ≈43 | ≈0.55 |
| Jetson Thor | 13 ms | ≈6 ms | ≈19 | ≈0.24 |

These are floors from bandwidth plus measured host overhead, not measurements: kernel launch latency at small SM counts, unified-memory contention with the code2wav stage on the same memory bus, and thermal limits push them up. The MPS 12 % run (§2) is the compute-side check of the Orin AGX case (16 of 132 SMs); a 4-bit talker (llama.cpp Q4_0 runs at 3.1 ms/token on 16 Xeon threads for the same backbone) would cut the streaming term by ≈3× on all rows. **Conclusion:** the branch's talker stays real-time on Orin AGX / Thor with 2–4× headroom by this model, is marginal on Orin NX, and its 15-pass code predictor is the term to attack (see §5).

## 5. Why the CPU path is slow

### 5.1 Measured attribution (vLLM-Omni CPU platform, 0.6B CustomVoice, 12+4 threads on one socket, bf16 with AMX)

| per frame (one talker step) | ms | share of the 80 ms frame |
|---|---|---|
| **code predictor loop** (15 sub-steps × 6.4 ms, 5-layer 1024-hidden model, batch 1) | **97.1** | 121 % |
| talker forward (28 layers, 0.45 B params) | 25.4 | 32 % |
| sampler (top-k/top-p on a 3072-way vocabulary, penalties, CPU) | 9.9 | 12 % |
| scheduler + bookkeeping | ≈2 | 2 % |
| **engine-core step** | **133.9** | **167 %** |

(`core.step` 134 ms here vs 147 ms in the earlier sweep: the attribution run had the host to itself.) The code2wav stage runs concurrently in its own process and does not add to this cadence.

### 5.2 The same split in llama.cpp (native C++, same host, 16 threads)

`llama-bench` on the converted talker GGUF gives the backbone alone at **126.7 tok/s f16 (7.9 ms/token)** and **325.8 tok/s Q4_0 (3.1 ms/token)**, yet `llama-tts` needs **85–97 ms per frame**: ≈80 ms of every frame is the fifteen code-predictor passes (≈5.3 ms each) plus sampling. llama.cpp's predictor pass is only ≈20 % cheaper than vLLM-Omni's 6.4 ms even though it is hand-written C++ with no Python — so the cost is not "Python overhead"; it is the structure of the work.

### 5.3 Root causes, in order

1. **Sixteen sequential forward passes per 80 ms frame.** One talker pass plus fifteen residual-codebook passes, each depending on the previous sample. Real time needs the whole chain under 80 ms, i.e. **< 5 ms per pass including sampling**. Both engines sit at 5.3–6.4 ms per predictor pass on 16 Xeon threads.
2. **Each pass is latency-bound, not bandwidth-bound.** A predictor pass reads ≈160 MB of bf16 weights, which is ≈1.6 ms at the ≈100 GB/s one socket-half delivers; the measured 6.4 ms is 4× that. The rest is per-operator cost: ≈100 small kernels per pass (5 layers × QKV/attention/RoPE/norms/MLP/gate) each paying OpenMP fork-join across 12 threads, cache-cold tiny GEMVs, and (in PyTorch) operator dispatch. The talker's 28-layer pass is at 2× its bandwidth floor (25 ms vs 12 ms) for the same reason.
3. **Sampling costs 10 ms on CPU.** vLLM's sampler applies penalties, top-k and top-p over the padded vocabulary per step with several full-vocabulary passes; on GPU this is 0.6 ms. The fifteen residual codebooks are also sampled (Gumbel top-k) inside the predictor loop.
4. **Threads stop helping at 16.** 16 → 32 threads changed the frame time by 8 % (147 → 136 ms) because the work is latency-bound (item 2) and OpenMP synchronization grows with thread count; the two stages also compete for the same memory channels.
5. **bf16/AMX is the only CPU precision on the branch.** There is no int8/int4 talker path in vLLM's CPU backend for this model; llama.cpp's Q4_0 backbone shows the bandwidth term can shrink 2.5×, but that only helps the part that is bandwidth-bound (talker pass, ≈25 → ≈12 ms), not the predictor's overhead floor.
6. **Vocoder**: not on the vLLM-Omni critical path (concurrent stage), but in llama.cpp the BigVGAN decoder costs a further 55 ms per frame serially, which is why its end-to-end RTF matches vLLM-Omni's despite the faster talker.

### 5.4 What would make CPU real-time, with expected effect

| change | where | expected per-frame effect (0.6B, 16 threads) | evidence |
|---|---|---|---|
| Fuse each predictor pass into one kernel / one OpenMP region (C++ or a compiled graph), drop per-op dispatch | predictor | 6.4 → ≈2–3 ms per pass, i.e. −50…−65 ms per frame | bandwidth floor 1.6 ms; llama.cpp's eager C++ is already at 5.3 ms, so fusion (not language) is the lever |
| Greedy or cheap top-k for the 15 residual codebooks; skip penalties for codec ids | sampler + predictor | −8 ms (sampler) and part of each pass | GPU sampler is 0.6 ms for the same work |
| Int8/int4 talker weights (AMX int8 or a GGUF-style kernel) | talker pass | 25 → ≈10–12 ms | llama-bench Q4_0 3.1 ms vs f16 7.9 ms per token |
| `torch.compile` (inductor) on the CPU runner | whole step | −6 % measured (`cpu16_inductor` in §8.4) | measured |
| Architectural: fewer residual codebooks per pass (predict 2–4 codebooks per pass) or a parallel/non-autoregressive predictor | model | 15 → 4–8 passes per frame | requires retraining or a different checkpoint; out of scope for the branch |

Adding the first three rows gives ≈134 → ≈45–60 ms per frame, i.e. real time with little margin on 16 Xeon threads; on a 4–8-core ARM box (≈2–3× slower per core, see the emulated `x86_cpu_avx2` case at RTF 12) the architectural row is the only one that reaches real time. That matches the study's original conclusion: CPU-only real-time TTS with this model family needs a lighter talker/predictor, not a faster runtime.
