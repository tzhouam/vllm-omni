# MiniCPM-o 4.5 on the edge: measured profile (2026-09-11)

Companion to [mobile_htp_simulation.md](mobile_htp_simulation.md) (Qwen3-TTS) using the same instruments: vLLM-Omni's
StepStats for per-stage attribution, the Xeon 8480C for CPU, one H200 through `canhazgpu` for GPU, and Qualcomm AI Hub
for real Snapdragon devices. Scripts and raw results: `experiments/edge_engine_compare/minicpm/` and
`experiments/edge_engine_compare/results/minicpmo/`.

## 1. What this model is, and why the shape matters

MiniCPM-o 4.5 is a three-stage pipeline in vLLM-Omni (`MINICPMO_4_5_PIPELINE`): an 8 B omni LLM (SigLIP vision tower +
Whisper audio encoder + text backbone) → a TTS head → Code2Wav. Weights are local (19 GB fp16, ≈9.5 B parameters).

The speech half is much lighter than the thinker, and much lighter than Qwen3-TTS's:

| component | parameters | rate |
|---|---|---|
| TTS head (Llama-style decoder, 20 layers × 768, 12 heads, no GQA, no q/k-norm) | 194 M | 1 token per 40 ms (25 Hz) |
| Code2Wav flow matching (CFM DiT, Step-Audio2 / CosyVoice2 lineage) | 156 M | 10 denoising steps per chunk |
| Code2Wav HiFT generator | 21 M | 1 chunk = 25 tokens = 1.0 s of audio |
| thinker (vision + audio + text, 8 B) | ≈9.2 B | prompt-rate |

**`num_vq = 1`**: one codebook, so one decode step per speech token. Qwen3-TTS emits 16 codebooks at 12.5 Hz, which
costs one talker step *plus fifteen* code-predictor sub-steps per 80 ms frame. That single architectural choice is worth
3.5× on a phone (§4). The vocoder is also genuinely streaming already — 3 tokens of lookahead, persistent flow and HiFT
caches — which is the design Qwen3-TTS's 72-frame-context vocoder still lacks.

## 2. GPU (one H200, 6 concurrent requests, text → speech)

Deploy `vllm_omni/deploy/minicpmo_4_5.yaml` (all three stages on one card), 6 prompts from `benchmarks/tts/prompts_12.txt`,
`max_tokens` 192 for the thinker and the stage-1 deploy defaults for the TTS head.

| metric | value |
|---|---|
| engine init (warm caches) | 253 s |
| first text | p50 393 ms, p90 598 ms |
| complete audio per request | p50 5.81 s, p90 7.60 s (offline path yields the whole waveform, so this is completion, not first-chunk latency) |
| batch throughput | 30.2 s of audio in 22.6 s wall → **RTF 0.748** at concurrency 6 |

Per-stage step attribution (mean over the run):

| stage | steps | mean step | mean model forward |
|---|---|---|---|
| 0 · thinker | 46 | 9.66 ms | 2.31 ms |
| 1 · TTS head | 274 | 6.53 ms | 1.72 ms |
| 2 · Code2Wav | 469 | 14.88 ms | – |

The largest single number in the table is not compute: stage 2's `hop.put_to_get` averages 3.1 s, i.e. codec chunks wait
in the shared-memory connector while the vocoder works through its backlog. On one card the vocoder is the queue.

## 3. CPU (Xeon 8480C, pinned)

| stage | 8 threads | 16 threads | real-time factor at 16 threads |
|---|---|---|---|
| TTS head, per 40 ms token | 17.1 ms | 11.6 ms | **0.29** |
| Code2Wav, per 1.0 s of audio | 1033 ms | 833 ms | **0.83** |
| one-time speaker setup (x-vector + prompt mel + caches) | 11.7 s | 9.9 s | cacheable per voice |

Inside the vocoder chunk the split is lopsided, and in the opposite direction from Qwen3-TTS:

| half | per 1.0 s of audio, 16 threads | share |
|---|---|---|
| flow matching (10-step CFM DiT) | 652 ms | **91 %** |
| HiFT generator | 65 ms | 9 % |

Qwen3-TTS's vocoder was elementwise-bound (76 % in Snake activations on sample-rate tensors) and therefore wanted a GPU;
MiniCPM-o's is matmul-bound in a DiT and therefore suits an NPU. The two models need opposite placements.

The step count is a config field (`tts_config.s3_stream_n_timesteps`), and it trades linearly:

| flow steps | per 1.0 s of audio | mel fidelity vs the 10-step default |
|---|---|---|
| 10 (default) | 542 ms | reference |
| 6 | 341 ms | **30.6 dB** |
| 4 | 236 ms | 16.5 dB |
| 2 | 127 ms | 9.6 dB |

Six steps is a 1.6× vocoder speed-up at 30.6 dB on the mel spectrogram — worth a listening test, since it costs nothing
but a config change.

## 4. Real Snapdragon devices (Qualcomm AI Hub)

The TTS head exported through `vllm_omni/edge/qnn_export.py` (Llama-style path: `qk_norm=False`, weight-normalised code
head), 256-token KV cache, QNN context binary, every operator on the Hexagon NPU.

| build | Galaxy S25 (8 Elite) | Galaxy S24 (8 Gen 3) |
|---|---|---|
| fp16, per 40 ms token | **7.08 ms** | **8.18 ms** |
| fp16 fidelity vs torch fp32 | 48.9 dB, top-1 token matches | 49.2 dB, top-1 matches |
| int8 weights (w8a16), per token | 4.88 ms | 6.15 ms |
| int8 fidelity | **2.3 dB, top-1 differs** | **0.1 dB, top-1 differs** |

Per second of speech, the comparison with Qwen3-TTS on the same phone is the headline:

| model, S25 NPU fp16 | per second of audio |
|---|---|
| **MiniCPM-o 4.5** (1 codebook @ 25 Hz, 194 M) | **177 ms** |
| Qwen3-TTS 0.6B (16 codebooks @ 12.5 Hz: talker 15.8 + predictor 33.9 per 80 ms frame) | 621 ms |

**Int8 fails here too.** Per-tensor int8 weights destroy MiniCPM-o's code head exactly as they destroy Qwen3-TTS's code
predictor (3.1 dB there, 2.3 dB here) — two unrelated families, same failure, so this is a property of speech-token heads
rather than of one model or one runtime. The faster int8 rows stay timing-only.

Not yet measured on device: the Code2Wav flow + HiFT (its streaming caches need the same flattening the KV cache needed),
and the 8 B thinker, which at fp16 is 16 GB and does not fit a phone at all — int4 would be mandatory, and §4 is the
reason to expect trouble there.

## 5. Reading

- **For a phone, MiniCPM-o's speech stack is the better architecture** and Qwen3-TTS's is the better size. One codebook at
  25 Hz costs 177 ms of NPU time per second of audio against 621 ms; but the thinker in front of it is 8 B, where
  Qwen3-TTS's whole pipeline is 0.6 B.
- **Placement is model-specific.** Qwen3-TTS wants transformer-on-NPU + vocoder-on-GPU. MiniCPM-o's vocoder is a DiT, so
  the NPU is plausible for both halves; that is the next measurement.
- **The cheapest real win is the flow step count**, not quantization: 10 → 6 steps is 1.6× at 30.6 dB, while every int8
  path measured in this study has been numerically broken.
- Two bugs found while profiling, both in the offline path rather than the model: `examples/offline_inference/minicpmo/end2end.py`
  builds two sampling-parameter sets and fails on this three-stage pipeline with `Expected 3 sampling params, got 2`, and
  its talker parameters cap the codec LM at `max_tokens=1`, which silently yields zero audio. The driver in
  `experiments/edge_engine_compare/minicpm/bench_minicpmo.py` passes three sets and the stage-1 deploy defaults.
