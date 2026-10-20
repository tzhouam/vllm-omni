# Edge engine comparison: Qwen3-TTS on vLLM-Omni (edge branch) vs native edge runtimes

Measured on 2026-09-09 on the same host as the rest of this study (Xeon 8480C, one NVIDIA L20X through `gpu run --gpus 1`), same 12 English prompts (`vllm-omni/benchmarks/tts/prompts_12.txt`), same model family (Qwen3-TTS-12Hz, 0.6B and 1.7B), same reference speaker wav, one utterance at a time. Raw results, driver scripts and logs live in [experiments/edge_engine_compare/](experiments/edge_engine_compare/); the merged table is [results_summary.md](experiments/edge_engine_compare/results_summary.md) (regenerate with `python summarize_compare.py --out results_summary.md`).

## 1. What was compared, and what could not be

| Engine (checkout) | Qwen3-TTS path | Status here |
|---|---|---|
| vLLM-Omni `edge/omni-edge` (this branch) | two-stage talker → code2wav pipeline, streaming chunks, edge profile; CPU platform from WP3 | measured on GPU and CPU |
| llama.cpp `e71b80510` | `llama-tts` + `libmtmd` (talker GGUF + mmproj holding the speaker encoder and the codec decoder; `conversion/qwen3tts.py`) | built out of tree (CPU and CUDA), models converted locally from the cached HF checkpoints (f16, Q8_0, Q4_0), measured on GPU and CPU |
| MNN `bef71b97` | `transformers/llm/engine/demo/qwen3_tts_demo` + `llmexport.py` | built (`MNNConvert`, `qwen3_tts_demo`), but the exporter imports the external `qwen_tts` package (the QwenLM/Qwen3-TTS GitHub repo), which is not on this host; cloning it is a download and was not done without permission. **Not measured.** |
| ExecuTorch `cfc6ecd516` | no Qwen3-TTS runner (the streaming TTS runner is Voxtral); the branch's decoder export targets it | needs `pip install executorch` (permission gate). **Not measured.** |
| ncnn `3b7bdba7` | no TTS model | not applicable |

Model/task parity: llama.cpp and MNN only run the **Base** checkpoints with a reference speaker (x-vector cloning); the branch's headline numbers are for **CustomVoice** (preset speakers). So this comparison runs vLLM-Omni on the *same* Base checkpoints with the *same* reference wav in `x_vector_only_mode`, and quotes the CustomVoice numbers separately where the reference-audio cost matters. The reference wav is a 4.5 s utterance synthesized earlier by the two-stage server; it drives timing only, not quality.

## 2. Metric definitions

- **TTFA**: vLLM-Omni: measured time from request submission to the first streamed PCM chunk (p50 over 12 prompts × repeats). llama.cpp: the tool is not streaming (all frames, then one vocoder pass, then the WAV), so the column is an **estimate**: prompt eval + 2 talker frames + 2 frames' share of the vocoder time (the edge profile's first chunk is 2 frames). llama.cpp's speaker encoding happens at load time and is excluded; vLLM-Omni's per-request x-vector extraction is included in its Base TTFA.
- **Talker ms/frame**: llama.cpp: generation time / frames from its own timing line. vLLM-Omni: the stage-0 engine-core step from the WP8 counters (one step = one frame, includes the 15-codebook predictor).
- **Vocoder ms/frame**: llama.cpp's vocoder time / frames. vLLM-Omni's code2wav runs concurrently in its own stage, so its cost does not add to the frame period; it is not tabulated.
- **RTF**: total processing wall time / seconds of audio (llama.cpp: its "total" line; vLLM-Omni: request wall time / audio).
- **Startup**: vLLM-Omni: engine init to ready (`init_s`, both stages). llama.cpp: process start + weight load + speaker encoding (`wall − total`, per invocation; the tool reloads everything for every utterance).
- **Memory**: peak RSS of the process tree (both), peak per-process GPU memory (vLLM-Omni, `nvidia-smi` samples; llama.cpp GPU memory was not sampled).

## 2b. Mobile (real devices via Qualcomm AI Hub, 2026-09-10)

Measured on Samsung Galaxy S24 (Snapdragon 8 Gen 3) and Galaxy S25 (Snapdragon 8 Elite): QNN context binaries on the Hexagon NPU, TFLite on the Adreno GPU and the phone CPU, AI Hub on-device medians. Details, job ids and fidelity in [mobile_htp_simulation.md §5](mobile_htp_simulation.md); export code in `vllm_omni/edge/qnn_export.py` and `decoder_export.py`.

| component (0.6B, real weights), per 80 ms frame | Hexagon NPU | Adreno GPU | phone CPU |
|---|---|---|---|
| talker decode step (v2 export), fp16 | **15.8 ms** | 29.9 ms | 29.8 ms |
| code predictor, 15 cached steps, fp16 | **33.9 ms** | 65.0 ms | 72.3 ms |
| Code2Wav vocoder, fp16 | 182 ms | **21.7 ms** | 109 ms |
| Code2Wav vocoder, fp32 (numerically exact) | – | **45.8 ms** | 109 ms |

Best placement: talker + predictor on the NPU, vocoder on the GPU. The two are separate stages exchanging codec tokens, so they pipeline: **49.7 ms per 80 ms frame, RTF 0.62** (96 ms, RTF 1.19 if run serially with the exact vocoder). Fidelity: NPU fp16 39.2 dB (vocoder) / 48.3 dB (predictor); GPU fp32 108.4 dB; GPU fp16 27.5 dB. Every automatic int8 build is 5–17× faster and numerically broken (vocoder −4.3 dB, predictor 3.1 dB), so int8 is not counted.

Runtime comparison on the same graphs and phone (per frame): the talker step costs 15.8 ms on a QNN context binary, 16.3 ms on TensorFlow Lite and 16.6 ms on ONNX Runtime — all NPU paths converge because they all end in QNN, so the export layout matters and the runtime does not. They separate elsewhere: the vocoder is 21.7 ms on TFLite/GPU, 122.5 ms on ONNX Runtime/GPU, 109–124 ms on the phone CPU, and cannot run on the NPU under TFLite or ONNX Runtime at all (fails after compiling; out of device memory).

Reading: with the heterogeneous split the branch's 0.6B pipeline is **real time on a 2024–2025 flagship phone, in faithful precision, without quantization**. llama.cpp and MNN have no Hexagon NPU path for Qwen3-TTS at all; llama.cpp's CPU-only pipeline extrapolates above RTF 2 on phone cores. The mistake to avoid is sending everything to the NPU: that is 3.7× slower than the split.

## 2c. Mobile inference engines on the same graphs (x86 CPU, 2026-09-10)

MNN, ncnn and ExecuTorch were the three engines this study could not reach earlier. All three are now installed
(`.venvs/edge-engines`) and fed the *same* exported graphs as the QNN/TFLite/ONNX-Runtime runs: the v2 talker decode
step (256-token cache), the code-predictor step (16-token cache) and the Code2Wav chunk (25 frames of audio from a
97-frame window). Median of 15 runs after 3 warm-ups, pinned to 8 or 16 physical cores of the Xeon 8480C, fp32.
Accuracy is signal-to-noise against the PyTorch output for the same input; the harness is
`experiments/edge_engine_compare/qnn/engines/bench_engines.py`.

| graph | PyTorch 2.14 | ONNX Runtime 1.29 | MNN 3.x | ExecuTorch (XNNPACK) | ncnn |
|---|---|---|---|---|---|
| code predictor step, 8 / 16 threads | 4.47 / 3.30 ms | 4.27 / **2.51** ms | **3.69** / 4.13 ms | 6.89 / 5.01 ms | segfault |
| talker decode step, 8 / 16 threads | 35.96 / 20.32 ms | **26.40 / 17.04** ms | 37.46 / 31.39 ms | 73.04 / 61.87 ms | segfault |
| Code2Wav chunk (2 s audio), 8 / 16 threads | **957 / 590** ms | 1562 / 1248 ms | 1621 / 1059 ms | 1059 / 633 ms | 3538 / 2331 ms, silent |
| accuracy vs PyTorch (16 threads) | reference | 86–114 dB | 85–109 dB | 84–107 dB | NaN inside the graph |

Reading:

- **ncnn cannot run this model.** pnnx converts all three graphs at its default optimisation level, but the two
  transformer steps segfault inside `Extractor::extract` and the vocoder fills with NaN partway through, which the
  final `clamp(-1, 1)` turns into constant silence — the timing column above is therefore meaningless for it. At
  `optlevel=0` the graph will not even load (`layer F.linear not exists or registered`). This is a converter-quality
  problem in the ncnn toolchain for these graphs, not a missing runtime feature.
- **MNN and ExecuTorch both work and are numerically faithful** (85–109 dB, i.e. fp32 arithmetic differences only).
  MNN is the fastest engine on the small predictor step at 8 threads; ExecuTorch is consistently the slowest on the
  transformer steps (2–3× ONNX Runtime) but mid-pack on the vocoder.
- **ONNX Runtime is the best CPU engine for the transformer steps** (1.4–1.8× MNN and ExecuTorch); **PyTorch is the
  best for the vocoder**, where its fused convolution/activation kernels beat every exported runtime.
- Caveat that matters: this is x86. MNN, ncnn and ExecuTorch are tuned primarily for ARM (NEON, dotprod, i8mm), so
  this ranks the engines *on this host*, not on a phone. The phone-side ranking is in §2b, where the same graphs run
  on the actual accelerators.
- MNN's Interpreter (C-style) API silently mis-feeds tensors of these ranks and produced −4 dB output; the numbers
  above come from its Module/expr API. Anyone benchmarking MNN on multi-input graphs should check parity first.

## 3. GPU (one L20X)

| engine | model | TTFA | talker ms/frame | vocoder ms/frame | RTF | startup | peak RSS |
|---|---|---|---|---|---|---|---|
| llama.cpp CUDA, all layers | 0.6B Base f16 | 109 ms (est.) | 17.7 | 3.6 | 0.31 | 4.5 s per invocation | not sampled |
| llama.cpp CUDA, all layers | 0.6B Base Q4_0 | 125 ms (est.) | 16.8 | 2.4 | 0.28 | 3.4 s | not sampled |
| llama.cpp CUDA, all layers | 1.7B Base f16 | 146 ms (est.) | 17.9 | 2.1 | 0.28 | 6.0 s | not sampled |
| vLLM-Omni edge profile | 0.6B Base, x-vector | 561 ms p50 (incl. per-request x-vector: chunks then 27–43 ms apart) | 6.0 | – | 0.20 | 148 s | 9.0 GiB + 36.6 GB GPU |
| vLLM-Omni edge profile | 1.7B Base, x-vector | 374 ms p50 (same) | 6.4 | – | 0.20 | 149 s | 8.8 GiB + 37.0 GB GPU |
| vLLM-Omni edge profile | 0.6B CustomVoice (no ref audio) | 41–54 ms p50 | 6.0 | – | 0.10 | 87 s | 8.7 GiB + 37 GB GPU |
| vLLM-Omni edge profile | 1.7B CustomVoice (no ref audio) | 44–46 ms p50 | 6.4 | – | 0.10 | 88 s | 7.8 GiB + 37 GB GPU |

Reading:
- **Steady-state generation** is where vLLM-Omni wins on the GPU: 6 ms per codec frame (CUDA graphs, talker-MTP graph, compiled code predictor) against 17–18 ms in llama.cpp, i.e. 3× faster per frame, and its code2wav runs in parallel in a second stage, so RTF is 0.10 (CustomVoice) / 0.20 (Base) against 0.28–0.31. Both are far below real time; the difference only matters for concurrency headroom or for smaller GPUs.
- **First audio** favours llama.cpp once reference-audio handling is in the picture: vLLM-Omni spends ≈0.3–0.5 s per request on x-vector extraction (the chunk cadence after the first chunk is normal), while llama.cpp encodes the speaker once at load. Without reference audio (CustomVoice) vLLM-Omni's measured TTFA of 41–54 ms is below llama.cpp's estimated 109–146 ms. Caching the x-vector per speaker on the server would bring the Base path to the same range; that is the obvious next fix on the branch.
- **Startup and footprint** favour llama.cpp by more than an order of magnitude: 3–6 s to load and be ready (per invocation, speaker encoding included) versus 87–150 s (and 355 s with cold compile caches) for vLLM-Omni, whose two stages each carry a vLLM engine process with CUDA-graph capture and a `torch.compile` warm-up, and 36–37 GB of reserved GPU memory (memory fractions, not need) versus a few GB of weights.

## 4. CPU only (NUMA node 1 of the Xeon 8480C, pinned)

Full grid in [results_summary.md](experiments/edge_engine_compare/results_summary.md); the WP3 CustomVoice sweep is in [experiments.md §8.4](experiments.md#84-cpu-platform-wp3-and-cpu-thread-sweep). Both engines pinned to the same cores of NUMA node 1 (stage 0 gets 3/4 of the threads in vLLM-Omni).

| engine | model | threads | TTFA | playback-start | talker ms/frame | vocoder ms/frame | RTF | startup | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 0.6B Base f16 | 8 / 16 / 32 | 443 / 413 / 407 ms (est.) | – | 94 / 97 / 93 | 55 | 1.9 / 2.0 / 2.0 | 3–4 s | not sampled (Q8_0 run: 6.8 GiB) |
| llama.cpp | 0.6B Base Q8_0 | 16 | 368 ms (est.) | – | 94 | 56 | 1.9 | 2.9 s | 6.8 GiB |
| llama.cpp | 0.6B Base Q4_0 | 8 / 16 | 342 / 332 ms (est.) | – | 85 / 87 | 54 | 1.9 / 1.9 | 2.2 s | 6.3 GiB |
| llama.cpp | 1.7B Base f16 | 16 | 541 ms (est.) | – | 114 | 56 | 2.2 | 5.1 s | 12.2 GiB |
| llama.cpp | 1.7B Base Q4_0 | 16 | 407 ms (est.) | – | 91 | 53 | 2.0 | 3.0 s | 8.0 GiB |
| vLLM-Omni CPU platform | 0.6B Base, x-vector | 8 / 16 / 32 | 855 / 581 / 707 ms | 8.4 / 5.1 / 4.6 s | ≈250 / 147 / 136 (from the CustomVoice sweep) | concurrent stage | 2.6 / 1.8 / 1.7 | 102–104 s | 13.2–13.4 GiB |
| **vLLM-Omni CPU platform after the 2026-09-10 CPU changes** ([mobile_htp_simulation.md §3](mobile_htp_simulation.md); faithful fp32 predictor path, the branch default) | 0.6B Base, x-vector | 16 / 32 | 686 / 486 ms (x-vector included) | 3.0 / 2.2 s | 109 / 91 mean (talker 25 / 21 + predictor 69 / 56 + sampler <1) | concurrent stage | **1.27 / 0.99** | ≈100 s | ≈13 GiB |
| same, CustomVoice (no reference audio) | 0.6B | 8 / 16 / 32 | 388 / 224 / 195 ms | 5.7 / 2.6 / 1.9 s | 148 / 93 / 77 | concurrent stage | 1.93 / **1.15 / 0.89** | ≈100 s | ≈13 GiB |
| same with the int8-dynamic predictor (measured, then **rejected**: 22 % top-1 agreement with the fp32 predictor, see §3 of the HTP document) | 0.6B Base x-vector / CustomVoice | 16 / 32 | 696 / 175 ms | 2.6 / 1.7 s | 80 / 59 | concurrent stage | 1.12 / 0.80 | ≈100 s | ≈13 GiB |
| vLLM-Omni CPU platform | 0.6B CustomVoice (WP3 sweep) | 8 / 16 / 32 | 426 / 292 / 252 ms | 9.4 / 5.3 / 5.0 s | 250 / 147 / 136 | concurrent stage | 2.8 / 1.8 / 1.7 | 110–156 s | 13.3 GiB |

Reading:
- Update 2026-09-10: after the KV-cached code predictor with a compiled shape-static step, the compiled post-step (lm head + top-k sampling + embedding) and the model-side sampler ([mobile_htp_simulation.md](mobile_htp_simulation.md) §3), the branch generates a frame in 77–109 ms and runs at RTF 0.89–1.27 on 16–32 threads (real time at 32 threads), i.e. 1.6–2.2× faster end-to-end than llama.cpp, whose RTF stays 1.9–2.0 because its talker frame (85–97 ms) is followed by a serial ≈55 ms vocoder pass per frame. At 8 threads the two are level (RTF 1.93 vs 1.9), where llama.cpp's Q4_0 talker frame (85 ms) is still shorter than the branch's 148 ms; an int8 predictor would close that gap but the only int8 path available in this torch build (dynamic per-tensor int8) was rejected for fidelity. The paragraphs below describe the state before that change.
- Neither engine reaches real time on this CPU class with the 0.6B model. llama.cpp's talker is faster per frame (85–97 ms vs 136–250 ms; vLLM-Omni's step includes the 15-codebook predictor eagerly on CPU) but llama.cpp then pays ≈55 ms per frame in its non-streamed BigVGAN vocoder, so the end-to-end RTF is the same ≈1.9–2.0 for llama.cpp and 1.7–2.6 for vLLM-Omni (whose decoder runs concurrently in the second stage).
- llama.cpp needs half the RAM (6–8 GiB vs 13 GiB) and starts in 2–5 s instead of ≈100 s. Quantization trims its talker (Q4_0: −10 %) and its footprint but not the f16 vocoder; vLLM-Omni's CPU platform has no quantized talker path on this branch (bf16/AMX), the first thing to add if the CPU tier matters.
- Playback-start latency is the honest streaming metric on CPU: vLLM-Omni streams but underruns (first audio at 0.6–0.9 s, then the player would stall for 4–8 s over the utterance); llama.cpp does not stream at all (the whole utterance is ready after ≈2× its duration). Threads beyond 16 help neither engine.

## 5. Bottom line for the edge plan

- On a Jetson/discrete-GPU class device, the branch's vLLM-Omni is the faster steady-state engine (3× per frame) and the better streaming server (measured 41–54 ms first audio, ramped chunks, concurrent decoder stage), but it costs 100 s+ of startup, a ~9 GiB host process and a Python/vLLM stack; llama.cpp starts in seconds, needs one process and a few GB, and is within 3× on generation speed. For an always-on edge server the branch is the right tier; for a device that must start on demand, or has < 8 GB, llama.cpp is.
- On CPU-only devices neither runs Qwen3-TTS in real time; the study's conclusion stands (real-time CPU TTS needs a smaller/quantized talker and a streamed, cheaper vocoder), and llama.cpp's quantized GGUF path is the better starting point there.
- MNN and ExecuTorch numbers are still missing for the reasons above; the exact unblock is `git clone https://github.com/QwenLM/Qwen3-TTS` (then re-run `llmexport.py` with `PYTHONPATH=<clone>`) and `pip install executorch` respectively.
