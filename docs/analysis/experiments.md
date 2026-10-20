# Experiments log

All runs recorded here were executed on 2026-09-08 from this session. Nothing was installed, downloaded, or modified in any checkout. Builds and outputs live in a session scratch directory; logs that matter are copied under [experiments/](experiments/). Numbers below are **measured on this specific host** and are labelled as such; none of them is an edge-device number (the host is an x86 server), so they only calibrate relative costs and validate the execution paths that the design documents rely on.

## 0. GPU scheduling procedure followed

Source: `/home/zhoutaichang/feature/GPU调度工具快速上手.pdf` (read in full, 3 pages) and the copy of its rules in `/data/zhoutaichang/CLAUDE.md`. The relevant rules, all honored:

| Rule (from the guide) | How it was applied |
|---|---|
| Use a personal user, never root | All commands ran as `zhoutaichang` (uid 2005). |
| `gpu status` before starting | Run before submission; all 8 L20X cards were AVAILABLE; a pending booking `84212f67` (another user, GPUs 0-3, 03:00-06:00 daily CI) was noted and avoided by not requesting specific card IDs. |
| Prefer `gpu run --gpus N -- cmd` (auto reserve, sets `CUDA_VISIBLE_DEVICES`, auto release) | Used for all three GPU submissions (runs 1–3). |
| Always `--timeout` on long jobs | `--timeout 30m`. |
| Add `--note` | `--note "tzhouam(claude): qwen3-tts-1.7B streaming TTFA baseline, 12 prompts"`. |
| Release immediately when done | `gpu run` auto-releases; `gpu release` was additionally invoked after the job and `gpu status` re-checked (see §2.4). |
| Never bypass the reservation | No process touched a GPU outside `gpu run`; CPU experiments used no GPU (`GGML_CUDA=OFF`). |

No dependency installation, model download (`HF_HUB_OFFLINE=1` was exported for the GPU job), or source change was made. Experiments judged expensive (multi-GPU, large models, long sweeps) were **deferred** and are listed in §5 with what permission they need.

## 1. Hardware and software environment

| Item | Value (verified with `lscpu`, `free`, `nvidia-smi`, `python -c`) |
|---|---|
| Host | `dedicated-developjob-8gpu2-a029z-…`, Ubuntu 22.04.5, kernel 5.10.134 |
| CPU | Intel Xeon Platinum 8480C, 224 hardware threads, x86_64 (AVX-512 class; `-march=native` build) |
| RAM | 2014 GB total |
| GPU | 8 × NVIDIA L20X, 143,771 MiB each, driver 570.133.20, CUDA 13.0 (`/usr/local/cuda/bin/nvcc`, not on `PATH`) |
| Toolchain | gcc 11.4.0, cmake 4.4.0, ninja 1.13 |
| Python | 3.12.13; `torch 2.13.0+cu129`, `transformers 5.14.1`, `vllm 0.28.0` (wheel), `vllm-omni 0.28.0rc2.dev22+gbe335a86f` (editable, from `/home/zhoutaichang/feature/vllm-omni-main`), `onnxruntime 1.29.0`, `torchao 0.9.0`, `gguf` importable |
| Analyzed commits | see [repository_inventory.md](repository_inventory.md) |

**Important mismatch.** The GPU experiment runs the *installed* vLLM-Omni (`be335a86f`) against the example script from the *analyzed* checkout (`7be014bc`). The two copies of `examples/offline_inference/text_to_speech/qwen3_tts/end2end.py` were confirmed byte-identical (`diff -q`), and the installed checkout's working tree is clean, but the library code differs by an unknown number of commits. Treat the TTS numbers as "vLLM-Omni around early Sept 2026", not as the exact analyzed commit.

## 2. Experiment A — vLLM-Omni Qwen3-TTS streaming baseline (GPU, scheduled)

**Purpose.** Establish a server-class reference for first-audio latency (TTFA), inter-chunk cadence, and end-to-end time of the two-stage talker → code2wav pipeline that [tts_edge_design.md](tts_edge_design.md) proposes to re-host on edge devices. Also verifies the `async_chunk` shared-memory streaming path described in [engines/vllm_omni.md](engines/vllm_omni.md).

**Model.** `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (already present in the HF cache, 4.3 GB; the example hard-codes the 1.7B variants, see `_build_inputs` in [`end2end.py`](../vllm-omni/examples/offline_inference/text_to_speech/qwen3_tts/end2end.py)). Deploy config: the model's default [`vllm_omni/deploy/qwen3_tts.yaml`](../vllm-omni/vllm_omni/deploy/qwen3_tts.yaml) (`async_chunk: true`, `SharedMemoryConnector` with `codec_streaming: true`, `initial_codec_chunk_frames: 1`, `codec_chunk_frames: 25`, stage 0 `gpu_memory_utilization: 0.3`, both stages on logical device `"0"`).

**Command (run 1, exact; runs 2 and 3 differ as listed in the run-history table below, run 3 invoking [tts_stream_bench.py](experiments/vllm_omni_tts/tts_stream_bench.py) with the same flags plus `--init-timeout 1500 --stage-init-timeout 1500`).**
```bash
export HF_HUB_OFFLINE=1
cd <scratch>/exp_tts
gpu run --gpus 1 --timeout 30m \
  --note "tzhouam(claude): qwen3-tts-1.7B streaming TTFA baseline, 12 prompts" -- \
  python /data/zhoutaichang/embedding_infer/vllm-omni/examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
    --query-type CustomVoice --streaming \
    --txt-prompts /data/zhoutaichang/embedding_infer/vllm-omni/examples/offline_inference/text_to_speech/qwen3_tts/benchmark_prompts.txt \
    --output-dir <scratch>/exp_tts/out --log-dir <scratch>/exp_tts/logs
gpu release
```
Prompts: the 12 English sentences shipped in [`benchmark_prompts.txt`](../vllm-omni/examples/offline_inference/text_to_speech/qwen3_tts/benchmark_prompts.txt). Requests are issued sequentially by the example (`main_streaming` loops over prompts one at a time), so this measures single-stream latency, not throughput.

**Job record.**

| Field | Value |
|---|---|
| Scheduler | `canhazgpu` via `gpu run`; type RUN |
| Physical GPU assigned | 6 (log line `Stage 0 logical-to-physical device mapping: 0->6`) |
| Main PID | 2197539 |
| Start | see [experiments/vllm_omni_tts/run1_start.txt](experiments/vllm_omni_tts/run1_start.txt) |
| Full stdout/stderr | [experiments/vllm_omni_tts/run1_stdout.log](experiments/vllm_omni_tts/run1_stdout.log) |

**Run history (three submissions, all through `gpu run`).**

| Run | Command variant | Outcome | Log |
|---|---|---|---|
| 1 | example `end2end.py --streaming`, default timeouts | **Failed at startup**: `TimeoutError: Orchestrator did not become ready within 300s`. Stage 0 was still in a cold `torch.compile` of the code predictor (talker CUDA graphs done at 00:13:12, predictor warmup at 00:14:22, timeout at 00:14:34). GPU 6 released by `gpu run`; no stray process. | [run1_stdout.log](experiments/vllm_omni_tts/run1_stdout.log) |
| 2 | same + `--init-timeout 1500 --stage-init-timeout 1500` | Completed (exit 0), 12 WAVs written, engine ready after 187.7 s (stage 0 startup 00:15:33→00:18:22, stage 1 00:18:22→00:19:22). But the example's per-chunk `logger.info` lines are not emitted (module logger with no handler), and in streaming mode `_save_wav` receives only the final message's audio, so the saved WAVs (0.08–1.92 s) are the last partial chunk, not the utterance. No latency data usable. | [run2_stdout.log](experiments/vllm_omni_tts/run2_stdout.log) |
| 3 | session driver [tts_stream_bench.py](experiments/vllm_omni_tts/tts_stream_bench.py) (imports the example's `_build_inputs`/`parse_args`, same flags, adds one warm-up request, records every chunk) | Completed (exit 0); engine ready after 156.8 s (compile caches warm). Results below. | [run3_stdout.log](experiments/vllm_omni_tts/run3_stdout.log), [bench_results.json](experiments/vllm_omni_tts/bench_results.json) |

**Results (run 3; 1 × L20X assigned by the scheduler; both stages co-located on it; installed vLLM-Omni `be335a86f` + vLLM 0.28.0; `HF_HUB_OFFLINE=1`).** The warm-up request (`w1`: TTFA 420.8 ms, first request after init) is excluded from the statistics.
| Metric (12 sequential requests, batch size 1) | p50 | p95 | max |
|---|---|---|---|
| TTFA, ms (request submit → first audio chunk received in the client coroutine) | 27.7 | 30.7 | 32.8 |
| Inter-chunk gap for 25-frame chunks, ms | 159.9 | 165.4 | 221.7 |
| RTF using streamed chunk audio only | 0.098 | 0.113 | 0.122 |
| RTF using streamed chunks + final-message tail | 0.083 | 0.085 | 0.086 |

Per-request detail (chunk arrival times in ms after submit; samples at 24 kHz mono):

| Req | Text (truncated) | TTFA (ms) | Chunk arrivals (ms) / samples | Streamed audio (s) | Final-message tail (s) | Wall (s) | RTF (streamed) | RTF (streamed+tail) |
|---|---|---|---|---|---|---|---|---|
| 0 | Hello, welcome to the voice synthesis … | 29.7 | 30/1920 190/48000 348/48000 | 4.08 | 0.40 | 0.380 | 0.093 | 0.085 |
| 1 | She said she would be here by noon, bu… | 30.6 | 31/1920 195/48000 353/48000 | 4.08 | 0.32 | 0.376 | 0.092 | 0.085 |
| 2 | The quick brown fox jumps over the laz… | 32.8 | 33/1920 190/48000 351/48000 | 4.08 | 0.08 | 0.360 | 0.088 | 0.086 |
| 3 | I can't believe how beautiful the suns… | 27.8 | 28/1920 187/48000 350/48000 510/48000 | 6.08 | 0.16 | 0.518 | 0.085 | 0.083 |
| 4 | Please remember to bring your identifi… | 26.5 | 27/1920 190/48000 351/48000 507/48000 | 6.08 | 0.40 | 0.536 | 0.088 | 0.083 |
| 5 | Have you ever wondered what it would b… | 30.7 | 31/1920 193/48000 353/48000 | 4.08 | 1.92 | 0.497 | 0.122 | 0.083 |
| 6 | The restaurant on the corner serves th… | 25.5 | 26/1920 191/48000 347/48000 | 4.08 | 1.20 | 0.436 | 0.107 | 0.083 |
| 7 | After the meeting, we should discuss t… | 27.5 | 28/1920 187/48000 346/48000 | 4.08 | 1.44 | 0.459 | 0.113 | 0.083 |
| 8 | Learning a new language takes patience… | 27.4 | 27/1920 188/48000 345/48000 503/48000 | 6.08 | 1.60 | 0.623 | 0.103 | 0.081 |
| 9 | The train leaves at half past seven, s… | 26.4 | 26/1920 189/48000 350/48000 503/48000 | 6.08 | 0.80 | 0.565 | 0.093 | 0.082 |
| 10 | Could you please turn down the music a… | 28.5 | 28/1920 250/48000 355/48000 | 4.08 | 1.28 | 0.450 | 0.110 | 0.084 |
| 11 | It was a dark and stormy night when th… | 27.6 | 28/1920 190/48000 348/48000 | 4.08 | 1.20 | 0.437 | 0.107 | 0.083 |

How to read these numbers:

- **Chunk structure matches the deploy config.** Every request delivers a first chunk of 1,920 samples (= 1 codec frame at 12.5 Hz, 80 ms of audio; `initial_codec_chunk_frames: 1`), then 48,000-sample chunks (25 frames = 2.0 s; `codec_chunk_frames: 25`). The streamed chunk count is therefore quantized to 1 + 25k frames, and the remainder arrives in the final message (0.08–1.92 s). The "streamed + tail" column is the best estimate of the true utterance length [Inferred: the tail was not verified to be disjoint from the last streamed chunk, but its RMS envelope is speech-like and its length varies per sentence].
- **TTFA of about 28 ms** covers talker prefill of the CustomVoice prompt, one talker decode step with the 15-codebook code predictor, code2wav of one frame, shared-memory transfer, and the async return path. This is a single-request, warm-graph number on a data-center GPU and is the floor the edge design has to be compared against, not a target for edge.
- **Steady-state cadence is about 160 ms per 25 frames**, i.e. roughly 6.4 ms per codec frame for talker decode + code predictor + code2wav together, well under the 80 ms real-time budget per frame (RTF ≈ 0.08–0.10).
- **Startup dominates**: 157–188 s of engine init (two vLLM engine processes, CUDA-graph capture, `torch.compile`), versus 0.4–0.6 s per utterance. This is the single most important qualitative fact for edge: the vLLM-Omni process model is built for long-running servers, not for cold-start on a device.
- Memory: stage 0 requested 30 % of the card (41.94 GiB budget by config, not need); actual model footprint was not isolated in this run [Unknown].

Artifacts kept: the JSON above and the logs. The 13 WAV files and the run-2 WAVs stay in the session scratch directory (not copied, ~2 MB).


## 3. Experiment B — llama.cpp CPU-only edge-native baseline (no GPU)

**Purpose.** Validate the "edge-native baseline" path that [vllm_omni_edge_feasibility.md](vllm_omni_edge_feasibility.md) compares against: HF checkpoint → GGUF → quantize → CPU inference, for a small VLM whose shape (SigLIP vision encoder + ~400M LLM) is representative of a VLA perception/language backbone. Also measure how much the vision encoder costs relative to token decode, which drives the VLA stage-placement argument in [vla_edge_design.md](vla_edge_design.md).

**Model.** `HuggingFaceTB/SmolVLM2-500M-Instruct` (cached, 1.9 GB safetensors). Chosen because it was the only cached checkpoint with weights that llama.cpp's converter supports for both text and vision (`conversion/smolvlm.py`, `tools/mtmd/mtmd-image.cpp::mtmd_image_preprocessor_idefics3`).

**Build (out-of-tree, checkout untouched).**
```bash
cmake -S llama.cpp -B <scratch>/llama_build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
cmake --build <scratch>/llama_build -j 32 \
      --target llama-cli llama-bench llama-simple llama-mtmd-cli llama-tts llama-quantize llama-server
```
Configure log: [experiments/llama_cpp_cpu/llama_cmake_configure.log](experiments/llama_cpp_cpu/llama_cmake_configure.log) (`GGML_SYSTEM_ARCH: x86`, CPU backend variant `-march=native`). Build finished with exit 0 ([tail](experiments/llama_cpp_cpu/llama_cmake_build.tail.log)). Resulting shared libraries: `libggml-base.so` 0.92 MB, `libggml-cpu.so` 1.53 MB, `libggml.so` 0.06 MB; `llama-cli` executable 1.3 MB (dynamically linked, x86 with AVX-512 code paths, so an ARM/Android build will differ).

**Conversion and quantization (CPU).**
```bash
PYTHONPATH=llama.cpp/gguf-py python3 llama.cpp/convert_hf_to_gguf.py <snapshot> --outfile smolvlm2-500m-f16.gguf --outtype f16
PYTHONPATH=llama.cpp/gguf-py python3 llama.cpp/convert_hf_to_gguf.py <snapshot> --outfile smolvlm2-500m-mmproj-f16.gguf --outtype f16 --mmproj
llama-quantize smolvlm2-500m-f16.gguf smolvlm2-500m-q8_0.gguf Q8_0
llama-quantize smolvlm2-500m-f16.gguf smolvlm2-500m-q4_0.gguf Q4_0
```
Logs: [convert_text.log](experiments/llama_cpp_cpu/convert_text.log), [convert_mmproj.log](experiments/llama_cpp_cpu/convert_mmproj.log), [quantize_q8_0.log](experiments/llama_cpp_cpu/quantize_q8_0.log), [quantize_q4_0.log](experiments/llama_cpp_cpu/quantize_q4_0.log). All exit 0.

| Artifact | Size (bytes) | Note |
|---|---|---|
| `smolvlm2-500m-f16.gguf` | 820,421,728 | text decoder, 409.25 M params as reported by `llama-bench` (the vision tower is excluded) |
| `smolvlm2-500m-q8_0.gguf` | 436,805,728 | |
| `smolvlm2-500m-q4_0.gguf` | 255,864,928 | |
| `smolvlm2-500m-mmproj-f16.gguf` | 199,467,680 | SigLIP vision encoder + projector, always f16 here |

**Text decode throughput (`llama-bench`, `-p 128 -n 64 -r 3`, CPU backend only, build `e71b80510`).** Raw output: [bench_results.md](experiments/llama_cpp_cpu/bench_results.md). Thread counts 4 and 8 were chosen to mimic a mobile big-core cluster; absolute numbers on a Xeon 8480C are **not** transferable to ARM, only the ratios are indicative.

| Quant | Threads | Prompt eval pp128 (tok/s) | Generation tg64 (tok/s) |
|---|---|---|---|
| F16 | 4 | 379.6 ± 0.5 | 48.2 ± 4.5 |
| Q8_0 | 4 | 997.3 ± 6.9 | 74.0 ± 13.3 |
| Q4_0 | 4 | 935.4 ± 7.7 | 135.6 ± 1.3 |
| F16 | 8 | 592.7 ± 1.2 | 68.8 ± 7.1 |
| Q8_0 | 8 | 1684.6 ± 73.1 | 150.1 ± 21.0 |
| Q4_0 | 8 | 1522.2 ± 165.3 | 220.1 ± 28.3 |

Observations (host-specific): decode is memory-bound, so Q4_0 gives ~2.8-3.2× the F16 decode rate; prompt processing is compute-bound and saturates at Q8_0. This is a backbone-only microbenchmark: a Qwen3-TTS frame additionally runs 15 code-predictor sub-steps (`qwen3_code_predictor.py`) and the speech decoder, neither of which is measured here, so it bounds only the talker backbone's share of an 80 ms frame budget and supports no real-time claim for any device.

**Vision-language path (`llama-mtmd-cli`, Q8_0 text + F16 mmproj, 8 threads, greedy, 48 new tokens).**
```bash
llama-mtmd-cli -m smolvlm2-500m-q8_0.gguf --mmproj smolvlm2-500m-mmproj-f16.gguf \
  --image MNN/resource/images/cat.jpg -p "Describe this image in one sentence." -t 8 -n 48 --temp 0
```
Output (verbatim from [mtmd_run.log](experiments/llama_cpp_cpu/mtmd_run.log)): "A small orange kitten is sitting on a rock outdoors, looking directly at the camera." Exit 0. The image is `MNN/resource/images/cat.jpg` (a repo asset, used only as input).

From the timestamps in [mtmd_run.err](experiments/llama_cpp_cpu/mtmd_run.err): the Idefics3 preprocessor split the image into 13 image chunks (27 prompt chunks in total, image and text interleaved); each image chunk was encoded in 596-1184 ms at 8 threads, i.e. roughly **9-11 s of vision encoding per image** with default tiling. With `--image-max-tokens 64` ([mtmd_run_maxtok64.err](experiments/llama_cpp_cpu/mtmd_run_maxtok64.err)) the preprocessor produced 2 image chunks of ~640 ms each (~1.3 s). Implication for VLA (see [vla_edge_design.md](vla_edge_design.md)): the tiling policy of the image preprocessor, not the LLM, dominates per-frame cost on CPU; an edge VLA must pin a single low-resolution tile per camera, and on CPU-only targets even one SigLIP-B/16-class tile at 512 px costs ~0.6 s at 8 x86 threads in llama.cpp's f16 path, which alone exceeds a 10 Hz control budget. This is a measured fact about this host and this engine, not a projection for ARM or GPU.

## 4. Experiments not run and why

| Experiment | Status | Reason / prerequisite |
|---|---|---|
| vLLM-Omni VLA (DreamZero / GR00T / InternVLA-A1) control-loop latency on GPU | **Deferred** | Weights are not in the cache (`GEAR-Dreams/DreamZero-DROID` has metadata only); a download of tens of GB requires permission. Also the DreamZero example is a client that expects a running server. |
| vLLM-Omni Qwen3-TTS 0.6B vs 1.7B TTFA comparison | Deferred | 0.6B weights are cached, but the example hard-codes 1.7B; running 0.6B needs a small script change (source change in an example → permission) or a custom driver. Cheap once approved. |
| vLLM CPU backend (`VLLM_TARGET_DEVICE=cpu`) on x86 | Deferred | The installed vLLM is the CUDA wheel; a CPU build requires a new environment (dependency installation). |
| ExecuTorch export of a TTS/VLA sub-module to `.pte` + XNNPACK run | Deferred | ExecuTorch is not installed and its submodules are uninitialized; `./install_executorch.sh` is a dependency installation. |
| MNN `llmexport.py` + `llm_demo` CPU run | Deferred | Needs an MNN build (cheap) plus its Python export deps; kept for the roadmap's phase 0 gate. |
| ncnn Vulkan build | Deferred | `glslang` submodule uninitialized; CPU-only build possible but there is no cached model in ncnn format. |
| Any ARM / Apple Silicon / Android / Jetson measurement | **Blocked** | No such hardware is reachable from this host. |

## 5. Reproducible benchmark protocols proposed for the edge work

These are proposals (not executed) that the roadmap's acceptance criteria refer to; they reuse only artifacts shown to exist above.

**P-TTS-1 (server reference).** Experiment A as-is, plus `--batch-size 1..8` with `--use-batch-sample`, reporting TTFA p50/p95, inter-chunk p95, RTF = generation wall time / audio seconds, and peak `nvidia-smi` memory. Acceptance for any edge port: same text set, same speaker, TTFA and RTF reported with the identical definitions, audio compared with a speech-recognition WER check on the 12 sentences.

**P-TTS-2 (edge talker).** Talker exported alone (see [tts_edge_design.md](tts_edge_design.md) for the split); decode tokens/s measured by the target engine's bench tool (`llama-bench`, MNN `llm_bench`, ExecuTorch `llama_main` timing) at 4 and 8 threads; must exceed 12.5 codec frames/s × (1 + margin) for the "real-time" gate.

**P-VLA-1 (perception cost).** Experiment B's `llama-mtmd-cli` run with `--image-max-tokens` swept over {64, 256, default}, plus the same encoder exported to ExecuTorch/MNN, reporting ms per frame at fixed resolution; gate: encoder + fusion + action head per control step below the chosen control period with p95 jitter below 20 % of it, measured in simulation first.

## 6. Cleanup

- GPU: `gpu run` released the reservation after each of the three runs; `gpu release` was run again after runs 2 and 3 and `gpu status --no-schedule` showed no reservation held by `zhoutaichang` ([cleanup_status.txt](experiments/vllm_omni_tts/cleanup_status.txt), [cleanup_status3.txt](experiments/vllm_omni_tts/cleanup_status3.txt)); `nvidia-smi` showed 0 MiB used on all eight cards afterwards.
- Build trees, GGUF files (about 1.7 GB), and generated WAV files live only in the session scratch directory and are not part of the repository; the logs copied under [experiments/](experiments/) total a few hundred KB.
- No checkout was modified (`git status --porcelain` empty for all six after the runs; see [repository_inventory.md](repository_inventory.md)).

## 7. Reference verification

All Markdown files under `analysis/` were scanned with [experiments/linkcheck.py](experiments/linkcheck.py): every relative link target must exist, and every `#Lnnn` anchor must be within the target file's length. Final pass: 2,677 links checked, 0 broken (three stale line anchors in the model-level report were corrected during the pass).

## 8. Edge branch experiments (2026-09-09)

All runs of this section were executed from the `edge/omni-edge` branch of `/data/zhoutaichang/embedding_infer/vllm-omni` with the two approved environments (`../.venvs/omni-cuda`: vllm 0.28.0 CUDA + editable vllm-omni; `../.venvs/omni-cpu`: vllm 0.28.0+cpu wheel). Results are committed JSON under `vllm-omni/benchmarks/tts/edge_results/` and are rendered as tables in [experiments/edge_branch/results_summary.md](experiments/edge_branch/results_summary.md) (regenerate with `python benchmarks/tts/edge_results/summarize.py`). The status per work package is in [edge_branch_plan.md §5](edge_branch_plan.md#5-implementation-status-branch-edgeomni-edge-2026-09-09).

### 8.1 GPU scheduling procedure

The rules of §0 were applied unchanged, with the user's constraint of **at most one GPU at a time**: every GPU job was one `gpu run --gpus 1 --timeout 40m --wait 30m --note "tzhouam(claude): <wp> ..." -- env HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn ...` invocation, executed **sequentially** by three shell chains ([chain1.sh](experiments/edge_branch/chain1.sh), [chain2.sh](experiments/edge_branch/chain2.sh), [chain3.sh](experiments/edge_branch/chain3.sh); each chain waits for the previous one to exit before it starts). Each chain ran from a detached git worktree snapshot of the branch so that code edits during a run could not change a running job. `gpu release` and `gpu status --no-schedule` followed every job; the job list from `gpu report --days 7` is in §8.9. CPU-only runs (CPU platform sweep, x86/ARM emulation, decoder export) used no GPU (`VLLM_TARGET_DEVICE=cpu`, `CUDA_VISIBLE_DEVICES=` empty, pinned to NUMA node 1 with `taskset`).

### 8.2 Streaming latency: default vs edge profile (WP0/WP1)

`benchmarks/tts/stream_latency_bench.py`, 12 prompts × 3 repeats after 1 warm-up, sequential single-stream requests, metrics from `stream_latency_metrics.py` (TTFA; playback-start latency = max over chunks of arrival time minus audio already buffered; stall at TTFA = playback-start − TTFA; RTF = generation wall time / audio seconds).

| model | deploy | run | init s | TTFA p50 / p90 ms | playback-start p50 ms | stall p50 ms | RTF | peak GPU MiB |
|---|---|---|---|---|---|---|---|---|
| 1.7B | default | 20260909-103504 | 271 | 32.0 / 35.9 | 116.2 | 84.7 | 0.10 | 46272 |
| 1.7B | edge | 20260909-145133 | 149 | 43.7 / 48.2 | 43.7 | 0.0 | 0.10 | 40876 |
| 0.6B | default | 20260909-104204 | 152 | 27.8 / 32.0 | 95.4 | 67.3 | 0.10 | 46846 |
| 0.6B | edge | 20260909-145500 | 144 | 40.7 / 45.2 | 40.7 | 0.0 | 0.10 | 41248 |

The edge profile removes the first-chunk stall entirely (the ramp's 2-frame first chunk lands before the 4-frame second chunk is needed) for ≈12 ms of TTFA; RTF and peak memory (per-process, both stages on one card) are unchanged within noise. With the profile's `parallel_stage_init` active (chain 2, runs 20260909-152439/152742, `edge_final2`): 1.7B init 87.7 s, TTFA p50 46.4 ms, stall 0; 0.6B init 86.7 s, TTFA p50 54.0 ms (p90 97 ms, one noisy repeat), stall 0 — the init gain (149 → 88 s) with the same steady-state latency.

### 8.3 Init timeline and cold start (WP2)

`benchmarks/tts/init_profile.py` (1.7B, one config per subprocess, `VLLM_OMNI_INIT_TIMELINE` phases from orchestrator and workers; run 20260909-112200 warm caches, 20260909-144054 cold caches):

| config | init s | first-request TTFA ms | engine_core_init s | load_weights s | talker MTP capture s | predictor compile s | code2wav graphs s |
|---|---|---|---|---|---|---|---|
| default | 152.0 | 646 | 66.2 | 8.9 | 27.4 | 11.6 | 3.6 |
| eager_stage1 | 149.9 | 781 | 67.1 | 6.4 | 29.1 | 11.4 | 0 |
| eager_both | 110.9 | 14956 | 27.8 | 6.2 | – | – | 0 |
| parallel_stage_init | 94.6 | 520 | 75.4 | 10.5 | 26.9 | 12.3 | 3.7 |
| edge (profile, serial init) | 145.7 | 512 | 51.4 | 12.7 | 7.2 | 5.1 | 5.8 |
| cold_cache (fresh VLLM/inductor/triton caches) | 355.5 | 465 | – | – | – | – | – |

Reading: the two stages initialise serially by default and each spends 50–75 s inside vLLM's `EngineCore.__init__` (device init, weight load, KV profiling, graph capture); running them concurrently on the shared card (`parallel_stage_init`) saves 57 s. Disabling graphs on both stages saves 41 s but the first request then pays the talker-MTP and predictor warm-up (15 s), so the edge profile keeps graphs. A cold inductor/Triton cache adds 200 s; the caches must ship with an edge image.

### 8.4 CPU platform (WP3) and CPU thread sweep

`benchmarks/tts/cpu_thread_sweep.py` (0.6B CustomVoice, `+cpu` wheel, bf16 with AMX on the 8480C, stage 0 gets 3/4 of the threads, `--step-stats`), run 20260909-145257 on NUMA node 1 while the GPU chains ran on other cores:

| threads (stage0+stage1) | init s | TTFA p50 ms | playback-start p50 ms | RTF | peak RSS GiB | talker step ms (mean / p95) | scheduler ms per step | chunk hop enqueue→put / get ms |
|---|---|---|---|---|---|---|---|---|
| 8 (6+2) | 112.7 | 426 | 9438 | 2.78 | 13.3 | 250.5 / 499.7 | 0.13 | 1.37 / 0.94 |
| 16 (12+4) | 109.5 | 292 | 5346 | 1.84 | 13.3 | 147.4 / 219.6 | 0.12 | 0.82 / 0.79 |
| 16 + inductor (`VLLM_OMNI_CPU_INDUCTOR=1`) | 103.4 | 234 | 3642 | 1.43 | 13.3 | 138.9 / 180.0 | 0.12 | 0.83 / 0.91 |
| 32 (24+8) | 155.7 | 252 | 4995 | 1.73 | 13.5 | 136.2 / 168.9 | 0.11 | 0.86 / 0.55 |

The talker step (one 80 ms codec frame per step, including the 15 code-predictor sub-steps) costs 136–250 ms, i.e. 1.7–3.1× the frame budget, and stops scaling beyond 16 threads; inductor gives 6–22 %. Scheduler work is 0.1 ms per step and the shared-memory chunk hop about 1 ms per chunk, together under 2 % of the budget. The earlier `cpu_smoke` run (12+4 threads from the static overlay) is in the summary file (TTFA 292 ms, RTF 1.7). Conclusion unchanged from the study: this CPU class serves offline synthesis, not real-time streaming; a 4-core/AVX2 box (§8.5) is 12× slower than real time.

### 8.5 Hardware emulation (WP4)

`benchmarks/tts/hw_emulation.py` with `deploy_profile="auto"` and `VLLM_OMNI_HW_PROFILE` overrides (0.6B, 12 prompts × 1). CPU cases ran in the CPU venv pinned to N cores with `ATEN_CPU_CAPABILITY` downgrades; GPU cases ran on the L20X with the derived absolute KV budgets and eager rules of the emulated class (the card's memory is not capped, so peak GPU MiB shows what the budget admits, not a hard limit).

| case | class chosen | dtype / threads or budget | init s | TTFA p50 ms | playback-start p50 ms | RTF | peak RSS GiB / GPU MiB |
|---|---|---|---|---|---|---|---|
| x86_cpu_avx2 (4 cores, 8 GiB) | x86_cpu | fp32, 3+1 threads, KV 1.2 GiB | 120 | 2651 | 54419 | 11.9 | 12.8 / – |
| x86_cpu_avx512 (8 cores, 16 GiB) | x86_cpu | 6+2 threads | 109 | 679 | 7874 | 2.45 | 14.4 / – |
| x86_cpu_amx (16 cores, 32 GiB) | x86_cpu | bf16, 12+4 threads | 116 | 513 | 5692 | 1.72 | 23.0 / – |
| arm64_cpu_logic (RK3588-like, logic only) | arm64_cpu | fp32, 3+1 threads, KV 2.2 GiB | 108 | 2183 | 50262 | 11.9 | 13.0 / – |
| jetson_orin_8g (unified 8 GiB) | jetson | eager both, absolute KV budget | 129 | 181 | 196 | 0.57 | – / 6236 |
| jetson_orin_32g (unified 32 GiB) | jetson | graphs on | 183 | 68 | 68 | 0.09 | – / 23376 |
| cuda_discrete_8g (8 GiB VRAM) | cuda_discrete | eager + small KV | 189 | 47 | 47 | 0.09 | – / 9680 |

The derivation never allowed preemption (checked by `tests/core/test_edge_scheduling.py` against the exact talker KV geometry, 112 KiB per token). The Jetson-8 GB emulation shows the cost of running eager on a small budget (RTF 0.57, TTFA 181 ms) and is the case where the S3 native path should be compared on real hardware.

### 8.6 Codec-token streaming: cadence and parity (WP5)

`benchmarks/tts/codec_stream_e2e.py` launches the talker-only server (eager stage 0), streams codec frames to the client decoder (CPU), then launches the full two-stage server (eager stage 0, same seed) for the reference PCM, and reports client TTFA and the length-trimmed SNR. Four runs were needed:

| run | outcome | root cause / fix |
|---|---|---|
| chain 1 (20260909-151221) | client crashed in `batched_chunked_decode` | the decoder takes a `[B, Q, F]` tensor, not a list; fixed in `536866b9` and verified against the real 0.6B decoder on CPU |
| chain 2 (20260909-154659) | frames decoded, but all codec frames arrived in one message ≈2.1 s after the request (client TTFA 2.2–2.7 s = whole generation), SNR −3 dB | see below |
| chain 4 (20260909-160935) | same, with counters: **1 engine output per request, first codes at output #1** | the speech path forced `FINAL_ONLY` outputs for Qwen3-TTS deployments without async chunks (`qwen3_full_payload`), which a talker-only deployment also is; fixed in `b22dabe2` (talker-only = final stage with latent output keeps DELTA) |
| chain 5 (20260909-162326) | server now emits per step (2 outputs by step 2) but failed: "inhomogeneous shape" in row alignment | in DELTA mode the payload is a list of per-step tensors; normalized in `380e8022` |
| chain 6 (20260909-163326) | 53 engine outputs for 52 rows, yet the client still got one message | each per-step payload is the *newest row only*; the alignment treated it as a cumulative snapshot (one yield at row 1, one at the end); fixed in `c893b638` (`accumulate_codec_rows`, unit-tested with per-step deltas) |
| chain 7 (20260909-164759) | **streams per step**: client TTFA 152–156 ms, playback-start = TTFA, stall 0, chunks 2/4/8/16/25 frames arriving 145 / 285 / 570 / 910 ms apart (≈36 ms per frame from the eager talker-only server + CPU decode), client RTF 0.5–0.6, first utterance 15 s (server warm-up) | – |

Parity was separated into two questions with `benchmarks/tts/codec_parity_diag.py` on the chain-4 codes (client now saves the received rows): (1) client chunked decode vs the exact non-chunked decode of the same codes: **72.7–75.9 dB SNR, correlation 0.99999999** for all four utterances, so the client-side decoder (ramp schedule, 72-frame context, stateful caches) reproduces the server's `Code2Wav` math; (2) the full server's PCM vs the exact decode of the client's codes: −3 dB, correlation ≈ 0 with identical frame counts, i.e. the two server processes sampled different code sequences for the same text and seed (the `seed` request field reaches both paths, so the divergence is numeric between the talker-only and the two-stage deployments; the speech API exposes no temperature, so a greedy comparison was not possible). The ≥40 dB end-to-end acceptance is therefore **not demonstrated**; decoder parity is, and a token-level tap on the two-stage server (or a greedy sampling switch) is the follow-up. The wire cost is unchanged from the design (16 int16 codebooks + 7-byte header per 80 ms frame ≈ 490 B/s). `tests/e2e/online_serving/test_qwen3_tts_codec_stream.py` passed in chain 2 (frames + decode), before the cadence fixes; it does not assert cadence.

### 8.7 Per-step attribution on GPU (WP8) and the CUDA-IPC connector hop (WP7)

`stream_latency_bench.py --step-stats` (0.6B, 12 prompts × 3, chains 2 and 3). Counters per process role; share = mean / 80 ms frame budget (one talker step = one frame).

| counter | default deploy (shm) | edge profile | cuda_ipc deploy |
|---|---|---|---|
| stage 0 engine-core step (talker + code predictor) | 6.43 ms (p95 10.8) = 8.0 % | 5.96 ms = 7.4 % | 5.79 ms = 7.2 % |
| stage 0 `schedule()` + `update_from_output()` | 0.12 + 0.12 ms = 0.3 % | 0.09 + 0.09 ms = 0.2 % | 0.09 + 0.08 ms = 0.2 % |
| stage 1 engine-core step (code2wav, incl. polling steps) | 0.54 ms | 0.38 ms | 0.35 ms |
| chunk hop: enqueue→put / put / get | 0.62 / 0.48 / 0.68 ms | 0.33 / 0.23 / 0.34 ms | 0.34 / 0.29 / 0.36 ms |
| chunk hop put→get (lock-file mtime → get, chain 3) | 2.42 ms (p95 2.58) = 3.0 % | – | 2.90 ms (p95 10.7) = 3.6 % |

Everything outside the model step (scheduler, transport, polling) is below 2 % of the frame budget per stage, and the whole hop from talker put to code2wav get is 2.4 ms with shared memory. CUDA-IPC does not improve it (2.9 ms mean, worse tail): on this edge the chunk tensor is already a small host tensor (codes), so the IPC export/ack adds work without removing a copy. **Decision: `qwen3_tts_cuda_ipc.yaml` stays optional; the edge profile keeps `SharedMemoryConnector`.** The orchestrator dispatch counter only fires under the event-driven orchestrator loop (`VLLM_OMNI_EVENT_DRIVEN_ORCH=1`); the default poll loop was used here, so it is absent from the tables.

### 8.8 Decoder export (WP4, phone/NPU class)

`python -m vllm_omni.edge.decoder_export --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --ramp 2 25` (CPU, no GPU): two `torch.export` programs (window 74 and 97 frames), exact match against the eager decoder, 460 MB fp32 each (manifest committed as `benchmarks/tts/edge_results/decoder_export_manifest_0.6B_torch_export.json`). ExecuTorch lowering not executed (package not installed; install is permission-gated).

### 8.9 Job records and cleanup

See the `gpu report --days 7` extract in [experiments/edge_branch/gpu_report.txt](experiments/edge_branch/gpu_report.txt) (all jobs of this section are the `tzhouam(claude): WP…` notes, type RUN, one card each, none overlapping). `gpu status --no-schedule` showed no reservation by `zhoutaichang` after the last chain; the worktrees `vllm-omni-run*` were removed; venvs stay outside the tree.
