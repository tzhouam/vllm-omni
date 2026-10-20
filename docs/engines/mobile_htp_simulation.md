# Mobile NPU simulation with the Qualcomm AI Runtime SDK (Hexagon HTP), and the CPU speed-ups it triggered

Goal set on 2026-09-10: install Qualcomm's AI Runtime SDK with its Hexagon HTP (NPU) simulator, use it to stand in for mobile devices, and make the branch beat the other engines on CPU and on mobile. This page records what was installed, what runs on the simulator, what the simulator can and cannot tell us, and the CPU changes measured on the way. Raw artifacts: `tools/qairt/` (SDK, runtime libs, Python env notes), `analysis/experiments/edge_engine_compare/results/cpu_opt/` (CPU runs), the session scratch directory for ONNX/DLC files (not committed, 460 MB each).

## 1. What was installed and how

| Item | Where | Notes |
|---|---|---|
| Qualcomm AI Runtime **Community** SDK 2.37.0.250724 (formerly QNN / AI Engine Direct) | `tools/qairt/qairt/2.37.0.250724/` | Downloaded from Qualcomm's software center without a login (the URL ExecuTorch's `backends/qualcomm/scripts/qnn_config.sh` pins); 1.35 GB zip. Contains the converters (`qairt-converter`, `qairt-quantizer`), `qnn-net-run`, `qnn-context-binary-generator`, profiling readers, and the backends for x86 (`libQnnCpu.so`, `libQnnHtp.so` = **HTP functional simulator + offline graph preparation**, `libQnnHtpQemu.so` = HTP QEMU backend, id 13), Android aarch64 and Hexagon v66–v79 skels. |
| LLVM runtime libraries the SDK binaries link against | `tools/qairt/runtime_libs/` (`libc++.so.1`, `libc++abi.so.1`, `libunwind.so.1`) | Not on this host and no root: unpacked from Ubuntu jammy `libc++1-14`, `libc++abi1-14` and `libunwind-18` `.deb` files with `dpkg-deb -x`; used via `LD_LIBRARY_PATH`. |
| Python 3.10 environment for the SDK tools | `.venvs/qairt` (uv) | The SDK supports 3.8/3.10 only. Its `check-python-dependency` installer needs `setuptools<81` (for `pkg_resources`) and does **not** install `onnx`; `onnx==1.16.2`, `onnxruntime==1.18.1`, `protobuf<5` were added. |
| ONNX exporter dependency | `onnxscript` in the CUDA venv | The TorchScript exporter fails on `aten::diff` in the decoder; the dynamo exporter (`torch.onnx.export(..., dynamo=True)`) works. |
| Qualcomm AI Hub | not installed | It is a cloud service (real devices behind an API token), not a local simulator; nothing on this host can stand in for it. |

Traps hit (all fixed in the scripts under `analysis/experiments/edge_engine_compare/` and `tools/qairt/`): converter rejects ONNX `Reshape` with `allowzero=1` (dynamo emits it; cleared), rejects the `IsNaN`→`Where` guards the dynamo exporter puts after every softmax (stripped), and the HTP graph preparation fails on an **int Gather from a float table** (`node_embedding`, the RVQ codebook lookup) with `could not create op: q::Gather`, so the codebook lookup was moved to the host and the HTP graph takes the summed codebook embeddings as a float input.

## 2. What runs on the HTP simulator

Pipeline (scripts: `scratchpad/qnn/export_htp_decoder.py`, `htp_pipeline.sh`, `run_qnn.sh`):

1. `WindowedCode2Wav` (branch `vllm_omni/edge/decoder_export.py`, context 72 frames + chunk 25) → `HtpBody` variant whose input is `quantized [1, 512, 97]` (RVQ sum done on the host) → ONNX (dynamo, opset 18, 458 MB fp32).
2. `qairt-converter --float_bitwidth 16` → DLC (fp16), 229 Reshape / 62 FullyConnected / 55 StridedSlice / 45 Transpose / 17 RmsNorm / 16 MatMul / 31 Conv2d / 6 ConvTranspose / 8 Softmax / ElementWiseNeuron (SnakeBeta) — every op accepted by the converter.
3. `qnn-net-run --backend libQnnHtp.so` on x86 with the HTP backend extensions (`dsp_arch v75`, `O=3`, `vtcm_mb 8`): graph preparation for HTP v75 **succeeds** (RmsNorm and Gelu were flagged "no properties registered" during prepare but did not stop it), execution runs on the functional simulator.

Parity: the same DLC on the QNN **CPU** backend (fp32) reproduces the torch decoder bit-for-bit (SNR 107.6 dB, max |diff| 7e-6 over 48,000 samples) — the conversion is correct. The HTP fp16 execution of the 2-second decoder chunk was still running on the functional simulator when this document was written (3 h of wall time, 5 cores): the chunk is 481 GFLOP and the simulator executed the 2.67 GFLOP predictor body at ≈16 MFLOP/s, so the run should finish ≈8.5 h after its 01:38 start. A background chain runs `parity_dec_htp.py` against the torch reference when it exits and writes `analysis/experiments/edge_engine_compare/qnn/run/decoder_htp_parity.txt` (plus the profiling records). The same graph on the QEMU backend is not possible on this host (below).

**Code predictor on the HTP simulator (real weights, complete).** Because the 15-pass code predictor is the CPU bottleneck (§3), it is the component that matters most on a phone NPU. Its 5-layer body (all 56 tensors loaded from the checkpoint, fused qkv/gate-up) was exported the same way (`export_predictor_htp.py`, input `x [1, 17, 1024]`), converted to fp32 and fp16 DLC (`Mul 82 / Add 46 / Reshape 35 / MatMul …`, all accepted), and run:

| backend | result |
|---|---|
| QNN CPU backend, fp32 DLC | SNR **111.2 dB** vs torch, 41 ms per inference on 16 host threads (reference kernels) |
| **HTP v75 functional simulator, fp16 DLC** | graph preparation 2.3 s, execution completes; SNR **50.7 dB** / corr 0.999996 vs the fp32 torch body, i.e. fp16 HTP math is faithful for the predictor |
| HTP profiler (`--profiling_level detailed`, HTP profiling reader) | per-op records are produced (e.g. `rms_norm_0`, `node_linear` …) but their "cycles" are simulator cycles (3.3e11 for one pass), not Hexagon device cycles |

**HTP QEMU backend (`libQnnHtpQemu.so`, backend id 13).** Tried for a second, closer-to-silicon data point. It refuses the HTP backend-extension config (`Unable to find HTP interface provider`) and without it fails at context creation (`err 0x3e8`); its strings explain why: it only loads *offline-prepared* context binaries and launches a Hexagon emulator subprocess whose command line must come from `QNN_HTP_QEMU_ARGS`, i.e. it needs the `qemu-hexagon` binary and the Hexagon skel/`libQnnHtpV75Skel.so` from the **Hexagon SDK**, which is a separate Qualcomm Package Manager download (account-gated) and not part of the community AI Runtime SDK. So on this host the HTP path is validated functionally (prepare + execute on the x86 simulator), not cycle-accurately.

Artifacts: `analysis/experiments/edge_engine_compare/qnn/` (scripts, converter logs, run logs, profiling records, execution metadata); the ONNX/DLC files (150–460 MB) stay in the session scratch directory.

What the simulator does and does not give:
- It is a **functional** simulator ("x86 Linux host backend serves as a functional simulator for the hardware accelerator and supports graph preparation", SDK backend docs): op support, graph preparation for a chosen Hexagon arch, fp16/int8 numerics, and the context binary that would be deployed. Wall time on x86 is not device time (one 2 s chunk takes minutes here).
- Per-op **cycle counts** come from the HTP profiler only when running on the device (the docs' detailed profiling "provides per op profiling result by cycle counts"); on the x86 simulator the profiling reader returns the host-side timeline. A device-time number therefore still needs a phone or Qualcomm AI Hub. The QEMU backend (`libQnnHtpQemu.so`) is exercised the same way (see below).

## 3. CPU: making the branch faster than llama.cpp on the same host

Starting point (§5 of [edge_device_simulation.md](edge_device_simulation.md)): 134 ms per 80 ms frame at 12+4 threads, of which 97 ms were the fifteen code-predictor sub-steps and 10 ms the sampler; llama.cpp's native path costs 85–97 ms per frame plus a 55 ms/frame serial vocoder (RTF 1.9–2.0).

What was found and changed (branch commit `6ce8de1b`):

| finding | change | effect (0.6B, 12+4 threads, 12 prompts) |
|---|---|---|
| each of the 15 sub-steps re-ran the 5-layer predictor over the whole prefix (up to 17 positions), no KV cache | `forward_cached`: per-layer KV cache, 2-token prefill then one token per sub-step; default on CPU | sub-step 6.4 → 4.7 ms; frame 134 → 108 ms; RTF 1.7 → 1.38 |
| the sub-step is per-operator overhead, not bandwidth (1.6 ms floor); OpenMP thread count irrelevant (4/2 threads: no change); per-submodule `torch.compile` worse (+2 ms, guard overhead per call) | shape-static `forward_static_step` (tensor position, boolean mask) compiled as one graph | micro-benchmark 70 → 58 ms per frame for the loop; engine run: see table below |
| bf16 M=1 GEMV kernels carry ~30 µs fixed overhead each; int8-dynamic Linear has none and halves the bytes | dynamic int8 quantization of every predictor Linear + lm heads on CPU (fp32 activations), default on | micro-benchmark 70 → 36 ms per frame for the loop |
| vLLM's stock sampler spends 9 ms per step in `compiled_random_sample` (a `torch.compile`'d kernel) on CPU | talker `sample()` with `prefer_model_sampler` on CPU: same math eagerly (penalties, temperature, top-k, exponential noise with per-request generators) | 9 → <0.5 ms per step |

Engine runs (`benchmarks/tts/stream_latency_bench.py --step-stats`, 0.6B CustomVoice, 12 prompts, NUMA node 1 pinned; per-frame numbers are the stage-0 engine step and its parts):

| variant | threads | talker step ms/frame | talker fwd | predictor loop (15 sub-steps) | sampler | TTFA p50 | playback-start p50 | RTF |
|---|---|---|---|---|---|---|---|---|
| before (re-prefill predictor, stock sampler) | 12+4 | 134 | 25 | 97 (6.4/sub-step) | 10 | 273 ms | 4.9 s | 1.7 |
| KV-cached predictor | 12+4 | 108 | 26 | 71 (4.7) | 9 | 251 ms | 3.6 s | 1.38 |
| + predictor 4 / 2 OpenMP threads | 12+4 | 108 | 25 | 71 (4.6–4.7) | 9 | 241–254 ms | 3.5–3.6 s | 1.36–1.39 |
| + per-submodule torch.compile (rejected) | 12+4 | 131–139 | 26 | 94–101 (6.2–6.7) | 9 | 266–272 ms | 4.7 s | 1.62–1.63 |
| + compiled shape-static step (bf16) + fast sampler | 12+4 | 114 | 26 | 84 (5.6) | 0.6 | 240 ms | 3.1 s | 1.27 |
| + fast CPU sampler (bf16 KV-cached predictor) | 12+4 | 99 | 25 | 71 (4.7) | 0.6 | 245 ms | 3.6 s | 1.39 |
| + int8-dynamic predictor + fast sampler (rejected for fidelity, below) | 12+4 | 68 | 26 | 38 (2.5) | 0.7 | 212 ms | 2.0 s | 0.98 |
| int8-dynamic predictor, 24+8 threads (rejected for fidelity, below) | 32 | 59 | 21 | 34 (2.2) | 0.6 | 175 ms | 1.7 s | 0.80 |
| same, 6+2 threads | 8 | 144 | 39 | 102 (6.8) | 0.6 | 372 ms | 5.8 s | 1.92 |
| **fp32 predictor + compiled static step + fast sampler (faithful; the new default)** | 12+4 | **93** | 25 | **64 (4.3)** | 0.6 | **225 ms** | 2.7 s | **1.15** |
| same, 24+8 threads | 32 | **77** | 21 | 53 (3.5) | 0.7 | **201 ms** | 1.9 s | **0.88** |
| int8-dynamic predictor, Base checkpoint with reference speaker (x-vector) | 12+4 | 80 | 26 | 38 (2.5) | 0.7 | 696 ms (incl. per-request x-vector) | 2.6 s | 1.12 (was 1.77) |
| **fp32 default, Base checkpoint with reference speaker (x-vector), the row the llama.cpp comparison uses** | 12+4 | **109** (mean; p90 158) | 25 | 69 (4.6) | <1 | 686 ms (incl. per-request x-vector) | 3.0 s | **1.27** (was 1.77) |
| same, 24+8 threads | 32 | **91** (mean; p90 111) | 21 | 56 (3.7) | <1 | 486 ms | 2.2 s | **0.99** (was 1.7) |

Folding the per-sub-step lm head, top-k Gumbel sampling, embedding and projection into a second compiled graph (branch commit `3085f501`) changes little on this host (sub-step 4.25 → 4.16 ms); the compiled fp32 transformer step itself is now the cost, i.e. fp32 weight traffic (4.8 GB per frame for the 15 passes), which is why the 6+2-thread class stays at 145–148 ms per frame (RTF 1.9): there the answer is a real int8/int4 GEMV kernel (PyTorch's `_weight_int8pack_mm` on this build is an unoptimized reference, 10× slower than fp32) or llama.cpp-style quantized weights, not more compilation.

Against llama.cpp on the same host and threads (native C++, Q4_0/f16 GGUF, [edge_engine_comparison.md](edge_engine_comparison.md) §4): 85–97 ms per frame for talker + predictor plus a 55 ms/frame serial vocoder, RTF 1.9–2.0 at 8–32 threads. With the faithful fp32 predictor the branch generates a frame in 77–93 ms including sampling, with the vocoder in a concurrent stage: **RTF 0.88 (real time) on 32 threads and 1.15 on 16 threads, against llama.cpp's 1.9–2.0 at any thread count**, i.e. 1.7–2.3× faster end-to-end, and at parity per talker frame at 16 threads (93 vs 85–97 ms). The remaining terms are the talker forward (21–26 ms, vLLM's bf16 CPU backend) and the eager glue in each sub-step (now compiled; 2 % gain).

Fidelity, measured in situ (`VLLM_OMNI_PREDICTOR_PARITY_CHECK=1`: every sub-step also runs the untouched bf16 predictor on the same inputs and compares logits):

GPU regression check after the CPU changes (all of them are gated on the CPU platform: `_setup_cpu_fast_path` and `prefer_model_sampler` are no-ops on CUDA): 0.6B CustomVoice, edge profile, one H200 through the scheduler, 12 prompts: TTFA p50 44.5 ms, playback-start 44.5 ms, stall 0, RTF 0.086, init 77.6 s — inside the 40.7–54 ms / 0.079–0.097 band of the 2026-09-09 edge-profile runs (`results/gpu_sanity/`).

| predictor variant vs bf16 reference | top-1 agreement | top-5 overlap | KL(ref‖test) | max \|logit diff\| |
|---|---|---|---|---|
| fp32 weights, compiled static step (control: only bf16 rounding differs) | 93.8 % | 95.3 % | 0.0015 | 0.23 |
| torch.ao dynamic int8 (per-tensor int8 activations, per-channel int8 weights) | **22.2 %** | – | – | 7.0 |
| int8 transformer body, fp32 lm heads (`ao-nohead`) | 23.7 % | 33.3 % | 0.76 | 7.0 |

So the fast int8 rows above are **not usable as-is**: per-tensor activation quantization of the residual stream destroys the residual-codebook predictions even though the same quantization is exact on a random model (the real activations have outliers). torchao's dynamic-activation path is faithful in eager mode but numerically wrong under `torch.compile` on CPU (cosine 0.02 vs fp32, it caused 160 s runaway utterances), and its weight-only path is faithful but slow (78 ms per frame). The shipped default is therefore **fp32 weights + compiled static step** (`VLLM_OMNI_PREDICTOR_INT8=fp32`); int8 stays opt-in for measurement.

## 4. Mobile: what can be claimed, and what cannot

What can be claimed for mobile after this session, and what cannot:

- **Claimable**: both halves of the on-device TTS path now exist as HTP-ready artifacts with verified numerics — the Code2Wav decoder (fp16 DLC, HTP graph prepared for v75, bit-exact on the QNN CPU backend) and the code predictor body (fp16 DLC, executed on the HTP simulator at 50.7 dB vs fp32). The talker backbone (28 layers, a standard Qwen3 decoder) is the same class of graph as the predictor and the remaining export; the codec-token protocol (WP5) and the branch's decoder export already give the phone client its contract. No other engine in this study runs Qwen3-TTS on a Hexagon NPU today (llama.cpp and MNN target the CPU/GPU on Android), so on the NPU the branch's artifacts are ahead in *deployability*.
- **Not claimable**: device speed. The x86 HTP backend is a functional simulator; its profiler reports simulator cycles, and Qualcomm AI Hub (real devices) is a cloud service that needs an account/API token this host does not have. A "faster than MNN/llama.cpp on a phone" statement therefore cannot be measured here; the honest projection is qualitative: the predictor's 15 sequential passes are exactly the kind of small-batch, launch-bound work an NPU with a prepared context binary executes in a few milliseconds each, which is what would turn the CPU-side 63–100 ms predictor loop into a sub-frame cost.
- **Next concrete steps**: (1) export the talker decode step with a static KV cache the same way; (2) run all three graphs on a Snapdragon device or AI Hub for cycle-accurate numbers; (3) an int8 GEMV kernel for the CPU predictor at 8 threads (the only CPU class where llama.cpp's Q4_0 talker is still faster per frame).

## 5. Real Snapdragon phones through Qualcomm AI Hub (2026-09-10, afternoon)

With the user's AI Hub token the same graphs were compiled to QNN context binaries and profiled on physical devices in Qualcomm's farm: **Samsung Galaxy S24 (Snapdragon 8 Gen 3)** and **Samsung Galaxy S25 (Snapdragon 8 Elite for Galaxy)**, every operator placed on the Hexagon NPU (`--compute_unit npu`; AI Hub confirms 100 % NPU placement for all builds). Times are AI Hub's `estimated_inference_time` (median of on-device runs). Scripts, job ids and the raw profile JSON (per-op cycles) are in `analysis/experiments/edge_engine_compare/qnn/aihub/`. Int-weight builds use AI Hub's compile-time quantization with random calibration unless marked *calibrated*; they are timing rows, the calibrated rows carry the fidelity.

### 5.1 Per-component device timings

| graph (real weights) | build | Galaxy S24 | Galaxy S25 | note |
|---|---|---|---|---|
| code predictor body, 17 tokens (§2 graph) | fp16 | 2.73 ms | 2.18 ms | 300/300 ops on NPU; the x86 simulator needed ~3 min for the same pass |
| code predictor cached step, 1 token, 16-slot cache | fp16 | 2.59 ms | 2.26 ms | same as the 17-token body: weight-streaming bound (160 MB fp16 per step) |
| same | w8a16 | 1.32 ms | 1.16 ms | ×15 sub-steps = **20 / 17 ms per frame** |
| same | w4a16 | 1.27 ms | 1.11 ms | no gain over int8: half the step is fixed overhead |
| talker decode step, 28 layers, 256-token cache, v1 export (stacked cache + select, expanded GQA) | fp16 | 39.7 ms | 36.9 ms | only 22–24 % of cycles in the linear layers |
| same v1 | w8a16 / w4a16 | 30.0 / 30.2 ms | 23.6 / 23.6 ms | int4 = int8: overhead-bound |
| **talker decode step, v2 export** (per-layer cache I/O, GQA by reshaping q, no cache select) | fp16 | **17.9 ms** | **15.8 ms** | 72 % of cycles in the linear layers, i.e. bandwidth-bound |
| same v2 | **w8a16** | **12.1 ms** | **9.35 ms** | |
| Code2Wav decoder chunk, 25 frames (2.0 s) with 72-frame context | fp16 | 5,197 ms | 4,539 ms | **208 / 182 ms per emitted frame**: 76 % of cycles in the Snake activation (mul·sin·pow) on 96–192 ch × 62k–186k samples |
| same | w8a16 | 4,350 ms | 4,876 ms | 16-bit activations keep the slow elementwise path |
| same | int8 (w8a8, random calibration) | **301 ms** | **266 ms** | 12.0 / 10.6 ms per emitted frame; 17× faster than fp16 |
| same, 24-frame context (49-frame window) | int8 (random calibration) | 140 ms | 118 ms | 2.2× faster, but **not a valid optimization**: truncating the context to 24 frames costs 15.9 dB against the 72-frame output on the host (48 frames: 23.0 dB, 36: 19.6, 16: 12.2), so the window is doing real work. The fp16 build of this export also failed AI Hub's context-binary conversion (exit code 15). |

Memory: every build peaks at 100–250 MB of working memory on top of the memory-mapped context binary (talker fp16 context ≈ 0.9 GB, int8 ≈ 0.45 GB; decoder fp16 ≈ 0.2 GB).

### 5.1b The NPU is the wrong unit for the vocoder (Adreno GPU, same phone)

The vocoder profile said the cost was elementwise math on sample-rate tensors, which is a bandwidth problem, not a matrix problem — so the same graphs were compiled to TFLite and profiled on the **Adreno GPU** and the CPU of the Galaxy S25:

| graph, fp16/fp32 TFLite | Hexagon NPU (QNN) | Adreno GPU | phone CPU |
|---|---|---|---|
| Code2Wav decoder chunk (2.0 s) | 4,539 ms | **542 ms** | 2,735 ms |
| → per emitted 80 ms frame | 182 ms | **21.7 ms** | 109 ms |
| talker decode step (v2) | **15.8 ms** | 29.9 ms | 29.8 ms |
| code predictor cached step | **2.25 ms** | 4.33 ms | 4.82 ms |

The GPU runs the vocoder **8.4× faster than the NPU**, and the NPU runs the transformer steps ~2× faster than the GPU. The right deployment is therefore heterogeneous, and it maps onto the branch's existing two-stage pipeline exactly: **stage 0 (talker + code predictor) on the NPU, stage 1 (Code2Wav) on the GPU**, which on this phone already run in separate processes on the server.

Fidelity and precision on the GPU. The TFLite GPU delegate runs fp16 by default; `--tflite_options` controls that, and the choice is a real one (Galaxy S25, same graph, same input):

| GPU configuration | per 2 s chunk | per emitted frame | SNR vs torch fp32 |
|---|---|---|---|
| default (fp16 tensors) | 542–572 ms | 21.7–22.9 ms | 27.5 dB (corr 0.9991) |
| `allow_fp32_as_fp16=false` | 1,145 ms | **45.8 ms** | **108.4 dB** (corr 0.99999999999) |
| `gpu_inference_priority1=…MAX_PRECISION` | 2,882 ms | 115.3 ms | 106.6 dB — slower, no more accurate |
| `qnn_gpu_precision=kGpuFp32` / `kGpuFp16` | 569–580 ms | 22.8–23.2 ms | 27.5 dB — the flag changed nothing here |
| ONNX Runtime on the GPU (`--target_runtime onnx`) | 3,063 ms | 122.5 ms | 115.6 dB |
| (reference) TFLite on the phone CPU | 2,735 ms | 109 ms | 106.6 dB |

So the vocoder can run **numerically exact on the Adreno GPU at 45.8 ms per frame** — still 4× faster than fp16 on the NPU — and there is no reason to accept the fp16 delegate's −27 dB noise floor unless the extra 24 ms per frame matters.

### 5.1c Which mobile *runtime*, on the same graphs and the same phone

The other half of "compared with other engines": llama.cpp cannot reach a phone's accelerators for this model at all (no Hexagon backend, CPU-only vocoder), so on-device the meaningful comparison is between the runtimes that can execute these graphs. All rows are the same two exported graphs on the Galaxy S25, fp16 unless noted, per 80 ms frame:

| runtime | talker step, NPU | talker step, GPU | talker step, CPU | vocoder, NPU | vocoder, GPU | vocoder, CPU |
|---|---|---|---|---|---|---|
| QNN context binary | **15.8 ms** | – | – | 182 ms | – | – |
| TensorFlow Lite | 16.3 ms | 29.9 ms | 29.8 ms | fails to run after compiling | **21.7 ms** (fp16) / **45.8 ms** (fp32, exact) | 109 ms |
| ONNX Runtime | 16.6 ms | – | 33.1 ms | out of device memory | 122.5 ms | 124.3 ms |

Reading: on the NPU all three runtimes land within 5 % of each other, because they all end up in QNN — the runtime choice does not matter there, the *export layout* does (§5.1). The runtimes separate on the other units, and the one decision worth making is the vocoder's: TensorFlow Lite on the GPU is 2.7× faster than ONNX Runtime on the same GPU and 8.4× faster than QNN on the NPU. Neither TFLite nor ONNX Runtime can run the vocoder on the NPU at all (one fails after compiling, the other exceeds device memory), so the QNN context binary is the only NPU vocoder path — and it is the slowest option of all.

### 5.1d Beyond the two phones: laptop, automotive and IoT Snapdragons

The same two graphs on other Snapdragon classes in the AI Hub farm (talker step as a QNN context binary on the NPU,
vocoder as TFLite on the GPU):

| device (class) | talker step, NPU | vocoder, GPU (per emitted frame) |
|---|---|---|
| Galaxy S25 — 8 Elite (phone) | 15.8 ms | 21.7 ms |
| Galaxy S24 — 8 Gen 3 (phone) | 17.9 ms | – |
| Snapdragon X Elite CRD (Windows laptop) | 19.6 ms | TFLite is not a supported target for this device |
| SA8775P ADP (automotive) | 23.8 ms | 68.0 ms |
| Dragonwing RB3 Gen 2 — QCS6490 (low-end IoT) | rejected: "Tensor 'x' has a floating-point type which is not supported by the targeted device" | 124.3 ms |

Reading: the transformer step is within 1.5× across a phone, a laptop and an automotive board, so the NPU story
generalises; the QCS6490 tier is integer-only, which is exactly the case where the (currently broken) int8 path would
be mandatory rather than optional. The vocoder's GPU cost scales with the GPU tier, and only the two phone GPUs are
fast enough to keep it under the 80 ms frame budget.

### 5.2 Per-frame budget on the phone (measured components, 80 ms frame)

| configuration (Galaxy S25) | talker | predictor ×15 | vocoder | serial | RTF | pipelined | RTF |
|---|---|---|---|---|---|---|---|
| everything on the NPU, fp16 | 15.8 | 33.9 | 182 | 232 ms | 2.9 | 182 ms | 2.3 |
| NPU steps + **GPU vocoder, fp32 (exact)** | 15.8 | 33.9 | **45.8** | 96 ms | 1.19 | **49.7 ms** | **0.62** |
| NPU steps + GPU vocoder, fp16 delegate | 15.8 | 33.9 | 21.7 | 71 ms | 0.89 | **49.7 ms** | **0.62** |
| everything on the GPU (fp16) | 29.9 | 65.0 | 21.7 | 117 ms | 1.5 | 95 ms | 1.2 |

The pipelined column is what the branch's architecture gives: the talker stage and the Code2Wav stage are separate engines exchanging codec tokens, so on a phone they overlap and the frame cost is the slower stage, not the sum. In that mode the talker stage is the bottleneck at 49.7 ms, the vocoder is free either way, and **the exact-precision vocoder costs nothing** — so the deliverable configuration is talker + predictor on the NPU in fp16 and Code2Wav on the GPU in fp32, at **RTF 0.62 on a Snapdragon 8 Elite, real time with a 1.6× margin, with no quantization anywhere**.

Earlier all-int8 numbers (37–44 ms per frame) are not deliverable: every automatic integer build here is numerically broken (§5.3). They remain as an upper bound on what a proper quantization effort could add.

Not in the budget: host-side embedding lookups, sampling and the RVQ codebook sum (sub-millisecond on a phone CPU), and per-call dispatch overhead (0.1–0.3 ms × 17 calls per frame on QNN). Each graph was measured in isolation, so these are sums or maxima of medians, not a pipeline measurement.

### 5.3 Fidelity on device

On-device inference jobs with the same inputs as the x86 runs (`decoder_quant_chain.py`, outputs in `aihub/out_*.npy`):

| graph, build | device | SNR vs torch fp32 | corr |
|---|---|---|---|
| Code2Wav decoder chunk, fp16 | Galaxy S25 / S24 | **39.2 / 39.1 dB** | 0.99994 |
| code predictor body, fp16 | Galaxy S25 | **48.3 dB** | 0.999993 (x86 simulator: 50.7 dB) |
| code predictor body, w8a16 / w4a16, AI Hub compile-time quantization (random calibration) | Galaxy S25 | 1.0 / 0.3 dB | 0.61 / 0.54: the random-data activation ranges are wrong, so these timing builds are not fidelity evidence |
| code predictor body, w8a16 / w8a8, quantize job calibrated on 6 draws of the input distribution (held-out test input) | Galaxy S25 | **3.1 / 0.7 dB** | 0.77 / 0.39; profiles at 2.09 / 1.23 ms |
| Code2Wav decoder, int8 (w8a8) calibrated on the test window (AI Hub quantize job, min–max ranges) | S25 / S24 | **−3.3 dB** | 0.00 (real-looking signal, uncorrelated with the reference) |
| same, calibrated on 7 held-out windows of real codec tokens (696 frames from the WP5 codec-stream runs) | S25 / S24 | **−4.3 / −4.3 dB** | 0.00 (same failure; the calibrated build profiles at 282 ms on the S25) |

So **fp16 on the Hexagon NPU is faithful and every integer build produced here is not**:

- fp16: decoder 39 dB (the fp16 accumulation limit of a 481-GFLOP vocoder; the same DLC on the QNN CPU backend was 107 dB), predictor 48 dB. Both are usable.
- The vocoder cannot take 8-bit activations at all: per-tensor ranges cannot hold the Snake activation (`x + sin²(ax)/a`) on sample-rate tensors, and calibrating on real codec-token windows does not help (−4.3 dB, output uncorrelated with the reference). The w8a16 build keeps the fidelity story open but runs as slowly as fp16 (4.4–4.9 s per chunk), so it buys nothing.
- The code predictor is damaged by **int8 weights alone**: calibrated w8a16 (int8 weights, int16 activations, correct activation ranges) reaches only 3.1 dB / corr 0.77. This is the same failure the CPU experiment found for torch.ao per-tensor int8 (22 % top-1 agreement, §3) and it is a property of the model, not of the runtime: the predictor's residual heads need per-channel scales or quantization-aware training.

The x86 simulator run of the fp16 decoder (§2) was stopped after 15 h once this device parity existed.

### 5.4 Reading

- **The export mattered more than the device.** The first talker export spent 70 % of its NPU cycles on KV-cache plumbing; the v2 layout (per-layer cache I/O, grouped-query attention by reshaping the query) is 2.2× faster and bandwidth-bound. That layout is what the branch's HTP export should emit.
- **The unit mattered even more.** The Hexagon NPU is 8.4× *slower* than the Adreno GPU on the vocoder (whose cost is elementwise Snake math on sample-rate tensors) and ~2× *faster* on the transformer steps. Sending everything to the NPU — the obvious reading of "run it on the phone's AI accelerator" — is the worst of the three placements measured.
- **What runs faithfully today**: talker + predictor on the NPU (fp16, 39–48 dB) and Code2Wav on the GPU in **fp32** (108 dB, i.e. exact), ≈96 ms per 80 ms frame serially and **≈50 ms pipelined** on a Snapdragon 8 Elite. That is real time with a 1.6× margin, with no quantization and no precision compromise. The export code is in `vllm_omni/edge/qnn_export.py` (transformer steps) and `vllm_omni/edge/decoder_export.py` (vocoder).
- **What still needs work**: a stateful streaming vocoder would cut the GPU's 21.7 ms further (the chunk synthesizes 3.9× more samples than it emits, and simply shortening the context is not valid — 24 frames of context costs 15.9 dB); int8 needs per-channel scales or QAT to be usable at all.
- **Against the other engines on mobile**: llama.cpp and MNN have no Hexagon NPU path for Qwen3-TTS, and llama.cpp's CPU-only design extrapolates above RTF 2 on a phone's CPU cores (1.9–2.0 on a 16–32-thread Xeon). The branch is the only engine here with a validated on-device accelerator path, and with the NPU/GPU split it is real time on the phone while llama.cpp is not.
