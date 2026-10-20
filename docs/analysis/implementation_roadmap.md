# Implementation roadmap

Phased plan for the designs in [tts_edge_design.md](tts_edge_design.md) and [vla_edge_design.md](vla_edge_design.md), following the strategy decision in [vllm_omni_edge_feasibility.md](vllm_omni_edge_feasibility.md) (S4 hybrid first for TTS, S3 lightweight orchestration as the main path, S5 edge-native numbers at every phase, S1 on Jetson as the reference tier). Each phase has entry conditions, deliverables, acceptance criteria, and the reproducible benchmark it must publish. Nothing below has been executed except the Phase 0 items marked "done" in [experiments.md](experiments.md).

## Phase 0 — Reference numbers and toolchain gates (1–2 weeks)

Entry: this analysis. No device hardware required for the first half.

| Item | Status | Acceptance |
|---|---|---|
| Server reference for Qwen3-TTS 1.7B streaming (TTFA, cadence, RTF) | **done** on L20X (p50 TTFA 27.7 ms, RTF ≈ 0.08–0.10) | Numbers and JSON archived; re-runnable with the recorded command |
| CPU edge-native baseline with llama.cpp (build, convert, quantize, bench, VLM run) | **done** on x86 (Q4_0 decode 135.6 tok/s @ 4 threads; SigLIP tile 0.6–1.2 s @ 8 threads) | Same |
| Qwen3-TTS 0.6B server reference | pending (needs a driver that overrides the example's hard-coded 1.7B ids; no source change) | TTFA/RTF for 0.6B on the same prompts |
| llama.cpp `llama-tts` Qwen3-TTS on x86 CPU (needs the GGUF conversion of the cached HF checkpoint; check whether `conversion/` supports it at this commit) | pending | WAV produced; RTF at 4/8 threads; codebook-0 agreement with the server under greedy sampling |
| MNN build with `MNN_BUILD_LLM` + `llmexport.py` of Qwen3-TTS 0.6B + `qwen3_tts_demo` on x86 CPU | pending (requires its Python export dependencies; permission to install) | Same metrics as above |
| ExecuTorch install + export of the Qwen3-TTS decoder (`modeling_qwen3_tts_tokenizer_v2.py`) to `.pte` on XNNPACK | pending (dependency installation; submodules) | Decoder PCM SNR vs torch ≥ 40 dB on recorded codes [Proposal threshold] |
| VLA reference: π0 or GR00T on the L20X through the OpenPI endpoint with recorded observations | pending (weights not cached; download needs permission) | Action chunks for ≥ 100 recorded observations saved as the parity oracle |
| Device procurement / access: one ARM Linux board, one Android phone with Qualcomm NPU, one Apple Silicon device, one Jetson | pending | Devices enumerated with SoC, RAM, OS, and thermal envelope |
| **Checkpoint/task parity gate**: pin one Qwen3-TTS checkpoint and task per engine (server benchmark used CustomVoice 1.7B; llama.cpp README shows 1.7B Base; MNN demo needs reference audio and hard-codes `hiddenSize = 1024`) | pending | A table stating, per engine, which size/task variant is confirmed to run, with the exact converted artifact; comparisons in later phases use only matching rows |
| **Playback-gap measurement**: rerun the server driver with `codec_chunk_ramp` / `codec_chunk_adaptive` enabled and report uninterrupted-playback latency alongside TTFA | pending (cheap, same GPU procedure) | Both metrics in the JSON schema; the default 1→25 schedule's ≈80 ms gap quantified |

Exit criterion: for each runtime kept in scope, a cached-model artifact and a one-command benchmark script exist and run on x86, so device runs in Phase 1 are only a change of host.

## Phase 1 — TTS hybrid (S4): server talker, on-device decoder (2–4 weeks)

Deliverables:

1. vLLM-Omni side: a codec-token streaming mode. This is a new wire contract, not only an adapter: `serving_speech_stream.py`'s `_generate_and_send` consumes PCM from the full pipeline today, so the deliverable is a talker-only deploy profile plus a stream message type carrying `codes.audio` frames (16 × int16 per frame) with defined frame ordering, EOS, cancel, and decoder-state/revision semantics (integration points in [vllm_omni_edge_feasibility.md](vllm_omni_edge_feasibility.md) §5). This is the only proposed change inside vLLM-Omni and must go through its review process (`review-pr`/`precheck-pr` conventions in the repo). **Gate**: the hybrid-first ordering holds only if this prototype and the on-device decoder measurement (item 2) both pass; otherwise Phase 2 proceeds first.
2. Device side: `IDecoderAdapter` implementations on llama.cpp (from the mtmd code2wav graph) and ExecuTorch (`.pte` from Phase 0), driven by the same chunk schedule as the server reference (ramp/adaptive from the deploy YAML, 72-frame left context; 1→25 kept as the measured baseline) and `reset()` on abort.
3. A small client that plays audio from the codec stream with a 2 s ring buffer and supports interruption.

Acceptance:

- TTFA (client-observed, including network) ≤ server TTFA under the same chunk schedule + RTT + first-chunk decode time; **uninterrupted-playback latency** (earliest start with no later underrun, using the ramped chunk schedule) reported alongside; inter-chunk gap p95 ≤ 100 ms on the device decoder for 25-frame chunks (else reduce steady chunk size).
- Decoder parity: PCM SNR ≥ 40 dB against the server decoder on identical codes; ASR WER on the 12 benchmark prompts within 1 point of server audio.
- Interruption: audio stops within one chunk length (≤ 2 s worst case, target ≤ 200 ms with a 2-frame ring) and the next utterance starts cleanly (no residual decoder state, verified by bit-identical output to a cold decode).

Benchmark: `P-TTS-1` and `P-TTS-2` from [experiments.md](experiments.md) §5, run on device and on x86, published as JSON in the same schema as `bench_results.json`.

## Phase 2 — TTS fully on device (S3) on llama.cpp and MNN (4–6 weeks)

Deliverables:

1. Edge TTS orchestrator (C++) implementing the loop in [tts_edge_design.md](tts_edge_design.md) §2.2 over `ITalkerAdapter`/`IDecoderAdapter`/`ISpeakerAdapter`.
2. llama.cpp binding: talker + predictor via the existing Qwen3-TTS mtmd graphs, chunked decode, abort check per step; Android JNI and Apple xcframework builds.
3. MNN binding: chunked `speech_decoder` calls with carried state (pattern from the Qwen2.5-Omni streaming path), talker at W4; Android app integration (`MnnLlmChat` has the host UI), QNN delegation of the decoder as an experiment.
4. Parity harness against vLLM-Omni offline outputs (greedy) for talker codes and decoder PCM.

Acceptance:

- 0.6B path: RTF ≤ 0.5 on the ARM board at 4 threads and on the phone's big cores; TTFA ≤ 300 ms; resident memory ≤ 2 GB; 10-minute continuous synthesis without thermal throttling below the RTF gate (log SoC temperature).
- 1.7B path: RTF ≤ 1.0 on Apple Silicon Metal and on the phone GPU/NPU; report, do not gate, on CPU.
- Codebook-0 greedy agreement ≥ 95 % on the first 25 frames of each prompt (differences after the first divergence are expected; compare prefix only) [Proposal metric].
- Interruption and backpressure behave as specified (tests with abort at frame 1, mid-chunk, and after EOS).

Benchmark: `P-TTS-2` plus the interruption test suite; results per device in a table with the same columns as the server run.

## Phase 3 — ExecuTorch TTS path for CoreML/QNN and Jetson reference (4 weeks, parallel with Phase 2)

Deliverables: talker export via `export_llm` (Qwen3 config, 8da4w), predictor and decoder as separate programs, a runner modelled on the Voxtral TTS runner; CoreML and QNN partitioner recipes; S1 reference run of vLLM-Omni on a Jetson (install, cold-start time, TTFA/RTF) for the same prompts.

Acceptance: same gates as Phase 2 on the ExecuTorch path; the Jetson S1 run documents cold start and memory (expected to fail the cold-start gate; recorded as the reference tier, not as a product path).

## Phase 4 — VLA parity and simulation (6–8 weeks)

Deliverables:

1. Exports of π0 (plain torch, CPU parity test exists in the repo) and GR00T N1.7 (HF wrapper) into ExecuTorch programs: `vision_encoder`, `fusion`, `action_head`; MNN exports of the same via ONNX for Android NPU experiments.
2. `IVlaRunner` implementations and the controller/safety split from [vla_edge_design.md](vla_edge_design.md) §2.1, with the chunk buffer, staleness check, and watchdog.
3. Parity suite: recorded observations → action chunks compared with the OpenPI server (max-abs, per-dimension RMS) at K = 4 (GR00T) and K = 10 (π0).
4. Simulation closed loop (external simulator, LIBERO-style tasks as available for the checkpoints) with the edge runner in the loop.

Acceptance:

- Parity: with the flow noise passed explicitly (π0 `sample_actions(noise=...)`) so both sides are deterministic, per-dimension RMS error ≤ 2 % of the action range at fp16 weights, ≤ 5 % at 8da4w [Proposal thresholds, to be revisited after the first measurements].
- Latency on device: inference wall time p50 ≤ 250 ms and p95 ≤ 400 ms for one 224² camera on the phone NPU or Apple ANE; report the three-camera case; chunk age never exceeds the staleness bound in a 10-minute run [Proposal].
- Simulation success rate within 5 points of the server policy over ≥ 50 episodes per task.
- Fault injection tests pass (delayed frames, dropped camera, CPU frequency cap) with the safety layer engaging as specified.

Benchmark: `P-VLA-1` from [experiments.md](experiments.md) §5 for perception cost, plus the parity and simulation suites; publish per-stage ms, jitter, memory, temperature.

## Phase 5 — Hardware-in-the-loop and hybrid re-planning (open-ended)

Only after Phase 4 passes: hardware-in-the-loop with an independent safety layer; optional cloud re-planning via vLLM-Omni for task text updates; long-duration soak (thermal, memory growth) on every target.

## Cross-cutting

- **Repository conventions**: any change proposed to vLLM-Omni or vLLM follows their `AGENTS.md` (uv environments, human-owned PRs, disclosure of AI assistance); llama.cpp explicitly forbids autonomous agent contributions, so llama.cpp work stays in a private fork or is authored by a human maintainer of this project.
- **Reproducibility**: every benchmark is a script plus a JSON output plus the exact commit of each engine; device runs record SoC, OS, thermal state, and governor settings.
- **Permission gates** (per the task's rules): dependency installs (ExecuTorch, MNN export deps), model downloads (π0/GR00T/DreamZero weights), and any source change in the six checkouts require explicit approval before they are executed.
