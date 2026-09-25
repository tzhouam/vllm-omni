# Independent Qwen3-TTS AMD NPU MLP rollout

This is a second **component and offline waveform replay**, not a live Omni
text-to-audio request. The checkpoint is Qwen3-TTS 0.6B CustomVoice revision
`85e237c12c027371202489a0ec509ded67b5e4b5`, with real weights. A different
seed-91 Ryan prompt about a blue kite generated 144 frames of 16 speech-code
indices on HX370 WSL CPU. The [source report](report.json) pins the prompt,
checkpoint and source hashes. Its 25-frame local ONNX waveform matched the
checkpoint's FP32 eager window at 9.86e-7 relative L2, and that window matched
the corresponding full-decoder segment at 6.34e-7.

The [fixture report](fixture_report.json) pins the unchanged CPU input
projection, attention and MLP ONNX graph hashes. At frames 95–115, the extracted
CPU layer and actual checkpoint layer agreed within 6.29e-7 relative L2 for
hidden and rolling KV. The [native NPU report](npu_report.json) pins the
VitisAI EP DLL, MLP graph and independent fixture. On native Windows 11 build
26200 with HX370 driver 32.0.203.329, ORT 1.30.0 and VitisAI EP 1.8.63.0,
the graph placed 12 VitisAI nodes (one warmup plus eleven measured calls) and
zero CPU nodes. The worst MLP hidden error versus matching ONNX CPU was
0.6739% relative L2. The [gzip-compressed raw profile](mlp_npu_profile_2026-09-25_14-56-36_033.json.gz)
and [captured outputs](npu_capture.npz) retain placement and tensors.

The [offline waveform replay](waveform_replay.json) injected those measured
NPU MLP outputs while keeping attention and KV on CPU, then ran the unchanged
seven-layer suffix and waveform decoder. All **11/11** two-frame chunks passed
the provisional 1% relative-L2 waveform gate; the worst was 0.7434% at frame
107. The joined 22-frame segment was 0.4871% relative L2. This extends the
earlier one-utterance 11/11 finding to a second generated utterance, but does
not establish listening quality, a resident stream or complete-request
behavior.

Same-process isolated-call nearest-rank p50/p95 was 0.045/0.048 ms for FP32
CPU MLP and 0.548/0.842 ms for NPU MLP, after one warmup each. NPU session
creation took 45.58 s; transfer, full decoder latency and power were not
measured. The NPU call remains about 12 times slower at p50 than CPU before
handoff. This narrow split fails the architecture's benefit gate and remains
**unintegrated**; the matrix cell stays **NOT E2E**. A coarser numerical and
speed candidate is needed before a live stage is justified.

Reproduce the generated codes with `probe_qwen_tts_real_code_window.py
--seed 91 --min-generated-frames 117` and the prompt in `report.json`. Run
`prepare_qwen_tts_independent_mlp_fixture.py` with the pinned source graphs and
hashes from `fixture_report.json`, then `probe_qwen_tts_mlp_npu.py` on native
Windows with the pinned MLP and fixture SHA-256 values. Finally run
`replay_qwen_tts_layer0_npu_suffix.py --injected-cache-source cpu` with the
retained code, replay-fixture and capture hashes. Full invocation arguments
are defined by those scripts; model and ONNX weights remain outside Git.
