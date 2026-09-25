# Qwen3-TTS rolling-KV transformer boundary on HX370

This is a **component experiment**, not a complete text-to-audio request or a
qualified AMD NPU backend. The checkpoint is Qwen3-TTS 0.6B CustomVoice revision
`85e237c12c027371202489a0ec509ded67b5e4b5`. The fixture uses real codes
from one generated 117-frame utterance. It tests consecutive two-frame decoder
steps beginning at frames 95 and 97, with an explicit 71-frame sliding KV state
for each of the source decoder's eight pre-transformer layers. The exported
state alone is 4,653,056 bytes per step; downstream convolution state, transfer,
workspace, weights, and admission have not been measured for this route.

The [export report](report.json) records the checkpoint, codes, ONNX and fixture
hashes. The wrapper calls the checkpoint's own `pre_transformer`; it does not
substitute trained weights. Its PyTorch hidden states and all rolling KV tensors
were bitwise identical to the source on both steps. ONNX Runtime CPU relative
L2 error against source was 5.96e-7 and 6.73e-7 for hidden state, with maximum
KV error 1.34e-7. The extracted first-layer graph also passed CPU parity.

On native Windows 11 build 26200, the HX370 AMD NPU driver was 32.0.203.329
and the device was detected through VitisAI EP 1.8.63.0 and ONNX Runtime 1.30.0.
The [environment record](environment.json) pins the EP DLL hash. The
[first-layer NPU report](layer0_npu_probe.json)
records a 244.586 s session build, two actual NPU partition events (one per
step) with 26 CPU node events across the two steps, and 6.904/3.287 ms cold/warm
single-step calls. Relative to the same ONNX CPU graph, hidden-state error was
**1.3449%/1.6589%**, above the provisional 1% component gate, while maximum KV
error was 0.7193%/0.8551%. These are only two steps of one fixture, not a
steady-state latency sample or a speech-quality assessment. The raw node trace
is in `layer0_npu_profile_2026-09-25_11-13-18_772.json`.

A repeated native capture saved the [actual NPU and CPU layer outputs](layer0_outputs_2step.npz).
The [unchanged CPU-suffix replay](layer0_cpu_suffix_replay.json) injected each
NPU hidden state into the checkpoint's remaining seven transformer layers and
carried forward the NPU layer-0 KV while preserving each later layer's CPU KV.
Its CPU control matched source decode to 1.44e-6/7.67e-7 waveform relative L2.
The NPU-layer-0 plus CPU-suffix chunks differed 0.8225%/0.9621% from the
source waveform, within the provisional 1% waveform gate on **these two
chunks only**. The first-layer hidden-state numerical gate still fails, and
this is an offline handoff replay rather than a live Omni stream. The repeated
native run had one NPU partition per step after a 254.102 s session build;
its [report](layer0_capture_probe.json) retains the artifact and capture hashes.

To test accumulation, an [extended fixture](step95_rollout11.npz) preserved the
same initial cache and exactly matched the first two convolution inputs before
adding nine more from the same 117-frame generated code stream. Its
[construction report](rollout_fixture.json) pins the fixture hash. The native
[eleven-step NPU report](layer0_rollout11_probe.json) records eleven actual
VitisAI node events, one per step; 143 additional node events ran on CPU. The
session build took 248.493 s. These isolated calls took 2.635–7.501 ms each,
without whole-request transfer or warmed repeated-request measurement. By
frame 115, first-layer hidden error reached 2.504% and maximum KV error reached
2.182% against the same ONNX CPU graph. The [captured outputs](layer0_outputs_11step.npz)
were replayed through the real seven-layer CPU suffix and waveform decoder.
The [replay report](layer0_cpu_suffix_rollout11.json) has all eleven chunk
errors: only **3/11** pass the provisional 1% waveform gate; the worst is
**2.197%** at frame 99. The CPU control's largest waveform error against source
decode is 3.39e-5 relative L2. This supersedes any inference that the two
early passing chunks establish a continuous NPU-assisted stream. The result
constrains this exact FP32 VitisAI first-layer export and one generated
utterance; it does not rule out a corrected artifact or another backend.

An [eleven-step CPU-KV reset diagnostic](layer0_cpu_state11_probe.json) ran the
same NPU graph with the CPU reference KV supplied before every new step. It
still placed eleven VitisAI partitions after a 246.555 s session build. The
maximum per-step new-KV error fell to 0.8953% (versus 2.182% with NPU-owned
rolling KV), but hidden-state error still reached 2.502%. In the
[matching CPU-suffix waveform replay](layer0_cpu_state11_suffix_replay.json),
the [captured outputs](layer0_outputs_cpu_state11.npz) again passed only 3/11
chunks, with a 2.265% worst waveform error at frame 99. Supplying exact CPU KV
therefore does not fix this artifact's waveform failures; the per-step placed
computation itself needs numerical work. This diagnostic requires a CPU cache
producer and is not a proposed deployment path or performance result.

A [checkpoint-output graph](checkpoint_graph_cpu.json) exposed eleven internal
tensors without changing the original ONNX CPU outputs. The native
[checkpoint probe](layer0_checkpoints_probe.json) retained one VitisAI partition
per step and nearly the original final hidden error (1.357%/1.675% versus
1.345%/1.659%). Its first exposed input-projection MatMul was already
1.144%/1.569% from ONNX CPU; attention softmax reached 2.475%/2.241%.
Those outputs are from this diagnostic graph, whose placement counts matched
the uninstrumented graph; they localize an early numerical divergence but do
not prove that only the input projection causes the final error. The native
checkpoint session took 456.397 s to build and the [captured tensors](layer0_checkpoints_cpu_state.npz)
are retained.

An exact [CPU input-projection split](input_projection_split_cpu.json) computes
the real layer's initial linear projection on CPU and passes its `[1,2,512]`
output to an ONNX suffix. CPU prefix+suffix matched the original layer across
all eleven real-code steps. In the first [native NPU suffix probe](input_projection_split2_npu.json),
one VitisAI partition ran per step; hidden error fell to 0.5744%/0.4873% and
maximum new-KV error to 0.7135%/0.4781%. The [offline waveform replay](input_projection_split2_waveform.json)
with CPU layer-0 KV passed both chunks at 0.5798%/0.3600% relative L2.
This is a two-step component candidate only. The suffix session build took
438.091 s, and its one cold/one warm call took 7.427/4.006 ms versus
2.261/0.649 ms for the corresponding ONNX CPU suffix calls, before CPU
projection and transfer. No speedup or continuous stream is established.

The [eleven-step NPU-owned-KV split probe](input_projection_split11_npu.json)
then placed eleven VitisAI partitions after a 289.178 s session build. All
eleven hidden states were within 1% of the matching CPU suffix, but maximum
KV error grew to 2.151% by frame 115. Replaying its [captured outputs](input_projection_split11_outputs.npz)
through the unchanged CPU suffix yielded [10/11 chunks](input_projection_split11_waveform.json)
within 1% waveform relative L2; frame 109 missed at **2.398%** (32.4 dB SNR,
CPU reference RMS 0.0164). The joined 22-frame segment was 0.591% relative L2
and 44.57 dB SNR. That aggregate does not erase the chunk failure. Eleven
individual NPU calls took 2.47–6.79 ms before CPU projection and transfer;
these are not complete-request latency samples.

The [same split with CPU reference KV supplied to each NPU step](input_projection_split11_cpu_state_npu.json)
placed eleven VitisAI partitions after a 272.605 s session build. Every
per-step hidden and new-KV output then met the provisional 1% tensor gate,
but the [offline waveform replay](input_projection_split11_cpu_state_waveform.json)
still passed only 10/11 chunks; frame 109 missed at **2.424%**. Its joined
22-frame segment was 0.603% relative L2 and 44.40 dB SNR. The retained
[CPU-state capture](input_projection_split11_cpu_state_outputs.npz) makes the
diagnostic auditable. Resetting KV therefore does not fix the remaining
chunk error, and computing the reference KV on CPU each step would duplicate
work. This split is **not a qualified live TTS backend**.

The full eight-layer ONNX graph passed the CPU check and exposed an NPU device,
but [the bounded full-graph attempt](full_graph_compile_cap.json) was manually
stopped after roughly ten minutes of session creation without an inference or
placement event. This is an inconclusive compile cap for that exact FP32 graph,
not proof that the model cannot run on the NPU. The partial [run report](npu_probe.json)
and empty profile file are retained so that no execution is inferred from the
attempt.

Reproduce with `probe_qwen_tts_stateful_transformer_export.py` using the pinned
checkpoint and the [generated-code source report](../../../e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/s25/real_code_window/report.json)
and its adjacent `codes.npz`; then run `probe_qwen_tts_stateful_npu.py --layer0`
against the exported first layer and fixture on the native Windows HX370 with
the packaged VitisAI EP.
The scripts require explicit SHA-256 values for the ONNX and fixture at NPU
probe time. For the longer run, `extend_qwen_tts_stateful_fixture.py` derives
the eleven-step fixture, `probe_qwen_tts_stateful_npu.py --layer0 --steps 11`
captures the native outputs, and `replay_qwen_tts_layer0_npu_suffix.py` measures
their effect on the CPU transformer suffix and waveform.
`probe_qwen_tts_stateful_npu.py --npu-state-source cpu` and
`replay_qwen_tts_layer0_npu_suffix.py --injected-cache-source cpu` reproduce
the CPU-KV reset diagnostic. The initial two-step
ONNX graph, source checkpoint and fixture remain outside Git; their hashes are
retained in the JSON reports. The captured NPU rollouts and the extended
fixture are retained here.
`prepare_qwen_tts_layer0_checkpoints.py` builds the output-instrumented graph;
`prepare_qwen_tts_input_projection_split.py` extracts the CPU prefix and NPU
suffix and verifies eleven-step CPU parity. The split's
[eleven-step fixture](step95_split11.npz) is retained here; its graph files are
reproducible from the pinned original ONNX hash and are kept outside Git.
Set `PYTHONPATH` to this fork's checkout for the Linux export and replay:
the installed `vllm_omni` wheel on the test host lacked the exact-state decoder
method, and running the replay without that override raised `AttributeError`
before any measurement. The recorded replay used the checkout source and
PyTorch 2.13.0+cpu; the native VitisAI probe used ONNX Runtime 1.30.0.

Next: localize the remaining frame-109 sensitivity within attention/MLP or the
waveform suffix, validate a corrected artifact over more utterances, and only
then test a warmed full-state NPU+CPU decoder with waveform quality, complete
Omni requests, explicit shared-RAM admission, cancellation, transfer-inclusive
latency and sustained power. The matrix cell remains
**NOT E2E**.
