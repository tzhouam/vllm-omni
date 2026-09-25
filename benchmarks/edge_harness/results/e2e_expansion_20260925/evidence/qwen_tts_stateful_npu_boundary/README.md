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
their effect on the CPU transformer suffix and waveform. The initial two-step
ONNX graph, source checkpoint and fixture remain outside Git; their hashes are
retained in the JSON reports. Both captured NPU rollouts and the extended
fixture are retained here.
Set `PYTHONPATH` to this fork's checkout for the Linux export and replay:
the installed `vllm_omni` wheel on the test host lacked the exact-state decoder
method, and running the replay without that override raised `AttributeError`
before any measurement. The recorded replay used the checkout source and
PyTorch 2.13.0+cpu; the native VitisAI probe used ONNX Runtime 1.30.0.

Next: isolate which placed operation causes the first-layer numerical drift,
validate a corrected rolling-state artifact over later frames and multiple
utterances, and only then test a warmed full-state NPU+CPU decoder with waveform
quality, complete Omni requests, explicit shared-RAM admission, cancellation,
transfer-inclusive latency and sustained power. The matrix cell remains
**NOT E2E**.
