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

On native Windows 11 build 26200, the HX370 AMD NPU was detected through VitisAI
EP 1.8.63.0 and ONNX Runtime 1.30.0. The [first-layer NPU report](layer0_npu_probe.json)
records a 244.586 s session build, two actual NPU partition events (one per
step) with 26 CPU node events across the two steps, and 6.904/3.287 ms cold/warm
single-step calls. Relative to the same ONNX CPU graph, hidden-state error was
**1.3449%/1.6589%**, above the provisional 1% component gate, while maximum KV
error was 0.7193%/0.8551%. These are only two steps of one fixture, not a
steady-state latency sample or a speech-quality assessment. The raw node trace
is in `layer0_npu_profile_2026-09-25_11-13-18_772.json`.

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
probe time. Large graph, weights and fixture files remain outside Git; their
hashes are retained in the JSON reports.

Next: isolate which placed operation causes the first-layer numerical drift,
validate a corrected rolling-state artifact over later frames and multiple
utterances, and only then test a warmed full-state NPU+CPU decoder with waveform
quality, complete Omni requests, explicit shared-RAM admission, cancellation,
transfer-inclusive latency and sustained power. The matrix cell remains
**NOT E2E**.
