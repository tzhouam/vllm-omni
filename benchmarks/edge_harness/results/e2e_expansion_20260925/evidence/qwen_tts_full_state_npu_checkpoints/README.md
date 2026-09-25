# Qwen3-TTS eight-layer hidden checkpoint diagnosis (HX370 AMD NPU)

This 2026-09-25/26 native Windows experiment exposes the eight layer-hidden
outputs of the same real-weight, rolling-K/V Code2Wav pre-transformer graph
used in the [uninstrumented two-step run](../qwen_tts_full_state_npu_extended/README.md).
It diagnoses component numerics only. It does not execute the text frontend,
predictor or complete text-to-audio request.

The [preparation report](preparation.json) pins the original ONNX graph
SHA-256 `2a74a49e99ffeddd4918cade906aea0a497bc9651279475eaaa583cd386b09fd`,
the source fixture SHA-256
`e132f412ee3b97a93b068548c2e52c5530a9c8226fd4d7ae268730d04e6f19dc`
and the output-instrumented graph SHA-256
`a707f33482fc87b3d6edf9f4b721d53a6078305a1ecd12b63cc0c5e91f37ba2d`.
The diagnostic graph and fixture are large local model artifacts outside Git.
The [preparation script](../../../../experiments/prepare_qwen_tts_full_state_checkpoints.py)
appends eight `[1,2,512]` FP32 layer-hidden outputs, checks the ONNX graph,
and confirms that all 17 original CPU graph outputs are unchanged on the
retained first-step fixture.

The [native probe](probe.json) used Windows 11 build 26200, ONNX Runtime
1.30.0 and VitisAI EP 1.8.63.0 on the HX370 NPU. Session creation took
**1,191.156 s**. The [raw ORT trace](profile_2026-09-25_23-43-28_461.json)
contains **two VitisAI and 26 CPU node events** across two consecutive
two-frame steps. The graph owned its next-step K/V. The two NPU calls took
22.096/17.107 ms, versus 6.601/5.132 ms for the same diagnostic graph on
CPU in that process. This is one cold and one later component call, with no
whole-request, warmed latency or power measurement.

The [checkpoint audit](audit.json) pins the native report, original report,
both [diagnostic captured tensors](full_state_checkpoints.npz) and original
captures by SHA-256. The diagnostic graph's first 17 CPU outputs and first
17 NPU outputs match their corresponding original-graph captures **exactly**
on both steps, and provider event counts are unchanged. That supports using
its extra outputs to localize the numerical divergence in this fixed case.

| Frame start | First hidden layer over 1% | Layer 0 hidden relative L2 | Final hidden relative L2 |
| --- | ---: | ---: | ---: |
| 95 | 0 | 1.3449% | 1.8908% |
| 97 | 0 | 1.6589% | 2.0738% |

All later layer-hidden outputs also exceed 1% on both steps. The separate
[all-layer K/V audit](../qwen_tts_full_state_npu_extended/all_layer_kv_audit.json)
first crosses 1% at layer 2 K on frame 95 and layer 1 K on frame 97. The
layer-0 hidden divergence therefore appears before any K/V output crosses
the provisional 1% gate. These are observed output boundaries, not proof
of the exact failing operation. The prior first-layer checkpoint run found
an input-projection error, but it is a different extracted graph and cannot
by itself attribute an operation inside this full graph.

The uninstrumented graph's downstream CPU vocoder replay remains **0/2**
waveform chunks within 1%; adding observations does not repair those errors.
No NPU-backed complete TTS request or performance benefit is established.
The HX370 AMD NPU Qwen3-TTS matrix cell remains **NOT E2E**. The next
numerical experiment should isolate the first full-graph layer's input
projection and compare an exact CPU projection boundary against the unchanged
eight-layer NPU rollout on both generated utterances before a live Omni stage.
