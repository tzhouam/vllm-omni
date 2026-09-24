# Two-frame X Elite QNN vocoder candidate

The same pinned [Qwen3-TTS two-frame source and fixture](../../s25/qnn_short_chunk/README.md)
compiled to [FP16 QNN DLC](compile_report.json) for Snapdragon X Elite CRD.
[NPU-requested inference](inference_report.json) succeeded on the retained
fixture. The [audit](audit_report.json) measures **0.930% relative L2** and
**40.63 dB SNR** against local ONNX Runtime CPU, with no saturated samples.
This is one synthetic, two-frame waveform without a listening-quality gate.
A separate [placement profile](profile_report.json) returned 100 component
samples at nearest-rank p50/p95 **3.662/3.676 s** for **0.16 s** audio,
with all 728 execution-detail rows on NPU and 15,855,616 bytes reported
inference peak. Its median component RTF is **22.89**. The profile used
Workbench-generated same-shape input, while the waveform inference used the
retained fixture; the [audit](audit_report.json) binds both jobs to the exact
target. This measured latency is a playable-stream blocker for this artifact.
The earlier
[25-frame X Elite NPU attempt](../qnn_npu/inference_report.json) returned a
waveform but its placement profile timed out during graph preparation; this
shorter artifact tests a separate memory/graph-preparation boundary. Neither
artifact establishes complete TTS support.
