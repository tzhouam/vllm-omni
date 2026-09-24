# Two-frame X Elite QNN vocoder candidate

The same pinned [Qwen3-TTS two-frame source and fixture](../../s25/qnn_short_chunk/README.md)
compiled to [FP16 QNN DLC](compile_report.json) for Snapdragon X Elite CRD.
An [NPU-requested inference job](inference_submission.json) is pending on
the retained fixture. A separate [placement profile](profile_submission.json)
is also pending for the same target and tensor shape. Compilation alone does
not establish numerical output, NPU placement, timing, or complete TTS
support. The earlier
[25-frame X Elite NPU attempt](../qnn_npu/inference_report.json) returned a
waveform but its placement profile timed out during graph preparation; this
shorter artifact tests a separate memory/graph-preparation boundary.
