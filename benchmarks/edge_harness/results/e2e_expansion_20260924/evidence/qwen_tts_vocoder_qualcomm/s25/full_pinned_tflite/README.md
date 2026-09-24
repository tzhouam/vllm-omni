# Revision-pinned 25-frame S25 vocoder candidate

The complete Qwen3-TTS 0.6B CustomVoice snapshot at revision
`85e237c12c027371202489a0ec509ded67b5e4b5` was exported to a fixed
72-history-plus-25-new-frame Code2Wav ONNX graph. The graph weighs roughly
400 MB and stays outside Git. The [export report](export_report.json) pins
its SHA256, decoder weight, full 97-frame fixture and 48,000-sample output.
Local ONNX Runtime CPU matched the eager decoder at **1.99e-6 relative L2**;
the pinned eager output matched the historical 25-frame ONNX CPU waveform at
**1.97e-6 relative L2** on that same synthetic fixture. Neither comparison
is a listening-quality test. The [exporter](../../../../../../experiments/export_qwen_tts_vocoder_short_chunk.py)
reproduces this graph with `--chunk-frames 25`.

The [source upload](upload_submission.json) and [one-sample input dataset](dataset_submission.json)
were submitted to Qualcomm AI Hub Workbench. The [S25 TFLite compile](compile_report.json)
and [GPU-requested inference](inference_report.json) succeeded on that exact
target. The [waveform audit](audit_report.json) measured **4.239% relative L2**
and **27.46 dB SNR** versus local ONNX Runtime CPU, with no saturated samples.
The new device waveform is [bitwise identical](historical_device_parity_report.json)
to the [historical 25-frame target's](../full_tflite_gpu_profile/README.md)
GPU output on this retained fixture. This supports numerical continuity on
one input, not identical historical export provenance or listening quality.
A separate [placement profile](profile_submission.json) is in progress; GPU
node placement and latency for this revision-pinned target remain unverified.
Complete-stream handoff and sustained behavior are also unverified.
