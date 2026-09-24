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
A separate [placement profile](profile_report.json) returned 100 component
samples at nearest-rank p50/p95 **0.671/1.454 s** for 2 s audio, with 403 CPU
and 335 GPU execution-detail rows and 380,100,608 bytes reported peak memory.
The median component RTF is **0.335**. Workbench generated profile input of
the same shape, while the numerical inference used the retained fixture; the
[audit](audit_report.json) binds both jobs to the exact target. Complete-stream
handoff, listening quality and sustained behavior remain unverified.

A separate [real generated-code window](../real_code_window/README.md) passed
local ONNX/eager parity at 1.02e-6 relative L2 and matched the corresponding
full FP32 decode segment at 7.26e-8 relative L2. The same fixture has a
successful inference on this pinned S25 target: the unsaturated device
waveform differed from local ONNX CPU by 1.153% relative L2 / 38.76 dB SNR.
This one real-code window is encouraging component evidence, not a speech
quality tolerance or complete mobile stream.
