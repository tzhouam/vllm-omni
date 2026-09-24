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
succeeded; [GPU-requested inference](inference_submission.json) and a separate
[placement profile](profile_submission.json) are pending on that target.
Device numerical waveform quality, node placement,
complete-stream handoff and sustained behavior remain unverified for this
revision-pinned artifact. The [historical 25-frame target](../full_tflite_gpu_profile/README.md)
is separate despite the local same-fixture numerical continuity.
