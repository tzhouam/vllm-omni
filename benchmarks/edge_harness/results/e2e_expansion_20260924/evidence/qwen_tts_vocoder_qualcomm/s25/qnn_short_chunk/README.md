# Two-frame S25 Code2Wav candidate

The original 25-frame vocoder graph takes 72 prior frames plus 25 new frames
and produced a 48,000-sample waveform. Its FP16 QNN DLC compiled for S25 but
failed inference with a reported device-memory limit. This candidate retains
the same pinned Qwen3-TTS 0.6B CustomVoice decoder and 72-frame history, and
reduces only the new chunk to two frames: `float32[1,512,74]` in and
`float32[1,3840]` out (0.16 s at 24 kHz). The RVQ codebook lookup remains
outside this graph, as in the original HTP export.

The [export report](export_report.json) pins the checkpoint snapshot, weight
hash, local fixture, output and ONNX artifact hashes. The ONNX file remains
outside Git under `/home/zhout/project/edge_infer/models/`. Local ONNX
Runtime CPU matched the eager decoder at **2.96e-6 relative L2**; the eager
two-frame waveform matched the first 3,840 samples of the retained 25-frame
CPU waveform at **2.83e-6 relative L2**. This is one fixed synthetic window,
not a speech-quality test. The [exporter](../../../../../../experiments/export_qwen_tts_vocoder_short_chunk.py)
reproduces the graph and parity gate.

The [local CPU component profile](cpu_profile_report.json) used one warmup and
20 serial samples on the HX370 WSL CPU with ONNX Runtime 1.29 and four threads.
Nearest-rank p50/p95 was **1.365/1.453 s** for 0.16 s of audio; component
median RTF **8.53** is far from playable in this configuration. This CPU
number is not an S25 estimate. The [profiler](../../../../../../experiments/profile_qwen_tts_short_vocoder_cpu.py)
records the raw samples. A hosted S25 artifact must independently pass
numerical, node-placement, memory and complete-stream gates before it can be
called supported.

A local [history-length numerical screen](history_sweep_report.json) on the
same checkpoint and synthetic first window tested 72, 64, 48, 32, 24, 16 and
8 prior frames while keeping two new frames. Reducing history to 64 frames
already changed the 3,840-sample waveform by **7.31% relative L2** versus
the 72-frame baseline; 32 frames changed it by **39.56%**. This one-window
result does not establish a quality threshold, but it rules out treating a
simple history cut as numerically equivalent. The
[probe](../../../../../../experiments/probe_qwen_tts_vocoder_history.py)
records the pinned inputs and every comparison.

The source [upload](upload_submission.json) and [one-sample dataset](dataset_submission.json)
were compiled to an [S25 FP16 QNN DLC](compile_report.json). Its
[NPU-requested same-fixture inference](inference_report.json) **failed** with
Workbench reporting that memory usage exceeded device limits. No device
waveform, node-placement rows or timing samples were returned. Reducing the
new chunk from 25 frames to two did not resolve memory admission for this
FP16 QNN artifact. This failure does not rule out a different graph layout,
precision or runtime route. The [compressed runtime log](logs/jg9zmvqvp_runtime.log.gz)
preserves the failed job trace. A separate S25 TFLite GPU candidate is tracked
under [tflite_short_chunk](../tflite_short_chunk/).
