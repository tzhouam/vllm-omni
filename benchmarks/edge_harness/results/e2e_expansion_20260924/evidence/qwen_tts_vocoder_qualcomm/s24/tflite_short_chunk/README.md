# Two-frame Code2Wav on hosted Galaxy S24

The pinned [two-frame source and fixture](../../s25/qnn_short_chunk/README.md)
retain 72 history frames and two new frames from Qwen3-TTS 0.6B CustomVoice.
The source [compiled to S24 TFLite](compile_report.json). Both routes below
use that exact compiled target and fixture, with explicit compute-unit requests.

The [CPU-requested inference](../tflite_short_chunk_cpu/inference_report.json)
returned finite unsaturated 3,840-sample audio. Its [audit](../tflite_short_chunk_cpu/audit_report.json)
measured **5.75e-6 relative L2** / **104.81 dB SNR** versus local ONNX
Runtime CPU on the same input. The [GPU-requested inference](inference_report.json)
returned a **fully saturated waveform**: all samples had absolute amplitude
at least 0.999, with relative L2 **24.218** / SNR **−27.68 dB** versus local
ONNX CPU ([audit](audit_report.json)). The tested S24 GPU delegate route
fails the numerical gate; requesting GPU alone does not establish its actual
node placement.

The [CPU placement profile](../tflite_short_chunk_cpu/profile_report.json)
returned **100** component samples at nearest-rank p50/p95
**3.433/3.873 s**, with all **738 execution-detail rows on CPU** and
reported 770,936,832-byte inference peak. This is **0.16 s audio** per call:
median component RTF **21.46**, far from playable streaming. The
[GPU-requested placement profile](profile_report.json) returned **100** samples
at p50/p95 **0.601/0.895 s**, with **403 CPU and 335 GPU** rows and reported
608,776,192-byte peak. Those timings describe a **grossly failed waveform**,
not usable audio. Workbench generated both profile inputs with the pinned
shape; the numerical inferences used the retained fixture.

CPU numerical parity on one synthetic window is component evidence only.
Neither route verifies a talker/predictor, resident history, stage handoff,
complete stream, listening quality or thermal behavior.
A separate [GPU FP32-preserving control](../tflite_short_chunk_gpu_fp32/inference_report.json)
requested `allow_fp32_as_fp16=false` against the same target and fixture.
It returned a waveform **bitwise identical** to the fully saturated default
GPU output ([audit](../tflite_short_chunk_gpu_fp32/audit_report.json)). This
option did not repair the tested S24 delegate route; it does not identify the
specific failing operator or rule out other artifacts/runtimes.
