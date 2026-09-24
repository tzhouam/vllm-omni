# S25 pinned vocoder on generated speech codes

The [reproducer](../../../../../../experiments/probe_qwen_tts_real_code_window.py)
ran the pinned Qwen3-TTS 0.6B CustomVoice checkpoint at revision
`85e237c12c027371202489a0ec509ded67b5e4b5` on HX370 WSL CPU. One
seed-42 Ryan utterance yielded 117 frames of 16 codebook indices and a
224,640-sample waveform. It decoded frames 0–96 as 72 history plus 25 new
frames through the FP32 decoder and the revision-pinned ONNX export.

The [local report](report.json) shows ONNX Runtime CPU versus the exact
25-frame eager window at **1.02e-6 relative L2**. The eager window versus
the same segment from an FP32 full 117-frame decode differed by **7.26e-8
relative L2**. The [code tensor](codes.npz), [quantized fixture](fixture.npz),
[eager](eager_output.npz), [ONNX CPU](ort_output.npz) and
[full-decoder segment](full_reference_segment.npz) are retained. This checks
the window boundary on one real generated-code stream; it is not a speech
quality or mobile pipeline pass.

The quantized fixture was uploaded as [Workbench dataset](dataset_submission.json)
and [GPU-requested inference](inference_report.json) succeeded against the
exact pinned S25 TFLite target. The [device waveform](device_output.npz) was
finite and unsaturated; the [audit](audit_report.json) measures **1.153%
relative L2 / 38.76 dB SNR** versus local ONNX CPU on this generated-code
window. The separate 100-sample component profile used Workbench-generated
input of the same shape and does not measure this specific fixture. One
generated window does not establish speech intelligibility, all utterances,
or device-local talker/predictor/vocoder integration.
