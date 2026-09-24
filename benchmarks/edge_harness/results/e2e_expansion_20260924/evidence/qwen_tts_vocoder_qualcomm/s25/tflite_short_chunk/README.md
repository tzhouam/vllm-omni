# Two-frame S25 TFLite GPU-requested Code2Wav component

The same pinned [72-history-plus-two-new-frame source](../qnn_short_chunk/README.md)
and [one-sample fixture](../qnn_short_chunk/dataset_submission.json) compiled
as a [TFLite artifact](compile_report.json) for Galaxy S25. A
[GPU-requested inference](inference_report.json) returned a finite,
unsaturated 3,840-sample waveform. The [audit](audit_report.json) binds the
source, fixture, target and output hashes: versus local ONNX Runtime CPU on
the same fixture, waveform relative L2 was **2.325%**, SNR **32.67 dB**, and
maximum absolute difference **0.00810**. This is a numerical measurement,
not a listening-quality qualification. The [raw output](device_output.npz)
is retained.

The [placement profile](profile_report.json) returned **100** component
samples at nearest-rank p50/p95 **0.559/1.052 s** for **0.16 s of audio**.
Its execution detail assigned **403 rows to CPU and 335 to GPU**, so this is
a mixed CPU/GPU route, not a GPU-only result. The reported inference peak was
**379,121,664 bytes**. The [audit](audit_report.json) ties that profile to
the same S25 target and tensor shape as the waveform check; Workbench generated
the profile input, while the numerical inference used the retained fixture.
Component median real-time factor
is **3.49**, before talker, predictor, transfer or playback overhead; this
route does not meet playable streaming latency on the measured fixture.

A complete Qwen3-TTS stream also needs a qualified talker/predictor, history
and stage handoff, co-resident memory admission, cancellation, speech-quality
checks and sustained playback.
