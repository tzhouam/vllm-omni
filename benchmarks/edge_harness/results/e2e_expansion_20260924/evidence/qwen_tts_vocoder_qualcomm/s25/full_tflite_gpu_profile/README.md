# S25 25-frame TFLite GPU-requested vocoder profile

The earlier [S25 GPU inference](../../s25_reference_inference.json) used the
retained 97-frame input (72 history + 25 new frames) and produced a 48,000-sample,
2-second waveform on the exact TFLite target `mng7kwxon`. Its
[waveform comparison](../../s25_cpu_comparison.json) to local ONNX Runtime CPU
was **4.2385% relative L2 / 27.46 dB SNR** on one synthetic fixture; there
is no listening-quality tolerance. The historical source export did not
attest an exact checkpoint revision.

A new [pinned eager-decoder comparison](pinned_eager_parity_report.json) used
the locally complete CustomVoice snapshot at revision
`85e237c12c027371202489a0ec509ded67b5e4b5` on the same 97-frame input.
Its full 48,000-sample waveform matched the historical ONNX Runtime CPU
output at **1.89e-6 relative L2 / 114.47 dB SNR**. This supports numerical
continuity on that fixture; it does not prove the historical ONNX export's
checkpoint revision. The [reproduction script](../../../../../../experiments/probe_qwen_tts_full_vocoder_parity.py)
pins artifact hashes and decoder weights.

A new [placement profile](profile_report.json) of that exact S25 target and
97-frame tensor shape returned **100** component samples: nearest-rank p50/p95
was **0.839/1.157 s** for 2 seconds of audio, with **403 CPU and 335 GPU**
execution-detail rows. Reported inference peak memory was **374,591,488
bytes**. The component-only p50/p95 real-time factors were **0.420/0.579**.
Workbench generated the profile input with the same shape; the historical
waveform inference used the retained fixture. The [audit](audit_report.json)
binds target, device, fixture, numerical output and profile. Its reported
estimated-time field is the **0.561 s sample minimum**, not the median.

This is a promising throughput candidate for a 25-frame vocoder stage, not a
complete or quality-qualified Qwen3-TTS stream. The talker, code predictor,
resident history, handoff, first-audio latency, cancellation, speech quality
and sustained co-resident memory/thermal behavior remain unverified. The
separate pinned [two-frame candidate](../tflite_short_chunk/README.md)
has a higher component median RTF and cannot be treated as the same artifact
or checkpoint revision.
