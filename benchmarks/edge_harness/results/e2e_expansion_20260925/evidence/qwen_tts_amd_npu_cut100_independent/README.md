# Independent Qwen3-TTS fixed-history Code2Wav control

**Disposition: CPU state-contract failure; NPU was not run in this follow-up.**
The fixed 72-history/two-new-frame Code2Wav prefix that previously passed
three NPU-prefix/CPU-suffix waveform windows is not a valid continuous-stream
substitute on a second generated utterance. This is a gate before compiling or
integrating the prefix for another input; it does not reverse the earlier
isolated NPU numerical observations.

The real Qwen3-TTS 0.6B CustomVoice checkpoint is pinned to revision
`85e237c12c027371202489a0ec509ded67b5e4b5`. The [independent source
report](../qwen_tts_stateful_npu_boundary/independent_codes/report.json) records
a seed-91 Ryan prompt about a blue kite, 144 generated speech-code frames,
the model/decoder hashes, and an ONNX CPU 25-frame waveform within 9.86e-7
relative L2 of the FP32 eager window. The reused real-weight 72-history
two-frame graph has SHA-256
`e563541bf77200ac4c4ca9d6e01cd7e25c09f9274233a8d82eeccc57580830d2`.
All windows were run on HX370 WSL CPU with ONNX Runtime 1.29.0, FP32
`[1,512,74]` input, and 24 kHz `[1,3840]` output. No VitisAI session was
created in this control.

| Generated-code frame start | Short CPU waveform vs matching 25-frame CPU segment, relative L2 | Provisional 1% chunk gate | Raw record |
|---:|---:|---|---|
| 0 | 0.0000% | pass | [preparation_0.json](preparation_0.json) |
| 2 | 1.3115% | fail | [preparation_2.json](preparation_2.json) |
| 23 | 2.2341% | fail | [preparation_23.json](preparation_23.json) |

The retained [frame-0](fixture_0.npz), [frame-2](fixture_2.npz), and
[frame-23](fixture_23.npz) inputs have corresponding
[0](reference_0.npz), [2](reference_2.npz), and [23](reference_23.npz) CPU
waveforms. Their SHA-256 values are in the respective preparation JSON files.
The source 25-frame waveform
is `independent_codes/ort_output.npz` (SHA-256
`db08a0cf91a36f52481e44bb4de1070e113503fd4f52867be97e7d3a51d3712e`).
The frame-0 control confirms exact alignment. The shifted fixed input window
does not preserve a source-equivalent output by frame 2 on this utterance,
before any NPU quantization error. The separate effects of omitted history
and position are not isolated here. These are numerical
comparisons, not listening-quality judgments. The 1% gate is provisional and
cannot itself qualify speech.

From the repository root, reproduce the frame-2 record with:

```bash
python benchmarks/edge_harness/experiments/prepare_qwen_tts_real_code_short.py \
  --source ../models/qwen3tts_code2wav_c2_ctx72_20260924.onnx \
  --real-fixture benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_stateful_npu_boundary/independent_codes/fixture.npz \
  --real-report benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_stateful_npu_boundary/independent_codes/report.json \
  --real-25-reference benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_stateful_npu_boundary/independent_codes/ort_output.npz \
  --expected-real-fixture-sha256 7427ec1bb4e16871dc53244ecf331dcb357bfc3646a64e85cb127744262aafcc \
  --expected-real-reference-sha256 db08a0cf91a36f52481e44bb4de1070e113503fd4f52867be97e7d3a51d3712e \
  --start-frame 2 \
  --fixture-output /tmp/qwen_tts_independent_fixture_2.npz \
  --reference-output /tmp/qwen_tts_independent_reference_2.npz \
  --report /tmp/qwen_tts_independent_preparation_2.json
```

Use `--start-frame 0` and `23` for the other records. The preparer checks
the recorded checkpoint revision and the export, fixture and reference hashes.
This result makes
an exact rolling/all-prior-history
state contract the next prerequisite for a useful NPU vocoder cut. The
existing narrow CPU-attention/NPU-MLP route remains offline component evidence;
no complete NPU text-to-audio request or whole-chain benefit is claimed.
