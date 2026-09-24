# Qwen3-TTS Code2Wav on the HX370 AMD NPU (2026-09-24)

**Disposition: component NPU compile failed; NOT E2E.** On native Windows
11 build 26200 with ONNX Runtime 1.30.0 and AMD VitisAI EP 1.8.63.0,
both fixed-shape, real-weight FP32 Code2Wav graphs passed a separate
same-file CPU reference check. Registering the AMD NPU and creating the
VitisAI session then terminated the process in `vaiml.dll` with an LLVM
`ArrayRef` assertion (`Index < Length`, line 260). Both process exit codes
were `0xC000001D` (signed `-1073741795`). The emitted profile files
are zero bytes. No NPU inference, node placement, waveform, latency, or
full Qwen3-TTS stream can be claimed from these attempts.

| Pinned graph | Input → output | ORT CPU versus retained reference | VitisAI result |
|---|---|---:|---|
| Historical 25-frame export, SHA-256 `7c07666229c6d404894132e22f784525e3212a49efad55ddd532eda5286c2b2b` | `[1,512,97]` → `[1,48000]` | relative L2 `6.46e-7`; unsaturated | Native compiler assertion |
| Pinned two-frame export at revision `85e237c12c027371202489a0ec509ded67b5e4b5`, SHA-256 `e563541bf77200ac4c4ca9d6e01cd7e25c09f9274233a8d82eeccc57580830d2` | `[1,512,74]` → `[1,3840]` | relative L2 `1.05e-6`; unsaturated | Same native compiler assertion |

The historical export's exact checkpoint revision remains unattested;
its file hash, fixture and retained CPU output are pinned in
[audit_report.json](audit_report.json). The two-frame graph derives from
the pinned CustomVoice snapshot described in the
[export report](../qwen_tts_vocoder_qualcomm/s25/qnn_short_chunk/export_report.json).
The FP32 ONNX files (about 424 MB each) and historical fixture are stored
outside Git under `/home/zhout/project/edge_infer/models/`; the two-frame
fixture and retained reference are in the linked S25 evidence directory.
No precision reduction, model substitution, or CPU fallback is counted as
AMD NPU execution.

The exact CPU controls are [25-frame](fp32_cpu_control.json) and
[two-frame](short_cpu_control.json). The raw mixed-encoding native logs are
[25-frame](fp32_attempt1.driver.log) and
[two-frame](short_attempt1.driver.log); they contain the assertion and
`vaiml.dll` compiler stack. Each zero-byte profile is retained to show
that ORT did not emit NPU placement events. These are single-fixture
component attempts with no warmup, repeated samples, power control, or
task-quality validation.

To reproduce, run
[probe_qwen_tts_amd_npu_vocoder.py](../../../../experiments/probe_qwen_tts_amd_npu_vocoder.py)
with the model, fixture and reference paths from the CPU reports,
`--expected-model-sha256`, `--expected-fixture-sha256` and
`--expected-reference-sha256` from the audit report, plus
`--context-frames`, `--output-samples`, `--report`,
`--profile-prefix` and `--ep-dir` pointing to the installed VitisAI
ExecutionProvider. `--cpu-only` reproduces the parity gate without
starting the NPU compiler. Run the NPU case in a separate process because
the native assertion cannot be caught by Python.

Next: bisect the specific ONNX operator/subgraph that trips this VitisAI
compiler, export a version it accepts without losing same-checkpoint
waveform quality, verify nonzero NPU node placement and measured coarse
stage benefit, then connect talker, codec, state, admission, cancellation,
and complete-stream quality gates through Omni. This compiler failure
constrains these graphs and EP version only.

## Compiler bisection and equivalent decoder rewrite

A follow-up [bisection audit](bisection_report.json) extracted reachable
prefixes from the pinned two-frame graph. All used its one fixed real-weight
fixture on the same HX370 Windows/VitisAI stack. Each row is **one component
inference**, with no warmup, repeated timing, waveform, or full-stream claim:

| Source cut / extracted nodes | VitisAI node events | Session creation | Intermediate relative L2 versus CPU |
|---|---:|---:|---:|
| 25 / 18 | 1 | 164.08 s | 0.958% |
| 50 / 50 | 1 | 301.30 s | 1.059% |
| 100 / 97 | 1 | 330.92 s | 0.391% |
| 200 / 192 | 1 | 395.85 s | 1.083% |

The first 100-node attempt was manually stopped after about five minutes
of active compilation; its longer retry passed. The 1% intermediate gate
used by the probe marks cuts 50 and 200 numerically unqualified. It is a
screening gate, not a validated waveform or listening tolerance. The
successful cuts show that the model prefix can run on this NPU; they do
not justify splitting a production vocoder at those boundaries.

The first decoder `ConvTranspose1d` isolated with its original weights
reproduced the exact `vaiml.dll` / LLVM `ArrayRef` assertion in
[its raw log](local_convtranspose0_probe.log). A
[shape-preserving rewrite](../../../../experiments/rewrite_qwen_tts_convtranspose1d.py)
expresses that operation as `Unsqueeze → ConvTranspose2d` with a
size-one height axis and `Squeeze`. The rewritten isolated operator was
bitwise identical on ORT CPU and [executed as one VitisAI NPU node](local_convtranspose0_2d_probe.json)
on a labeled synthetic activation; its NPU output differed by 1.636%
relative L2 from CPU. That value is not a task-quality pass.

Applying the same transformation to all six upsampling convolutions
produced a [full two-frame graph](full_short_2d_rewrite.json) whose
`[1,3840]` CPU waveform was bitwise identical to the original graph on
the pinned fixture. The ~424 MB rewritten artifact remains outside Git;
its SHA-256 is pinned in the report. A full-graph NPU waveform, placement
profile and complete TTS stream are still unqualified in this evidence.
The [extraction/probe tool](../../../../experiments/bisect_qwen_tts_vitisai_graph.py)
records cut hashes, CPU execution, actual provider events and intermediate
numerics. No precision, weights or checkpoint revision changed in the
rewrite.
