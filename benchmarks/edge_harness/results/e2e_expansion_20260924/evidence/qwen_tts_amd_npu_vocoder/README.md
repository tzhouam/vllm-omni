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

The initial compiler failure constrains these graphs and EP version only.
The follow-up below isolates one trigger and tests a numerically equivalent
ONNX rewrite.

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
its SHA-256 is pinned in the report. A [full rewritten graph NPU attempt](full_short_2d_npu_probe.json)
then completed with **two VitisAI NPU node events and 139 CPU node events**,
so this was mixed execution, not an all-NPU vocoder. Its finite,
unsaturated waveform differed from the same-graph CPU result by **15.696%
relative L2 / 16.08 dB SNR** on the fixed fixture, failing the probe's
1% waveform gate. Session creation took **2515.0 s**. One cold inference
took **2.205 s** for nominal 0.16 s audio (single-sample component RTF
**13.78**); there was no warmup or repeated latency profile. The
[raw profile](full_short_2d_profile_2026-09-24_19-07-20_388.json)
and [native log](full_short_2d_npu_probe.log) are retained.
An [isolated original `Sin` node](local_sin0_probe.json) on a labeled
synthetic activation executed entirely on ORT CPU (one CPU event, zero
VitisAI events) with exact CPU parity; the full graph likewise lists
CPU sine nodes. That control identifies a real fallback boundary but
does not by itself explain the 15.696% waveform difference.

A real-weight [cut at source node 486](cut486_probe_paired.json), just
before the first decoder transposed convolution, executed one NPU
partition and one CPU node. Its `[1,1024,74]` activation differed from
CPU by **2.265% relative L2** on the pinned input. Feeding the captured
CPU activation into the [original CPU decoder suffix](decoder_suffix_extract.json)
reproduced the retained waveform at `1.05e-6` relative L2. Feeding
the captured NPU activation through that **same** CPU decoder yielded a
[waveform](boundary_handoff_report.json) **9.689% relative L2 /
20.27 dB SNR** from the CPU-boundary control. The paired activations
and output waveforms are pinned in the audit report. This is one
fixed-shape component handoff, without streaming, representative
quality or a warm performance profile.

The rewrite therefore fixes the isolated compiler crash and keeps CPU
waveform behavior exact, but the NPU prefix already misses the waveform
gate and the full mixed AMD NPU route differs still more. [AMD's
model-support documentation](https://ryzenai.docs.amd.com/projects/WinML/en/latest/model_support.html)
states that its VitisAI EP automatically converts float CNN and
Transformer models to BF16 during compilation. That makes compiled
precision a candidate explanation, **not a diagnosis established by
these profiles**. An [unvectorized-layout control](cut486_unvectorized_probe.json)
kept the same source graph, input and cut while setting AMD's
[documented VitisAI option](https://ryzenai.docs.amd.com/projects/WinML/en/stable/modelrun.html)
`preferred_data_storage=unvectorized` with `optimize_level=1`.
It still executed one NPU partition and one CPU node, and its
pre-decoder difference rose to **2.287%**. The
[same CPU decoder handoff](boundary_unvectorized_handoff_report.json)
produced a waveform **10.074%** from the CPU-boundary control.
This setting did not fix the one-fixture quality failure.

Next, isolate which NPU prefix operations cause the boundary error,
test a numerically justified precision or calibrated QDQ alternative
against the same waveform gate, and measure a useful coarse stage
against CPU after warmup. Only then connect talker, codec,
state, admission, cancellation and complete-stream quality through
Omni.
The [extraction/probe tool](../../../../experiments/bisect_qwen_tts_vitisai_graph.py)
records cut hashes, CPU execution, actual provider events and intermediate
numerics. No precision, weights or checkpoint revision changed in the
rewrite.
