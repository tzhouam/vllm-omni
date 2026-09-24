# InternVLA Cosmos Conv13 batch-one recovery on HX370 AMD NPU

**Disposition (2026-09-25): one real-weight encoder convolution passed a
fixed-fixture NPU numerical check after rebatching; InternVLA on this device
is still NOT E2E.** The first 150 encoder nodes, remaining encoder nodes and
real Place_Markpen policy were not executed together with this NPU stage in a
live Omni request. The NPU ran only the first Conv13 in a replayed boundary;
the rest of the encoder and synthetic policy-sensitivity run used CPU.

The [earlier six-frame A16W8 attempt](../../../e2e_expansion_20260924/evidence/internvla_cosmos_amd_npu_fp32/README.md)
placed one VitisAI partition but had 4.4948 aggregate relative L2 error.
A [per-frame audit](batch_fault_analysis.json), reproduced by
[`analyze_internvla_batch_fault.py`](../../../../experiments/analyze_internvla_batch_fault.py),
found that frame 0 matched quantized CPU within **1.11e-5 relative L2**, while
frames 1–5 were all the same constant value and each had **4.924 relative
L2**. The pattern input and CPU outputs were identical across six frames.
Moving the real Conv bias to an unchanged FP32 Add gave [bitwise CPU parity](../internvla_amd_npu_conv_bias/rewrite_report.json)
but the NPU still corrupted frames 1–5 (each **2.531 relative L2**;
[placed probe](../internvla_amd_npu_conv_bias/npu_probe.json)). Thus bias
handling affects the error but does not explain the batch-index failure.
The fault is observed on these graph layouts and EP version; it is not a
general statement about all AMD NPU convolutions.

The [batch-one exporter and probe](../../../../experiments/probe_internvla_cosmos_batch1.py)
changed the fixed shape from `[6,128,64,64] → [6,256,64,64]` to six
independent `[1,128,64,64] → [1,256,64,64]` calls, without changing weights
or operations. The batch-one FP32 CPU graph reproduced the original six-frame
FP32 graph **bitwise** on both ramp and pattern fixtures
([preparation report](prepare_report.json)). The source boundary SHA-256 is
`ff2ebad2c59a472b561536fa49cdd8bc43cf8e7bcf000a8ff4420f8e3a4766c8`;
the batch-one FP32 graph is
`ced07f35fa68a909693c3049cd5e399c5aaf37acc75045442ccbf540a76dc89e`.
All large artifacts and fixtures remain outside Git.

An initial A16W8 per-channel QDQ candidate calibrated on ramp frames 0–3
and one pattern frame failed the CPU numerical gate on ramp frames 4–5 at
**1.353%/3.979% relative L2** ([failed candidate](quantization_report.json)).
The retained [seven-frame calibration](quantization_report_cal7.json) added
those ramp frames and passed the provisional 1% CPU gate on this *same*
synthetic set, with a maximum **0.949% relative L2**. This is not independent
task calibration or broad quality validation. The resulting 307,187-byte
candidate SHA-256 is
`2ae37f405cb7cb469f01d7022c0e789650e486d81279713715d0e9808af15f67`.

On native Windows 11 build 26200, HX370 NPU driver `32.0.203.329`, ORT
`1.30.0` and VitisAI EP `1.8.63.0`, the [final placement/numerical probe](npu_probe_final.json)
ran 12 correctness frames plus one warmup and 20 repeated six-frame pattern
boundaries. Its [raw ORT profile](profile_final_2026-09-25_00-39-52_965.json)
contains **133 VitisAI node events** and 399 CPU events: one NPU partition
and three CPU operations per frame call. Across the 12 correctness frames,
maximum NPU-versus-same-QDQ-CPU relative L2 was **1.72e-5** and maximum
NPU-versus-original-FP32-batch-six error was **0.949%**. A separate
[assembled-output probe](npu_probe_assembled.json) pins both six-frame NPU
outputs outside Git at SHA-256
`64a77fa5c20d55b103b8b08134f137ee8f42bdc18452214da00609b69e374e66`.

Within one Windows process, 20 alternating-order repeated *pattern*
boundaries measured **23.778/26.527 ms p50/p95** for the original FP32 CPU
batch-six graph and **14.468/22.106 ms** for six sequential NPU batch-one
calls. These are complete isolated boundary wall times, not whole-encoder
or policy timings. Session creation was 1.460 s. Existing caches were not
cleared; power and thermals were not controlled ([hardware snapshot](hardware.json)).
The boundary transfers about 12 MiB in and 24 MiB out at this shape, so
same-process timings cannot establish a WSL↔Windows Omni speedup.

A [CPU encoder-suffix replay](downstream_report.json) removed Conv13 from
the original real-weight ONNX encoder and supplied the measured NPU tensor.
The CPU-boundary control reproduced the unchanged full encoder latent
bitwise. The NPU-injected final latent differed by **0.807% relative L2**
on ramp and **0.562%** on pattern; the suffix did not cause a larger
numerical failure on these fixtures. Its output is pinned outside Git at
SHA-256 `d63ea16b124fde1196d7eb28fe63b653841a96d3c900c3facf4c69127f23556c`.
The [reproducer](../../../../experiments/probe_internvla_npu_boundary_downstream.py)
checks source/weight, pixel and NPU-output hashes before replay.

Finally, the [real Place_Markpen policy action-sensitivity probe](action_sensitivity_report.json)
injected that pattern latent into the CPU policy under fixed synthetic
state and zero noise. It returned finite `[1,50,32]` actions, repeated the
source actions bitwise, and measured **0.876% action relative L2**, maximum
absolute change **0.01039**, cosine **0.999973**. The [policy reproducer](../../../../experiments/probe_internvla_npu_latent_actions.py)
pins the checkpoint SHA-256 and requires `PYTHONPATH` to point at this fork
checkout; the neighboring editable Omni installation has a different
InternVLA/Transformers pairing and fails to import. There is no
real-observation reference action, physical unit/order/step-time tolerance,
live NPU-to-policy handoff, complete-request latency, admission/cancellation
or sustained power result. The device/model cell therefore remains NOT E2E.

Next, test representative non-calibration observations, then integrate a
coarse encoder plan in a native Windows policy worker only if complete-policy
quality, transfer-inclusive latency and shared-RAM admission justify it.
