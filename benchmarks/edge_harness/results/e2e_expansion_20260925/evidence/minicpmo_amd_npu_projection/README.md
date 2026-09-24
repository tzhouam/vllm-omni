# MiniCPM-o speech-token projection on HX370 AMD NPU (2026-09-25)

**Disposition: one isolated real-weight projection passed the fixed-fixture
numerical gate on the AMD NPU; MiniCPM-o on this configuration remains NOT E2E.**
The earlier [complete 20-layer speech-head step](../../../e2e_expansion_20260924/evidence/minicpmo_amd_npu_speech_head/README.md)
ran on the NPU but changed K/V cache outputs by as much as 4.353% relative L2.
This experiment leaves the 20-layer speech head and all state on CPU and sends
only its final normalized `[1,1,768]` activation to the projection stage.
It does not run the thinker, encoders, continuous speech loop or vocoder.

The [export script](../../../../experiments/probe_minicpmo_amd_npu_projection.py)
checked the pinned 776,341,465-byte ONNX source (SHA-256
`78cf64804f11ad269ee4180da61588ccc5eefe2942773227345499756e4cc229`),
44-input synthetic cache fixture and retained 41-output CPU reference. An
augmented CPU run reproduced **all 41 outputs bitwise**. The extracted FP32
MatMul+Gather projection reproduced its logits bitwise. The FP32 projection
artifact is 20,160,199 bytes, SHA-256
`c81c2cdd5ca4af4e4ca398baf83a685518a42b649ebc26cd198d3d85a0799f70`;
the captured activation fixture is SHA-256
`2304a867cfb9d232212a93a046cd8e789438a6085eeffec0480ebdd28cc00309`.
An actual CPU-prefix graph removes the final MatMul and Gather and their
weights, returning the 40 new-cache tensors plus that activation. Its
756,181,266-byte artifact has SHA-256
`21be0ea89add8fe4b885e5a0d00400c30b1190dd0287f56515446709bd30a040`.
On the retained fixture, all 40 cache tensors and the activation matched
the uncut CPU graph **bitwise**. The isolated projection then supplies the
41st output. This checks one split step, not continuous cache evolution or
an Omni handoff.

The original export's exact checkpoint revision remains unattested, so these
hashes pin the result, not a claimed model release revision.

Windows ONNX Runtime 1.30.0 generated an A16W8 per-channel QDQ projection from
**one captured activation**, using MinMax calibration. The 5,073,759-byte
candidate has SHA-256
`ca4f9a95af4933363e05da5391681a6dabfbeb5e123bc6fb1ad3875df675118e`.
Its CPU output preserved top token **1867** and differed by **0.861% relative
L2** from FP32 CPU logits. On native Windows 11 build 26200, HX370 AMD NPU
driver `32.0.203.329` and VitisAI EP `1.8.63.0`, 20/20 measured NPU outputs
also preserved token 1867. Their maximum relative L2 was **5.65e-6 versus the
same quantized graph on CPU** and **0.861% versus FP32 CPU**. The retained
[final report](npu_probe_final.json) and [profile](npu_profile_final_2026-09-25_00-23-56_274.json)
show 21 VitisAI node events and 63 CPU node events for one warmup plus 20 calls:
one NPU partition and three CPU operations per call. This is a component C
pass on one activation, not a stateful or task-quality pass.

The [hardware record](hardware.json) pins the observed driver and package;
power and thermal conditions were not controlled or sampled.

The isolated warm-call p50/p95 was **0.338/0.406 ms on NPU**, **8.925/13.019 ms
for the same QDQ graph on CPU**, and **0.057/0.360 ms for the original FP32
projection on CPU** (20 calls each, nearest-rank percentiles). NPU session
creation took 1.497 s. These are same-process calls without cross-OS transfer,
Omni dispatch, model co-residency or power control. The practical CPU
alternative is the FP32 projection, not the slow QDQ CPU graph. On this
fixture the NPU projection alone is already slower than FP32 CPU, before
handoff costs. The accepted architecture therefore does **not** justify
integrating this per-token split as a default execution plan. A larger
numerically qualified coarse stage or measured overlap benefit is needed.

The CPU prefix, FP32 projection, QDQ candidate and captured activation remain outside Git
under `/home/zhout/project/edge_infer/models/minicpmo_amd_npu_projection_20260925/`
and `C:\Users\zhout\w2\minicpmo_npu_projection_20260925\`; their hashes and
commands are encoded by the [export report](export_report.json),
[quantization report](quantization_report.json), and final probe. The first
[NPU probe](npu_probe.json) and intermediate [repeat](npu_probe_repeat.json)
are retained rather than substituted for the final paired comparison.

Next, obtain representative speech-head activations from full MiniCPM-o
requests and test a coarse, state-safe NPU stage against the exact FP32 CPU
reference. Only if the complete request benefits after transfer, startup,
memory and power costs should an Omni external stage be made eligible.
