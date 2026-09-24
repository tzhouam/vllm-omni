# Spark-X2.5-1.7B output head on the HX370 AMD NPU

**Depth: B+C, not P.** A real-weight output-head component passed numerical and placement checks through the Omni `external.graph.v1` Windows worker. No Spark prefill, continuous decode, KV state, sampling, output-head handoff from vLLM, or complete text request ran on this route. The [60-cell matrix](../../../e2e_profiling_20260922/evidence/summary/README.md) therefore remains **NOT E2E** for this pairing.

The source is `XHToken/Spark-X2.5-1.7B` revision `448e61eb392c00f2c403185c5b56d5e0665bfaab`. The FP32 ONNX output head (SHA-256 `f5f0c31eb438e9940ab242da8bf6f0f5dad6ebc9f6ef02d5475c21b744e595bc`) takes a pre-final-norm `[1,1,2048]` activation and returns all `[1,131072]` logits. The four captured real activations and source-reference logits are [retained here](real_acts_lmhead.npz), SHA-256 `cfa2ed8439f8fe01730319667bfd01abf2e39cf5851490346a0331304cf63194`. The FP32 source graph matched those captured references at 133.66–137.37 dB SNR, top-1 4/4. These four activation cases are a small numerical check, not a task-quality suite.

The qualified ONNX graph uses opset 21, RMS norm plus four distinct 32,768-column real-weight MatMul branches and Concat, MatMul-only QDQ with QUInt16 activations/QInt8 weights calibrated by MinMax on the four captured activations, and FP32 I/O. Its SHA-256 is `e2ae9c3fd7071e8ff628f209e6140c8a023bfaad59814b80a936c6b263f33eaf`. The 268 MB graph and 1.1 GB FP32 source remain in local artifact storage; they are not Git files. The [local bundle manifest record](local_bundle_manifest.json) pins the graph, input, validation, revision, layout and precision. That manifest references `graph.onnx` in the local bundle and cannot be executed from this checkout alone. [The preparation script](../../../../experiments/prepare_spark_amd_npu_bundle.py) verifies the graph and source hashes and numerical gate before making a local Omni bundle.

Hardware was Ryzen AI 9 HX 370 on Windows 11 build 26200, with AMD NPU driver `32.0.203.329`, VitisAI Windows Workload EP `1.8.63.0`, ONNX Runtime `1.30.0`, ONNX `1.22.0`, Python `3.12.10`, and the current Omni source. The [direct ORT report](composite_report.json), [raw log](composite_raw.log), and [profile](composite_profile.json) show four actual VitisAI node events over four activations, with 28 CPU node events. Full-logit SNR was 55.43–56.81 dB with finite outputs and top-1 4/4. The [Omni NPU run](spark_npu_graph_stage_with_norm_20260923.json) admitted a 4 GiB shared-RAM reservation and verified **one fused NPU partition plus seven CPU nodes** per request (NPU fraction 0.125). Its [worker profile](omni_npu_placement_profile.json) is retained; there was no silent CPU-only substitution. The reservation was released after shutdown.

The following timings are one warmup plus 20 serial repeats of **one captured activation** at concurrency 1, through the same Omni graph-stage boundary. They exclude whole-model inference and handoff; nearest-rank p50/p95 are reported, and power/thermal conditions were not captured.

| Output-head stage | p50 / p95 | Startup | Peak worker RSS | Placement |
|---|---:|---:|---:|---|
| [A16W8 NPU](spark_npu_graph_stage_with_norm_20260923.json) | 10.21 / 12.44 ms | 18.21 s | 3.52 GiB | 1 NPU + 7 CPU nodes |
| [Original FP32 CPU](spark_cpu_fp32_graph_stage_20260923.json) | 17.97 / 20.29 ms | 4.48 s | 2.06 GiB | 3 CPU nodes |
| [Same A16W8 graph on CPU](spark_cpu_graph_stage_with_norm_20260923.json) | 69.90 / 82.83 ms | 2.04 s | 0.84 GiB | 24 CPU nodes |

The NPU result was faster than this original FP32 CPU *component* baseline but used longer startup and more peak worker memory. The same-graph CPU comparison includes QDQ overhead. Neither comparison establishes a benefit for moving a token-by-token output head off the vLLM backend; that requires an actual handoff and full-generation measurement.

Failed candidates remain evidence: the [monolithic QDQ output head](failures/full_qdq_report.json) and [full-width MatMul](failures/full_matmul_report.json) had **zero** NPU node events despite listing VitisAI; they ran on CPU. Four independently compiled, unnamed co-resident shards gave [bad full-logit numerics](failures/unnamed_co_resident_report.json), including one wrong top-1. Distinct graph/tensor names fixed the separate-session check ([report](full_head_report.json)); the final single composite graph avoids that separate-session arrangement. These failures apply to those graph formats, not to Spark or the NPU in general.

Reproduce the direct composite probe on this host, after regenerating or locating the pinned FP32 source graph:

```powershell
C:\Users\zhout\npu-ep\Scripts\python.exe benchmarks/edge_harness/experiments/probe_spark_amd_npu_lm_head.py `
  --model \\wsl.localhost\Ubuntu\home\zhout\project\edge_infer\analysis\experiments\spark_edge\onnx\lm_head.onnx `
  --calibration benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/spark_amd_npu_output_head/real_acts_lmhead.npz `
  --ep-dir 'C:\Program Files\WindowsApps\WindowsWorkload.EP.AMD.VitisAI.Framework.1.8_1.8.63.0_x64__8wekyb3d8bbwe\ExecutionProvider' `
  --composite-with-norm --output-dir C:\Users\zhout\w2\spark_npu_lmhead_repro
```

Follow-up: a [live CPU decoder + AMD NPU head experiment](../../../e2e_expansion_20260924/evidence/spark_amd_npu_live_split/README.md) connected this component to vLLM with joint memory admission and completed 20 measured requests. Exact greedy-token parity failed at one quantization-sensitive BF16 top-logit tie. Broader quality, cancellation, paired whole-request benefit and sustained power remain open, so this split is not qualified.
