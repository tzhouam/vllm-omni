# Spark output-head WSL-to-Windows handoff probe

**Depth B+C and measured component transport, not P.** The Omni `ExternalStage` sent four previously captured *real* Spark-X2.5-1.7B pre-final-norm activations from WSL to a native Windows VitisAI worker, returned full logits, and verified the actual execution profile. This was offline replay; vLLM did not generate those activations during this run. It establishes the transport cost for this boundary, not a complete Spark request or a beneficial split.

The source checkpoint is `XHToken/Spark-X2.5-1.7B` revision `448e61eb392c00f2c403185c5b56d5e0665bfaab`. The A16W8 opset-21 composite output-head graph (RMS norm plus full 131,072-logit projection) is SHA-256 `e2ae9c3fd7071e8ff628f209e6140c8a023bfaad59814b80a936c6b263f33eaf`; the four captured inputs/source logits are SHA-256 `cfa2ed8439f8fe01730319667bfd01abf2e39cf5851490346a0331304cf63194`. The graph remains in local native-NTFS artifact storage. Its creation and prior component numerical validation are in the [original output-head record](../../../e2e_expansion_20260923/evidence/spark_amd_npu_output_head/README.md).

The run used the HX370 laptop, WSL2 Ubuntu caller with the Omni 0.29.0rc2 source and installed vLLM 0.28.0, and a native Windows 11 build 26200 worker with AMD NPU driver `32.0.203.329`, VitisAI EP `1.8.63.0` and ORT `1.30.0`. The host power state and sustained temperature were not measured. An explicit `npu:amd` placement with minimum NPU node fraction 0.125 admitted; the retained [placement profile](placement_profile.json) attributes **one fused node to VitisAI and seven nodes to CPU**. One profiled warmup preceded 20 serial measured runs at concurrency one, cycling the four captured activations five times each. All 20 full-logit arrays were finite and matched the source top-1; SNR was 55.43–56.81 dB. [Raw measurements and metadata](report.json) contain every request and worker statistics.

| Measured boundary | Nearest-rank p50 / p95 |
|---|---:|
| Windows worker inference | 7.529 / 7.769 ms |
| WSL↔Windows transport and dispatch (round trip minus worker) | 1.453 / 4.276 ms |
| Full graph-stage round trip | **9.037 / 11.754 ms** |

Session creation took 11.958 s, excluding worker launch. The earlier [unbudgeted 20-run replay](unbudgeted_report.json) observed a 3.773 GB worker peak against a generic **1.946 GB** whole-stage budget; its [placement profile](unbudgeted_placement_profile.json) also verified one NPU node. The planner now accepts a device-specific measured peak hint, reserves it with 10% headroom, and checks the actual peak before requests begin. A fresh run using the previous 3,772,710,912-byte hint reserved **5.492 GB** including graph, runtime, workspace and external margin; its **3.775 GB** worker peak stayed within the reserved graph+runtime envelope. A separate [real no-hint control](unhinted_refusal.json) loaded the same graph and then explicitly refused `budget_exceeds_device` before any measured request; its [placement profile](unhinted_refusal_placement_profile.json) still showed one NPU node. The 3.774 GB load peak exceeded the generic graph+runtime reservation of about 0.604 GB. Unknown load peaks therefore still require a first controlled qualification run; this is not a pre-load safety guarantee for unprofiled graphs.

This worker-RSS admission pass covers only the isolated NPU graph stage; device allocations outside process working set are not measured. The same CPU/iGPU/NPU host RAM pool must also account for a resident Spark vLLM session, its KV and load peak before a joint plan can be admitted. The earlier Windows-native original FP32 CPU head measured 17.97/20.29 ms p50/p95 on a different run; these timings do not establish whole-generation speedup or a paired WSL CPU comparison.

Reproduce on this host with the pinned local graph:

```bash
PYTHONPATH=. /home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python benchmarks/edge_harness/experiments/profile_spark_npu_wsl_handoff.py \
  --graph /mnt/c/Users/zhout/w2/spark_npu_graph_bundle_with_norm_20260923/graph.onnx \
  --fixture benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/spark_amd_npu_output_head/real_acts_lmhead.npz \
  --output benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/spark_amd_npu_wsl_handoff/report.json \
  --profile-dir /mnt/c/Users/zhout/w2/spark_npu_wsl_handoff_20260924 \
  --worker-peak-rss-hint-bytes 3772710912 --repeats 20
```

Next gate: admit the *combined* vLLM+NPU load under current Windows/WSL shared RAM, connect the pre-final-norm activation to the live vLLM Spark decode path without changing checkpoint/precision or state semantics, then compare whole-request latency, token parity, cancellation and sustained memory/power against the unsplit backend. The current disposition remains **NOT E2E**.
