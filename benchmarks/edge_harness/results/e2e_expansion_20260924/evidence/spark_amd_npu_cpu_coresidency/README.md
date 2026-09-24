# Spark BF16 CPU and AMD NPU output-head co-residency

**Depth P for the existing CPU text route, C for the NPU output head; no split P.** A WSL Omni/vLLM Spark-X2.5-1.7B BF16 CPU engine and the native Windows VitisAI output-head worker stayed loaded together on the HX370 laptop. The CPU requests used vLLM's own output head. The NPU worker replayed previously captured activations *after* the CPU request series, while the CPU model remained resident. No live activation crossed from vLLM into the NPU, and no request was jointly computed by both backends.

The CPU checkpoint is `XHToken/Spark-X2.5-1.7B` revision `14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8`, with its pinned [BF16 manifest](../spark_bf16_wsl_cpu/weight_manifest.json). The A16W8 output-head graph came from revision `448e61eb392c00f2c403185c5b56d5e0665bfaab`. Before this co-residency run, an [exact tensor comparison](../spark_amd_npu_wsl_handoff/weight_identity.json) checked the graph's FP32 RMS-norm weight and full transposed output projection against the newer BF16 checkpoint: both were bitwise equal after BF16→FP32 conversion. This establishes **head-weight identity only**, not equality of every model weight or live activation. The graph SHA-256 is `e2ae9c3fd7071e8ff628f209e6140c8a023bfaad59814b80a936c6b263f33eaf`; the captured activation fixture SHA-256 is `cfa2ed8439f8fe01730319667bfd01abf2e39cf5851490346a0331304cf63194`. The graph and BF16 weights remain outside Git.

The tested route was WSL2 Ubuntu CPU with installed vLLM `0.28.0` and Omni `0.29.0rc2` source, plus Windows 11 build 26200, AMD NPU driver `32.0.203.329`, VitisAI EP `1.8.63.0` and ORT `1.30.0`. A Windows PowerShell physical-RAM check and WSL available-RAM check preceded both loads. CPU and NPU plans reserved **8.202 GB** and **5.492 GB** respectively, **13.694 GB combined**, below both the contemporaneous WSL **30.624 GB** and Windows **30.423 GB** available readings. This is an explicit sum of two plan budgets on the same physical RAM pool, not two independent capacities. The NPU budget used the prior measured worker-load peak plus 10% headroom. After both loads, available RAM was 24.219 GB in WSL and 20.169 GB in Windows. The CPU process-tree sampled attributable peak was 6.949 GB; the native worker's lifetime peak RSS was 3.774 GB. These are distinct counters and measurement windows, not an additive observed whole-system peak. Windows available RAM had not returned to its starting value after teardown; file cache and other processes were uncontrolled.

One warmup preceded **20 serial complete CPU text requests** (28 prompt tokens, 64 generated tokens, concurrency one) while the NPU worker stayed loaded but idle. All 20 produced the exact token-ID hash from the earlier [standalone BF16 CPU profile](../spark_bf16_wsl_cpu/profile20_branch_report.json). Nearest-rank complete-request wall p50/p95 was **4.804/4.909 s**; the earlier 5.411/5.751 s profile is a separate run under uncontrolled cache and power conditions, so this is not a co-residency speedup. The CPU stage reported `device_type=cpu` and BF16. Afterward, 20 serial captured-activation NPU calls, cycling four cases, matched source top-1 **20/20** at round-trip p50/p95 **8.607/9.976 ms**. The [placement profile](placement_profile.json) attributes one fused node to VitisAI and seven nodes to CPU. [Raw report](report.json) retains plans, memory readings, timings, hashes and per-request records. Power, thermal and simultaneous CPU+NPU compute were not measured.

Reproduce on this host, from the fork checkout root:

```bash
PYTHONPATH=. /home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python benchmarks/edge_harness/experiments/verify_spark_npu_head_weight_identity.py \
  --source-graph /home/zhout/project/edge_infer/analysis/experiments/spark_edge/onnx/lm_head.onnx \
  --bf16-model /home/zhout/project/edge_infer/models/Spark-X2.5-1.7B-BF16 \
  --report benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/spark_amd_npu_wsl_handoff/weight_identity.json
PYTHONPATH=. /home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python benchmarks/edge_harness/experiments/profile_spark_cpu_npu_coresidency.py \
  --model /home/zhout/project/edge_infer/models/Spark-X2.5-1.7B-BF16 \
  --graph /mnt/c/Users/zhout/w2/spark_npu_graph_bundle_with_norm_20260923/graph.onnx \
  --fixture benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/spark_amd_npu_output_head/real_acts_lmhead.npz \
  --reference-profile benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/spark_bf16_wsl_cpu/profile20_branch_report.json \
  --weight-identity benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/spark_amd_npu_wsl_handoff/weight_identity.json \
  --profile-dir /mnt/c/Users/zhout/w2/spark_cpu_npu_coresidency_20260924 \
  --worker-peak-rss-hint-bytes 3772710912 \
  --report benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/spark_amd_npu_cpu_coresidency/report.json
```

Follow-up: a [live CPU decoder + AMD NPU head run](../spark_amd_npu_live_split/README.md) now handed current pre-final-norm activations across this boundary for 20 measured complete requests. Exact greedy-token parity failed at a BF16 top-logit tie, and cancellation plus paired latency/power remain open. The split is therefore still **not qualified**.
