# Spark BF16 CPU versus CPU+AMD NPU paired-order profile

**Disposition (2026-09-25): scoped complete-request quality passed; no NPU
latency benefit on this fixture.** On the Ryzen AI 9 HX 370 laptop, Omni/vLLM
kept the 1.7B BF16 Spark decoder, KV and sampler on WSL CPU. The opt-in
hybrid sent each post-final-norm token activation to the pinned A16W8 AMD
NPU projection, then re-ranked its top 64 candidates with the resident BF16
CPU head for greedy decoding. Both routes generated the same reference token
hash on every measured request. The [independent audit](audit.json) verifies
the [phase report](paired_report.json), each complete-request report, the
two worker reports and raw ORT traces. The run did **not** use the Radeon or
RTX; this is the CPU+NPU cell, not joint CPU+iGPU+NPU execution.

The fixed workload was one 28-prompt-token inventory question, 64 greedy
output tokens with EOS ignored, single concurrency, one excluded warmup then
20 measured requests per phase. Four fresh sessions ran in CPU/NPU/NPU/CPU
order. These are nearest-rank complete-request wall times, including Omni
handoff but excluding model/stage startup:

| Phase | Complete requests | p50 / p95 wall | Reference hash | Placement |
|---|---:|---:|---|---|
| [01 CPU](01_cpu_report.json) | 20/20 | 4.766 / 4.868 s | 20/20 | CPU |
| [02 CPU+NPU](02_npu_report.json) | 20/20 | 5.003 / 5.133 s | 20/20 | one VitisAI, six CPU nodes |
| [03 CPU+NPU](03_npu_report.json) | 20/20 | 5.030 / 5.167 s | 20/20 | one VitisAI, six CPU nodes |
| [04 CPU](04_cpu_report.json) | 20/20 | 4.829 / 4.908 s | 20/20 | CPU |

The NPU-to-CPU phase-p50 ratios were **1.050 and 1.042**. The two [native
ORT traces](02_npu_ort_profile.json) and
[second trace](03_npu_ort_profile.json) independently record one VitisAI and
six CPU graph nodes on each placement warmup. Each [NPU worker
report](02_npu_worker.json) and [second worker report](03_npu_worker.json)
records 1,344 live graph calls for 21 × 64 tokens, with about 3.78 GB sampled
worker peak RSS. The hybrid's declared shared-RAM budget was 13.694 GB,
below contemporaneous Windows and WSL available RAM. The two phase reports
also identify the installed vLLM 0.28.0 wheel and Omni 0.29.0rc2 source;
this tested version pairing remains a caveat.

The [hardware snapshot](hardware.json) was captured after the run: Windows
11 build 26200, NPU driver 32.0.203.329 and Performance power scheme. Native
ORT was 1.30.0 with VitisAI EP 1.8.63.0. The BF16 checkpoint index SHA-256,
graph SHA-256, source calibration record, tokenizer and execution plans are
in the phase reports and [pinned source spec](../../../e2e_expansion_20260924/evidence/spark_amd_npu_live_split/spec_postnorm_profile20.json).
The checkpoint and graph weights remain outside Git at those pinned local
paths. CPU and NPU package power, NPU energy, temperature and thermal
frequency were not sampled, so this is a paired-order latency check rather
than a sustained power/thermal comparison. Four phase restarts also leave
cache effects; the order limits but does not eliminate them.

Reproduce with
[`profile_spark_cpu_npu_paired.py`](../../../../experiments/profile_spark_cpu_npu_paired.py)
from the repository root using the pinned BF16 model, source spec and
[reference CPU profile](../../../e2e_expansion_20260924/evidence/spark_bf16_wsl_cpu/profile20_branch_report.json),
and a fresh `--output-dir`. The script delegates to the two existing Omni
profilers, verifies every phase's token hash and real NPU placement, and
preserves reports and logs. Re-run
[`audit_spark_cpu_npu_paired.py`](../../../../experiments/audit_spark_cpu_npu_paired.py)
on this directory. An [earlier attempt](../spark_cpu_npu_paired/README.md)
was explicitly refused at NPU placement verification because the new harness
passed a WSL-only profiling path to the native Windows ORT worker; its raw
failure is retained and is not a graph-support conclusion.

```bash
PYTHONPATH=. VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  /home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python \
  benchmarks/edge_harness/experiments/profile_spark_cpu_npu_paired.py \
  --model /home/zhout/project/edge_infer/models/Spark-X2.5-1.7B-BF16 \
  --source-spec benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/spark_amd_npu_live_split/spec_postnorm_profile20.json \
  --reference-profile benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/spark_bf16_wsl_cpu/profile20_branch_report.json \
  --output-dir /tmp/spark_cpu_npu_paired_fresh
```

For this measured short request, keep the whole CPU route as the default.
The opt-in NPU split remains a scoped experimental quality-passing path;
broader prompts, long context, non-greedy behavior, concurrency, fault
recovery and sustained power/thermal behavior remain unqualified.
