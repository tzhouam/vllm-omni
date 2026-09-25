# Spark-X2.5 BF16 RTX 5090 Laptop decoder + HX370 AMD NPU output head

**Disposition: scoped complete-request execution and greedy-token quality
passed; the NPU split is slower on this tested workload.** The real Spark
1.7B BF16 decoder, KV and sampler stayed on WSL RTX CUDA. An opt-in Omni
`ExternalStage` sent each post-final-norm single-token activation to the
native-Windows HX370 NPU A16W8 output-head graph, then returned the full
logits to CUDA. The resident BF16 head re-ranked the NPU top-64 candidate
tokens on CUDA; it did not silently run the full head on CPU. This retains
the existing vLLM decode loop and checkpoint, with no model or precision
substitution.

The [raw report](report.json) and [audit](audit.json) record one excluded
warmup plus **20/20 measured complete 64-token greedy text requests** per
phase, first RTX-only and then RTX+NPU, on the same 28-token inventory prompt.
Every request in both phases had the same pinned token-ID SHA-256
`a8d7798bf439de0b9cec3ba65661bee9e06df2a8af3f8c0d46ad7957c38e4162`.
The EOS was ignored and concurrency was one. These nearest-rank values include
whole-request Omni handoff and generation, excluding stage startup:

| Phase | Complete requests | p50 / p95 wall | Median TTFT | Same 64 tokens |
|---|---:|---:|---:|---:|
| RTX-only | 20/20 | 0.728 / 0.779 s | 26.9 ms | 20/20 |
| RTX+AMD NPU | 20/20 | 1.438 / 1.465 s | 48.0 ms | 20/20 |

The joint p50 was **1.98× the RTX-only p50** in this CUDA-then-joint run.
That is evidence against selecting the split by default for this short
fixture; it is not a paired-order or power-conditioned speedup study. The
RTX-only and joint load times were 16.10 and 41.71 s, respectively. The NPU
session itself took 18.03 s to create. The pinned [NPU worker](npu_worker.json)
recorded 1,346 projection/refinement calls: 1,344 generated tokens across
21 requests and two extra initialization calls. All refinement records name
`cuda:0`. Its [retained raw ORT trace](npu_raw_profile.json) has one VitisAI
and six CPU graph-node events; the same open session served live tokens.

The [as-run spec](npu_spec.json) pins the A16W8 graph SHA-256
`0e711ed95d64d1f929117a2f90425b0a92702b1bff6d4945cec56bd21f2a602a`
and the checkpoint index SHA-256 is
`cc2b212985f5d0469bf926903e4bd7ee81856687baa0c6c6b55ff200d4cdc63f`.
The graph remains at its recorded local Windows path; the earlier
[CPU+NPU artifact record](../spark_cpu_npu_paired_verified/README.md)
documents its calibration and numerical history. The CUDA session plan
reserved 8.20 GB VRAM; the NPU stage reserved 5.49 GB shared RAM plus an
explicit 2 GiB host-process allowance. Windows and WSL available RAM both
exceeded that combined host budget at admission. The native NPU worker's
sampled peak RSS was 3.78 GB, below its reservation. GPU memory sampling is
a lower bound and host/GPU loading peaks and power were not jointly measured.

WSL Ubuntu 26.04 ran PyTorch 2.13.0+cu130, vLLM 0.29.0, this Omni checkout,
and NVIDIA driver 610.71. Windows 11 build 26200 ran AMD NPU driver
32.0.203.329, ORT 1.30.0 and VitisAI EP 1.8.63.0. The two stage shutdowns
and the full driver log are [retained](driver.log). An
[earlier harness attempt](../spark_cuda_amd_npu_joint/README.md) completed
both phases but wrongly treated its two initialization calls as a runtime
failure; the corrected run here is independent evidence.

The changed adapter also passed a fresh [CPU+NPU regression request](cpu_regression_report.json):
64/64 greedy tokens matched the existing CPU reference, with 64 VitisAI
calls and 64 CPU re-ranking records. Its [worker](cpu_regression_worker.json)
and [raw trace](cpu_regression_raw_profile.json) are retained and checked by
the same audit. This checks that adding CUDA tensors did not change the
earlier CPU+NPU route on the fixed prompt.

Reproduce from the repository root with
[`profile_spark_cuda_npu_joint.py`](../../../../experiments/profile_spark_cuda_npu_joint.py),
the pinned BF16 checkpoint and [source spec](../../../e2e_expansion_20260924/evidence/spark_amd_npu_live_split/spec_postnorm_profile20.json).
Use the `omni-cuda-029` environment with `PYTHONPATH` set to this checkout,
`VLLM_TARGET_DEVICE=cuda`, `CUDA_VISIBLE_DEVICES=0`,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, and a fresh output directory. Re-run
[`audit_spark_cuda_npu_joint.py`](../../../../experiments/audit_spark_cuda_npu_joint.py)
on the saved evidence and model directory. The raw trace is copied from the
as-run Windows profile path into this record.

Long context, non-greedy sampling, broader prompt quality, concurrent
admission, cancellation/recovery and sustained power/thermal behavior remain
open. The default remains an unsplit RTX session under the whole-chain
benefit rule.
