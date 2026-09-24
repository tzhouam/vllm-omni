# Spark live CPU decoder + AMD NPU output-head split

**Depth P for a narrow 28-prompt-token / 64-output-token text request, with failed exact-token parity; not a qualified split.** vLLM retained the BF16 Spark-X2.5-1.7B decoder, KV cache and sampler. The current Omni source routed each live pre-final-norm token activation through an admitted `ExternalStage` to the A16W8 RMS-norm plus full output-projection graph on native Windows VitisAI. No CPU fallback or model substitution was enabled. [The request profile](profile20.json) and [worker report](worker_profile20.json) contain the per-request and per-token measurements.

The HX370 laptop ran WSL2 Ubuntu for vLLM/Omni and Windows 11 build 26200 for ORT 1.30.0, VitisAI EP 1.8.63.0 and AMD NPU driver 32.0.203.329. The Omni checkout was 0.29.0rc2 source with installed vLLM 0.28.0 and PyTorch 2.13.0+cpu; this version mismatch remains a runtime caveat. The BF16 checkpoint is `XHToken/Spark-X2.5-1.7B-BF16` revision `14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8`. Its norm and tied projection weights were verified bitwise equal after BF16-to-FP32 conversion to those in the graph's FP32 source; [the identity record](../spark_amd_npu_wsl_handoff/weight_identity.json) narrows the observed output difference to execution/quantization, not an unrelated head checkpoint. The A16W8 graph SHA-256 is `e2ae9c3fd7071e8ff628f209e6140c8a023bfaad59814b80a936c6b263f33eaf`; the retained real-activation fixture SHA-256 is `cfa2ed8439f8fe01730319667bfd01abf2e39cf5851490346a0331304cf63194`. The graph remains in local NTFS artifact storage, as recorded in [the experiment spec](spec_profile20.json).

The joint CPU and NPU host-RAM plans reserved 13.694 GB before loading, below the observed Windows and WSL available memory. One warmup and **20/20** serial complete requests ran at concurrency one with the same 28-token prompt, 64 greedy output tokens, `ignore_eos=True` and 4096-token context. The nearest-rank p50/p95 whole-request wall time was **5.012/5.393 s**, and TTFT was **0.125/0.133 s**. All 21 requests produced the same token IDs. The NPU worker reported **1,344 live graph calls** (21×64), with a 3.772 GB peak RSS. Per-token graph-stage round trip across WSL/Windows was 9.475/9.908 ms p50/p95; worker inference was 8.046/8.283 ms. The [ORT placement profile](placement_profile.json) attributes one fused projection partition to VitisAI and seven norm/quantization/concatenation nodes to CPU. The CPU process-tree sampled peak was 7.764 GB RSS, including its pre-load baseline; this is not additive to the Windows worker RSS or the shared-pool plan.

**Exact token parity failed.** The previous unsplit BF16 CPU run yielded a different 64-token hash, and this split first diverged at zero-based output index 22 (CPU token 275, NPU token 518). In a separate [64-step diagnostic request](probe_diag64.json), the resident CPU head was evaluated on each *same live activation* for measurement only while sampling still used NPU logits. It agreed with the NPU top-1 on 63/64 steps; the sole disagreement was index 22. CPU BF16 logits tied tokens 275 and 518 at -2.640625, selecting 275, while A16W8 NPU logits selected 518 at -2.5864 versus -2.6897 for 275. The diagnostic request reproduced the same split token sequence; [raw comparisons](worker_diag64.json) have all logits-error and top-1 rows. This is a quantization-sensitive tie, not proof of broad model quality. A separate unsplit CPU profile measured 5.411/5.751 s p50/p95 on the same prompt, but the runs were not paired for power, cache or thermal state, so their difference does not establish a beneficial split. The diagnostic run is excluded from the timing profile because CPU reference-head execution changes its cost.

The split is still experimental: cancellation propagation during an NPU call, recovery, long context, broader prompts, numerical/task tolerance, power and thermal behavior have not been validated. The normal Spark route is unchanged when `VLLM_OMNI_SPARK_EXTERNAL_HEAD_SPEC` is unset. The next gate is a quality-passing recalibration or artifact and paired whole-request testing against the unsplit backend; only then should the split become an eligible execution plan.

Reproduce the 20-request profile on this host with the retained local graph and checkpoint:

```bash
PYTHONPATH=. /home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python \
  benchmarks/edge_harness/experiments/probe_spark_live_npu_head.py \
  --model /home/zhout/project/edge_infer/models/Spark-X2.5-1.7B-BF16 \
  --spec benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/spark_amd_npu_live_split/spec_profile20.json \
  --reference-profile benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/spark_bf16_wsl_cpu/profile20_branch_report.json \
  --report benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/spark_amd_npu_live_split/profile20.json \
  --max-tokens 64 --warmup-requests 1 --measured-requests 20
```

## Broader acceptance, recalibration, and explicit CPU candidate re-rank

The original four-activation A16W8 calibration did not generalize to the
12-prompt BF16 text acceptance set. All 12 requests completed 128 output
tokens, but only **4/12** matched the unsplit CPU token sequences. The raw
[request report](accept12.json) says `failed` because the first harness version
required exactly 1,536 NPU calls; the [worker report](worker_accept12.json)
records 1,537 calls and all 12 request records are complete. The additional
call occurs in the long-prompt run. The corrected harness checks at least one
NPU call per generated token and retains the original raw report. A focused
[`plain_6` capture](plain6.json) exposed live activations reaching 35–36,
versus a maximum of 4.178 in the original four calibration inputs. The old
QDQ graph reproduced its bad logits on ORT CPU; VitisAI and ORT CPU agreed
closely on the *same* graph. This isolated the principal failure to the tested
calibration, rather than proving a general NPU execution defect.

Two pinned per-channel W8 candidates were built from the bitwise-verified
source head. [Calibration v1](recalibration_manifest.json) combined the four
original samples with six live `plain_6` activations. Its graph SHA-256 is
`497e8974d3e1faa6f66715dfa692172bdb304f1de0fffe12a09fd91afdafdd2e`.
The [v1 12-prompt run](accept12_per_channel.json) matched **7/12** exact BF16
sequences; the old graph had matched 4/12. A second
[48-sample calibration](recalibration_v2_manifest.json) added live activation
extremes and divergence-neighbor steps from the 12-prompt capture. Its graph
SHA-256 is
`88321b607a6def7f2a76824009a322f1bfa9f60f34c2b93344c3de498d9d6072`.
The [65-activation ORT CPU check](recalibration_v2_validation.json) reduced
the worst source-relative L2 from 2.597% for v1 to 0.259% for v2 on that
selection, while four BF16 top-token mismatches remained. The
[VitisAI-versus-QDQ replay](recalibration_v2_provider_vs_qdq.json) verified
one NPU and seven CPU graph nodes on v2, with same-graph numerical checks.
The [v2 live 12-prompt run](accept12_v2.json) matched **8/12** exact CPU
sequences. Same-live-activation NPU versus resident BF16 head top-1 matched
1,532/1,536 measured output steps; logits relative L2 nearest-rank p50/p95
was 0.181%/0.208%, maximum 0.865%. The BF16 winner was within NPU top-64
on all 1,536 measured steps. These 12 prompts overlap the calibration set,
so this is not a held-out quality result.

An **explicit greedy-only hybrid** now treats the v2 NPU head as a 64-token
candidate retriever. The already resident BF16 CPU head re-scores only those
64 weight rows, and the sampler receives those scores with all other token
logits masked. This CPU work is recorded in
[the worker report](worker_accept12_v2_refine64.json), rather than being
reported as NPU-only execution. The [12-prompt hybrid run](accept12_v2_refine64.json)
matched **12/12** unsplit BF16 token sequences, 128 tokens each. On all 1,536
measured output steps, the BF16 top token was in the NPU top-64 set, sparse
BF16 scores equaled full-head BF16 scores exactly, and the re-rank changed six
NPU top choices. There were 1,537 worker calls because the long prompt added
one non-output call. The [separate 20-request profile](profile20_v2_refine64.json)
passed 20/20 repeated 64-token complete requests after one warmup, at
nearest-rank whole-request wall p50/p95 **5.356/5.580 s** and TTFT
**0.131/0.151 s**. Its graph-stage round trip was 9.768/10.190 ms p50/p95;
the sparse CPU re-rank was 1.108/1.747 ms p50/p95 per call. The 13.694 GB
joint CPU/NPU shared-RAM budget was admitted; the worker peak RSS was
3.780 GB. The independent BF16 CPU 20-request profile was 5.411/5.751 s,
but the runs were not paired for power, cache, or thermal state, so no
speedup is established. The timed run did not execute the full CPU comparison
head on each token.

A separate [hybrid cancellation/restart probe](restart_v2_refine64.json)
completed a 128-token greedy request, cancelled a longer request after eight
observed token events, and completed the same 128-token sequence with a fresh
session handle in the same engine. The backend abort and zero in-flight count
were recorded; the retired epoch delivered no late output and the stale
handle was rejected. The [NPU worker report](worker_restart_v2_refine64.json)
records 266 graph and CPU re-rank calls with one VitisAI plus seven CPU nodes.
The cancellation occurred after tokens had started streaming; it does not
isolate a cancellation precisely during an NPU graph call or test recovery
from a crashed NPU worker.

Six [prompts outside the calibration set](heldout_prompts.json) were run first
on unsplit BF16 CPU, then through the hybrid under explicit joint admission.
The [paired request record](heldout_v2_refine64.json) matched **5/6** complete
token sequences. On the hybrid trajectory, NPU top-64 contained the resident
BF16 winner, and sparse scoring matched the full resident BF16 head on all
768 measured steps. `heldout_weather` diverged from the separate CPU run at
zero-based output index 25. Three independent unsplit CPU repeats of that
prompt had the same original token sequence
([raw repeat](heldout_weather_cpu_repeat.json)); the cause of the hybrid's
different activation or token choice is not yet isolated. This is a real
held-out failure, so the hybrid remains **scoped experimental E2E**, not a
qualified general Spark CPU+NPU route. It is also not a stochastic-sampling
implementation: masking to the NPU top-64 is intended only for greedy
decoding. Mid-graph cancellation, NPU-worker failure recovery, broader
quality/context, and sustained power/thermal behavior remain open.

The calibration input arrays and captured live activations are retained in
this evidence directory with hashes in their manifests and worker reports.
The large ONNX graphs remain in local NTFS artifact storage at the paths in
the [v2 experiment spec](spec_accept12_v2_refine64.json); this repository
records their SHA-256 digests and the
[recalibration](../../../../experiments/recalibrate_spark_amd_npu_head.py),
[same-graph provider check](../../../../experiments/compare_spark_npu_qdq_provider.py),
[CPU numeric check](../../../../experiments/validate_spark_recalibration.py),
and [held-out request probe](../../../../experiments/probe_spark_heldout_hybrid.py).

## After-normalization boundary and independent-prompt recovery

The preceding hybrid deferred Spark's final norm and sent the pre-norm sum to
an NPU graph containing its own norm. Its `heldout_weather` output differed
from three stable unsplit BF16 CPU repeats despite exact sparse/full CPU
scores *on the hybrid's own activations*. A second, explicit stage boundary
now leaves vLLM's original final RMSNorm in place and sends its normalized
single-token tensor to a four-shard A16W8 NPU projection. The resident BF16
CPU head still re-ranks the NPU top-64 candidates for **greedy decoding
only**. This preserves the exact vLLM decoder, norm, KV, sampler and state
path. Both the NPU projection and CPU re-rank are reported; this is a joint
CPU+NPU route, not an NPU-only output head. Moving the boundary and replacing
the artifact happened together, so these experiments do not isolate which
change caused the earlier weather mismatch.

The historical four-sample normalized-input graph failed a live
[weather control](postnorm_weather.json): its NPU top-64 contained the BF16
winner on only 3/128 steps. The unsplit BF16 CPU run then
[captured](postnorm_cpu_capture.json) **1,665** exact post-final-norm
activations across the 12 acceptance prompts and weather (13×128 output
steps plus one extra long-prompt graph input). Its 12 acceptance token hashes
matched the earlier CPU reference, and weather matched three stable CPU
repeats. The captured tensor range was −41.5 to 19.375, unlike the small
original calibration. The retained [activation array](postnorm_cpu_capture.npz)
has SHA-256 `f76dfc221f52ef5c4f3cb23fa2f51f51c8fd05aad6d4ed32da178cfb22b864c4`.

A [58-sample calibration](postnorm_calibration_manifest.json) of the pinned
four-shard normalized-input source produced per-channel A16W8 graph SHA-256
`0e711ed95d64d1f929117a2f90425b0a92702b1bff6d4945cec56bd21f2a602a`.
The [quantization report](postnorm_recalibration_report.json) pins its source
and input hashes. A [66-activation ORT CPU check](postnorm_cpu_validation.json)
matched the FP32 source top token on all 66 selected inputs; source-relative
logits L2 had median 0.086% and maximum 0.164%, compared with 0.647% median
and 6.881% maximum for the original graph on the same inputs. A separate
[VitisAI-versus-ORT CPU replay](postnorm_provider_vs_qdq.json) matched top-1
on 7/7 sampled real activations, with maximum same-graph relative L2 below
3.5e-6. ORT profiling verified one VitisAI partition and six CPU graph nodes.

The rebuilt route passed the formerly failing
[paired weather request](postnorm_weather_recal.json), matching the unsplit
BF16 CPU sequence for all 128 tokens. Its
[12-prompt live acceptance](postnorm_accept12.json) matched **12/12** exact
128-token CPU sequences. Across 1,536 measured output steps, the BF16 winner
was in the NPU top-64 candidates, sparse BF16 scoring equaled the full
resident BF16 head, and the refined top token equaled that full head on every
step. The worker made 1,537 graph/re-rank calls, including one non-output
long-prompt call. Because these 12 prompts contributed calibration samples,
they are not a held-out task-quality result. Six
[independent prompts](postnorm_heldout_prompts.json) excluded from this
calibration passed [paired unsplit-CPU versus hybrid requests](postnorm_heldout.json)
at **6/6** exact 128-token sequences and 768/768 top-64 recall and exact
sparse/full CPU head scores. The single weather prompt was part of
calibration and is reported separately, not counted in that 6/6 figure.

The [separate timing run](postnorm_profile20.json) completed one warmup plus
20/20 repeated 28-prompt-token/64-output-token requests with the same token
sequence as the BF16 CPU reference. Nearest-rank whole-request wall p50/p95
was **4.984/5.075 s**, TTFT **0.126/0.133 s**, NPU graph-stage round trip
**9.426/9.886 ms** per call, and sparse BF16 CPU re-rank **1.161/1.544 ms**
per call. The plans jointly reserved **13.694 GB** of shared host RAM before
loading; the NPU worker peak RSS was **3.781 GB**. The separate unsplit BF16
CPU 20-request profile was 5.411/5.751 s, but these runs were not paired for
cache, power or thermal state, so no speedup claim follows. The timed hybrid
run did not execute the diagnostic full CPU head per token. A final
[same-engine cancellation/restart](postnorm_restart.json) passed backend
abort after eight streamed token events, no late events, stale-handle
rejection, and an identical 128-token request through a fresh session with
the NPU stage still active.

This is **scoped E2E P evidence for Spark text on this HX370 CPU+AMD NPU
configuration**. It does not qualify general text quality, stochastic
sampling, concurrent traffic, mid-NPU-call cancellation, recovery from an
NPU-worker crash, longer context transitions, sustained power/thermal
behavior or a beneficial split versus the unsplit backend. The full graph
remains in local NTFS storage at the path and SHA-256 in
[the final experiment spec](spec_postnorm_accept12.json); source and
calibration fixtures plus the CPU capture are retained with hashes. The
one-step [stage-open input fixture](postnorm_stage_input.npz) is also retained
with the same SHA-256 as the NTFS input named in the spec.
