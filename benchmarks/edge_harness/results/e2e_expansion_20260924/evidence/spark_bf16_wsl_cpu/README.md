# Spark-X2.5 1.7B BF16 on WSL CPU

The complete upstream BF16 checkpoint, previously missing its two weight
shards, was downloaded outside Git from
`XHToken/Spark-X2.5-1.7B` at revision
`14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8`. The
[weight manifest](weight_manifest.json) verifies both upstream LFS SHA256
digests, file sizes and every tensor name in the checkpoint index. This is a
separate artifact from the previously tested INT8 and GGUF variants.

On the Ryzen AI 9 HX 370 WSL CPU, the current Omni local text path admitted
the dense BF16 artifact with a 4,096-token context and an explicit 7.64 GiB
host-RAM budget. The [fork-branch acceptance run](accept_branch_report.json) completed all 12
named prompts with 128 output tokens each, reported actual CPU placement, and
retired an in-flight cancelled request without a stale event. The separate
[fork-branch parity run](parity_branch_report.json) compared the Omni path against standalone
vLLM on the same checkpoint and found exact greedy token-ID agreement for
12/12 prompts at 128 output tokens. The installed vLLM was 0.28.0 while the
tested fork-branch Omni package identified as 0.29.0rc2.dev71, so that
version pairing and its compatibility warning must be retained with the result.
The branch acceptance run observed a 30.36 s load and approximately 6.45 GiB
process-tree RSS load delta; neither is a cold-start guarantee or an
attribution to model weights alone.
The [branch run manifest](branch_run_manifest.json) pins the imported runtime,
source-file hashes and all three raw report hashes.

The [fork-branch 20-request complete-text profile](profile20_branch_report.json) used one warmup,
then 20 serial requests with the same 28-token prompt and 64 generated tokens
per request. Nearest-rank p50/p95 complete-request wall time was **5.411/5.751 s**;
time to first token was **0.129/0.146 s**. These are measurements of the admitted
fork-branch Omni BF16 CPU route, not mobile estimates or sustained thermal results. The
[profiler](../../../../experiments/profile_spark_bf16_omni_cpu.py) retains all
20 raw request samples and runtime provenance. A separately retained
[acceptance](accept_report.json), [parity](parity_report.json) and
[20-request profile](profile20_report.json) came from a neighboring Omni
checkout (`3834de310`, package `0.28.1.dev154`) with the same vLLM 0.28.0
and checkpoint. Its 4.808/5.089 s wall p50/p95 and 19.88 s acceptance load
are distinct run observations, not the fork-branch timing.

A separate [fork-branch restart probe](restart_branch_report.json) verified
the state lifecycle after cancelling generation at eight token events. The
backend reported zero in-flight requests and delivered no retired-epoch
events. Reusing the old session handle was explicitly refused as stale; a
fresh session then produced the **exact same 128 greedy token IDs** as the
pre-cancel request. The [probe script](../../../../experiments/verify_spark_bf16_restart.py)
records the two token-sequence hashes, state refusal and runtime provenance.
This covers one prompt and one cancellation depth, not concurrent recovery.

Reproduce from the repository root with the installed CPU environment, after
downloading the pinned checkpoint into the path below:

```bash
export PYTHONPATH=/home/zhout/project/edge_infer/vllm-omni-edge
/home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python -m vllm_omni.edge.local accept \
  --model /home/zhout/project/edge_infer/models/Spark-X2.5-1.7B-BF16 \
  --max-model-len 4096 --max-tokens 128 --json /tmp/spark_bf16_accept_repeat.json
/home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python -m vllm_omni.edge.local parity \
  --model /home/zhout/project/edge_infer/models/Spark-X2.5-1.7B-BF16 \
  --max-model-len 4096 --max-tokens 128 --json /tmp/spark_bf16_parity_repeat.json
/home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python \
  benchmarks/edge_harness/experiments/profile_spark_bf16_omni_cpu.py \
  --model /home/zhout/project/edge_infer/models/Spark-X2.5-1.7B-BF16 \
  --report /tmp/spark_bf16_profile_repeat.json
/home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python \
  benchmarks/edge_harness/experiments/verify_spark_bf16_restart.py \
  --model /home/zhout/project/edge_infer/models/Spark-X2.5-1.7B-BF16 \
  --report /tmp/spark_bf16_restart_repeat.json
```

This validates one complete text path and cancellation on WSL CPU. It does
not establish native Windows BF16 execution, mobile generation, broader text
quality, sustained load, or an incremental GGUF token-stream contract.
