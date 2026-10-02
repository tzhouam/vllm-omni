# Batch-1 Spark CUDA profiler smoke (2026-10-02)

**Diagnostic run only; it does not pass the qualification protocol.** The
current `profile_local_text.py` completed its three serial input bands and
slow-consumer/cancel/recovery path through Omni with the real Spark-X2.5-4B
BF16 checkpoint on the HX370 workstation’s RTX 5090 Laptop GPU under WSL.
The pinned plan set `max_num_seqs=1`; every recorded request declares batch
size 1 and concurrency 1. [Raw report](report.json) and
[request records](requests.jsonl) preserve the actual checkpoint, runtime,
placement and stream observations.

| Input band | Actual prompt tokens | Output tokens | One measured request wall | TTFT |
| --- | ---: | ---: | ---: | ---: |
| Short | 56 | 128 | 2.545 s | 0.034 s |
| Medium | 488 | 128 | 2.690 s | 0.083 s |
| Long | 1928 | 128 | 2.896 s | 0.358 s |

There was one warmup and **one measured request per band**, not the required
20. The sustained phase was disabled, and no power/thermal profile or new
high-precision quality comparison was run. The cancellation left zero
remaining events and a subsequent 128-token request completed. The GPU load
delta is a whole-card NVML observation; WSL did not expose per-process NVML
attribution. The existing Spark CUDA cell therefore stays scoped and
unqualified. These single observations are not p50/p95 estimates or a route
comparison.

Reproduce with a fresh output directory:

```bash
PYTHONPATH=$PWD:$PWD/packages/omni-stage-contracts \
  /home/zhout/project/edge_infer/.venvs/omni-cuda/bin/python \
  benchmarks/edge_harness/profile_local_text.py \
  --model /home/zhout/project/edge_infer/models/Spark-X2.5-4B \
  --out /tmp/omni_batch1_text_smoke_new \
  --batch-size 1 --concurrency 1 --repeats 1 --sustained-seconds 0
```
