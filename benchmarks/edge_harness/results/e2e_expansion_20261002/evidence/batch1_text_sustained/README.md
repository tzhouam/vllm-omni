# Spark-X2.5-4B desktop batch-1 sustained profile (2026-10-02)

**Depth: complete local text-request timing protocol, not release
qualification.** The real Spark-X2.5-4B BF16 checkpoint ran through
Omni/vLLM CUDA on the HX370 laptop's NVIDIA GeForce RTX 5090 Laptop GPU under
WSL2. A separate warmup and 20 measured requests completed in each of three
input lengths. Then **613 sequential medium requests ran for 1,801.720 s**
with batch size 1, one active request, 128 generated tokens per request and
no recorded output or metric failure. The read-only [analysis](analysis.json)
reports `timing_protocol_complete=true`, no violations and a 99.9566% active
request fraction during the sustained phase. The [report](report.json),
[raw per-request output and event records](requests.jsonl),
[1 Hz whole-device GPU telemetry](gpu_telemetry.jsonl) and
[Windows power snapshot during the run](power_snapshot_during.json) are
retained. The copied raw files were SHA-256 matched to the completed source.

The checkpoint is `/home/zhout/project/edge_infer/models/Spark-X2.5-4B`:
five dense BF16 safetensors shards totaling 8,224,192,408 bytes, with the
combined weight SHA-256
`5c91fc4a3664bc5744ef6cb654da8a792d52e104101874641eb8d6e7e932a826`.
The report pins config and tokenizer hashes and actual CUDA placement. The
plan used eager vLLM, a 4,096-token context, a 512-token prefill chunk,
754,974,720 bytes of budgeted KV cache, and one sequence. Its active stack
was WSL2 Linux 6.18.33.2, NVIDIA driver 610.71 at archive time, Python
3.12.13, PyTorch 2.13.0+cu130, vLLM 0.28.0 and Omni
`0.29.0rc2.dev71+g00510ce38.d20260922` loaded from the source checkout
`def6ebd00485dd587bbeb6a975ec053dd89d9988-dirty`. Startup took
43.610 s with existing disk/JIT caches and is excluded from request timing.
The contemporaneous Windows snapshot recorded the Performance power scheme,
battery status code 2 and 100% charge; no controlled alternative power
condition was run.

Nearest-rank p50/p95 from **20 measured complete requests per band**, excluding
warmups and sustained requests:

| Input tokens | Complete wall p50 / p95 | First token p50 / p95 | Decode rate p50 / p95 |
| ---: | ---: | ---: | ---: |
| 56 | 2.571 / 2.722 s | 0.034 / 0.046 s | 49.81 / 51.29 tokens/s |
| 488 | 2.750 / 3.011 s | 0.092 / 0.106 s | 47.26 / 48.90 tokens/s |
| 1,928 | 3.087 / 3.614 s | 0.377 / 0.402 s | 46.64 / 48.93 tokens/s |

The **613 sustained medium requests** had complete-wall p50/p95
2.962/3.128 s, first-token p50/p95 0.095/0.107 s and decode-rate p50/p95
44.31/48.63 tokens/s. The first 100 versus last 100 sustained requests had
complete-wall p50/p95 **2.783/3.101 versus 2.928/3.139 s**, and first-token
p50/p95 **0.091/0.104 versus 0.094/0.103 s**. These are observed cohorts,
not a controlled thermal or power-mode comparison. Within each fixed input
band the 20 measured greedy token sequences matched; all 613 sustained
medium requests also returned the same 128-token sequence. This is a
repeatability check on three named prompts, not a high-precision reference
or broad task-quality gate.

During the sustained window, 1,792 whole-device 1 Hz samples recorded GPU
board power **45.70–114.76 W** (median **84.80 W**), temperature **64–72 °C**
(median **67 °C**) and SM clock **645–2,235 MHz**. A 0.25 s memory sampler
observed up to **16,413,110,272 bytes** of whole-device GPU memory in use;
baseline-subtracted whole-card increase was **10,020,851,712 bytes**. The
declared admission peak budget was **12,553,385,449 bytes** and the request
ran without OOM. The sampled process-tree RSS peak was **5,226,020,864
bytes**. These counters have different accounting, cannot be added, and
shorter peaks may have been missed. WSL GPU-PV did not expose this engine's
per-process compute-memory attribution; neither power nor GPU memory can be
assigned exclusively to Omni from these counters.

The deliberately slow consumer completed with a bounded queue (high-water
4/4 chunks) and 128 tokens. An in-flight cancellation retired its epoch,
left zero in-flight requests and delivered zero remaining events; a fresh
128-token request then completed. This supplies the named recovery check,
but this timing run did not rerun a high-precision output comparison, broad
text-quality suite, longer context, fault-injection campaign or matched
unsplit-versus-staged route comparison. The WSL RTX Spark matrix cell remains
scoped and unqualified for release.

Reproduce from the repository root with a fresh output directory:

```bash
PYTHONPATH=$PWD:$PWD/packages/omni-stage-contracts \
  /home/zhout/project/edge_infer/.venvs/omni-cuda/bin/python \
  benchmarks/edge_harness/profile_local_text.py \
  --model /home/zhout/project/edge_infer/models/Spark-X2.5-4B \
  --out /tmp/omni_batch1_spark_sustained_new \
  --batch-size 1 --concurrency 1 --repeats 20 \
  --sustained-seconds 1800 --gpu-telemetry-interval-s 1

PYTHONPATH=benchmarks/edge_harness \
  /home/zhout/project/edge_infer/.venvs/omni-cuda/bin/python \
  benchmarks/edge_harness/analyze_text_sustained.py \
  /tmp/omni_batch1_spark_sustained_new > /tmp/omni_batch1_spark_sustained_new_analysis.json
```
