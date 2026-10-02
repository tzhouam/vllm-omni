# Spark CUDA batch-1, 20-request-per-band profile (2026-10-02)

**Depth: scoped complete text-request timing, not release qualification.**
The real Spark-X2.5-4B BF16 checkpoint ran through Omni/vLLM CUDA on the
HX370 workstation's RTX 5090 Laptop GPU under WSL2. The [report](report.json)
pins the exact combined weight SHA-256
`5c91fc4a3664bc5744ef6cb654da8a792d52e104101874641eb8d6e7e932a826`,
config hash, loaded package versions, plan, observed placement and startup.
The run used batch size 1, one active request, greedy 128-token outputs, one
warmup and 20 measured requests in each length band. Every measured request
finished, and each band's 20 token sequences were identical. Raw
[per-request events](requests.jsonl) retain the samples.

| Band | Actual input tokens | Measured requests | Complete wall p50 / p95 | TTFT p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| Short | 56 | 20 | 2.498 / 2.563 s | 0.035 / 0.044 s |
| Medium | 488 | 20 | 2.735 / 2.850 s | 0.088 / 0.103 s |
| Long | 1928 | 20 | 2.969 / 3.027 s | 0.358 / 0.371 s |

Percentiles are nearest-rank statistics over the 20 measured requests per
band, excluding warmups, startup and recovery probes. Engine startup was
21.29 s. Slow-consumer and post-cancel requests completed; cancellation
returned zero remaining events. The separate [1 Hz NVML samples](nvml_1s.csv)
and [contemporaneous Windows power snapshot](power_condition.json) document
the condition. Across the first-to-last measured-request envelope, 167 NVML
samples (including six between-band warmup samples) showed GPU
board power **53.61–100.37 W** (median 88.23 W), temperature **58–73 °C**
and SM clock **727–1822 MHz**; these are device-wide readings, not whole-laptop
energy or per-process attribution. Battery status was code 2 at 100% charge
under the Windows Performance scheme. The WSL driver reported 610.71 and no
power-limit value.

The profiler deliberately disabled the sustained phase (`--sustained-seconds 0`).
The summarizer therefore marks its 20-per-band and required-metrics
checks true but its 30-minute and overall batch-1 protocol checks false. No
new high-precision quality comparison, broad prompts or image path was run.
The existing CUDA matrix cell stays **scoped and unqualified**; these data do
not promote a new backend placement or establish a thermal limit.

Reproduce with a fresh output directory and record the active power condition:

```bash
PYTHONPATH=$PWD:$PWD/packages/omni-stage-contracts \
  /home/zhout/project/edge_infer/.venvs/omni-cuda/bin/python \
  benchmarks/edge_harness/profile_local_text.py \
  --model /home/zhout/project/edge_infer/models/Spark-X2.5-4B \
  --out /tmp/omni_batch1_text_profile20_new \
  --batch-size 1 --concurrency 1 --repeats 20 --sustained-seconds 0
```
