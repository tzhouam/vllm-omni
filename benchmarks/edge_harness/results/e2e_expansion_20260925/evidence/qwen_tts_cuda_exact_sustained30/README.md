# Qwen3-TTS exact-state CUDA under a 33 W GPU cap

**Failed real-time qualification, deliberately interrupted after 32 completed
sustained requests.** The last request ended 365 s after phase start; harness
shutdown ended at 374 s, short of the planned 1,800 s. This is the same
two-stage `AsyncOmni`
CustomVoice 0.6B checkpoint and exact x-vector sliding-KV decoder used by the
earlier [20-request medium pass](../qwen_tts_cuda_exact_live20/README.md), on
the HX370 + RTX 5090 Laptop under WSL2. The checkpoint revision is
`85e237c12c027371202489a0ec509ded67b5e4b5`. The [raw report](report.json)
pins the runtime, source path, launch arguments and sampled memory. The run
used existing disk/JIT caches, one medium warmup, 20 serial measured medium
requests and then a planned 30-minute medium-request loop at concurrency one.
The stream remained finite, ordered and complete for all 53 retained requests
including warmup: 24 kHz PCM and exactly one terminal event per request.

The [raw request records](requests.jsonl) and [derived audit](partial_analysis.json)
show the following nearest-rank results. Playback deficits are simulated from
audio-chunk arrival times; no sound device was used.

| Phase | Complete requests | Wall p50/p95 | TTFA p50/p95 | RTF p50/p95 | Requests with simulated deficit |
|---|---:|---:|---:|---:|---:|
| Measured | 20 | 11.287/11.809 s | 447/473 ms | 1.418/1.434 | 20/20 |
| Interrupted sustained | 32 | 11.150/12.839 s | 448/481 ms | 1.435/1.448 | 32/32 |

After persistent RTF above one and deficits on every complete request, the
operator sent SIGINT to the owned benchmark process. The in-flight request was
aborted and both Omni stages shut down; the harness `finally` block saved an
end time and telemetry status. `report.json` retains `status=running` and
`sustained_wall_s=0` because the planned loop did not finish. The derived
audit labels this interruption explicitly; **it is not a 30-minute pass**.
The first 20 requests also contained inference-time Triton JIT compilation,
so this run is not an isolated steady-state latency baseline.

The contemporaneous [GPU power-state record](gpu_power_state_during_sustained.txt)
reports a **33 W current software power cap**, 95 W default, and no active
hardware thermal slowdown. [Windows power state](windows_power_state_during_sustained.json)
reports the Performance power scheme, battery status 2 and 100% charge; it
does not reveal why firmware or software applied the GPU cap. Across 1,463
device-wide [GPU samples](gpu_telemetry.jsonl) during the interrupted phase,
nearest-rank p50 power was 32.91 W, SM clock 900 MHz, memory clock 810 MHz,
utilization 95%, and temperature 58 °C. The telemetry includes other GPU
activity and cannot isolate engine energy. An attempt to set the 95 W default
through `nvidia-smi -pl` was reported as unsupported on both WSL and Windows;
the cap was not changed.

The earlier exact-state medium run measured 1.542 s wall p50, 104.9 ms TTFA
p50 and 0.194 RTF p50, with 83.39 W device-wide GPU power p50 during its
short measured window. The different power condition, possible other GPU
activity and inference-time JIT mean these are **not paired runs**; the new
failure does not establish a source-code regression. A full-power repeat and
late-abort audio fence remain necessary before claiming sustained playable
streaming. The HX370 AMD NPU was not involved.

The partial analysis is reproducible without rerunning the model:

```bash
PY=/home/zhout/project/edge_infer/.venvs/omni-cuda-029/bin/python
$PY benchmarks/edge_harness/experiments/analyze_tts_power_limited_run.py \
  --run-dir benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_cuda_exact_sustained30 \
  --out /tmp/qwen_tts_power_limited_analysis.json
```
