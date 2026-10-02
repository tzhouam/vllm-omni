# Local model qualification and profiling

These runners exercise existing Omni implementations with real weights. They
do not add device or model support. A completed timing run is distinct from
reference-quality, streaming, fault-recovery and release qualification.
**The current qualification protocol measures batch size 1 and one active
request only.** Historical results at concurrency 2/4 remain archived as
observations, but they do not enter current route selection or qualification.

## Matrix and prerequisites

The [60-cell audit](results/model_device_matrix_20260922/README.md) covers twelve
PC/mobile/embedded configurations and five model families. Its raw evidence
distinguishes complete pipelines from components and backend-only probes.
Unavailable complete pipelines retain a reason and next repair step; a missing
measurement is never represented as zero latency.
That audit and the five-run summarizer below preserve the original 2026-09-22
baseline. The [rolling 60-cell matrix and qualification ledger](results/e2e_profiling_20260922/evidence/summary/README.md)
carry later evidence and are the current coverage source.

Use the actual installed runtime and exact checkpoint identified in the result,
including quantization, rather than substituting another family member. The
current Spark CPU runner uses 1.7B INT8 and the CUDA runner uses 4B BF16. Their
results do not establish same-model acceleration. Qwen3-TTS uses the pinned
0.6B CustomVoice checkpoint, English/Vivian, through both Omni stages.

Windows and WSL are distinct configurations on the same physical laptop.
Run benchmarks serially: they share CPU, physical RAM, GPU and power limits.
GPU board power is not whole-device energy. Windows used RAM includes WSL;
do not add Windows and WSL counters as independent memory pools.

## Run the workload

Run from the repository root using the model's validated Python environment:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  python benchmarks/edge_harness/profile_local_text.py \
  --model /absolute/path/to/checkpoint --out /absolute/path/to/new-run

PYTHONPATH=. HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  python benchmarks/edge_harness/profile_local_tts.py \
  --model /absolute/path/to/customvoice-checkpoint --out /absolute/path/to/new-run
```

Set the validated CPU thread/KV-cache settings or CUDA compiler environment
before launching. Native Windows requires UTF-8 Python mode before interpreter
startup (`PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`). The common benchmark helper
uses the active environment's `torch/lib/cudart64_13.dll` when present and a
short `~/c29` vLLM cache path unless explicitly overridden. These reproduce
settings required by the tested Windows installation; they are not a general
claim about other CUDA/runtime versions.

Both runners require a new output directory and enforce:

- Three input lengths × batch size 1 × concurrency 1 × 20 measured requests per length.
- Separate warmup requests, excluded from group percentiles.
- At least 1800 seconds of sequential medium-input requests.
- Slow-consumer, cancellation and subsequent-request checks.

The runners record `profile_protocol.name=single_request_batch1_v1` and
`batch_size=1, concurrency=1` in each timed request. The CLI rejects any
other batch size or concurrency. `max_num_batched_tokens` is a token budget,
not a request batch-size setting. A one-band or shortened instrumentation run
is useful diagnostically but cannot satisfy the full timing protocol.

Text timing forces 128 output tokens with greedy sampling and records actual
input lengths, token IDs, sequence/epoch metadata and queue high-water marks.
It does not recommend those sampling settings for applications. TTS requires
DELTA output, records every audio chunk including terminal audio, verifies
finite PCM, and archives warmup float32 chunks with hashes. Playback underruns
are simulated from chunk arrivals; no sound-device playback or perceptual
quality is implied.

`profile_host_command.py` wraps an explicit command on Windows to collect
one-second host-memory, AC/battery and whole-NVIDIA-device telemetry. Each host
directory retains the exact command, exit status and combined log. Existing
disk/JIT caches are used. Constructor time is not a cold-disk measurement and
must remain separate from request latency. Sampled memory peaks are lower
bounds, not proof that every transient allocation was observed.

On Windows, a live reader can deny replacement of a report snapshot. The
writer defers such running updates, records the permission error, and requires
the final snapshot to persist. Request JSONL is the primary sample record.
Read live reports through Linux when they reside in WSL, or use a Windows
reader that permits file deletion/replacement. Failed profile/trace reports
produce a nonzero exit code in the current runners; historical attempts may
require inspection of the report rather than relying on the process exit code.

## Separate diagnostics

After timing runs finish, use `profile_trace_diagnostic.py --kind spark|tts`
with the same `--model` and a fresh `--out`. It invokes Omni's profiler RPCs and
measures three requests before, during and after tracing. Current text traces
use 32 generated tokens and disable expensive shape/memory instrumentation.
Compare paired diagnostic requests, not these 32-token times against the
128-token baseline. Profiler start/stop/export costs are recorded separately
when supported. The initial archived CPU diagnostic used full instrumentation
and 128 tokens; its configuration and limitations are retained in its report.

Run `check_text_reliability.py` or `check_tts_reliability.py` separately. They
test real-model interruption/recovery and deliberately terminate a worker
belonging to their own engine after verifying process ancestry. Text also
checks actual consumer-queue saturation, stale handles and oversized-plan
rejection. TTS's slow consumer does not prove saturation of every internal
buffer. Inspect individual outcomes: `status=completed` means the probe ran,
not that all checks passed. Any forced cleanup remains a finding.

## Analyze and qualify

For a separate TTS stage-attribution diagnostic, enable the existing
`VLLM_OMNI_STEP_STATS_DIR=/absolute/path/to/counters` hook and use
`VLLM_OMNI_STEP_STATS_SYNC=0`. Run `profile_local_tts.py` with
`--length-band medium --concurrency 1 --batch-size 1 --repeats 20 --sustained-seconds 0`
and a fresh output directory. This is not a full baseline run. The counters
include warmup, slow-consumer, abort and recovery activity, and nested timers
overlap. Preserve per-process summaries rather than pooling their percentiles.
These are host-side timers without accelerator synchronization. Periodic
flushing can omit the final counter tail after forced process termination.
Comparisons with an uninstrumented baseline must retain cache/thermal and
stochastic speech-length caveats.
Core-step counts include idle/no-work iterations. The current orchestrator
dispatch hook is only in the optional event-driven loop; the default loop
does not produce that counter. Absent counters are not zero-time measurements.

```bash
python benchmarks/edge_harness/summarize_e2e_profiles.py \
  --matrix benchmarks/edge_harness/results/model_device_matrix_20260922/matrix.json \
  --runs /absolute/path/to/run-root --out /absolute/path/to/summary

python benchmarks/edge_harness/analyze_profile_resources.py \
  --runs /absolute/path/to/run-root --out /absolute/path/to/resource-summary.json

python benchmarks/edge_harness/experiments/verify_e2e_summary_table.py \
  benchmarks/edge_harness/results/e2e_profiling_20260922/evidence/summary/README.md \
  --json-out benchmarks/edge_harness/results/e2e_profiling_20260922/evidence/summary/qualification_status.json
```

Use run directory names `spark-cpu-wsl`, `spark-cuda-wsl`,
`spark-cuda-windows`, `tts-cuda-wsl` and `tts-cuda-windows`, with `-host`,
`-trace` and `-reliability` suffixes for their corresponding records. The matrix
summarizer verifies all sixty pairings and the original evidence hashes. It
reports nearest-rank p50/p95 and sample counts, checks the three serial
single-request groups, flags recorded output-contract violations, and retains
unavailable metrics. It labels the historical concurrency sweep separately
and never promotes those archived measurements to a current batch-1 pass.
Delivery-spacing statistics pool adjacent intervals within each request; their
sample count is the number of intervals. Text output updates may coalesce
tokens, so this is delivery timing rather than isolated kernel inter-token
latency. Effective prefill throughput includes all work before the first output.

Parse large traces only after inference timing has finished. The resource
analyzer streams trace JSON, separates summed event duration from interval
union, and excludes telemetry gaps over five seconds from energy integration.
Actual device placement, fallback, transfer and synchronization conclusions
must be supported by the trace/runtime records, not merely the intended plan.
The TTS diagnostic starts each stage with a distinct profiler prefix because
both workers have rank 0 and the current runtime otherwise overwrites their
shared trace filename. Both stages are active during traced requests. Sequential
stop/export RPCs can extend later-stage trace windows with idle/export time;
trace-window length is not model execution time or GPU utilization.
Startup summaries retain both constructor time and inclusive host-launch to
first warmup submission/output. The latter includes artifact hashing, plan
construction and imports; constructor timing alone excludes that preparation.

Keep compile/load/run failures and their exact configurations. Before declaring
a pairing qualified, review model-specific reference quality, complete declared
modalities, state/stream correctness, memory admission, executed placement and
the single-request performance protocol. Missing artifacts, backends, device
access, sensors or reference suites require explicit reasons and repair actions.
AI Hub component or host-replayed jobs are separately labeled from a resident
device-local request; a component-time sum is an estimate, not whole-request
latency. A capacity refusal requires measured device RAM and a pinned artifact
budget, never just parameter count.

After every writer exits, archive evidence with reversible compression of large
uncompressed files and an original/stored byte-hash manifest:

```bash
python benchmarks/edge_harness/archive_profiles.py create /path/to/run-root /path/to/new-archive
python benchmarks/edge_harness/archive_profiles.py verify /path/to/new-archive
python benchmarks/edge_harness/archive_profiles.py restore /path/to/new-archive /path/to/new-run-root
```

Restore before using original report-relative paths for compressed artifacts.
The utility does not rewrite raw reports. Integrity verification establishes
that archived bytes match, not that a model passes its qualification gates.
