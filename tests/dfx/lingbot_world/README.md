# LingBot World regression gates

The suite drives the real `WS /v1/realtime/video` endpoint. CPU tests validate
gate logic separately and are not model performance evidence.

## Prerequisites and commands

Use the inference environment matching `docker/Dockerfile.ci` (vLLM 0.31.0)
and development dependencies. Reserve devices with the host scheduler before
inference. Full CUDA profiles use four GPUs while reserving all eight GPU slots
of a node in the H100 pool; NPU reserves eight Ascend A2 devices and uses four. The smoke
uses the real 14B checkpoint at 192×320 on one GPU. Full performance/quality
profiles remain 480×832. CI provides exclusive devices.

The CUDA performance preset prevents other GPU jobs from sharing the eight-GPU
node during paired measurements. The launcher verifies all eight reserved
devices, records their UUIDs in `allocation.json`, then preserves the original
device mapping while exposing only four to inference. This retains SP4, memory
and precision settings. It changes scheduling isolation rather than regression
thresholds. A four-GPU allocation cannot silently run an exclusive profile.
For a local full CUDA run, reserve eight devices with `gpu run --gpus 8 -- ...`
before invoking the launcher.

NPU generation and long-pair collection use the existing `a2b3_npu_8`
scheduler preset. Before model preparation, a separate preflight process
requires eight runtime-visible devices and records their names, memory sizes
and CANN runtime identifiers in `allocation.json`. The launcher then exports
the first four identifiers through `ASCEND_RT_VISIBLE_DEVICES`; inference
rechecks the device count, mapping and properties. These identifiers are not
physical UUIDs. The eight-slot reservation reduces device co-location while
retaining SP4, the workload and the reference/candidate regression limits.
It does not prove exclusive access to host CPU, storage or networking.

```bash
pytest -q --confcutdir=tests/benchmarks \
  tests/benchmarks/test_lingbot_world_realtime.py \
  tests/benchmarks/test_lingbot_world_regression.py \
  tests/benchmarks/test_lingbot_world_precision.py \
  tests/benchmarks/test_lingbot_world_video_quality.py \
  tests/benchmarks/test_lingbot_world_ci.py
```

The `unit` stage captures CPU test artifacts without loading a checkpoint.
The model stages download the pinned checkpoint if `LINGBOT_MODEL_PATH` is unset.
Only that explicit provisioning subprocess enables Hub access; inference keeps
the runner's offline setting.
For an existing local or mirrored checkpoint, export that path first. All model
files and images must match `assets.lock.json`; directory names do not matter.

```bash
BUILDKITE=true bash .buildkite/common/scripts/run-lingbot-ci.sh unit \
  tests/dfx/lingbot_world/quality.json lingbot-artifacts/unit
bash .buildkite/common/scripts/run-lingbot-ci.sh smoke \
  tests/dfx/lingbot_world/configs/cuda_smoke.json lingbot-artifacts/cuda-smoke
bash .buildkite/common/scripts/run-lingbot-ci.sh generate \
  tests/dfx/lingbot_world/configs/cuda_nightly.json lingbot-artifacts/cuda-nightly
```

Source the CANN and ATB environments and substitute `npu_nightly.json` for Ascend.
The script prepares a clean, separately checked-out reference revision and
runs pytest with the correct platform/card selection and `--run-level full_model`.
Direct equivalent after preparing `/tmp/lingbot-reference`:

```bash
export LINGBOT_ALLOCATION_FILE="$(pwd)/lingbot-artifacts/cuda-nightly/allocation.json"
CUDA_VISIBLE_DEVICES="$(python -m tests.dfx.lingbot_world.prepare devices \
  --profile tests/dfx/lingbot_world/configs/cuda_nightly.json \
  --root lingbot-artifacts/cuda-nightly)"
export CUDA_VISIBLE_DEVICES
pytest -sv tests/dfx/lingbot_world/test_generate.py \
  --lingbot-profile tests/dfx/lingbot_world/configs/cuda_nightly.json \
  --lingbot-reference-root=/tmp/lingbot-reference \
  --lingbot-output lingbot-artifacts/cuda-nightly \
  -m 'full_model and diffusion and cuda and cards_4' --run-level full_model \
  --junitxml=lingbot-artifacts/cuda-nightly/pytest.xml
```

NPU profiles use four A2 cards with eager execution and component-level CPU
text-encoder offload; DiT/VAE stay on the NPU. The all-resident configuration
OOMed on the target runner at the first warmup (64GB cards expose 60.96GiB).
The CI deployment retains the full resolution, KV window and rollout lengths;
`gpu_memory_fraction=0.1` avoids allocating extra pages above the one-session
minimum after offloading frees device memory. This placement is identical for
reference/candidate and is included in the hashed deployment contract.
The same deployment disables eager multithreaded safetensors loading for both
roles: four ranks otherwise read up to sixteen whole shards concurrently from
the shared checkpoint filesystem. The serial iterator preserves the checkpoint
tensors and leaves the 1,200-second initialization budget unchanged. Startup
success must still be established on the target runner.
The NPU profiles pin `cpu_threads_per_worker=1`. Both reference and candidate
servers and fixed-input captures receive `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
and `OPENBLAS_NUM_THREADS` before importing Torch, while retaining CANN's
backend import paths. This bounds host-side parallelism across the four ranks.
The setting is part of the hashed profile; changing it requires fresh
independent reference calibration and target-hardware validation.

The NPU bootstrap reference is `8d7c354674c7e1dc57544f248689574fccb71af9`,
which places unselected DiT weights before direct AR projections and releases
selected text weights before video conditioning. Those placement fixes are
required for this component-offload deployment. CUDA keeps its original reference;
every platform still calibrates independent reference starts before checking candidates.

## Performance, precision and lifecycle

Three reference and three candidate server starts run in alternating AB/BA
order. Each reference/candidate pair shares its device reservation and dependencies.
Raw benchmark JSON retains unmeasured warmup chunk arrivals and startup timings
under `warmups`, separately from measured `sessions` and aggregate `metrics`.
These records are serialized after measurement and add no requests or settling
delays. They help distinguish warmup behavior from subsequent TTFC failures.
If a warmup or measured session fails, the client writes a separate
`benchmark-{warmup|session}-N-failure.json` containing received chunk timings,
protocol events and the original error. This diagnostic has `complete: false`
and no aggregate metrics; it cannot establish acceptance. Trace writes occur
after failure and preserve the original exception even if the filesystem fails.
Lifecycle checks use a one-second playback window: the initial nine-frame
block fits, but admitting the following twelve-frame block requires playback
progress. The held-playhead assertion therefore exercises backpressure before
second-block inference, independently of model speed. These instrumented
requests remain outside performance measurements.
Each case receives an unmeasured
eight-chunk warmup session. The first six and terminal chunks stay in raw results and
playback/FPS accounting, but are excluded from steady cadence by `workload.py`.

Reference cadence CV above 3% invalidates calibration. Each candidate must
meet measured reference limits with the profile's 5% regression allowance.
For TTFC and accumulated playback stalls only, the allowance is the greater of
5% of the worst reference and three reference population standard deviations.
The profile pins this multiplier (at most three); both allowances are reported.
H100 TTFC also uses a reference-only 220 ms jitter floor. Two matching nightly
reference cohorts have maximum cross-cohort excess 174.791 ms; adding 25%
reserve and rounding up to 5 ms gives 220 ms. On 36 separate weekly reference
comparisons this removes three same-code false rejections. The calibration
record retains all 54 reference measurements and artifact hashes; candidate
values never enter that calculation. `ttfc-calibration.json` retains the rule
and record hash with each CUDA generation result. Cadence, FPS, P99 and stall
limits do not receive this TTFC allowance. NPU uses no TTFC floor. Independent
final-head hardware validation remains required.
CUDA additionally uses a 135 ms floor for the total-stall allowance. Two
independent three-start reference cohorts falsely rejected two same-code
reference runs under the original rule. All their stalls occurred during
startup; their largest cross-cohort startup excess was 107.778 ms. Adding 25%
reserve and rounding up to 1 ms gives 135 ms, using reference inputs only.
This floor requires a second, independent post-startup stall gate without
the floor: later stalls cannot hide behind initial playback jitter. The
startup boundary is `max(2, warmup_chunks)` media blocks (six in CI), so the
first nine-frame block and the configured attention ramp are distinguished
from steady playback. Both components stay in total-stall accounting. The
one-chunk playback buffer is explicitly passed and checked. NPU retains its
own calibration with zero constant floor. TTFC and cadence limits are unchanged.
This accounts for startup/discrete-stall jitter observed in real H100 runs, and
is a fixed calibration rule, not a confidence interval or a retry of a failed
candidate. Stationary cadence limits and the 3% reference CV cap are unchanged.
Candidate measurements never calibrate their own limits. This is a relative
regression gate, not a promise of 12 FPS. The MP4 label and playback FPS basis
are recorded separately. P99 is always reported but gates only when each
independent start has at least 1000 steady observations; long profiles use
two measured 600-chunk sessions per start.

CI partitions weekly's 18 cases into three CUDA groups of six cases and six NPU
groups of three cases. Each case still receives all three reference and three
candidate starts. Generated effective profiles retain the source profile SHA256;
generation and evaluation use the same `LINGBOT_CASE_SHARD=INDEX/COUNT` selection.
No scene, seed, rollout length or measurement is removed.

NPU long collection runs as three independent reference/candidate pairs, followed
by a blocking CPU aggregation job. Each start still has two 600-chunk sessions
and at least 1,186 steady observations. The first pair also runs both complete
600-chunk memory/control diagnostics. Pairs may run on different nodes, but the
aggregate rejects different hardware/dependency contracts and unstable reference
populations. It verifies all three complete manifests, revisions and checksums,
rejects duplicated sessions, recomputes performance from raw chunks, and applies
the same performance and seven VAE precision checks. Collection success alone is
not regression acceptance. Quality jobs depend on the aggregate, retain scores
for complete failed populations, and require both aggregate and quality gates
to pass for acceptance.

These partitions keep individual jobs below the effective timeout observed in
the CI environment. A configured step timeout does not override a smaller
[pipeline/organization maximum](https://buildkite.com/docs/pipelines/configure/build-timeouts).

Fixed-input VAE diagnostics run outside performance windows. Both checkouts
load the real Wan VAE and decode the same CPU-seeded BF16 latent. Independent
same-code repeats must be exact before reference/candidate tolerances apply.
This is VAE numerical evidence, not a full DiT replay or GPU/NPU bitwise-parity
claim. Generated videos additionally cover full-pipeline behavior.

Existing public endpoint tests check chunk order, pacing, stop and successful
session reuse. An additional instrumented rollout sends a live SE3 camera update
(with the fixed action script disabled, as required by the pipeline),
requires its event ID in streamed started/active/completed metadata, and samples
device-wide memory on the allocated devices. The post-warmup growth budget is
256 MiB per die. Diagnostic timings never enter performance limits. Server
child processes must exit after teardown.

## GPU evaluation, including NPU videos

The evaluator uses a separate Python 3.12 environment with resolved dependencies
in `requirements/lingbot-quality.txt`. `evaluators.lock.json` pins code revisions,
submodules and asset checksums. Provisioning validates downloads before scoring.
An optional `LINGBOT_EVALUATOR_CACHE` supplies files at the lock's relative
paths; cached files still undergo checksum verification.
The Buildkite cache key includes both the evaluator/weight lock and the resolved
dependency lock. A completed interpreter is never resynchronized underneath an
active scorer. If either lock changes, use a fresh custom evaluator root.

```bash
export LINGBOT_EVALUATOR_ROOT=/path/to/evaluator-cache
python -m tests.dfx.lingbot_world.prepare evaluators \
  --root "$LINGBOT_EVALUATOR_ROOT"
export LINGBOT_EVALUATOR_PYTHON="$LINGBOT_EVALUATOR_ROOT/venv/bin/python"
pytest -sv tests/dfx/lingbot_world/test_quality.py \
  --lingbot-profile tests/dfx/lingbot_world/configs/npu_nightly.json \
  --lingbot-generation lingbot-artifacts/npu-nightly \
  --lingbot-output lingbot-artifacts/npu-nightly-quality \
  -m 'full_model and diffusion and cuda and H100 and cards_1' \
  --run-level full_model --junitxml=lingbot-artifacts/npu-nightly-quality/pytest.xml
```

Transfer the complete generation directory to a GPU evaluator. Candidate commit,
profile and artifact checksums must match. Seven VBench custom-input dimensions
and six WBench video-quality dimensions are checked per case. Continuous scores
are checked for every candidate repeat. Both pinned dynamic-degree evaluators
return a binary classification for each complete video. Activity therefore uses
a two-sided difference in active counts across three independent starts: the
unchanged 0.34 fractional budget permits one changed classification, while two
or three changes fail. All six individual classifications remain in the report.
H100 reference cohorts from builds [17426](https://buildkite.com/vllm/vllm-omni/builds/17426)
and [17431](https://buildkite.com/vllm/vllm-omni/builds/17431) differed by at most one
classification per scene; the latter returned `[0, 1, 0]` for both reference
and candidate in the blossom case. A fractional per-video range limit cannot
represent that binary trial noise. This count budget is not a confidence test;
target-runner validation is still required for every profile and platform.
The [reference-only calibration record](calibration/cuda_nightly_reference.json)
retains all six reference scores, the matching contracts and raw-score hashes;
candidate scores do not enter that record.
Excessive continuous reference
spread fails calibration; scores are never averaged across dimensions/scenes.
`quality.json` separately pins `max_reference_spreads` by generation platform for reproducibility and
per-suite candidate tolerances for degradation. The candidate floor remains the
worst reference minus its candidate tolerance; a larger reference-spread ceiling
never increases that tolerance. Reports retain both budgets and the measured
reference spread. Preliminary CUDA continuous ceilings use the maximum
per-scene range across six frozen-reference starts, plus 25% reserve, rounded
up to 0.005; an existing larger ceiling is retained. The calibration record
contains the reference values and the exact calculation. VBench cohorts were
scored on target H100 evaluators; the historical WBench cohorts were scored
independently on L20X. Build 17443 also executes all seven VBench and six WBench
dimensions on H100. Its three frozen-reference boardwalk HPSv3 scores span
0.064710581354398, exceeding the initial 0.05 reference ceiling. Applying the
same 25% reserve and 0.005 rounding rule to these target-H100 references yields
0.085. The HPSv3 candidate tolerance stays 0.02; candidate scores do not set
either value. The calibration record retains the nine reference HPSv3 scores,
video/score hashes and evaluator provenance. The original failed job remains
failed; a fresh independent target-H100 campaign must validate this calibration.
The first completed target-H100 600-chunk WBench reference population also
has imaging-quality range 0.02171683116488965, above its preliminary 0.02
reference ceiling. The same reference-only rule yields 0.03; the candidate
imaging tolerance stays 0.01 and NPU limits stay separate. The calibration
record retains the three reference scores, actual video hashes, frozen source,
long-profile hash and native evaluator provenance. The ongoing independent
long campaign is the holdout; these calibration inputs do not accept it.
Every weekly/long profile also
needs held-out target validation. NPU retains its existing separate ceilings
until its own reference campaign completes. Reference ceilings must be verified on the target evaluator;
local diagnostic scores alone do not establish hardware acceptance.
Candidate tolerances are also keyed by generation platform. CUDA VBench
aesthetic uses 0.025: independent same-code reference deficits reached
0.018552, exceeding the initial 0.01 budget, before adding 25% reserve and rounding
up to 0.005. Other native candidate budgets stay at 0.01 (HPSv3: 0.02).
NPU retains its initial native budgets. These reference-only calculations are
recorded separately from the reference-spread ceilings and still require fresh
target-runner validation; candidate observations never set their own limits.
Standard VBench background consistency encodes every native-selected frame
through CLIP in batches of 32 with gradients disabled. The pinned native
implementation otherwise retains a full-rollout gradient graph and exceeds
even 140 GiB on a 600-chunk reference video. This adapter changes only that
metric's image encoding; native transforms, normalization, cosine aggregation
and frame selection are preserved. Other metrics keep their native paths and
the evaluator checkout stays immutable. Reports bind the adapter source hash
and retain per-video encoded-frame/batch counters. CI requires every expected
frame and rejects a missing counter, changed batch size or enabled gradients.

Both native score populations are saved before calibration checks. A reference
calibration error is recorded as a failed case check; remaining cases and pixel
checks still run and retain their evidence. The final pytest assertion fails
if any native calibration, candidate score or pixel check fails.
All selected scores from these pinned evaluator revisions use a 0–1 scale.

Video diagnostics validate full frame/geometry/FPS coverage, compute PSNR on
every frame and SSIM every 24 frames, and compare candidate divergence against
all three pairs of independent same-code reference repeats. Candidate PSNR
must stay within 3 dB of the worst reference distance. Sampled SSIM may drop by
0.04 on CUDA or 0.03 on NPU. Independent CUDA reference SSIM deficits reached
0.031969, exceeding the initial 0.03 margin; 25% reserve rounded upward to 0.005
gives 0.04. Their maximum PSNR deficit was 2.535 dB, so the existing 3 dB margin
is retained. Only failed reference-only budgets are adjusted; the CUDA values
are not copied to NPU. These are approximate rollout-divergence budgets; the separate fixed-input
VAE numerical gate still requires exact repeats. PSNR is also gated independently in every two-second
window. Perfect repeats use
an explicit minimum PSNR budget rather than arithmetic on infinity. Failed pixel
checks retain the reference noise floors and every candidate result.

The fixed-photograph CI profiles additionally reject any decoded frame whose
pixels all have the same RGB value, in both reference and candidate outputs.
A real 600-chunk reference population admitted an inserted 32-frame black
window under relative pixel budgets (5.45 dB against a 4.69 dB floor). The
independent spatial-content check catches that collapse without widening or
tightening the measured noise budgets. Every decoded frame is checked; reports
retain `uniform_frame_indices`, and missing coverage fails selected CI tests.
Standalone comparison diagnostics retain support for intentionally uniform
videos unless this CI policy is explicitly selected.

These are fixed-input CI video-quality evaluations, not official leaderboard
totals. WBench navigation, reconstruction and API/VLM dimensions are outside
this video-quality suite.

## Buildkite and baseline maintenance

- READY: a dedicated CPU unit gate and a one-GPU real-checkpoint
  stream/cancellation smoke. The unit job uses the standard L4 CI image with
  accelerators masked, and retains JUnit, logs, the quality policy and package
  versions. Selected skips fail through the same pytest hook as model jobs.
- NIGHTLY: independent CUDA/NPU generation followed by one-GPU evaluation.
  The NPU build provisions its matching CUDA evaluator image in the same build.
- WEEKLY/manual (`weekly-test`): three scenes, three camera patterns and two
  seeds (18 cases), 160 chunks each across the groups above, plus the complete
  600-chunk long profile. NPU long quality waits for all three collected pairs
  and their CPU performance/precision aggregation gate.

Selected jobs are blocking. Missing prerequisites and skipped CI tests fail.
NPU generation uses the existing infrastructure retry codes (`-1`, `128`),
with one retry per code. Pytest failures and OOM are never automatically retried.
JUnit, raw chunks, measured limits, precision captures, memory/events, videos,
scores and logs are retained through `artifact_paths`, including failed jobs.
A failed placeholder JUnit report is written before provisioning so a forced
SIGKILL still leaves failure evidence. Pytest replaces it on completion; normal
failure and SIGTERM also retain structured launcher errors.
Provisioning records host available/total RAM, cgroup membership, raw memory
limits/peaks/pressure counters, available v1/v2 CPU quota/throttling/pressure
counters and CPU-set masks,
and the resolved checkpoint filesystem in `model.json`.
Pytest failure JSON captures fresh kernel resource evidence
after cleanup; peak counters can still expose load-time pressure. Kernel paths
remain explicit because cgroup namespace layouts differ. Unavailable interfaces
remain absent/null; read errors are retained without replacing the original
test failure.
These read-only snapshots run before serving or after failure, outside timing.
An unstable reference remains a failed calibration in the complete generation
manifest. Other cases, all seven VAE checks and video checksums are still
retained, so independent scoring can inspect the failed population.
Quality steps also run after a failed generation gate: when all videos and
their checksums are retained, both native suites and pixel checks still record
their evidence. `quality.json` separates `generation_passed` and
`quality_passed`; overall `passed` requires both. Missing, unhashed, corrupt or
incompatible generation populations fail before scoring. The generation job
remains blocking, and evaluator-image failures still prevent scoring.

To reproduce one weekly group, set `LINGBOT_CASE_SHARD=0/3` for CUDA or `0/6`
for NPU when invoking both `generate` and `evaluate`. To reproduce NPU long
collection, invoke `collect` three times with `LINGBOT_MEASUREMENT_REPEAT=0`,
`1` and `2`, writing to `lingbot-artifacts/npu-long-measurements/pair-0`,
`pair-1` and `pair-2`. Each invocation requires the usual four-NPU reservation.
Then run the CPU gate without loading a checkpoint:

```bash
bash .buildkite/common/scripts/run-lingbot-ci.sh aggregate \
  tests/dfx/lingbot_world/configs/npu_long.json lingbot-artifacts/npu-long \
  lingbot-artifacts/npu-long-measurements lingbot-npu-weekly-long
```

For a local run, leave `BUILDKITE` unset so the already collected directories
are used directly. On CI the launcher downloads every pair from the named
matrix step. Use the aggregate's output directory for the subsequent GPU evaluator.

For source attribution, use `candidate_revision` from the retained
`generation.json`. Buildkite can test a synthetic merge containing upstream
changes absent from the PR branch. Compare that merge with `reference_revision`
and its upstream parent before attributing failures to the PR or host noise.
Comparing only the unmerged branch cannot establish source equivalence.
Upstream differences alone do not justify replacing the reference or widening
limits; preserve the failed observations while investigating their cause.

Advance a reference only by changing its full SHA and inspecting three new
independent reference starts on every target runner. Model, input, evaluator,
topology or precision changes require reviewed lock/profile updates and fresh
measurements. Include matching-head Buildkite URLs in the PR. Never drop failed
observations, automatically widen budgets, or treat CPU tests and older hardware
experiments as acceptance of the new commit.

For a quality calibration update, retain two independent three-start reference
cohorts, their raw-score hashes and matching numerical contracts. Evaluate
each cohort as held-out reference data against the other before adjusting any
failed budget. Candidate outputs do not enter this calculation. Update the
reference-only record and its platform's policy together, then run
`pytest tests/benchmarks/test_lingbot_world_regression.py -m 'core_model and cpu'`
and a fresh target-runner campaign. CUDA quality jobs reject calibration reuse
after deployment, hardware, runtime package, precision or model/reference changes;
they retain the applied policy, generation profile and calibration record.
The preliminary CUDA record does not establish NPU or weekly/long acceptance.

The pinned `startup_timeout_s` bounds both the outer server-readiness wait and
the server's `--init-timeout` / `--stage-init-timeout`. This avoids silently using
a shorter internal default while cold checkpoints load. Initialization timeouts
still fail pytest and retain their evidence; measurement thresholds are separate.
