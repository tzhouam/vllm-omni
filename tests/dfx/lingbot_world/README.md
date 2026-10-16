# LingBot World regression gates

The suite drives the real `WS /v1/realtime/video` endpoint. CPU tests validate
gate logic separately and are not model performance evidence.

## Prerequisites and commands

Use the inference environment matching `docker/Dockerfile.ci` (vLLM 0.31.0)
and development dependencies. Reserve devices with the host scheduler before
inference. Full CUDA profiles use four GPUs while reserving all eight GPU slots
of a node in the H100 pool; NPU uses the four-device Ascend A2 pool. The smoke
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

The NPU bootstrap reference is `8d7c354674c7e1dc57544f248689574fccb71af9`,
which places unselected DiT weights before direct AR projections and releases
selected text weights before video conditioning. Those placement fixes are
required for this component-offload deployment. CUDA keeps its original reference;
every platform still calibrates independent reference starts before checking candidates.

## Performance, precision and lifecycle

Three reference and three candidate server starts run in alternating AB/BA
order. Each reference/candidate pair shares its device reservation and dependencies.
Each case receives an unmeasured
eight-chunk warmup session. The first six and terminal chunks stay in raw results and
playback/FPS accounting, but are excluded from steady cadence by `workload.py`.

Reference cadence CV above 3% invalidates calibration. Each candidate must
meet measured reference limits with the profile's 5% regression allowance.
For TTFC and accumulated playback stalls only, the allowance is the greater of
5% of the worst reference and three reference population standard deviations.
The profile pins this multiplier (at most three); both allowances are reported.
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
not regression acceptance; quality jobs require the aggregate gate to pass.

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
and six WBench video-quality dimensions are checked per case and candidate
repeat. Dynamic degree uses a two-sided activity bound. Excessive reference
spread fails calibration; scores are never averaged across dimensions/scenes.
`quality.json` separately pins `max_reference_spreads` for reproducibility and
per-suite candidate tolerances for degradation. The candidate floor remains the
worst reference minus its candidate tolerance; a larger reference-spread ceiling
never increases that tolerance. Reports retain both budgets and the measured
reference spread. Reference ceilings must be verified on the target evaluator;
local diagnostic scores alone do not establish hardware acceptance.
All selected scores from these pinned evaluator revisions use a 0–1 scale.

Video diagnostics validate full frame/geometry/FPS coverage, compute PSNR on
every frame and SSIM every 24 frames, and compare candidate divergence against
all three pairs of independent same-code reference repeats. Candidate PSNR
must stay within 3 dB of the worst reference distance, and sampled SSIM within
0.03. These are approximate rollout-divergence budgets; the separate fixed-input
VAE numerical gate still requires exact repeats. PSNR is also gated independently in every two-second
window so a whole-video average cannot hide late corruption. Perfect repeats use
an explicit minimum PSNR budget rather than arithmetic on infinity. Failed pixel
checks retain the reference noise floors and every candidate result.

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

Advance a reference only by changing its full SHA and inspecting three new
independent reference starts on every target runner. Model, input, evaluator,
topology or precision changes require reviewed lock/profile updates and fresh
measurements. Include matching-head Buildkite URLs in the PR. Never drop failed
observations, automatically widen budgets, or treat CPU tests and older hardware
experiments as acceptance of the new commit.

The pinned `startup_timeout_s` bounds both the outer server-readiness wait and
the server's `--init-timeout` / `--stage-init-timeout`. This avoids silently using
a shorter internal default while cold checkpoints load. Initialization timeouts
still fail pytest and retain their evidence; measurement thresholds are separate.
