# LingBot World regression gates

The suite drives the real `WS /v1/realtime/video` endpoint. CPU tests validate
gate logic separately and are not model performance evidence.

## Prerequisites and commands

Use the inference environment matching `docker/Dockerfile.ci` (vLLM 0.31.0)
and development dependencies. Reserve devices with the host scheduler before
inference. CUDA nightly needs four GPUs; NPU uses the four-device Ascend A2 pool. The smoke
uses the real 14B checkpoint at 192×320 on one GPU. Full performance/quality
profiles remain 480×832. CI provides exclusive devices.

```bash
pytest -q --confcutdir=tests/benchmarks \
  tests/benchmarks/test_lingbot_world_realtime.py \
  tests/benchmarks/test_lingbot_world_regression.py \
  tests/benchmarks/test_lingbot_world_precision.py \
  tests/benchmarks/test_lingbot_world_video_quality.py \
  tests/benchmarks/test_lingbot_world_ci.py
```

The launcher downloads the pinned checkpoint if `LINGBOT_MODEL_PATH` is unset.
For an existing local or mirrored checkpoint, export that path first. All model
files and images must match `assets.lock.json`; directory names do not matter.

```bash
bash .buildkite/common/scripts/run-lingbot-ci.sh smoke \
  tests/dfx/lingbot_world/configs/cuda_smoke.json lingbot-artifacts/cuda-smoke
bash .buildkite/common/scripts/run-lingbot-ci.sh generate \
  tests/dfx/lingbot_world/configs/cuda_nightly.json lingbot-artifacts/cuda-nightly
```

Source the CANN environment and substitute `npu_nightly.json` for Ascend.
The script prepares a clean, separately checked-out reference revision and
runs pytest with the correct platform/card selection and `--run-level full_model`.
Direct equivalent after preparing `/tmp/lingbot-reference`:

```bash
pytest -sv tests/dfx/lingbot_world/test_generate.py \
  --lingbot-profile tests/dfx/lingbot_world/configs/cuda_nightly.json \
  --lingbot-reference-root=/tmp/lingbot-reference \
  --lingbot-output lingbot-artifacts/cuda-nightly \
  -m 'full_model and diffusion and cuda and cards_4' --run-level full_model \
  --junitxml=lingbot-artifacts/cuda-nightly/pytest.xml
```

## Performance, precision and lifecycle

Three reference and three candidate server starts run in alternating AB/BA
order on the same devices and dependencies. Each case receives an unmeasured
eight-chunk warmup session. The first six and terminal chunks stay in raw results and
playback/FPS accounting, but are excluded from steady cadence by `workload.py`.

Reference cadence CV above 3% invalidates calibration. Each candidate must
meet measured reference limits with the profile's 5% regression allowance.
Candidate measurements never calibrate their own limits. This is a relative
regression gate, not a promise of 12 FPS. The MP4 label and playback FPS basis
are recorded separately. P99 is always reported but gates only when each
independent start has at least 1000 steady observations; long profiles use
two measured 600-chunk sessions per start.

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
Native units differ: VBench imaging is 0–100, WBench imaging is normalized.

Video diagnostics validate full frame/geometry/FPS coverage, compute PSNR on
every frame and SSIM every 24 frames, and compare candidate divergence against
same-code reference repeats. Perfect repeats use an explicit minimum PSNR
budget rather than arithmetic on infinity.

These are fixed-input CI video-quality evaluations, not official leaderboard
totals. WBench navigation, reconstruction and API/VLM dimensions are outside
this video-quality suite.

## Buildkite and baseline maintenance

- READY: CPU gates and a one-GPU real-checkpoint stream/cancellation smoke.
- NIGHTLY: independent CUDA/NPU generation followed by one-GPU evaluation.
  The NPU build provisions its matching CUDA evaluator image in the same build.
- WEEKLY/manual (`weekly-test`): three scenes, three camera patterns and two
  seeds (18 cases), 160 chunks each, plus a separate 600-chunk long profile.

Selected jobs are blocking. Missing prerequisites and skipped CI tests fail.
JUnit, raw chunks, measured limits, precision captures, memory/events, videos,
scores and logs are retained through `artifact_paths`, including failed jobs.

Advance a reference only by changing its full SHA and inspecting three new
independent reference starts on every target runner. Model, input, evaluator,
topology or precision changes require reviewed lock/profile updates and fresh
measurements. Include matching-head Buildkite URLs in the PR. Never drop failed
observations, automatically widen budgets, or treat CPU tests and older hardware
experiments as acceptance of the new commit.
