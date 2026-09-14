# LingBot-World four-GPU spatial VAE benchmark

These manual benchmarks integrate LingBot-World latent generation with the
existing Wan VAE width-sharded decoder (`install_wan_spatial_shard_decode`).
They do not add a new VAE backend or change production serving. All four VAE
ranks share the same four visible GPUs as the DiT workers; this is feature-map
spatial sharding with halo exchange, not channel/weight tensor parallelism or
independent overlapping image tiles.

Requirements: the normal vLLM-Omni CUDA environment, four visible GPUs, a
LingBot-World v2 causal checkpoint with an untiled FP32 Wan VAE (`patch_size=None`),
and sufficient memory for both DiT and VAE workers. Run from the repository root.
The validation phase additionally runs an unpatched single-GPU VAE reference.
The benchmark never allocates or reserves GPUs itself; use your cluster's
scheduler before launching it. Select an unused `--port` base (base through
base + 10).

## Decode saved real latents first

The probe accepts ten normalized BCTHW FP32 CPU tensors named
`steady_latent_00.pt` through `steady_latent_09.pt`. Each represents three latent
frames. It runs two sessions with a reset between them. Only load trusted tensor
files; loading uses `weights_only=True`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/lingbot_world/probe_spatial_vae.py \
  --model /path/to/lingbot-world-v2-14b-causal-fast-diffusers \
  --latents /path/to/saved/latents \
  --output /path/to/new/probe-result --port 29500
```

Validation compares the **first four chunks** of session zero against:

- a fresh-cache, concatenated four-rank decode (`atol=rtol=1e-5`);
- an unpatched, untiled single-GPU FP32 VAE decode (maximum absolute error 0.03).

The comparison uses floating-point pixels before uint8 conversion. Its tensors
are saved in `validation_tensors.pt`; validation is outside timed decode. This
prefix check does not establish whole-rollout equivalence. Every chunk must be
finite and contain nine output frames initially, then twelve per chunk. All
uint8 chunks, per-chunk hashes, timings, input hashes and validation results are
saved. No lossless/bitwise claim is made for spatial versus single-GPU decode.

## Integrate with LingBot latent generation

Provide a condition image, prompt, and JSONL file with **exactly ten** events.
Each row has an integer `event_id`, optional `prompt`, and optional `frames`
containing three lists of camera actions. For a static-camera smoke test:

```bash
python - <<'PY'
import json
from pathlib import Path
Path('/tmp/lingbot-events.jsonl').write_text(''.join(
    json.dumps({'event_id': i, 'frames': None}) + '\n' for i in range(10)))
PY
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/lingbot_world/async_vae_spatial_lingbot.py \
  --model /path/to/lingbot-world-v2-14b-causal-fast-diffusers \
  --image /path/to/condition.png --prompt 'A still landscape.' \
  --events /tmp/lingbot-events.jsonl --output-dir /path/to/new/integrated-result \
  --ulysses-degree 4 --enforce-eager --epochs 2 --case D --port 29500
```

Case D performs DiT then decode sequentially. Case E is an **experimental**
previous-chunk decode / next-chunk DiT overlap comparison on the same GPUs, with
at most one pending decode. It is not a serving scheduler and does not validate
backpressure or cancellation. Overlap may lose performance through contention;
there is no claimed speedup. Ulysses2 and Ulysses4 are supported with TP1 only.
Each session has ten chunks; the first session is warmup. Use the same generated
inputs and inspect saved outputs when comparing schedules or implementations.

The scripts preserve state across chunks and reset state between sessions.
An isolated plain VAE reference is retained only for validation. Rank failures
terminate sibling ranks rather than waiting for collective cleanup indefinitely.
Both entrypoints stop their decoder supervisor on normal exit, timeout, or error.

## Measurement boundaries and artifacts

`decode.jsonl` records transfer/control completion, spatial decode completion,
CPU-ready uint8 time, shapes, hashes and peak memory. Integrated runs also record
parent receive time, DiT chunk timings and per-session timings in `summary.json`.
The pipe returns **metadata**, not decoded pixel payloads. CPU-ready and parent
receipt are not video encoding, network delivery, or client playback times.

All epochs save generated latents and decoded uint8 chunks. The second epoch
also exports `steady_latent_*.pt` for the probe. Saving, hashing, synchronization
and CPU-mediated latent transfer affect whole-session times. Do not report those
as an uninstrumented production throughput result. The decoder-ready timestamp
precedes output serialization. FP32 prefix reference tensors are separate from
full-rollout uint8 artifacts.

For the integrated runner, `LINGBOT_BENCH_PROFILE=1` enables a diagnostic trace
on the selected fixed warm-session chunks. Keep profiled results separate from
normal timing. No additional models, encoded videos, or generated frames are
committed to the repository.

These scripts were extracted from a local LingBot analysis harness. Previous
harness runs are historical evidence, not validation of this branch or a claim
that the complete LingBot performance plan has passed.
