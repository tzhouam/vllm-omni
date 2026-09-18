# Reproducing the LingBot-World v2 served realtime optimization

Self-contained: this branch holds the optimized vLLM-Omni tree, the benchmark client, and this
directory with the run scripts, deploy configs and an optional timing overlay. An empty machine needs
this branch, the model from Hugging Face, and four H200-class GPUs.

## 0. What you get

Served WebSocket path (`WS /v1/realtime/video`), LingBot-World v2 14B causal-fast, 480x832, 4 DMD steps,
Ulysses SP-4 with regional torch.compile, one session at a time. Steady-state median ms per chunk (12 frames):

| tree | ms / chunk | RTF @12 fps | RTF @16 fps | stall | output vs stock |
| --- | ---: | ---: | ---: | ---: | --- |
| stock upstream main 507cb1d83 | 1326 | 1.33 | 1.77 | 100% | reference |
| + PR #7648, lossless set (19 commits) | 916 / 920 | 0.94 | 1.25 | 21% | bit-identical |
| + PR #7651, width-sharded VAE decode (1 commit) = this branch | **721 / 718** | 0.74 | 0.99 | 4% | shard-border accumulation order, max abs 0.0195 on [-1, 1] |
| + PR #7749 `fused` (draft, VAE kernel path) | 685 / 684 | 0.70 | 0.94 | | not bit-exact, PSNR 54-55 dB vs plain decode |

Two numbers per row are two runs of the same tree; the run-to-run floor is about +-10 ms. RTF is wall time over
video time, so below 1.0 is real time. State the fps basis with every RTF: the checkpoint declares none.

## 1. Hardware and host

- Four GPUs of the H200 class (measured: 141 GB, 132 SMs, driver 570.133.20, CUDA 13.0) on one NUMA node.
  On smaller cards, lower `gpu_memory_fraction` in the deploy config first.
- Host RAM: the four workers load 18.5 B parameters each through host memory. Keep at least ~230 GB
  MemAvailable free when a server starts, or an out-of-memory killer ends a worker during weight load with only a
  bare "Shutdown signal received" in the log. `chain_example.sh` gates on this.
- A quiet host. CPU load above ~120 on a 224-core box turned a 55 ms steady std into 645 ms. Sustained
  tensor-core load hits the power cap and drops SM clocks ~25%; that affects every arm equally, but lock clocks if
  you compare single kernels.
- Time: one arm (server start, compile, 1 warm-up + 3 measured sessions of 40 chunks) takes 35-45 minutes.

## 2. Software

| component | version used | note |
| --- | --- | --- |
| Python | 3.12.13 | |
| torch | 2.13.0+cu130 | |
| vllm | 0.29.0 | this tree is upstream main after the 0.29 rebase |
| vllm-omni | this branch | `pip install -e .` from the repo root, or `PYTHONPATH=<repo root>` |
| diffusers | 0.40.0 | model_index.json says 0.35.0.dev0; 0.40 loads it |
| transformers | 5.14.1 | UMT5 text encoder |
| triton | 3.7.1 | needed by #7749's fused kernels only |
| flashinfer | installed | attention backends as bundled with vLLM 0.29 |
| websockets 17.1, av 18.1, imageio-ffmpeg 0.6 | client, MP4 demux, `--save-video` | |

**Pin what you serve.** `vllm serve` is a console script and puts its own `bin/` first on `sys.path`, so an
editable install silently serves whichever tree it was installed against, not the directory you are in.
`run_arm.sh` exports `PYTHONPATH` to the repo root, prints `vllm_omni.__file__` into `provenance.txt`, and aborts
if it is not under the repo.

## 3. Model

Hugging Face `robbyant/lingbot-world-v2-14b-causal-fast-diffusers` (diffusers layout: `model_index.json` with
`LingBotWorldCausalDMDPipeline`, `transformer/` = `CausalLingBotWorldTransformer3DModel`, UMT5 text encoder and
tokenizer, Wan `AutoencoderKLWan`, UniPC scheduler). `run_arm.sh` passes the HF id by default; set `MODEL` to a
local snapshot path to avoid the download on every host.

## 4. Code

This branch is upstream main 507cb1d83 plus, in order:

| range | what | PR |
| --- | --- | --- |
| 419181bd1..6198a81cb (19 commits) | eleven bit-identical removals of repeated work in the realtime path | [#7648](https://github.com/vllm-project/vllm-omni/pull/7648) `lingbot/exact-realtime-optimizations` |
| 910f8753c (cherry-pick of 956f56a1d) | width-sharded streaming VAE decode across the Ulysses ranks, on whenever SP > 1 | [#7651](https://github.com/vllm-project/vllm-omni/pull/7651) `lingbot/vae-decode-precision` |
| 56f2a2cee (cherry-pick of ef2d36e3e) | the realtime benchmark client and deploy configs | [#7645](https://github.com/vllm-project/vllm-omni/pull/7645) `benchmark/lingbot-world-realtime` |
| top commit | this directory | |

To rebuild it from the PR branches on the fork `https://github.com/tzhouam/vllm-omni.git` instead:

```bash
git clone https://github.com/vllm-project/vllm-omni.git && cd vllm-omni
git fetch https://github.com/tzhouam/vllm-omni.git \
  lingbot/exact-realtime-optimizations lingbot/vae-decode-precision benchmark/lingbot-world-realtime
git checkout -b lingbot-served 6198a81cb   # #7648 head
git cherry-pick -x 956f56a1d               # #7651 head (auto-merges cleanly)
git cherry-pick -x ef2d36e3e               # #7645 head (auto-merges cleanly)
```

What #7648 removes, each bit-identical: flat K/V slot pools allocated as base tensors; the source image decoded
once; camera modulation built once per AR block and reused across a stepwise block; both cross-attention
all-to-all collectives under Ulysses; K/V restaged only for what moved (opt-in `reuse_history_staging`); block-table
metadata copied asynchronously; realtime chunks delivered as uint8 frames straight from the device; the
condition no longer re-encoded once the encoder reaches a fixed point; a whole chunk per stepwise runner call;
the timestep projection staged in one static buffer; the sharded causal-conv input assembled in one pass; the
batch index mapping built without a device read-back.

What #7651 adds: the Wan decoder is split along the width across the same ranks the DiT runs on, with halo
exchange at every spatial convolution and one all-gather so every rank keeps the frame. The per-session decoder
cache shrinks by the same factor. No option; `vae_patch_parallel_size` stays rejected for this pipeline.

Draft [#7749](https://github.com/vllm-project/vllm-omni/pull/7749) (`lingbot/vae-decode-fast-path`, head
8ad08677a) adds `lingbot_vae_decode_fast_path: exact|fused` for the decoder. Its branch still carries the previous
form of #7651 (83e480737, switched on through `vae_patch_parallel_size: 4`), so use that branch head as is, with
that key, until it is rebased onto 956f56a1d.

## 5. Deploy configs (`configs/`)

All three are `benchmarks/lingbot_world/configs/usp4_compiled.yaml` with at most one added key under
`model_config.ar_diffusion_kv_config`:

| file | added key | measures |
| --- | --- | --- |
| `stock.yaml` | none | the stock number on any commit |
| `kvreuse.yaml` | `reuse_history_staging: true` | #7648's lossless set (the K/V restage reuse is opt-in) |
| `both_prs.yaml` | same as `kvreuse.yaml` | this branch: the shard needs no key under SP-4 |

Fixed in all of them: `dtype: bfloat16` (which also makes the VAE decode bf16; a separate decode-dtype option
measured as a no-op, 1027 vs 1026, and was removed), `enforce_eager: false`, `sequence_parallel_size: 4`,
`ulysses_degree: 4`, `max_num_seqs: 1`, `ar_diffusion_height/width: 480/832`, `gpu_memory_fraction: 0.6`,
`warmup_cudagraph: true`, `num_inference_steps: 4`, `flow_shift: 5.0`, `devices: "0,1,2,3"` (rank to card inside
`CUDA_VISIBLE_DEVICES`).

## 6. Serve and measure

| file | role |
| --- | --- |
| `run_arm.sh OUT [PORT] [CONFIG]` | one arm: optional `CODE_REV` checkout, provenance check, server start, wait for `/health`, client run, server stop. Env: `CODE_ROOT`, `CODE_REV`, `PY`, `MODEL`, `NUM_CHUNKS`, `SESSIONS`, `TARGET_FPS`, `OVERLAY_DIR`. |
| `chain_example.sh` | several arms in one GPU reservation with the host-RAM/load gate and three attempts per arm; edit the `measure` lines. |
| `eval_table.py ROOT [ARM ...]` | steady median/std/p99, RTF at 12 and 16 fps, stall, plus single-chunk e2e latency and per-stage split from the overlay spans. |
| `decompose.py` | helper for `eval_table.py` (span timeline per rank). |
| `overlay/sitecustomize.py` | optional: per-rank span timing into `LINGBOT_TRACE_DIR`; `LINGBOT_SYNC_DEBUG=1` arms `torch.cuda.set_sync_debug_mode("warn")` in every worker after model load and prints the vllm_omni frames of each synchronising call. First on `PYTHONPATH`. One harmless "module unavailable" line at start is expected. |
| `sync_debug_run.sh` | 12-chunk run with sync tracing armed. |

Quick start on a fresh machine:

```bash
git clone -b lingbot/reproduce-served-20260918 https://github.com/tzhouam/vllm-omni.git && cd vllm-omni
pip install -e .            # into an env that has torch 2.13 / vllm 0.29 (section 2)
pip install websockets av imageio-ffmpeg
export CUDA_VISIBLE_DEVICES=0,1,2,3
RESULTS=$PWD/results bash benchmarks/lingbot_world/reproduce/chain_example.sh   # stock, lossless, both, control
python benchmarks/lingbot_world/reproduce/eval_table.py results stock lossless both_prs both_prs_b
```

Manual equivalent for one arm:

```bash
export PYTHONPATH=$PWD                               # the repo root
python -c "import vllm_omni; print(vllm_omni.__file__)"   # must print a path under $PWD
vllm serve robbyant/lingbot-world-v2-14b-causal-fast-diffusers --omni \
  --deploy-config benchmarks/lingbot_world/reproduce/configs/both_prs.yaml \
  --port 8771 --served-model-name lingbot-world > server.log 2>&1 &
until curl -sf http://127.0.0.1:8771/health >/dev/null; do sleep 10; done   # 5-10 min: weights + compile
python benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py \
  --host 127.0.0.1 --port 8771 --model lingbot-world --target-fps 12 \
  --num-chunks 40 --sessions 3 --warmup-sessions 1 --print-chunks \
  --save-video last_session.mp4 --output-json bench.json
```

Protocol behind the table in section 0: one warm-up session (pays compile and CUDA-graph capture), then three
measured sessions of 40 chunks; steady state = chunks after the sixth (the 18-latent-frame sliding window is
full from chunk 6), pooled over the three sessions (n about 100). Fresh server per arm, one GPU reservation per
chain, `--target-fps 12` for the real-time verdict (the default is the 16 fps mux label). Measure pairs of arms
on the same card set; a single arm is not evidence. Re-measure any arm whose steady std exceeds 60 ms; that was a
noisy host.

## 7. Checks before believing a number

- `provenance.txt` in the run dir shows the served `vllm_omni.__file__`, commit, cards, model and config. If the
  path is not this repo, the number is about some other code.
- `server.log` must show all four ranks alive through the run; a worker exit with no traceback during load is
  the OOM killer, not the code.
- `bench.json` carries the git revision and hostname; quote its steady block.
- Lossless arms: with `enforce_eager: true` and the same seed the produced video is byte-identical to stock
  (compare `last_session.mp4` checksums). In compiled mode cuBLAS/cuDNN choices can differ between launches, so
  byte parity only holds run-to-run with the library pinned deterministic; a compiled-mode bit difference is not a
  bug in the lossless set.
- Quality for the lossy items (#7651, #7749 fused): watch the 40-chunk `last_session.mp4`. Reference-similarity
  metrics against a same-seed run say "is it the same video", not "is it still good".

## 8. What not to repeat

- CUDA-graph trees on the served DiT (`reduce-overhead`): 725/727 vs 716/716, worse.
- Same-device VAE/DiT overlap: the DiT saturates the cards, no gain.
- A VAE decode dtype option: `dtype: bfloat16` already decodes in bf16.
- DiT kernel fusion: a same-run nsys ledger of the served path has no fusable chain above 5 ms/chunk/rank; the
  DiT is FA3 + cuBLAS GEMMs + all-to-all (~555 of ~644 ms GPU per chunk per rank at 721).
- Per-step device syncs in shared runner code: one that crept in during review cost 50 ms/chunk (a device
  `arange` read back with `.tolist()`); `sync_debug_run.sh` finds these.

Remaining levers are lossy on the DiT: drop the commit forward (~-120 ms), 3 steps instead of 4 (~-120 ms),
FP8/INT8 GEMMs (~-50 ms); or a separate decode device to hide the VAE.
