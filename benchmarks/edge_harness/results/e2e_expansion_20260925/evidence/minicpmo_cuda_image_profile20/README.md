# MiniCPM-o image-to-text+speech: repeated WSL RTX profile

**Disposition (2026-09-25): scoped complete-request pass for one synthetic image, with a shutdown cleanup failure.** The real MiniCPM-o 4.5 BF16 checkpoint (prior provenance revision `503e754207c94da6bb26850b4469f367c9ea3582`) ran in the existing three-stage Omni/vLLM pipeline on Ryzen AI 9 HX 370 + RTX 5090 Laptop GPU, Ubuntu 26.04 under WSL2 (kernel `6.18.33.2-microsoft-standard-WSL2`), driver 610.71, PyTorch `2.13.0+cu130`, vLLM `0.29.0`, and base checkout `c9d6ef05b` with the profiler's image-input extension in the working tree. The four local shard, config and tokenizer hashes are in [model_sha256.txt](model_sha256.txt). This is one 448×448 red-square image, not an image benchmark or complete mobile/AMD NPU deployment.

The [raw report](report.json) and [request JSONL](report.jsonl) retain one warmup and **20/20 serial measured complete image→text+speech requests**. All 21 requests had unique IDs, returned the identical correct description (“The image shows a solid red square centered on a white background. There are no other elements, text, or details visible.”) and thinker token IDs, plus finite nonzero 24 kHz speech of 124,800 samples (5.2 s). Nearest-rank measured complete-request p50/p95 was **16.102/16.560 s** (range 15.887–16.896 s), excluding 96.269 s three-stage startup. Complete-request RTF was 3.10/3.18; this is neither playable streaming nor first-audio latency. The earlier [single-image output](../../../e2e_expansion_20260923/README.md) used the same deployment plan and image; the current 20 WAVs were not retained, so waveform identity across runs is not claimed.

The [GPU samples](gpu_telemetry.csv) start during startup and continue after shutdown (1,655 raw samples, with 896 through final-stage shutdown at roughly 0.5 s): device-wide VRAM use peaked at **24,047/24,463 MiB**, temperature at 71°C and GPU-only power at 99.44 W within that run window. These include any other GPU use and are not model-attributed peaks or a 30-minute thermal test. The request sampler saw WSL available RAM as low as 17.264 GB, swap use up to 1.132 GB and summed process-tree RSS up to 16.751 GB; RSS can double-count shared pages and the sampler did not cover loading peak. The 5 GiB thinker host offload, 2,048-token thinker context, 512-token talker context and one-image admission limit are fixed in the [deployment YAML](../../../e2e_expansion_20260923/evidence/minicpmo_wsl_image/minicpmo_24gb_image_offline.yaml).

The [run log](run.log.gz) shows stage 0 was force-killed at shutdown and Python's resource tracker reported **21 leaked shared-memory objects**; stage 1 and 2 shutdowns completed. This repeats a known lifecycle weakness and prevents a clean-release claim. An [initial failed launch](run_attempt1.log.gz) tried FlashInfer sampler JIT without `nvcc` in this WSL environment; the measured run explicitly set `VLLM_USE_FLASHINFER_SAMPLER=0`, matching prior CUDA MiniCPM-o runs. The profiler's raw `scope` string still says text-to-speech because it was a pre-existing fixed label; `query_type=image`, the image input hash and the actual response show what ran. The script now writes a query-specific scope label.

On the **earlier single-image WAV**, a separate [pinned Whisper tiny.en ASR proxy](prior_single_image_asr.json) transcribed only “The image shows a solid red square centered on a white background. There are no” against the full displayed answer (WER **0.286**). This is not one of the 20 measured WAVs, and one small ASR model does not prove the speech was cut off. It does leave text/speech alignment and intelligibility unqualified; the [ASR log](prior_single_image_asr.log.gz) is retained. The [derived summary](summary.json) hashes the raw files and records the request, resource and lifecycle checks.

Reproduce the measured run from the repository root with the same local checkpoint, image and plan:

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
  VLLM_TARGET_DEVICE=cuda VLLM_USE_FLASHINFER_SAMPLER=0 \
  /home/zhout/project/edge_infer/.venvs/omni-cuda-029/bin/python \
  benchmarks/edge_harness/profile_minicpmo_text_speech.py \
  --model /home/zhout/project/edge_infer/models/MiniCPM-o-4_5 \
  --deploy-config benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_wsl_image/minicpmo_24gb_image_offline.yaml \
  --query-type image \
  --image-path benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_wsl_image/red_square.png \
  --output /tmp/minicpmo_cuda_image_profile20.json \
  --warmups 1 --repeats 20 --init-timeout 900 --stage-init-timeout 600
```

Next: fix and recheck stage-0/shared-memory shutdown, retain representative measured WAVs for text/audio alignment, validate a real held-out image suite and combined audio+image/video inputs, then profile concurrency and sustained power under explicit memory admission. This run does not change the mobile or AMD NPU MiniCPM-o cells.
