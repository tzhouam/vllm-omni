# Qwen3-TTS reliability probe startup failure

This is a retained failed run of `check_tts_reliability.py` against the pinned
Qwen3-TTS 0.6B CustomVoice checkpoint on WSL2 RTX 5090 Laptop. The
[raw report](report.json) says stage initialization failed before any speech
request. The [compressed startup log](startup.log.gz) identifies the root:
vLLM's dummy sampler selected FlashInfer top-k/top-p sampling, whose JIT path
raised `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`.
The installed CUDA Python packages include an `nvcc` binary in a nondefault
location, but this harness had not set the sampler option used by the
successful streaming profiler. No output, abort, or crash behavior was
measured. The first identical failed attempt retained its
[report](../qwen_tts_cuda_exact_reliability_33w/report.json) without a full log.

The harness was changed to default `VLLM_USE_FLASHINFER_SAMPLER=0`, matching
`profile_local_tts.py`. The subsequent [real reliability run](../qwen_tts_cuda_exact_reliability_33w_fixed/README.md)
passed abort, late-audio, recovery and explicit worker-failure checks. This
failure is a launch-configuration issue, not evidence that the checkpoint or
the GPU cannot run Qwen3-TTS.
