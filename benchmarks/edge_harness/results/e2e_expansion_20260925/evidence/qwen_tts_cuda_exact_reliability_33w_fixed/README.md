# Qwen3-TTS exact-state CUDA abort and recovery

This is one targeted current-source two-stage `AsyncOmni` reliability run on
WSL2 HX370 + RTX 5090 Laptop with pinned Qwen3-TTS 0.6B CustomVoice revision
`85e237c12c027371202489a0ec509ded67b5e4b5`. It used the same `edge`
deployment profile and `VLLM_USE_FLASHINFER_SAMPLER=0` setting as the earlier
[20-stream exact-state profile](../qwen_tts_cuda_exact_live20/README.md).
The [GPU power reading after the run](gpu_power_state_after_run.txt) still
showed the 33 W cap; the cap was not sampled continuously during this probe.
This tests request lifecycle, not real-time throughput, broad speech quality
or sound-device playback.

The [raw report](report.json) and [compressed worker log](run.log.gz) retain
the following observed outcomes:

| Gate | Result |
|---|---|
| Baseline short request | Six finite events, 111,360 audio samples, terminal output |
| Slow consumer | Five finite events, 105,600 audio samples, terminal output after deliberate three-second pause |
| Abort after first audio | 3,840 samples received before abort; acknowledgement 1.802 ms |
| Late-output fence | One later terminal event with **zero audio samples**; no late PCM |
| Fresh request after abort | Five finite events, 96,000 audio samples, terminal output |
| Owned vocoder failure | Deliberate termination reported `OmniEngineDeadError` after 3.749 s |
| Owned worker cleanup | Zero processes required harness cleanup |

This verifies the late-audio fence and fresh-request recovery for the tested
single-request stream. The stage failure is explicit; it does not prove that
the same failed stage can restart without a new engine. The separate
[power-limited throughput run](../qwen_tts_cuda_exact_sustained30/README.md)
failed simulated playback for every complete request, so playable streaming
under that condition remains unqualified.

The first two attempts stopped during stage startup because this harness had
not disabled FlashInfer sampling, and FlashInfer attempted to JIT-build a
kernel without a discoverable CUDA `nvcc`. The [retained startup failure](../qwen_tts_cuda_exact_reliability_33w_retry/README.md)
records the root error. The harness now sets the same sampler option as
`profile_local_tts.py`; this successful run is the validation of that change.

Reproduce with a fresh output directory:

```bash
PYTHONPATH=/home/zhout/project/edge_infer/vllm-omni-edge \
  /home/zhout/project/edge_infer/.venvs/omni-cuda-029/bin/python \
  benchmarks/edge_harness/check_tts_reliability.py \
  --model /home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  --out /tmp/qwen_tts_exact_reliability_fresh
```
