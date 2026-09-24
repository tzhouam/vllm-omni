# Qwen3-TTS exact CUDA state in a live Omni request

The current-source two-stage Omni pipeline ran the pinned Qwen3-TTS 0.6B CustomVoice checkpoint on WSL2 HX370 + RTX 5090 Laptop with `async_chunk: true`. Stage 0 generated speech codes and stage 1 delivered PCM through the new x-vector sliding-KV decoder route. The [raw run report](report.json), [per-request events](requests.jsonl) and five [retained warmup PCM chunks](audio_chunks/) are the evidence. The five archived chunk hashes match their request-event hashes. This is a short scoped full text-to-audio execution, not a sustained or quality-qualified release result.

The process initialized both admitted CUDA stages in **79.41 s**. Four serial requests completed with finite audio and request IDs in order:

| Phase | Text band | Total audio | Request wall | TTFA | Total RTF | Simulated underruns |
|---|---|---:|---:|---:|---:|---:|
| Warmup | Short | 3.28 s | 1.011 s | 508.5 ms | 0.308 | 0 |
| Measured | Short | 2.80 s | 0.603 s | 84.7 ms | 0.215 | 0 |
| Slow consumer | Medium | 7.36 s / 92 codec frames | 1.619 s | 98.3 ms | 0.220 | 1 |
| After abort | Short | 3.04 s | 0.635 s | 101.0 ms | 0.209 | 0 |

The medium request crossed the decoder's 72-frame attention window and completed while the consumer deliberately slept 0.2 s after each event; its single simulated playback deficit therefore does not establish an intrinsic decoder stall. An in-flight long request was aborted after its first output with **2.99 ms acknowledgement**; the harness did not verify a late-audio fence. A subsequent short request and both stage shutdowns completed. One warmup and one measured short request are not a latency distribution. Actual sound-device playback, transcript/speaker/listening quality, parity to a full decoder on *these* generated codes, long-running memory and power remain unverified.

The raw report identifies source import paths, vLLM/Omni/PyTorch versions, the model revision, the `edge` deployment profile and sampled memory. The run loaded `qwen3_tts_code2wav.py` SHA-256 `aa6d326ec420cbd91ec5fd3ebbd231b59c36fc7e3f06a552e9bfc3ba55572478` and `segmented_graph_wrapper.py` SHA-256 `2d8467bc961d36033c9e951c91864bf45556cd8ec3a0662efb94fb1018f4fb3c`; the report marks the checkout dirty because these edits had not yet been committed. The 0.25 s memory sampler observed 12,415,819,776 bytes peak device-wide GPU allocation and a 25,567,870,976-byte minimum available WSL host RAM. It is a sampled lower bound and includes other GPU processes. The [separate fixed-code numerical audit](../qwen_tts_cuda_incremental/README.md) demonstrates full-decode parity for 117 generated frames, including 25- and two-frame schedules; this live run checks complete behavior but does not repeat that parity calculation.

Reproduce from the repository root with the CUDA 0.29 environment:

```bash
PYTHONPATH=/home/zhout/project/edge_infer/vllm-omni-edge \
  /home/zhout/project/edge_infer/.venvs/omni-cuda-029/bin/python \
  benchmarks/edge_harness/profile_local_tts.py \
  --model /home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  --out benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_cuda_exact_live \
  --repeats 1 --sustained-seconds 0 --length-band short --concurrency 1
```

The output directory must not already exist for a rerun. Choose a new evidence directory to preserve these files.
