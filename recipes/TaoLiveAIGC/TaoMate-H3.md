# TaoMate-H3: realtime streaming MiniMax-H3 with the TaoMate LoRA

TaoMate-H3 ([TaoLiveAIGC/TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3)) is a
rank-128 LoRA over [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) that turns the
bidirectional five-second text-to-video-and-audio model into a causal live streamer: a
presenter that keeps talking and acting while the prompt of each next five-second request
is chosen just in time. vLLM-Omni hosts it on the AR-Diffusion realtime runtime with pure
Ulysses sequence parallelism.

## How it works

Each five-second request (124 native frames for the first request, 119 afterwards, at
24 fps with 32 kHz stereo audio) is generated as four causal phases of 34/34/34/17 frames:

| Step | What runs | Where |
| --- | --- | --- |
| Prompt | Qwen3-VL text encoder (tensor parallel across the Ulysses ranks) | at session start and whenever the client updates the prompt |
| Audio teacher | base MiniMax-H3 (LoRA disabled), nine audio-only forwards over `[text \| 1 s clean reference tail \| new audio]` | once per request, before its first phase |
| Phase denoise | three LoRA forwards of the phase document `[text \| audio \| video]`, attending to text, the persistent clean audio/video KV of earlier phases and the current chunk | per phase |
| Clean commit | one sigma-zero forward that recomputes the phase's K/V from its clean latents and appends them to the KV history (first-phase video sink + two most recent phases; audio history dropped every twelve requests) | per phase |
| Decode | tile-parallel temporal-window video VAE decode plus sliding-window audio VAE decode | per phase, streamed as one fragmented-MP4 chunk with an AAC track |

The LoRA is never merged: the same resident BF16 weights serve the LoRA student and the
LoRA-free audio teacher, so one 62 GB DiT copy per rank is enough. TP is fixed at 1; the
attention heads are split across the Ulysses ranks and each rank keeps the clean KV
history of its own head shard.

## Requirements

- 4 GPUs with at least 96 GB each (measured 85 GB per rank at 480x864 after load: DiT
  62 GB, text encoder TP4 16 GB, VAEs, LoRA 1.2 GB, model-owned session state 12.6 GB).
- Hopper or newer (`fa3_fwd_interface` / FlashAttention-3 dense kernels; FA4 on Blackwell).
- The MiniMax-H3 snapshot (`FL2VA/` partition) and the TaoMate-H3 adapter directory
  (`adapter_model.safetensors` + `config.json`).

## Serve

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/taomate_h3_usp4_realtime.yaml \
  --lora-path /path/to/TaoMate-H3 --port 8000
```

`--model` must point at the `FL2VA/` partition directory (the snapshot root carries the
Diffusers modular index and is resolved as the FastH3 modular checkpoint). The deploy
config pins `tensor_parallel_size: 1`, `ulysses_degree: 4`, `text_encoder_tp_size: 4`,
`vae_patch_parallel_size: 4`, the AR-Diffusion engine, step execution and streaming output.
Per-deployment knobs live under `model_config`:

| Key | Default | Meaning |
| --- | --- | --- |
| `taomate_h3_width` / `taomate_h3_height` | 480 / 864 | The fixed session canvas (32-aligned, short edge 480, 768 or 1088) |
| `taomate_h3_seed` | 8301 | Default seed; request `k` draws audio noise from `seed + k` and video noise from `seed + k * 1000003` |
| `taomate_h3_audio_kv_reset_requests` | 12 | Drop the audio KV history every N requests (video sink and recents are kept) |
| `taomate_h3_allow_no_lora` | false | Stream the base H3 without the adapter (debugging only) |
| `taomate_h3_pad_text_tokens` | unset | Prompt token budget that pins every phase and teacher document to one packed length per kind (for compiled or CUDA-graph runs) |

The deploy config keeps `ar_diffusion_kv_config.warmup_cudagraph: true`: the AR runner runs
one throwaway five-second request at load time (the pipeline opts into this warmup in eager
mode as well), so the first chunk of a session arrives in about 3 s instead of 17-20 s on
a cold server.

## Stream

Open `WS /v1/realtime/video`, send `session.start` with the prompt, `width`, `height`,
`fps: 24`, `seed` and `num_frames` (the frame budget decides how many five-second
requests the session runs; `124 + 119 * (n - 1)` frames is `n` requests), and receive one
binary fragmented-MP4 chunk per phase. The bundled client works unchanged:

```bash
python examples/online_serving/streaming_video_generation/streaming_video_client.py \
  --port 8000 --model /path/to/MiniMax-H3/FL2VA --size 480x864 --fps 24 --num-frames 362 \
  --seed 8301 --no-helios-distilled-preset --prompt "..." \
  --prompt-updates '[{"at": 6.0, "prompt": "..."}]'
```

A `session.interaction` prompt update is applied at the next chunk boundary and takes
effect for the *next five-second request* (the audio teacher and all four phases of a
request share one prompt), which is TaoMate's just-in-time prompt lock. Every chunk's
`video.chunk_metadata` carries `num_frames`, `num_audio_samples` and `audio_sample_rate`;
the stream has an H.264 video track and an AAC audio track. Chunk 1 of a session is
39 frames of latents but 34 frames of pixels: the decoder holds the last five frames of
each temporal window until the next phase, and flushes them with the final chunk.

## Measured (4x H200-class 141 GB, eager BF16, 480x864)

Steady-state chunk period after the cold start, one session, measured locally:

| Chunk | Frames | Period |
| --- | --- | --- |
| phase 0 of a request (includes the teacher's nine forwards and, after a prompt update, the text encode) | 34 | 1.7-2.3 s |
| phases 1-2 | 34 | 0.85-0.95 s |
| phase 3 | 17 | 0.7-0.85 s |

A 13-request session (1552 frames, 64.6 s of video and audio) took 4.1 s of wall time per
4.958 s request in steady state (real-time factor 0.82); the first chunk arrived after
2.9 s with the load-time warmup. Regional `torch.compile` (`enforce_eager: false`,
`VLLM_OMNI_TORCH_DYNAMO_RECOMPILE_LIMIT=64`) gave no steady-state gain over eager here.

## Limits

- One session per server (`max_num_seqs: 1`, `session_capacity: 1`).
- The canvas is fixed per deployment; a request with another size is rejected.
- Only text prompts (T2VA); the TaoMate release has no image or audio conditioning.
- Video is not bit-reproducible run to run (the release runtime has the same property).
