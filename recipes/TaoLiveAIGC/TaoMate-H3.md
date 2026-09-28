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
| `taomate_h3_log_timings` | false | Log per-phase stage timings (adds device synchronizations) |
| `taomate_h3_hold_for_prompt` | false | Just-in-time lock: request `k >= 1` starts only after a `session.interaction` prompt update arrived since request `k-1` started; until then the stream idles at the request boundary (send `session.ping` to keep the stall timer fresh). For clients that choose every request's prompt at the last moment |
| `taomate_h3_hold_poll_seconds` | 0.02 | Idle-step period while a request boundary is held |
| `taomate_h3_teacher_cuda_graph` | false | Replay the audio teacher's nine forwards from CUDA graphs, one graph per document shape (prompt length, first or later request, reference tail or not). The first capture is checked with `torch.cuda.set_sync_debug_mode("error")`; any capture failure falls back to eager for the rest of the process, agreed across the Ulysses group |
| `taomate_h3_cuda_graph_max_entries` | 16 | Resident teacher graphs (least recently used shape evicted); they share one memory pool |
| `taomate_h3_warmup_requests` | 4 | Requests of the load-time warmup session. Later requests cycle through three audio latent counts (198, 198, 199 per channel), so four requests visit every teacher document shape: the graphs are captured and every phase size has been allocated before the first client connects |
| `taomate_h3_teacher_graph_text_lengths` | unset | Prompt token counts (`"lo-hi"`) whose teacher graphs are captured during the load-time warmup, three document shapes per count (about 0.5 s and 5 MB each, estimate); these graphs are pinned against LRU eviction by later shapes. A teacher graph is keyed by the prompt's token count, so without this each new prompt length captures inside the stream (about 0.7 s per shape). Needs `taomate_h3_pad_text_tokens` and enough `taomate_h3_cuda_graph_max_entries` |
| `taomate_h3_text_encoder_cuda_graph` | false | Replay the text-only prompt encode of a prompt update from a CUDA graph (one graph per token count, captured for `taomate_h3_teacher_graph_text_lengths` at load, exact: the encoder's own modules run with graph-safe indexing). Prompts with images or videos, offloaded encoders and non-encoder ranks keep the eager path |
| `taomate_h3_vae_decoder_tile_size` | unset (checkpoint: 256) | Decoder tile edge of the video VAE in pixels (multiple of 16). At 480x864 the checkpoint's 256 px tiles form a 3x5 grid covering 2.4x the canvas; 480 gives two 480x480 tiles (1.1x the canvas), one per tile rank at USP2. Fewer tiles than `vae_patch_parallel_size` falls back to the slower whole-frame decode (a warning is logged) |
| `taomate_h3_vae_decoder_tile_overlap_min` | unset (checkpoint: 64) | Minimum overlap between decoder tiles in pixels (multiple of 16) |

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

A 13-request session (1552 frames, 64.6 s of video and audio) took 3.9-4.1 s of wall time
per 4.958 s request in steady state (real-time factor 0.78-0.82); the first chunk arrived
after 2.3-2.9 s with the load-time warmup. Stage timings per 34-frame phase
(`model_config.taomate_h3_log_timings: true`): teacher 0.7-0.8 s per request, three student
forwards 0.42 s, clean commit 0.15 s, video decode 0.2 s, audio decode 0.03 s. Regional
`torch.compile` (`enforce_eager: false`, `VLLM_OMNI_TORCH_DYNAMO_RECOMPILE_LIMIT=64`) and
FP8 online linears (`quantization: fp8`) gave no steady-state gain over eager here (4.3 s per
request); the launch-bound teacher forwards and the per-phase VAE decode are the next targets.

## Two GPUs (USP2): `vllm_omni/deploy/taomate_h3_usp2_realtime.yaml`

The two-GPU config keeps TP=1 and splits the heads across two Ulysses ranks
(`sequence_parallel_size: 2`, `text_encoder_tp_size: 2`, `vae_patch_parallel_size: 2`).
Memory per rank in eager BF16 is 100.7 GB after load (measured locally). Eager BF16 is
**not** real time on two H200-class GPUs at 480x864 (measured locally, 2026-09-28, five
requests, constant prompt, `taomate_h3_log_timings: true`, rank 0):

| Stage | 34-frame phase | 17-frame phase |
| --- | --- | --- |
| audio teacher (once per request, 9 forwards) | 0.78-0.84 s | - |
| 3 student forwards | 0.77-0.81 s | 0.43 s |
| clean-commit forward + cache append | 0.29-0.32 s | 0.16 s |
| tile-parallel video VAE decode | 0.35-0.37 s | 0.18 s |
| audio VAE decode | 0.03 s | 0.03 s |

Per request: 2.29 + 1.55 + 1.59 + 0.86 = 6.3 s of wall time for 4.958 s of content
(real-time factor 1.27; the client saw 6.4 s between request starts). Against the USP4
breakdown the DiT forwards take 1.9x longer per rank (2200 instead of 1100 rows through the
linears, twice the heads in attention), the VAE decode 1.8x, and the launch-bound teacher
is unchanged.

The two-GPU deploy config therefore enables the levers that leave the generated content
unchanged, plus a decoder tiling that only changes how the VAE is split across the ranks:

- `quantization: fp8` for the DiT linears (the GEMMs are about 60% of a student forward at
  USP2, estimate from parameter and row counts);
- `taomate_h3_teacher_cuda_graph` (the teacher's forwards run a few hundred rows and are
  launch-bound at about 85-90 ms each; a replay takes 38 ms);
- `taomate_h3_vae_decoder_tile_size: 480` (two 480x480 tiles, one per rank, instead of the
  checkpoint's 3x5 grid of 256 px tiles that decodes 2.4x the canvas);
- `taomate_h3_pad_text_tokens: 256` so every document kind has one packed length, and
  `taomate_h3_teacher_graph_text_lengths: "1-96"` so the graphs of every plausible prompt
  length exist before the first client connects.

Measured locally (2026-09-28, 480x864, two H200-class GPUs, five requests, constant prompt
of 32 tokens, `taomate_h3_log_timings: true`, rank 0; the warmup already holds the graphs):

| Stage | 34-frame phase | 17-frame phase |
| --- | --- | --- |
| audio teacher (once per request, 9 graph replays) | 0.35 s | - |
| 3 student forwards (FP8) | 0.67-0.70 s | 0.42-0.45 s |
| clean-commit forward + cache append | 0.24-0.26 s | 0.15-0.17 s |
| tile-parallel video VAE decode (2 x 480 px tiles) | 0.21-0.22 s | 0.10 s |
| audio VAE decode | 0.03 s | 0.03 s |
| phase preparation (packing, RoPE table) | 0.01-0.03 s | 0.00 s |

Per request the phases take 1.51 + 1.19 + 1.24 + 0.76 = 4.70 s of stage wall time (stage
ends synchronized with the device by the timing log), and the wall
time between the ends of consecutive requests (phases plus the runner's output handling) is
4.91-4.97 s for 4.958 s of content: real-time factor 0.99-1.00, against 1.27 for eager BF16.
Without the timing instrumentation a ten-request session gave 4.98 s per request on average
between the first chunks of consecutive requests (range 4.70-5.10 s; measured locally), so
the device synchronizations of the log cost nothing. This is real time with no headroom: a
playback buffer of about one second is needed, and the two events below each stall the
stream once. The 0.05 s per phase between the end of a phase and the next preparation is
the runner's output transport (34 uint8 frames, 42 MB, leave the worker process per phase).

- **A prompt length without a resident graph** captures the teacher graphs of that length
  inside the stream: about 0.7 s per shape (three shapes for a session's first prompt, two
  per later prompt update), plus one or two slow commits (0.5-0.9 s) right after a capture
  while the allocator regrows. Measured without the length range: a session start cost
  2.1 s before the first chunk, and each prompt update with a new token count stretched
  the next request by 2-3 s. With `taomate_h3_teacher_graph_text_lengths: "8-96"` the
  warmup captured 267 graphs in 149-165 s (measured locally; about 1.3 GB of static inputs)
  and the live sessions captured nothing.
- **A prompt update** re-encodes the text on the workers (Qwen3-VL at TP2): 0.13-0.27 s
  for 21-32 tokens, inside the phase in which the update arrives, so a request that also
  applies an update took 5.1-5.3 s of wall time (measured locally). A client that changes
  the prompt every request therefore runs at real-time factor 1.03-1.07 and drains its
  buffer by 0.2-0.3 s per request; one that changes it every few requests stays level.
  Measured over a 100-request session with a different prompt (3-60 words) every request:
  5.09 s per request once warm (requests 20-100; real-time factor 1.03), 20 s of drift over
  the 101 requests, four graph captures for a 3-token prompt because the range then started
  at 8 tokens (the config now starts at 1).

**Prompt length caveat.** Every number above is for prompts of 3-60 words (up to 96 tokens
of the FL2VA tokenizer). Persona prompts of the live agent measure 345-360 tokens (measured
locally by a peer session), above the config's `taomate_h3_pad_text_tokens: 256`: such
prompts run unpinned (one warning), so every teacher document shape is captured lazily and
the pinned-length assumptions do not hold. For that workload set `taomate_h3_pad_text_tokens`
at or above the longest prompt (e.g. 448) and a graph range around the real lengths (e.g.
`"300-430"`, about 390 graphs), and expect the larger documents (about 320 more text rows per
phase) to run somewhat slower than measured here; the two-GPU request period for that setting
is not measured yet. On four GPUs (the USP4 recipe, eager BF16, `taomate_h3_hold_for_prompt`
on) persona prompts of 330-405 tokens with a prompt update on almost every request measured a
median of 4.29 s and a maximum of 4.53 s between consecutive requests' first chunks over 54
requests, real-time factor about 0.86 (measured locally by a peer session, 2026-09-28).

Next levers, in order of expected gain per effort (estimates from the stage breakdown, not
measured): move the frame transport off the step loop or into shared memory (0.2 s per
request, runner change); hide or graph the prompt re-encode (0.13-0.27 s per update);
serve the student from a second FP8 weight set with the LoRA merged in (about 300 fewer
launches per forward, 0.2-0.25 s per request, small numeric change); CUDA graphs for the
student forwards over preallocated KV history buffers (the forwards are launch-bound at
about 60 ms of CPU per forward, up to 0.9 s per request, large change).

Teacher graphs are keyed by the prompt's token count because the H3 attention treats the
document's valid rows as a prefix whose length is a Python int of the forward (a fixed
text length would need extra rows inside that prefix, which changes the attention), so
the graphs of a length range are captured up front instead. Reusing the last denoise
step's K/V instead of the clean-commit forward (the LingBot-World `reuse_last_step_kv`
pattern) is not offered: TaoMate's last student forward runs at sigma 0.853 of the shift-12
schedule (the ladder is 1.0, 0.961, 0.853, 0), far from the clean K/V the model was trained
to attend to.

## Limits

- One session per server (`max_num_seqs: 1`, `session_capacity: 1`).
- The canvas is fixed per deployment; a request with another size is rejected.
- Only text prompts (T2VA); the TaoMate release has no image or audio conditioning.
- Video is not bit-reproducible run to run (the release runtime has the same property).
