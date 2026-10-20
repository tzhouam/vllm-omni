# Codec-token streaming (on-device decoding)

For edge deployments the expensive part of a TTS pipeline that must run near
the user is the audio decoder, while the autoregressive talker can stay on a
server. Codec-token streaming serves the **talker only** and streams its codec
frames to a client that decodes them locally (Qwen3-TTS: 16 codebooks at
12.5 Hz, about 400 B/s of payload before framing).

## Server

Deploy the talker-only pipeline:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni --port 8001 \
  --deploy-config qwen3_tts_talker_only.yaml --trust-remote-code
```

`qwen3_tts_talker_only.yaml` selects the registered `qwen3_tts_talker_only`
pipeline (stage 0 only, `final_output_type: latent`). The talker publishes its
per-step `codes.audio` rows to the client (`omni_client_multimodal_output_keys`),
and the WebSocket handler emits them as binary frames.

## Protocol (`/v1/audio/speech/stream`)

Client → server: the usual `session.config` with

```json
{"type": "session.config", "output_mode": "codec_tokens", "stream_audio": true,
 "response_format": "pcm", "voice": "vivian", "task_type": "CustomVoice"}
```

then `input.text` / `input.done`. `input.cancel` aborts the in-flight
generation (works for PCM sessions too).

Server → client per utterance:

| message | content |
|---|---|
| `codec.start` (JSON) | `codebooks`, `codebook_size`, `frame_rate_hz`, `sample_rate`, `decoder_id`, `header_format` |
| binary frames | header `<IHB` = `seq:uint32, n_frames:uint16, flags:uint8`, then `int16[n_frames × codebooks]`; flags `EOS=1`, `SEGMENT_END=2`, `RESET=4` |
| `codec.done` (JSON) | `total_frames`, `seq_last`, `error`, `cancelled` |
| `session.done` | as for PCM sessions |

Frames are emitted as they are produced (one talker step = one frame); the
chunking policy for the decoder (first small chunk, ramp to 25 frames, 72-frame
left context) is the client's choice. The wire format lives in
`vllm_omni/entrypoints/openai/codec_stream.py`.

## Client

`examples/online_serving/text_to_speech/qwen3_tts/edge_decoder_client.py`
loads `Qwen3TTSTokenizerV2Decoder` from the checkpoint's `speech_tokenizer/`
folder (CPU or CUDA), applies the same chunk schedule and state carry-over as
the server's Code2Wav stage, writes a WAV and reports TTFA, playback-start
latency and RTF. `codec_parity_check.py` compares the client-decoded audio with
the full two-stage server's output (same `tts_local_seed`) and reports SNR.

## Limits

- Only adapters with `supports_codec_stream = True` (Qwen3-TTS today) accept
  `output_mode: codec_tokens`; others return an `error` message.
- The talker's per-step output carries the cumulative code tensor, so a long
  utterance costs O(N²) bytes between engine and API server (about 9 MB for
  30 s of audio); the wire stream itself is incremental.
- `word_timestamps` and `speed != 1.0` are not available in this mode.
