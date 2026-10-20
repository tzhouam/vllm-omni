# Edge benchmark results

Committed JSON produced by the edge branch benchmarks (kept small; WAVs are never committed):

- `stream_latency/<model>/<deploy>/<timestamp>[_tag].json` — `stream_latency_bench.py` output (schema 1): `meta` (model, deploy config, git SHA, GPU), `init` (`init_s`, optional init timeline), `resources` (peak RSS, peak GPU MiB, sampling mode), `requests[]` (per-chunk arrivals, TTFA, playback-start, stall at TTFA, RTF streamed / total) and `summary` (p50/p90/max).
- `init_profile/<timestamp>/summary.json` — `init_profile.py` comparison of initialization phases per configuration.
- `hw_emulation/<timestamp>/summary.json` — `hw_emulation.py` matrix (emulated hardware class x TTFA / playback-start / RTF / RSS; per-case bench JSON, `hw_profile.json` and init timeline next to it).
- `cpu_sweep/<timestamp>/summary.json` — `cpu_thread_sweep.py` (CPU-only 8/16/32 threads + inductor variant, with per-step attribution).
- `stream_latency/step_stats/<timestamp>/*.json` — raw per-process counters from `--step-stats` (merged into the bench JSON under `step_stats`).
- `codec_stream/` — `codec_stream_e2e.py` parity (server PCM vs client-decoded codec tokens, SNR) and client latency.
- `stream_latency/timelines/` — init timelines (`VLLM_OMNI_INIT_TIMELINE`) of the committed runs.

Metric definitions live in `benchmarks/tts/stream_latency_metrics.py`.
