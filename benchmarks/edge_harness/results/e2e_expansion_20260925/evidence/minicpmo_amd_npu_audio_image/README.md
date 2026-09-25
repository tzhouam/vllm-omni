# MiniCPM-o HX370 CPU+AMD NPU audio+image requests

This is a **scoped complete Omni request** on the HX370: one spoken-audio input
and one image enter the BF16 MiniCPM-o 4.5 thinker, which returns text and
passes through the talker and codec to produce speech. The opt-in image
resampler KV projection runs through the native-Windows AMD NPU VitisAI worker;
the rest of the three-stage model runs on WSL CPU. It extends the earlier
[image-only NPU route](../minicpmo_amd_npu_vision/README.md). It is not a
default placement or representative quality qualification.

The checkpoint revision is `503e754207c94da6bb26850b4469f367c9ea3582`,
with `model.safetensors.index.json` SHA-256
`e578de05a95804bb15237a6fd7c236414e0160cd76751da82c9f7f0d134596e7`.
The opt-in A16W8 projection graph SHA-256 is
`330bbbd0d18caaf6903aa41f836dbeaef57643a8afbaba333df68e3c1e722aeb`.
The spoken input was the pinned Qwen3-TTS ["Hello from the local computer"
WAV](../../../e2e_expansion_20260923/evidence/qwen_tts_native_cpu/qwen_tts_cpu_20260923.wav),
SHA-256 `8a41cd36a87981b0ee5b8f77e8a411a4e89b76211fd65b88ad63789f1dd813f4`.
The two images were the [red square](../../../e2e_expansion_20260923/evidence/minicpmo_wsl_image/red_square.png)
and [blue circle/green triangle](../minicpmo_amd_npu_vision/heldout_blue_circle.png),
whose hashes are in the request reports. These are generated inputs, not
natural-scene or human-recorded speech tests.

The [CPU control](cpu_report.json) and [CPU+NPU run](npu_report.json) each
completed two serial audio+image → text+speech requests in one Omni session,
with a clean three-stage shutdown. Both responses correctly transcribed the
spoken phrase and described the matching image. The [comparison audit](comparison.json)
finds **2/2 exact text-token sequences** against the CPU control and 124,800
finite nonzero output-audio samples per request on each route. Audio waveforms
were not retained or compared, and no listening-quality gate was applied.

The [raw NPU events](audio_image_kv_events.jsonl) record one graph placement
with one VitisAI node and two CPU nodes, then two live projection calls and a
clean worker close. The [gzip-compressed ORT placement trace](audio_image_kv_profile/resampler_kv_projection_32x32__2026-09-25_15-14-53_343.json.gz)
decompresses to SHA-256
`316328ed15c0f96bc84d06f588f81d701c4ab49b4db6803f1a5c54bceab78fbc`.
It records placement verification; the same open session served both request
calls. The NPU worker peak RSS was 387,694,592 bytes. Native Windows 11 build
26200 used AMD NPU driver 32.0.203.329, ORT 1.30.0 and VitisAI EP 1.8.63.0;
WSL Ubuntu used PyTorch 2.13.0+cpu, vLLM 0.28.0 and Omni 0.29 source. The
version mismatch remains a tested boundary.

CPU control startup was 97.33 s and request walls were 57.61/20.16 s. NPU
startup was 100.83 s and request walls were 60.74/20.43 s. NPU graph-stage
round trips were 35.7/162.0 ms; these are two observations, not a p50/p95
profile. The NPU run's sampled WSL RAM use reached 32.04 GB with 5.74 GB swap,
versus 31.72 GB and 5.46 GB for the separate CPU control. Run order, cache,
memory pressure and power were uncontrolled, so no speedup follows. The route
remains opt-in under the whole-chain benefit rule.

Reproduce from the repository root with `profile_minicpmo_image_suite.py`,
the two image paths above, `--audio` pointing to the pinned WAV, and the
[CPU config](cpu_audio_image.yaml) or [CPU+NPU config](cpu_npu_audio_image.yaml).
Use `VLLM_TARGET_DEVICE=cpu`, `VLLM_CPU_KVCACHE_SPACE=1`,
`OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `CUDA_VISIBLE_DEVICES=`,
`VLLM_ENABLE_V1_MULTIPROCESSING=0` and the pinned `omni-cpu` environment.
The [as-run NPU config](cpu_npu_audio_image_as_run.yaml) preserves the exact
SHA-256 in `npu_report.json`; its event/profile paths pointed to the adjacent
image-only evidence directory. Those raw files were moved here after the run,
and the reproducible config now points here. Run `audit_minicpmo_image_suite.py`
on both reports and the event log to check matched inputs, complete outputs,
actual NPU placement and worker closure.

Next: test natural images and human-recorded speech with defined output-quality
gates, retain and compare output waveforms, then run paired warmed complete
requests with explicit shared-RAM loading peaks, cancellation/restart and
sustained power/thermal measurements before considering default selection.
