# MiniCPM-o HX370 CPU+AMD NPU abort and fresh natural-input request

**Disposition (2026-09-25): scoped same-engine recovery passed.** An
`AsyncOmni.generate` request with recorded speech and the Chelsea cat photo
was aborted after its first streamed text token. The same engine then completed
a request with the same speech and a different astronaut photo. Both
requests routed the thinker's resampler KV projection through the native
Windows VitisAI worker; thinker, talker, codec and resampler suffix stayed on
WSL CPU. This is an opt-in recovery experiment, not default placement.

The [passing report](cancel3_report.json) records 107.85 s stage startup,
47.27 s to the cat's first text token, and a 3.27 ms abort acknowledgement.
Two terminal events arrived after abort but carried **zero audio samples**.
The fresh astronaut request finished in 39.66 s, returned nonempty text
describing both the narrated ship and the woman in an orange spacesuit, and
emitted 265,920 finite 24 kHz audio samples. The [independent audit](cancel3_audit.json)
checks distinct image hashes, the nonterminal cancellation point, no late
audio, a terminal fresh event, and complete NPU event groups.

The [raw NPU events](cancel3_kv_events.jsonl) show one VitisAI and two CPU
nodes in the graph session, one call for the cat's 1,014 projection rows,
four tiles for the astronaut's 3,105 rows, request groups `[1,2]`, and
clean worker closure after five calls. Same-input BF16 CPU projection
relative L2 was 0.640%/0.647%, below the configured provisional 1% gate.
The [ORT profile](cancel3_kv_profile/) and [driver log](cancel3_driver.log)
retain provider and lifecycle details. The worker's reported peak RSS was
394,739,712 bytes; it is not total CPU+NPU shared-RAM pressure.

Two earlier attempts are retained. The [first report](cancel_report.json)
and [log](cancel_driver.log) show that the harness itself raised
`AttributeError` when it treated a text-only `CompletionOutput` as
multimodal output after a successful first-token abort. The corrected
[second report](cancel2_report.json) and [events](cancel2_kv_events.jsonl)
showed zero late audio and a complete fresh cat request, but only one NPU
projection call: the repeated image reused a cached image representation.
That run could not prove a fresh NPU graph execution. The final run changed
the fresh image and required two NPU request groups and five calls.

The pinned checkpoint revision is
`503e754207c94da6bb26850b4469f367c9ea3582`, and the ONNX graph
SHA-256 is
`330bbbd0d18caaf6903aa41f836dbeaef57643a8afbaba333df68e3c1e722aeb`;
the exact graph is [retained with the prior waveform evidence](../minicpmo_amd_npu_waveform/resampler_kv_a16w8.onnx).
The as-run [deploy config](cpu_npu_audio_image_cancel3.yaml) pins the
numerical gate, model, backend and stage limits. Its graph path points to
the local original copy; set that path to the retained artifact when
reproducing elsewhere. The [source fixtures and attribution](../minicpmo_amd_npu_natural/inputs/input_manifest.json)
and the [probe](../../../../experiments/probe_minicpmo_npu_cancel.py) pin
the inputs and request sequence. Re-run the
[audit](../../../../experiments/audit_minicpmo_npu_cancel.py) on the final
report and event log.

The abort happened after NPU projection but before a produced audio chunk.
This does not test interruption inside an NPU call, cancellation after
PCM begins, worker-crash recovery, long sessions, broad input quality,
paired latency benefit, loading peak, or sustained power and thermal use.
