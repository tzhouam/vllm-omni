# Gemma 4 pinned artifact lineage probe

`probe_gemma_lineage.py` observes the exact Gemma 4 QAT Q4 GGUF and F16 vision
projector used by the native Agent route. It checks the pinned download manifest
and source metadata, fetches only two small public Hugging Face JSON responses
at the exact commit (model metadata and file tree), and compares their license,
revision, LFS SHA-256 IDs, and sizes. Only after that match does it hash the two
already-downloaded local files. Local file names do not establish lineage.

From the repository root, run with native Windows Python and this checkout on
`PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Get-Item .).FullName
python -X utf8 -m benchmarks.edge_agent.experiments.probe_gemma_lineage
```

The command neither downloads model bytes nor sends any authentication header.
It writes a unique, gitignored `benchmarks/edge_agent/results/gemma_lineage_*.json`
with source/config/manifest hashes, pinned API response hashes, observed local
byte hashes, and a whole-record SHA-256. `status=observed` means the official
artifact commit, file IDs, declared license, and local bytes agreed. It is
still **candidate evidence for human review**, not a signed qualification or
a claim that the declared unquantized parent checkpoint was independently
verified. The raw record stays private; a reviewer may bind it to a separate
`checkpoint_lineage` gate receipt after examining the source and license.

## Native placement snapshot

`snapshot_runtime_placement.py` independently rehashes a completed native
cancel/recovery record, its model, projector, and llama-server binary, then
copies the sanitized startup log into a new content-addressed private file.
It checks one unambiguous offload report, every layer assignment, model buffer
reports, the exact model-loaded marker, the configured GPU name, and the
recovered request's Omni worker generation, stream, and terminal event. It
refuses to overwrite a prior snapshot. Run it with native Windows Python:

```powershell
$env:PYTHONPATH = (Get-Item .).FullName
python -X utf8 -m benchmarks.edge_agent.experiments.snapshot_runtime_placement `
  --record benchmarks/edge_agent/results/native_cancel_gemma4_20261006.jsonl `
  --startup-log benchmarks/edge_agent/results/native_cancel_gemma4_20261006.gemma4-31b-qat-q4-windows-hybrid40.log `
  --output benchmarks/edge_agent/results/gemma4_placement_snapshot_<unique>.json
```

The raw input and snapshot remain in gitignored `results/`. The snapshot is
`observed_not_qualified`: the copied log was saved after the recovered worker
finished and does not embed its PID or generation. Its association with the
second worker comes from the harness manifest and recovered stage event.
Observed 40/61 layer offload and CPU/Vulkan model buffer sizes do not verify
each operation's compute placement, prove latency benefit, or satisfy the
independent signed placement gate. A reviewer must inspect the raw files and
bind any gate receipt to the exact full-profile identity separately.

## Native cancel/recovery and profile binding

`native_cancel_recovery.py` runs one batch-1 stream, cancels after its first
delta, releases the worker and state, then sends one recovery request. Its
record and sanitized startup-log copy use exclusive creation; rerunning with
the same output name cannot replace earlier evidence. From native Windows
Python, pass `--config <native config> --record <private new .jsonl path>`.

For a **new** post-profile cancellation or memory-admission run, add
`--profile-index <private index.json>`. Both probes audit the full profile's raw
requests, then require the same config, route, stable hardware identity, OS,
GPU driver, power condition, imported Omni source, and loaded runtime before
model work begins. A successful bound manifest contains the nested
`omni-agent-profile-binding-v1` object with exact profile-index and raw-request
SHA-256 hashes. An unbound run records `profile_binding: null`; earlier runs
must be repeated after profiling, not relabeled. These are private diagnostic
observations and do not sign qualification gates.
