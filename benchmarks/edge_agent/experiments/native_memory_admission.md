# Native Windows memory-admission probe

This independent diagnostic loads one exact, hash-pinned Omni Agent route and
runs one complete Agent turn at batch size 1 with one active request. It reads
host-wide used RAM from `psutil` and NVIDIA GPU-0 used VRAM from NVML before
loading, during cold loading, during the complete request, and after release.
It records every timestamped reading, the actual Omni execution plan, the
host coordinator's reserved bytes before and after release, and the result of
an actual `ResourceLedger` refusal for each pool at its measured controller
ceiling plus one byte. The refusal changes only a declaration; it performs no
large allocation.

From native Windows PowerShell at the repository root, with the native Python
environment active and the model and llama-server files already installed at
the paths in the selected config:

```powershell
$repo = (Resolve-Path -LiteralPath (Get-Location)).Path
$env:PYTHONPATH = $repo
python -X utf8 -m benchmarks.edge_agent.experiments.native_memory_admission `
  --config (Join-Path $repo 'benchmarks/edge_agent/configs/windows_laptop_gemma4_31b_hybrid.experimental.json') `
  --route-id gemma4-31b-qat-q4-windows-hybrid40
```

After a full 20×3-request plus 30-minute profile has completed, add
`--profile-index <private profile index.json>` to bind a **new** diagnostic to
that profile. The probe audits the profile's raw requests before loading the
model, then compares its exact config hash, route, stable CPU/GPU/RAM identity,
OS, driver, power condition, imported Omni source, and loaded runtime against
the live Windows process. A mismatch refuses before cold load and leaves a
failed private record. The successful record's manifest contains
`profile_binding` (`omni-agent-profile-binding-v1`) with the profile index and
raw SHA-256 hashes. Runs without the option contain `profile_binding: null`;
older runs cannot be retroactively bound.

The command prints a path under
`%LOCALAPPDATA%\OmniEdgeAgent\memory-probes\`. Its JSONL and exact temporary
config are private raw evidence and must not be committed. The same filename
cannot be reused. Each row is flushed immediately so a failed run retains
its readings. If an operation deadline expires, an
`operation_deadline_exceeded` row is flushed before teardown. A hung native
loader can still hold its worker thread during teardown, so a final `outcome`
row and process exit are not guaranteed in that case. A missing RAM or VRAM
sensor fails the probe.

`sampled_global_incremental_peak_bytes` is the largest sampled used-byte
reading minus the pre-load baseline across the cold load and complete turn.
Its difference from the route's declared reservation is diagnostic only:
other processes contribute to the global reading, and periodic sampling can
miss a brief peak. The loader's and Omni's own placement and ledger reports
remain separate evidence. This probe does not issue signed memory-admission
receipts, qualify model quality, establish p95 latency, or make any route a
default. Run it while other GPU/large-memory workloads are idle to reduce
interference.
