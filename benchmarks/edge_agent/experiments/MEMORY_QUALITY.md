# Native bilingual memory-quality experiment

`native_memory_quality.py` is an **independent quality probe**, not a timing
protocol or a signed router qualification. It uses the normal native Windows
Omni Agent controller and its encrypted DPAPI memory. The pinned Gemma 4
route remains explicitly experimental.

The probe generates four nonce-like identifiers from a private seed. It runs
English→English, Chinese→Chinese, English→Chinese, and Chinese→English recall.
Each target has a similarly named distractor with a different identifier.
One controller session submits eight source/distractor observation turns in
sequence. After closing it, a new controller opens the **same encrypted
database** and submits one recall request per case. It then deletes the exact
source event and submits the same request again. Thus the run has 16 complete
Agent requests, batch size 1, and at most one active request. It does not
insert facts directly into SQLite or answer with a host-side model substitute.

A case passes only if the source and distractor were both actually retrieved
into the second session, the final event derives from both exact source event
IDs, the answer exactly matches the independent reference, the source deletion
cascades to that final event and its search entries, the distractor remains,
and the later answer is exactly `UNKNOWN`. Ordered full-Agent traces and
declared placement must also hold. An answer that happens to match without
source lineage fails. The retrieved candidates are recorded separately from
the final lineage so ranking or top-5 crowding is visible. The native route's
configured recall limit is used unchanged; a miss is a measured failure.

From the repository root in native Windows PowerShell, with the native Python
environment active:

```powershell
$repo = (Resolve-Path -LiteralPath (Get-Location)).Path
$env:PYTHONPATH = $repo
python -X utf8 -m benchmarks.edge_agent.experiments.native_memory_quality `
  --config (Join-Path $repo 'benchmarks/edge_agent/configs/windows_laptop_gemma4_31b_hybrid.experimental.json') `
  --lineage (Join-Path $repo 'benchmarks/edge_agent/configs/windows_laptop_gemma4_31b_hybrid.lineage.experimental.json')
```

After a full native profile has completed, a **new** independent quality run
can add `--profile-index <private-profile-index.json>`. Before its first model
request, the probe audits the raw profile and checks the exact source config,
route, imported Omni source, loaded runtime, environment fingerprint, and live
hardware, OS, driver, and power condition. It checks the binding again when
the second app session opens. A mismatch fails the run; it never silently
uses another route. The raw manifest records `profile_binding_requested` and
the verified `omni-agent-profile-binding-v1` object, or `null` for an unbound
run. Binding is provenance evidence only and does not sign or qualify a gate.

The command prints the private index path. A unique
`benchmarks/edge_agent/results/memory_quality_*/` directory contains raw
JSONL, index, the final encrypted database, phase-specific native configs,
and local worker logs. The raw JSONL includes full prompts, answers, event
traces, exact source IDs, reference/answer SHA-256 hashes, observed placement,
and per-case checks. `results/.gitignore` excludes the entire run directory.
The two phase configs use a temporary local NTFS SQLite path during the run;
after both controllers close, the encrypted database is copied to the private
results directory. A failure still gets a hash-bound private index and failure
record. No public aggregate is generated automatically; a reviewer must
inspect the raw records before publishing any sanitized aggregate or using a
result as independent task-quality evidence.

This probe does **not** establish 20-request p95 latency, a 30-minute
endurance result, open-domain memory quality, or permission to install a
default route. Those require the separate profiling and signed-review gates.

The manually reviewed [2026-10-06 aggregate](../public_evidence/agent_memory_quality_20261006.json)
preserves a paired 3/4 result before the encrypted-index ranking and
current-request exclusion fix, and 4/4 after it, using the same private seed
and pinned Gemma route. Its hashes identify the unmodified private runs;
neither result is a signed qualification. The previous failure was a real
top-five miss: the target and similarly named distractor were omitted from
the recalled source lineage while the target source still existed.
Those two earlier raw runs predate the full profile, have no profile binding,
and remain diagnostic; their records are not rewritten or promoted
retroactively. A third, manually reviewed
[sanitized aggregate entry](../public_evidence/agent_memory_quality_20261006.json)
records `memory_quality_442380f3ffa6407c95481631f45af084`, bound to
the later current-source Gemma fixed-memory profile and loaded runtime. Its
four English/Chinese source-and-distractor, exact-answer, and cascading
deletion cases passed **4/4** with the same batch-one, single-request
condition. The profile and probe raw SHA-256 digests are published in the
aggregate; prompts, answers, and source contents remain private. This small
fixture is not open-domain memory validation or a signed reference-quality
gate, and it does not install a default route.
