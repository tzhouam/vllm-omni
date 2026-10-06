# Unsigned Gemma gate evidence assembly

`assemble_gemma_gate_evidence.py` compares independent private observations
with one audited batch-size-1 Gemma **memory-task** profile. It does not sign a
review, create a gate receipt, or install a router default. A matching model
file and route configuration alone cannot bind an experiment to the profiled
runtime. Older memory and cancellation runs without an explicit
`profile_binding` stay diagnostic even when their raw measurements are valid.

Run the module with native Windows Python and this checkout on `PYTHONPATH`.
Provide the exact private files from the intended run; the output defaults to
`%LOCALAPPDATA%\OmniEdgeAgent\gate-assemblies\gate_assembly_<id>\`.

```powershell
python -X utf8 -m benchmarks.edge_agent.experiments.assemble_gemma_gate_evidence `
  --profile-index <audited-native-profile-index.json> `
  --memory-probe <private-memory-probe.jsonl> `
  --cancel-record <private-cancel-recovery.jsonl> `
  --lineage-probe <private-lineage-probe.json> `
  --download-manifest <pinned-download-manifest.json> `
  --source-metadata <pinned-source-metadata.json> `
  --placement-snapshot <private-placement-snapshot.json> `
  --quality-index <private-bilingual-quality-index.json>
```

All inputs after `--profile-index` are optional. Missing or inconsistent input
is named in `report.json`. A lineage, cancellation, or quality source with
verified observations **and** explicit profile/runtime/hardware binding gets
an unsigned `<gate>.raw.json` in the format expected by the independent
qualification reviewer. An older run with valid observations but missing
binding gets `<gate>.diagnostic.json` instead; that file uses a distinct record
type and cannot be used as a gate source. The current memory probe always stays
diagnostic because sampled whole-host increments cannot bound unsampled
transients or isolate the model process. Startup-log placement also stays
diagnostic because it does not establish per-operation compute placement. Raw
prompts, event traces, worker logs, and the output directory remain private;
the command prints only paths, gate names, and blocker codes.

The independent memory/cancellation/placement/quality probes must record a
`profile_binding` object generated against the **same audited profile**:

```json
{
  "schema": "omni-agent-profile-binding-v1",
  "profile_index_sha256": "<index file SHA-256>",
  "profile_raw_sha256": "<audited samples SHA-256>",
  "source_config_sha256": "<profile source config SHA-256>",
  "route_id": "<route ID>",
  "environment_fingerprint": "<profile fingerprint>",
  "imported_omni_source_sha256": "<live imported-source digest>",
  "loaded_runtime_sha256": "<live Agent/runtime digest>",
  "hardware_identity": {
    "os": "<observed OS>",
    "machine": "<observed machine>",
    "cpu": "<observed CPU>",
    "host_ram_total_bytes": 0,
    "gpu_name": "<observed GPU>",
    "gpu_driver": "<observed driver>",
    "power_condition": "<observed power condition>"
  }
}
```

The assembler compares every field to the profile and to the probe's own
hardware record, then recomputes the environment fingerprint. Copying profile
fields into an old record after execution would not establish this binding.
The source profile itself must match the currently imported runtime; otherwise
the report names `profile_runtime_stale` and a new full profile is required.
Even with five raw gate sources, independent task review and an Ed25519 signed
receipt remain separate manual steps.
