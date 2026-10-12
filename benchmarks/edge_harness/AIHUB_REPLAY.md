# AI Hub hosted functional replay

`aihub_replay.py` is an evidence journal for a **single batch-1 request**. A
model adapter supplies the complete neural-stage topology, real checkpoint
inputs, preprocessing, sampling, state updates, final output and reference
checks. The runner submits each neural stage to the exact Qualcomm AI Hub
target registration, waits for and downloads its output, and gives that output
to the adapter for the next dependent job. It does not execute a neural stage
on the host. It does not turn a component test into complete-model evidence.

Use `/home/zhout/project/edge_infer/.venvs/aihub056/bin/python` (qai-hub
0.56.0). `submit_inference_job` defaults to `profile=True`; the runner passes
`profile=False` explicitly and checks that the returned job reports
`profiling=False`. AI Hub still ran 100 timed iterations in our first S24
`profile=False` smoke. The runner therefore also forces
`--max_profiler_iterations 1` and archives the raw profile response so the
actual count remains visible. SDK 0.55.0 is rejected before submission.
Separate successful profile jobs can be attached for compute-unit evidence.
The [raw S24 transport evidence](results/e2e_expansion_20261002/evidence/aihub_runner_sdk056_s24_single_iteration_smoke/README.md)
records job `jp3z31zx5` (`profile=False`, no iteration option, 100 timing
samples) and job `jglyz2we5` (`profile=False`,
`--max_profiler_iterations 1`, one timing sample). These are component jobs,
not model E2E evidence.

```python
from benchmarks.edge_harness.aihub_replay import HubArtifact, HubReplay

with HubReplay(
    evidence_dir, target_name="Samsung Galaxy S25", model=model_name,
    checkpoint=checkpoint_id, revision=checkpoint_revision,
    request_id=request_id,
    stage_topology={"encoder": [], "decoder": ["encoder"]},
) as replay:
    first = replay.execute(
        "encoder-0", stage_id=0, stage_name="encoder", artifact=encoder_artifact,
        inputs={"image": image_array},
        input_provenance={"image": "request:image"},
        options="--compute_unit npu",
    )
    second = replay.execute(
        "decoder-0", stage_id=1, stage_name="decoder", artifact=decoder_artifact,
        inputs={"features": first.outputs["features"]},
        input_provenance={"features": "step:encoder-0:features"},
    )
    # The adapter must validate the *complete* output against a reference,
    # then call replay.finish(..., required_stages=["encoder", "decoder"]).
```

A `HubArtifact` binds target model ID, SHA-256 of its compiled artifact (or
hashed compiled manifest), checkpoint/revision, precision, target registration,
runtime, shape bucket and state-layout version. Each input and downloaded
output is saved as a separate NPZ with both byte and tensor-content hashes.
`run.json` includes the run ID, exact registration, SDK version, stage order,
input provenance, job/dataset IDs and placement evidence. Use
`audit_run(evidence_dir)` to recheck local bytes, topology, lineage, profile
records and completion status before matrix promotion.

Valid provenance forms are `request:<name>`, `step:<step-id>:<output-name>`
for byte-identical stage transfer, or
`host:<allowed-transform>:<source>` for declared non-neural processing.
Direct transfers are hash-checked. A crash after a job ID is durably saved
reattaches to that same job. A crash between SDK submission and saving the ID
is marked ambiguous and requires manual job reconciliation before any retry;
the runner never silently submits a duplicate. For a known terminal failure,
rerun only the corrected stage with a new step ID using saved completed
predecessor outputs, then call
`supersede_failed_step(failed_id, successor_step_id=..., reason=...)`. The
failed job remains in the journal and is excluded from the final successful
job list; an unresolved failed job prevents completion.

`finish` can mark `complete` only after all declared neural stages, passing
model-specific checks and separately verified compute-unit profiles are
present. Passing output without those profiles becomes
`functional_complete_placement_unverified`. The matrix verifier must also
apply its model-specific stage and final-output gates in
`experiments/verify_e2e_summary_table.py`; an adapter's smaller topology is not
independently proof of full-model coverage.

Workbench may release and reassign the physical device between jobs. The
host-side state snapshots and lock do not prove device-resident memory,
same-device state continuity, full-request device latency, playback or thermal
behavior. Stage profiles are not summed into a whole-request latency.

For a real SDK transport check that remains component-only, run
`experiments/validate_aihub_replay_sdk.py` with an already audited inference
job ID, checkpoint label, revision label and fresh evidence directory. Its
output is a `component_transport_smoke`, never a matrix pass.
