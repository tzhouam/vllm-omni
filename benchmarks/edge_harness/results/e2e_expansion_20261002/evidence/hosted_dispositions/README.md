# Exact-target AI Hub replay blockers (2026-10-02)

These ten JSON reports apply to the five exact registrations in the rolling
matrix. They are named **external blockers**, not measured model or device
incompatibilities, full inference runs, or capacity refusals. Each report
identifies its raw source, its SHA-256 and a next experiment. The machine-readable
[hosted evidence index](../../../e2e_profiling_20260922/evidence/summary/hosted_e2e_index.json)
maps each report to one matrix cell; the summary verifier checks the report,
scope, source-file existence, and release qualification.

| Model | Exact registrations | Observed external dependency |
| --- | --- | --- |
| InternVLA-A1 | S25, S24, X Elite CRD, SA8775P ADP, RB3 Gen 2 | The [matching Place_Markpen reference archive access probe](../internvla_dataset_access/access_probe.json) returned an approval-required error. A real task-quality comparison cannot be made with the synthetic observations. Full target policy and action artifacts are independently unfinished. |
| Qwen3.8-27B | S25, S24, X Elite CRD, SA8775P ADP, RB3 Gen 2 | The [SDK 0.56.0 target inventory](../aihub_device_inventory/device_inventory_sdk056.json) has no exact installed or usable RAM for these registrations. Full target artifacts and peak-memory measurements are also unfinished. Neither fit nor an unsupported capacity verdict is established. |

All ten reports record batch size 1 and one active request as the intended
replay protocol, but no full request was submitted for those cells. The other
15 hosted cells retain an open engineering gap until a complete neural-stage
chain passes or a concrete target-specific failure is captured. Component
jobs must not be promoted into completed hosted replay.
