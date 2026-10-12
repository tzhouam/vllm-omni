# AI Hub 0.56 single-iteration transport check

This is a **real S24 component inference**, not a Spark-X2.5 end-to-end run.
It reused the audited one-layer W8A16 full-attention artifact and the real
input tensors from historical job `j57e4k9lp`. The exact registration is
Samsung Galaxy S24, Android 14. The compiled QNN DLC downloaded from AI Hub
has SHA-256 `8f1518f1f392434306dba11fd7115fe7629657ab7bfc834c0f78ee6b684e5154`.
The original checkpoint revision was not recorded in that historical job, so
the journal labels it `original-revision-unrecorded` and is ineligible for
whole-model promotion.

| Inference job | Options | Remote `profiling` | Timing samples returned | Per-op rows |
| --- | --- | --- | ---: | ---: |
| [`jp3z31zx5`](https://workbench.aihub.qualcomm.com/jobs/jp3z31zx5/) | `--compute_unit npu`; SDK `profile=False` | `False` | 100 | 0 |
| [`jglyz2we5`](https://workbench.aihub.qualcomm.com/jobs/jglyz2we5/) | `--compute_unit npu --max_profiler_iterations 1`; SDK `profile=False` | `False` | **1** | 0 |

The two jobs used different uploaded dataset IDs but their saved input NPZ
byte hashes match (`5a439e6c6d148f1720938770daa110039a39edde014f1c1592c823c7d7dfc8ed`).
Their output NPZ byte hashes also match
(`67f1236f5866c601791f3d8656591adeec3250f864b56a902542d6b84b937c4a`).
Compared with the historical job, the three output maximum absolute
differences are `6.5893e-05`, `3.38197e-04`, and `0`—a repeatability check,
not checkpoint-quality validation. The full raw inference profile responses,
job and dataset IDs, tensor hashes and downloaded outputs are retained in the
two adjacent evidence directories. Both replay journals pass `audit_run` and
remain at status `running` because a one-layer component cannot satisfy a
complete model topology.

The first journal initially mislabeled the historical artifact format as a
context binary. The SDK identifies it as `QNN_DLC`; its `run.json` contains
explicit corrections and the original journal snapshot. No job, tensor or
sample measurement was changed. Neither inference job provides per-op
compute-unit attribution, so NPU is a **requested** placement here; a
separate successful profile job is needed to attach actual placement to a
full-chain replay.

Reproduce with the configured SDK token and `qai-hub==0.56.0`:

```bash
PYTHONPATH=. /home/zhout/project/edge_infer/.venvs/aihub056/bin/python \
  benchmarks/edge_harness/experiments/validate_aihub_replay_sdk.py \
  --source-job-id j57e4k9lp \
  --checkpoint Spark-X2.5-1.7B-historical-component \
  --revision original-revision-unrecorded \
  --evidence-dir /tmp/omni-aihub-sdk056-new-component-smoke
```

That command submits a new component job only when the evidence directory is
fresh. Re-running against a completed journal verifies local evidence and
returns the same job ID without resubmitting.
