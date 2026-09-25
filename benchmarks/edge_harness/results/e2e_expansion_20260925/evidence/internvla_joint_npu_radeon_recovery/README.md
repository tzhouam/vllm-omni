# InternVLA joint CPU+AMD NPU+Radeon abort and fresh-stage recovery

**Disposition (2026-09-25): scoped synthetic recovery passed through public
`AsyncOmni.generate`.** A native Windows HX370 run loaded the pinned
Place_Markpen checkpoint with CPU policy/Cosmos prefix, six AMD NPU Conv13
calls, and the Radeon 890M-requested DirectML Cosmos suffix. It returned a
terminal finite `[1,50,32]` action, then aborted a second request after the
policy worker logged that computation had started. No action from that
request reached the client. A newly initialized Omni stage returned the
same action hash under a different worker generation. This extends the
[joint whole-policy profile](../internvla_joint_npu_radeon/README.md);
the observation remains synthetic, and `control_ready=false`.

The [passing report](report3.json) measured initial/restart startup at
145.85/143.62 s and separate baseline/fresh complete requests at
4.435/4.372 s. The in-flight abort acknowledgement took 0.389 s.
Baseline, fresh-stage and the [prior public request](../internvla_joint_npu_radeon/public.json)
all returned action SHA-256
`f17ae6dcd4a6dd5f1cc7ee29198606e2499edc762f830d54df25504a887796e0`.
The worker generations changed from
`ff13606a94d440d2a694465b517d66d4` to
`2b3210a454724f828f1077714b76fd6b`. The
[independent audit](audit3.json) checks the hash, terminal events, no stale
action, worker-start marker, two stage shutdowns, and placement on both
loads. The two [worker logs](worker3.log) and
[restart log](worker3_restarted.log) record VitisAI warmup placement
(one NPU node each) and a DirectML suffix with 357 DirectML and 34 CPU
nodes, requested on adapter 1. The [driver log](driver3.log) retains the
abort and stage lifecycle. These traces verify the provider sessions; they
do not provide a new per-request node trace.

The backend deliberately retires the blocking policy worker on an in-flight
abort. The [first recovery attempt](report2.json) correctly observed a
dead stage when it tried to submit another request to that same instance.
Its [driver](driver2.log) and [worker](worker2.log) logs are retained. The
final probe creates a **new** stage rather than replaying state into the
retired worker. An earlier [startup failure](driver.log) was caused by
Windows' GBK decoding of a PyTorch kernel template; setting
`PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` repaired the test
environment before either complete-policy attempt.

Reproduce using the pinned paths and PowerShell command in the
[joint-profile evidence](../internvla_joint_npu_radeon/README.md), replacing
the profiler with
[`probe_omni_internvla_entrypoint.py`](../../../../experiments/probe_omni_internvla_entrypoint.py),
adding `--abort-recovery`, and choosing fresh worker-log and report paths.
Set `PYTHONUTF8=1`. The
[`audit_internvla_joint_restart.py`](../../../../experiments/audit_internvla_joint_restart.py)
arguments are the new report, previous public report, first/restart worker
logs and driver log. The model, graphs, ORT VitisAI/DirectML versions, device
selection and 16 GiB shared-RAM reservation are pinned as in that profile.

The synthetic action hash establishes deterministic restart for this
fixture, not task success. Real A2D observations and reference actions,
physical units/order/step time, paired whole-request benefit, worker-crash
recovery and sustained power/thermal behavior remain unverified.
