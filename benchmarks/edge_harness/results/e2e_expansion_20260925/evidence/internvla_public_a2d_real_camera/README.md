# InternVLA-A1 real A2D camera input across PC placements

Seven **complete** Omni whole-policy requests passed on one pinned public A2D
observation: WSL CPU, native Windows CPU, native Windows Radeon 890M+CPU,
native Windows AMD NPU+CPU, native Windows AMD NPU+Radeon+CPU, WSL RTX 5090
Laptop CUDA and native Windows RTX CUDA. Each strict-loaded the real
Place_Markpen checkpoint, consumed two frames from each of three real camera
streams plus the dataset's joint/effector state and shared diffusion noise,
and returned finite `[1,50,32]` padded and `[1,50,16]` decoded physical action
buffers in a terminal epoch/sequence-tagged event. Every plan released its
host/VRAM reservation after shutdown.

**This is an execution test, not Place_Markpen task validation.** The public
[`pepijn223/task_374`](https://huggingface.co/datasets/pepijn223/task_374)
sample is the A2D *laundry/personal-care sorting* task, while the checkpoint
training config names `a2d_real/a2d_pen_holder`. The state/action/camera schema
matches, but the task and scene do not. The dataset card declares CC BY 4.0;
the derived [fixture](observation.npz) is attributed to that dataset and its
revision `dade034a5780c23de5ab98fd236fee99f0758021`. The three source
videos and Parquet files stay outside Git; their full SHA-256 digests are in
the [audit](audit.json). The 3.5 MB fixture is sample index 15 of episode 0,
seed 1234, with camera offsets −15 and 0 frames. The mean absolute difference
between the two decoded frames is 0.00470/0.00385/0.00405 for head/left/right.
Pinned PyAV 18.1.0 decoded the actual AV1 videos because this host's
torchvision 0.28.0+cpu lacks `torchvision.io.VideoReader`. A second preparation
run reproduced the fixture SHA-256
`b0b195ee8b1f1d6bb17df84e69167064f75e14c0ca260006756f7ab66f881fd5`
bitwise.

| Whole-policy placement | One request wall (s) | Padded-action L2 vs native Windows CPU | Physical-action L2 vs dataset action |
|---|---:|---:|---:|
| [WSL CPU](cpu_report.json) | 3.315 | 0.840% | 31.71% |
| [Windows CPU](windows_cpu_report.json) | 5.118 | reference | 31.74% |
| [Windows Radeon+CPU](radeon_report.json) | 4.324 | 0.640% | 31.73% |
| [Windows AMD NPU+CPU](npu_report.json) | 5.118 | 0.701% | 31.76% |
| [Windows AMD NPU+Radeon+CPU](joint_report.json) | 4.918 | 0.729% | 31.76% |
| [WSL RTX CUDA](cuda_report.json) | 1.386 | 0.844% | 31.77% |
| [Windows RTX CUDA](windows_cuda_report.json) | 2.350 | 0.780% | 31.73% |

The physical-action column is **not an accuracy score** for Place_Markpen:
the reference trajectory belongs to the different laundry task. It is
retained to make the mismatch and missing task gate explicit. The [independent
audit](audit.json) reconstructs each 16-value action from the checkpoint's
normalization statistics and raw state, checks action and input hashes,
terminal event metadata, zero final ledger, distinct real camera frames and
the requested device evidence. All seven decoded buffers match that
reconstruction bitwise. The one-request wall values have different OS,
cache, load order and uncontrolled power/thermal conditions; they are **not**
p50/p95 distributions, a paired speedup or an accelerator-selection result.

Placement is observed rather than inferred from a requested flag. The
Windows NPU worker reports a VitisAI NPU node on the six batch-one Conv13
calls, and the [joint raw ORT NPU trace](joint_worker_npu_profile_2026-09-25_21-47-54_249.json)
is retained. The joint suffix worker reports 357 DirectML and 34 CPU nodes
with requested adapter 1, previously mapped to the Radeon 890M. The
Radeon-only `.pt2` Cosmos graph returned its output on
`AMD Radeon(TM) 890M Graphics`; this output-device check does not prove every
internal operation ran there. CPU policy/flow remained on CPU in all AMD
routes. The CUDA workers report the actual RTX 5090 Laptop GPU and CUDA
policy placement. The Radeon route explicitly uses an FP32 Cosmos graph;
the CPU/CUDA policy routes use their recorded BF16 components. Full worker,
driver and report files are retained beside the action arrays.

The HX370 host ran Windows 11 build 26200 and WSL Ubuntu 26.04. Windows
Radeon/NVIDIA driver versions were `32.0.22018.6001` / `32.0.16.1071`; the
NPU route used VitisAI EP 1.8.63.0 and ORT 1.30.0. The separate DirectML
suffix worker used ORT 1.24.4; the Radeon `.pt2` worker used torch-directml.
The WSL CPU controller used PyTorch 2.13.0+cpu; Windows policy workers used
PyTorch 2.13.0+cu130 with CPU or CUDA selected explicitly. The checkpoint,
processor, Cosmos artifacts, graph/interpreter paths and hashes are in each
request report. Each route admitted 16 GiB shared host RAM; CUDA routes also
reserved 16 GiB VRAM. The observed loaded process-tree RSS is not a loading
peak, and no sustained power or thermal measurement was made.

To reproduce, download `meta/info.json`, `meta/stats.json`,
`meta/tasks.parquet`, `meta/episodes/chunk-000/file-000.parquet`,
`data/chunk-000/file-000.parquet` and the `file-000.mp4` streams under
`videos/observation.images.{head,hand_left,hand_right}/chunk-000/` from the
above immutable dataset revision into a local directory. Then run
[`prepare_internvla_a2d_fixture.py`](../../../../experiments/prepare_internvla_a2d_fixture.py)
with `--model-dir` pointing to the pinned checkpoint, `--dataset-dir` to that
directory, `--source pepijn223/task_374@dade034a5780c23de5ab98fd236fee99f0758021`,
`--sample-index 15`, and `--output-fixture observation.npz
--output-manifest observation.json`. The existing
[`probe_omni_internvla_policy.py`](../../../../experiments/probe_omni_internvla_policy.py)
commands in the [CPU/Radeon](../../../e2e_expansion_20260923/evidence/internvla_omni_windows_hybrid/README.md),
[AMD NPU](../internvla_omni_amd_npu_live/README.md),
[joint](../internvla_joint_npu_radeon/README.md) and
[CUDA](../../../e2e_expansion_20260923/evidence/internvla_omni_cuda/README.md)
records can be rerun with the matching `--placement`, artifacts and explicit
budgets, plus `--input-fixture observation.npz --fixture-manifest
observation.json --warmups 0 --repeats 1 --output-physical-actions`.
Recompute [the audit](../../../../experiments/audit_internvla_public_a2d.py)
with the saved evidence directory, checkpoint and optional downloaded dataset
directory. The audit needs `--include-radeon --include-cuda
--include-windows-cuda` for all seven reports.

The next VLA quality gate is an authorized *matching* Place_Markpen camera,
state and reference-action set with declared units, joint order, step time
and task tolerance. Then measure paired full-request benefit, loading peaks,
concurrency, cancellation/recovery and sustained power. `control_ready` stays
false; no action here is qualified for robot actuation.
