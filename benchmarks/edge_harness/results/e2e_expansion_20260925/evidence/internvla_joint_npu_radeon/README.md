# InternVLA whole-policy CPU + AMD NPU + Radeon 890M experiment

**Disposition (2026-09-25): scoped synthetic complete-policy pass through Omni, not robot-task qualification or a demonstrated beneficial split.** One HX370 laptop ran the real Place_Markpen checkpoint on native Windows 11 build 26200. Its BF16 Qwen3-VL/flow action policy remained on CPU. A pinned FP32 Cosmos ONNX prefix produced six frame boundaries, an A16W8 QDQ Conv13 graph ran as six batch-one calls on the AMD NPU, and the pinned FP32 Cosmos suffix ran through an isolated ONNX Runtime DirectML worker requested on adapter 1. The policy returned finite float32 `[1,50,32]` actions with a terminal action event and `control_ready=false`. The [environment capture](environment.json) fixes the NPU/Radeon drivers, runtimes and uncontrolled power condition. No robot-control units, joint order, step time or real-observation quality gate is available.

The [one-warmup, 20-request whole-policy profile](profile20_handoff.json) passed **20/20** serial synthetic observation-to-action requests with one repeated action hash. Nearest-rank complete-request wall p50/p95 was **4.525/4.683 s**, excluding **124.146 s** startup. The admitted plan reserved **16 GiB shared host RAM**, measured **10.23 GB** loaded process-tree RSS including the DirectML child, and limited capacity to one request. An earlier [20-request profile before stage-timing instrumentation](profile20.json) also passed at 4.541/4.621 s p50/p95, with the same action hash. These are separate ordered runs, not a controlled speedup study.

Both accelerators have execution evidence. The plan's VitisAI warmup reported one NPU node for the Conv13 graph; a [later one-request report](smoke_profiled.json) retains the [raw NPU trace](smoke_profiled_worker_npu_profile_2026-09-25_02-59-41_341.json) with one VitisAI and three CPU node events. The [whole-policy DirectML placement report](profile20_handoff.json) counted **357 DirectML and 34 CPU node events** in the suffix, with the [raw ORT node trace](profile20_handoff_dml_profile.json) retained. The requested DirectML `device_id=1` maps to `AMD Radeon(TM) 890M Graphics` in the separately installed `torch-directml` enumeration (index 0 is the RTX 5090 Laptop). ORT's DML profile names the provider, not the physical adapter; Radeon attribution uses that adapter-index mapping. There is no claim that every suffix operation ran on the iGPU.

The profiled joint handoff had nearest-rank p50/p95 **10.71/12.57 ms** for the CPU prefix, **16.91/44.61 ms** for six NPU Conv13 calls, **532.17/639.84 ms** inside the DirectML suffix worker, and **568.56/665.23 ms** for its full round trip. The measured serialization, transport and dispatch portion was **31.85/43.21 ms**. A separate fixed-component [DirectML suffix probe](suffix_dml_probe.json) used a CPU Conv13 boundary and measured 6.45e-7 latent relative L2 versus the CPU suffix, with 357/34 provider events in its [raw profile](suffix_dml_profile_2026-09-25_02-28-44.json). Its same-process 10-request p50/p95 was **0.138/0.620 s** on DML versus **0.771/0.794 s** on CPU. These component timings have different inputs and process conditions from the whole-policy handoff.

The [pinned action comparison](action_comparison.json) measures the joint output against separately retained synthetic CPU, CPU+NPU and CPU+Radeon outputs: relative L2 **3.188%**, **1.363%** and **2.614%**, respectively. The jointly executed action is finite and repeatable but no task-level acceptance tolerance exists. Earlier separate native Windows complete-policy p50 values were 4.878 s CPU, 4.720 s CPU+NPU and 4.575 s CPU+Radeon. The joint 4.525 s p50 is only 50 ms below the separate Radeon run; run order, clocks, file cache, thermal state and power were not paired. The architecture's benefit gate is therefore **not met**, and this placement remains opt-in experimental.

The [1 GiB refusal](refusal.json) stopped before loading weights or launching a worker. The measured 20-request run then cancelled an in-flight request with no stale output, zero reserved/quarantined host RAM, and all six recorded worker processes exited. A separate public [`AsyncOmni.generate` request](public.json) returned the same action hash in **4.162 s** after **126.037 s** startup, with terminal sequence/epoch metadata. The first one-request [smoke](smoke.json), [action tensor](smoke_actions.npy), two 20-request [action](profile20_actions.npy) [tensors](profile20_handoff_actions.npy), and [worker logs](profile20_handoff_worker.log) are retained. The [component input and CPU/DML latents](suffix_dml_outputs.npz) are retained as well.

The implementation reuses Omni's whole-policy `StageRuntime` and existing external ORT worker. `amd-npu-radeon-cosmos` pins the checkpoint, ONNX prefix/Conv13/suffix, VitisAI DLL and isolated DirectML interpreter by SHA-256; it rejects missing provider nodes, wrong shape/precision, insufficient reservation and unverified device selection. The two ORT versions remain in separate processes because the VitisAI EP needs ORT 1.30.0 while this DirectML worker uses ORT 1.24.4. The stage owns the action event, bounded request, cancellation epoch and cleanup; no alternate model is selected silently. Large model graphs remain outside Git at the exact paths and hashes recorded in the raw reports.

Reproduce the measured run in native PowerShell with the pinned local model files:

```powershell
$root = '\\wsl.localhost\Ubuntu\home\zhout\project\edge_infer'
$repo = Join-Path $root 'vllm-omni-edge'
$onnx = Join-Path $root 'models\InternVLA-cosmos-onnx'
$evidence = Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260925\evidence\internvla_joint_npu_radeon'
$py = 'C:\Users\zhout\w2\omni029venv\Scripts\python.exe'
$ep = Join-Path (Get-AppxPackage -Name 'WindowsWorkload.EP.AMD.VitisAI.Framework.1.8').InstallLocation 'ExecutionProvider'
$env:PYTHONPATH = $repo
$env:VLLM_ENABLE_V1_MULTIPROCESSING = '0'
& $py (Join-Path $repo 'benchmarks\edge_harness\experiments\probe_omni_internvla_policy.py') `
  --model-dir (Join-Path $root 'models\InternVLA-A1-3B-FT-Place_Markpen') `
  --processor-dir (Join-Path $root 'models\Qwen3-VL-2B-Instruct-processor') `
  --cosmos-dir (Join-Path $root 'models\Cosmos-Tokenizer-CI8x8') `
  --graph-file 'C:\Users\zhout\w2\internvla_batch1_20260925\boundary_batch1_a16w8_cal7.onnx' `
  --prefix-file (Join-Path $onnx 'cosmos_prefix_to_group_norm_20260925.onnx') `
  --suffix-file (Join-Path $onnx 'cosmos_encoder_suffix_after_conv13_20260925.onnx') `
  --ep-dir $ep --dml-python-bin 'C:\Users\zhout\w2\internvla_ort_dml_venv\Scripts\python.exe' `
  --python-bin $py --placement amd-npu-radeon-cosmos `
  --capacity-gib 16 --reserve-gib 16 --warmups 1 --repeats 20 --abort-check `
  --reference-actions (Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260923\evidence\internvla_omni_windows_cpu\cpu_profile20_actions.npy') `
  --log-file (Join-Path $evidence 'profile20_handoff_worker.log') `
  --output-report (Join-Path $evidence 'profile20_handoff.json') `
  --output-actions (Join-Path $evidence 'profile20_handoff_actions.npy')
```

Next gates are real Place_Markpen observations and reference actions with a declared tolerance, physical action metadata, paired same-workload whole-request benefit against CPU and CPU+Radeon, loading-peak/shared-GPU memory, concurrency, post-cancel restart, sustained power/thermal behavior and deadline handling. The checkpoint's named real dataset was inaccessible in the earlier [access record](../../../e2e_expansion_20260924/evidence/internvla_real_observation_access/README.md); synthetic actions cannot stand in for task success.
