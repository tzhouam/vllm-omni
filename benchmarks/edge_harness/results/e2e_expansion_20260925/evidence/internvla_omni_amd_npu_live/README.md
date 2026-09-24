# InternVLA native Windows CPU + AMD NPU live policy probe

**Disposition (2026-09-25): experimental complete synthetic action requests; not a qualified action-policy or beneficial split.** The real Place_Markpen checkpoint ran through Omni `StageRuntime` on Windows 11 build 26200 / HX370. The policy and Cosmos prefix/suffix ran on CPU; six batch-one Conv13 calls ran through ORT 1.30.0 / VitisAI EP 1.8.63.0 on AMD NPU driver 32.0.203.329. This is a single local device, not an AI Hub run. The GPU was not used. The checkpoint is BF16 in the policy; the ONNX encoder prefix/suffix is FP32 and the Conv13 artifact is A16W8 QDQ. The observations were six patterned synthetic camera frames with zero robot state/noise and one fixed instruction. Action units, joint order and step time remain unverified; `control_ready=false`.

The [one-warmup, 20-request raw report](profile20.json) records 20/20 complete finite `[1,50,32]` action chunks with one action hash, nearest-rank complete-request wall p50/p95 **4.720/4.750 s**, startup 123.070 s, a 16 GiB host-RAM reservation, 9.307 GB loaded process-tree RSS, and an admission-controlled capacity of one. The worker's initial ORT trace contained one `vitisai` node event and the plan pinned the graph, prefix, suffix, EP DLL, interpreter and weights by SHA-256. Cancellation of an in-flight request produced no stale output and released the RAM reservation. [Worker log](profile20_worker.log) and [first measured action](profile20_actions.npy) are retained. A [one-request smoke](smoke2.json) passed with 4.966 s wall time. The initial [failed smoke](smoke.json) stopped at a verifier-label mismatch (`VitisAIExecutionProvider` versus ORT's actual `vitisai`); its [log](smoke_worker.log) is retained. The placement check was corrected before the measured run.

Against the separately measured [native Windows CPU Omni profile](../../../e2e_expansion_20260923/evidence/internvla_omni_windows_cpu/README.md), the live NPU route's synthetic actions differ by **3.176% relative L2**, maximum absolute **0.0870**, cosine **0.999521**. There is no task-level acceptance threshold or real-observation reference action. The CPU profile's separate 20-request wall p50/p95 was **4.575/4.725 s**; run order, OS cache, clocks and power were not paired or controlled, so the data show no demonstrated whole-request benefit rather than a controlled slowdown. The isolated Conv13 component was faster on a fixed activation, but that does not justify this split for policy deployment. The earlier [fixed-fixture component and replay](../internvla_amd_npu_batch1/README.md) remain distinct evidence; their 0.876% synthetic action error was an injected latent from a different input, not this live request.

The stage route is opt-in as `amd-npu-conv13` in the existing Omni whole-policy backend. It refuses missing/mismatched graph hashes, absent NPU hardware, zero NPU node events, invalid tensor shapes, and inadequate reservation. The [probe](../../../../experiments/probe_omni_internvla_policy.py) supplied the full plan. Large ONNX artifacts stay outside Git: the Conv13 graph is `C:\Users\zhout\w2\internvla_batch1_20260925\boundary_batch1_a16w8_cal7.onnx` (SHA-256 `2ae37f405cb7cb469f01d7022c0e789650e486d81279713715d0e9808af15f67`), and the pinned prefix/suffix paths and hashes are in `profile20.json`. Model/processor/Cosmos checkpoint hashes are there as well. The [final same-session smoke](final_smoke.json) verifies the placement trace and action after the code was tightened to reuse the profiled ORT session; the earlier 20-request run used a second identically configured session after the placement warmup.

An explicit [1 GiB admission-refusal run](admission_refusal.json) rejected the
weights, graph and workspace/headroom before worker launch, with no reservation
left in the ledger.

The next gate is independent, representative observations and reference actions with a declared task tolerance. A coarser or fuller NPU encoder artifact must then beat a paired CPU whole-request profile after transfer, shared-RAM peak, and sustained power/thermal costs; until then the default CPU placement remains the qualified local route for this synthetic policy scope.

Reproduce the 20-request run in native PowerShell after generating the pinned
prefix/suffix and batch-one graph described in the [component record](../internvla_amd_npu_batch1/README.md):

```powershell
$root = '\\wsl.localhost\Ubuntu\home\zhout\project\edge_infer'
$repo = Join-Path $root 'vllm-omni-edge'
$onnx = Join-Path $root 'models\InternVLA-cosmos-onnx'
$evidence = Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260925\evidence\internvla_omni_amd_npu_live'
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
  --ep-dir $ep --python-bin $py --placement amd-npu-conv13 `
  --capacity-gib 16 --reserve-gib 16 --warmups 1 --repeats 20 --abort-check `
  --reference-actions (Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260923\evidence\internvla_omni_windows_cpu\cpu_profile20_actions.npy') `
  --log-file (Join-Path $evidence 'profile20_worker.log') `
  --output-report (Join-Path $evidence 'profile20.json') `
  --output-actions (Join-Path $evidence 'profile20_actions.npy')
```
