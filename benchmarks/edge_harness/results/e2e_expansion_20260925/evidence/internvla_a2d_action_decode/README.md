# InternVLA A2D physical-action boundary on WSL and native Windows CPU

**Disposition (2026-09-25): synthetic complete-policy action decoding passed on WSL CPU, native Windows CPU and native Windows CPU+Radeon 890M; robot-task E2E remains unqualified.** The pinned Place_Markpen checkpoint is revision `0a557c6a503e6545b369bff2f34c76a410bbd190`, `model.safetensors` SHA-256 `586799539888c69c35f84d083e559cc5f0ba887801d000e70e65f58d3f85c313`. Its `train_config.json` declares A2D delta actions; `stats.json` SHA-256 is `354710650bb91fc0e84a461987b3554b30bef9d5de47567480624484cdfbf3a2`. All three runs used the same patterned synthetic three-camera observation, zero normalized state and noise, and the 16-value raw state set to the checkpoint's state mean. There are no reference task actions in this fixture.

The [checkpoint-bound codec](../../../../../../vllm_omni/edge/internvla_actions.py) checks that the raw A2D state normalizes to the exact padded policy input before dispatch. The raw state stays in the Omni controller; the worker still receives the original normalized state only. For each returned `[1,50,32]` normalized padded chunk, the controller unnormalizes the 14 joint and two effector coordinates using checkpoint stats and adds the observed joint positions to the 14 delta coordinates. It emits a separate finite `[1,50,16]` `physical_actions` buffer and explicit metadata. Units, controller joint order and action step time remain unverified, and `control_ready=false`. Padded coordinates 16–31 are never presented as physical controls.

| Placement | Complete-request result | Independent reconstruction |
|---|---|---|
| HX370 WSL CPU; Python 3.12, PyTorch 2.13.0+cpu, vLLM 0.28.0, Omni checkout 0.29 | One warmup and one measured request passed. Startup 40.352 s; measured whole-request wall 3.220 s. 16 GiB host-RAM reservation under a 29 GiB declared WSL capacity; ledger cleared on shutdown. | Independent Torch unnormalize-plus-joint-delta calculation was bitwise identical to the emitted physical buffer. [Report](cpu_report.json), [normalized actions](cpu_normalized_actions.npy), [physical actions](cpu_physical_actions.npy), [worker log](cpu_worker.log). |
| HX370 native Windows 11 build 26200 CPU; Python 3.12.10, PyTorch 2.13.0+cu130 executing on CPU, vLLM 0.29.0, Omni checkout 0.29 | One warmup and one measured request passed. Startup 129.668 s; measured whole-request wall 4.601 s. 16 GiB host-RAM reservation and capacity; ledger cleared on shutdown. | The same independent Torch calculation was bitwise identical to the emitted physical buffer. [Report](windows_cpu_report.json), [normalized actions](windows_cpu_normalized_actions.npy), [physical actions](windows_cpu_physical_actions.npy), [worker log](windows_cpu_worker.log). |
| HX370 native Windows 11 build 26200 CPU+Radeon 890M; same Python/PyTorch/vLLM stack with separate Torch-DirectML worker | One warmup and one measured request passed. The BF16 policy stayed on CPU and the pinned FP32 Cosmos encoder graph ran on the selected Radeon 890M, as reported by the worker. Startup 138.581 s; measured whole-request wall 4.112 s. 16 GiB shared-RAM reservation and capacity; ledger cleared on shutdown. | Independent Torch reconstruction was bitwise identical to the emitted physical buffer. [Report](windows_radeon_report.json), [normalized actions](windows_radeon_normalized_actions.npy), [physical actions](windows_radeon_physical_actions.npy), [worker log](windows_radeon_success_worker.log). |

The WSL-versus-Windows **normalized** policy outputs differ 2.1795% relative L2 on this synthetic fixture, consistent with the prior environment comparison; there is no robot-task tolerance to interpret that difference. These one-request runs are contract checks, not latency distributions or power profiles. The first WSL attempt was refused before model load because a 30 GiB declared capacity exceeded currently available WSL RAM; the recorded successful run declared 29 GiB with the same CPU model and precision.

The Radeon-versus-native-CPU normalized policy output differs 2.4391% relative L2 on this fixture, with no task tolerance; the separate one-request wall times do not demonstrate a speedup. The first Radeon attempt [failed during load](windows_radeon_worker.log) because `VLLM_OMNI_EXTERNAL_PYTHON_TORCH_DML` was unset in that shell. Setting it to the existing `C:\Users\zhout\w2\qwen_dml_venv\Scripts\python.exe` produced the passing run. The failed attempt did not execute a policy request.

A separate [WSL abort rerun](cpu_abort_report.json) used the same raw-state action path, returned one measured physical action chunk, then canceled another in-flight request. The abort emitted no stale output; reserved and quarantined host RAM were both zero after abort and shutdown. Its [worker log](cpu_abort_worker.log) is retained. This checks cancellation lifecycle for the new output contract on WSL CPU, not automatic same-stage recovery or concurrent action sessions.

The [fixture preparer](../../../../experiments/prepare_internvla_a2d_fixture.py) now writes a v2 A2D observation with raw state and 50 reference physical actions, pinned to model, training config, stats, dataset files and source revision. The [Omni profiler](../../../../experiments/probe_omni_internvla_policy.py) keeps reference actions outside the worker request and records single-sample MAE, max error and relative L2 for that fixture. Legacy v1 fixtures remain accepted without physical decoding. The v2 route has **not** run with real Place_Markpen observations: the matching archive is still gated for this account. Task quality, physical units/order/time, deadlines, sustained latency and power remain open.

Reproduce the WSL contract run from the repository root:

```bash
ROOT=/home/zhout/project/edge_infer
EVIDENCE=benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/internvla_a2d_action_decode
PYTHONPATH="$PWD" "$ROOT/.venvs/omni-cpu/bin/python" benchmarks/edge_harness/experiments/probe_omni_internvla_policy.py \
  --model-dir "$ROOT/models/InternVLA-A1-3B-FT-Place_Markpen" \
  --processor-dir "$ROOT/models/Qwen3-VL-2B-Instruct-processor" \
  --cosmos-dir "$ROOT/models/Cosmos-Tokenizer-CI8x8" \
  --python-bin "$ROOT/.venvs/omni-cpu/bin/python" --placement cpu \
  --capacity-gib 29 --reserve-gib 16 --warmups 1 --repeats 1 --decode-a2d-actions \
  --log-file "$EVIDENCE/cpu_worker.log" --output-report "$EVIDENCE/cpu_report.json" \
  --output-actions "$EVIDENCE/cpu_normalized_actions.npy" \
  --output-physical-actions "$EVIDENCE/cpu_physical_actions.npy"
```

For native Windows, use the [existing Windows CPU reproduction setup](../../../e2e_expansion_20260923/evidence/internvla_omni_windows_cpu/README.md) with `--capacity-gib 16 --reserve-gib 16 --warmups 1 --repeats 1 --decode-a2d-actions`, and write the four outputs under this directory with the `windows_cpu_` prefix. For the Radeon run, add `--placement radeon-cosmos --graph-file \\wsl.localhost\Ubuntu\home\zhout\project\edge_infer\models\InternVLA-cosmos-encoder-batch6.pt2` and set `VLLM_OMNI_EXTERNAL_PYTHON_TORCH_DML` as above; write outputs with the `windows_radeon_` prefix. A v2 real-data run should instead use `--input-fixture` and `--fixture-manifest`; it compares one physical action chunk with its reference but does not by itself establish task quality.
