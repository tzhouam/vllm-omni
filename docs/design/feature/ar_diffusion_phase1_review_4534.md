# AR-Diffusion Phase 1 Review for PR 4534

Review target: `[AR-Diffusion] Phase 1: engine-level KV cache management for DreamZero`.

Local commit reviewed: `ef3b67e27`.

## Summary

The AR-Diffusion Phase 1 implementation is directionally sound: it keeps the existing DreamZero attention kernels, moves persistent self-attention and cross-attention KV into runner-owned engine pools, and gates actual AR-Diffusion KV behavior behind `AR_DIFFUSION_KV_ENABLE`. The focused AR-Diffusion unit suite passes locally.

The remaining concerns are around lifecycle and configuration consistency rather than the core paged write/gather path. The highest-risk issue is that AR-Diffusion session state is not evicted with DreamZero's model-local session LRU, so long-running serving with many session IDs can leak pool ownership until the AR-Diffusion KV pool is exhausted.

## Review Findings

### P1: AR-Diffusion Session KV Is Not Evicted With DreamZero Session State

`DreamZeroPipeline` bounds model-local session state with an LRU in `_states`, but `ARDiffusionModelRunner` keeps `_bde_states` in a plain dict and never removes old sessions. Each AR-Diffusion state owns two adapters, one positive and one negative CFG branch, and those adapters can keep resident pool blocks after the corresponding `DreamZeroState` has been evicted.

Impact: session churn in a long-running server can exhaust the finite AR-Diffusion KV block pool even though the model-local state map remains bounded.

Recommended fix: mirror the pipeline LRU in `ARDiffusionModelRunner`, or add an explicit session teardown path that calls `ARDiffusionKVCache.end_request()` for both adapters before removing the AR-Diffusion state.

### P2: DreamZero AR-Diffusion Config Helper Uses Old Causal-Block Granularity

`benchmarks/ar_diffusion/dreamzero_config.py` derives:

- `chunk_size = num_frame_per_block * frame_seqlen`
- `window_chunks = local_attn_size // num_frame_per_block`

The runtime AR-Diffusion path in `ARDiffusionModelRunner._preallocate_kv_cache()` overwrites `chunk_size` to `frame_seqlen` for frame-granular paging but preserves any configured `window_chunks`. A helper-produced config for `local_attn_size=21` and `num_frame_per_block=3` therefore becomes a 7-frame runtime window instead of the intended 21-frame window.

Recommended fix: update the helper and tests to use `chunk_size = frame_seqlen` and `window_chunks = local_attn_size`.

### P2: `frame_seqlen` Is Inferred From Deploy Resolution

`ARDiffusionModelRunner._infer_frame_seqlen()` derives frame token count from `policy_server_config.image_resolution`, while DreamZero runtime derives it from actual latent shape and the loaded transformer also carries `frame_seqlen`. If deploy config and model/runtime geometry drift, AR-Diffusion can allocate and write with the wrong chunk size.

Recommended fix: use the loaded transformer's `frame_seqlen` as source of truth, and only use deploy-derived geometry as an assertion if needed.

### P2/P3: DreamZero Routes to AR-Diffusion Runner by Default Even When KV Is Disabled

`default_engine_backend_for_model("DreamZeroPipeline")` returns `"ar_diffusion"`, so every DreamZero load builds `ARDiffusionEngine` / `ARDiffusionModelRunner` even when `AR_DIFFUSION_KV_ENABLE` is unset and actual AR-Diffusion KV is disabled. The production topology called out in the PR is still not fully validated, so this widens blast radius beyond users explicitly testing AR-Diffusion KV.

Recommended fix: gate DreamZero-to-AR-Diffusion routing behind the same enable switch or add a clear config path to force the base `DiffusionEngine`.

### P3: Duplicate AR-Diffusion Helper Surfaces Are Drifting

`BDEPipelineMixin` is exported from `vllm_omni.experimental.ar_diffusion.kv_cache` but `DreamZeroPipeline` does not inherit it. The two surfaces already differ: the mixin update path does not accept the real `seq_len` argument used by the current pipeline bridge.

The cross-attention `project_kv()` helpers in `causal_wan_model.py` are also unused because `_kv_populate_cross()` inlines the projection directly.

Recommended fix: delete the unused helpers or wire the production path through them so there is only one implementation to maintain.

## Local Validation

### Unit Tests

Command:

```bash
/feature/.venv/bin/python -m pytest tests/bde/
```

Result:

- `72 passed`
- Runtime warning observed: local `vLLM-Omni` and `vLLM` versions report different major/minor versions (`0.22.1.dev76` vs `0.23.0`).

### Performance Harness

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/models /feature/.venv/bin/python examples/offline_inference/dreamzero/ar_diffusion_perf_compare.py --num-chunks 12 --tag local_kv_off --latents outputs/bde_review/perf_kv_off.pt --timing outputs/bde_review/perf_kv_off.json
```

```bash
AR_DIFFUSION_KV_ENABLE=1 CUDA_VISIBLE_DEVICES=0 HF_HOME=/models /feature/.venv/bin/python examples/offline_inference/dreamzero/ar_diffusion_perf_compare.py --num-chunks 12 --tag local_bde_on --latents outputs/bde_review/perf_bde_on.pt --timing outputs/bde_review/perf_bde_on.json
```

Environment:

- Deploy config: `vllm_omni/deploy/dreamzero.yaml`
- Model: `GEAR-Dreams/DreamZero-DROID`
- Workload: 13 forwards, consisting of 1 prefill and 12 chunk forwards
- GPU: single local GPU, reported by `nvidia-smi` as `NVIDIA L20X`; per workspace guidance this hardware should be treated as H200
- `enforce_eager=True`, matching the PR harness

Local timing results:

- KV-off steady mean: `4729.96 ms`
- AR-Diffusion-on steady mean: `4780.05 ms`
- Delta: `+50.08 ms`
- Relative: `1.0106x` AR-Diffusion-on vs KV-off, or about `1.1%` slower locally
- KV-off prefill: `4996.03 ms`
- AR-Diffusion-on prefill: `5095.58 ms`
- KV-off load: `27.87 s`
- AR-Diffusion-on load: `28.05 s`

Precision result:

- Latents shape: `(1, 16, 30, 44, 80)`
- `max_abs_diff = 0`
- `PSNR = inf`

Saved artifacts:

- `outputs/bde_review/perf_kv_off.json`
- `outputs/bde_review/perf_bde_on.json`
- `outputs/bde_review/perf_kv_off.pt`
- `outputs/bde_review/perf_bde_on.pt`

## Comparison With PR Performance Claim

The local run matches the PR's main correctness claim: AR-Diffusion-on is bit-exact against KV-off for the tested DreamZero rollout.

The local run also broadly matches the PR's performance-parity claim, but not the exact relative speedup:

- PR reports baseline steady mean `4930 ms` and current AR-Diffusion `4813 ms`, or `0.976x` versus baseline.
- Local run reports KV-off steady mean `4729.96 ms` and AR-Diffusion-on `4780.05 ms`, or `1.0106x` versus KV-off.

Interpretation: AR-Diffusion is at parity on this local run, but it is slightly slower than the local KV-off run rather than slightly faster. The AR-Diffusion-on absolute number is close to the PR number (`4780 ms` local vs `4813 ms` PR). The local KV-off baseline is faster than the PR baseline, which explains the relative-delta mismatch.

## Merge Readiness

I would not block the PR on the core AR-Diffusion write/gather implementation based on this review. I would address the session-lifecycle issue before production serving, because it can turn session churn into a hard KV pool exhaustion failure. The config-helper and frame geometry issues should also be fixed before relying on helper-generated configs outside the specific PR harness path.
