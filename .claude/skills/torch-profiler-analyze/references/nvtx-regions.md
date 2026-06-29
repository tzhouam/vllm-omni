# Named regions to key off (record_function / NVTX)

`trace_reader.py --per-region` rolls up `cat=user_annotation` / `python_function`
durations by name. These names come from `record_function(...)`,
`annotate_context_manager(...)` (which wraps `torch.profiler.record_function`), and NVTX
`range_push/pop`. Known region names in vllm-omni today — search the current tree for the
authoritative list (`grep -rE 'record_function\(|annotate_context_manager\(|range_push\('`):

| Region name | Where | Notes |
|-------------|-------|-------|
| `pipeline_forward` | `vllm_omni/diffusion/worker/diffusion_model_runner.py` | diffusion pipeline step |
| `diffusion_forward` | diffusion worker path (`annotate_context_manager`) | one full forward |
| `diffusion_step` | diffusion worker path (`annotate_context_manager`) | one sampler step |
| (layer-wise NVTX) | `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py` | gated by runtime flag `enable_nvtx_profile`; LocDiT estimator layers via `range_push(name)` |

## Producing the annotations

- `record_function` / `annotate_context_manager` ranges always show up in the chrome
  trace as `cat=user_annotation` (no special flag needed).
- NVTX `range_push/pop` in voxcpm2 is gated by `enable_nvtx_profile` in the model runtime
  config — enable it to get per-layer regions.
- Caveat: `gpu_generation_model_runner.py` deliberately avoids NVTX inside hook functions
  during tracing — do not assume every layer is annotated.

## Adding regions for BDE

The reader is name-driven and needs no code changes to pick up new regions. To get
stage-level attribution for a BDE/DiT forward path, wrap the stages of interest in the
engine code with `with torch.profiler.record_function("bde.<stage>"):` (or the wrapper's
`annotate_context_manager("bde.<stage>")`), then re-run with the profiler. Use a stable
`bde.` prefix so `--per-region` output groups cleanly and is easy to grep.
