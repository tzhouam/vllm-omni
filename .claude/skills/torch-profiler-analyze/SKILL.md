---
name: torch-profiler-analyze
description: Read and understand PyTorch torch.profiler artifacts for any vLLM Omni model (LLM, diffusion, TTS, BDE). Use when asked to analyze a torch profiler trace.json/.json.gz timeline, read or render a flamegraph from export_stacks output, find GPU idle/free bubbles, attribute time to record_function/NVTX regions, compare ranks for stragglers, or locate host/Python hotspots. Model-agnostic; pairs with diffusion-perf-opt (which owns the diffusion optimization playbook).
---

# Read & Understand torch.profiler Output

Turn the artifacts `OmniTorchProfilerWrapper` already writes into an understanding of
*where time goes*: GPU idle bubbles, hot kernels, per-stage (region) cost, rank imbalance,
and host/Python hot paths via flamegraphs. This skill is timing-only and diagnostic — it
does not produce final latency claims (use a non-profiler baseline) or tensor-shape
analysis (use the `ops_rankN.xlsx` `by_shape` sheet).

Works for any vLLM Omni model. For diffusion-specific optimization strategy (USP/CFG/HSDP,
VAE, attention, ROI), use the **diffusion-perf-opt** skill — it delegates raw reading here.

## Where the artifacts are

One timestamped session dir per run, files named `{stem}_rank{N}{suffix}`. See
`references/artifact-formats.md` for the full table and formats. The ones this skill reads:

- `trace_rankN.json[.gz]` — Chrome/Perfetto timeline (always written).
- `stacks_cpu_rankN.txt` / `stacks_cuda_rankN.txt` — folded stacks for flamegraphs (only
  when `torch_profiler_with_stack=True`).

Which artifacts exist depends on `ProfilerConfig` flags in
`vllm_omni/profiler/omni_torch_profiler.py`:
`torch_profiler_with_stack` → stacks + `by_stack` sheet; `torch_profiler_record_shapes` →
`by_shape` sheet; `torch_profiler_with_memory` → memory snapshot. If you need flamegraphs,
profile with `with_stack=True`.

## Workflow

1. **Timeline first.** Run `trace_reader.py` on rank 0. Read `idle_pct`, the largest `GAP`
   blocks (each mapped to the enclosing CPU/Python call), and the top GPU operators. High
   `idle_pct` → host/launch/sync bound; flat busy timeline dominated by few kernels →
   compute bound.
2. **Attribute to stages.** Add `--per-region` to roll up `record_function`/NVTX regions
   (e.g. `pipeline_forward`, `diffusion_step`, `bde.*`). See `references/nvtx-regions.md`
   for known names and how to add BDE regions. Remember nested annotations overcount.
3. **Compare ranks.** Pass all `trace_rank*.json*` in one command (cross-rank summary is
   automatic with >1 trace) to spot stragglers / load imbalance before opening per-rank
   detail.
4. **Flamegraph for host/Python hot paths.** When idle gaps point at CPU/Python work, run
   `flamegraph_reader.py` on `stacks_cpu_rankN.txt` (or `stacks_cuda_rankN.txt` for device
   self-time) to see the hottest leaves, inclusive frames, and stack paths — and render an
   interactive flamegraph.
5. **Cross-check.** Validate the top operators against the `ops_rankN.xlsx` `summary` sheet
   / `profiler_out_N.txt` key_averages table for the same run.

## Commands

Run from the repo root with `/feature/.venv`.

```bash
# Timeline: idle %, gaps, top ops, per-stage regions, cross-rank straggler check
python .claude/skills/torch-profiler-analyze/scripts/trace_reader.py \
  <session>/trace_rank*.json.gz --min-gap-ms 1 --topn 20 --per-region

# Flamegraph: hot-path summary + interactive speedscope (+ SVG if flamegraph.pl/inferno present)
python .claude/skills/torch-profiler-analyze/scripts/flamegraph_reader.py \
  <session>/stacks_cuda_rank0.txt --top 30 --speedscope fg.json --svg fg.svg
```

- `trace_reader.py`: `--min-gap-ms` (gap threshold; lower to 1 for host-stack traces),
  `--topn`, `--per-region`, `--ranks` (force cross-rank summary on a single file).
- `flamegraph_reader.py`: `--top`, `--speedscope <json>` (dependency-free; open at
  speedscope.app), `--svg <svg>` (needs `flamegraph.pl` or `inferno-flamegraph`). With no
  render flag it prints the textual summary only.

## Notes

- Treat `cat=user_annotation` NCCL ranges as enclosing annotations; prefer `cat=kernel`
  for real device communication time.
- `Command Buffer Full` and similar CUPTI entries are profiler overhead, not model targets.
- torch `export_stacks` output needs normalization before `flamegraph.pl` (pytorch#73556);
  `flamegraph_reader.py` handles this — see `references/artifact-formats.md`.
- Scope: torch-profiler artifacts only. For nsys/ncu (system timeline / kernel SOL%,
  roofline) this skill does not apply.
