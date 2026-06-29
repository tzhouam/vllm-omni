# torch.profiler artifact formats

All artifacts are written by `OmniTorchProfilerWrapper`
(`vllm_omni/profiler/omni_torch_profiler.py`) into one timestamped session directory
per profiling run, e.g. `…/20260403-034200_stage_0_llm/`. Filenames use the pattern
`{stem}_rank{N}{suffix}` (`_artifact_path`). A run typically contains:

| File | Produced when | Reader |
|------|---------------|--------|
| `trace_rankN.json` / `trace_rankN.json.gz` | always (gz if `gzip` available) | `trace_reader.py` |
| `stacks_cpu_rankN.txt` | `torch_profiler_with_stack=True` | `flamegraph_reader.py` |
| `stacks_cuda_rankN.txt` | `with_stack=True` **and** CUDA-like activity | `flamegraph_reader.py` |
| `ops_rankN.xlsx` | always; sheets `summary`, `by_shape`*, `by_stack`* | open directly |
| `profiler_out_N.txt` | `torch_profiler_dump_cuda_time_total=True` | read directly |
| `memory_snapshot_rankN.pickle` | `torch_profiler_with_memory=True` | `torch.cuda` memory viz |

\* `by_shape` needs `torch_profiler_record_shapes=True`; `by_stack` needs `with_stack=True`.

## Chrome trace JSON (`trace_rankN.json[.gz]`)

Produced by `prof.export_chrome_trace()`. Chrome/Perfetto-compatible. Either a top-level
object with a `traceEvents` list, or (some exporters) the raw event list directly —
`trace_reader.py` handles both.

Each duration event has `name`, `cat`, `ts` (µs), `dur` (µs), `pid`, `tid`. Categories
used by the reader:

- **GPU work** — `cat in {kernel, gpu_memcpy, gpu_memset}` → busy/idle union, top operators.
- **CPU/host** — `cat in {python_function, user_annotation, cpu_op, cuda_runtime, cuda_driver}`
  → gap attribution and named-region rollup.
- `record_function("name")` ranges appear as `cat=user_annotation`. Rolling these up
  (`--per-region`) gives logical stage weights, but **nested annotations overcount** — a
  parent region's `dur` includes its children. Treat the rollup as relative weight, not
  exclusive time. For real device work prefer `cat=kernel` over `cat=user_annotation`
  NCCL ranges.

## Folded stacks (`stacks_cpu_rankN.txt`, `stacks_cuda_rankN.txt`)

Produced by `torch.profiler.export_stacks(path, metric=...)` with
`metric="self_cpu_time_total"` (cpu file) or `"self_cuda_time_total"` (cuda file).
Brendan-Gregg "folded" / "collapsed" format, one stack per line:

```
frameA;frameB;...;leaf <value>
```

`<value>` is the metric in **microseconds**. Frames are `;`-separated, root first, leaf
last. Note torch frames frequently **contain spaces** (e.g. `torch/nn/module.py(1): forward`),
so the value must be split off the *last* whitespace token — `flamegraph_reader.py` uses
`rpartition(" ")`, not a naive `split`.

### flamegraph.pl compatibility caveat (pytorch#73556)

torch's `export_stacks()` output is not always accepted as-is by Brendan Gregg's
`flamegraph.pl` (stray formatting / non-numeric trailing tokens). `flamegraph_reader.py`
**normalizes** every parsed line back to `stack<space>integer` before invoking
`flamegraph.pl`/`inferno`, and drops lines whose trailing token is not numeric.

### Rendering options

- `--speedscope out.json` — dependency-free. Emits the speedscope **sampled** schema
  (`$schema`, `shared.frames`, `profiles[0].{samples,weights}`). Open at
  <https://www.speedscope.app> (or the offline app). Each folded line → one sample,
  weight = µs value.
- `--svg out.svg` — needs `flamegraph.pl` or `inferno-flamegraph` on `PATH`. If neither
  is present the normalized `.folded` file is left on disk and a hint is printed.
- Default (neither flag) — textual summary only: hottest leaves (self time), hottest
  frames (inclusive time), hottest full stack paths.
