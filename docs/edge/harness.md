# The measurement harness

`benchmarks/edge_harness/` — the tooling every number in this fork was produced
with. It exists because five earlier conclusions were wrong; see
[methodology.md](methodology.md).

## Running a comparison

Arms are subprocess invocations of a single-run benchmark, so the measurement
code stays the one that was verified; the harness decides what to run, when, and
how to read it.

```bash
python benchmarks/edge_harness/bench_harness.py \
    --spec arms.json --passes 3 --metric rss_anon_tree_mb \
    --keys ready_s,init_s,rss_anon_tree_mb,peak_rss_anon_mb \
    --cores 56-79 --out summary.json
```

`arms.json` is `{"arms": [{"tag": ..., "command": [...], "env": {...}}]}`.
Give each arm its own `VLLM_CACHE_ROOT` if the arms differ in anything that
changes the traced graph.

It rotates arm order every pass, reports spread, labels contended passes, refuses to
start when the target cores are busy, prints `NOT A RANKING` when the top two
arms differ by less than their own spread, and stamps provenance into the
output. `--check expectations.json` turns it into a regression gate.

## The single-run benchmarks

| script | measures |
|---|---|
| `bench_init_memory.py` | `ready_s` (process start → serving), peak `RssAnon` sampled through load, resident memory over the **process tree**, memory at a filled context |
| `bench_cpu_vllm.py` | pp512 / tg128, mirroring `llama-bench` semantics |
| `profile_cpu_step.py` | per-op decode cost, measured by difference so it matches the wall clock |
| `memory_ledger.py` | attributes resident memory to tensor storages, KV cache and mappings, with `mincore(2)` residency |
| `fidelity_kl.py` | top-1 agreement and KL against a bf16 reference, comparable with `llama-perplexity --kl-divergence` |

Two arguments matter more than they look:

- `--context-len` on `bench_init_memory.py`. Without it the KV cache is
  allocated and never touched, and a memory number taken that way says nothing
  about cache sizing. That is how "KV sizing is not a lever" happened.
- `--prompt-len` on `profile_cpu_step.py`. The default of 1 measures an empty
  cache; Spark's sliding layers have a 512-token window, so attention cost at a
  1-token prompt is not attention cost at context.

## The agent harness

`benchmarks/edge_harness/android_agent/` drives a phone (real, over `adb`, or a
scripted simulator that emits real uiautomator XML) and scores tasks by the
phone's own settings and screen rather than by the transcript.

Scoring reads the *setting* where one exists and the UI dump otherwise, so the
same predicate works on both backends; an agent that turns Wi-Fi on and then
presses Home has still turned Wi-Fi on. Tasks carry variants — a renamed
control, reordered rows, a plausible decoy — because repeats at temperature 0
against one fixture measure repeatability, not whether the agent can do the task.

```bash
python benchmarks/edge_harness/android_agent/run_suite.py \
    --base-url http://127.0.0.1:8077/v1 --repeat 5          # simulator
python benchmarks/edge_harness/android_agent/run_suite.py --device adb --repeat 5
```

Tasks needing flip history (a phone cannot report which switches were touched)
are skipped with a reason on a real device rather than scored from final state.

## Tests

```bash
pytest benchmarks/edge_harness/test_bench_harness.py \
       benchmarks/edge_harness/test_fidelity_kl.py \
       benchmarks/edge_harness/android_agent/test_android_agent.py
```

They pin judgement, not plumbing: that arm order rotates, that `best` is the
minimum for memory, that a difference inside the noise is not reported as a
ranking, that the alarm oracle needs a saved alarm rather than a typed digit.
