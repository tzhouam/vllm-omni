# Local Agent whole-request profiling

`profile.py` records **batch size 1, one active request** for a complete local
Agent route: observation, retrieval, model output, tool decisions, verification,
and final answer. It is a library harness. A model-stage timing callback or a
reference answer without the Agent loop is recorded as incomplete evidence.

The [candidate catalog](../../vllm_omni/edge/agent/catalog.py) is a research
queue, with separate unpruned, pruned, and capacity-only artifacts. Its GB
figures are declared file sizes, not measured load peaks. Where exact pinned
LFS bytes replace a rounded plan estimate, the original number remains in
`declared_plan_size_gb_decimal`. A download preflight
requires a pinned artifact commit and explicit proof that a local runtime
supports that exact candidate and its quantization; a passing result permits
measurement only. No catalog entry becomes a default route automatically.
The three capacity-only entries always refuse whole-weight downloads here.

The [GSQ-RCO unpruned Q2_0](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
and [pruned Coder IQ1_M](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
are distinct artifacts. Both require their two GGUF shards, and vision also
requires the BF16 projector. Their smaller resident estimates require a
runtime-proven lazy mapping mode for the 28.8 GB n-gram shard. The
[Qwen3.5 REAP 76B checkpoint](https://huggingface.co/0xSero/Qwen3.5-76B)
is text-only, so it is excluded from screen routes; the
[Qwen3-Coder-Next REAP 40B GGUF](https://huggingface.co/mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-GGUF)
names a separate pruned checkpoint. The
[DeepSeek V4 REAP25 artifact](https://huggingface.co/ljupco/DeepSeek-V4-Flash-0731-REAP25-GGUF)
is built for ds4/DwarfStar and needs its own native runtime probe. Model cards
and license metadata must be reconciled before redistribution.

Call `await run_profile(...)` with:

- One or more exact `ProfileRoute` values, including checkpoint revision,
  artifact SHA-256, precision, backend, and expected placement.
- `AgentCase` values for each of `short`, `medium`, and `long`. A run covers one
  task class, but can include Chinese and English cases. The same cases and
  repetition order are used for every route.
- An async runner `(route, case, emit) -> AgentRunResult` that executes the
  complete Agent request. Call `emit("assistant_text_delta", text)` on the first
  user-visible token and subsequent text deltas. Report the **actual** model,
  artifact, backend, placement, placement evidence, tool decisions, and final
  answer. `complete_agent_trace=True` and `trace_scope="agent_e2e"` must be
  justified by the execution trace; the harness does not infer these from a
  stage result.
- An independent evaluator `(case, result) -> Evaluation` for task success,
  quality, and tool safety. A telemetry callback returns observed RAM/VRAM,
  power, temperature, and other numeric fields. Their sampled peaks are lower
  bounds on instantaneous peaks.
- A `prepare(route) -> Preparation` callback that times a confirmed cold load.
  A no-op or already-loaded preparation cannot provide cold-start evidence.

The default `ProfileConfig` performs separate warmups, **20 measured complete
requests per input length**, and a **30-minute sequential** run for each route.
The 30-minute threshold counts active whole-Agent request time; wall time and
per-request evaluator time are separate, so a slow offline evaluator cannot
substitute for sustained Agent operation. TTFT starts at request submission and
the answer timer stops when the runner finishes, before reference evaluation.
Short configurations are useful for harness tests, but their summaries have
`protocol_compliant=false` and cannot qualify a route. The output directory
contains a unique run subdirectory with `samples.jsonl` and `summary.json`.
Every request record includes its full prompt and answer, event timestamps,
evaluation, actual route, and raw telemetry samples. Keep this directory in an
appropriate private location when test cases contain sensitive observations.

`RouteProfile.qualification_fields(...)` maps raw counts and timings to the
router's `Qualification` fields. It deliberately defaults memory admission,
cancellation/recovery, and independently verified placement to false; callers
must supply those checks from separate evidence. The resulting route is still
unqualified when any length lacks the required samples or first-token events,
when sustained operation is missing, or when a stage-only trace was supplied.

## Native Windows paired-Agent entrypoint

`native_profile.py` connects this harness to `native_app.build_controller` and
the same Omni `StageRuntime`/llama.cpp route used by the desktop Agent. It
starts one route cold, keeps it resident while requests run serially, and
records actual ordered Agent events, tool decisions, first visible deltas,
full-answer times, RAM/VRAM samples, NVIDIA GPU power and temperature. System
power is not available from these sensors. Each run pins the native config
hash, model GGUF hash, optional projector hash, runtime binary hash, stated
checkpoint revision, precision, OS/driver, and AC/battery condition.
New profiles distinguish the loaded Omni source version from installed
distribution metadata and record SHA-256 digests for the imported Omni source
and loaded Agent/dependency runtime, including imported vLLM and shared stage
contract Python sources. Release review requires those digests to
match the code currently loaded by the app, along with the full native behavior
config. Older runs without the digests cannot qualify the current release.

The fixed task set pairs Chinese and English at short, medium, and long input
lengths. `browser_text` covers a local text page on CPU or another text route;
`browser_vision` separately covers browser screenshot and Windows desktop
screen capture on an admitted image route. The other classes cover read-only
mouse-speed setting, seeded encrypted-memory recall, and a narrow Python-output
question. Visual identifiers are rendered in an image, not in page body text.
The length buckets use fixed prompt text and record characters, UTF-8 bytes,
and a prompt SHA-256; model-specific tokenizer counts are not inferred.
Browser navigation is confined to a loopback fixture;
browser writes and setting changes are rejected at the tool boundary. A
`browser_open` URL is allowed without approval only when it exactly matches
an explicit URL in the trusted user task; model output and recalled page or
memory text cannot authorize another destination. Following a page link without
approval additionally requires that the visible, same-origin target URL was
explicitly present in that trusted task; other link targets require exact-action
approval. The managed browser opens a fresh isolated Edge context with no
restored tabs. It installs HTTP and WebSocket guards before the first page
exists. Only an exact authorized top-level GET navigation can reach the
network; page-initiated background HTTP requests, frames, redirects, and
WebSockets are blocked. This deliberately limits ordinary open-web images,
scripts, fonts, and stylesheets; inline/data assets remain available, and
the fixed visual fixture embeds its image as a data URL.

Production `browser_click` and `browser_fill` return an explicit refusal
before target inspection, approval, or page action. Native form POST is not
enabled: even constructing `FormData` can dispatch page JavaScript, while a
click or fill may invoke handlers or external protocols outside HTTP routing.
Injected mock backends exercise the generic approval contract but do not
establish production browser-write support. The context is discarded when
the Agent closes, so browser cookies and tabs do not persist between app
sessions. Browser process-level traffic is outside page routing; the fixed
local fixture is not evidence of general open-web safety.
`screen_capture` observes the visible foreground window, crops to its
physical screen rectangle before model resizing, and records that rectangle
and original/output dimensions. It is available only when the trusted task
explicitly requests desktop capture; a request to inspect a browser image grants
only `browser_screenshot`. The desktop fixture uses a unique Edge title, then
verifies that exact window is foreground before the timed request and at the
actual capture. If Windows denies focus, setup fails with a visibility blocker
instead of scoring a blank or obscured capture as model quality. A
separate, local-NTFS encrypted SQLite database is reset and seeded before each
timed request, with setup evidence in the raw record. This avoids memory
leakage across repeats or candidate routes. The code case checks static code
reasoning only; no code-execution tool is qualified by it.

Use native Windows Python with this checkout on `PYTHONPATH` and `-X utf8`:

```powershell
$repo = (Get-Item .).FullName
$env:PYTHONPATH = $repo
python -X utf8 -m vllm_omni.edge.agent.native_app `
  --config (Join-Path $repo 'benchmarks/edge_agent/configs/windows_laptop_gemma4_31b_hybrid.experimental.json')
```

That starts the PySide6 desktop Agent. The bundled route is visibly
experimental until a reviewed qualification bundle is installed; its model
and projector files must exist at the pinned paths in the local config. For
the whole-request profiler, run:

```powershell
$repo = (Get-Item .).FullName
$env:PYTHONPATH = $repo
python -X utf8 -m benchmarks.edge_agent.native_profile `
  --config (Join-Path $repo 'benchmarks/edge_agent/configs/windows_laptop_spark_cpu.experimental.json') `
  --lineage (Join-Path $repo 'benchmarks/edge_agent/configs/windows_laptop_spark_cpu.lineage.experimental.json') `
  --output-dir (Join-Path $repo 'benchmarks/edge_agent/results') `
  --task-class memory --smoke
```

Omit `--smoke` for the full protocol: separate warmups, 20 measured requests
per length, and 30 minutes of consecutive single-request Agent work. The
short smoke runs one measured pass over every Chinese/English case per length
and no endurance; it is always `protocol_compliant=false`. Text-only routes
are recorded as blocked for the visual suite until a pinned vision projector
is available. Power-condition changes abort before the next request and
invalidate telemetry samples in the active request.

The output `index.json` points to each `summary.json` and SHA-256-bound raw
`samples.jsonl`. Every new native run includes sanitized full Agent events,
so `evidence.audit_summary(path)` can recompute event order, answer and tool
evaluation, TTFT, telemetry, counts, and protocol flags. Older smoke runs
without full Agent event records fail that audit. Profiling never writes a
`qualification_file`; independent memory admission, cancellation/recovery,
placement logs, checkpoint lineage, and broader task-quality evidence are
still required before a route may become a default. Do not load an unreviewed
claim-only qualification JSON into `native_app`. The gated promotion and
loader contract is described in [QUALIFICATION.md](QUALIFICATION.md).

For the pinned llama.cpp Vulkan backend, `Vulkan_Host+Vulkan0` is a distinct
experimental placement. It requires `host_mapped_expert_layers` and explicit
host/GPU weight budgets; `cpu_moe_layers` and `gpu_layers` must be unset. The
backend still passes `--n-cpu-moe` to llama.cpp, but verifies all requested
expert tensor overrides as exactly `Vulkan_Host`, every layer assignment as
`Vulkan0`, and the vision backend as `Vulkan0` when a projector is loaded.
`Vulkan_Host` model buffers are charged to host RAM. Startup override logs
establish only selection of a preferred buffer type: mmap or pinned-memory
fallback can change final storage, and they do not establish where expert
compute runs. The execution plan marks this as `override_selection_only`, and
the release qualification placement gate refuses this route pending deeper
runtime evidence.

The [public aggregate](public_evidence/agent_native_smokes_20261005.json)
lists each completed run ID, raw/index hashes, audited counts, failures, and
protocol status without publishing private prompts or screen/page/tool content.
The [rolling status](STATUS.md) explains what those observations establish.
Raw `results/native_*/index.json`, summaries, JSONL, and unsanitized logs are
gitignored and available only on the profiling host. Earlier six-of-six memory
fixtures used the wrong seed event kind and cannot validate retrieval; a
corrected Gemma memory smoke is recorded separately. All short smokes remain
below the 20-per-length and 30-minute qualification thresholds.

Four completed Gemma fixed-memory runs passed the 20-per-length and
30-minute protocol at their recorded code states. The newest completed
[hash-bound run](public_evidence/agent_native_smokes_20261005.json)
`native_4bba0301e4d04f4f9af9dd45e1c6deea` passed 60/60 measured
fixed-memory requests and 611/611 sequential endurance requests over
1,802.26 seconds of active Agent work. Raw trace, measurement protocol and
fixed-suite case audits pass. Its raw JSONL SHA-256 is
`26f119de4e9db43756084cc1838a4423bb276d2f63c100422c45c30981a76ed5`;
the [rolling status](STATUS.md) gives the index/summary hashes and p50/p95.
Its source/runtime digests matched at measurement time, but subsequent browser
boundary changes make it historical for final-code release matching. The
earlier three full profiles, including `native_37aad46efd4f4893a497a7bfaa83599d`,
are also historical. No default route is qualified.

The post-hardening browser-text-only smoke
`native_f97dc44f38024b2d82794e73b1b745de` passed 6/6 with a valid raw
trace audit. It has two measured requests per length and no endurance. A full
browser-text protocol is running, with no audited outcome to report yet. An
earlier browser-vision smoke passed 12/12, but a later desktop capture run
without foreground verification reconstructed only 6/12 and cannot isolate
model quality from visibility. A later visual attempt completed one
warmup and one measured browser screenshot, then stopped before the next
desktop case when Windows could not verify the Edge fixture in the
foreground. All indexed visual attempts predate the latest browser hardening;
no indexed browser or visual class has completed the full protocol on that
source. The separate
[bilingual memory-quality probe](public_evidence/agent_memory_quality_20261006.json)
retains a 3/4 failure and a later 4/4 observation, both unsigned; the 4/4
run predates the later exact-content memory index. Default-route promotion
still requires independent signed memory, cancellation, placement, lineage,
and quality evidence. The current Gemma artifact probe binds official
GGUF/projector LFS bytes, revision and license metadata but does not verify
base-model provenance or replace human review. The signed quality gate accepts
only `comparison: exact_sha256` with equal lowercase answer/reference hashes;
semantic or graded quality needs a separate reviewed comparator.
