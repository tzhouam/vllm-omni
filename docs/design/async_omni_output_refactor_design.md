# Refactor Design — Async Omni Output (`gpu_ar_model_runner.py`)

Target: the async-output machinery introduced by PR #4476 ("Async Omni output materialization").
It works and is perf-validated, but ~750 lines of it are interleaved into a 2224-line runner with
implicit contracts. This design makes it **understandable, testable, and rebase-safe** without
changing behaviour. Line refs: branch `dev/worker-runner-audit-comments`.

---

## 1. Problems today (why it's hard to maintain)

| # | Problem | Evidence |
|---|---|---|
| P1 | **Implicit snapshot boundary.** What the background thread may read is encoded in a closure capturing ~14 locals (`output_builder`, `:2162`) plus ad-hoc copies (`:2117-2135`). Nothing enforces "builder reads only frozen state". | `:2117-2138` |
| P2 | **Live-state access from the background thread.** In async mode the deferred builder's `accumulate_full_payload_output` block reads live `self.requests` AND **writes** connector accumulation state (`:1920-1923`) while the main thread's next step mutates both. Likely unreachable in async mode (accumulate is full-payload; the async gate requires `async_chunk`) — but nothing proves or asserts it. (Note: `_resolve_global_request_id` is main-thread-only — called from `execute_model:1073` — not a race.) | `:1912` |
| P3 | **Two output paths that can drift.** Sync mode and async mode share `_build_omni_model_runner_output_from_snapshot` but differ in who runs postprocess, whether payloads were pre-copied, and which branches (prefix cache) are reachable. The differences live in scattered `if`s. | `:2137-2186` |
| P4 | **Model-specific gating scattered as flag probes.** `use_async_omni_output`, `eager_omni_postprocess_before_async_output`, `omni_pooler_payload_include_hidden`, `async_chunk` — read via `getattr` in 4 places; only qwen3_omni uses them. | `:1628-1658` |
| P5 | **Invisible cross-step coupling.** Producer `set_async_sampled_token_ids` (`:2220`) and consumer `_build_model_sampler_output_token_ids` (`:404`) live 1800 lines apart; the `-1`-backfill contract between them is untested. | `:2214`, `:404` |
| P6 | **Fragile construction & lifecycle.** `**kwargs` + `pop()` constructor (`:157-168`); `torch.Event()` ambiguity (`:174`); event recorded but never waited by the builder (`:197-201`); daemon-thread exception silent unless `get_output()` is called (`:209`); dead assignment (`:206`). | `:151-216` |
| P7 | **Everything inline in the runner module.** 5 free payload helpers, 3 data classes, the async class, the policy, the copy stream — all in `gpu_ar_model_runner.py`; the generation runner back-imports from it. | `:51-254` |

## 2. Design principles

1. **One builder, two execution modes.** The same `build(snapshot)` runs inline (sync) or on the
   background thread (async). Mode changes *when* it runs, never *what* it computes.
2. **The snapshot is a type, not a convention.** Everything the builder may read is a field of a
   frozen dataclass. If it's not in the snapshot, the builder can't see it — the P1/P2 contract
   becomes structural instead of tribal knowledge.
3. **Models declare, the runner decides.** Model-side capability is one declarative spec object;
   runner-side compatibility (scheduling mode, prefix cache, spec decode) is one policy function.
4. **Paired things live together.** The sampler-feedback producer/consumer become one class.
5. **Mechanical moves first, semantic fixes second** — each step shippable and hash-verifiable
   against the PR #4476 baseline.

## 3. Target structure

```
vllm_omni/worker/output/
    __init__.py          # re-exports (keeps old import paths alive for NPU/XPU + generation runner)
    payload_copy.py      # clone→copy-stream→event trio (pure-ish, unit-testable)
    snapshot.py          # OmniStepSnapshot — THE frozen builder input
    policy.py            # AsyncOutputSpec (model-declared) + should_use_async_output(runner) gate
    builder.py           # OmniOutputBuilder: (snapshot, services) -> OmniModelRunnerOutput
    async_output.py      # OmniAsyncGPUModelRunnerOutput (thread + event lifecycle only)
    sampler_feedback.py  # AsyncSamplerFeedback: publish() / backfill() pair
    state.py             # ExecuteModelState (kw-only) — moved from the AR module
```

`gpu_ar_model_runner.py` keeps only thin orchestration (~40 lines in `sample_tokens`):

```python
spec = self.model_state.async_output_spec()          # model-declared (P4)
mode = should_use_async_output(self, spec)           # runner compatibility (one gate)
if mode.eager_postprocess:
    self.model_state.run_postprocess(..., eager=True)  # side-effectful part stays on the hot path
snapshot = OmniStepSnapshot.capture(self, spec, ...)   # freeze everything (P1)
if mode.is_async:
    return OmniAsyncGPUModelRunnerOutput(builder=self._output_builder, snapshot=snapshot, ...)
return self._output_builder.build(snapshot)            # same builder, inline (P3)
```

## 4. Component design

### 4.1 `payload_copy.py` — the D2H snapshot trio
Move `_to_cpu_contiguous`, `_clone_cuda_tensor_payload`, `_copy_tensor_payload_to_cpu`,
`_snapshot_tensor_payload_to_cpu_async`, `_AsyncCPUPayloadSnapshot` verbatim. Fixes bundled:
- `torch.cuda.Event()` explicitly (resolves the `:174` ambiguity), or `torch.Event(device=...)`
  with an assert that it's an accelerator event.
- Docstring the **clone-before-copy invariant**: *clone on the producing stream so CUDA-graph
  buffer reuse by step N+1 cannot corrupt the payload; copy stream `wait_stream()`s the producer
  before D2H; `wait()` = event sync + release clones.*
- Unit tests run the dict/list/tuple recursion + CPU-passthrough paths without a GPU; one `@cuda`
  test covers the stream/event path.

### 4.2 `snapshot.py` — `OmniStepSnapshot` (fixes P1, P2)
One frozen dataclass with **every** field the builder reads:

```python
@dataclass(frozen=True)
class OmniStepSnapshot:
    # step geometry
    scheduler_output: SchedulerOutput          # replace()-copied
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    num_scheduled_tokens_np: np.ndarray        # copied
    query_start_loc_cpu: torch.Tensor          # cloned
    seq_len: int
    # sampler results
    valid_sampled_token_ids: list[list[int]]
    logprobs_lists: Any
    prompt_logprobs_dict: dict
    num_nans_in_logits: Any
    # payload tensors (async: CPU snapshot via payload_copy; sync: live GPU refs)
    hidden_states: torch.Tensor
    staged_hidden_states_cpu: torch.Tensor | None
    multimodal_outputs: Any
    async_payload: AsyncCPUPayloadSnapshot | None
    # per-request state COPIED OUT of live dicts (fixes the P2 race)
    accumulation_req_states: dict[str, Any]    # resolved on the main thread at capture time
    postprocess_already_applied: bool
    # pass-through
    kv_connector_output: Any; ec_connector_output: Any
    cudagraph_stats: Any; kv_extracted_req_ids: list[str] | None
```

Key move for **P2**: the live-state access in the builder graph is the `accumulate_full_payload_output`
block (`:1920-1923`), which reads `runner.requests` and writes connector accumulation state from the
background thread. Resolve the needed `req_state`s (or the accumulation itself) **at capture time on the
main thread** and store them in the snapshot. Rule, enforced by review + a lint-style test: **`builder.py`
must not touch `runner.requests`, `runner.input_batch`, or `runner.model_intermediate_buffer`.**
A `capture(runner, spec, ...)` classmethod is the single place all copying happens — the current
`:2117-2135` block becomes its body. (`_resolve_global_request_id` needs no move — it runs only on the
main thread via `execute_model:1073`.)

### 4.3 `policy.py` — declarative gating (fixes P4)
```python
@dataclass(frozen=True)
class AsyncOutputSpec:                  # declared by the MODEL (one object, not 4 flags)
    enabled: bool = False
    include_hidden_payload: bool = True   # talker: False (code2wav only needs codes)
    eager_postprocess: bool = False       # talker: True (writes hidden_states['last'])

def should_use_async_output(runner, spec) -> OutputMode:
    # runner-side compatibility, one place, each condition commented with WHY:
    # async scheduling on; no prefix cache (snapshot doesn't cover cache branches);
    # no spec decode; async_chunk on; no routed-experts return;
    # spec.enabled; postprocess implies spec.eager_postprocess.
```
qwen3_omni declares `AsyncOutputSpec(enabled=True, …)` on thinker/talker; every `getattr(model,
"use_async_omni_output"/…)` probe disappears. When B-align lands, the spec is served by
`OmniModelState.async_output_spec()`.

### 4.4 `builder.py` — one builder, two modes (fixes P3)
`OmniOutputBuilder` is constructed once per runner with its **stable** collaborators (model-state
handle, pooler-payload helpers, sparse-audio router). `build(snapshot)`:
1. `snapshot.async_payload.wait()` if present (async mode),
2. the current `_build_omni_model_runner_output_from_snapshot` body, reading **only** snapshot
   fields + stable collaborators,
3. postprocess only if `not snapshot.postprocess_already_applied`.
The sync path calls the same method inline — deleting the mode-conditional drift surface. The
prefix-cache branches stay (sync-only by policy) but now visibly so: `assert snapshot.async_payload
is None or self.omni_prefix_cache is None`.

### 4.5 `async_output.py` — lifecycle only (fixes P6)
`OmniAsyncGPUModelRunnerOutput` keeps exactly: upstream-compatible sampled-token D2H, the
background thread, `get_output()`. Changes:
- **Explicit kw-only constructor** (`sampled_token_ids=`, `logprobs_tensors=`,
  `invalid_req_indices=`, `copy_stream=`, `vocab_size=`, `routed_experts=`, `builder=`,
  `snapshot=`, `device=`) — deletes the `**kwargs`+`pop`+guard block (`:157-168`) and the dead
  assignment (`:206`).
- **Event contract made explicit:** the async-copy event gates only the *sampler feedback*
  tensors, consumed post-join by `super().get_output()` — rename it
  `sampler_feedback_ready_event` and add one comment + one assert in `_build_output_in_background`
  that the builder consumes only `snapshot` (whose own `async_payload` event it waits).
- **Exceptions surface immediately:** `logger.exception` inside the thread at catch time (the
  re-raise in `get_output()` stays); a dropped output object no longer swallows errors silently.
- Direct attribute access instead of the `getattr(self, "_background_thread", None)` defaults (`:239`).

### 4.6 `sampler_feedback.py` — pair the coupling (fixes P5)
```python
class AsyncSamplerFeedback:
    def publish(self, sampled_token_ids_cpu, ready_event): ...   # today: :2220
    def backfill(self, histories, prev_req_id_to_index): ...     # today: :404-451 resolve loop
```
Both ends of the `-1`-placeholder protocol in one file, with a unit test that publishes a fake
tensor+event and asserts backfill resolves placeholders (and the trailing-`-1` truncation question
from the base/AR divergence gets decided here, once). `input_batch` keeps only the storage slot.

### 4.7 `state.py` — `ExecuteModelState` neutral home
Move the NamedTuple out of the AR module (generation runner's backwards import, positional-`None`
construction — audit "Fork-fragility"). Kw-only construction helper + a field-parity test vs
upstream's `ExecuteModelState`.

## 5. Invariants — written down and tested

| Invariant | Enforced by |
|---|---|
| Builder reads only `OmniStepSnapshot` + stable collaborators | snapshot type + no-live-state test (grep-style) |
| CUDA payloads cloned on producing stream before copy-stream D2H | docstring + `@cuda` test in `payload_copy` |
| Eager postprocess ran ⇔ `snapshot.postprocess_already_applied` | policy sets it; builder asserts |
| Async mode ⇒ no prefix cache | `should_use_async_output` + builder assert |
| Sampler-feedback tensors read only after `get_output()` join | event rename + comment + `AsyncSamplerFeedback` test |
| Sync output ≡ async output | golden test: same fake snapshot through both modes, compare |

## 6. Migration plan (each step shippable, hash-verified)

| Step | Change | Risk | Guard |
|---|---|---|---|
| 0 | Characterization: builder golden test on a synthetic `snapshot`-shaped input; keep PR #4476's 11 unit tests green; record text/wav sha256 e2e baseline | — | — |
| 1 | **Mechanical move** of trio + 3 classes to `worker/output/` with re-export shims (NPU/generation imports unchanged) | low | import test + step 0 |
| 2 | Explicit constructor; event/exception/dead-code fixes (§4.5) | low | step 0 tests |
| 3 | Introduce `OmniStepSnapshot.capture()`; replace the closure; sync path through the same builder | **medium** (this is the real refactor) | golden + e2e hash |
| 4 | `AsyncSamplerFeedback` pairing; decide the trailing-`-1` divergence | medium | new backfill unit test |
| 5 | `AsyncOutputSpec` + policy; delete flag probes; qwen3_omni declares the spec | low | policy truth-table test |
| 6 | (with B-align) fold spec/builder into `OmniModelState`; the package becomes the state's output component | — | — |

## 7. Open questions (resolve during step 3)

1. **P2 race audit:** enumerate every live access in the builder graph (the
   `accumulate_full_payload_output` block at `:1920`; anything inside
   `_stage_deferred_prefix_cache_mm_outputs`) and either move to capture-time, prove
   unreachable in async mode + assert, or prove main-thread-only.
   (`_resolve_global_request_id` checked: main-thread-only via `execute_model:1073`.)
2. **`torch.Event()`** — confirm it resolves to a CUDA event on this build before renaming
   (if it's a CPU event today, the fix may change timing; measure).
3. **NPU runner** — it copies parts of this machinery; decide whether NPU imports from
   `worker/output/` (preferred) or stays forked until the platform cleanup.
4. Does the generation runner ever need async omni output? If provably not, its
   `ExecuteModelState` construction can be simplified against `state.py` directly.
