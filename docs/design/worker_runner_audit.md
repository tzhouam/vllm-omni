# Worker / ModelRunner Audit

**Level 1 = error type** (model-specific issues listed individually). **Level 2 = `file:line` — description.**
Branch `dev/worker-runner-audit-comments` (line numbers match the inline annotations). Comments-only;
no behaviour change. `file` is relative to `vllm_omni/worker/` unless noted. Rescan focus: `gpu_model_runner.py`.

---

## Dead / unused code
- `gpu_model_runner.py:83` — `_omni_last_model_output` written by `_model_forward` (`:2090`) but never read; remove (NPU mirrors the dead write).
- `gpu_model_runner.py:217` — `_sampled_token_ids_cpu_override` dead extension point: no model declares `supports_sampled_token_ids_cpu_override`, so it's always `None`.
- `gpu_model_runner.py:1355` — `_resolve_prompt_embeds_cpu` (tensor+payload aware, cleaner) is only called by the dead function below.
- `gpu_model_runner.py:1384` — `_decode_and_store_request_payloads` is never called; it's the intended (unwired) consolidation of the two inline decoders in `_update_states`.
- `gpu_model_runner.py:1534` — unused parameter `scheduler_output` in `_process_additional_information_updates`.
- `gpu_model_runner.py:978` — `_dummy_sampler_run` overload exists only for CPU hidden_states from dummy-weight loads (`:977` "can we guarantee GPU tensor?"); removable if the invariant holds.
- `gpu_ar_model_runner.py:742` — `_maybe_update_prefix_cache` is **dead in the GPU path** (verified: only the NPU runner calls its own copy; GPU `execute_model` uses the async `schedule_async_write` pipeline). Remove or relocate to NPU.
- `gpu_ar_model_runner.py:1604` — dead `getattr(self, "model_config")` → `vllm_config.model_config` fallback: `self.model_config` is always set by upstream `GPUModelRunner.__init__`. Same pattern in `_should_use_async_omni_output`.
- `gpu_generation_model_runner.py:538` — `if hasattr(self.model, "forward")` is always True for any `nn.Module`, so the branch is always taken and the following `raise` is unreachable (and its message advertises `sample`/`diffuse` interfaces that are never probed).
- `gpu_generation_model_runner.py:955` — `output = None` in `profile_run` is a dead local (never read; upstream assigns a real result and `del`s it).
- `gpu_ar_model_runner.py:207` — dead assignment in `OmniAsyncGPUModelRunnerOutput.__init__` (immediately overwritten below); drop the line.
- `gpu_ar_model_runner.py:241` — the `getattr(self, "_background_thread", None)` / `_background_exception` defaults are dead defensive code (both always set in `__init__`); access the attributes directly.

## Stale over-provisioned workaround (verified)
- `gpu_model_runner.py:145` — FA3 `scheduler_metadata` resize in `initialize_metadata_builders` (`:144`); **verified** (kernel probe + real Qwen2.5-0.5B FA3 full-cudagraph run) size doesn't scale with `num_splits` and upstream pre-alloc already covers it (129 ≥ 97; this computed 1025 → ~10× over-alloc). Remove behind a capture-size regression test.

## Deprecated / legacy remnant
- `gpu_model_runner.py:1483` — `runtime_additional_information` back-compat alias emitted alongside `model_intermediate_buffer` ("need to be removed in the refactor"); remove.
- `gpu_model_runner.py:1630` — `_update_additional_information` is deprecated ("should be removed"); remove.
- `gpu_model_runner.py:1826` — its call site in `_preprocess` should use `_update_intermediate_buffer` directly (inline comment "should use the update model intermediate buffer helper instead").
- `gpu_model_runner.py:2133` — `_merge_additional_information_update` is a deprecated alias (warns + forwards to `_update_intermediate_buffer`); callers in `_preprocess` should call the new name directly.
- `gpu_generation_model_runner.py:422` — `pooler_output` "no longer used for multimodal data"; class docstring `:53` still contradicts this.

## Possibly-redundant subsystem: cross-stage prompt_embeds transfer
- `gpu_model_runner.py:1610` — "double check if prompt_embeds are really needed — the vLLM EngineCoreRequest already carries prompt_embeds." If upstream now provides them natively, the whole omni prompt_embeds decode+overlay path is redundant.
- Sites of that path: inline decode in `_update_states` (`:711`), `_resolve_prompt_embeds_cpu` (`:1355`), `_decode_and_store_request_payloads` (`:1384`), `_collect_additional_information_for_prefill` (`:1602`), overlay in `_preprocess` (`:1839`). Also flagged `:1396` "same prompt embedding necessary problem", `:1601` "if not needed, this function can be removed".

## Possibly-unnecessary subsystem: process-level memory estimation
- `base.py:104` — `determine_available_memory` does per-process NVML accounting (`nvmlDeviceGetComputeRunningProcesses`, `:111-118`) so **multiple stages can initialize in parallel** on one GPU without double-counting each other's reserved memory; supporting code: `base.py:138` (`get_process_gpu_memory`), `gpu_memory_utils.py:22` (`is_process_scoped_memory_available`), `:37` (`parse_cuda_visible_devices`), `:59` (`get_device_handle`), `:69` (`get_process_gpu_memory`), `memory_utils.py:19` (`request_memory_tolerant`).
- **This whole per-process estimation may no longer be needed.** Investigate before keeping it:
  1. **Does the orchestrator still use parallel stage init?** The per-process accounting only earns its keep if stages initialize concurrently on the same device. If init is now sequential (or stages are on separate devices), plain profiling suffices and this NVML machinery can go.
  2. **Does engine init actually cost significant wall-clock?** Confirm whether parallel init was solving a real startup-latency problem (measure single-stage vs pipeline init time). If init is cheap, the complexity isn't justified.
  3. **Can we do rapid init instead?** Investigate a faster init path (e.g. shared/pre-warmed memory pool, lazy stage bring-up, or reusing a profiled estimate) that would remove the need for per-process NVML estimation altogether.
- If (1)–(3) show it's unnecessary: drop `determine_available_memory`'s NVML branch back to upstream profiling, and retire `gpu_memory_utils.py` / `is_process_scoped_memory_available` / `request_memory_tolerant`. (Guard with the Phase-0 characterization tests first — currently untested.)

## `additional_information` subsystem — rename / remove-if-unused / extract to a Mixin
Three coordinated actions on the whole `additional_information` / `model_intermediate_buffer` /
`runtime_additional_information` cluster (the naming/deprecation items below are facets of this):
- **Rename to one consistent name.** Settle on `model_intermediate_buffer` (or another meaningful,
  non-generic name) and retire the aliases `additional_information` / `runtime_additional_information`.
  Sites: `gpu_model_runner.py:657, 1375, 1394, 1408, 1414, 1483 (alias emit), 1526, 1599, 1627`.
- **Remove what's no longer used.** `_update_additional_information` (`:1630`, deprecated),
  `_merge_additional_information_update` alias (`:2133`), the `runtime_additional_information`
  back-compat alias (`:1483`), and — pending the prompt_embeds investigation above —
  `_decode_and_store_request_payloads` (`:1384`) / `_resolve_prompt_embeds_cpu` (`:1355`) /
  `_collect_additional_information_for_prefill` (`:1602`).
- **Extract the rest into a Mixin** (`AdditionalInformationMixin` / `IntermediateBufferMixin`, later
  folded into `OmniModelState`). The subsystem is ~11 methods currently interleaved into
  `OmniGPUModelRunner`: `_gather_runtime_additional_information` (`:1415`), `_build_model_kwargs_extra`
  (`:1476`, partial), `_process_additional_information_updates` (`:1529`),
  `_collect_additional_information_for_prefill` (`:1602`), `_update_additional_information` (`:1630`),
  `_store_value` (`:2094`), `_update_intermediate_buffer` (`:2112`),
  `_merge_additional_information_update` (`:2133`), `_update_streaming_input_additional_info`
  (`:2137`), plus the inline decode in `_update_states` (`:727`) and the `model_intermediate_buffer`
  field itself. Pulling them into a Mixin slims the runner and gives the B-align migration a single
  unit to move (worker_v2 already has `OmniIntermediateBuffer`).

## Divergent duplicate — needs merge (rfc8 A1)
- `gpu_model_runner.py:345` ↔ `gpu_ar_model_runner.py:382` — `_build_model_sampler_output_token_ids`; the AR copy adds a trailing `-1` truncation (`gpu_ar_model_runner.py:448`) the base lacks → behaviour differs. Also should be split (`:344`) into "build history" + "resolve async placeholders".
- `gpu_model_runner.py:406` ↔ `gpu_ar_model_runner.py:461` — `_sampling_metadata_for_model_sampler` (AR adds a `skips_model_sampler_output_token_history` short-circuit).
- `gpu_model_runner.py:989` ↔ `gpu_generation_model_runner.py:552` — `_dummy_run` ~90% identical to upstream, ~88% cross-runner. Verified deltas in the generation copy: (a) **missing** the base `has_preprocess` inputs_embeds input branch (see capture≠replay below), (b) added MammothModa2 dummy-runtime-info, (c) drops the talker-MTP cudagraph-record block, (d) returns `(hidden, None)` vs base `(hidden, hidden[logit_indices])`. Extract shared body + hook the 4 deltas.
- `gpu_ar_model_runner.py:1008` (`execute_model`, ~430 L) ↔ `gpu_generation_model_runner.py:110` — the connector-recv/flush + ngram `scheduler_output` copy + KV-preemption + routed-experts-clear **preamble is copy-pasted and divergent** (AR adds warmup-clear, prefix-cache drain, KV-transfer-before-update-states, `commit_deferred_mm_outputs`). Extract a shared `OmniGPUModelRunner` preamble helper + hook the omni deltas.
- `gpu_generation_model_runner.py:901` — `profile_run` is a near-duplicate of upstream `GPUModelRunner.profile_run` (only intended delta: skip the last-rank sampler/pooler dummy); already drifted (`logger.info` vs upstream `logger.info_once`).
- `gpu_generation_model_runner.py:77` — `_update_request_states` (generation-only, async_chunk path) hand-rolls upstream `_update_states`' persistent-batch remove/add + M-RoPE; no AR/base counterpart, fork-drift hazard (marked `# OMNI:`).

## Correctness — CUDA-graph capture == replay
- `gpu_generation_model_runner.py:857` — `_dummy_run`'s input branch is **missing the base's `has_preprocess` path** (which loads both `input_ids` and `inputs_embeds`, `gpu_model_runner.py:1215`; verified by reading both). A generation-stage model declaring `has_preprocess` would capture the graph on `input_ids` only while runtime `_preprocess` feeds `inputs_embeds` → **silent capture≠runtime bug**. (High.)
- `gpu_generation_model_runner.py:425-428` — the bare-`torch.Tensor` return branch asserts `shape[0]==1` **and** `shape[0]==num_reqs`, together forcing `num_reqs==1`; batched requests are silently unsupported on that path. Document/guard or slice per-request.

## Correctness — logic bugs (deep AR/generation review)
- **`gpu_generation_model_runner.py:183`** (High) — the unconditional no-tokens early return **shadows** the careful no-tokens block ~10 lines below (`:194`), making it **unreachable**. So a 0-token step on the generation runner skips (a) the `external_launcher` + `data_parallel_size>1` `_dummy_run(1)` that prevents a **DP coordinate_batch_across_dp out-of-sync hang**, and (b) the `kv_connector_no_forward` path. The AR runner has only the careful block. Remove the early return / merge the two.
- **`gpu_ar_model_runner.py:2013`** (High, verify) — `sample_tokens` vocab correction does `prompt_token_ids.clamp(max=logits_vocab)`, but `logits_vocab` is an **out-of-bounds** index for a `logits_vocab`-wide tensor (valid `0..logits_vocab-1`); penalty gather would index OOB. Almost certainly should be `logits_vocab - 1`.
- **`gpu_ar_model_runner.py:883`** (`_unwrap_lists` in `_build_combined_prefix_cache_mm_payload`) — on an out-of-range index it silently returns `v[0]` (request 0's payload) instead of failing, so a per-request length mismatch **ships the first request's mm output for request `idx`**. (The sibling `_build_omni_mm_payload` at least warns on the same mismatch.)
- **`gpu_ar_model_runner.py:1065`** — `execute_model` mutates `scheduler_output`'s finished-request `custom_metadata` **in place**; `scheduler_output` can be shared with the engine-core process (the ngram block ~40 lines below `replace()`-copies to avoid exactly this). Copy before mutating.
- `gpu_ar_model_runner.py:578` (`_is_sparse_audio_marker`) — the `bool(value)` fallback **raises** "Boolean value of Tensor with more than one element is ambiguous" if the marker is ever a multi-element tensor/ndarray (it comes from `multimodal_outputs`). Guard the tensor case.
- `gpu_generation_model_runner.py:481` — `sample_tokens` `list` branch asserts `len(list)==1` but **not** `num_reqs==1`, so it always builds a length-1 `per_req_payloads` and **misaligns with `req_ids`** for batched requests (silent), unlike the tensor branch.
- `gpu_ar_model_runner.py:1351` — `execute_model` assigns `hidden_states` from the aux-unpack then **immediately overwrites** it via `extract_multimodal_outputs(model_output)`; the first assignment is dead (only `aux_hidden_states` survives). Fold the two.
- `gpu_ar_model_runner.py:1477` (`_sample`) — if a `prefer_model_sampler` model's `sample()` returns `None`, control **silently falls through to the default `self.sampler`** (wrong tokens for a custom sampler). Verify the contract; document the opt-out or raise.

## Implicit cross-phase state (fragile)
- `gpu_model_runner.py:81` / `:1497` — `_omni_num_scheduled_tokens_np` is written in `_preprocess` and read later in `_build_model_kwargs_extra` and `sample_tokens`; it's an implicit instance-attribute hand-off (the note says "we must keep this … or pass it out of `_preprocess` (too complex)"). Candidate to make a field of `ExecuteModelState` instead of a mutable attribute.
- `gpu_ar_model_runner.py:1473` — `self.kv_connector_output` is stashed on the instance at the end of `execute_model` and read/cleared in `sample_tokens`; every other hand-off rides `ExecuteModelState`. Make it an `ExecuteModelState` field.
- `gpu_ar_model_runner.py:1482`, `gpu_generation_model_runner.py:415` — `hasattr(self, "_positions_cpu")` (set in `_preprocess`) gates routed-experts D2H; a miss silently skips it with no error.
- `gpu_ar_model_runner.py:2117` — `sample_tokens` reads `_omni_num_scheduled_tokens_np` via `getattr` (same implicit attr as above); rename/miss silently recomputes.

## Base method reaching subclass-only attributes (hasattr smell)
- `gpu_model_runner.py:597` — inside base `_update_states`, `_downstream_payload_cache` "only appears on the AR model runner"; it's popped via `hasattr` guard. Same pattern for `_talker_mtp_generators` and the `cleanup_finished_request` gating. The base cleaning up attributes that live only on subclasses is a layering smell → own these per-request caches in one place (`OmniModelState.remove_request`).

## Wrapper-vs-raw model (`get_model()`) inconsistency
- `gpu_model_runner.py:1551` — `has_postprocess` check uses `self.model`; should use `self.get_model()`.
- `gpu_model_runner.py:480`, `:482` — M-RoPE uses `self.model.get_mrope_input_positions` / `_filter_mrope_kwargs_for_model(self.model, …)` right after `supports_mrope(self.get_model())` (isinstance must unwrap; attr access works only by wrapper delegation).

## Dead defensive guard
- `gpu_model_runner.py:1544` — `if callable(query_start_loc_cpu)` never fires: `CpuGpuBuffer.cpu` is a tensor attribute, never callable (every other site indexes it directly).
- `gpu_ar_model_runner.py:1570` — same dead guard (in `_snapshot_query_start_loc_cpu`).

## Silent failure / swallowed exception
- `gpu_model_runner.py:723` — inline prompt_embeds decode swallows all exceptions and drops the payload ("should not silently fail, should raise"); masks the native-tensor drop.
- `gpu_memory_utils.py:126` — silent NVML/device fallback returns `None`, hiding misconfiguration.
- `gpu_model_runner.py:1490`, `:1597` — `traceback.print_exc()` to stdout in `_build_model_kwargs_extra` / `_process_additional_information_updates`; use `logger.exception`.
- `gpu_ar_model_runner.py:1066` — `execute_model` bare `except Exception` logs a warning and **drops the model's custom KV-transfer metadata**; a real bug surfaces later as a corrupt/incomplete KV transfer. Narrow or re-raise.
- `gpu_ar_model_runner.py:1414` — `compute_logits` uses `except TypeError` as a signature probe (twice); a genuine `TypeError` raised *inside* `compute_logits` is swallowed and silently re-run without `sampling_metadata`.
- `gpu_generation_model_runner.py:454` — unsupported multimodal-output type is logged and the key silently omitted from the payload (downstream stage loses that output). Raise / narrow accepted types.

## Branch conflation
- `gpu_memory_utils.py:104` — local-rank→device mapping conflates "no `CUDA_VISIBLE_DEVICES`" with "`local_rank` out of range of the mask"; only the former is correct for direct indexing.

## Redundant sync
- `base.py:123` — ROCm `synchronize()` in `determine_available_memory` is likely redundant (already done in upstream `profile_run`); verify and drop.

## Redundant D2H / perf
- `utils/mm_outputs.py:76`, `:79` — redundant D2H copies when `gpu_resident_buffer_keys` keeps tensors on GPU.
- `gpu_ar_model_runner.py:1381` — `execute_model` does a **blocking** synchronous `.to("cpu")` on the default stream for the full-prefix opt-out path (qwen3-tts-talker), *inside* the async-write block whose whole purpose is to avoid per-step blocking copies — defeating it for those models.
- `gpu_ar_model_runner.py:1926` — in non-async-chunk mode `pooler_inter` and `pooler_client` are the **same list object**, so `_build_multimodal_outputs` (with the full `_ensure_tensor_values` conversion) runs **twice over identical data**. Convert once when they're the same object.
- `gpu_ar_model_runner.py:804` — `_maybe_get_combined_prefix_cache_tensors` calls `_model_needs_full_prefix_hidden_states()` twice on the critical path; bind to a local once.

## Async / threading correctness
(Refactor design for this whole cluster: `async_omni_output_refactor_design.md` — PR #4476 machinery.)
- `gpu_ar_model_runner.py:176` (`torch.Event()` vs `torch.cuda.Event()`), `:193` (fragile truthiness), `:197-202`/`_build_output_in_background` (event recorded but the background builder never waits on it), `:207` (dead assignment), `:211` daemon-thread-in-`__init__` — the `OmniAsyncGPUModelRunnerOutput` cluster (already marked inline).
- `gpu_ar_model_runner.py:2220` — `set_async_sampled_token_ids` writes `sampled_token_ids_cpu` + `async_copy_ready_event` onto `input_batch`; the *next* step's `_build_model_sampler_output_token_ids` (`:404`) reads + syncs them. Invisible, untested cross-step producer↔consumer coupling — move the two together.
- `gpu_ar_model_runner.py:1912` — **cross-thread live-state access**: in async-omni-output mode the deferred builder runs on the background thread, but its `accumulate_full_payload_output` block reads live `self.requests` **and writes connector accumulation state** while the main thread's next step mutates both. Likely unreachable in async mode (accumulate is full-payload; the async gate requires `async_chunk`) — but nothing proves or asserts it. Resolve at snapshot time, or assert unreachable.
- `gpu_ar_model_runner.py:2156` — **implicit snapshot boundary**: the `output_builder` closure captures ~14 locals; "everything the background thread reads must be snapshotted" is a convention encoded in ad-hoc copies (`:2117-2135`), not a type — a missed copy compiles fine and races silently. Replace with an explicit frozen `OmniStepSnapshot` + `capture()`.
- `gpu_ar_model_runner.py:1760` — **dual-mode builder with an implicit contract**: `_build_omni_model_runner_output_from_snapshot` runs either inline (sync) or on the background thread (async); mode differences (`postprocess_already_applied`, pre-copied payloads, prefix-cache branches being sync-only) are scattered `if`s with nothing enforcing "async reads only snapshotted args". One builder + typed snapshot + `assert async ⇒ no prefix cache`.

## Stale-memoization hazard
- `gpu_ar_model_runner.py:493` — `_request_needs_downstream_stage_payload` caches its result forever keyed by `req_id`, but the value derives from `model_intermediate_buffer`, which may be unpopulated on the first call (`final_stage_id` `None` → caches `True` permanently, never refreshed when the marker later arrives).

## Fork-fragility (rebase hazard)
- `gpu_ar_model_runner.py:286` / `gpu_generation_model_runner.py:41`,`:392` — `ExecuteModelState` is a `typing.NamedTuple` **copied** from upstream (can't be extended by inheritance) with omni fields inserted mid-tuple (`hidden_states_cpu`, `multimodal_outputs`) and undocumented drift (`logits` widened to `|None`, `slot_mappings` default, `Any` types). The generation runner imports it from the AR runner (backwards coupling) and builds it with **6 positional `None` placeholders** (`:392`) — a field add/reorder silently misassigns slots. Fix: move to a neutral module, use keyword construction, add a field-parity test, mark the OMNI deltas.
- `gpu_ar_model_runner.py:634` — `shutdown` imports `vllm.compilation.breakable_cudagraph`, an upstream-internal module that may be renamed/removed (an ImportError would break teardown).
- `gpu_ar_model_runner.py:1594` — `_snapshot_scheduler_output_for_async_omni_output` uses `dataclasses.replace` guarded by `except TypeError`; if `SchedulerOutput` stops being a dataclass on rebase this silently returns the un-snapshotted original (aliasing live dicts).
- `gpu_ar_model_runner.py:469` — `_sampling_metadata_for_model_sampler` `dataclasses.replace(SamplingMetadata, …)` assumes it stays a dataclass (already marked inline).
- `gpu_ar_model_runner.py:650` — `_capture_talker_mtp_graphs` does a fragile 5-tuple positional unpack of upstream `_determine_batch_execution_and_padding`; an arity/order change silently misassigns `batch_desc`.
- `gpu_ar_model_runner.py:373` — the `_make_buffer` pin-memory workaround (a context manager flips `self.pin_memory` to dodge a Ray low-`ulimit -l` alloc failure) assumes base `_make_buffer` chooses pinning from `self.pin_memory`; a rebase changing how pinning is selected (e.g. a per-call param) silently no-ops and the Ray failure returns. Mark `# OMNI:`.

## Condition style
- `gpu_model_runner.py:366` — truthiness check should be explicit `is not None` (format consistency).

---

## Model-specific: HunyuanImage3 output-token-ids
- `gpu_model_runner.py:241` — `_maybe_enable_output_token_ids_for_model_sampler` ("Only required by HunyuanImage3"); flips `input_batch.logitsprocs_need_output_token_ids` for one model. Move to a model-declared capability.

## Model-specific: custom model sampler (prefer_model_sampler) history
- `gpu_model_runner.py:345`, `:406` (+ AR dups `gpu_ar:330`, `:400`) — sampler-history reconstruction is "required by cosyvoice3, higgs_audio3, hunyuanimage3, glm_tts" (`:405`). Only `prefer_model_sampler` models need it → move to `OmniModelState` sampler hook (also the duplicate-merge target above).

## Model-specific: talker-MTP baked into the generic runner
- `gpu_model_runner.py:246` — `_init_talker_mtp` (fish_speech / qwen3_tts / qwen3_omni); split detect/graph/hidden/buffer, move to a `TalkerMTP` component on `OmniModelState` (`:245`, `:254`).
- `gpu_model_runner.py:600` — talker-MTP per-request cleanup inside `_update_states`.
- `gpu_model_runner.py:1266` — talker-MTP CUDA-graph recording inside `_dummy_run`.
- `gpu_model_runner.py:1991` — `qwen3_tts_request_seed` extraction inside `_talker_mtp_forward`.
- `gpu_ar_model_runner.py:650` — `_capture_talker_mtp_graphs` (AR-side capture).

## Model-specific: Fish-KV attention baked into the generic runner
- `gpu_model_runner.py:301` — `_prewarm_attention_capture_workspaces` ("Only required by Fish-KV backend").
- `gpu_model_runner.py:316` — `_maybe_attach_attention_metadata_extensions` ("if no other extension is needed, collapse into the fish-kv backend").
- `gpu_model_runner.py:1185` — Fish-KV-only branch inside `_dummy_run` ("only for fish kv").

## Model-specific: GLM-Image M-RoPE decode fixup
- `gpu_model_runner.py:514`, `:517` — `_calc_mrope_positions` / `_fixup_precomputed_mrope_decode_positions` ("ONLY required by GLM-Image, should be split to the model state"); only `glm_image_ar` declares `precomputed_mrope_decode`.

## Model-specific: Higgs `omni_query_start_loc`
- `gpu_model_runner.py:224` (load-time flag) / `:1508` (injection in `_build_model_kwargs_extra`) — only `higgs_audio_v3_talker` declares `supports_omni_query_start_loc`.

## Model-specific: MammothModa2 `generated_len` + dummy runtime info
- `gpu_model_runner.py:1432` — `generated_len` injected in `_gather_runtime_additional_information` for every request but only MammothModa2's image-grid EOL constraint uses it; generalize or remove.
- `gpu_generation_model_runner.py:877` — `get_dummy_runtime_additional_information` probe baked into `_dummy_run` (verified: sole def in `diffusion/models/mammoth_moda2/…`, consumed only there). Move behind a model-declared dummy-info hook.

## Model-specific: MiMoAudio class-name check
- `gpu_model_runner.py:1651` — `_maybe_attach_mimo_audio_req_infos` gates on `self.model.__class__.__name__ == "MiMoAudioForConditionalGeneration"`.

## Model-specific: `has_preprocess` capture/runtime buffer coupling
- `gpu_model_runner.py:1219` — the `has_preprocess` inputs_embeds load in `_dummy_run` must stay in lock-step with the `_preprocess` buffer path (capture==runtime); belongs in one hook.

## Model-specific: connector arch allowlist hardcoded + divergent
- `gpu_ar_model_runner.py:330` ↔ `gpu_generation_model_runner.py:68` — `_OMNI_CONNECTOR_INIT_ARCHS` inlined in both `__init__`s and **verified divergent**: entry 9 is `IndexTTS2TalkerForConditionalGeneration` (AR) vs `IndexTTS2S2MelDecoder` (generation), and AR passes `kv_transfer_manager=` while generation omits it (has no `OmniKVTransferManager`). Two hand-maintained sets drift silently; also two-places-coupled with `omni_scheduling_coordinator._FULL_PAYLOAD_INPUT_STAGES` (silent Stage-1 hang). Single-source (connector-owner handoff).

## Model-specific: sparse-audio marker routing
- `gpu_ar_model_runner.py:526`/`552`/`570` — `_sparse_mm_req_ids` / `_resolve_sparse_mm_routing` / `_is_sparse_audio_marker`: the `"meta.sparse_audio"` / `"meta.req_id"` marker protocol + string parsing is only meaningful for sparse-audio TTS models.
- `gpu_ar_model_runner.py:928` — hardcoded `{"meta.req_id","meta.sparse_audio"}` keys skipped inside the generic `_build_omni_mm_payload`.

## Model-specific: prefix-cache-per-model capability
- `gpu_ar_model_runner.py:713` — `_model_needs_full_prefix_hidden_states` probes `requires_full_prefix_cached_hidden_states` (only `higgs_audio_v3_talker` / `qwen3_tts_talker` set it False — verified).
- `gpu_ar_model_runner.py:730` — `_deferred_prefix_cache_mm_keys` probes `deferred_prefix_cache_mm_keys={"codes.audio"}` (same two models — verified).

## Model-specific: async-omni-output cluster (qwen3_omni)
- `gpu_ar_model_runner.py:1634`/`1675`/`1716` — `_should_use_async_omni_output` + `_snapshot_omni_output_tensors_for_async_output` + `_maybe_run_eager_omni_postprocess_before_async_output` + copy-stream gate on qwen3_omni-only flags (`use_async_omni_output` / `eager_omni_postprocess_before_async_output` / `async_chunk` / `omni_pooler_payload_include_hidden` — verified).

## Model-specific: `engine_output_type == "audio"` string
- `gpu_ar_model_runner.py:508` — `_resolve_pooler_payload_req_ids` branches on a hardcoded `"audio"` engine_output_type string (single-stage AR TTS override, e.g. VoxCPM2).

## Model-specific: scattered `getattr(self.model, …)` capability probes
- `gpu_model_runner.py:1649` (generic marker) — ~30 `getattr/hasattr(self.model, …)` sites across the three runners; consolidate into a declared capability object on `OmniModelState`. Verified AR sites: `_clear_warmup_state` (`:1019`, bagel), `get_kv_transfer_metadata` (`:1038`, bagel), `on_requests_finished` (`:1123`, moss_tts_nano / voxcpm2_talker), `prepare_runner_inputs` (`:1277`, bagel), `flush_pending_metadata` (`:1333`, bagel).

---

## Naming inconsistency (`additional_information`)
- `gpu_model_runner.py:657`, `:1375`, `:1394`, `:1408`, `:1414`, `:1526`, `:1599`, `:1627` — "additional_information" is too generic and overlaps `model_intermediate_buffer` / `runtime_additional_information` (3 names, 1 concept). `:657`: "if it's just an input naming it should be more meaningful." Settle on one; retire aliases.
- `gpu_ar_model_runner.py:475` — `_request_final_stage_id` reads the same concept from two names (`model_intermediate_buffer` vs `req_state.additional_information_cpu`); the `getattr` on the upstream `RequestState` internal is also fork-fragile.
- `gpu_ar_model_runner.py:2234` — `_resolve_global_request_id` reaches into `model_intermediate_buffer` by a `"global_request_id"` magic key (the buffer doubling as a generic side-channel).

## Missing / incomplete docstrings (systemic)
Nearly every omni-added method across `worker/` lacks a complete docstring — **~120 flagged sites**,
by file: `gpu_ar_model_runner.py` 49, `gpu_model_runner.py` ~14 ("enrich the doc string"),
`gpu_generation_model_runner.py` 8, `base.py` 7, `gpu_memory_utils.py` 4, `payload_span.py` 3,
`mixins.py` 1, `memory_utils.py` 1. Treat as one mechanical sweep (add purpose / args / returns /
how-it-works), not per-line items. Distinct from the three *wrong* (drifted) docstrings, which are bugs:
- `gpu_model_runner.py:1699` — `_preprocess` "Align with v0.14.0" is outdated.
- `gpu_generation_model_runner.py:53` — class docstring claims outputs return "via `pooler_output`", but `sample_tokens` sets `pooler_output=None` (`:484`) and ships tensors via `multimodal_outputs`/`inter_stage_outputs`.
- `gpu_generation_model_runner.py:517` — `_run_generation_model` docstring's `Args:` lists `scheduler_output`, not a parameter of the method.

## Wrong type annotation
- `gpu_model_runner.py:1539` — `query_start_loc_cpu: object` should be `torch.Tensor` ("why object?").
- `gpu_model_runner.py:958` — `extract_multimodal_outputs` annotated `-> dict` but returns a `(hidden, mm)` tuple.
- `gpu_model_runner.py:1602` — `_collect_additional_information_for_prefill` annotated `-> dict[str, dict]` but returns `None` (also stale name/docstring — see prompt_embeds subsystem).

## Import inside function (should hoist)
- `mixins.py:11`; `gpu_model_runner.py:132`, `:527`, `:1487`; `gpu_ar_model_runner.py:364` (`ray_utils.utils` — import-safe, it guards `import ray` internally, so **not** a keep-lazy dep) — hoist to module top. **Exception:** `gpu_model_runner.py:305`, `:327` are deliberate lazy `fish_kvcache_backend` imports — keep lazy.

## Unclear signature (`*args`/`**kwargs`)
- `base.py:40`, `:57`, `:183`; `gpu_model_runner.py:77` (`__init__`), `:205` (`load_model`, note at `:203`) — make signatures explicit.
- `gpu_ar_model_runner.py:160` — `OmniAsyncGPUModelRunnerOutput.__init__` takes `**kwargs` then `kwargs.pop(...)` for each real arg (`sampled_token_ids`, `logprobs_tensors`, … each marked "list this in the parameters directly"); list them as explicit params, after which the leftover-kwargs guard (`:167`, "should be removed in the refactor") becomes unnecessary.

## Over-long / should-split function
- `gpu_model_runner.py:564` — `_update_states` (`:563` "too long and complex"); `:591` — its finished-request cleanup block should be its own function.
- `gpu_model_runner.py:989` — `_dummy_run` (`:987` ~90% same as vllm, split).
- `gpu_model_runner.py:1529` — `_process_additional_information_updates` (`:1528` "need to break down to smaller functions").
- `gpu_model_runner.py:345` — `_build_model_sampler_output_token_ids` split build/resolve (`:344`).
- `gpu_ar_model_runner.py:1008` — `execute_model` (~430 L; also the divergent-duplicate above).
- `gpu_ar_model_runner.py:1961` — `sample_tokens` (~240 L: sample/draft/bookkeep + async-omni snapshot cluster + output-builder closure all inline).

## Questionable indirection / inner helper
- `gpu_model_runner.py:1677` — `_maybe_run_batch_preprocess` ("really need to be wrapped to a function?") is a thin wrapper over a `getattr` call.
- `gpu_model_runner.py:1845` — the `flush_decode_batch` closure inside `_preprocess` ("no inner helper define allowed") should be a method.
- `gpu_ar_model_runner.py:2029` — the inner closure `propose_draft_token_ids` **shadows the method `self.propose_draft_token_ids`** it wraps (same name) — confusing/error-prone; rename the local and/or lift to a method.

## Misplaced helper — free functions belong in `utils`
- `gpu_model_runner.py:51` — `_filter_mrope_kwargs_for_model` should move to `utils`; reconsider `**kwargs`.
- `gpu_ar_model_runner.py:53`, `:61`, `:84`, `:124`, `:254` — the module-level payload helpers `_to_cpu_contiguous`, `_clone_cuda_tensor_payload`, `_copy_tensor_payload_to_cpu`, `_snapshot_tensor_payload_to_cpu_async`, `_ensure_tensor_values` live in the runner file ("utils should be moved to the utils.py file").

## Inline dataclasses belong in a data module
- `gpu_ar_model_runner.py:100` (`_AsyncCPUPayloadSnapshot`), `:144` (`_OmniOutputTensorSnapshot` NamedTuple), `:151` (`OmniAsyncGPUModelRunnerOutput`), plus `ExecuteModelState` (NamedTuple) — output/state types defined inline in the runner module ("should create a separate data file for the dataclass"). Moving them to a neutral `worker/` data module also fixes the generation runner's backwards import of `ExecuteModelState` / `_ensure_tensor_values` from the AR runner (see Fork-fragility).

## Fragile hack
- `gpu_model_runner.py:1584` — `**postprocess_kwargs` unpack + `hidden_states`-key exclusion to dodge a positional clash; pass `req_infos` as a single payload arg (existing TODO).

## Missing tests (untested modules)
- `base.py`, `memory_utils.py`, `payload_span.py` — no direct unit tests; add L1 CPU characterization tests before any refactor that moves them.

---

## Suggested order
0. **Correctness first:** the `_dummy_run` capture≠replay gap (`gpu_generation_model_runner.py:857`,
   silent wrong output for a `has_preprocess` generation model) and the silent KV-transfer-metadata drop
   (`gpu_ar_model_runner.py:1066`) are real bugs — fix + test before the cleanup passes.
1. **Now (low risk, CPU-testable):** dead/stale, deprecated, dead guard, silent-failure, wrapper-vs-raw,
   naming/docs/type-hints, non-lazy import hoists, inner-helper extraction. Guard with `tests/worker/`;
   add the missing `base`/`memory_utils`/`payload_span` characterization tests first.
2. **Investigate (may unlock large removals):**
   (a) the cross-stage prompt_embeds subsystem — confirm upstream `EngineCoreRequest` carries
   prompt_embeds; if so delete the omni decode/overlay path;
   (b) process-level memory estimation — check orchestrator parallel init, engine-init cost, and
   rapid-init feasibility; if unneeded, drop the NVML per-process machinery back to profiling.
3. **`additional_information` subsystem:** rename to one name + remove dead members + extract the rest
   into a Mixin (seeds the B-align move into `OmniModelState` / `OmniIntermediateBuffer`).
4. **Structural (MR-V2 / B-align):** the divergent duplicates and every **Model-specific** type above →
   evict into `worker_v2/OmniModelState`, reusing its per-model units rather than re-solving on v1.
