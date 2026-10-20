# Omni Windows branch migration to 0.32

Status (2026-10-20): source migration and scoped regression complete;
**native Windows 0.32 wheel/build and real-model qualification remain pending**.

## Branch identity and preservation

- New branch: `codex/omni-windows-0.32.0`, based on official `v0.32.0rc1`.
  This is an Omni prerelease, not a final `v0.32.0` release.
- Original branch retained: `edge-infer/omni-0.29.0`, tip `00510ce38`.
- All 70 historical local commits are represented by their final cumulative tree
  delta against `v0.29.0rc1`. This avoids replaying an old upstream merge and
  restoring obsolete intermediate implementations. The original commit history
  remains on the old branch; the new branch has a cumulative migration commit
  followed by compatibility corrections.
- Paired vLLM branch: [codex/windows-0.32.0](https://github.com/tzhouam/vllm/tree/codex/windows-0.32.0),
  based on stable `v0.32.0`, source checks at `ddd8a8fbf`.
- The separate current Strata/Agent development checkout `vllm-omni-edge` and
  installed Windows `omni029venv` are unchanged. This branch migrates the complete
  historical Windows/edge/TTS branch, not a deployment of the newer Agent checkout.

## Adaptations

Startup/profile selection and initialization timelines now use OmniEngineBase
and StageRuntime. Upstream stage launch, readiness, shutdown and CUDA graph
memory accounting remain authoritative. Edge CPU admission, Windows compatibility,
external-stage contracts, Spark and talker-only deployment remain present.
Diffusion imports stay lazy for registered text/TTS routes.

The local CPU incremental predictor has a separate cached-forward entry point
alongside upstream tensor-indexed caching. It honors supplied per-request uniforms
and greedy sampling, and parity tests cover repeat requests with intervening state.
The codec frame generator is retained alongside upstream speech cleanup. Streaming
cancellation state is request-local; commitment cleanup also owns codec requests.
Talker timing uses StepStats around the new post-sampling entry point.

Upstream dependencies and registry additions are retained. The CPU runner test
now checks `_postprocess_tensors` (the 0.32 API); the deployment test checks the
existing explicit 640 MiB talker KV budget rather than the obsolete 4 GiB value.
An AST audit found no missing locally added functions except the two startup
helpers deliberately moved/replaced by OmniEngineBase initialization.

## Validation

See [results.json](results.json), [static.json](static.json), and the adjacent logs.
Run from the repository root with the isolated 0.32 environment:

```sh
.venv/bin/python benchmarks/migration_032/run_validation.py
```

| Group | Result | Boundary |
| --- | --- | --- |
| Edge, Windows compatibility, config, CPU runner and initialization | 345 passed, 2 skipped | WSL tests; Windows-specific behavior is mocked where applicable |
| CPU incremental predictor | 7 passed | Tiny numerical fixtures, including supplied uniforms and request-state isolation |
| Speech/codec and CUDA predictor | 122 passed | WSL CUDA, official Linux vLLM 0.32 wheel |
| Python syntax and distributed re-exports | Passed | Static syntax and installed-wheel symbol checks |

Total: **474 passed, 2 skipped**, in separate processes. These are correctness
checks, not batch-one performance qualification. Exact Python, Torch,
Transformers and vLLM versions and commands are recorded in results.json.

During migration, tests caught a lost codec generator, eager diffusion import,
and obsolete CPU API/budget assertions; these were corrected. Initial runs also
lacked pytest-mock/onnxruntime/onnxscript; the final environment includes them.
A combined CPU-mocking/CUDA-graph run failed with a default-stream capture error;
the unchanged CUDA tests pass in their own process. The final runner deliberately
isolates these groups. This mixed-process test interaction remains a limitation.

The vLLM full pre-commit run could not initialize markdownlint because nodejs.org
archive downloads repeatedly ended with IncompleteRead. Relevant source hooks
passed independently; the complete hook suite is not claimed passed.

## Next qualification

1. Build the paired vLLM source with native Windows MSVC/CUDA in a new environment.
2. Install this Omni source there, check actual loaded paths and Windows patch report.
3. Run kernel/control-plane checks and Spark/TTS complete requests, cancellation,
   resource release and parity with the same checkpoint/precision/input.
4. Only then migrate the deployed Agent environment and measure batch=1,
   concurrency=1 latency/memory. Preserve historical evidence under its original
   runtime identity; this migration does not renew its performance qualification.