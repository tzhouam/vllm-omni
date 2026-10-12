# Initial AI Hub 0.56 component transport check

Inference job [`jp3z31zx5`](https://workbench.aihub.qualcomm.com/jobs/jp3z31zx5/)
ran one historical S24 Spark attention layer with real saved input and SDK
`profile=False`. The remote job confirmed `profiling=False`, yet its raw
response contained **100** timing samples. See the adjacent
[single-iteration follow-up](../aihub_runner_sdk056_s24_single_iteration_smoke/README.md)
for job `jglyz2we5`, which used `--max_profiler_iterations 1` and returned
one sample. Both tests are component-only.

`run_initial_before_runtime_correction.json` preserves the first journal.
The live `run.json` records two explicit metadata corrections: the artifact
was a QNN DLC rather than a context binary, and the service's raw profile
response was archived after collection. The job ID and input/output NPZ
files remained unchanged.
