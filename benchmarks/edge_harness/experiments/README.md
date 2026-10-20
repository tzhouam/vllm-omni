# One-off experiment drivers

Scripts that produced a specific result rather than tooling meant to be reused.
Kept because the results in `../results/` are only checkable if the script that
made them is here.

| group | what |
|---|---|
| `aihub_*.py` | Qualcomm AI Hub on-device runs — Snapdragon layer timings, calibration, w8a16/w4a16 parity. Authentication comes from `~/.qai_hub/client.ini`; no credential is in these files |
| `bench_int4_kernels.py` | the 4-bit GEMM microbenchmark — kernel-level shapes, used to size decisions before building anything |
| `quality_int4.py`, `ppl_vllm.py`, `parity_hf_vs_vllm.py`, `ref_hf.py` | fidelity: greedy divergence, perplexity, and parity against a HuggingFace reference |
| `quantize_int8.py`, `export_gptq_cpu.py`, `dequantize_int4.py`, `verify_export.py` | building and round-tripping checkpoints |
| `capture_real_acts.py`, `capture_lm_head.py` | activation capture for calibrated quantization |
| `cpu_busy.py` | a load generator, used to test that the harness's contention guard actually refuses |

**They are not portable as written.** Most carry absolute paths from the machine
they ran on (`/data/<user>/embedding_infer/...`) and expect checkpoints under
`models/`. They are a record of what was run, not a supported interface — the
supported entry points are the tools one level up, described in
[docs/edge/harness.md](../../../docs/edge/harness.md).
