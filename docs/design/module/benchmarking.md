---
title: Benchmarking
kind: module
status: draft
owners:
  - "@alex-jw-brooks"
  - "@Bounty-hunter"
primary_code_paths:
  - vllm_omni/benchmarks/**
  - vllm_omni/entrypoints/cli/benchmark/**
related_code_paths:
  - benchmarks/**
depends_on:
  - entrypoints.md
  - profiling.md
validation_paths:
  - tests/benchmarks/**
  - tests/dfx/perf/**
upstream_refs:
  - vllm.benchmarks
last_reviewed: 2026-07-16
---

# Benchmarking

Benchmarking defines reproducible workloads, metrics, warm-up, measurement,
and result metadata for comparing vLLM-Omni performance.

## Candidate invariants

### BENCH-INV-001: Results are reproducible

**Rule:** Results MUST identify the model, revision, commit, hardware,
configuration, workload, concurrency, warm-up, and measurement window.

### BENCH-INV-002: Correctness precedes performance

**Rule:** A benchmark MUST validate successful and semantically acceptable
outputs before using timings as performance evidence.

### BENCH-INV-003: Metric definitions are explicit

**Rule:** Metrics MUST state their unit, population, and aggregation.

## Safe-change guide

Test workload parsing and metric calculations with deterministic fixtures. Do
not compare materially different workloads or runtime settings.

The [LingBot World regression suite](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/lingbot_world/README.md)
wraps the served realtime workload in blocking pytest jobs. It measures a frozen
reference on the same hardware/dependencies, calibrates from three independent
server starts, and checks candidate performance separately. Fixed-input VAE
replay and instrumented lifecycle/memory checks run outside timing windows.
Calibration reuse requires recorded optimization settings: its nonempty
`engine_environment` string mapping must match the measured contract. Missing
settings are invalid even when both records omit the field.
Calibration compatibility errors retain raw measurements, numerical results and
quality scores for review; candidate comparisons stay unjudged and block acceptance.
NPU weekly collection separates each case shard's three independent
reference/candidate pairs into bounded jobs. A mandatory CPU gate verifies the
complete matching population per shard before quality evaluation; missing or
cross-shard pairs fail that gate. Partitioning jobs must preserve all cases,
sessions, chunks, starts, hardware and regression budgets.
GPU evaluators consume checksum-verified CUDA or NPU videos and enforce native
per-dimension quality limits. Incomplete populations, unstable references,
missing prerequisites and skipped selected CI tests cannot establish acceptance.
