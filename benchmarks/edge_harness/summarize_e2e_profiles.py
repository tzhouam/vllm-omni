# SPDX-License-Identifier: Apache-2.0
"""Combine the archived 2026-09-22 matrix with measured local profiling runs.

A completed benchmark is not automatically a quality-qualified model release.
Missing runs and insufficient samples remain explicit rather than becoming zeroes.
For current coverage use the rolling 60-cell ledger, not this historical matrix.
"""

import argparse
import hashlib
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from single_request_protocol import (
    LENGTH_BANDS,
    MIN_MEASURED_PER_BAND,
    MIN_SUSTAINED_SECONDS,
    PROTOCOL,
)

RUNS = {
    ("pc_cpu_wsl", "spark"): "spark-cpu-wsl",
    ("pc_cuda_wsl", "spark"): "spark-cuda-wsl",
    ("pc_cuda_windows", "spark"): "spark-cuda-windows",
    ("pc_cuda_wsl", "tts"): "tts-cuda-wsl",
    ("pc_cuda_windows", "tts"): "tts-cuda-windows",
}


def blocker_codes(case):
    device, model = case["device"], case["model"]
    if model in ("minicpm", "vla") and device in ("pc_cpu_wsl", "pc_cpu_windows", "pc_cuda_wsl", "pc_cuda_windows"):
        return ["COMPLETE_MATCHING_CHECKPOINT_MISSING"]
    if device == "pc_joint":
        return ["JOINT_PLAN_NOT_QUALIFIED", "NO_E2E_CORESIDENCY_OR_OVERLAP_EVIDENCE"]
    if device.startswith(("mobile_", "embedded_")) or device == "pc_xelite":
        return [
            "NO_DEVICE_LOCAL_APPLICATION_ACCESS",
            "STATEFUL_LOCAL_PIPELINE_MISSING",
            "COMPLETE_TARGET_ARTIFACT_NOT_QUALIFIED",
        ]
    if device == "pc_cpu_windows":
        return ["CPU_ATTENTION_OPERATORS_ABSENT", "NATIVE_CPU_MODEL_BACKEND_NOT_QUALIFIED"]
    if device == "pc_cpu_wsl" and model == "tts":
        return ["VLLM_OMNI_VERSION_INCOMPATIBILITY", "GENERATION_FAILED_BEFORE_AUDIO"]
    if device.startswith("pc_cuda") and model == "qwen27":
        return ["OMNI_PIPELINE_NOT_REGISTERED"]
    if device == "pc_npu" and model == "qwen27":
        return ["TESTED_VISION_ARTIFACT_REJECTED_BY_NPU_PARTITIONER", "COMPLETE_PIPELINE_MISSING"]
    if device in ("pc_igpu", "pc_npu"):
        return ["COMPLETE_TARGET_ARTIFACT_NOT_QUALIFIED", "STATEFUL_MODEL_BACKEND_MISSING"]
    if model in ("minicpm", "vla"):
        return ["COMPLETE_MATCHING_CHECKPOINT_MISSING"]
    return ["COMPATIBLE_ARTIFACT_AND_BACKEND_NOT_AVAILABLE"]


def stats(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None}
    return {
        "n": len(values),
        "p50": values[max(0, math.ceil(len(values) * 0.50) - 1)],
        "p95": values[max(0, math.ceil(len(values) * 0.95) - 1)],
        "mean": sum(values) / len(values),
        "min": values[0],
        "max": values[-1],
    }


def read_lines(path):
    if not path.is_file():
        return []
    rows = []
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A writer may still be flushing its final line. Never count it as a sample.
            if index != len(lines) - 1 or raw.endswith("\n"):
                raise
    return rows


def execution_report(path):
    """A killed process can leave a running snapshot; the host exit is authoritative."""
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    host_path = path.with_name(path.name + "-host") / "status.json"
    if report.get("status") in ("running", "pending") and host_path.is_file():
        host = json.loads(host_path.read_text(encoding="utf-8"))
        if host.get("status") == "exited":
            report = {
                **report,
                "status": "failed",
                "error": report.get("error")
                or (
                    f"Process exited with code {host.get('returncode')} "
                    "while the last benchmark snapshot was nonterminal."
                ),
            }
    return report


def diagnostic(path, kind):
    """Expose executed checks without promoting a completed probe into a pass."""
    report_path = path / "report.json"
    if not report_path.is_file():
        return missing_report(path)
    report = execution_report(path)
    keys = (
        (
            "median_wall_s",
            "traced_over_untraced_before",
            "traced_over_untraced_after",
            "start_rpc_s",
            "stop_rpc_s",
            "traces",
            "overhead_caveat",
            "requests",
            "profiler_config",
            "stage_trace_prefixes",
        )
        if kind == "trace"
        else (
            "saturation_reached",
            "same_tokens_after_saturation",
            "stale_handle_rejected",
            "cancel",
            "abort",
            "crash",
            "crash_injection",
            "oversized_plan_refused",
            "cleanup_error",
            "owned_processes_requiring_harness_cleanup",
        )
    )
    return {
        "status": report["status"],
        "error": report.get("error"),
        "source": str(report_path),
        "qualification": "Inspect individual outcomes; completion means the probe ran, not that all gates passed.",
        **{key: report[key] for key in keys if key in report},
    }


def missing_report(path):
    host = path.with_name(path.name + "-host")
    status_path = host / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    if status.get("status") == "exited":
        return {
            "status": "failed",
            "source": str(status_path),
            "error": f"Process exited with code {status.get('returncode')} before producing a benchmark report.",
            "log": str(host / "command.log"),
            "failure_scope": "Inspect the launcher log before attributing failure to the model or device.",
        }
    return {
        "status": "pending",
        "source": str(path / "report.json"),
        "reason": "Serialized benchmark has not produced its initial report.",
    }


def output_checks(rows):
    """Check recorded invariants only; no reference-quality inference."""
    findings, ids = [], set()
    for row in rows:
        rid = row.get("request_id")
        errors = []
        if rid in ids:
            errors.append("duplicate_request_record")
        ids.add(rid)
        if not row.get("finished") or row.get("error"):
            errors.append("incomplete_or_error")
        if "arrivals" in row:
            events = row["arrivals"]
            seq = [e["sequence"] for e in events]
            if any(b <= a for a, b in zip(seq, seq[1:])):
                errors.append("nonincreasing_sequence")
            if len({e["epoch"] for e in events}) != 1:
                errors.append("missing_or_mixed_epochs")
            if sum(e["kind"] == "done" for e in events) != 1:
                errors.append("missing_or_duplicate_completion")
            if row.get("output_tokens") != 128 or len(row.get("output_token_ids", [])) != 128:
                errors.append("wrong_output_length")
            stream = row.get("stream", {})
            for unit in ("chunks", "bytes"):
                if stream.get("high_water_" + unit, 0) > stream.get("max_" + unit, 0):
                    errors.append("queue_bound_exceeded_" + unit)
        elif "chunks_all" in row:
            chunks = row["chunks_all"]
            if not chunks or sum(c.get("samples", 0) for c in chunks) <= 0:
                errors.append("no_audio")
            if any(not c.get("finite") for c in chunks):
                errors.append("nonfinite_audio")
            if [c["idx"] for c in chunks] != list(range(len(chunks))):
                errors.append("nonsequential_audio_chunks")
            if sum(bool(c.get("terminal")) for c in chunks) != 1 or not chunks[-1].get("terminal"):
                errors.append("missing_duplicate_or_nonfinal_terminal")
            if any(c.get("request_id", rid) != rid for c in chunks):
                errors.append("wrong_request_id")
            if any(c.get("sr", row["sr"]) != row["sr"] for c in chunks):
                errors.append("sample_rate_changed")
        else:
            errors.append("no_recorded_output_contract")
        if errors:
            findings.append({"request_id": rid, "errors": errors})
    return {
        "requests_checked": len(rows),
        "findings": findings,
        "recorded_invariants_pass": bool(rows) and not findings,
        "scope": "Recorded stream metadata and finite audio only; not reference, perceptual or task quality.",
    }


def delivery_spacing(samples):
    """Pool adjacent deliveries within each request, never across requests."""
    updates, consumer, audio = [], [], []
    update_counts = []
    for row in samples:
        times = row.get("token_unix", [])
        updates.extend((b - a) * 1000 for a, b in zip(times, times[1:]))
        if times:
            update_counts.append({"updates": len(times), "output_tokens": row.get("output_tokens")})
        times = [e["elapsed_s"] for e in row.get("arrivals", []) if e["kind"] == "token"]
        consumer.extend((b - a) * 1000 for a, b in zip(times, times[1:]))
        times = [c["t_ms"] for c in row.get("chunks_all", []) if c.get("samples", 0) > 0]
        audio.extend(b - a for a, b in zip(times, times[1:]))
    return {
        "scope": "Pooled adjacent delivery intervals within measured requests; n counts intervals, not requests. "
        "Text timestamps describe output updates, which can coalesce tokens; not isolated per-token kernel latency.",
        "text_output_update_gap_ms": stats(updates),
        "text_consumer_event_gap_ms": stats(consumer),
        "audio_nonempty_chunk_gap_ms": stats(audio),
        "text_delivery_counts": update_counts,
        "effective_prefill_tokens_per_s": stats(
            [
                row["prompt_tokens"] / row["ttft_s"]
                for row in samples
                if row.get("prompt_tokens") is not None and row.get("ttft_s", 0) > 0
            ]
        ),
        "prefill_scope": (
            "Input tokens divided by TTFT, including queue/host/first-output work; not isolated prefill compute."
        ),
    }


def profile(path):
    report_path = path / "report.json"
    if not report_path.exists():
        return missing_report(path)
    report = execution_report(path)
    rows = read_lines(path / "requests.jsonl")
    recorded_protocol = report.get("profile_protocol") or {}
    protocol_name = recorded_protocol.get("name", "historical_concurrency_sweep_v1")
    single_request = protocol_name == PROTOCOL
    protocol_violations = []
    if single_request:
        if recorded_protocol.get("request_batch_size") != 1 or recorded_protocol.get("max_active_requests") != 1:
            protocol_violations.append("report_does_not_declare_batch1_concurrency1")
        for row in rows:
            if row.get("phase") in ("warmup", "measured", "sustained") and (
                row.get("batch_size") != 1 or row.get("concurrency") != 1
            ):
                protocol_violations.append(f"non_single_request_sample:{row.get('request_id')}")
    elif protocol_name != "historical_concurrency_sweep_v1":
        protocol_violations.append(f"unknown_protocol:{protocol_name}")
    groups = defaultdict(list)
    for row in rows:
        if row["phase"] == "measured":
            groups[(row["length_band"], row["concurrency"])].append(row)
    measurements = []
    for (length, concurrency), samples in sorted(groups.items()):
        token_sequences = {tuple(s["output_token_ids"]) for s in samples if "output_token_ids" in s}
        reference_sequences = {
            tuple(s["output_token_ids"]) for s in groups.get((length, 1), []) if "output_token_ids" in s
        }
        measurements.append(
            {
                "length_band": length,
                "concurrency": concurrency,
                "n": len(samples),
                "delivery_spacing": delivery_spacing(samples),
                "distinct_token_sequences_for_fixed_prompt": len(token_sequences) if token_sequences else None,
                "requests_with_sequences_not_seen_at_concurrency_1": sum(
                    tuple(s["output_token_ids"]) not in reference_sequences for s in samples
                )
                if token_sequences and reference_sequences
                else None,
                "reproducibility_note": (
                    "Identical prompt/greedy requests produced multiple sequences; "
                    "numerical vs state cause requires investigation, not inferred from timings."
                )
                if len(token_sequences) > 1
                else None,
                "metrics": {
                    key: stats([s.get(key) for s in samples])
                    for key in (
                        "wall_s",
                        "ttft_s",
                        "decode_tok_per_s",
                        "ttfa_ms",
                        "rtf_total",
                        "stall_at_ttfa_ms",
                        "total_wall_s",
                        "playback_start_ms",
                    )
                },
                "actual_prompt_tokens": sorted(
                    {s["prompt_tokens"] for s in samples if s.get("prompt_tokens") is not None}
                ),
                "requests_with_playback_stalls": sum((s.get("stall_at_ttfa_ms") or 0) > 0 for s in samples),
            }
        )
    expected = set(itertools.product(LENGTH_BANDS, (1,) if single_request else (1, 2, 4)))
    enough = all(len(groups.get(key, [])) >= MIN_MEASURED_PER_BAND for key in expected)
    if single_request:
        unexpected = set(groups) - expected
        if unexpected:
            protocol_violations.append(f"unexpected_measured_groups:{sorted(unexpected)}")
        for length in LENGTH_BANDS:
            if not any(row.get("phase") == "warmup" and row.get("length_band") == length for row in rows):
                protocol_violations.append(f"missing_warmup:{length}")
        if not any(row.get("phase") == "sustained" for row in rows):
            protocol_violations.append("missing_sustained_requests")
    metric_gaps = []
    for key, samples in groups.items():
        for sample in samples:
            required = (
                ("total_wall_s", "ttfa_ms", "rtf_total", "stall_at_ttfa_ms")
                if "ttfa_ms" in sample
                else ("wall_s", "ttft_s", "decode_tok_per_s")
            )
            missing = [name for name in required if sample.get(name) is None or not math.isfinite(float(sample[name]))]
            if missing:
                metric_gaps.append({"configuration": key, "request_id": sample.get("request_id"), "fields": missing})
    metrics_complete = enough and not metric_gaps
    recorded_output = output_checks(rows)
    sustained = report.get("sustained_wall_s", 0) >= MIN_SUSTAINED_SECONDS
    timing_complete = report["status"] == "completed" and metrics_complete and sustained
    timing_complete = timing_complete and not protocol_violations
    if single_request:
        # A timing protocol cannot be complete when its archived requests
        # include malformed, unfinished, duplicated, or non-finite output.
        # This remains a metadata check, not a task-quality verdict.
        timing_complete = timing_complete and recorded_output["recorded_invariants_pass"]
    return {
        "status": report["status"],
        "error": report.get("error"),
        "startup_s": report.get("startup_s"),
        "report_write_permission_errors": report.get("report_write_permission_errors", 0),
        "last_report_write_error": report.get("last_report_write_error"),
        "source": str(report_path),
        "profile_protocol": protocol_name,
        "protocol_violations": protocol_violations,
        "historical_protocol_complete": (
            timing_complete if protocol_name == "historical_concurrency_sweep_v1" else False
        ),
        "batch1_protocol_complete": timing_complete if single_request else False,
        "minimum_20_per_configuration": enough,
        "sustained_30_minutes": sustained,
        "required_metrics_complete": metrics_complete,
        "metric_gaps": metric_gaps,
        "profile_protocol_complete": timing_complete if single_request else False,
        "sustained_wall_s": report.get("sustained_wall_s"),
        "quality_gate": report.get("quality_gate", "not established by this profiling run"),
        "recorded_output_checks": recorded_output,
        "measurements": measurements,
        "cancel": report.get("cancel"),
        "memory": report.get("memory") or report.get("usage_after_close", {}).get("peaks"),
        "placement": report.get("placement", "See actual stage/backend configuration and execution log"),
        "limitations": [
            "Existing disk/JIT caches; no cold-disk startup claim.",
            "Whole-GPU power/memory cannot be attributed exclusively to this model.",
            "No whole-device energy sensor; no NPU or CPU energy attribution.",
            "Detailed kernel trace and its overhead are not measured by this run.",
            "Quality and fault-injection gates remain separate from timing completion.",
        ],
    }


def build(matrix_path, run_root):
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    pairs = [(c["device"], c["model"]) for c in matrix["cases"]]
    assert len(pairs) == len(set(pairs)) == 60
    assert set(pairs) == set(itertools.product(matrix["devices"], matrix["models"]))
    for source in matrix["sources"].values():
        assert hashlib.sha256((matrix_path.parent / source["file"]).read_bytes()).hexdigest() == source["sha256"]
    result = {
        "schema": 1,
        "generated_unix": time.time(),
        "source_matrix": str(matrix_path),
        "scope": (
            "Archived 2026-09-22 60-pair matrix and five baseline runs; "
            "current rolling coverage is tracked in the separate qualification ledger."
        ),
        "percentiles": "nearest rank; warmups and sustained runs excluded from configuration percentiles",
        "devices": matrix["devices"],
        "models": matrix["models"],
        "cases": [],
    }
    for case in matrix["cases"]:
        pair = case["device"], case["model"]
        current = {**case, "prior_status": case["status"]}
        if pair in RUNS:
            current["profiling"] = profile(run_root / RUNS[pair])
            current["diagnostics"] = {
                "trace": diagnostic(run_root / (RUNS[pair] + "-trace"), "trace"),
                "reliability": diagnostic(run_root / (RUNS[pair] + "-reliability"), "reliability"),
            }
            current["status"] = "PROFILE_" + current["profiling"]["status"].upper()
            current["e2e_qualification"] = (
                "not fully qualified: reference-quality and remaining reliability/performance gates must be reviewed"
            )
        else:
            current["status"] = "NOT_E2E_PROFILEABLE"
            current["profiling"] = {
                "status": "not_run",
                "reason": case["finding"],
                "reason_codes": blocker_codes(case),
                "evidence": case["evidence"],
                "scope": "No currently qualified complete local pipeline; component data retained separately.",
            }
            current["e2e_qualification"] = (
                "unsupported in the current tested configuration; not a universal hardware/model claim"
            )
        result["cases"].append(current)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = build(a.matrix, a.runs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "matrix.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E2E check and profiling",
        "",
        "This is the archived 2026-09-22 matrix. For current 60-cell coverage, use the rolling "
        "qualification ledger. Profiling completion does not imply full release qualification.",
        "",
        "## Configuration × model",
        "",
        "NOT E2E means the current artifacts/runtime cannot run the complete requested pipeline. "
        "See the per-cell reasons below.",
        "",
        "| Device | " + " | ".join(result["models"].values()) + " |",
        "|---|" + "---|" * len(result["models"]),
    ]
    by_pair = {(c["device"], c["model"]): c for c in result["cases"]}
    for device, label in result["devices"].items():
        cells = []
        for model in result["models"]:
            case = by_pair[device, model]
            cells.append(
                "NOT E2E"
                if case["status"] == "NOT_E2E_PROFILEABLE"
                else case["status"].replace("PROFILE_", "Profile ").lower()
            )
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Per-cell reasons and next steps",
            "",
            "| Device | Model | Current profiling disposition | Reason / remaining work | Next step |",
            "|---|---|---|---|---|",
        ]
    )
    for case in result["cases"]:
        reason = case["profiling"].get("reason") or case["profiling"].get("error") or case["e2e_qualification"]
        next_step = case["next_step"].replace(chr(10), " ").replace("|", ";")
        lines.append(
            f"| {result['devices'][case['device']]} | {result['models'][case['model']]} | {case['status']} "
            f"| {reason.replace(chr(10), ' ').replace('|', ';')} | {next_step} |"
        )
    lines.extend(
        [
            "",
            "## Measured request timing",
            "",
            "Nearest-rank p50/p95; warmups and sustained requests excluded. Values are observations "
            "under recorded power/cache conditions, not latency targets. Spark CPU uses 1.7B INT8; "
            "CUDA uses 4B BF16, so these are not same-model acceleration comparisons.",
            "",
        ]
    )

    def number(value):
        return "unavailable" if value is None else f"{value:.3f}"

    for case in result["cases"]:
        if (case["device"], case["model"]) not in RUNS:
            continue
        prof = case["profiling"]
        lines.extend(
            [
                f"### {result['devices'][case['device']]} — {result['models'][case['model']]}",
                "",
                f"Status: {prof['status']}. Full timing protocol complete: "
                f"{prof.get('profile_protocol_complete', False)}. Startup: {number(prof.get('startup_s'))} s. "
                f"Sustained run: {number(prof.get('sustained_wall_s'))} s.",
                "",
                "| Length | Concurrent requests | n | Metric | p50 | p95 |",
                "|---|---:|---:|---|---:|---:|",
            ]
        )
        keys = (
            ("ttfa_ms", "rtf_total", "stall_at_ttfa_ms")
            if case["model"] == "tts"
            else ("ttft_s", "wall_s", "decode_tok_per_s")
        )
        for group in prof.get("measurements", []):
            for key in keys:
                metric = group["metrics"][key]
                lines.append(
                    f"| {group['length_band']} | {group['concurrency']} | {metric['n']} | {key} "
                    f"| {number(metric['p50'])} | {number(metric['p95'])} |"
                )
        lines.append("")
    (a.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['cases'])} pair dispositions to {a.out}")


if __name__ == "__main__":
    main()
