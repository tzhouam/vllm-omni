# SPDX-License-Identifier: Apache-2.0
"""Read-only batch-1 text timing analysis of an existing complete profile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from single_request_protocol import LENGTH_BANDS, MIN_MEASURED_PER_BAND, MIN_SUSTAINED_SECONDS, PROTOCOL
from summarize_e2e_profiles import output_checks, read_lines, stats

METRICS = ("wall_s", "ttft_s", "decode_tok_per_s")
TELEMETRY = ("gpu_temperature_c", "gpu_power_mw", "gpu_clock_sm_mhz", "gpu_clock_memory_mhz",
             "gpu_utilization_pct", "gpu_memory_used_bytes", "host_available_bytes")
MIN_SUSTAINED_ACTIVE_FRACTION = 0.95  # Allow per-request bookkeeping, not a long idle interval.


def _finite(value: Any) -> bool:
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {name: stats([row.get(name) for row in rows]) for name in METRICS}


def analyze(report: dict[str, Any], rows: list[dict[str, Any]], telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the archived single-request window; never infer reference quality."""
    violations: list[str] = []
    protocol = report.get("profile_protocol") or {}
    settings = report.get("settings") or {}
    if report.get("status") != "completed":
        violations.append("report_not_completed")
    if (protocol.get("name") != PROTOCOL or protocol.get("request_batch_size") != 1
            or protocol.get("max_active_requests") != 1):
        violations.append("wrong_or_missing_batch1_protocol")
    if settings.get("batch_size") != 1 or settings.get("concurrency") != 1:
        violations.append("settings_not_batch1_concurrency1")
    if report.get("requests") != len(rows):
        violations.append("report_request_count_mismatch")

    previous_end: float | None = None
    metric_gaps: list[dict[str, Any]] = []
    for row in rows:
        rid = row.get("request_id")
        if row.get("batch_size") != 1 or row.get("concurrency") != 1:
            violations.append(f"non_single_request_sample:{rid}")
        submitted, wall = row.get("submitted_unix"), row.get("wall_s")
        if not _finite(submitted) or not _finite(wall) or float(wall) <= 0:
            violations.append(f"invalid_request_interval:{rid}")
        else:
            # submit is recorded inside engine.submit, after the profiler's
            # monotonic start. submit + wall conservatively overstates the end.
            if previous_end is not None and float(submitted) < previous_end - 0.01:
                violations.append(f"overlapping_request_interval:{rid}")
            previous_end = float(submitted) + float(wall)
        if row.get("phase") in ("measured", "sustained"):
            missing = [name for name in METRICS if not _finite(row.get(name)) or float(row[name]) <= 0]
            if row.get("output_tokens") != 128:
                missing.append("output_tokens_128")
            if (_finite(row.get("ttft_s")) and _finite(wall)
                    and float(row["ttft_s"]) > float(wall) + 0.01):
                missing.append("ttft_exceeds_request_wall")
            first, last = row.get("first_token_unix"), row.get("last_token_unix")
            stamps = row.get("token_unix")
            if (not _finite(first) or not _finite(last) or not _finite(submitted)
                    or float(first) < float(submitted) or float(last) <= float(first)
                    or not isinstance(stamps, list) or len(stamps) < 2
                    or any(not _finite(t) for t in stamps)
                    or any(b < a for a, b in zip(stamps, stamps[1:]))
                    or abs(float(stamps[0]) - float(first)) > 1e-6
                    or abs(float(stamps[-1]) - float(last)) > 1e-6):
                missing.append("token_timestamp_contract")
            else:
                if _finite(row.get("ttft_s")) and abs(float(row["ttft_s"]) - (float(first) - float(submitted))) > 1e-6:
                    missing.append("ttft_disagrees_with_raw_timestamps")
                if (_finite(row.get("decode_tok_per_s")) and row.get("output_tokens") == 128
                        and not math.isclose(float(row["decode_tok_per_s"]), 127 / (float(last) - float(first)),
                                             rel_tol=1e-6, abs_tol=1e-6)):
                    missing.append("decode_rate_disagrees_with_raw_timestamps")
            if type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] <= 0:
                missing.append("prompt_tokens")
            if missing:
                metric_gaps.append({"request_id": rid, "fields": missing})

    measured = {band: [r for r in rows if r.get("phase") == "measured" and r.get("length_band") == band]
                for band in LENGTH_BANDS}
    for band in LENGTH_BANDS:
        warmup_indices = [i for i, r in enumerate(rows) if r.get("phase") == "warmup" and r.get("length_band") == band]
        measured_indices = [i for i, r in enumerate(rows) if r.get("phase") == "measured" and r.get("length_band") == band]
        if not warmup_indices:
            violations.append(f"missing_warmup:{band}")
        elif measured_indices and min(warmup_indices) >= min(measured_indices):
            violations.append(f"warmup_after_measurement:{band}")
        if len(measured[band]) < MIN_MEASURED_PER_BAND:
            violations.append(f"insufficient_measured_requests:{band}")
    prompt_lengths = {}
    for band in LENGTH_BANDS:
        lengths = {r.get("prompt_tokens") for r in measured[band] if type(r.get("prompt_tokens")) is int}
        if len(lengths) != 1:
            violations.append(f"inconsistent_prompt_length:{band}")
        else:
            prompt_lengths[band] = next(iter(lengths))
    if len(prompt_lengths) == len(LENGTH_BANDS) and not all(
        prompt_lengths[a] < prompt_lengths[b] for a, b in zip(LENGTH_BANDS, LENGTH_BANDS[1:])
    ):
        violations.append("nonincreasing_prompt_length_bands")
    if any(r.get("phase") == "measured" and r.get("length_band") not in LENGTH_BANDS for r in rows):
        violations.append("unexpected_measured_length_band")
    sustained = [r for r in rows if r.get("phase") == "sustained"]
    active_wall = sum(float(r["wall_s"]) for r in sustained if _finite(r.get("wall_s")) and float(r["wall_s"]) > 0)
    start, duration = report.get("sustained_start_unix"), report.get("sustained_wall_s")
    if not sustained:
        violations.append("missing_sustained_requests")
    if not _finite(duration) or float(duration) < MIN_SUSTAINED_SECONDS:
        violations.append("sustained_wall_under_30_minutes")
    observed_window = None
    if not _finite(start):
        violations.append("missing_sustained_start")
    elif sustained:
        last = sustained[-1]
        if _finite(last.get("submitted_unix")) and _finite(last.get("wall_s")):
            observed_window = float(last["submitted_unix"]) + float(last["wall_s"]) - float(start)
        if observed_window is None or observed_window < MIN_SUSTAINED_SECONDS - 0.01:
            violations.append("sustained_request_window_under_30_minutes")
        elif active_wall < MIN_SUSTAINED_ACTIVE_FRACTION * observed_window:
            violations.append("sustained_active_fraction_low")
        if any(r.get("length_band") != "medium" for r in sustained):
            violations.append("sustained_workload_changed")
        if ("medium" in prompt_lengths and any(r.get("prompt_tokens") != prompt_lengths["medium"] for r in sustained)):
            violations.append("sustained_prompt_length_changed")
    output = output_checks(rows)
    if not output["recorded_invariants_pass"]:
        violations.append("recorded_output_invariants_failed")
    if metric_gaps:
        violations.append("measured_or_sustained_metric_gaps")

    first, last = sustained[:100], sustained[-100:]
    telemetry_ranges = {}
    for name in TELEMETRY:
        values = [float(r[name]) for r in telemetry if _finite(r.get(name))]
        telemetry_ranges[name] = {"n": len(values), "min": min(values) if values else None,
                                  "max": max(values) if values else None}
    return {
        "status": report.get("status"),
        "timing_protocol_complete": not violations,
        "violations": violations,
        "measured": {band: {"requests": len(measured[band]), "metrics": _metric_summary(measured[band]),
                            "prompt_tokens": sorted({r["prompt_tokens"] for r in measured[band]
                                                     if type(r.get("prompt_tokens")) is int})}
                     for band in LENGTH_BANDS},
        "sustained": {"requests": len(sustained), "declared_wall_s": duration,
                      "observed_request_window_s": observed_window, "active_request_wall_s": active_wall,
                      "active_fraction": active_wall / observed_window if observed_window and observed_window > 0 else None,
                      "metrics": _metric_summary(sustained),
                      "first_vs_last": {"cohort_size": 100, "nonoverlapping": len(sustained) >= 200,
                                        "first_wall_s": stats([r.get("wall_s") for r in first]),
                                        "last_wall_s": stats([r.get("wall_s") for r in last]),
                                        "first_ttft_s": stats([r.get("ttft_s") for r in first]),
                                        "last_ttft_s": stats([r.get("ttft_s") for r in last])}},
        "output": output,
        "metric_gaps": metric_gaps,
        "telemetry": {"samples": len(telemetry), "sampler_error": (report.get("gpu_telemetry") or {}).get("error"),
                      "ranges": telemetry_ranges, "scope": "Whole GPU/host, not process-attributed; brief peaks may be missed."},
        "interval_scope": "submitted_unix + wall_s is a conservative end estimate; the source loop submits sequentially.",
        "qualification": "Timing and recorded stream invariants only; reference quality and release gates remain separate.",
        "percentiles": "nearest rank over raw requests; warmups and sustained requests excluded from length-band statistics",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    rows = read_lines(args.run / "requests.jsonl")
    telemetry = read_lines(args.run / "gpu_telemetry.jsonl")
    print(json.dumps(analyze(report, rows, telemetry), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
