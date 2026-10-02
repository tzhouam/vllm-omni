# SPDX-License-Identifier: Apache-2.0
"""Read-only batch-1 TTS timing analysis from an existing profile directory.

Print JSON to stdout; do not mutate the run. A timing-protocol pass is deliberately
separate from reference, speaker, listening, and device-local playback quality.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from single_request_protocol import LENGTH_BANDS, MIN_MEASURED_PER_BAND, MIN_SUSTAINED_SECONDS, PROTOCOL
from summarize_e2e_profiles import output_checks, read_lines, stats

METRICS = ("total_wall_s", "ttfa_ms", "playback_start_ms", "rtf_total", "stall_at_ttfa_ms")
TELEMETRY = (
    "gpu_temperature_c",
    "gpu_power_mw",
    "gpu_clock_sm_mhz",
    "gpu_clock_memory_mhz",
    "gpu_throttle_reasons",
    "gpu_utilization_pct",
    "gpu_memory_used_bytes",
    "host_available_bytes",
    "host_cpu_utilization_pct",
)


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _startup_buffer_ms(row: dict[str, Any]) -> float | None:
    """Minimum delay after first audio to avoid stalls for observed arrivals."""
    first, start = row.get("ttfa_ms"), row.get("playback_start_ms")
    if not (_finite(first) and _finite(start)):
        return None
    return max(0.0, float(start) - float(first))


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **{name: stats([row.get(name) for row in rows]) for name in METRICS},
        "minimum_underrun_free_startup_buffer_ms": stats([_startup_buffer_ms(row) for row in rows]),
    }


def _playback_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stalls = [stall for row in rows for _, stall in row.get("underruns", [])]
    buffers = [value for row in rows if (value := _startup_buffer_ms(row)) is not None]
    return {
        "requests": len(rows),
        "requests_with_simulated_underruns": sum(bool(row.get("underruns")) for row in rows),
        "simulated_underrun_count": len(stalls),
        "simulated_underrun_max_ms": max(stalls) if stalls else 0.0,
        "simulated_underrun_total_ms": sum(stalls),
        "minimum_underrun_free_startup_buffer_ms": {
            "minimum": min(buffers) if buffers else None,
            "maximum": max(buffers) if buffers else None,
            "p50": stats(buffers)["p50"],
            "p95": stats(buffers)["p95"],
        },
        "scope": "Arrival-based simulated playback, starting at first audio; no sound-device measurement.",
    }


def _telemetry_ranges(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranges = {}
    for name in TELEMETRY:
        values = [float(row[name]) for row in rows if _finite(row.get(name))]
        ranges[name] = {
            "n": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    times = sorted(float(row["unix"]) for row in rows if _finite(row.get("unix")))
    return {
        "samples": len(rows),
        "first_unix": times[0] if times else None,
        "last_unix": times[-1] if times else None,
        "max_sample_gap_s": max((b - a for a, b in zip(times, times[1:])), default=None),
        "ranges": ranges,
        "scope": "Sampled whole GPU/host, not process-attributed; brief peaks may be missed.",
    }


def analyze(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]],
    *,
    cohort_size: int = 100,
) -> dict[str, Any]:
    """Analyze raw records without changing the report or inferring audio quality."""
    if cohort_size < 1:
        raise ValueError("cohort_size must be positive")
    settings = report.get("settings") or {}
    declared = report.get("profile_protocol") or {}
    violations: list[str] = []
    if report.get("status") != "completed":
        violations.append("report_not_completed")
    if (
        declared.get("name") != PROTOCOL
        or declared.get("request_batch_size") != 1
        or declared.get("max_active_requests") != 1
    ):
        violations.append("wrong_or_missing_batch1_protocol")
    if settings.get("batch_size") != 1 or settings.get("concurrency") != 1:
        violations.append("settings_not_batch1_concurrency1")
    if report.get("requests") != len(rows):
        violations.append("report_request_count_mismatch")

    seen: set[str] = set()
    previous_finished: float | None = None
    metric_gaps = []
    for row in rows:
        rid = row.get("request_id")
        if not rid or rid in seen:
            violations.append(f"missing_or_duplicate_request_id:{rid}")
        seen.add(rid)
        if row.get("batch_size") != 1 or row.get("concurrency") != 1:
            violations.append(f"non_single_request_sample:{rid}")
        submitted, finished = row.get("submitted_unix"), row.get("finished_unix")
        if not (_finite(submitted) and _finite(finished)) or float(finished) < float(submitted):
            violations.append(f"invalid_request_interval:{rid}")
        else:
            if previous_finished is not None and float(submitted) < previous_finished - 1e-3:
                violations.append(f"overlapping_requests:{rid}")
            previous_finished = float(finished)
        if row.get("phase") in ("measured", "sustained"):
            gaps = [name for name in METRICS if not _finite(row.get(name))]
            if _startup_buffer_ms(row) is None:
                gaps.append("minimum_underrun_free_startup_buffer_ms")
            if gaps:
                metric_gaps.append({"request_id": rid, "fields": gaps})

    measured = {
        band: [row for row in rows if row.get("phase") == "measured" and row.get("length_band") == band]
        for band in LENGTH_BANDS
    }
    for band in LENGTH_BANDS:
        if not any(row.get("phase") == "warmup" and row.get("length_band") == band for row in rows):
            violations.append(f"missing_warmup:{band}")
        if len(measured[band]) < MIN_MEASURED_PER_BAND:
            violations.append(f"insufficient_measured_requests:{band}")
    if any(row.get("phase") == "measured" and row.get("length_band") not in LENGTH_BANDS for row in rows):
        violations.append("unexpected_measured_length_band")

    sustained = [row for row in rows if row.get("phase") == "sustained"]
    start = report.get("sustained_start_unix")
    duration = report.get("sustained_wall_s")
    if not sustained:
        violations.append("missing_sustained_requests")
    if not _finite(duration) or float(duration) < MIN_SUSTAINED_SECONDS:
        violations.append("sustained_wall_under_30_minutes")
    if not _finite(start):
        violations.append("missing_sustained_start")
    elif sustained:
        first, last = sustained[0], sustained[-1]
        if not (_finite(first.get("submitted_unix")) and _finite(last.get("finished_unix"))):
            violations.append("sustained_request_times_missing")
        elif (
            float(first["submitted_unix"]) < float(start) - 1
            or float(last["finished_unix"]) - float(start) < MIN_SUSTAINED_SECONDS - 1
        ):
            violations.append("sustained_request_window_under_30_minutes")
        if any(row.get("length_band") != "medium" for row in sustained):
            violations.append("sustained_workload_changed")

    output = output_checks(rows)
    if not output["recorded_invariants_pass"]:
        violations.append("recorded_output_invariants_failed")
    if metric_gaps:
        violations.append("measured_or_sustained_metric_gaps")

    failures = {
        "recorded_output_failure_requests": len(output["findings"]),
        "nonfinite_audio_requests": sum(
            any(not chunk.get("finite") for chunk in row.get("chunks_all", [])) for row in rows
        ),
        "unfinished_requests": sum(not row.get("finished") for row in rows),
        "nonfinite_or_missing_metric_requests": len(metric_gaps),
    }

    first_cohort = sustained[:cohort_size]
    last_cohort = sustained[-cohort_size:] if sustained else []
    cohorts_nonoverlapping = len(sustained) >= cohort_size * 2
    sustained_end = float(start) + float(duration) if _finite(start) and _finite(duration) else None
    sustained_telemetry = (
        [
            row
            for row in telemetry_rows
            if _finite(row.get("unix")) and float(start) <= float(row["unix"]) <= sustained_end
        ]
        if sustained_end is not None
        else []
    )
    telemetry_error = (report.get("gpu_telemetry") or {}).get("error")
    return {
        "status": report.get("status"),
        "timing_protocol_complete": not violations,
        "violations": violations,
        "measured": {
            band: {
                "requests": len(measured[band]),
                "metrics": _metric_summary(measured[band]),
                "playback": _playback_summary(measured[band]),
            }
            for band in LENGTH_BANDS
        },
        "sustained": {
            "requests": len(sustained),
            "declared_wall_s": duration,
            "observed_request_window_s": (
                float(sustained[-1]["finished_unix"]) - float(start)
                if sustained and _finite(start) and _finite(sustained[-1].get("finished_unix"))
                else None
            ),
            "metrics": _metric_summary(sustained),
            "playback": _playback_summary(sustained),
            "first_vs_last": {
                "cohort_size": cohort_size,
                "nonoverlapping": cohorts_nonoverlapping,
                "first_n": len(first_cohort),
                "last_n": len(last_cohort),
                "first_total_wall_s": stats([row.get("total_wall_s") for row in first_cohort]),
                "last_total_wall_s": stats([row.get("total_wall_s") for row in last_cohort]),
                "first_ttfa_ms": stats([row.get("ttfa_ms") for row in first_cohort]),
                "last_ttfa_ms": stats([row.get("ttfa_ms") for row in last_cohort]),
            },
        },
        "output": output,
        "failures": failures,
        "finite_metric_gaps": metric_gaps,
        "telemetry": {
            "sampler_error": telemetry_error,
            "whole_run": _telemetry_ranges(telemetry_rows),
            "sustained_window": _telemetry_ranges(sustained_telemetry),
        },
        "qualification": (
            "Timing and recorded finite/output checks only; audio reference, speaker, listening, "
            "device-local playback, and release qualification remain separate."
        ),
        "percentiles": (
            "nearest rank over raw requests; warmups, recovery probes and sustained requests "
            "excluded from per-band measured statistics"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed profile directory containing report.json and requests.jsonl")
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    rows = read_lines(args.run / "requests.jsonl")
    telemetry_rows = read_lines(args.run / "gpu_telemetry.jsonl")
    print(json.dumps(analyze(report, rows, telemetry_rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
