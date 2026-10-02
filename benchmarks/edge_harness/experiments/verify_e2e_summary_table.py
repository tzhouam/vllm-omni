#!/usr/bin/env python3
"""Verify 12×5 summary-table coverage and its stated E2E depth counts."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


# Archived batch-1, single-request timing protocols passed for these exact
# cells. This does not imply model quality, playback, or release qualification.
BATCH1_TIMING_EVIDENCE = {
    ("PC HX370 + RTX 5090 Laptop / WSL", "Spark-X2.5"): {
        "raw_requests": "../../../e2e_expansion_20261002/evidence/batch1_text_sustained/requests.jsonl",
        "analysis": "../../../e2e_expansion_20261002/evidence/batch1_text_sustained/analysis.json",
    },
    ("PC HX370 + RTX 5090 Laptop / WSL", "Qwen3-TTS 0.6B CustomVoice"): {
        "raw_requests": "../../../e2e_expansion_20261002/evidence/batch1_tts_sustained/requests.jsonl",
        "analysis": "../../../e2e_expansion_20261002/evidence/batch1_tts_sustained/analysis.json",
    },
}


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def verify(text: str) -> dict:
    """Check the rolling 60-cell table and return its conservative status ledger."""
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines)
                  if line.startswith("| Device | Spark-X2.5 | Qwen3-TTS"))
    models = lines[header].strip("|").strip().split(" | ")[1:]
    rows = []
    coverage = None
    for line in lines[header + 2:]:
        if not line.startswith("| "):
            break
        cells = line.strip().strip("|").strip().split(" | ")
        if len(cells) != 6:
            raise ValueError(f"matrix row has {len(cells)} columns: {line[:80]}")
        if cells[0] == "Coverage across 12 configurations":
            coverage = cells
            break
        rows.append(cells)
    if len(rows) != 12 or len({row[0] for row in rows}) != 12:
        raise ValueError("matrix must contain twelve distinct device configurations")
    complete = synthetic = unverified = 0
    complete_by_model = [0] * 5
    synthetic_by_model = [0] * 5
    cases = []
    for row in rows:
        hosted_target = row[0].startswith(("PC Snapdragon X Elite CRD", "Mobile Galaxy", "Embedded "))
        for model, cell in enumerate(row[1:]):
            batch1_evidence = BATCH1_TIMING_EVIDENCE.get((row[0], models[model]))
            if cell.startswith(("scoped E2E", "experimental scoped E2E")):
                complete += 1
                complete_by_model[model] += 1
                disposition = "scoped_complete_request"
            elif model == 4 and cell.startswith(("synthetic policy pass",
                                                  "experimental synthetic Omni policy")):
                synthetic += 1
                synthetic_by_model[model] += 1
                disposition = "synthetic_policy_only"
            elif cell.startswith("NOT E2E"):
                unverified += 1
                disposition = "no_complete_workload"
            else:
                raise ValueError(f"unrecognized depth for {row[0]}: {cell[:100]}")
            cases.append({
                "device_configuration": row[0],
                "model": models[model],
                "disposition": disposition,
                "release_qualification": "not_qualified",
                "qualification_outcome": "open",
                "batch1_protocol": "verified_timing_only" if batch1_evidence else "not_yet_verified",
                "batch1_protocol_evidence": batch1_evidence,
                "hosted_full_replay": "not_yet_verified" if hosted_target else "not_applicable",
                "evidence_summary": cell,
            })
    if coverage is None:
        raise ValueError("matrix coverage row is absent")
    for model in range(4):
        match = re.fullmatch(r"(\d+) scoped E2E.*; (\d+) not E2E", coverage[model + 1])
        if (match is None or tuple(map(int, match.groups())) !=
                (complete_by_model[model], 12 - complete_by_model[model])):
            raise ValueError(f"coverage row disagrees for model column {model + 1}")
    vla = re.match(r"(\d+) synthetic-policy passes; (\d+) without a complete policy", coverage[5])
    if vla is None or tuple(map(int, vla.groups())) != (synthetic_by_model[4], 12 - synthetic_by_model[4]):
        raise ValueError("InternVLA coverage row disagrees with policy depth")
    claim = re.search(
        r"\*\*(\d+) scoped complete-request E2E paths, "
        r"(\d+) synthetic InternVLA policy-only paths, and "
        r"(\d+) paths without a verified complete workload\*\*", text)
    if claim is None or tuple(map(int, claim.groups())) != (complete, synthetic, unverified):
        raise ValueError("opening summary count disagrees with the configuration matrix")
    if sum(case["hosted_full_replay"] == "not_yet_verified" for case in cases) != 25:
        raise ValueError("expected five hosted device configurations x five models")
    if sum(case["batch1_protocol"] == "verified_timing_only" for case in cases) != len(BATCH1_TIMING_EVIDENCE):
        raise ValueError("missing archived batch-1 timing cell")
    reason_header = "| Device | Model | Current profiling disposition | Reason / remaining work | Next step |"
    reason_start = lines.index(reason_header)
    reasons = {}
    for line in lines[reason_start + 2:]:
        if not line.startswith("| "):
            break
        parts = line.strip().strip("|").strip().split(" | ")
        if len(parts) != 5 or not all(parts):
            raise ValueError(f"malformed per-cell reason: {line[:100]}")
        key = parts[0], parts[1]
        if key in reasons:
            raise ValueError(f"duplicate per-cell reason: {key}")
        reasons[key] = parts[2:]
    expected_reasons = {(case["device_configuration"], case["model"]) for case in cases}
    if set(reasons) != expected_reasons:
        raise ValueError("per-cell reasons must match every one of the 60 matrix cells")
    for case in cases:
        disposition, reason, next_step = reasons[case["device_configuration"], case["model"]]
        case.update(recorded_disposition=disposition, remaining_work=reason, next_step=next_step)
    return {
        "schema_version": 1,
        "qualification_protocol": "single_request_batch1_v1",
        "scope": "Rolling 12x5 evidence-depth ledger; no cell is release-qualified by this table.",
        "qualification_outcomes": [
            "open",
            "qualified_full_workload",
            "hosted_functional_replay_pending_device",
            "measured_unsupported",
            "external_blocker",
        ],
        "counts": {
            "scoped_complete_request": complete,
            "synthetic_policy_only": synthetic,
            "no_complete_workload": unverified,
            "release_qualified": 0,
            "batch1_timing_verified": len(BATCH1_TIMING_EVIDENCE),
            "open": len(cases),
        },
        "cases": cases,
    }


def verify_timing_evidence(ledger: dict, readme: Path) -> None:
    """Check the archived timing claim against its files before writing a ledger.

    ``verify(text)`` deliberately remains a pure table parser. This separate
    file check prevents a missing or stale profile from becoming a verified
    timing cell just because the table still names it.
    """
    for case in ledger["cases"]:
        if case["batch1_protocol"] != "verified_timing_only":
            continue
        text_route = case["model"] == "Spark-X2.5"
        evidence = case["batch1_protocol_evidence"]
        label = f"{case['device_configuration']} x {case['model']}"
        raw = readme.parent / evidence["raw_requests"]
        analysis_path = readme.parent / evidence["analysis"]
        report_path = raw.parent / "report.json"
        telemetry_path = raw.parent / "gpu_telemetry.jsonl"
        power_path = raw.parent / "power_snapshot_during.json"
        for path in (raw, analysis_path, report_path, telemetry_path, power_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"{label}: missing or empty timing evidence: {path}")

        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
        if analysis.get("status") != "completed" or report.get("status") != "completed":
            raise ValueError(f"{label}: timing analysis and report must both be completed")
        if analysis.get("timing_protocol_complete") is not True or analysis.get("violations"):
            raise ValueError(f"{label}: analysis did not pass the batch-1 timing protocol")
        if analysis.get("metric_gaps") or analysis.get("finite_metric_gaps"):
            raise ValueError(f"{label}: analysis has missing or non-finite timing metrics")
        output = analysis.get("output") or {}
        if output.get("recorded_invariants_pass") is not True or output.get("findings"):
            raise ValueError(f"{label}: archived output invariants did not pass")
        protocol = report.get("profile_protocol") or {}
        settings = report.get("settings") or {}
        if (protocol.get("name") != "single_request_batch1_v1"
                or protocol.get("request_batch_size") != 1
                or protocol.get("max_active_requests") != 1
                or settings.get("batch_size") != 1
                or settings.get("concurrency") != 1
                or settings.get("length_band") is not None):
            raise ValueError(f"{label}: report is not a three-band batch-1 single-request run")
        sustained_wall = report.get("sustained_wall_s")
        analyzed_wall = (analysis.get("sustained") or {}).get("declared_wall_s")
        if not _finite_number(sustained_wall) or sustained_wall < 1800:
            raise ValueError(f"{label}: report lacks a completed 30-minute phase")
        if not _finite_number(analyzed_wall) or not math.isclose(
                sustained_wall, analyzed_wall, rel_tol=0, abs_tol=0.001):
            raise ValueError(f"{label}: analysis and report sustained durations disagree")
        if (report.get("requests") != len(rows)
                or output.get("requests_checked") != len(rows)):
            raise ValueError(f"{label}: raw, report and analysis request counts disagree")
        if text_route:
            plan = report.get("plan") or {}
            manifest = plan.get("manifest") or {}
            if (plan.get("backend") != "vllm:cuda"
                    or (plan.get("selected_device") or {}).get("name") != "NVIDIA GeForce RTX 5090 Laptop GPU"
                    or manifest.get("weight_sha256") !=
                    "5c91fc4a3664bc5744ef6cb654da8a792d52e104101874641eb8d6e7e932a826"
                    or manifest.get("dtype") != "bfloat16"):
                raise ValueError(f"{label}: Spark checkpoint or CUDA placement disagrees with archived claim")

        counts = Counter((row.get("phase"), row.get("length_band")) for row in rows)
        measured = analysis.get("measured") or {}
        for band in ("short", "medium", "long"):
            n = counts[("measured", band)]
            if (n < 20 or n != (measured.get(band) or {}).get("requests")
                    or counts[("warmup", band)] < 1):
                raise ValueError(f"{label}: raw and analysis {band} sample counts disagree")
        sustained = counts[("sustained", "medium")]
        if sustained < 1 or sustained != (analysis.get("sustained") or {}).get("requests"):
            raise ValueError(f"{label}: raw and analysis sustained counts disagree")
        if any(phase == "sustained" and band != "medium" for phase, band in counts):
            raise ValueError(f"{label}: unexpected sustained workload")
        seen_ids = set()
        previous_finished = None
        for row in rows:
            request_id = row.get("request_id")
            if not request_id or request_id in seen_ids:
                raise ValueError(f"{label}: missing or duplicate raw request ID")
            seen_ids.add(request_id)
            if (row.get("batch_size") != 1 or row.get("concurrency") != 1
                    or row.get("finished") is not True):
                raise ValueError(f"{label}: unfinished or non-single-request raw sample")
            submitted = row.get("submitted_unix")
            if text_route:
                wall = row.get("wall_s")
                if (not _finite_number(submitted) or not _finite_number(wall)
                        or wall <= 0
                        or (previous_finished is not None and submitted < previous_finished - 0.01)):
                    raise ValueError(f"{label}: invalid or overlapping raw request interval")
                previous_finished = submitted + wall
                arrivals = row.get("arrivals")
                if (row.get("output_tokens") != 128
                        or len(row.get("output_token_ids") or []) != 128
                        or not isinstance(arrivals, list)
                        or sum(event.get("kind") == "done" for event in arrivals) != 1
                        or not isinstance(row.get("token_unix"), list)):
                    raise ValueError(f"{label}: raw text output invariants failed")
                if row.get("phase") in ("measured", "sustained") and any(
                        not _finite_number(row.get(metric)) or row[metric] <= 0
                        for metric in ("wall_s", "ttft_s", "decode_tok_per_s")):
                    raise ValueError(f"{label}: raw timing metric is missing or non-finite")
            else:
                finished = row.get("finished_unix")
                if (not _finite_number(submitted) or not _finite_number(finished)
                        or finished < submitted
                        or (previous_finished is not None and submitted < previous_finished - 0.001)):
                    raise ValueError(f"{label}: invalid or overlapping raw request interval")
                previous_finished = finished
                chunks = row.get("chunks_all")
                if (not isinstance(chunks, list) or not chunks
                        or [chunk.get("idx") for chunk in chunks] != list(range(len(chunks)))
                        or sum(bool(chunk.get("terminal")) for chunk in chunks) != 1
                        or chunks[-1].get("terminal") is not True
                        or sum(chunk.get("samples", 0) for chunk in chunks) <= 0
                        or any(chunk.get("finite") is not True for chunk in chunks)):
                    raise ValueError(f"{label}: raw audio output invariants failed")
                if row.get("phase") in ("measured", "sustained") and any(
                        not _finite_number(row.get(metric)) for metric in
                        ("total_wall_s", "ttfa_ms", "playback_start_ms", "rtf_total", "stall_at_ttfa_ms")):
                    raise ValueError(f"{label}: raw timing metric is missing or non-finite")
        # Parse the remaining named evidence as well; its presence alone is
        # insufficient if capture stopped in the middle of a JSON record.
        if not any(line.strip() for line in telemetry_path.read_text(encoding="utf-8").splitlines()):
            raise ValueError(f"{label}: empty GPU telemetry")
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)
        json.loads(power_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("readme", type=Path)
    parser.add_argument("--json-out", type=Path,
                        help="Write a conservative machine-readable ledger after validation.")
    args = parser.parse_args()
    result = verify(args.readme.read_text(encoding="utf-8"))
    verify_timing_evidence(result, args.readme)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = result["counts"]
    print(f"verified {len(result['cases'])} cells: {counts['scoped_complete_request']} scoped E2E, "
          f"{counts['synthetic_policy_only']} synthetic policy-only, "
          f"{counts['no_complete_workload']} without E2E")


if __name__ == "__main__":
    main()
