#!/usr/bin/env python3
"""Verify 12×5 summary-table coverage and its stated E2E depth counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
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

# These are exact AI Hub registrations, not interchangeable device families.
HOSTED_TARGET_NAMES = {
    "PC Snapdragon X Elite CRD / AI Hub": "Snapdragon X Elite CRD",
    "Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy": "Samsung Galaxy S25",
    "Mobile Galaxy S24 / Snapdragon 8 Gen 3": "Samsung Galaxy S24",
    "Embedded SA8775P ADP / AI Hub": "SA8775P ADP",
    "Embedded RB3 Gen 2 / QCS6490 / AI Hub": "Dragonwing RB3 Gen 2 Vision Kit",
}

# A successful runner journal is necessary but not sufficient: a one-layer
# component can supply its own tiny topology. These checks are the per-model
# acceptance contract for promoting a hosted matrix cell to full replay.
HOSTED_REQUIRED_CHECKS = {
    "Spark-X2.5": {"real_prompt", "full_neural_coverage", "continuous_128",
                   "cache_512", "cache_1024", "reference_quality", "state_order"},
    "Qwen3-TTS 0.6B CustomVoice": {"real_text", "talker_predictor_vocoder",
                                  "pcm_tail", "reference_audio_quality", "state_order"},
    "Qwen3.8-27B": {"real_image", "complete_vision_language_chain",
                     "language_state", "reference_quality", "peak_memory_admission"},
    "MiniCPM-o 4.5": {"real_image_and_speech", "complete_multimodal_chain",
                       "text_and_audio_quality", "state_order"},
    "InternVLA-A1": {"real_observation", "complete_policy_and_action",
                      "matching_action_reference", "action_units_timing_order"},
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_replay_file(root: Path, name: object, digest: object) -> Path:
    if (not isinstance(name, str) or not name or Path(name).is_absolute()
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise ValueError("hosted replay evidence needs a relative file and SHA-256")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or _file_sha256(path) != digest:
        raise ValueError("hosted replay evidence file is missing or changed")
    return path


def verify_hosted_model_acceptance(record: dict, root: Path) -> None:
    """Reject a complete-looking component journal lacking whole-model gates."""
    identity = record["identity"]
    model = identity["model"]
    required = HOSTED_REQUIRED_CHECKS[model]
    completion = record["completion"]
    quality = completion["quality"]
    if identity["revision"] not in quality.get("reference", ""):
        raise ValueError("hosted replay reference is not pinned to checkpoint revision")
    checks = quality.get("checks")
    if not isinstance(checks, list) or any(not isinstance(check, dict) for check in checks):
        raise ValueError("hosted replay lacks model-specific checks")
    by_name = {check.get("name"): check for check in checks}
    if len(by_name) != len(checks) or not required.issubset(by_name):
        raise ValueError("hosted replay omits whole-model acceptance checks")
    for name in required:
        check = by_name[name]
        if check.get("passed") is not True:
            raise ValueError(f"hosted replay quality gate {name} did not pass")
        _checked_replay_file(root, check.get("evidence_path"), check.get("sha256"))
    output_path = _checked_replay_file(
        root, completion["final_output_path"], completion["final_output_sha256"])
    output = json.loads(output_path.read_text(encoding="utf-8"))
    if model == "Spark-X2.5":
        tokens = output.get("token_ids")
        if (not isinstance(tokens, list) or len(tokens) < 128
                or any(type(token) is not int or token < 0 for token in tokens)
                or output.get("generated_tokens") != len(tokens)
                or output.get("cache_boundaries_verified") != [512, 1024]
                or not isinstance(output.get("prompt_length"), int)):
            raise ValueError("Spark hosted replay lacks 128 tokens or 512/1024 state evidence")
        if sum("decode" in step["stage_name"] for step in record["steps"]) < 128:
            raise ValueError("Spark hosted replay lacks 128 dependent decode jobs")
    elif model in {"Qwen3-TTS 0.6B CustomVoice", "MiniCPM-o 4.5"}:
        _checked_replay_file(root, output.get("pcm_path"), output.get("pcm_sha256"))
        if (not isinstance(output.get("sample_rate"), int) or output["sample_rate"] <= 0
                or not isinstance(output.get("samples"), int) or output["samples"] <= 0
                or output.get("tail_flushed") is not True):
            raise ValueError("hosted audio replay lacks complete PCM with tail")
        if model == "MiniCPM-o 4.5" and (not output.get("text")
                                            or not output.get("image_sha256")
                                            or not output.get("speech_input_sha256")):
            raise ValueError("MiniCPM-o hosted replay lacks real multimodal input and text")
    elif model == "Qwen3.8-27B":
        if (not output.get("text") or not output.get("image_sha256")
                or not output.get("token_ids")
                or not isinstance(output.get("peak_memory_bytes"), int)
                or not isinstance(output.get("usable_ram_bytes"), int)
                or output["peak_memory_bytes"] <= 0 or output["usable_ram_bytes"] <= 0):
            raise ValueError("27B hosted replay lacks image, text, state or RAM evidence")
    elif model == "InternVLA-A1":
        actions = output.get("actions")
        times = output.get("timestamps_ms")
        if (not isinstance(actions, list) or not actions
                or not isinstance(times, list) or len(actions) != len(times)
                or any(not _finite_number(value) for value in times)
                or any(b <= a for a, b in zip(times, times[1:]))
                or not output.get("units") or not output.get("observation_sha256")):
            raise ValueError("VLA hosted replay lacks ordered, unit-tagged actions")


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


def apply_hosted_evidence(ledger: dict, index_path: Path) -> None:
    """Promote hosted cells only from an audited full-chain run or named evidence.

    The index may omit unfinished cells. In particular, a component job or a
    synthetic host policy is never promoted by its mere presence in the matrix.
    """
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != 1 or not isinstance(index.get("entries"), list):
        raise ValueError("hosted evidence index must contain schema_version=1 and entries")
    cases = {(case["device_configuration"], case["model"]): case
             for case in ledger["cases"]}
    seen: set[tuple[str, str]] = set()
    repository_root = Path(__file__).resolve().parents[3]
    for entry in index["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("hosted evidence entry must be an object")
        key = entry.get("device_configuration"), entry.get("model")
        if key in seen or key not in cases or key[0] not in HOSTED_TARGET_NAMES:
            raise ValueError(f"duplicate, unknown or non-hosted evidence cell: {key}")
        seen.add(key)
        path_value = entry.get("evidence_path")
        if not isinstance(path_value, str) or not path_value or Path(path_value).is_absolute():
            raise ValueError(f"{key}: evidence_path must be a relative file path")
        evidence_path = (index_path.parent / path_value).resolve(strict=True)
        if not evidence_path.is_relative_to(repository_root) or not evidence_path.is_file():
            raise ValueError(f"{key}: hosted evidence must be a repository file")
        outcome = entry.get("outcome")
        case = cases[key]
        target = HOSTED_TARGET_NAMES[key[0]]
        if outcome == "hosted_functional_replay_pending_device":
            if evidence_path.name != "run.json":
                raise ValueError(f"{key}: full replay evidence must be run.json")
            runner_root = Path(__file__).resolve().parents[1]
            if str(runner_root) not in sys.path:
                sys.path.insert(0, str(runner_root))
            from aihub_replay import audit_run

            audit = audit_run(evidence_path.parent)
            record = json.loads(evidence_path.read_text(encoding="utf-8"))
            if (audit["status"] != "complete" or audit["model"] != key[1]
                    or audit["target_name"] != target or not audit["job_ids"]
                    or record.get("tier") != "aihub_hosted_functional_replay"
                    or record.get("facility") != "Qualcomm AI Hub Workbench"
                    or record.get("target_registration", {}).get("name") != target):
                raise ValueError(f"{key}: incomplete, wrong-target or unaudited hosted replay")
            verify_hosted_model_acceptance(record, evidence_path.parent)
            case["hosted_full_replay"] = "verified_functional_replay"
            case["qualification_outcome"] = outcome
            case["hosted_replay_job_ids"] = audit["job_ids"]
        elif outcome in {"measured_unsupported", "external_blocker"}:
            report = json.loads(evidence_path.read_text(encoding="utf-8"))
            if (report.get("status") != outcome or report.get("model") != key[1]
                    or report.get("target_name") != target
                    or not isinstance(report.get("reason"), str) or not report["reason"]
                    or not isinstance(report.get("next_step"), str) or not report["next_step"]):
                raise ValueError(f"{key}: failure or blocker report lacks exact scope")
            if outcome == "measured_unsupported" and (
                    not report.get("job_id") or not report.get("artifact_sha256")
                    or not report.get("raw_failure_path")):
                raise ValueError(f"{key}: unsupported verdict lacks an exact failed artifact/job")
            if outcome == "external_blocker" and not report.get("source_evidence"):
                raise ValueError(f"{key}: external blocker lacks source evidence")
            source_name = ("raw_failure_path" if outcome == "measured_unsupported"
                           else "source_evidence")
            source_value = report[source_name]
            if not isinstance(source_value, str) or Path(source_value).is_absolute():
                raise ValueError(f"{key}: {source_name} must name a relative raw evidence file")
            source_path = (evidence_path.parent / source_value).resolve(strict=True)
            if (not source_path.is_relative_to(repository_root)
                    or not source_path.is_file() or source_path.stat().st_size == 0):
                raise ValueError(f"{key}: {source_name} is missing or empty")
            hash_name = ("raw_failure_sha256" if outcome == "measured_unsupported"
                         else "source_sha256")
            if (not isinstance(report.get(hash_name), str)
                    or re.fullmatch(r"[0-9a-f]{64}", report[hash_name]) is None
                    or _file_sha256(source_path) != report[hash_name]):
                raise ValueError(f"{key}: {source_name} hash is missing or changed")
            if outcome == "measured_unsupported" and (
                    re.fullmatch(r"[0-9a-f]{64}", report["artifact_sha256"]) is None
                    or re.fullmatch(r"j[a-z0-9]+", report["job_id"]) is None):
                raise ValueError(f"{key}: failed artifact or job identity is malformed")
            case["hosted_full_replay"] = outcome
            case["qualification_outcome"] = outcome
        else:
            raise ValueError(f"{key}: unknown hosted outcome {outcome!r}")
        case["hosted_evidence_path"] = str(evidence_path.relative_to(repository_root))
        case["release_qualification"] = "not_qualified"

    ledger["counts"].update({
        "hosted_replay_verified": sum(case["hosted_full_replay"] == "verified_functional_replay"
                                      for case in ledger["cases"]),
        "hosted_artifact_unsupported": sum(case["hosted_full_replay"] == "measured_unsupported"
                                            for case in ledger["cases"]),
        "hosted_external_blocker": sum(case["hosted_full_replay"] == "external_blocker"
                                       for case in ledger["cases"]),
        "open": sum(case["qualification_outcome"] == "open" for case in ledger["cases"]),
    })


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
    parser.add_argument("--hosted-index", type=Path,
                        help="Evidence index for AI Hub full-chain outcomes; defaults to the sibling index.")
    args = parser.parse_args()
    result = verify(args.readme.read_text(encoding="utf-8"))
    verify_timing_evidence(result, args.readme)
    hosted_index = args.hosted_index or args.readme.parent / "hosted_e2e_index.json"
    if hosted_index.is_file():
        apply_hosted_evidence(result, hosted_index)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = result["counts"]
    print(f"verified {len(result['cases'])} cells: {counts['scoped_complete_request']} scoped E2E, "
          f"{counts['synthetic_policy_only']} synthetic policy-only, "
          f"{counts['no_complete_workload']} without E2E; "
          f"hosted full replay {counts.get('hosted_replay_verified', 0)}/25, "
          f"external blocker {counts.get('hosted_external_blocker', 0)}/25")


if __name__ == "__main__":
    main()
