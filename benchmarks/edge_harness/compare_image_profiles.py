# SPDX-License-Identifier: Apache-2.0
"""Compare closed image profiles; independent audit reports are trusted inputs.

This checks receipt/sample bindings, not every raw file again. It does not
establish causal speedup, broad task quality, or release qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

LENGTHS = ("short", "medium", "long")
FIELDS = ("whole_agent_task_s_excludes_load", "first_visible_output_s")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_closed(run, audit_path):
    run = Path(run).resolve()
    audit = read(audit_path)
    receipt = read(run / "receipt.json")
    require(audit.get("status") == "passed" and audit.get("mode") == "full", "full independent audit required")
    require(audit.get("receipt_sha256") == digest(run / "receipt.json"), "audit receipt binding differs")
    require(receipt.get("phase") == "completed" and receipt.get("profile_mode") == "full", "run is not closed/full")
    for flag in ("profile_completed", "quality_passed", "resource_drain_verified", "source_stable", "prerequisites_stable", "output_writers_quiesced"):
        require(receipt.get(flag) is True, "missing closure: " + flag)
    require(receipt.get("batch_size") == 1 and receipt.get("concurrency") == 1, "batch/concurrency differs")
    require(receipt.get("endurance_active_request_s", 0) >= 1800, "active endurance incomplete")
    for name in ("profile-samples.json", "hardware.json", "effective-launch.json"):
        expected = receipt["raw_files"][name]
        path = run / name
        require(path.stat().st_size == expected["size_bytes"] and digest(path) == expected["sha256"], "raw binding differs: " + name)
    rows = read(run / "profile-samples.json")
    require(len(rows) == audit["requests"] and rows == receipt["samples"], "sample/receipt/audit mismatch")
    require(all(r.get("status") == "passed" for r in rows), "failed request present")
    groups = {length: [r for r in rows if r["phase"] == "measured" and r["length"] == length] for length in LENGTHS}
    for length, values in groups.items():
        require(len(values) >= 20, "insufficient measurements: " + length)
        require([r["ordinal"] for r in values] == list(range(len(values))), "sample sequence differs")
        for row in values:
            for field in FIELDS:
                value = row[field]
                require(type(value) in (int, float) and math.isfinite(value) and value > 0, "invalid timing")
    return dict(run=str(run), receipt=receipt, audit_sha256=digest(audit_path), groups=groups, hardware=read(run / "hardware.json"))


def nearest_rank(values, q):
    return sorted(values)[math.ceil(len(values) * q) - 1]


def compare(baseline, candidate):
    a, b = baseline["receipt"], candidate["receipt"]
    for key in ("precision_binding", "runtime_identity", "runtime_sha256", "source_before"):
        require(a[key] == b[key], "execution identity differs: " + key)
    for key in ("launch_sha256", "reference_sha256", "image_sha256", "engine_runner_sha256"):
        require(a["prerequisite"][key] == b["prerequisite"][key], "prerequisite differs: " + key)
    for key in ("os", "machine", "cpu", "host_ram_total_bytes", "gpu_name", "gpu_driver", "vram_total_bytes", "power_condition"):
        require(key in baseline["hardware"] and baseline["hardware"][key] == candidate["hardware"].get(key), "hardware/power differs: " + key)
    result = {}
    for length in LENGTHS:
        left, right = baseline["groups"][length], candidate["groups"][length]
        require(len(left) == len(right), "unequal sample count")
        for x, y in zip(left, right):
            for key in ("usage", "answer", "prompt_utf8_bytes"):
                require(x[key] == y[key], "request workload differs: " + length + "/" + key)
        result[length] = {"n": len(left)}
        for field in FIELDS:
            x, y = [r[field] for r in left], [r[field] for r in right]
            record = {"baseline_samples": x, "candidate_samples": y}
            for label, q in (("p50", .5), ("p95", .95)):
                av, bv = nearest_rank(x, q), nearest_rank(y, q)
                record[label] = {"baseline_s": av, "candidate_s": bv, "candidate_minus_baseline_s": bv-av, "candidate_over_baseline": bv/av}
            result[length][field] = record
    return dict(schema="omni-image-profile-comparison-v1", lengths=result,
        baseline_run=baseline["run"], candidate_run=candidate["run"],
        baseline_audit_sha256=baseline["audit_sha256"], candidate_audit_sha256=candidate["audit_sha256"],
        method="nearest rank, measured requests only; warmup/endurance excluded",
        qualified=False, causal_speedup_established=False,
        limitations=["Fixed screenshot task family only", "No control of cache state, background activity or continuous power conditions", "Equal token counts and reference bindings do not prove identical raw prompts", "Independent audits trusted; this comparison does not re-audit all raw tensors/events", "Sidecar observer versions/overhead must be compared separately"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "baseline-audit", "candidate", "candidate-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(args.baseline.resolve() != args.candidate.resolve(), "distinct runs required")
    for root in (args.baseline, args.candidate):
        require(not args.output.resolve().is_relative_to(root.resolve()), "output must be outside closed evidence")
    result = compare(load_closed(args.baseline, args.baseline_audit), load_closed(args.candidate, args.candidate_audit))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
