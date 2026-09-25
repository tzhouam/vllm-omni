#!/usr/bin/env python3
"""Audit direct-vLLM Spark batch-slot variants against the earlier Omni run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omni-report", type=Path, required=True)
    parser.add_argument("--omni-requests", type=Path, required=True)
    parser.add_argument("--auto", type=Path, required=True)
    parser.add_argument("--reduction", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.omni_report.read_text(encoding="utf-8"))
    if prior["status"] != "completed" or prior["plan"]["backend"] != "vllm:cuda":
        raise ValueError("prior Omni CUDA profile is not complete")

    omni: dict[tuple[str, int], Counter[tuple[int, ...]]] = defaultdict(Counter)
    with args.omni_requests.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if item.get("phase") == "measured":
                omni[(item["length_band"], item["concurrency"])][
                    tuple(item["output_token_ids"])
                ] += 1
    if any(sum(omni[(band, batch)].values()) != 20
           for band in ("medium", "long") for batch in (1, 2, 4)):
        raise ValueError("earlier Omni comparison is not the expected 20-request profile")

    profiles = {}
    baseline = None
    for label, path, expected_override in (
        ("auto", args.auto, "auto"),
        ("reduction", args.reduction, "1"),
        ("fp32", args.fp32, "fp32"),
    ):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report["status"] != "passed"
                or report.get("gate_reduction_override", "auto") != expected_override
                or len(report["cases"]) != 6):
            raise ValueError(f"{label} reference run is incomplete")
        artifact = (report["model"], report["weight_sha256"],
                    report["config_sha256"], report["prior_report_sha256"],
                    report["engine_kwargs"])
        if (report["prior_report_sha256"] != sha256(args.omni_report)
                or report["config_sha256"] != prior["plan"]["manifest"]["config_sha256"]
                or report["engine_kwargs"] != prior["plan"]["engine_kwargs"]):
            raise ValueError(f"{label} does not match the prior Omni plan")
        aggregate = hashlib.sha256()
        for name, digest in report["weight_sha256"].items():
            aggregate.update(name.encode("utf-8"))
            aggregate.update(digest.encode("ascii"))
        if aggregate.hexdigest() != prior["plan"]["manifest"]["weight_sha256"]:
            raise ValueError(f"{label} does not match the prior Omni weights")
        if baseline is not None and artifact != baseline:
            raise ValueError(f"{label} used a different model, baseline or engine plan")
        baseline = artifact
        cases = []
        for case in report["cases"]:
            band, batch = case["band"], case["batch"]
            if len(case["slots"]) != batch:
                raise ValueError(f"{label} {band}/{batch} has the wrong slot count")
            slots = []
            for slot in case["slots"]:
                ids = tuple(slot["token_ids"])
                historical_count = omni[(band, batch)][ids]
                if historical_count == 0:
                    raise ValueError(f"{label} {band}/{batch} slot {slot['slot']} is a new variant")
                diff = slot["first_difference_from_batch1"]
                if diff is not None and diff["index"] not in (84, 89):
                    raise ValueError(f"{label} has an unexpected first difference")
                slots.append({
                    "slot": slot["slot"],
                    "first_difference_from_batch1": diff["index"] if diff else None,
                    "matching_prior_omni_requests": historical_count,
                    "probe_top_two": slot.get("probe_logprobs", [])[:2],
                })
            cases.append({"band": band, "batch": batch, "slots": slots})
        profiles[label] = {"report_sha256": sha256(path), "cases": cases}

    summary = {
        "scope": "same BF16 Spark checkpoint, direct vLLM vs prior Omni; greedy output variants only",
        "omni_requests_sha256": sha256(args.omni_requests),
        "omni_report_sha256": sha256(args.omni_report),
        "model": baseline[0],
        "weight_sha256": baseline[1],
        "profiles": profiles,
        "conclusion": "Direct vLLM reproduces the Omni batch-slot variants; FP32 gate alone does not remove long-prompt variation. Task quality remains unqualified.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "conclusion": summary["conclusion"]}))


if __name__ == "__main__":
    main()
