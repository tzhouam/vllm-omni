#!/usr/bin/env python3
"""Compare fixed and sixteen-slot Spark CPU full-cache bucket rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed-512", "bucket-512", "fixed-1024", "bucket-1024", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for prompt, fixed_path, bucket_path in (
        (500, args.fixed_512, args.bucket_512),
        (1000, args.fixed_1024, args.bucket_1024),
    ):
        fixed, bucket = load(fixed_path), load(bucket_path)
        if (fixed["model_index_sha256"] != bucket["model_index_sha256"]
                or fixed["model_config_sha256"] != bucket["model_config_sha256"]
                or fixed["prompt_ids_sha256"] != bucket["prompt_ids_sha256"]
                or not (fixed["prompt_tokens"] == bucket["prompt_tokens"] == prompt)
                or not (fixed["decode_steps"] == bucket["decode_steps"] == 128)
                or len(fixed["steps"]) != 128 or len(bucket["steps"]) != 128
                or not (fixed["export_arithmetic"] == bucket["export_arithmetic"]
                        == "hf_bf16_reference")
                or not fixed["ordered_sliding"] or not bucket["ordered_sliding"]
                or bucket["full_bucket_width"] != 16
                or len(bucket["bucket_transitions"]) != 8):
            raise ValueError(f"{prompt}: source, inputs, arithmetic or buckets changed")
        token_fields = ("position", "input_token", "hf_next_token", "export_next_token")
        if any(any(left[field] != right[field] for field in token_fields)
               for left, right in zip(fixed["steps"], bucket["steps"])):
            raise ValueError(f"{prompt}: bucket changed teacher-forced token path")
        steps = bucket["steps"]
        worst = max(steps, key=lambda row: row["max_new_kv_relative_l2"])
        rows.append({
            "prompt_tokens": prompt,
            "fixed_report_sha256": sha256(fixed_path),
            "bucket_report_sha256": sha256(bucket_path),
            "prompt_ids_sha256": bucket["prompt_ids_sha256"],
            "step_count": len(steps),
            "top1_matches": bucket["top1_matches"],
            "logit_error_changed_steps": sum(
                left["logits_relative_l2"] != right["logits_relative_l2"]
                for left, right in zip(fixed["steps"], steps)),
            "new_kv_error_changed_steps": sum(
                left["max_new_kv_relative_l2"] != right["max_new_kv_relative_l2"]
                for left, right in zip(fixed["steps"], steps)),
            "max_logits_relative_l2": bucket["max_logits_relative_l2"],
            "max_new_kv_relative_l2": bucket["max_new_kv_relative_l2"],
            "fixed_max_new_kv_relative_l2": fixed["max_new_kv_relative_l2"],
            "first_new_kv_over_1pct_position": next(
                (row["position"] for row in steps
                 if row["max_new_kv_relative_l2"] > 0.01), None),
            "worst_new_kv_position": worst["position"],
            "worst_new_kv": worst["worst_new_kv"],
            "fixed_full_capacity": fixed["full_cache_capacity"],
            "bucket_initial_capacity": bucket["initial_full_cache_capacity"],
            "bucket_final_capacity": bucket["full_cache_capacity"],
            "bucket_transition_count": len(bucket["bucket_transitions"]),
        })
    report = {
        "scope": "real-weight BF16 CPU Spark decode with self-owned KV; fixed versus 16-slot full-cache buckets, not mobile execution",
        "rows": rows,
        "status": ("buckets_do_not_repair_state_gate"
                   if any(row["max_new_kv_relative_l2"] > 0.01 for row in rows)
                   else "bucketed_cpu_state_gate_passed"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "max_new_kv_relative_l2": [row["max_new_kv_relative_l2"]
                                                 for row in rows]}))


if __name__ == "__main__":
    main()
