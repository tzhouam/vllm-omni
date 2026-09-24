#!/usr/bin/env python3
"""Profile serial complete Spark BF16 text requests through the Omni CPU stage."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path

from vllm_omni.edge.local.engine import LocalTextEngine
from vllm_omni.edge.local.manifest import runtime_versions
from vllm_omni.edge.local.plan import plan_text_session
from vllm_omni.edge.local.prompts import acceptance_prompts


def rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


async def profile(args: argparse.Namespace) -> dict:
    plan = plan_text_session(
        str(args.model), max_model_len=4096, max_num_seqs=1,
        max_num_batched_tokens=2048, enforce_eager=True,
    )
    if not plan.admitted or plan.selected is None or plan.selected.device_id != "cpu":
        raise RuntimeError("the pinned Spark BF16 run was not admitted on CPU")
    name, prompt = acceptance_prompts()[0]
    runs = []
    async with LocalTextEngine(plan) as engine:
        for index in range(args.warmups + args.repeats):
            session = engine.open_session()
            started = time.perf_counter()
            request_id, stream = await engine.submit(
                session, prompt, max_tokens=args.max_tokens,
                temperature=0.0, ignore_eos=True,
            )
            async for _ in stream:
                pass
            wall_s = time.perf_counter() - started
            record = engine.records[request_id]
            if record.error or record.cancelled or not record.finished or record.output_tokens != args.max_tokens:
                raise RuntimeError(f"request {index} did not complete the specified text workload")
            digest = hashlib.sha256(json.dumps(record.output_token_ids).encode()).hexdigest()
            runs.append({
                "kind": "warmup" if index < args.warmups else "measured",
                "prompt_name": name,
                "prompt_tokens": record.prompt_tokens,
                "output_tokens": record.output_tokens,
                "token_ids_sha256": digest,
                "wall_s": wall_s,
                "ttft_s": record.ttft_s,
                "decode_tok_per_s": record.decode_tok_per_s,
                "stream": stream.stats(),
            })
            engine.close_session(session.session_id)
        placement = engine.report_placement()
        usage = engine.report_usage()
        load_s = engine.load_seconds
    measured = runs[args.warmups:]
    if len({row["token_ids_sha256"] for row in runs}) != 1:
        raise RuntimeError("serial greedy requests produced different token sequences")
    report = {
        "scope": "Spark-X2.5-1.7B dense BF16 complete text requests through Omni on WSL HX370 CPU",
        "status": "scoped_e2e_profiled",
        "model": str(args.model.resolve()),
        "runtime": runtime_versions().to_dict(),
        "plan": plan.to_dict(),
        "placement": placement,
        "usage": usage,
        "load_s_not_request_timing": load_s,
        "warmup_count": args.warmups,
        "measured_count": args.repeats,
        "max_new_tokens": args.max_tokens,
        "runs": runs,
        "nearest_rank_p50_wall_s": rank([r["wall_s"] for r in measured], 0.5),
        "nearest_rank_p95_wall_s": rank([r["wall_s"] for r in measured], 0.95),
        "nearest_rank_p50_ttft_s": rank([r["ttft_s"] for r in measured], 0.5),
        "nearest_rank_p95_ttft_s": rank([r["ttft_s"] for r in measured], 0.95),
        "limits": [
            "One short prompt and one concurrency level; the acceptance and parity suites are separate evidence.",
            "The host power condition was not controlled or recorded as a sustained thermal test.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 20 or args.max_tokens < 1:
        parser.error("expected at least one warmup, 20 measured requests and positive token count")
    report = asyncio.run(profile(args))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "measured_count", "nearest_rank_p50_wall_s", "nearest_rank_p95_wall_s",
        "nearest_rank_p50_ttft_s", "nearest_rank_p95_ttft_s",
    )}, indent=2))


if __name__ == "__main__":
    main()
