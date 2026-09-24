# SPDX-License-Identifier: Apache-2.0
"""Repeat one pinned Spark BF16 CPU prompt in fresh sessions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from vllm_omni.edge.local.engine import LocalTextEngine
from vllm_omni.edge.local.plan import plan_text_session


async def probe(args: argparse.Namespace) -> dict:
    prompts = json.loads(args.prompts.read_text())
    prompt = next(item["prompt"] for item in prompts if item["name"] == args.name)
    plan = plan_text_session(
        str(args.model), max_model_len=4096, max_num_seqs=1,
        max_num_batched_tokens=2048, enforce_eager=True,
    )
    if not plan.admitted or plan.selected is None or plan.selected.device_id != "cpu":
        raise RuntimeError("BF16 CPU reference was not admitted")
    rows = []
    async with LocalTextEngine(plan) as engine:
        for index in range(args.repeats):
            session = engine.open_session()
            try:
                request_id, stream = await engine.submit(
                    session, prompt, max_tokens=args.max_tokens,
                    temperature=0.0, ignore_eos=True,
                )
                async for _ in stream:
                    pass
                record = engine.records[request_id]
                if record.error or record.cancelled or not record.finished:
                    raise RuntimeError("BF16 CPU repeat did not complete")
                ids = list(record.output_token_ids)
                rows.append({
                    "run": index, "output_token_ids": ids,
                    "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                })
            finally:
                engine.close_session(session.session_id)
    report = {
        "scope": "repeated unsplit BF16 CPU greedy sequence on a held-out prompt",
        "name": args.name, "model": str(args.model), "rows": rows,
        "all_repeats_equal": len({row["token_ids_sha256"] for row in rows}) == 1,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(probe(args))
    print(json.dumps({"all_repeats_equal": report["all_repeats_equal"],
                      "hashes": [row["token_ids_sha256"] for row in report["rows"]]}, indent=2))


if __name__ == "__main__":
    main()
