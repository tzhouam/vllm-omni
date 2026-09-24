#!/usr/bin/env python3
"""Verify Spark BF16 can generate again after an in-flight Omni cancellation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from vllm_omni.edge.local.engine import LocalTextEngine
from vllm_omni.edge.local.manifest import runtime_versions
from vllm_omni.edge.local.plan import plan_text_session
from vllm_omni.edge.local.prompts import acceptance_prompts


def token_digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


async def run(model: Path) -> dict:
    plan = plan_text_session(
        str(model), max_model_len=4096, max_num_seqs=1,
        max_num_batched_tokens=2048, enforce_eager=True,
    )
    if not plan.admitted or plan.selected is None or plan.selected.device_id != "cpu":
        raise RuntimeError("pinned BF16 CPU plan was not admitted")
    name, prompt = acceptance_prompts()[0]
    requests = []
    async with LocalTextEngine(plan) as engine:
        session = engine.open_session()

        async def generate(label: str, active_session: object) -> list[int]:
            rid, stream = await engine.submit(
                active_session, prompt, max_tokens=128, temperature=0.0, ignore_eos=True,
            )
            async for _ in stream:
                pass
            record = engine.records[rid]
            if record.error or record.cancelled or not record.finished or record.output_tokens != 128:
                raise RuntimeError(f"{label} did not finish the specified text workload")
            ids = list(record.output_token_ids)
            requests.append({"label": label, "request_id": rid,
                             "session_id": active_session.session_id,
                             "epoch": record.epoch, "output_tokens": len(ids),
                             "token_ids_sha256": token_digest(ids),
                             "ttft_s": record.ttft_s,
                             "stream": stream.stats()})
            return ids

        baseline = await generate("before_cancel", session)
        rid, stream = await engine.submit(
            session, prompt, max_tokens=4096, temperature=0.0, ignore_eos=True,
        )
        token_events = 0
        async for event in stream:
            if event.kind == "token":
                token_events += 1
            if token_events >= 8:
                break
        cancel = await engine.cancel(rid)
        late_events = []
        while True:
            try:
                event = await asyncio.wait_for(stream.get(), timeout=1.0)
            except asyncio.TimeoutError:
                break
            if event is None:
                break
            late_events.append(event.to_dict())
        cancel["token_events_before_cancel"] = token_events
        cancel["late_events"] = late_events
        try:
            stale_rid, _ = await engine.submit(
                session, prompt, max_tokens=128, temperature=0.0, ignore_eos=True,
            )
        except RuntimeError as exc:
            stale_error = str(exc)
            stale_rejected = "stale" in stale_error and "open a new session" in stale_error
        else:
            await engine.cancel(stale_rid)
            stale_error = "old session was unexpectedly accepted"
            stale_rejected = False
        engine.close_session(session.session_id)
        fresh_session = engine.open_session()
        new_session = await generate("new_session_after_cancel", fresh_session)
        engine.close_session(fresh_session.session_id)
        placement = engine.report_placement()
        usage = engine.report_usage()
        load_s = engine.load_seconds
    passed = (bool(cancel.get("aborted_in_backend"))
              and cancel.get("inflight_after") == 0
              and not late_events
              and stale_rejected
              and baseline == new_session)
    return {
        "scope": "Spark 1.7B BF16 128-token WSL CPU text restart after cancellation",
        "status": "restart_after_cancel_pass" if passed else "restart_after_cancel_fail",
        "prompt_name": name,
        "runtime": runtime_versions().to_dict(),
        "plan": plan.to_dict(),
        "placement": placement,
        "usage": usage,
        "load_s_not_request_timing": load_s,
        "requests": requests,
        "cancel": cancel,
        "stale_handle_rejected": stale_rejected,
        "stale_handle_error": stale_error,
        "exact_token_sequence_agreement": baseline == new_session,
        "limits": [
            "One prompt, 128 output tokens, one engine instance and one cancellation depth.",
            "The retired session is rejected by design; restart uses a fresh session handle.",
            "No concurrent traffic or sustained power/thermal behavior was tested.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(run(args.model))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "cancel_before": report["cancel"]["token_events_before_cancel"],
                      "late_events": len(report["cancel"]["late_events"]),
                      "stale_handle_rejected": report["stale_handle_rejected"],
                      "exact": report["exact_token_sequence_agreement"]}, indent=2))
    if report["status"] != "restart_after_cancel_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
