# SPDX-License-Identifier: Apache-2.0
"""Capture the exact post-final-norm BF16 CPU boundary for Spark head export."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from vllm_omni.edge.local.engine import LocalTextEngine
from vllm_omni.edge.local.plan import plan_text_session


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def capture(args: argparse.Namespace) -> dict:
    prompts = json.loads(args.prompts.read_text())
    plan = plan_text_session(
        str(args.model), max_model_len=4096, max_num_seqs=1,
        max_num_batched_tokens=2048, enforce_eager=True,
    )
    if not plan.admitted or plan.selected is None or plan.selected.device_id != "cpu":
        raise RuntimeError("BF16 CPU capture plan refused")
    if os.environ.get("VLLM_OMNI_SPARK_EXTERNAL_HEAD_SPEC"):
        raise ValueError("CPU norm capture cannot run with an external head")
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.raw.unlink(missing_ok=True)
    os.environ["VLLM_OMNI_SPARK_CPU_NORM_CAPTURE_BIN"] = str(args.raw.resolve())
    rows = []
    try:
        async with LocalTextEngine(plan) as engine:
            for item in prompts:
                session = engine.open_session()
                try:
                    request_id, stream = await engine.submit(
                        session, item["prompt"], max_tokens=args.max_tokens,
                        temperature=0.0, ignore_eos=True,
                    )
                    async for _ in stream:
                        pass
                    record = engine.records[request_id]
                    if (record.error or record.cancelled or not record.finished
                        or record.output_tokens != args.max_tokens):
                        raise RuntimeError(f"capture request {item['name']} failed")
                    ids = list(record.output_token_ids)
                    rows.append({
                        "name": item["name"], "prompt_tokens": record.prompt_tokens,
                        "output_tokens": len(ids),
                        "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    })
                finally:
                    engine.close_session(session.session_id)
    finally:
        os.environ.pop("VLLM_OMNI_SPARK_CPU_NORM_CAPTURE_BIN", None)
    raw = np.fromfile(args.raw, dtype=np.float32)
    if raw.size % 2048:
        raise RuntimeError("incomplete Spark norm capture")
    activations = raw.reshape(-1, 1, 2048)
    if len(activations) < len(prompts) * args.max_tokens:
        raise RuntimeError("fewer norm activations than generated tokens")
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.npz, x=activations)
    report = {
        "scope": "unsplit BF16 CPU post-final-norm activations before the full output projection",
        "model": str(args.model), "plan": plan.to_dict(),
        "prompt_file_sha256": sha256(args.prompts), "rows": rows,
        "raw_sha256": sha256(args.raw), "capture_sha256": sha256(args.npz),
        "activation_shape": list(activations.shape),
        "activation_min": float(activations.min()),
        "activation_max": float(activations.max()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(capture(args))
    print(json.dumps({k: report[k] for k in (
        "scope", "activation_shape", "activation_min", "activation_max", "capture_sha256"
    )}, indent=2))


if __name__ == "__main__":
    main()
