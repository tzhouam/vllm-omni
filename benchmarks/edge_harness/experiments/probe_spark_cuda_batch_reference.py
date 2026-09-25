#!/usr/bin/env python3
"""Compare greedy Spark CUDA tokens by batch slot against a standalone vLLM reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def top_logprobs(candidate: object) -> list[dict[str, float | int]]:
    if not candidate:
        return []
    return [
        {"token_id": int(token), "logprob": float(value.logprob)}
        for token, value in sorted(
            candidate.items(), key=lambda pair: pair[1].logprob, reverse=True
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bands", nargs="+", choices=("short", "medium", "long"),
                        default=("medium", "long"))
    args = parser.parse_args()

    prior = json.loads(args.prior_report.read_text(encoding="utf-8"))
    manifest = prior["plan"]["manifest"]
    if (args.model.resolve() != Path(manifest["model_dir"]).resolve()
            or prior["plan"]["backend"] != "vllm:cuda"):
        raise ValueError("prior CUDA artifact and requested model differ")
    weight_sha = {name: sha256(args.model / name) for name in manifest["weight_files"]}
    aggregate = hashlib.sha256()
    for name, digest in weight_sha.items():
        aggregate.update(name.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
    if aggregate.hexdigest() != manifest["weight_sha256"]:
        raise ValueError("Spark weight bytes differ from the prior profile")
    config_sha = sha256(args.model / "config.json")
    if config_sha != manifest["config_sha256"]:
        raise ValueError("Spark config bytes differ from the prior profile")

    # Importing Omni registers Spark with vLLM; the reference below uses vLLM's
    # scheduler and model executor directly, without LocalTextEngine/AsyncOmni.
    import vllm_omni  # noqa: F401
    from vllm import LLM, SamplingParams

    engine_kwargs = dict(prior["plan"]["engine_kwargs"])
    started = time.perf_counter()
    llm = LLM(model=str(args.model.resolve()), trust_remote_code=False,
              disable_log_stats=True, **engine_kwargs)
    load_s = time.perf_counter() - started
    params = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True,
                            logprobs=5)
    prompts = {
        name: ("The garden has trees, flowers, and a small pond. " * n)
              + "\nDescribe the garden in detail."
        for name, n in (("short", 4), ("medium", 40), ("long", 160))
    }
    report = {
        "scope": "standalone vLLM Spark 4B BF16 greedy batch-slot reference; no Omni request path",
        "model": str(args.model.resolve()),
        "weight_sha256": weight_sha,
        "config_sha256": config_sha,
        "prior_report_sha256": sha256(args.prior_report),
        "engine_kwargs": engine_kwargs,
        "python": sys.executable,
        "platform": platform.platform(),
        "gate_reduction_override": os.environ.get("VLLM_OMNI_SPARK_GATE_REDUCTION", "auto"),
        "load_s": load_s,
        "cases": [],
    }
    for band in args.bands:
        baseline = None
        for batch in (1, 2, 4):
            begun = time.perf_counter()
            outputs = llm.generate([prompts[band]] * batch, params, use_tqdm=False)
            wall_s = time.perf_counter() - begun
            slots = []
            for index, output in enumerate(outputs):
                completion = output.outputs[0]
                ids = list(completion.token_ids)
                if len(ids) != 128:
                    raise ValueError(f"{band} batch {batch} slot {index}: only {len(ids)} tokens")
                if baseline is None:
                    baseline = ids
                difference = first_difference(baseline, ids)
                decision = None
                if difference is not None:
                    logprobs = completion.logprobs or []
                    decision = {
                        "index": difference,
                        "baseline_token": baseline[difference],
                        "slot_token": ids[difference],
                        "slot_top_logprobs": top_logprobs(logprobs[difference]),
                    }
                slots.append({"slot": index, "token_ids": ids,
                              "first_difference_from_batch1": decision,
                              "probe_logprobs": top_logprobs((completion.logprobs or [])[{
                                  "short": 107, "medium": 84, "long": 89
                              }[band]])})
            report["cases"].append({"band": band, "batch": batch,
                                    "wall_s": wall_s, "slots": slots})
            print(f"{band} batch={batch}: differences="
                  f"{[x['first_difference_from_batch1']['index'] if x['first_difference_from_batch1'] else None for x in slots]}",
                  flush=True)
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
