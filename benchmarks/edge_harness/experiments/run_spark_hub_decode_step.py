#!/usr/bin/env python3
"""Replay one real-weight 28-layer Spark decode step through eight S25 jobs.

This component experiment begins with one real prompt token and empty KV. It
does not claim a complete prompt prefill, 128 generated tokens, or model E2E.
Every neural stage, including the embedding and tied head, runs on the named
AI Hub target; the host only supplies masks/rotary angles and routes hidden
state between dependent jobs. The generic runner journals each job and tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.edge_harness.aihub_replay import HubArtifact, HubReplay


REVISION = "14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8"
CHECKPOINT = "XHToken/Spark-X2.5-1.7B"
TARGET = "Samsung Galaxy S25"


def manifest_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def angles(position: int, dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(0, dim, 2, dtype=np.float32)
    inv = 1.0 / np.power(np.float32(theta), index / dim)
    phase = position * inv
    cos = np.concatenate((np.cos(phase), np.cos(phase))).reshape(1, 1, 1, dim)
    sin = np.concatenate((np.sin(phase), np.sin(phase))).reshape(1, 1, 1, dim)
    return cos.astype(np.float32), sin.astype(np.float32)


def get_spec(model: object) -> dict[str, object]:
    groups = model.input_spec
    if len(groups) != 1:
        raise ValueError("expected one compiled target input signature")
    return {spec.name: spec for specs in groups.values() for spec in specs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports", type=Path, required=True)
    ap.add_argument("--compiles", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--token-input", type=Path, required=True,
                    help="NPZ containing token_id from the pinned chat prompt")
    ap.add_argument("--reference-logits", type=Path,
                    help="optional same-checkpoint HF BF16 one-token logits NPZ")
    args = ap.parse_args()
    with np.load(args.token_input, allow_pickle=False) as archive:
        token_id = int(archive["token_id"].item())
    if token_id < 0 or token_id >= 131072:
        raise ValueError("token ID is outside the pinned Spark vocabulary")
    records = []
    for block in range(8):
        matches = list(args.exports.glob(f"spark_block{block}_*_ctx128.json"))
        if len(matches) != 1:
            raise ValueError(f"missing unique source manifest for block {block}")
        source = json.loads(matches[0].read_text())
        attempt = "trunc64" if block == 0 else "base"
        submission = json.loads((args.compiles / f"block{block}_{attempt}_compile_submission.json").read_text())
        report = json.loads((args.compiles / f"block{block}_{attempt}_compile_report.json").read_text())
        if (source["checkpoint_revision"] != REVISION or source["context"] != 128
                or source["sha256"] != submission["onnx_sha256"]
                or report["job_id"] != submission["job_id"]
                or report["status"] != "SUCCESS" or not report["target_model_id"]
                or submission["device"] != TARGET):
            raise ValueError(f"block {block} does not have a matching successful compile")
        identity = {
            "source_onnx_sha256": source["sha256"],
            "compile_job_id": report["job_id"],
            "compile_options": submission["options"],
            "target_model_id": report["target_model_id"],
            "device": TARGET,
        }
        records.append((source, report, identity))
    from qai_hub import Client
    client = Client()
    topology = {f"block{index}": ([f"block{index - 1}"] if index else [])
                for index in range(8)}
    cos_sw, sin_sw = angles(0, 256, 10000.0)
    cos_full, sin_full = angles(0, 64, 5000000.0)
    shared = {"cos_sw": cos_sw, "sin_sw": sin_sw,
              "cos_full": cos_full, "sin_full": sin_full}
    previous_hidden = None
    with HubReplay(
        args.evidence, target_name=TARGET, model="Spark-X2.5",
        checkpoint=CHECKPOINT, revision=REVISION,
        request_id=f"spark-first-chat-token-{token_id}-all28-ctx128",
        stage_topology=topology, client=client,
    ) as replay:
        for block, (source, report, identity) in enumerate(records):
            target_id = report["target_model_id"]
            spec = get_spec(client.get_model(target_id))
            artifact = HubArtifact(
                artifact_id=f"spark-block{block}-{report['job_id']}",
                target_model_id=target_id,
                sha256=manifest_hash(identity),
                checkpoint=CHECKPOINT, revision=REVISION,
                layout_version=1, precision="QNN float16 from FP32 ONNX",
                target_name=TARGET, runtime="qnn_dlc", shape_bucket="context-128",
                hash_source="compiled_manifest",
            )
            inputs = {}
            provenance = {}
            for name, signature in spec.items():
                shape = tuple(signature.shape)
                if name == "token_id":
                    value = np.asarray([[token_id]], dtype=np.int32)
                    source_lineage = "host:cast:host:tokenize:request:chat_prompt"
                elif name == "x":
                    if previous_hidden is None:
                        raise ValueError("later block lacks predecessor hidden output")
                    value = previous_hidden
                    source_lineage = f"step:token000-block{block - 1}:output_0"
                elif name in shared:
                    value = shared[name]
                    source_lineage = "host:position:request:chat_prompt"
                elif name.startswith("mask_"):
                    value = np.full(shape, -np.inf, dtype=np.float32)
                    value[..., -1] = 0
                    source_lineage = "host:mask:request:chat_prompt"
                elif name.startswith(("k_cache_", "v_cache_")):
                    value = np.zeros(shape, dtype=np.float32)
                    source_lineage = "request:zero_initial_kv"
                else:
                    raise ValueError(f"unexpected compiled input {name}")
                if value.shape != shape or str(value.dtype) != signature.dtype:
                    raise ValueError(f"compiled input contract mismatch for {name}")
                inputs[name] = value
                provenance[name] = source_lineage
            result = replay.execute(
                f"token000-block{block}", stage_id=block,
                stage_name=f"block{block}", artifact=artifact,
                inputs=inputs, input_provenance=provenance,
                options="--compute_unit npu",
            )
            if "output_0" not in result.outputs:
                raise ValueError(f"block {block} has no primary target output")
            previous_hidden = result.outputs["output_0"]
            if not np.isfinite(previous_hidden).all():
                raise ValueError(f"block {block} returned non-finite hidden/logits")
            print(f"block {block}: inference {result.job_id} complete", flush=True)
    logits = previous_hidden
    if logits is None or logits.shape != (1, 131072):
        raise ValueError("last target block did not return full-vocabulary logits")
    quality = {"top1_token_id": int(np.argmax(logits)), "finite": True}
    if args.reference_logits:
        with np.load(args.reference_logits, allow_pickle=False) as archive:
            reference = archive["logits"]
        if reference.shape != logits.shape:
            raise ValueError("HF reference logits shape mismatch")
        norm = float(np.linalg.norm(reference.astype(np.float64)))
        quality.update({
            "reference_top1_token_id": int(np.argmax(reference)),
            "top1_match": bool(np.argmax(reference) == np.argmax(logits)),
            "logits_relative_l2": float(np.linalg.norm(
                reference.astype(np.float64) - logits.astype(np.float64)) / max(norm, 1e-12)),
        })
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.evidence / "decode_step_quality.json").write_text(json.dumps({
        "depth": "C: one 28-layer decode step from the first real prompt token",
        "not_e2e": True,
        "not_128_generated_tokens": True,
        "context_capacity": 128,
        "token_id": token_id,
        "job_ids": [row["job_id"] for row in json.loads((args.evidence / "run.json").read_text())["steps"]],
        "quality": quality,
    }, indent=2) + "\n")
    print(json.dumps(quality, indent=2), flush=True)


if __name__ == "__main__":
    main()
