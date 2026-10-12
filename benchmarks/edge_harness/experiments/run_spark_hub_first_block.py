#!/usr/bin/env python3
"""Run a real first Spark decode block on S25; record C-tier evidence only.

This is deliberately not a model E2E runner: without on-target blocks 1–7,
full prompt prefill and 128 dependent generations, it must never call
``HubReplay.finish`` or mark a matrix cell complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.edge_harness.aihub_replay import HubArtifact, HubReplay


def digest_manifest(value: dict) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    return float(np.linalg.norm(reference.astype(np.float64) - actual.astype(np.float64))
                 / max(float(np.linalg.norm(reference.astype(np.float64))), 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--compile-submission", type=Path, required=True)
    ap.add_argument("--compile-report", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--token-int32", action="store_true",
                    help="satisfy a compiled --truncate_64bit_io target signature")
    args = ap.parse_args()
    source = json.loads(args.manifest.read_text())
    submission = json.loads(args.compile_submission.read_text())
    report = json.loads(args.compile_report.read_text())
    if source["block"] != 0 or report["status"] != "SUCCESS":
        raise ValueError("requires successful first-block target compile")
    if report["job_id"] != submission["job_id"] or submission["onnx_sha256"] != source["sha256"]:
        raise ValueError("compile evidence does not match the source artifact")
    target = report["target_model_id"]
    if not target:
        raise ValueError("compile success has no target model ID")
    compiled_identity = {
        "source_onnx_sha256": source["sha256"],
        "compile_job_id": report["job_id"],
        "compile_options": submission["options"],
        "target_model_id": target,
        "device": submission["device"],
    }
    artifact = HubArtifact(
        artifact_id=f"spark-block0-{report['job_id']}",
        target_model_id=target,
        sha256=digest_manifest(compiled_identity),
        checkpoint="XHToken/Spark-X2.5-1.7B",
        revision=source["checkpoint_revision"],
        layout_version=1,
        precision="QNN float16 from FP32 ONNX",
        target_name=submission["device"],
        runtime="qnn_dlc",
        shape_bucket=f"context-{source['context']}",
        hash_source="compiled_manifest",
    )
    with np.load(args.inputs, allow_pickle=False) as saved:
        inputs = {key: saved[key] for key in saved.files}
    if args.token_int32:
        token = int(inputs["token_id"].item())
        if token < 0 or token >= 131072:
            raise ValueError("token ID outside the pinned Spark vocabulary")
        inputs["token_id"] = inputs["token_id"].astype(np.int32)
    with np.load(args.reference, allow_pickle=False) as saved:
        reference = {key: saved[key] for key in saved.files}
    # The QNN compiler's --truncate_64bit_io maps this bounded vocabulary ID
    # to int32. Preserve the source int64 value and let the target signature
    # decide whether the SDK accepts it; never reinterpret an arbitrary ID.
    provenance = {}
    for key in inputs:
        if key == "token_id":
            provenance[key] = ("host:cast:host:tokenize:request:chat_prompt"
                               if args.token_int32 else
                               "host:tokenize:request:chat_prompt")
        elif key.startswith(("k_cache_", "v_cache_")):
            provenance[key] = "request:zero_initial_kv"
        elif key.startswith("mask_"):
            provenance[key] = "host:mask:request:chat_prompt"
        else:
            provenance[key] = "host:position:request:chat_prompt"
    with HubReplay(
        args.evidence, target_name=submission["device"], model="Spark-X2.5",
        checkpoint=artifact.checkpoint, revision=artifact.revision,
        request_id="spark-first-real-prompt-token-block0",
        stage_topology={"embedding_layers_0_1": []},
    ) as replay:
        result = replay.execute(
            "token000-block0", stage_id=0, stage_name="embedding_layers_0_1",
            artifact=artifact, inputs=inputs, input_provenance=provenance,
            options="--compute_unit npu",
        )
    source_order = ["hidden", "k_new_0", "v_new_0", "k_new_1", "v_new_1"]
    if set(result.outputs) == set(reference):
        name_map = {name: name for name in source_order}
    else:
        target_order = [f"output_{index}" for index in range(len(source_order))]
        if set(result.outputs) != set(target_order) or set(reference) != set(source_order):
            raise ValueError("target output signature differs from the source graph")
        name_map = dict(zip(target_order, source_order))
    comparison = {
        key: {
            "source_output": name_map[key],
            "relative_l2_vs_fp32_export": relative_l2(reference[name_map[key]], actual),
            "max_abs_vs_fp32_export": float(np.max(np.abs(reference[name_map[key]] - actual))),
            "finite": bool(np.isfinite(actual).all()),
        }
        for key, actual in result.outputs.items()
    }
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.evidence / "comparison.json").write_text(json.dumps({
        "depth": "C; first real-weight embedding plus two decoder layers only",
        "not_e2e": True,
        "inference_job_id": result.job_id,
        "compile_job_id": report["job_id"],
        "compiled_identity": compiled_identity,
        "comparison": comparison,
    }, indent=2) + "\n")
    print(json.dumps({"inference_job_id": result.job_id,
                      "comparison": comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
