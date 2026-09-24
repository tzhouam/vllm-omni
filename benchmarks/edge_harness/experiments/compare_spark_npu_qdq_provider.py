# SPDX-License-Identifier: Apache-2.0
"""Compare the same Spark output-head QDQ graph on ORT CPU and AMD NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from vllm_omni.edge.hardware_probe import load_profile
from vllm_omni.edge.local.capabilities import FORMAT_ONNX_A16W8, enumerate_devices
from vllm_omni.edge.local.external.stage import ExternalStage, plan_external_stage
from vllm_omni.edge.local.manifest import build_graph_artifact


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[2, 7, 8, 16, 32, 64, 127])
    parser.add_argument(
        "--expected-graph-sha256", type=str,
        default="e2ae9c3fd7071e8ff628f209e6140c8a023bfaad59814b80a936c6b263f33eaf",
    )
    args = parser.parse_args()

    prior = json.loads(args.capture_report.read_text())
    recorded = prior.get("captured_activations", {})
    if recorded.get("sha256") != sha256(args.capture):
        raise ValueError("captured live activations changed")
    with np.load(args.capture, allow_pickle=False) as archive:
        activations = archive["x"].copy()
    if activations.ndim != 4 or activations.shape[1:] != (1, 1, 2048):
        raise ValueError("unexpected Spark activation shape")
    if min(args.indices) < 0 or max(args.indices) >= len(activations):
        raise ValueError("requested index is outside capture")

    artifact = build_graph_artifact(
        args.graph, fmt=FORMAT_ONNX_A16W8, opset=21,
        source_model="XHToken/Spark-X2.5-1.7B",
        source_revision="448e61eb392c00f2c403185c5b56d5e0665bfaab",
        component="spark_output_head",
        exporter="probe_spark_amd_npu_lm_head.py --composite-with-norm",
    )
    if artifact.sha256 != args.expected_graph_sha256:
        raise ValueError("Spark graph differs from the requested artifact")
    plan = plan_external_stage(
        artifact, enumerate_devices(load_profile(use_torch=False)),
        require="npu:amd", min_fraction_on_target=0.125,
        worker_peak_rss_hint_bytes=3772710912,
    )
    if not plan.admitted:
        raise RuntimeError(plan.summary())
    cpu = ort.InferenceSession(str(args.graph), providers=["CPUExecutionProvider"])
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ExternalStage(plan) as stage:
        placement = stage.open({"x": activations[args.indices[0]]}, profile_dir=args.profile_dir)
        if placement.target_nodes < 1 or placement.ep != "vitisai":
            raise RuntimeError("VitisAI placement was not verified")
        for index in args.indices:
            x = activations[index]
            expected = cpu.run(None, {"x": x})[0]
            outputs, timing = stage.run({"x": x})
            actual = outputs["logits_concatenated"]
            rows.append({
                "index": index,
                "activation_min": float(x.min()),
                "activation_max": float(x.max()),
                "qdq_cpu_top1": int(expected.argmax()),
                "vitisai_top1": int(actual.argmax()),
                "vitisai_vs_qdq_cpu_relative_l2": float(
                    np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                ),
                **timing.to_dict(),
            })
        stats = stage.stats()
    report = {
        "graph_sha256": artifact.sha256,
        "capture_sha256": recorded["sha256"],
        "placement": placement.to_dict(),
        "worker_stats": stats,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"target_nodes": placement.target_nodes, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
