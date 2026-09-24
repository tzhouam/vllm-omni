# SPDX-License-Identifier: Apache-2.0
"""Profile captured Spark output-head activations across Omni's WSL/Windows boundary.

This is a component handoff probe. It does not run vLLM generation or prove a
useful split plan. The graph and fixture hashes pin the previously qualified
artifact; results include the planner's budget and observed worker peak.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from vllm_omni.edge.hardware_probe import load_profile
from vllm_omni.edge.local.capabilities import FORMAT_ONNX_A16W8, enumerate_devices
from vllm_omni.edge.local.external.stage import ExternalStage, plan_external_stage
from vllm_omni.edge.local.manifest import build_graph_artifact


GRAPH_SHA = "e2ae9c3fd7071e8ff628f209e6140c8a023bfaad59814b80a936c6b263f33eaf"
FIXTURE_SHA = "cfa2ed8439f8fe01730319667bfd01abf2e39cf5851490346a0331304cf63194"
REVISION = "448e61eb392c00f2c403185c5b56d5e0665bfaab"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def nearest(samples: list[float], q: float) -> float:
    return sorted(samples)[math.ceil(q * len(samples)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--worker-peak-rss-hint-bytes", type=int, default=None)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[3]
    stage_source = Path(inspect.getfile(plan_external_stage)).resolve()
    if stage_source != checkout / "vllm_omni/edge/local/external/stage.py":
        raise RuntimeError(
            f"loaded Omni stage from {stage_source}, not this checkout; "
            "run with PYTHONPATH=. from the repository root"
        )
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if sha256(args.graph) != GRAPH_SHA or sha256(args.fixture) != FIXTURE_SHA:
        raise ValueError("graph or fixture differs from the pinned qualified component")
    with np.load(args.fixture, allow_pickle=False) as archive:
        activations = archive["x"].copy()
        reference = archive["hidden_out"].copy()
    if activations.shape != (4, 1, 1, 2048) or reference.shape != (4, 1, 131072):
        raise ValueError("unexpected fixture shape")

    artifact = build_graph_artifact(
        args.graph,
        fmt=FORMAT_ONNX_A16W8,
        opset=21,
        source_model="XHToken/Spark-X2.5-1.7B",
        source_revision=REVISION,
        component="spark_output_head",
        exporter="probe_spark_amd_npu_lm_head.py --composite-with-norm",
        calibration={"fixture_sha256": FIXTURE_SHA, "samples": 4},
        parity={"previous_top1_matches": 4, "previous_snr_db_min": 55.43},
    )
    plan = plan_external_stage(
        artifact,
        enumerate_devices(load_profile(use_torch=False)),
        require="npu:amd",
        # This composite has one fused NPU partition and seven CPU nodes.
        min_fraction_on_target=0.125,
        worker_peak_rss_hint_bytes=args.worker_peak_rss_hint_bytes,
    )
    if not plan.admitted:
        raise RuntimeError(plan.summary())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    started = time.time()
    with ExternalStage(plan) as stage:
        placement = stage.open({"x": activations[0]}, profile_dir=args.profile_dir)
        if placement.target_nodes < 1 or placement.fraction_on_target is None:
            raise RuntimeError("NPU execution was not verified")
        for index in range(args.repeats):
            case = index % len(activations)
            outputs, timing = stage.run({"x": activations[case]})
            logits = outputs["logits_concatenated"]
            expected = reference[case]
            error = np.linalg.norm((logits - expected).astype(np.float64))
            scale = np.linalg.norm(expected.astype(np.float64))
            relative_l2 = float(error / scale)
            snr_db = float(20 * np.log10(scale / error))
            samples.append({
                "case": case,
                **timing.to_dict(),
                "top1_match": bool(np.argmax(logits) == np.argmax(expected)),
                "relative_l2": relative_l2,
                "snr_db": snr_db,
                "finite": bool(np.isfinite(logits).all()),
            })
        worker_stats = stage.stats()

    metrics = {}
    for key in ("worker_s", "transport_s", "round_trip_s"):
        values = [sample[key] for sample in samples]
        metrics[key] = {"p50": nearest(values, 0.50), "p95": nearest(values, 0.95)}
    report = {
        "schema_version": 1,
        "scope": "captured_real_activation_component_handoff_only; no vLLM generation",
        "source_model": artifact.source_model,
        "source_revision": REVISION,
        "graph_sha256": GRAPH_SHA,
        "fixture_sha256": FIXTURE_SHA,
        "platform": platform.platform(),
        "omni_stage_source": str(stage_source),
        "started_unix": started,
        "ended_unix": time.time(),
        "concurrency": 1,
        "warmup": "one profiled stage-open inference",
        "repeats": args.repeats,
        "placement": placement.to_dict(),
        "planner_budget_bytes": plan.budget_bytes,
        "worker_peak_rss_hint_bytes": args.worker_peak_rss_hint_bytes,
        "worker_peak_rss_bytes": worker_stats.get("peak_rss_bytes"),
        "planner_bounds_observed_worker_peak": plan.budget_bytes >= worker_stats.get("peak_rss_bytes", 0),
        "worker_stats": worker_stats,
        "samples": samples,
        "metrics": metrics,
        "power_thermal": "not measured",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"metrics": metrics, "top1": sum(s["top1_match"] for s in samples),
                      "planner_bounds_observed_worker_peak": report["planner_bounds_observed_worker_peak"]}, indent=2))


if __name__ == "__main__":
    main()
