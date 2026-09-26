"""Compare one placed external ONNX stage with CPU source and quantized graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from vllm_omni.edge.hardware_probe import load_profile
from vllm_omni.edge.local.capabilities import enumerate_devices
from vllm_omni.edge.local.external.stage import ExternalStage, plan_external_stage
from vllm_omni.edge.local.manifest import build_graph_artifact


def _error(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(expected)),
        "max_abs": float(np.max(np.abs(difference))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--source-graph", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path)
    args = parser.parse_args()
    with np.load(args.inputs) as loaded:
        inputs = {name: loaded[name] for name in loaded.files}
    artifact = build_graph_artifact(
        args.graph,
        fmt="onnx:a16w8",
        opset=21,
        source_model="Qwen3.8-27B-FP8",
        component="qwen38_vision_merger_448",
        exporter="probe_qwen38_vision_merger_npu.py",
    )
    plan = plan_external_stage(
        artifact,
        enumerate_devices(load_profile()),
        require="npu:amd",
        min_fraction_on_target=0.10,
        worker_peak_rss_hint_bytes=1073741824,
    )
    if not plan.admitted:
        raise RuntimeError(plan.summary())
    source = ort.InferenceSession(str(args.source_graph), providers=["CPUExecutionProvider"])
    quantized = ort.InferenceSession(str(args.graph), providers=["CPUExecutionProvider"])
    source_result = source.run(None, inputs)[0]
    quantized_result = quantized.run(None, inputs)[0]
    with ExternalStage(plan) as stage:
        placement = stage.open(inputs, profile_dir=args.profile_dir)
        target_result, timing = stage.run(inputs)
        target = target_result[next(iter(target_result))]
        stats = stage.stats()
    if args.output_npz:
        np.savez(args.output_npz, output=target)
    report = {
        "source_graph": str(args.source_graph),
        "quantized_graph": str(args.graph),
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "output_shape": list(target.shape),
        "source_vs_quantized_cpu": _error(quantized_result, source_result),
        "quantized_cpu_vs_npu": _error(target, quantized_result),
        "source_vs_npu": _error(target, source_result),
        "placement": placement.to_dict(),
        "timing": timing.to_dict(),
        "stats": stats,
        "plan": plan.to_dict(),
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in {"stats", "plan"}}, indent=2))


if __name__ == "__main__":
    main()
