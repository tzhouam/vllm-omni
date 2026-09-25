#!/usr/bin/env python3
"""Audit every layer's K/V output in a captured Qwen3-TTS AMD NPU rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "probe-report", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-capture-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--tensor-gate", type=float, default=0.01)
    args = parser.parse_args()
    if args.tensor_gate <= 0:
        parser.error("tensor gate must be positive")
    probe = json.loads(args.probe_report.read_text(encoding="utf-8-sig"))
    if (sha256(args.capture) != args.expected_capture_sha256
            or probe.get("capture", {}).get("sha256") != args.expected_capture_sha256
            or probe.get("artifact_sha256", {}).get("model") != args.expected_model_sha256
            or probe.get("npu_state_source") != "self"
            or probe.get("node_providers", {}).get("vitisai", 0) < 1):
        raise ValueError("capture, model, self-owned state or NPU placement changed")
    with np.load(args.capture, allow_pickle=False) as source:
        data = {name: np.ascontiguousarray(source[name]) for name in source.files}
    steps = sorted(int(name.removeprefix("npu_step").removesuffix("_out0"))
                   for name in data if name.startswith("npu_step") and name.endswith("_out0"))
    if not steps or steps != list(range(len(steps))):
        raise ValueError("captured steps are missing or out of order")
    rows = []
    for step in steps:
        layer_rows = []
        for layer in range(8):
            values = {}
            for kind, offset in (("key", 1 + 2 * layer), ("value", 2 + 2 * layer)):
                cpu = data[f"cpu_step{step}_out{offset}"]
                npu = data[f"npu_step{step}_out{offset}"]
                if (cpu.shape != npu.shape or cpu.dtype != npu.dtype
                        or not np.isfinite(cpu).all() or not np.isfinite(npu).all()):
                    raise ValueError(f"layer {layer} {kind} contract changed at step {step}")
                values[f"{kind}_relative_l2"] = relative_l2(cpu, npu)
                values[f"{kind}_max_abs"] = float(np.max(np.abs(
                    npu.astype(np.float64) - cpu.astype(np.float64))))
            layer_rows.append({"layer": layer, **values,
                               "tensor_gate_pass": max(values["key_relative_l2"],
                                                       values["value_relative_l2"])
                               <= args.tensor_gate})
        rows.append({"step": step, "start_frame": 95 + 2 * step,
                     "layers": layer_rows,
                     "first_gate_failure_layer": next((item["layer"] for item in layer_rows
                                                       if not item["tensor_gate_pass"]), None),
                     "max_all_layer_state_relative_l2": max(
                         max(item["key_relative_l2"], item["value_relative_l2"])
                         for item in layer_rows)})
    report = {
        "scope": "all-layer captured NPU-versus-ONNX-CPU rolling K/V numerical audit; not an operation attribution or complete TTS",
        "model_sha256": args.expected_model_sha256,
        "capture_sha256": args.expected_capture_sha256,
        "probe_report_sha256": sha256(args.probe_report),
        "tensor_gate_relative_l2": args.tensor_gate,
        "rows": rows,
        "status": "all_layer_output_audit_complete",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "first_gate_failure_layers": [row["first_gate_failure_layer"]
                                                    for row in rows]}))


if __name__ == "__main__":
    main()
