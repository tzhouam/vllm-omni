#!/usr/bin/env python3
"""Audit an output-instrumented Qwen3-TTS rolling-state NPU graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation", "probe", "capture", "baseline-probe",
                 "baseline-capture", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tensor-gate", type=float, default=0.01)
    args = parser.parse_args()
    preparation = json.loads(args.preparation.read_text(encoding="utf-8-sig"))
    probe = json.loads(args.probe.read_text(encoding="utf-8-sig"))
    baseline_probe = json.loads(args.baseline_probe.read_text(encoding="utf-8-sig"))
    if (preparation.get("status") != "diagnostic_cpu_output_parity_pass"
            or probe.get("artifact_sha256", {}).get("model") != preparation.get("candidate_sha256")
            or baseline_probe.get("artifact_sha256", {}).get("model") != preparation.get("source_sha256")
            or probe.get("artifact_sha256", {}).get("fixture") != preparation.get("fixture_sha256")
            or baseline_probe.get("artifact_sha256", {}).get("fixture") != preparation.get("fixture_sha256")
            or probe.get("npu_state_source") != "self"
            or baseline_probe.get("npu_state_source") != "self"
            or probe.get("node_providers", {}).get("vitisai", 0) < 2
            or baseline_probe.get("node_providers", {}).get("vitisai", 0) < 2
            or probe.get("node_providers") != baseline_probe.get("node_providers")
            or probe.get("capture", {}).get("sha256") != sha256(args.capture)
            or baseline_probe.get("capture", {}).get("sha256") != sha256(args.baseline_capture)):
        raise ValueError("model, fixture, placement, state, or capture provenance changed")
    with np.load(args.capture, allow_pickle=False) as source:
        candidate = {name: np.ascontiguousarray(source[name]) for name in source.files}
    with np.load(args.baseline_capture, allow_pickle=False) as source:
        baseline = {name: np.ascontiguousarray(source[name]) for name in source.files}
    names = preparation["checkpoint_output_names"]
    if len(names) != 8 or len(set(names)) != 8:
        raise ValueError("eight-layer checkpoint output contract changed")
    rows = []
    for step in range(2):
        layers = []
        for layer, name in enumerate(names):
            index = 17 + layer
            cpu = candidate[f"cpu_step{step}_out{index}"]
            npu = candidate[f"npu_step{step}_out{index}"]
            if (cpu.shape != (1, 2, 512) or npu.shape != cpu.shape
                    or not np.isfinite(cpu).all() or not np.isfinite(npu).all()):
                raise ValueError("checkpoint shape changed")
            layers.append({"layer": layer, "name": name,
                           "hidden_relative_l2": relative_l2(cpu, npu)})
        source_output_errors = []
        baseline_npu_output_errors = []
        for index in range(17):
            cpu = candidate[f"cpu_step{step}_out{index}"]
            npu = candidate[f"npu_step{step}_out{index}"]
            baseline_cpu = baseline[f"cpu_step{step}_out{index}"]
            baseline_npu = baseline[f"npu_step{step}_out{index}"]
            if (cpu.shape != baseline_cpu.shape or npu.shape != baseline_npu.shape
                    or not all(np.isfinite(value).all() for value in
                               (cpu, npu, baseline_cpu, baseline_npu))):
                raise ValueError("source output shape or finiteness changed")
            source_output_errors.append(relative_l2(baseline_cpu, cpu))
            baseline_npu_output_errors.append(relative_l2(baseline_npu, npu))
        if max(source_output_errors) != 0 or max(baseline_npu_output_errors) != 0:
            raise ValueError("instrumentation changed an original CPU or NPU output")
        rows.append({"step": step, "start_frame": 95 + 2 * step,
                     "layers": layers,
                     "first_hidden_gate_failure_layer": next(
                         (item["layer"] for item in layers
                          if item["hidden_relative_l2"] > args.tensor_gate), None),
                     "source_cpu_output_max_relative_l2": max(source_output_errors),
                     "baseline_npu_output_max_relative_l2": max(baseline_npu_output_errors),
                     "final_hidden_relative_l2": relative_l2(
                         candidate[f"cpu_step{step}_out0"],
                         candidate[f"npu_step{step}_out0"]),
                     "baseline_final_hidden_relative_l2": relative_l2(
                         baseline[f"cpu_step{step}_out0"],
                         baseline[f"npu_step{step}_out0"])})
    report = {
        "scope": "diagnostic output-instrumented eight-layer NPU graph; attribution to the baseline graph requires matching provider partition and final-output behavior",
        "preparation_sha256": sha256(args.preparation),
        "probe_sha256": sha256(args.probe),
        "baseline_probe_sha256": sha256(args.baseline_probe),
        "capture_sha256": sha256(args.capture),
        "baseline_capture_sha256": sha256(args.baseline_capture),
        "tensor_gate_relative_l2": args.tensor_gate,
        "provider_nodes": probe["node_providers"],
        "baseline_provider_nodes": baseline_probe["node_providers"],
        "rows": rows,
        "status": "diagnostic_checkpoint_audit_complete",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "first_hidden_gate_failure_layers": [
                          row["first_hidden_gate_failure_layer"] for row in rows],
                      "baseline_npu_output_max_relative_l2": [
                          row["baseline_npu_output_max_relative_l2"] for row in rows]}))


if __name__ == "__main__":
    main()
