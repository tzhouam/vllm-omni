#!/usr/bin/env python3
"""Replay a real HX370 NPU Cosmos Conv13 output through the CPU encoder suffix."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


SOURCE_SHA = "fee79864e393df35475ba243e14b4995c5651aa33e7e182cd1fda726425668b7"
WEIGHTS_SHA = "b9bc3411c10f05daec2ff977698ee100979ef756ed7f0d685275403517558b58"
FIXTURES_SHA = "dcbaf1391ca914d4bbccc1cb384daf05666e771d9cdae90c4f069ec8fcf05368"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_l2(reference, actual) -> float:
    import numpy as np

    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return float(np.linalg.norm(difference) / max(np.linalg.norm(reference.astype(np.float64)), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fixtures", "assembled", "assembled-report", "suffix", "latent-output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import onnx
    import onnxruntime as ort

    if (sha256(args.source) != SOURCE_SHA
            or sha256(args.source.with_name(args.source.name + ".data")) != WEIGHTS_SHA
            or sha256(args.fixtures) != FIXTURES_SHA):
        raise ValueError("real Cosmos source, weights or pixels changed")
    assembled_report = json.loads(args.assembled_report.read_text(encoding="utf-8-sig"))
    if (assembled_report["status"] != "component_npu_numeric_pass"
            or assembled_report["assembled_outputs_sha256"] != sha256(args.assembled)):
        raise ValueError("AMD NPU six-frame boundary output changed")
    source = onnx.load(str(args.source), load_external_data=True)
    suffix = onnx.utils.Extractor(source).extract_model(["pixels", "conv2d_13"], ["latent"])
    if any(node.name == "node_conv2d_13" for node in suffix.graph.node):
        raise ValueError("encoder suffix still computes the replaced convolution")
    onnx.checker.check_model(suffix)
    args.suffix.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(suffix, args.suffix)
    full_cpu = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    suffix_cpu = ort.InferenceSession(str(args.suffix), providers=["CPUExecutionProvider"])
    if {item.name for item in suffix_cpu.get_inputs()} != {"pixels", "conv2d_13"}:
        raise ValueError("encoder suffix input contract changed")
    metrics = {}
    latents = {}
    with np.load(args.fixtures, allow_pickle=False) as fixture, np.load(args.assembled, allow_pickle=False) as boundary:
        for case in ("ramp", "pattern"):
            pixels = np.ascontiguousarray(fixture[f"{case}_pixels"])
            cpu_boundary = np.ascontiguousarray(boundary[f"{case}_cpu"])
            npu_boundary = np.ascontiguousarray(boundary[f"{case}_npu"])
            if pixels.shape != (6, 3, 256, 256) or cpu_boundary.shape != npu_boundary.shape != (6, 256, 64, 64):
                raise ValueError("encoder replay tensor shape changed")
            reference = full_cpu.run(None, {"pixels": pixels})[0]
            cpu_control = suffix_cpu.run(None, {"pixels": pixels, "conv2d_13": cpu_boundary})[0]
            started = time.perf_counter()
            npu_injected = suffix_cpu.run(None, {"pixels": pixels, "conv2d_13": npu_boundary})[0]
            suffix_s = time.perf_counter() - started
            if reference.shape != cpu_control.shape or reference.shape != npu_injected.shape or not np.isfinite(npu_injected).all():
                raise ValueError("encoder suffix latent contract failed")
            metrics[case] = {
                "cpu_suffix_vs_full_relative_l2": relative_l2(reference, cpu_control),
                "npu_boundary_vs_cpu_boundary_relative_l2": relative_l2(cpu_boundary, npu_boundary),
                "npu_injected_latent_vs_full_relative_l2": relative_l2(reference, npu_injected),
                "npu_injected_latent_cosine": float(
                    np.sum(reference.astype(np.float64) * npu_injected.astype(np.float64))
                    / max(np.linalg.norm(reference.astype(np.float64)) * np.linalg.norm(npu_injected.astype(np.float64)), 1e-12)),
                "cpu_suffix_one_inference_s": suffix_s,
            }
            latents[f"{case}_reference"] = reference
            latents[f"{case}_npu_injected"] = npu_injected
    if any(value["cpu_suffix_vs_full_relative_l2"] > 1e-5 for value in metrics.values()):
        raise ValueError("CPU encoder suffix did not reproduce the source")
    args.latent_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.latent_output, **latents)
    report = {
        "scope": "real Cosmos encoder CPU suffix with one HX370 NPU Conv13 output injected; synthetic six-frame fixtures, no full policy",
        "source_sha256": SOURCE_SHA, "weights_sha256": WEIGHTS_SHA,
        "fixtures_sha256": FIXTURES_SHA,
        "npu_boundary_sha256": assembled_report["assembled_outputs_sha256"],
        "suffix_sha256": sha256(args.suffix), "suffix_bytes": args.suffix.stat().st_size,
        "latent_output_sha256": sha256(args.latent_output),
        "onnxruntime": ort.__version__, "metrics": metrics,
        "status": "encoder_suffix_numeric_reported",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
