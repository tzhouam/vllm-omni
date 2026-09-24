#!/usr/bin/env python3
"""Measure waveform impact of an AMD NPU prefix handed to the original CPU decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(reference: np.ndarray, actual: np.ndarray) -> dict:
    if reference.shape != actual.shape:
        raise ValueError("waveform shape changed")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    relative_l2 = float(np.linalg.norm(delta) / max(
        np.linalg.norm(reference.astype(np.float64)), 1e-12))
    return {
        "finite": bool(np.isfinite(actual).all()),
        "relative_l2": relative_l2,
        "snr_db": float(-20 * np.log10(max(relative_l2, 1e-12))),
        "max_absolute_error": float(np.max(np.abs(delta))),
        "saturation_fraction": float(np.mean(np.abs(actual) >= 0.999)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("decoder", "paired-activations", "reference", "report", "waves"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("decoder", "paired-activations", "reference"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    args = parser.parse_args()
    import onnxruntime as ort

    hashes = {
        "decoder": sha256(args.decoder),
        "paired_activations": sha256(args.paired_activations),
        "reference": sha256(args.reference),
    }
    for name, actual in hashes.items():
        expected = getattr(args, f"expected_{name}_sha256")
        if actual != expected:
            raise ValueError(f"{name} hash changed")
    with np.load(args.paired_activations, allow_pickle=False) as pair:
        if set(pair.files) != {"cpu", "npu"}:
            raise ValueError("paired activation schema changed")
        cpu_activation = np.ascontiguousarray(pair["cpu"])
        npu_activation = np.ascontiguousarray(pair["npu"])
    with np.load(args.reference, allow_pickle=False) as source:
        if source.files != ["wav"]:
            raise ValueError("reference schema changed")
        reference = np.asarray(source["wav"])
    if cpu_activation.shape != (1, 1024, 74) or npu_activation.shape != cpu_activation.shape:
        raise ValueError("boundary activation shape changed")
    if cpu_activation.dtype != np.float32 or npu_activation.dtype != np.float32:
        raise ValueError("boundary activation dtype changed")
    if reference.shape != (1, 3840) or reference.dtype != np.float32:
        raise ValueError("reference waveform contract changed")

    started = time.perf_counter()
    decoder = ort.InferenceSession(str(args.decoder), providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - started
    inputs = decoder.get_inputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 1024, 74]:
        raise ValueError("decoder input contract changed")
    input_name = inputs[0].name
    started = time.perf_counter()
    cpu_wave = decoder.run(None, {input_name: cpu_activation})[0]
    cpu_inference_s = time.perf_counter() - started
    started = time.perf_counter()
    hybrid_wave = decoder.run(None, {input_name: npu_activation})[0]
    hybrid_inference_s = time.perf_counter() - started
    cpu_vs_reference = compare(reference, cpu_wave)
    if not cpu_vs_reference["finite"] or cpu_vs_reference["relative_l2"] > 1e-4:
        raise ValueError("CPU prefix plus decoder fails retained waveform parity")
    hybrid_vs_cpu = compare(cpu_wave, hybrid_wave)
    hybrid_vs_reference = compare(reference, hybrid_wave)
    args.waves.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.waves, cpu=cpu_wave, hybrid=hybrid_wave)
    report = {
        "scope": "one fixed two-frame Code2Wav component, NPU prefix to CPU decoder; not full TTS",
        "host": platform.platform(),
        "onnxruntime_version": ort.__version__,
        "artifact_sha256": hashes,
        "waves_file": str(args.waves.resolve()),
        "waves_sha256": sha256(args.waves),
        "decoder_load_s": load_s,
        "cpu_prefix_cpu_decoder_inference_s": cpu_inference_s,
        "npu_prefix_cpu_decoder_inference_s": hybrid_inference_s,
        "cpu_prefix_cpu_decoder_vs_reference": cpu_vs_reference,
        "npu_prefix_cpu_decoder_vs_cpu": hybrid_vs_cpu,
        "npu_prefix_cpu_decoder_vs_reference": hybrid_vs_reference,
        "numeric_disposition": (
            "single_fixture_waveform_gate_pass"
            if hybrid_vs_cpu["finite"]
            and hybrid_vs_cpu["relative_l2"] <= 0.01
            and hybrid_vs_cpu["saturation_fraction"] < 0.01
            else "single_fixture_waveform_gate_failed"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"numeric_disposition": report["numeric_disposition"],
                      "relative_l2": hybrid_vs_cpu["relative_l2"]}))


if __name__ == "__main__":
    main()
