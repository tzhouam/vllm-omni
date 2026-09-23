#!/usr/bin/env python3
"""Run a pinned Qwen3-TTS Code2Wav TFLite binary on local LiteRT CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from audit_qwen_tts_vocoder_qualcomm import compare, read_output


TFLITE_SHA256 = "29826da6ecc1ec2efc0e3589c3cebb42b20d7e20c0975a037d9f51625d6638cd"
FIXTURE_SHA256 = "dfca9e8d2a724a568a31b6cd18ff6a52fdc205fd34e13cbf1cd75f0f7937e82d"
REFERENCE_SHA256 = "8de78b94b59f01e49852f7706adbd2508a9976b8e2705cfd3b94d07bb3606cb1"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "fixture", "reference", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    for name, expected in {
        "model": TFLITE_SHA256,
        "fixture": FIXTURE_SHA256,
        "reference": REFERENCE_SHA256,
    }.items():
        if sha256(getattr(args, name)) != expected:
            raise ValueError(f"{name} differs from the pinned artifact")

    with np.load(args.fixture, allow_pickle=False) as data:
        if data.files != ["quantized"]:
            raise ValueError("unexpected input dataset layout")
        input_value = np.asarray(data["quantized"])
    if input_value.shape != (1, 512, 97) or input_value.dtype != np.float32:
        raise ValueError("expected one FP32 [1,512,97] input")

    interpreter = Interpreter(model_path=str(args.model), num_threads=args.threads)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1 or inputs[0]["name"] != "quantized":
        raise ValueError("unexpected TFLite I/O contract")
    if tuple(inputs[0]["shape"]) != input_value.shape or inputs[0]["dtype"] != np.float32:
        raise ValueError("TFLite input shape/type differs")
    if tuple(outputs[0]["shape"]) != (1, 48000) or outputs[0]["dtype"] != np.float32:
        raise ValueError("TFLite output shape/type differs")

    interpreter.set_tensor(inputs[0]["index"], input_value)
    started = time.monotonic()
    interpreter.invoke()
    elapsed = time.monotonic() - started
    actual = np.asarray(interpreter.get_tensor(outputs[0]["index"]))
    if actual.shape != (1, 48000) or actual.dtype != np.float32 or not np.isfinite(actual).all():
        raise ValueError("TFLite output violates finite waveform contract")
    np.savez_compressed(args.output, wav=actual)

    source = read_output(args.reference, "wav")
    report = {
        "scope": "local x86-64 LiteRT CPU execution of one retained Code2Wav TFLite binary; no hosted-device timing or complete TTS",
        "status": "component_cpu_numeric_parity_on_one_fixture",
        "runtime": {"ai_edge_litert": importlib.metadata.version("ai-edge-litert"), "numpy": np.__version__},
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "processor": cpu_model()},
        "threads": args.threads,
        "model_sha256": sha256(args.model),
        "fixture_sha256": sha256(args.fixture),
        "reference_sha256": sha256(args.reference),
        "output_sha256": sha256(args.output),
        "source_onnx_cpu_vs_litert_cpu": compare(source, actual),
        "single_invoke_seconds_not_a_profile": elapsed,
        "limits": [
            "Local x86-64 CPU execution cannot attest SA8775P or RB3 device CPU behavior.",
            "LiteRT CPU output is not evidence that the Android GPU delegate is correct.",
            "One fixed vocoder fixture has no listening-quality tolerance or complete-stream gate.",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "comparison": report["source_onnx_cpu_vs_litert_cpu"]}, indent=2))


if __name__ == "__main__":
    main()
