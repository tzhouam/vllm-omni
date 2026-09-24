#!/usr/bin/env python3
"""Derive a pinned two-frame vocoder fixture from a generated Qwen3-TTS code stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_SHA = "e563541bf77200ac4c4ca9d6e01cd7e25c09f9274233a8d82eeccc57580830d2"
REAL_FIXTURE_SHA = "4220eec4ee00e5fbb0df6f195d883c6d4549accf399853c3a8e2f83821724452"
REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def relative_l2(reference, actual) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = actual.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "real-fixture", "real-report", "real-25-reference",
                 "fixture-output", "reference-output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.start_frame <= 23:
        parser.error("start frame must leave 72 context and two new frames inside 97")

    import numpy as np
    import onnxruntime as ort

    provenance = json.loads(args.real_report.read_text(encoding="utf-8-sig"))
    if (sha256(args.source) != SOURCE_SHA
            or sha256(args.real_fixture) != REAL_FIXTURE_SHA
            or provenance["model_revision"] != REVISION
            or provenance["files"]["fixture"]["sha256"] != REAL_FIXTURE_SHA
            or provenance["generated_frames"] < 97):
        raise ValueError("real generated-code input or model source changed")
    with np.load(args.real_fixture, allow_pickle=False) as data:
        quantized = np.ascontiguousarray(data["quantized"])
    with np.load(args.real_25_reference, allow_pickle=False) as data:
        long_wave = np.ascontiguousarray(data["wav"])
    if (quantized.shape != (1, 512, 97) or quantized.dtype != np.float32
            or long_wave.shape != (1, 48000)
            or not np.isfinite(quantized).all()):
        raise ValueError("real generated-code fixture contract changed")
    short = np.ascontiguousarray(quantized[:, :, args.start_frame:args.start_frame + 74])
    session = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    wave = session.run(None, {"quantized": short})[0]
    if wave.shape != (1, 3840) or not np.isfinite(wave).all():
        raise ValueError("two-frame CPU waveform contract failed")
    for path, key, value in ((args.fixture_output, "quantized", short),
                             (args.reference_output, "wav", wave)):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{key: value})
    report = {
        "scope": "two-frame Code2Wav input derived from one real generated CPU Qwen3-TTS code stream; no NPU yet",
        "source_sha256": SOURCE_SHA,
        "real_25_frame_fixture_sha256": REAL_FIXTURE_SHA,
        "model_revision": REVISION,
        "start_frame": args.start_frame,
        "onnxruntime": ort.__version__,
        "fixture_sha256": sha256(args.fixture_output),
        "cpu_reference_sha256": sha256(args.reference_output),
        "short_wave_vs_matching_25_frame_segment_relative_l2": relative_l2(
            long_wave[:, args.start_frame * 1920:(args.start_frame + 2) * 1920], wave
        ),
        "status": "real_code_short_fixture_prepared",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
