# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A16W8 quantization, which is the only door into the XDNA2 NPU.

The overlays shipped in the VitisAI EP are named for what they take --
``8x4_psu_model_a16w8_qdq.xclbin`` and friends -- and they mean it. A8W8, which
is what :func:`onnxruntime.quantization.quantize_static` produces by default,
and plain fp32 are both declined **with no error at all**: the session builds,
runs, returns correct numbers, and every node executed on the CPU. So the
precision is not a tuning knob here, it is the entry condition, and
:func:`vllm_omni.edge.npu_ryzenai.quantization_kwargs` is where it is written
down.

**Calibration is part of the artifact.** AGENTS.md requires a quantized export
to carry the data it was calibrated on, and this project has the receipts for
why: the Qwen3-TTS vocoder calibrated to int8 measured -4.3 dB with correlation
0.00 against its fp16 export, and uncalibrated w4a16 Spark builds scored
-1.6 dB -- uncorrelated with the reference, while looking like a normal model.
So :class:`ImageCalibrationReader` records what it fed the quantizer, and
:func:`quantize_a16w8` returns that record for the manifest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

CALIBRATION_IMAGES = 16
"""Small on purpose: the quantizer walks the whole graph per sample, and this
is a ranging pass, not training. Recorded either way -- the number matters less
than being able to say which images they were."""


@dataclass
class CalibrationRecord:
    """What a quantized artifact was calibrated on."""

    kind: str
    count: int
    sources: list[str] = field(default_factory=list)
    digest: str = ""
    shape: list[int] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "count": self.count,
            "sources": self.sources,
            "digest": self.digest,
            "shape": self.shape,
            "note": self.note,
        }


class ArrayCalibrationReader:
    """Feeds pre-built tensors to ``quantize_static`` and remembers them."""

    def __init__(self, input_name: str, samples: list[np.ndarray], record: CalibrationRecord) -> None:
        self.input_name = input_name
        self.samples = samples
        self.record = record
        self._iter: Iterator[np.ndarray] | None = None

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iter is None:
            self._iter = iter(self.samples)
        sample = next(self._iter, None)
        return None if sample is None else {self.input_name: sample}

    def rewind(self) -> None:
        self._iter = None


def image_calibration(
    model_dir: str | Path,
    image_paths: list[str | Path],
    *,
    image_size: int,
    input_name: str = "patches",
    dtype: Any = np.float32,
) -> ArrayCalibrationReader:
    """Real images, put through the checkpoint's own image processor.

    Using the shipped ``Qwen2VLImageProcessor`` rather than a hand-rolled
    patchify is the point: the tower's input layout (temporal pairs, 16x16
    patches, channel-major within a patch) is exactly the kind of thing that is
    easy to get subtly wrong, and a calibration set in the wrong layout ranges
    the activations of a model nobody is going to run.
    """
    from PIL import Image
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    pixels = image_size * image_size
    samples: list[np.ndarray] = []
    used: list[str] = []
    hasher = hashlib.sha256()

    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB").resize((image_size, image_size))
        except Exception:
            continue  # an unreadable file is skipped, and absent from `sources`
        encoded = processor(images=image, return_tensors="np",
                            min_pixels=pixels, max_pixels=pixels)
        sample = np.asarray(encoded["pixel_values"], dtype=dtype)
        samples.append(sample)
        used.append(str(path))
        hasher.update(sample.tobytes())

    if not samples:
        raise ValueError(f"no readable images among {len(image_paths)} paths")

    record = CalibrationRecord(
        kind="real images, resized to a square and run through the checkpoint's "
             "Qwen2VLImageProcessor",
        count=len(samples),
        sources=used,
        digest=hasher.hexdigest()[:32],
        shape=list(samples[0].shape),
        note=(
            "a ranging set, not a quality claim: these are whatever images this "
            "repository happens to contain, not a task-representative sample"
        ),
    )
    return ArrayCalibrationReader(input_name, samples, record)


def quantize_a16w8(
    source: str | Path,
    target: str | Path,
    reader: ArrayCalibrationReader,
    *,
    op_types_to_quantize: tuple[str, ...] | None = None,
    per_channel: bool | None = None,
) -> dict[str, Any]:
    """Quantize an ONNX graph to A16W8 QDQ, the shape XDNA2 partitions.

    The default quantization settings come from
    :mod:`vllm_omni.edge.npu_ryzenai`. ``per_channel`` permits a recorded
    experimental variant; NPU placement and numerical validation must still
    be checked for that exported graph.
    """
    from onnxruntime.quantization import QuantFormat, quantize_static

    from vllm_omni.edge.npu_ryzenai import quantization_kwargs

    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    kwargs = quantization_kwargs()
    if per_channel is not None:
        kwargs["per_channel"] = per_channel

    quantize_static(
        str(source),
        str(target),
        reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=list(op_types_to_quantize) if op_types_to_quantize else None,
        **kwargs,
    )
    reader.rewind()

    return {
        "activation_type": str(kwargs["activation_type"]),
        "weight_type": str(kwargs["weight_type"]),
        "per_channel": kwargs["per_channel"],
        "quant_format": "QDQ",
        "calibration": reader.record.to_dict(),
        "source_sha256_prefix": _digest(source)[:16],
    }


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
