"""Rebuild and numerically gate a same-checkpoint Qwen3.8 vision merger.

This is a component probe, not a complete multimodal request. It records
real-image activations from the unmodified checkpoint tower and leaves device
placement to the external-stage CLI, which profiles actual ORT node assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor

from vllm_omni.edge.local.artifacts.quantize import (
    ArrayCalibrationReader,
    CalibrationRecord,
    quantize_a16w8,
)
from vllm_omni.edge.local.artifacts.vision_tower import (
    FixedImageVisionTower,
    load_tower,
    tower_shape,
)
from vllm_omni.edge.qnn_export import sanitize_onnx


class Merger(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.merger = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merger(x)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-12)),
        "max_abs": float(np.max(np.abs(delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--quantize-ops", choices=["all", "gemm"], default="all")
    parser.add_argument("--per-channel", action="store_true")
    args = parser.parse_args()
    if len(args.images) < 3:
        parser.error("use at least two calibration images and one held-out image")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(12)
    shape = tower_shape(args.model)
    tower = load_tower(args.model, dtype=torch.float32)
    fixed = FixedImageVisionTower(tower, shape).eval()
    merger = Merger(tower.merger).eval()
    processor = AutoImageProcessor.from_pretrained(str(args.model))
    activations: list[np.ndarray] = []
    references: list[np.ndarray] = []
    image_sha256: list[str] = []

    def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        activations.append(inputs[0].detach().cpu().numpy().copy())

    hook = tower.merger.register_forward_pre_hook(capture)
    for image_path in args.images:
        image_sha256.append(_file_sha256(image_path))
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").resize((shape.image_size, shape.image_size))
        encoded = processor(
            images=image,
            return_tensors="np",
            min_pixels=shape.image_size**2,
            max_pixels=shape.image_size**2,
        )
        patches = torch.from_numpy(np.asarray(encoded["pixel_values"], dtype=np.float32))
        if tuple(patches.shape) != (shape.patches, shape.patch_dim):
            raise ValueError(f"unexpected processed image shape: {tuple(patches.shape)}")
        with torch.inference_mode():
            references.append(fixed(patches).detach().cpu().numpy().copy())
    hook.remove()
    if len(activations) != len(args.images):
        raise ValueError("merger activation capture count differs from image count")

    source = args.out / "merger_fp32.onnx"
    torch.onnx.export(
        merger,
        (torch.from_numpy(activations[0]),),
        str(source),
        input_names=["x"],
        output_names=["y"],
        opset_version=21,
        dynamo=True,
        optimize=True,
    )
    sanitize = sanitize_onnx(source)
    variant = f"{args.quantize_ops}{'_perchannel' if args.per_channel else ''}"
    qdq = args.out / f"merger_a16w8_{variant}.onnx"
    calibration_arrays = activations[:-1]
    digest = hashlib.sha256()
    for array in calibration_arrays:
        digest.update(array.tobytes())
    record = CalibrationRecord(
        kind="same-checkpoint merger activations from resized real images",
        count=len(calibration_arrays),
        sources=[str(path) for path in args.images[:-1]],
        digest=digest.hexdigest(),
        shape=list(calibration_arrays[0].shape),
        note="last image is held out from quantizer calibration",
    )
    quantization = quantize_a16w8(
        source,
        qdq,
        ArrayCalibrationReader("x", calibration_arrays, record),
        op_types_to_quantize=("Gemm",) if args.quantize_ops == "gemm" else None,
        per_channel=True if args.per_channel else None,
    )

    cpu_fp32 = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    cpu_qdq = ort.InferenceSession(str(qdq), providers=["CPUExecutionProvider"])
    cases = []
    for image_path, image_sha, array, reference in zip(
        args.images, image_sha256, activations, references, strict=True
    ):
        fp32 = cpu_fp32.run(None, {"x": array})[0]
        quantized = cpu_qdq.run(None, {"x": array})[0]
        cases.append(
            {
                "image": str(image_path),
                "image_sha256": image_sha,
                "calibration": image_path != args.images[-1],
                "input_shape": list(array.shape),
                "output_shape": list(reference.shape),
                "fp32_vs_torch": _metrics(fp32, reference),
                "a16w8_vs_torch": _metrics(quantized, reference),
            }
        )
    np.savez(args.out / "heldout_input.npz", x=activations[-1])
    report = {
        "model": str(args.model),
        "model_config_sha256": _file_sha256(args.model / "config.json"),
        "outside_safetensors_sha256": _file_sha256(args.model / "outside.safetensors"),
        "checkpoint_dtype": "visual BF16 loaded as FP32",
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "shape": shape.to_dict(),
        "sanitization": sanitize,
        "quantization": quantization,
        "quantize_ops": args.quantize_ops,
        "artifacts": {
            "fp32_sha256": _file_sha256(source),
            "fp32_external_data_sha256": _file_sha256(
                source.with_suffix(source.suffix + ".data")
            ),
            "a16w8_sha256": _file_sha256(qdq),
        },
        "cases": cases,
    }
    (args.out / f"component_numeric_{variant}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"cases": cases, "quantization": quantization}, indent=2))


if __name__ == "__main__":
    main()
