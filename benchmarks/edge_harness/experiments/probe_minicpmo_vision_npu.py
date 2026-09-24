#!/usr/bin/env python3
"""Export a pinned MiniCPM-o vision-encoder prefix for AMD NPU feasibility.

The source checkpoint's BF16 vision stack is converted explicitly to FP32.
This exports a fixed 448px, no-padding vision block, not a complete model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_l2(reference, actual) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = actual.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def export(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from safetensors import safe_open

    sys.path.insert(0, str(args.model_dir.resolve()))
    from modeling_navit_siglip import SiglipVisionConfig, SiglipVisionTransformer

    config_path = args.model_dir / "config.json"
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    source_path = args.model_dir / "modeling_navit_siglip.py"
    revisions = {
        (args.model_dir / ".cache/huggingface/download" / (path.name + ".metadata"))
        .read_text(encoding="utf-8").splitlines()[0]
        for path in (config_path, shard, source_path)
    }
    if len(revisions) != 1:
        raise ValueError("MiniCPM-o vision config, source and weights differ in revision")
    revision = revisions.pop()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vision_config = SiglipVisionConfig(**config["vision_config"])
    vision_config._attn_implementation = "eager"
    torch.set_num_threads(args.threads)
    model = SiglipVisionTransformer(vision_config)
    with safe_open(shard, framework="pt", device="cpu") as source:
        keys = [key for key in source.keys() if key.startswith("vpm.")]
        state = {key[4:]: source.get_tensor(key) for key in keys}
    model.load_state_dict(state, strict=True)
    model.eval()
    del state

    # This is the same normalized, fixed 448px no-padding image bucket as the
    # model processor; the source is a retained MiniCPM-o request image.
    image = Image.open(args.image).convert("RGB").resize((448, 448))
    pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    pixels = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)[None]))
    mask = torch.ones((1, 32, 32), dtype=torch.bool)
    sizes = torch.tensor([[32, 32]], dtype=torch.int32)
    with torch.inference_mode():
        bf16 = model.to(torch.bfloat16)
        source_embedding = bf16.embeddings(pixels.to(torch.bfloat16), mask, sizes)
        source_value = source_embedding
        for layer in bf16.encoder.layers[: args.layers]:
            source_value = layer(source_value, None)[0]
        source_value = source_value.float().numpy()
        candidate = model.float()
        embedding = candidate.embeddings(pixels, mask, sizes)
        fp32_value = embedding
        for layer in candidate.encoder.layers[: args.layers]:
            fp32_value = layer(fp32_value, None)[0]
        fp32_value = fp32_value.numpy()
    if embedding.shape != (1, 1024, 1152) or not np.isfinite(fp32_value).all():
        raise RuntimeError("fixed vision boundary contract failed")

    class Prefix(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = torch.nn.ModuleList(layers)

        def forward(self, hidden):
            for layer in self.layers:
                hidden = layer(hidden, None)[0]
            return hidden

    prefix = Prefix(list(candidate.encoder.layers[: args.layers])).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            prefix, (embedding,), str(args.output), input_names=["hidden"],
            output_names=["hidden_after_prefix"], opset_version=17,
            dynamo=False, do_constant_folding=True,
        )
    export_s = time.perf_counter() - started
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.output), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    observed = session.run(None, {"hidden": embedding.numpy()})[0]
    parity = relative_l2(fp32_value, observed)
    if observed.shape != (1, 1024, 1152) or parity > 1e-4:
        raise RuntimeError(f"ONNX CPU vision-prefix parity failed: {parity}")
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fixture, hidden=embedding.numpy(),
                        torch_fp32=fp32_value, torch_bf16=source_value,
                        ort_fp32=observed)
    report = {
        "scope": f"MiniCPM-o 4.5 real-weight {args.layers}-layer vision prefix on a retained synthetic image; component export only",
        "status": "fp32_vision_prefix_cpu_parity_pass",
        "model_config_sha256": sha256(config_path),
        "model_source_sha256": sha256(source_path),
        "source_shard_sha256": sha256(shard),
        "image_sha256": sha256(args.image),
        "model_revision": revision,
        "layers": args.layers,
        "input_shape": list(embedding.shape),
        "output_shape": list(observed.shape),
        "precision_change": "checkpoint BF16 to explicit FP32 ONNX",
        "torch_fp32_vs_bf16_relative_l2": relative_l2(source_value, fp32_value),
        "ort_fp32_vs_torch_fp32_relative_l2": parity,
        "export_s": export_s,
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "fixture_sha256": sha256(args.fixture),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "threads": args.threads,
        "next_gate": "A16W8 numerical calibration and actual AMD NPU node placement; then complete-vision and downstream model quality",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layers <= 27 or not 1 <= args.threads <= 24:
        parser.error("layers must be 1..27 and threads 1..24")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    export(args)


if __name__ == "__main__":
    main()
