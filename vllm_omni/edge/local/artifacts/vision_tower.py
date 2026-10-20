# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exporting Qwen3.8-27B's vision tower as a stage for the iGPU or the NPU.

Of everything in the three models this project targets, this tower is the
component that best fits what an external device can actually do. The
constraint measured in M3.0 is that leaving the process costs real time --
2.8 ms across the WSL/Windows boundary for a megabyte each way -- so a stage
only pays for itself when it is compute-heavy and I/O-light. The tower is 27
transformer blocks run **once per image**, against 4.8 MB in and 4.0 MB out;
a decode step, run once per token, is the opposite shape and does not qualify.

It is also free of the usual obstacles. The 461 M parameters are already bf16
in the FP8 checkpoint -- ``quantization_config.modules_to_not_convert`` excludes
every visual module -- so the DirectML path needs no quantization story at all.
And the tower carries no KV and no cross-request state: pixels in, one
``[num_merged_tokens, out_hidden_size]`` tensor out.

**Which definition this exports.** ``transformers``'
:class:`Qwen3_5VisionModel`, whose 333 parameters match the checkpoint's
``model.visual.*`` tensors exactly, name for name. No third definition of the
model is written here -- open question 8 in the proposal is precisely about
export code drifting from the implementation. :class:`FixedImageVisionTower`
holds the real module and only fixes its shapes, the same way
:class:`~vllm_omni.edge.decoder_export.WindowedCode2Wav` does for the vocoder.

**What fixing the shape buys.** ``Qwen3_5VisionModel.forward`` derives bilinear
interpolation indices, position ids and ``cu_seqlens`` from ``grid_thw`` on
every call. At a fixed image size all three are constants, so they are computed
once here and kept as buffers -- and with them go every integer ``Gather`` in
the graph, which is the op most likely to have no kernel on a DirectML or NPU
backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_IMAGE_SIZE = 448
"""One image, 448x448: a 28x28 patch grid, 784 patches in, 196 merged tokens
out. Large enough that the 27 blocks dominate, small enough to stay a single
attention segment."""


@dataclass(frozen=True)
class TowerShape:
    """The geometry an exported tower is fixed at."""

    image_size: int
    grid_t: int
    grid_h: int
    grid_w: int
    patches: int
    patch_dim: int
    merged_tokens: int
    out_hidden_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "grid_thw": [self.grid_t, self.grid_h, self.grid_w],
            "patches": self.patches,
            "patch_dim": self.patch_dim,
            "merged_tokens": self.merged_tokens,
            "out_hidden_size": self.out_hidden_size,
        }


def _vision_config(model_dir: str | Path) -> Any:
    from transformers.models.qwen3_5 import Qwen3_5VisionConfig

    config = json.loads((Path(model_dir) / "config.json").read_text())
    if "vision_config" not in config:
        raise ValueError(
            f"{model_dir} has no vision_config; this checkpoint carries no tower "
            "and there is nothing here to place on an external device"
        )
    return Qwen3_5VisionConfig(**config["vision_config"])


def tower_shape(model_dir: str | Path, image_size: int = DEFAULT_IMAGE_SIZE) -> TowerShape:
    config = _vision_config(model_dir)
    if image_size % config.patch_size:
        raise ValueError(f"image_size {image_size} is not a multiple of patch_size {config.patch_size}")
    grid = image_size // config.patch_size
    merge = config.spatial_merge_size
    if (grid * grid) % (merge * merge):
        raise ValueError(f"a {grid}x{grid} grid does not divide by spatial_merge_size {merge}")
    return TowerShape(
        image_size=image_size,
        grid_t=1,
        grid_h=grid,
        grid_w=grid,
        patches=grid * grid,
        patch_dim=config.in_channels * config.temporal_patch_size * config.patch_size**2,
        merged_tokens=(grid * grid) // (merge * merge),
        out_hidden_size=config.out_hidden_size,
    )


def load_tower(
    model_dir: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    attn_implementation: str = "sdpa",
) -> nn.Module:
    """The real tower, with the checkpoint's weights, on the CPU.

    Weights are read from whichever shard holds ``model.visual.*`` -- in
    Qwen3.8-27B-FP8 that is ``outside.safetensors``, the same file the hot
    ``lm_head`` and the cold ``embed_tokens`` live in. ``strict=True``: a
    silently half-loaded tower would export cleanly and produce plausible
    garbage, which is the failure this project has the most scars from.
    """
    from safetensors.torch import load_file
    from transformers.models.qwen3_5 import Qwen3_5VisionModel

    config = _vision_config(model_dir)
    config._attn_implementation = attn_implementation

    weights: dict[str, torch.Tensor] = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        for name, tensor in load_file(str(shard)).items():
            if ".visual." in name:
                weights[name.split("model.visual.", 1)[1]] = tensor.to(dtype)
    if not weights:
        raise ValueError(f"no model.visual.* tensors in {model_dir}")

    tower = Qwen3_5VisionModel(config).to(dtype)
    tower.load_state_dict(weights, strict=True)
    return tower.eval()


class FixedImageVisionTower(nn.Module):
    """The tower at one image size, with its index arithmetic folded away.

    Everything ``forward`` would recompute from ``grid_thw`` -- interpolated
    position embeddings, rotary cos/sin, the attention segment boundaries -- is
    a function of the grid alone, so at a fixed size it is a constant. Folding
    them into buffers leaves a graph of convolution, GEMM, softmax and
    normalisation, with no integer gather anywhere in it.
    """

    def __init__(self, tower: nn.Module, shape: TowerShape) -> None:
        super().__init__()
        self.tower = tower
        self.shape = shape
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            get_vision_bilinear_indices_and_weights,
            get_vision_cu_seqlens,
            get_vision_position_ids,
        )

        grid = torch.tensor([[shape.grid_t, shape.grid_h, shape.grid_w]], dtype=torch.long)
        with torch.no_grad():
            indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
                grid,
                num_grid_per_side=tower.num_grid_per_side,
                spatial_merge_size=tower.config.spatial_merge_size,
                kwargs={},
            )
            position_ids = get_vision_position_ids(grid, tower.spatial_merge_size, kwargs={})
            cu_seqlens = get_vision_cu_seqlens(grid, kwargs={})

            # The interpolated position embedding is a weighted gather over a
            # constant table at constant indices: constant.
            pos_embeds = (tower.pos_embed(indices) * bilinear_weights[:, :, None]).sum(0)
            rotary = tower.rotary_pos_emb(position_ids).reshape(shape.patches, -1)
            emb = torch.cat((rotary, rotary), dim=-1)

        dtype = next(tower.parameters()).dtype
        self.register_buffer("pos_embeds", pos_embeds.to(dtype), persistent=False)
        self.register_buffer("rope_cos", emb.cos().to(dtype), persistent=False)
        self.register_buffer("rope_sin", emb.sin().to(dtype), persistent=False)
        # Kept as a tensor because the block signature takes one; for a single
        # image it is [0, patches], so the attention runs as one segment.
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """``[patches, patch_dim]`` -> ``[merged_tokens, out_hidden_size]``."""
        hidden = self.tower.patch_embed(patches)
        hidden = hidden + self.pos_embeds.to(hidden.dtype)
        hidden = hidden.reshape(self.shape.patches, -1)
        position_embeddings = (self.rope_cos, self.rope_sin)
        for block in self.tower.blocks:
            hidden = block(
                hidden,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )
        return self.tower.merger(hidden)

    def example_input(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or next(self.tower.parameters()).dtype
        generator = torch.Generator().manual_seed(0)
        return torch.randn(self.shape.patches, self.shape.patch_dim, generator=generator).to(dtype)


def build(
    model_dir: str | Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    dtype: torch.dtype = torch.float32,
) -> FixedImageVisionTower:
    shape = tower_shape(model_dir, image_size)
    # ``.eval()`` on the wrapper too, not just the tower it holds: the exporter
    # reads ``module.training`` on the top-level module and warns -- and any
    # future dropout in a block would key off its own flag anyway.
    return FixedImageVisionTower(load_tower(model_dir, dtype=dtype), shape).eval()


def export_onnx(
    module: FixedImageVisionTower,
    path: str | Path,
    *,
    opset: int = 21,
    external_data: bool = True,
) -> dict[str, Any]:
    """Export to ONNX and make the result loadable, which is a second step.

    ``sanitize_onnx`` is not optional here. torch's dynamo exporter emits
    ``Reshape`` with ``allowzero=1``, and onnxruntime's DirectML provider
    rejects that at **session initialisation** with a bare ``E_INVALIDARG`` --
    whereupon onnxruntime does not raise, it prints a fallback notice and hands
    back a working CPU session. Measured here: the unsanitized 27B vision tower
    placed 0 of 1171 nodes on the iGPU and computed correct features on the CPU;
    the same graph with 58 ``allowzero`` attributes cleared placed 85 of 113.
    The function was written for Qualcomm's converter and turns out to fix a
    generic dynamo wart.

    opset 21 by default because A16W8 needs it: 16-bit QDQ is only expressible
    without contrib ops from 21 on.
    """
    import torch

    from vllm_omni.edge.qnn_export import sanitize_onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    example = module.example_input()
    torch.onnx.export(
        module,
        (example,),
        str(path),
        input_names=["patches"],
        output_names=["embeds"],
        opset_version=opset,
        dynamo=True,
        optimize=True,
        external_data=external_data,
    )
    stats = sanitize_onnx(path)
    return {"path": str(path), "opset": opset, "sanitize": stats, "shape": module.shape.to_dict()}


def _main(argv: list[str] | None = None) -> int:
    import argparse

    import torch

    parser = argparse.ArgumentParser(
        prog="python -m vllm_omni.edge.local.artifacts.vision_tower",
        description="Export Qwen3.8-27B's vision tower as a fixed-size stage graph.",
    )
    parser.add_argument("--model", required=True, help="checkpoint directory")
    parser.add_argument("--out", required=True, help="destination .onnx")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--opset", type=int, default=21)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--check", action="store_true",
                        help="compare the fixed-shape wrapper against the real forward first")
    args = parser.parse_args(argv)

    module = build(args.model, image_size=args.image_size, dtype=getattr(torch, args.dtype))
    print(f"shape: {module.shape.to_dict()}")

    if args.check:
        grid = torch.tensor(
            [[module.shape.grid_t, module.shape.grid_h, module.shape.grid_w]], dtype=torch.long
        )
        example = module.example_input()
        with torch.no_grad():
            wrapped = module(example)
            reference = module.tower(example, grid).pooler_output
        identical = torch.equal(wrapped, reference)
        delta = (wrapped - reference).abs().max().item()
        print(f"wrapper vs Qwen3_5VisionModel.forward: identical={identical} max_abs={delta:.3e}")
        if not identical and delta > 1e-4:
            print("the fixed-shape wrapper does not reproduce the model; refusing to export")
            return 1

    result = export_onnx(module, args.out, opset=args.opset)
    print(f"exported {result['path']} (opset {result['opset']}), sanitize={result['sanitize']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
