# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 DiT with TaoMate streaming attention in the 50 main blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

from .attention import TaoMateH3StreamingAttention

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig


class TaoMateH3DiTModel(MiniMaxH3DiTModel):
    """Checkpoint-compatible H3 DiT whose block attention can stream over clean KV.

    The token refiner keeps the upstream attention: TaoMate never caches text
    K/V, and prompt refinement runs on replicated rows before sequence
    parallel sharding.
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        diffusers_weights: bool | None = None,
    ) -> None:
        super().__init__(
            od_config,
            quant_config,
            diffusers_weights=diffusers_weights,
            attention_cls=TaoMateH3StreamingAttention,
        )
        for index, block in enumerate(self.blocks):
            block.attn.layer_name = f"blocks.{index}.attn"

    @property
    def streaming_local_heads(self) -> int:
        """Heads per rank after the Ulysses all-to-all (the persistent KV head count)."""
        return int(self.blocks[0].attn.num_sp_heads)


EntryClass = TaoMateH3DiTModel

__all__ = ["TaoMateH3DiTModel"]
