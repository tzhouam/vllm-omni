# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prompt interaction handler for TaoMate-H3 streaming sessions.

TaoMate-H3 encodes one prompt per five-second request and applies it to the
whole request (the audio teacher and all four phases), so a prompt update is a
hard switch that takes effect at the next request boundary. Prompt lengths
differ between updates, which rules out the generic handler's embedding
interpolation; this subclass forces an immediate transition.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from typing_extensions import override

from vllm_omni.diffusion.interaction.modality_handlers.prompt import PromptInteractionHandler
from vllm_omni.diffusion.interaction.types import InteractionPayload
from vllm_omni.diffusion.worker.utils import StepRequestState


class TaoMateH3PromptInteractionHandler(PromptInteractionHandler):
    modality: ClassVar[str] = "prompt"

    @classmethod
    @override
    def from_pipeline(cls, pipeline: Any) -> TaoMateH3PromptInteractionHandler:
        # The H3 DiT keeps mixed parameter dtypes and exposes no ``dtype``; the
        # text encoder returns BF16 hidden states regardless.
        return cls(encode_prompt=pipeline.encode_prompt, device=pipeline.device, dtype=torch.bfloat16)

    @override
    def enqueue(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        received_at: float,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        del transition_chunks
        super().enqueue(
            state,
            event_id=event_id,
            received_at=received_at,
            payload=payload,
            transition_chunks=0,
        )


__all__ = ["TaoMateH3PromptInteractionHandler"]
