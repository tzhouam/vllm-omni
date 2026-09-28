# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TaoMate-H3 audio teacher: base-H3 audio guidance for one five-second request.

TaoMate-H3 does not publish the LoRA student's audio. For every request the
*base* MiniMax-H3 model (LoRA disabled) runs an audio-only ten-state schedule
(nine forwards) over ``[text | 1 s clean reference tail | new audio noise]``
and the student's audio rows are overwritten with the teacher's states 3, 6 and
9 after its three denoise steps. The reference tail ("tail-40 rollover") is the
last forty latents per channel of the previous request's clean audio, so
speech continues seamlessly across requests.

The released implementation runs this teacher as a separate BF16 process on
its own GPUs. Here it shares the resident weights with the student: the LoRA
is a switchable hook set, so disabling it for nine forwards yields the base
model exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch

from .cuda_graph import GraphedForward
from .geometry import REQUEST_AUDIO_LATENTS, REQUEST_VIDEO_LATENTS, CanvasGeometry, StreamPlan
from .lora import TaoMateLoRAAdapter
from .packed import taomate_audio_only_frozen_prefix_packed_layout, taomate_audio_only_packed_layout
from .schedule import TEACHER_STATE_NUMBERS, euler_eta0_update_, teacher_sigmas
from .transformer import LocalEmbedPlan, local_embed_plan

ROLLOVER_LATENTS_PER_CHANNEL = 40
AUDIO_ROW_WIDTH = 32
VIDEO_ROW_WIDTH = 96


def official_request_noise(
    *,
    seed: int,
    canvas: CanvasGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the official standalone request noise in the release's generator order.

    Returns ``(video [1, 24, 37, lh, lw], audio rows [2 * 207, 32])`` drawn from one
    CPU generator: video first, then audio. The teacher and the student both use
    the audio draw; the video draw is part of the audio identity even when the
    caller discards it.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    video = torch.randn(
        1,
        24,
        REQUEST_VIDEO_LATENTS,
        canvas.latent_h,
        canvas.latent_w,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    audio = torch.randn(
        2 * REQUEST_AUDIO_LATENTS, AUDIO_ROW_WIDTH, generator=generator, dtype=torch.float32, device="cpu"
    )
    return video, audio


@dataclass
class TeacherResult:
    """Milestones of one request: clean audio rows at teacher states 3, 6 and 9."""

    milestones: list[torch.Tensor]
    audio_latent_count: int
    forwards: int

    @property
    def clean(self) -> torch.Tensor:
        return self.milestones[-1]


class TaoMateAudioTeacher:
    """Runs the base-model audio schedule on the shared DiT."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        lora: TaoMateLoRAAdapter | None,
        device: torch.device,
        graph: GraphedForward | None = None,
    ) -> None:
        self.transformer = transformer
        self.lora = lora
        self.device = device
        # Optional CUDA-graph replay of the nine fixed-shape forwards.
        self.graph = graph
        self.sigmas_video, self.sigmas_audio = teacher_sigmas()
        self.forwards = 0
        # One embedding plan per document shape. A captured graph reads the
        # plan's index tensors by address, so the plan of a shape must stay
        # the same object for as long as the process serves that shape.
        self._embed_plans: dict[tuple[int, ...], LocalEmbedPlan | None] = {}

    def build_branch(
        self,
        *,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        canvas: CanvasGeometry,
        audio_latent_count: int,
        previous_audio_latent_count: int | None,
        seq_len: int | None = None,
    ) -> MiniMaxH3DenoiseBranch:
        text_len = int(text_embeddings.shape[0])
        if previous_audio_latent_count is None:
            packed = taomate_audio_only_packed_layout(
                text_len=text_len,
                audio_t=audio_latent_count,
                latent_h=canvas.latent_h,
                latent_w=canvas.latent_w,
                seq_len=seq_len,
            )
        else:
            packed = taomate_audio_only_frozen_prefix_packed_layout(
                text_len=text_len,
                ref_audio_t=ROLLOVER_LATENTS_PER_CHANNEL,
                audio_t=audio_latent_count,
                latent_h=canvas.latent_h,
                latent_w=canvas.latent_w,
                reference_time_start=text_len + previous_audio_latent_count - ROLLOVER_LATENTS_PER_CHANNEL,
                target_time_start=text_len + previous_audio_latent_count,
                seq_len=seq_len,
            )
        tags = packed["token_tags"].clone()
        tags[packed["text_pos"].view(-1)] = text_tags.detach().to("cpu", torch.long)
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=tags,
            device=self.device,
        )
        branch.prepare_rope_table(self.transformer)
        attach_teacher_timestep_slots(branch)
        branch.taomate_embed_plan = self._embed_plan_for(branch, text_pos=packed["text_pos"])
        return branch

    def _embed_plan_for(self, branch: MiniMaxH3DenoiseBranch, *, text_pos: torch.Tensor) -> LocalEmbedPlan | None:
        plan_local = getattr(self.transformer, "plan_local_embed", None)
        if not callable(plan_local):
            return None
        shape = (
            int(branch.seq_len),
            int(branch.text_len),
            int(branch.audio_pos.numel()),
            int((~branch.audio_update_mask).sum()),
        )
        if shape not in self._embed_plans:
            self._embed_plans[shape] = plan_local(
                img_pos=branch.img_pos,
                audio_pos=branch.audio_pos,
                text_pos=text_pos,
                seq_len=branch.seq_len,
                device=self.device,
            )
        return self._embed_plans[shape]

    @torch.inference_mode()
    def run_request(
        self,
        *,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        canvas: CanvasGeometry,
        plan: StreamPlan,
        audio_seed: int,
        previous_clean: torch.Tensor | None,
        previous_audio_latent_count: int | None,
        seq_len: int | None = None,
    ) -> TeacherResult:
        """Generate the request's audio milestones with the LoRA disabled."""
        active_audio_latents = plan.audio_latent_count
        transport_prefix = REQUEST_AUDIO_LATENTS - active_audio_latents
        _, official_audio = official_request_noise(seed=audio_seed, canvas=canvas)
        target_noise = (
            official_audio.view(2, REQUEST_AUDIO_LATENTS, AUDIO_ROW_WIDTH)[:, transport_prefix:]
            .contiguous()
            .view(2 * active_audio_latents, AUDIO_ROW_WIDTH)
        )
        if previous_clean is None:
            reference = None
            initial_audio = target_noise
        else:
            if previous_audio_latent_count is None:
                raise ValueError("previous_audio_latent_count is required with previous_clean")
            reference = (
                previous_clean.detach()
                .to(device="cpu", dtype=torch.float32)
                .view(2, int(previous_audio_latent_count), AUDIO_ROW_WIDTH)[:, -ROLLOVER_LATENTS_PER_CHANNEL:]
                .contiguous()
                .view(2 * ROLLOVER_LATENTS_PER_CHANNEL, AUDIO_ROW_WIDTH)
            )
            initial_audio = torch.cat((reference, target_noise), dim=0)
        branch = self.build_branch(
            text_embeddings=text_embeddings,
            text_tags=text_tags,
            canvas=canvas,
            audio_latent_count=active_audio_latents,
            previous_audio_latent_count=None if previous_clean is None else previous_audio_latent_count,
            seq_len=seq_len,
        )
        audio_rows = initial_audio.to(device=self.device, dtype=torch.float32)
        target = branch.audio_update_mask_dev
        ref_rows = int((~branch.audio_update_mask).sum())
        video_rows = torch.empty((0, VIDEO_ROW_WIDTH), dtype=torch.float32, device=self.device)
        captured: dict[int, torch.Tensor] = {}
        steps = len(self.sigmas_video) - 1
        lora_scope = self.lora.disabled() if self.lora is not None else _NullScope()
        plan: LocalEmbedPlan | None = getattr(branch, "taomate_embed_plan", None)
        with lora_scope, local_embed_plan(plan):
            for step in range(steps):
                s_v = self.sigmas_video[step]
                s_a, s_a_next = self.sigmas_audio[step], self.sigmas_audio[step + 1]
                t_v, t_a = 1.0 - s_v, 1.0 - s_a
                forward_kwargs = teacher_forward_kwargs(
                    branch,
                    video_rows=video_rows,
                    audio_rows=audio_rows,
                    t_video=t_v,
                    t_audio=t_a,
                )
                _, velocity_audio = self._forward(forward_kwargs, plan=plan)
                self.forwards += 1
                velocity_target = velocity_audio.float()[target]
                target_rows = audio_rows[ref_rows:]
                euler_eta0_update_(target_rows, velocity_target, sigma_curr=s_a, sigma_next=s_a_next)
                state_number = step + 1
                if state_number in TEACHER_STATE_NUMBERS:
                    captured[state_number] = target_rows.detach().clone()
        milestones = [captured[number] for number in TEACHER_STATE_NUMBERS]
        return TeacherResult(milestones=milestones, audio_latent_count=active_audio_latents, forwards=steps)

    def _forward(self, forward_kwargs: dict, *, plan: LocalEmbedPlan | None) -> tuple[torch.Tensor, torch.Tensor]:
        graph = self.graph
        if graph is None or not graph.enabled:
            return self.transformer(**forward_kwargs)
        # The graph is captured with the LoRA hooks disabled and this
        # document's embedding plan installed; both are part of its identity.
        variant = ("teacher", "lora_off", None if plan is None else plan.fingerprint)
        return graph(variant=variant, keep_alive=() if plan is None else (plan,), **forward_kwargs)


# Timestep slots of a teacher document: every row's AdaLN timestep is one of
# three values, so ``unique_timesteps`` is written as a fixed three-row vector
# and ``inverse_indices`` as a constant slot map. This replaces the per-forward
# ``torch.unique`` (a host synchronization) and keeps the timestep tensors at
# one shape for CUDA-graph replay. Duplicate slot values (both timesteps are 0
# at the first step) are harmless: the DiT gathers per row, uniqueness is not
# required.
TEACHER_SLOT_TEXT = 0  # text and padding rows: the video timestep
TEACHER_SLOT_REFERENCE = 1  # clean reference audio rows: timestep 1
TEACHER_SLOT_TARGET = 2  # denoised audio rows: the audio timestep


def attach_teacher_timestep_slots(branch: MiniMaxH3DenoiseBranch) -> torch.Tensor:
    """Store the branch's constant slot map (``branch.taomate_inverse_indices``)."""
    inverse = torch.full((branch.seq_len,), TEACHER_SLOT_TEXT, dtype=torch.long)
    audio_pos = branch.audio_pos.view(-1).to(torch.long)
    update = branch.audio_update_mask.view(-1).to(torch.bool)
    inverse[audio_pos[~update]] = TEACHER_SLOT_REFERENCE
    inverse[audio_pos[update]] = TEACHER_SLOT_TARGET
    if branch.img_pos.numel():
        raise ValueError("teacher documents carry no video rows")
    branch.taomate_inverse_indices = inverse.to(branch.device)
    return branch.taomate_inverse_indices


def teacher_forward_kwargs(
    branch: MiniMaxH3DenoiseBranch,
    *,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    t_video: float,
    t_audio: float,
) -> dict:
    """The branch's forward kwargs with fixed-slot timesteps (no host sync)."""
    inverse = getattr(branch, "taomate_inverse_indices", None)
    if inverse is None:
        inverse = attach_teacher_timestep_slots(branch)
    x = branch.x_base.clone()
    if video_rows.shape[0]:
        x[0].index_copy_(0, branch.img_pos_dev, video_rows)
    audio_x = branch.audio_x_base.clone()
    audio_x[0].index_copy_(0, branch.audio_pos_dev, audio_rows)
    unique = torch.tensor([float(t_video), 1.0, float(t_audio)], dtype=torch.float32, device=branch.device)
    return {
        **branch.static_kwargs,
        "x": x,
        "audio_x": audio_x,
        "unique_timesteps": unique,
        "inverse_indices": inverse,
    }


class _NullScope:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


__all__ = [
    "ROLLOVER_LATENTS_PER_CHANNEL",
    "TaoMateAudioTeacher",
    "TeacherResult",
    "attach_teacher_timestep_slots",
    "official_request_noise",
    "teacher_forward_kwargs",
]
