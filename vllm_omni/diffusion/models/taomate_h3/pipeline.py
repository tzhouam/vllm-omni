# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TaoMate-H3: realtime streaming MiniMax-H3 with the TaoMate LoRA.

TaoMate-H3 (``TaoLiveAIGC/TaoMate-H3``) is a rank-128 LoRA over MiniMax-H3 that
turns the bidirectional five-second T2VA model into a causal streamer: each
request is generated as four phases of 34/34/34/17 frames, every phase runs
three denoise steps conditioned on the clean (sigma = 0) keys and values of
earlier phases, and the base model provides the audio. This pipeline hosts that
contract on vLLM-Omni's AR-Diffusion runtime (LingBot-World's realtime path):

* one AR chunk = one TaoMate phase (``chunk_num_steps = 3``) with the clean-KV
  commit and the incremental VAE decode in ``post_decode``;
* the persistent clean audio/video KV is model-owned (``CleanAVKVCache``) and
  read by ``TaoMateH3StreamingAttention`` after the Ulysses all-to-all;
* the prompt of the *next* request may change at any chunk boundary through
  the streaming ``session.interaction`` API (TaoMate's just-in-time lock);
* the LoRA stays unmerged so the resident BF16 weights also serve the LoRA-free
  audio teacher.

The pipeline reuses the upstream MiniMax-H3 pipeline for component loading
(DiT, Qwen3-VL text encoder with tensor parallel, VAEs) and only differs in
the DiT attention class and the execution contract. Serving topology: pure
Ulysses sequence parallelism (``tensor_parallel_size == 1``).
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.models.interface import SupportsStepExecution
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec, ARDiffusionKVCacheSpec
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionChunkMetadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import PreparedEncoderInputs

from .attention import StreamContext, StreamMode, _ulysses_state, stream_context
from .audio_teacher import (
    ROLLOVER_LATENTS_PER_CHANNEL,
    TaoMateAudioTeacher,
    TeacherResult,
    official_request_noise,
)
from .cuda_graph import GraphedForward
from .geometry import (
    AUDIO_SAMPLE_RATE,
    PHASE_GROUP_COUNTS,
    REQUEST_AUDIO_LATENTS,
    REQUEST_NATIVE_FRAMES,
    REQUEST_VIDEO_LATENTS,
    VIDEO_FPS,
    CanvasGeometry,
    StreamPhase,
    StreamPlan,
    phases_for_frames,
    request_plan,
)
from .kv_cache import CleanAVKVCache, KVContract
from .lora import TaoMateLoRAAdapter, is_taomate_lora_dir
from .packed import taomate_phase_packed_layout
from .schedule import euler_eta0_update_, student_sigmas
from .stream_decode import StreamingAudioDecoder, StreamingVideoDecoder, samples_at_frame
from .transformer import TaoMateH3DiTModel

logger = init_logger(__name__)

NUM_PHASES = len(PHASE_GROUP_COUNTS)
# Largest per-phase audio latent count over the first and the steady-state
# request geometries (audio boundaries are rounded on the global timeline).
_MAX_PHASE_AUDIO_LATENTS = tuple(
    max(request_plan(index).phases[phase].audio_latent_count for index in range(0, 8)) for phase in range(NUM_PHASES)
)
STUDENT_STEPS = 3
DEFAULT_SEED = 8301
DEFAULT_HEIGHT = 864
DEFAULT_WIDTH = 480
DEFAULT_AUDIO_KV_RESET_REQUESTS = 12
# Ten minutes of 24 fps video when a session does not name a frame budget: the
# WebSocket client ends the session by disconnecting.
DEFAULT_SESSION_FRAMES = 10 * 60 * VIDEO_FPS
_VIDEO_NOISE_REQUEST_STRIDE = 1_000_003
_VIDEO_ROW_WIDTH = 96
_AUDIO_ROW_WIDTH = 32
_AR_BRANCH = "main"


def _prompt_version(state: StepRequestState) -> int:
    """Count of prompt updates applied to this stream (bumped by the prompt interaction handler)."""
    session = state.interaction_sessions.get("prompt")
    return int(getattr(session, "version", 0) or 0)


def _validate_parallel_config(od_config: OmniDiffusionConfig) -> None:
    parallel = od_config.parallel_config
    tp = int(getattr(parallel, "tensor_parallel_size", 1) or 1)
    if tp != 1:
        raise ValueError(
            "TaoMate-H3 runs with pure Ulysses sequence parallelism; set tensor_parallel_size=1 "
            f"(got {tp}). The native LoRA is replicated per rank and the clean KV is sharded by heads."
        )
    ulysses = int(getattr(parallel, "ulysses_degree", 1) or 1)
    sp = int(getattr(parallel, "sequence_parallel_size", 1) or 1)
    ring = int(getattr(parallel, "ring_degree", 1) or 1)
    allgather = int(getattr(parallel, "allgather_degree", 1) or 1)
    if ring != 1 or allgather != 1 or sp != ulysses:
        raise ValueError(
            "TaoMate-H3 requires sequence_parallel_size == ulysses_degree with ring_degree=1 and "
            f"allgather_degree=1 (got sp={sp}, ulysses={ulysses}, ring={ring}, allgather={allgather})."
        )
    if str(getattr(parallel, "ulysses_mode", "strict") or "strict") != "strict":
        raise ValueError("TaoMate-H3 supports ulysses_mode='strict' only.")
    if bool(getattr(parallel, "ulysses_a2a_permute", False)):
        raise ValueError("TaoMate-H3 does not support ulysses_a2a_permute.")
    if int(getattr(parallel, "cfg_parallel_size", 1) or 1) != 1:
        raise ValueError("TaoMate-H3 is CFG-distilled; cfg_parallel_size must be 1.")
    if int(getattr(parallel, "pipeline_parallel_size", 1) or 1) != 1:
        raise ValueError("TaoMate-H3 does not support pipeline parallelism.")


@dataclass
class _RequestState:
    """One five-second request of a session."""

    index: int
    plan: StreamPlan
    text_embeddings: torch.Tensor
    text_tags: torch.Tensor
    video_rows: torch.Tensor  # initial noise rows of the request's active video latents
    audio_rows: torch.Tensor  # initial noise rows [2 * active_audio, 32]
    audio_seed: int
    video_seed: int
    frame_rows: int
    teacher: TeacherResult | None = None
    teacher_seconds: float = 0.0

    @property
    def text_len(self) -> int:
        return int(self.text_embeddings.shape[0])

    def phase_video_rows(self, phase: StreamPhase) -> torch.Tensor:
        start = phase.video_latent_start * self.frame_rows
        stop = phase.video_latent_stop * self.frame_rows
        return self.video_rows[start:stop].clone()

    def phase_audio_rows(self, phase: StreamPhase, rows: torch.Tensor | None = None) -> torch.Tensor:
        source = self.audio_rows if rows is None else rows
        total = self.plan.audio_latent_count
        return (
            source.view(2, total, _AUDIO_ROW_WIDTH)[:, phase.audio_latent_start : phase.audio_latent_stop]
            .reshape(-1, _AUDIO_ROW_WIDTH)
            .clone()
        )


@dataclass
class _PhaseState:
    phase: StreamPhase
    branch: MiniMaxH3DenoiseBranch
    condition_rows: torch.Tensor
    media_rows: torch.Tensor
    token_tags: torch.Tensor
    commit_mask: torch.Tensor
    seq_len: int
    started_at: float = field(default_factory=time.perf_counter)
    timings: dict[str, float] = field(default_factory=dict)


class _Session:
    """Model-owned state of one streaming session (one WebSocket request)."""

    def __init__(
        self,
        *,
        session_id: str,
        canvas: CanvasGeometry,
        seed: int,
        contract: KVContract,
        video_decoder: StreamingVideoDecoder,
        audio_decoder: StreamingAudioDecoder,
        audio_kv_reset_requests: int,
        pad_text_tokens: int = 0,
    ) -> None:
        self.session_id = session_id
        self.canvas = canvas
        self.pad_text_tokens = int(pad_text_tokens)
        self.seed = int(seed)
        self.cache = CleanAVKVCache(contract)
        self.video_decoder = video_decoder
        self.audio_decoder = audio_decoder
        self.audio_kv_reset_requests = int(audio_kv_reset_requests)
        self.media_time_origin: int | None = None
        self.request_index = 0
        self.video_latent_offset = 0
        self.audio_latent_offset = 0
        self.frames_published = 0
        self.renorm_anchor: tuple[torch.Tensor, torch.Tensor] | None = None
        self.teacher_previous_clean: torch.Tensor | None = None
        self.teacher_previous_count: int | None = None
        self.request: _RequestState | None = None
        self.phase: _PhaseState | None = None
        self.sigmas_video, self.sigmas_audio = student_sigmas()
        self.stats: dict[str, float] = {"teacher_seconds": 0.0, "student_forwards": 0, "clean_forwards": 0}

    # -- request lifecycle -------------------------------------------------

    def begin_request(
        self,
        *,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        device: torch.device,
    ) -> _RequestState:
        index = self.request_index
        plan = request_plan(index)
        transport_video = REQUEST_VIDEO_LATENTS - plan.video_latent_count
        transport_audio = REQUEST_AUDIO_LATENTS - plan.audio_latent_count
        audio_seed = self.seed + index
        video_seed = (self.seed + index * _VIDEO_NOISE_REQUEST_STRIDE) % (2**63)
        _, official_audio = official_request_noise(seed=audio_seed, canvas=self.canvas)
        audio_rows = (
            official_audio.view(2, REQUEST_AUDIO_LATENTS, _AUDIO_ROW_WIDTH)[:, transport_audio:]
            .contiguous()
            .view(2 * plan.audio_latent_count, _AUDIO_ROW_WIDTH)
        )
        video_noise, _ = official_request_noise(seed=video_seed, canvas=self.canvas)
        frame_rows = self.canvas.frame_rows
        video_rows = minimax_h3_patchify_video_latent(video_noise, patch_size=(1, 2, 2))[transport_video * frame_rows :]
        text_len = int(text_embeddings.shape[0])
        if self.media_time_origin is None:
            self.media_time_origin = text_len
        if index > 0 and self.audio_kv_reset_requests > 0 and index % self.audio_kv_reset_requests == 0:
            dropped = self.cache.drop_audio_history()
            logger.info(
                "TaoMate-H3 session %s: dropped %d audio KV rows at request %d", self.session_id, dropped, index
            )
        self.request = _RequestState(
            index=index,
            plan=plan,
            text_embeddings=text_embeddings.detach(),
            text_tags=text_tags.detach(),
            video_rows=video_rows.to(device=device, dtype=torch.float32),
            audio_rows=audio_rows.to(device=device, dtype=torch.float32),
            audio_seed=audio_seed,
            video_seed=video_seed,
            frame_rows=frame_rows,
        )
        return self.request

    def finish_request(self) -> None:
        request = self.request
        assert request is not None
        self.video_latent_offset += request.plan.video_latent_count
        self.audio_latent_offset += request.plan.audio_latent_count
        self.request_index += 1
        self.request = None
        self.phase = None

    # -- pinned shapes ---------------------------------------------------------

    def pinned_phase_seq_len(self, phase: StreamPhase, text_len: int) -> int | None:
        """Fixed packed length for this phase kind, or ``None`` when unpinned."""
        if self.pad_text_tokens <= 0:
            return None
        if text_len > self.pad_text_tokens:
            logger.warning_once(
                "TaoMate-H3 prompt has %d tokens, above taomate_h3_pad_text_tokens=%d; this phase runs unpinned",
                text_len,
                self.pad_text_tokens,
            )
            return None
        audio_rows = 2 * _MAX_PHASE_AUDIO_LATENTS[phase.index]
        used = self.pad_text_tokens + audio_rows + phase.video_latent_count * self.canvas.frame_rows
        return -(-used // 64) * 64

    def pinned_teacher_seq_len(self, text_len: int, *, with_reference: bool) -> int | None:
        if self.pad_text_tokens <= 0 or text_len > self.pad_text_tokens:
            return None
        rows = 2 * (REQUEST_AUDIO_LATENTS + (ROLLOVER_LATENTS_PER_CHANNEL if with_reference else 0))
        return -(-(self.pad_text_tokens + rows) // 64) * 64

    # -- phases ------------------------------------------------------------

    def begin_phase(self, phase_index: int, *, transformer: TaoMateH3DiTModel, device: torch.device) -> _PhaseState:
        request = self.request
        assert request is not None and self.media_time_origin is not None
        phase = request.plan.phases[phase_index]
        packed = taomate_phase_packed_layout(
            text_len=request.text_len,
            phase=phase,
            latent_h=self.canvas.latent_h,
            latent_w=self.canvas.latent_w,
            media_time_origin=self.media_time_origin,
            video_latent_offset=self.video_latent_offset,
            audio_latent_offset=self.audio_latent_offset,
            seq_len=self.pinned_phase_seq_len(phase, request.text_len),
        )
        tags = packed["token_tags"].clone()
        tags[packed["text_pos"].view(-1)] = request.text_tags.detach().to("cpu", torch.long)
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=request.text_embeddings,
            token_tags=tags,
            device=device,
        )
        # The rank-local RoPE table is built lazily inside the first forward:
        # the sequence-parallel span is only known under the runner's forward
        # context, and ``prepare_next_chunk`` runs outside it.
        seq_len = int(packed["seq_len"])
        condition_rows = packed["text_pos"].view(-1).to(device=device, dtype=torch.long)
        media_rows = torch.cat((packed["audio_pos"].view(-1), packed["img_pos"].view(-1))).sort().values
        media_rows = media_rows.to(device=device, dtype=torch.long)
        commit_mask = torch.zeros(seq_len, dtype=torch.bool)
        commit_mask[media_rows.cpu()] = True
        self.phase = _PhaseState(
            phase=phase,
            branch=branch,
            condition_rows=condition_rows,
            media_rows=media_rows,
            token_tags=tags.to(device=device, dtype=torch.long),
            commit_mask=commit_mask.to(device=device),
            seq_len=seq_len,
        )
        return self.phase

    def stream_context(self, mode: StreamMode) -> StreamContext:
        phase = self.phase
        assert phase is not None
        return StreamContext(
            cache=self.cache,
            mode=mode,
            condition_rows=phase.condition_rows,
            media_rows=phase.media_rows,
            token_tags=phase.token_tags,
            commit_mask=phase.commit_mask,
            seq_len=phase.seq_len,
        )

    def renorm_clean_video_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Match generated phases to the first phase's per-feature statistics."""
        current = rows.detach().float()
        mean = current.mean(dim=0, keepdim=True)
        std = current.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        if self.renorm_anchor is None:
            self.renorm_anchor = (mean, std)
            return rows
        anchor_mean, anchor_std = self.renorm_anchor
        return (current - mean).div(std).mul(anchor_std).add(anchor_mean).to(dtype=rows.dtype)

    def close(self) -> None:
        self.cache.clear()
        self.video_decoder.reset()
        self.audio_decoder.reset()
        self.request = None
        self.phase = None
        self.renorm_anchor = None
        self.teacher_previous_clean = None
        self.teacher_previous_count = None


class TaoMateH3Pipeline(MiniMaxH3Pipeline, SupportsStepExecution, InteractionMixin):
    """Streaming TaoMate-H3 on the AR-Diffusion runtime."""

    supports_step_execution: ClassVar[bool] = True
    # The three student steps of a phase run back to back; the scheduler regains
    # control at the phase boundary, where prompt interactions are applied.
    supports_chunk_step_grouping: ClassVar[bool] = True
    # Generic warmup cannot synthesize a session; the AR runner's rollout
    # warmup (one throwaway request) replaces it, in eager mode too: a cold
    # server otherwise spends 17-20 s on its first chunk, 3 s when warm.
    dummy_run_num_frames: ClassVar[int] = 0
    ar_diffusion_warmup_eager: ClassVar[bool] = True
    _transformer_cls: ClassVar[type] = TaoMateH3DiTModel

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        _validate_parallel_config(od_config)
        if bool(getattr(od_config, "step_execution", False)) is False:
            logger.warning(
                "TaoMate-H3 is a streaming model; serve it with step_execution=true and streaming_output=true "
                "on the AR-Diffusion engine (see vllm_omni/deploy/taomate_h3_usp4_realtime.yaml)."
            )
        super().__init__(od_config=od_config, prefix=prefix)
        model_config = dict(getattr(od_config, "model_config", None) or {})
        self._tm_default_height = int(model_config.get("taomate_h3_height", DEFAULT_HEIGHT))
        self._tm_default_width = int(model_config.get("taomate_h3_width", DEFAULT_WIDTH))
        self._tm_default_canvas = CanvasGeometry(height=self._tm_default_height, width=self._tm_default_width)
        self._tm_default_seed = int(model_config.get("taomate_h3_seed", DEFAULT_SEED))
        self._tm_audio_kv_reset_requests = int(
            model_config.get("taomate_h3_audio_kv_reset_requests", DEFAULT_AUDIO_KV_RESET_REQUESTS)
        )
        self._tm_allow_no_lora = bool(model_config.get("taomate_h3_allow_no_lora", False))
        # Optional fixed packed lengths: with a prompt budget every phase kind
        # and the teacher documents keep one shape across prompts, which is
        # what compiled blocks and CUDA graphs need to stay warm.
        pad_text = model_config.get("taomate_h3_pad_text_tokens")
        self._tm_pad_text_tokens = int(pad_text) if pad_text else 0
        # Per-phase stage timings in the log (adds device synchronizations).
        self._tm_log_timings = bool(model_config.get("taomate_h3_log_timings", False))
        # Opt-in CUDA-graph replay of the audio teacher's nine launch-bound
        # forwards (one graph per document shape; see cuda_graph.py).
        self._tm_teacher_cuda_graph = bool(model_config.get("taomate_h3_teacher_cuda_graph", False))
        self._tm_cuda_graph_max_entries = int(model_config.get("taomate_h3_cuda_graph_max_entries", 16))
        if self._tm_cuda_graph_max_entries < 1:
            raise ValueError("taomate_h3_cuda_graph_max_entries must be at least 1")
        # Opt-in just-in-time prompt lock for clients that choose each request's
        # prompt at the last moment: request k >= 1 starts only after a prompt
        # update was applied since request k-1 started. Until then the session
        # idles at the request boundary (one short no-op step per poll), so the
        # stream cannot run ahead of the client and start a request on a stale
        # prompt. Off by default: the free-running stream keeps the last prompt.
        self._tm_hold_for_prompt = bool(model_config.get("taomate_h3_hold_for_prompt", False))
        self._tm_hold_poll_seconds = float(model_config.get("taomate_h3_hold_poll_seconds", 0.02))
        # The executor returns the output of DiT rank 0; the other ranks take
        # part in the collectives but do not convert or decode media.
        self._tm_output_rank = int(self._dit_rank) == 0
        self.taomate_lora: TaoMateLoRAAdapter | None = None
        self._tm_teacher: TaoMateAudioTeacher | None = None
        self._tm_teacher_graph: GraphedForward | None = None
        self._tm_sessions: dict[str, _Session] = {}
        self._ar_diffusion_kv_state: ARDiffusionKVState | None = None
        if getattr(self, "transformers_ref", None) is not None:
            raise ValueError("TaoMate-H3 serves the FL2VA partition only; do not load the Ref2VA DiT")
        if not self.load_text_encoder:
            raise ValueError("TaoMate-H3 encodes each request prompt locally; keep text_encoder loaded")

    # -- weights ----------------------------------------------------------

    def load_weights(self, weights):  # type: ignore[override]
        loaded = super().load_weights(weights)
        lora_path = getattr(self.od_config, "lora_path", None)
        if isinstance(lora_path, (list, tuple)):
            lora_path = lora_path[0] if len(lora_path) == 1 else None
        if lora_path and is_taomate_lora_dir(lora_path):
            self.taomate_lora = TaoMateLoRAAdapter.load(
                lora_path,
                transformer=self.transformer,
                device=self.device,
                dtype=torch.bfloat16,
            )
        elif lora_path:
            raise ValueError(
                f"--lora-path {lora_path!r} is not a TaoMate-H3 native adapter directory "
                "(expected adapter_model.safetensors with <target>.lora_a/lora_b keys and config.json)"
            )
        elif not self._tm_allow_no_lora:
            raise ValueError(
                "TaoMate-H3 requires --lora-path pointing at the TaoLiveAIGC/TaoMate-H3 adapter; "
                "set model_config.taomate_h3_allow_no_lora=true to stream the base H3 without it"
            )
        graph: GraphedForward | None = None
        if self._tm_teacher_cuda_graph:
            _, _, group = _ulysses_state()
            graph = GraphedForward(
                self.transformer,
                device=self.device,
                max_entries=self._tm_cuda_graph_max_entries,
                group=group,
                name="TaoMate-H3 teacher",
            )
            if not graph.enabled:
                logger.warning("TaoMate-H3 teacher CUDA graphs requested but the device is not CUDA; running eager")
        self._tm_teacher_graph = graph
        self._tm_teacher = TaoMateAudioTeacher(
            self.transformer, lora=self.taomate_lora, device=self.device, graph=graph
        )
        return loaded

    @property
    def lora_is_fused(self) -> bool:
        # The adapter is applied by this pipeline; keep the dynamic LoRA manager away.
        return True

    # -- prompts ----------------------------------------------------------

    def _prepared_text_inputs(self, prompt: str, *, canvas: CanvasGeometry) -> PreparedEncoderInputs:
        if not isinstance(prompt, str) or not prompt.strip():
            raise OmniClientError("TaoMate-H3 requires a non-empty text prompt")
        media = MiniMaxH3EncoderMediaInput(
            task="t2va",
            height=canvas.height,
            width=canvas.width,
            num_frames=int(REQUEST_VIDEO_LATENTS),
            latent_t=REQUEST_VIDEO_LATENTS,
            audio_t=REQUEST_AUDIO_LATENTS,
        )
        return PreparedEncoderInputs(
            prompt=prompt,
            media=media,
            images=[],
            qwen_videos=[],
            video_timestamps=[],
            condition_labels=[],
        )

    def encode_prompt(  # type: ignore[override]
        self,
        prepared: PreparedEncoderInputs | None = None,
        *,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one prompt to ``(hidden [L, 5120] bf16, tags [L])`` on every DiT rank.

        Also the entry point of the streaming prompt interaction handler, which
        calls it with ``prompt=`` and the generic keyword arguments.
        """
        del kwargs
        if prepared is None:
            if prompt is None:
                raise ValueError("encode_prompt needs prepared inputs or a prompt string")
            prepared = self._prepared_text_inputs(prompt, canvas=self._tm_default_canvas)
        return super().encode_prompt(prepared)

    # -- AR-Diffusion capability --------------------------------------------

    def _kv_contract(self) -> KVContract:
        return KVContract(
            num_layers=int(self.transformer.arch.num_layers),
            local_heads=int(self.transformer.streaming_local_heads),
            head_dim=int(self.transformer.arch.attention_head_dim),
            dtype=torch.bfloat16,
        )

    def _dense_kv_rows_bound(self, canvas: CanvasGeometry) -> int:
        # Sink (12 video latents) + two recent AV chunks (10 latents each) +
        # one staged chunk (12 latents) plus their audio rows.
        video_latents = 12 + 10 + 10 + 12
        audio_rows = 2 * (60 + 56 + 56 + 60)
        return video_latents * canvas.frame_rows + audio_rows

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        """Declare the (tiny) paged geometry and the model-owned dense state."""
        canvas = self._tm_default_canvas
        contract = self._kv_contract()
        # The dense history is rebuilt by concatenation at commit time, so two
        # copies of one layer coexist transiently; budget the whole cache twice.
        dense_bytes = 2 * contract.bytes_for_rows(self._dense_kv_rows_bound(canvas))
        latent_bytes = 24 * REQUEST_VIDEO_LATENTS * canvas.latent_h * canvas.latent_w * 4 * 2
        audio_bytes = 2 * _AUDIO_ROW_WIDTH * 4 * REQUEST_AUDIO_LATENTS * 8
        decoder_bytes = (
            24 * 20 * canvas.latent_h * canvas.latent_w * 4 + 3 * 5 * canvas.height * canvas.width * 4 + 64 * 1024**2
        )
        return ARDiffusionKVCacheSpec(
            num_layers=contract.num_layers,
            num_kv_heads=contract.local_heads,
            head_size=contract.head_dim,
            tokens_per_frame=canvas.frame_rows,
            frames_per_block=1,
            window_frames=1,
            sink_frames=0,
            kv_branches=(ARDiffusionKVBranchSpec(_AR_BRANCH, 0),),
            session_capacity=1,
            model_owned_state_bytes_per_session=dense_bytes + latent_bytes + audio_bytes + decoder_bytes,
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id: str, state: ARDiffusionKVState) -> Iterator[None]:
        if self._ar_diffusion_kv_state is not None:
            raise RuntimeError("TaoMate-H3 AR-Diffusion state is already bound")
        if state.session_id != session_id:
            raise ValueError(f"TaoMate-H3 bound session mismatch: {state.session_id!r} != {session_id!r}")
        self._ar_diffusion_kv_state = state
        try:
            yield
        finally:
            self._ar_diffusion_kv_state = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self._release_session(session_id)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self._release_session(session_id)

    def _release_session(self, session_id: str) -> None:
        session = self._tm_sessions.pop(session_id, None)
        if session is not None:
            session.close()
            if torch.cuda.is_available():
                torch.accelerator.empty_cache()

    def _require_bound_ar_state(self) -> None:
        if self._ar_diffusion_kv_state is None:
            raise RuntimeError(
                "TaoMate-H3 step execution requires the AR-Diffusion engine "
                "(engine_backend: vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine)"
            )

    def _bound_session_id(self, state: StepRequestState) -> str:
        bound = self._ar_diffusion_kv_state
        return bound.session_id if bound is not None else state.request_id

    def _session(self, state: StepRequestState) -> _Session:
        key = state.extra.get("taomate_session_id") or self._bound_session_id(state)
        session = self._tm_sessions.get(key)
        if session is None:
            raise RuntimeError(f"TaoMate-H3 has no session {key!r} for request {state.request_id!r}")
        return session

    # -- request mode: a whole session in one call (offline use and the AR warmup) --

    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:  # type: ignore[override]
        """Generate every phase of the requested frame budget and return the joined media.

        Request mode under the AR-Diffusion runner (the session KV must be
        bound by the runner, so this is not a standalone offline path): it
        drives the same step contract to completion in one runner invocation
        and serves the runner's load-time warmup. Streaming clients use the
        WebSocket realtime endpoint.
        """
        if request.num_reqs != 1:
            raise OmniClientError("TaoMate-H3 request mode serves one request at a time")
        req = request.requests[0]
        state = StepRequestState(request_id=req.request_id, sampling=req.sampling_params, prompt=req.prompt)
        self.prepare_encode(state)
        state.extra["taomate_no_hold"] = True  # one call, no client to wait for
        self.prepare_next_chunk(state)
        frames: list[np.ndarray] = []
        audio: list[np.ndarray] = []
        chunk_metadata: list[dict[str, Any]] = []
        last: DiffusionOutput | None = None
        while not state.request_denoise_completed:
            for _ in range(STUDENT_STEPS):
                velocity = self.denoise_step(cast(InputBatch, None), states=[state])
                assert velocity is not None
                self.step_scheduler(state, velocity)
            last = self.post_decode(state)
            envelope = cast(dict[str, Any], last.output)
            frames.append(envelope["payload"]["video"])
            audio.append(envelope["payload"]["audio"])
            chunk_metadata.append(envelope["metadata"]["taomate_h3"])
            if not state.request_denoise_completed:
                self.prepare_next_chunk(state)
        assert last is not None
        session_id = self._bound_session_id(state)
        video = np.concatenate([item for item in frames if item.shape[0]], axis=0) if frames else np.zeros((0, 1, 1, 3))
        waveform = np.concatenate(audio, axis=0) if audio else np.zeros((0, 2), np.float32)
        return DiffusionOutput(
            output={
                "payload": {"video": video, "audio": waveform},
                "metadata": {
                    "video": {"fps": float(VIDEO_FPS)},
                    "audio": {"sample_rate": AUDIO_SAMPLE_RATE},
                    "ar_diffusion": ARDiffusionChunkMetadata(
                        session_id=session_id,
                        request_id=req.request_id,
                        chunk_index=state.chunk_index - 1,
                        applied_event_ids=(),
                    ).to_dict(),
                    "taomate_h3": {"chunks": chunk_metadata},
                },
            },
            chunk_index=state.chunk_index - 1,
            total_chunks=state.total_chunks,
            finished=True,
        )

    def ar_diffusion_warmup_requests(self, session_id: str) -> Iterator[OmniDiffusionRequest]:
        """One five-second request that exercises every phase shape, the teacher and the decoders."""
        yield OmniDiffusionRequest(
            prompt="A presenter smiles at the camera and greets the audience warmly.",
            sampling_params=OmniDiffusionSamplingParams(
                num_frames=REQUEST_NATIVE_FRAMES,
                height=self._tm_default_height,
                width=self._tm_default_width,
                seed=self._tm_default_seed,
                extra_args={"session_id": session_id, "reset": True},
            ),
            request_id=f"taomate-h3-warmup-{session_id}",
        )

    # -- step execution ------------------------------------------------------

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        del kwargs
        self._require_bound_ar_state()
        if self._tm_teacher is None:
            raise RuntimeError("TaoMate-H3 weights are not loaded")
        sampling = state.sampling
        if int(getattr(sampling, "num_outputs_per_prompt", 1) or 1) != 1:
            raise OmniClientError("TaoMate-H3 produces one stream per session; num_outputs_per_prompt must be 1")
        prompt_text, _ = self._extract_prompt(state.prompt)
        height = int(getattr(sampling, "height", None) or self._tm_default_height)
        width = int(getattr(sampling, "width", None) or self._tm_default_width)
        try:
            canvas = CanvasGeometry(height=height, width=width)
        except ValueError as exc:
            raise OmniClientError(str(exc)) from exc
        if canvas != self._tm_default_canvas:
            raise OmniClientError(
                f"TaoMate-H3 was deployed for {self._tm_default_width}x{self._tm_default_height}; "
                f"request asked for {width}x{height}. Set model_config.taomate_h3_width/height."
            )
        seed = getattr(sampling, "seed", None)
        seed = self._tm_default_seed if seed is None else int(seed)
        num_frames = int(getattr(sampling, "num_frames", None) or 0) or DEFAULT_SESSION_FRAMES
        total_chunks = phases_for_frames(num_frames)
        hidden, tags = self.encode_prompt(prompt=prompt_text)
        state.prompt_embeds = hidden
        session_id = self._bound_session_id(state)
        previous = self._tm_sessions.pop(session_id, None)
        if previous is not None:
            previous.close()
        session = _Session(
            session_id=session_id,
            canvas=canvas,
            seed=seed,
            contract=self._kv_contract(),
            video_decoder=StreamingVideoDecoder(
                self.video_vae,
                device=self.device,
                height=canvas.height,
                width=canvas.width,
                emit_frames=self._tm_output_rank,
            ),
            audio_decoder=StreamingAudioDecoder(self.audio_vae, device=self.device),
            audio_kv_reset_requests=self._tm_audio_kv_reset_requests,
            pad_text_tokens=self._tm_pad_text_tokens,
        )
        self._tm_sessions[session_id] = session
        state.chunk_num_steps = STUDENT_STEPS
        state.total_chunks = total_chunks
        state.chunk_index = 0
        state.step_index = 0
        state.step_in_chunk = 0
        state.do_true_cfg = False
        state.timesteps = torch.tensor(
            [1.0 - sigma for sigma in session.sigmas_video[:-1]], dtype=torch.float32, device=self.device
        )
        state.extra = {"text_tags": tags, "audio_rows": None, "taomate_session_id": session_id}
        logger.info(
            "TaoMate-H3 session %s: %dx%d, seed=%d, %d chunk(s) (%d request(s)), prompt=%.40r",
            session_id,
            canvas.width,
            canvas.height,
            seed,
            total_chunks,
            total_chunks // NUM_PHASES,
            prompt_text,
        )
        return state

    def prepare_next_chunk(self, state: StepRequestState) -> None:
        """Start the next phase (and the next request at every fourth chunk)."""
        if state.request_denoise_completed:
            return
        session = self._session(state)
        phase_index = state.chunk_index % NUM_PHASES
        if phase_index == 0:
            version = _prompt_version(state)
            if (
                self._tm_hold_for_prompt
                and state.chunk_index >= NUM_PHASES
                and not state.extra.get("taomate_no_hold")
                and version <= int(state.extra.get("taomate_prompt_version", 0))
            ):
                # Hold at the request boundary until the client locks this
                # request's prompt; denoise_step polls (see _tm_hold_for_prompt).
                state.extra["taomate_held"] = True
                # the runner batches every scheduled state: keep a valid current timestep
                # and never re-decode the finished phase while idling
                state.step_in_chunk = 0
                state.step_index = 0
                # Idle steps emit nothing: stop the AR runner from grouping them as
                # one chunk's denoise steps, so every idle step returns to the
                # scheduler (where the client's prompt update arrives).
                self.supports_chunk_step_grouping = False
                return
            self.__dict__.pop("supports_chunk_step_grouping", None)  # back to the class policy
            state.extra["taomate_held"] = False
            state.extra["taomate_prompt_version"] = version
            if state.prompt_embeds is None:
                raise RuntimeError("TaoMate-H3 request needs prompt embeddings")
            text_embeddings = state.prompt_embeds
            # The step runner batches prompt embeddings as [1, L, D]; TaoMate
            # packs plain text rows.
            if text_embeddings.ndim == 3 and text_embeddings.shape[0] == 1:
                text_embeddings = text_embeddings[0]
            if text_embeddings.ndim != 2:
                raise RuntimeError(
                    f"TaoMate-H3 prompt embeddings must be [L, 5120], got {tuple(text_embeddings.shape)}"
                )
            tags = state.extra.get("text_tags")
            if tags is None or int(tags.shape[0]) != int(text_embeddings.shape[0]):
                # A prompt interaction replaced the embeddings; TaoMate prompts are plain text rows.
                tags = torch.ones(int(text_embeddings.shape[0]), dtype=torch.long, device=text_embeddings.device)
                state.extra["text_tags"] = tags
            session.begin_request(text_embeddings=text_embeddings, text_tags=tags, device=self.device)
        request = session.request
        assert request is not None
        phase_state = session.begin_phase(phase_index, transformer=self.transformer, device=self.device)
        state.latents = request.phase_video_rows(phase_state.phase)
        state.extra["audio_rows"] = request.phase_audio_rows(phase_state.phase)
        state.timesteps = torch.tensor(
            [1.0 - sigma for sigma in session.sigmas_video[:-1]], dtype=torch.float32, device=self.device
        )
        state.step_index = 0
        state.step_in_chunk = 0

    def _run_teacher(self, session: _Session) -> None:
        request = session.request
        assert request is not None and self._tm_teacher is not None
        started = time.perf_counter()
        result = self._tm_teacher.run_request(
            text_embeddings=request.text_embeddings,
            text_tags=request.text_tags,
            canvas=session.canvas,
            plan=request.plan,
            audio_seed=request.audio_seed,
            previous_clean=session.teacher_previous_clean,
            previous_audio_latent_count=session.teacher_previous_count,
            seq_len=session.pinned_teacher_seq_len(
                request.text_len, with_reference=session.teacher_previous_clean is not None
            ),
        )
        request.teacher = result
        request.teacher_seconds = time.perf_counter() - started
        session.stats["teacher_seconds"] += request.teacher_seconds
        session.teacher_previous_clean = result.clean
        session.teacher_previous_count = result.audio_latent_count
        # The request's clean audio is final now; make it available to the
        # sliding-window audio decoder ahead of the phases that publish it.
        session.audio_decoder.append(
            minimax_h3_unpack_audio_tokens(result.clean, audio_t=2 * result.audio_latent_count, audio_channel=2)
        )

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        del input_batch, kwargs
        if states is None or len(states) != 1:
            raise ValueError("TaoMate-H3 step execution serves one session per forward")
        self._require_bound_ar_state()
        state = states[0]
        if state.extra.get("taomate_held"):
            # Still at a held request boundary: apply any prompt the client sent
            # since the last step (the same interaction sequence reaches every
            # rank, so all ranks decide alike), then start the request or idle.
            self.apply_interaction_at_chunk_boundary(state)
            self.prepare_next_chunk(state)
            if state.extra.get("taomate_held"):
                time.sleep(self._tm_hold_poll_seconds)
                return None
        session = self._session(state)
        request = session.request
        phase_state = session.phase
        if request is None or phase_state is None or state.latents is None:
            raise RuntimeError("TaoMate-H3 denoise_step called before prepare_next_chunk")
        if request.teacher is None:
            self._run_teacher(session)
        step = state.step_in_chunk
        if step >= STUDENT_STEPS:
            raise RuntimeError(f"TaoMate-H3 phase has only {STUDENT_STEPS} denoise steps, got step {step}")
        t_video = 1.0 - session.sigmas_video[step]
        t_audio = 1.0 - session.sigmas_audio[step]
        if "rope_table" not in phase_state.branch.static_kwargs:
            phase_state.branch.prepare_rope_table(self.transformer)
        forward_kwargs = phase_state.branch.forward_kwargs(
            video_rows=state.latents,
            audio_rows=state.extra["audio_rows"],
            t_video=t_video,
            t_audio=t_audio,
            imgvid_cond_timestep=t_video,
            audio_ref_cond_timestep=1.0,
        )
        started = time.perf_counter()
        with stream_context(session.stream_context(StreamMode.NOISY)), torch.inference_mode():
            velocity_video, _ = self.transformer(**forward_kwargs)
        if self._tm_log_timings:
            if velocity_video.is_cuda:
                torch.accelerator.synchronize()
        phase_state.timings["denoise"] = phase_state.timings.get("denoise", 0.0) + (time.perf_counter() - started)
        session.stats["student_forwards"] += 1
        return velocity_video.float()

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor | None, **kwargs: Any) -> None:
        del kwargs
        if noise_pred is None and state.extra.get("taomate_held"):
            return  # idle step at a held request boundary: nothing advances
        session = self._session(state)
        request = session.request
        phase_state = session.phase
        if request is None or phase_state is None or state.latents is None or request.teacher is None:
            raise RuntimeError("TaoMate-H3 step_scheduler called out of order")
        step = state.step_in_chunk
        video_rows = state.latents.clone()
        euler_eta0_update_(
            video_rows,
            noise_pred.float(),
            sigma_curr=session.sigmas_video[step],
            sigma_next=session.sigmas_video[step + 1],
        )
        state.latents = video_rows
        # The student's audio velocity is discarded: audio comes from the base
        # teacher's milestone for this step (states 3, 6, 9).
        state.extra["audio_rows"] = request.phase_audio_rows(phase_state.phase, request.teacher.milestones[step])
        state.step_in_chunk += 1
        state.step_index += 1

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        self._require_bound_ar_state()
        session = self._session(state)
        request = session.request
        phase_state = session.phase
        if request is None or phase_state is None or state.latents is None or request.teacher is None:
            raise RuntimeError("TaoMate-H3 post_decode called out of order")
        phase = phase_state.phase
        audio_rows = state.extra["audio_rows"]
        clean_video = session.renorm_clean_video_rows(state.latents)
        # Sigma-zero forward: recompute this chunk's K/V from its clean latents
        # and append them to the persistent history.
        cache = session.cache
        commit_started = time.perf_counter()
        cache.begin_clean_commit(cache.committed_blocks)
        try:
            forward_kwargs = phase_state.branch.forward_kwargs(
                video_rows=clean_video,
                audio_rows=audio_rows,
                t_video=1.0,
                t_audio=1.0,
                imgvid_cond_timestep=1.0,
                audio_ref_cond_timestep=1.0,
            )
            with stream_context(session.stream_context(StreamMode.CLEAN_COMMIT)), torch.inference_mode():
                self.transformer(**forward_kwargs)
            cache.commit()
        except BaseException:
            cache.rollback()
            raise
        cache.retain_sink_and_recent_commits()
        session.stats["clean_forwards"] += 1
        if self._tm_log_timings and torch.cuda.is_available():
            torch.accelerator.synchronize()
        phase_state.timings["commit"] = time.perf_counter() - commit_started

        # Publish this phase: incremental video decode plus the aligned audio.
        decode_started = time.perf_counter()
        latent = minimax_h3_unpatchify_video_tokens(
            clean_video,
            latent_shape=(phase.video_latent_count, session.canvas.latent_h // 2, session.canvas.latent_w // 2, 24),
            patch_size=(1, 2, 2),
        )
        frame_start, frames = session.video_decoder.push(latent)
        phase_state.timings["video_decode"] = time.perf_counter() - decode_started
        completed_chunk_index = state.chunk_index
        state.chunk_index += 1
        last_phase = (completed_chunk_index % NUM_PHASES) == NUM_PHASES - 1
        session_done = state.request_denoise_completed
        if session_done:
            flush_start, flushed = session.video_decoder.flush()
            if flushed is not None:
                frames = flushed if frames is None else np.concatenate((frames, flushed), axis=0)
                if frames is flushed:
                    frame_start = flush_start
        num_frames = 0 if frames is None else int(frames.shape[0])
        audio_started = time.perf_counter()
        if self._tm_output_rank:
            audio = session.audio_decoder.decode_range(
                samples_at_frame(frame_start), samples_at_frame(frame_start + num_frames)
            )
        else:
            # The executor returns rank 0's output; peers keep the decoder
            # timeline but skip the waveform decode.
            session.audio_decoder.skip_range(samples_at_frame(frame_start + num_frames))
            audio = np.zeros((0, 2), dtype=np.float32)
        phase_state.timings["audio_decode"] = time.perf_counter() - audio_started
        session.frames_published = frame_start + num_frames
        phase_seconds = time.perf_counter() - phase_state.started_at
        if self._tm_log_timings:
            graph = self._tm_teacher_graph
            logger.info(
                "TaoMate-H3 %s request %d phase %d: %.3f s total (teacher %.3f, denoise x3 %.3f, commit %.3f, "
                "video decode %.3f, audio decode %.3f), %d frames, history %d rows%s",
                session.session_id,
                request.index,
                phase.index,
                phase_seconds,
                request.teacher_seconds if phase.index == 0 else 0.0,
                phase_state.timings.get("denoise", 0.0),
                phase_state.timings.get("commit", 0.0),
                phase_state.timings.get("video_decode", 0.0),
                phase_state.timings.get("audio_decode", 0.0),
                num_frames,
                cache.history_tokens,
                "" if graph is None else f", teacher graphs {graph.stats()}",
            )
        payload: dict[str, Any] = {
            "video": frames
            if frames is not None
            else np.zeros((0, session.canvas.height, session.canvas.width, 3), np.uint8),
            "audio": audio,
        }
        metadata: dict[str, Any] = {
            "video": {"fps": float(VIDEO_FPS)},
            "audio": {"sample_rate": AUDIO_SAMPLE_RATE},
            "ar_diffusion": ARDiffusionChunkMetadata(
                session_id=session.session_id,
                request_id=state.request_id,
                chunk_index=completed_chunk_index,
                applied_event_ids=(),
            ).to_dict(),
            "taomate_h3": {
                "request_index": request.index,
                "phase_index": phase.index,
                "frame_start": frame_start,
                "num_frames": num_frames,
                "audio_samples": int(audio.shape[0]),
                "history_tokens": cache.history_tokens,
                "teacher_seconds": round(request.teacher_seconds, 4),
                "phase_seconds": round(phase_seconds, 4),
                **{f"{name}_seconds": round(value, 4) for name, value in phase_state.timings.items()},
            },
        }
        if last_phase:
            session.finish_request()
        return DiffusionOutput(
            output={"payload": payload, "metadata": metadata},
            chunk_index=completed_chunk_index,
            total_chunks=state.total_chunks,
            finished=session_done,
        )

    # -- interaction hooks ----------------------------------------------------

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        """Frames of the next phase (approximate: the decoder holds a five-frame overlap)."""
        chunk = state.chunk_index
        plan = request_plan(chunk // NUM_PHASES)
        phase = plan.phases[chunk % NUM_PHASES]
        fps = getattr(state.sampling, "fps", None) or float(VIDEO_FPS)
        return ChunkMediaSpec(
            num_media_frames=phase.frame_count,
            fps=float(fps),
            num_latent_frames=phase.video_latent_count,
        )


def get_taomate_h3_post_process_func(od_config: OmniDiffusionConfig):
    """Pass the payload/metadata envelope through untouched."""
    del od_config

    def post_process_func(output: Any, output_type: str = "np", sampling_params: Any | None = None) -> Any:
        del output_type, sampling_params
        if isinstance(output, dict) and isinstance(output.get("payload"), dict):
            return output
        return {"payload": {"video": output}, "metadata": {}}

    return post_process_func


__all__ = ["TaoMateH3Pipeline", "get_taomate_h3_post_process_func"]
