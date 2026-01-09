# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GlmImagePipeline implementation for vLLM-Omni.

This pipeline implements GLM-Image text-to-image generation with:
- AR stage: GlmImageForConditionalGeneration generates prior tokens
- DiT stage: GlmImageTransformer2DModel performs diffusion denoising
- VAE: AutoencoderKL decodes latents to images
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
from collections.abc import Iterable
from math import sqrt

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import (
    ByT5Tokenizer,
    GlmImageForConditionalGeneration,
    GlmImageProcessor,
    T5EncoderModel,
)

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.glm_image.glm_image_transformer import (
    GlmImageTransformer2DModel,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)


def get_glm_image_post_process_func(od_config: OmniDiffusionConfig):
    """Get post-processing function for GLM-Image pipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        block_out_channels = vae_config.get("block_out_channels", [128, 256, 512, 512])
        vae_scale_factor = 2 ** (len(block_out_channels) - 1)

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    base_shift: float = 0.25,
    max_shift: float = 0.75,
) -> float:
    """Calculate timestep shift based on image sequence length."""
    m = (image_seq_len / base_seq_len) ** 0.5
    mu = m * max_shift + base_shift
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps.
    Handles custom timesteps and sigmas schedules.
    """
    accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())

    if timesteps is not None and sigmas is not None:
        # Both provided - check if scheduler supports both
        if not accepts_timesteps and not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep or sigma schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is not None:
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        if not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigma schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    """Extract latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class GlmImagePipeline(nn.Module):
    """
    GLM-Image Pipeline for text-to-image and image-to-image generation.

    This pipeline integrates:
    - AR model (GlmImageForConditionalGeneration): Generates prior image tokens
    - Text encoder (T5EncoderModel): Encodes glyph/text embeddings
    - DiT model (GlmImageTransformer2DModel): Diffusion transformer
    - VAE (AutoencoderKL): Encodes/decodes images to/from latent space

    The pipeline flow:
    1. AR generates prior_token_ids from text prompt
    2. T5 encodes glyph text for text rendering
    3. DiT performs iterative denoising conditioned on prior tokens
    4. VAE decodes final latents to image
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()

        model = od_config.model
        local_files_only = os.path.exists(model)

        if local_files_only:
            model_path = model
        else:
            model_path = download_weights_from_hf_specific(model, od_config.revision, ["*"])

        # Load scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler", local_files_only=True
        )

        # Load AR model (vision_language_encoder)
        logger.info("Loading GlmImageForConditionalGeneration (AR model)...")
        self.vision_language_encoder = GlmImageForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="vision_language_encoder",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.vision_language_encoder.eval()

        # Load processor for AR model
        self.processor = GlmImageProcessor.from_pretrained(model_path, subfolder="processor", local_files_only=True)

        # Load text encoder (T5 for glyph embeddings)
        logger.info("Loading T5EncoderModel (glyph encoder)...")
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.text_encoder.eval()

        # Load tokenizer for glyph encoding
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)

        # Load VAE
        logger.info("Loading AutoencoderKL (VAE)...")
        self.vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", local_files_only=True, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.vae.eval()

        # Load transformer (DiT)
        logger.info("Loading GlmImageTransformer2DModel (DiT)...")
        self.transformer = GlmImageTransformer2DModel(od_config=od_config)

        # Weight sources for DiT loading
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        # Configure scale factors
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = 128

        # Get transformer config for patch size
        self._patch_size = getattr(self.transformer, "patch_size", 2)

    # ==================== AR Stage Methods ====================

    @staticmethod
    def _build_image_grid_thw(
        token_h: int,
        token_w: int,
        prev_token_h: int,
        prev_token_w: int,
        existing_grid: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build image grid tensor for AR model."""
        if existing_grid is None or existing_grid.numel() == 0:
            return torch.tensor(
                [
                    [1, token_h, token_w],
                    [1, prev_token_h, prev_token_w],
                ],
                device=device,
            )
        else:
            return torch.cat(
                [existing_grid.to(device), torch.tensor([[1, token_h, token_w]], device=device)],
                dim=0,
            )

    @staticmethod
    def _calculate_ar_generation_params(
        token_h: int, token_w: int, prev_token_h: int, prev_token_w: int, is_text_to_image: bool
    ) -> tuple[int, int]:
        """Calculate AR generation parameters."""
        large_image_tokens = token_h * token_w
        small_image_tokens = prev_token_h * prev_token_w

        if is_text_to_image:
            max_new_tokens = small_image_tokens + large_image_tokens + 1
            large_image_start_offset = small_image_tokens
        else:
            max_new_tokens = large_image_tokens + 1
            large_image_start_offset = 0

        return max_new_tokens, large_image_start_offset

    @staticmethod
    def _extract_large_image_tokens(
        outputs: torch.Tensor, input_length: int, large_image_start_offset: int, large_image_tokens: int
    ) -> torch.Tensor:
        """Extract large image tokens from AR output."""
        generated_tokens = outputs[0][input_length:]
        large_image_start = large_image_start_offset
        large_image_end = large_image_start + large_image_tokens
        return generated_tokens[large_image_start:large_image_end]

    @staticmethod
    def _upsample_token_ids(token_ids: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
        """Upsample token IDs by 2x using nearest neighbor interpolation."""
        token_ids = token_ids.view(1, 1, token_h, token_w)
        token_ids = torch.nn.functional.interpolate(token_ids.float(), scale_factor=2, mode="nearest").to(
            dtype=torch.long
        )
        token_ids = token_ids.view(1, -1)
        return token_ids

    @staticmethod
    def _build_prompt_with_shape(
        prompt: str,
        height: int,
        width: int,
        is_text_to_image: bool,
        factor: int = 32,
    ) -> tuple[str, int, int, int, int]:
        """Build prompt with shape information for AR model."""
        token_h = height // factor
        token_w = width // factor
        ratio = token_h / token_w
        prev_token_h = int(sqrt(ratio) * (factor // 2))
        prev_token_w = int(sqrt(1 / ratio) * (factor // 2))

        if is_text_to_image:
            expanded_prompt = f"{prompt}<sop>{token_h} {token_w}<eop><sop>{prev_token_h} {prev_token_w}<eop>"
        else:
            expanded_prompt = f"{prompt}<sop>{token_h} {token_w}<eop>"

        return expanded_prompt, token_h, token_w, prev_token_h, prev_token_w

    @torch.inference_mode()
    def generate_prior_tokens(
        self,
        prompt: str,
        height: int,
        width: int,
        image: list[PIL.Image.Image] | None = None,
        factor: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
        """
        Generate prior tokens using the AR model.

        Args:
            prompt: Text prompt for generation
            height: Target image height
            width: Target image width
            image: Optional condition images for image-to-image
            factor: Token factor (default 32)

        Returns:
            Tuple of (prior_token_ids, prior_token_image_ids, pixel_height, pixel_width)
        """
        device = self.vision_language_encoder.device
        height = (height // factor) * factor
        width = (width // factor) * factor
        is_text_to_image = image is None or len(image) == 0

        expanded_prompt, token_h, token_w, prev_h, prev_w = self._build_prompt_with_shape(
            prompt, height, width, is_text_to_image
        )

        # Build message content
        content = []
        if image is not None:
            for img in image:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": expanded_prompt})
        messages = [{"role": "user", "content": content}]

        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Build image grid
        existing_grid = inputs.get("image_grid_thw")
        inputs["image_grid_thw"] = self._build_image_grid_thw(
            token_h,
            token_w,
            prev_h,
            prev_w,
            existing_grid=existing_grid if not is_text_to_image else None,
            device=device,
        )

        max_new_tokens, large_image_offset = self._calculate_ar_generation_params(
            token_h, token_w, prev_h, prev_w, is_text_to_image
        )
        large_image_tokens = token_h * token_w

        inputs = inputs.to(device)
        input_length = inputs["input_ids"].shape[-1]

        # Process condition images if provided
        prior_token_image_ids = None
        if image is not None and existing_grid is not None:
            prior_token_image_embed = self.vision_language_encoder.get_image_features(
                inputs["pixel_values"], existing_grid
            )
            prior_token_image_embed = torch.cat(prior_token_image_embed, dim=0)
            prior_token_image_ids = self.vision_language_encoder.get_image_tokens(
                prior_token_image_embed, existing_grid
            )

        # Generate with AR model
        outputs = self.vision_language_encoder.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )

        # Extract and upsample tokens
        prior_token_ids_d32 = self._extract_large_image_tokens(
            outputs, input_length, large_image_offset, large_image_tokens
        )
        prior_token_ids = self._upsample_token_ids(prior_token_ids_d32, token_h, token_w)

        pixel_height = token_h * factor
        pixel_width = token_w * factor

        return prior_token_ids, prior_token_image_ids, pixel_height, pixel_width

    # ==================== Text Encoding Methods ====================

    def get_glyph_texts(self, prompt: str | list[str]) -> list[str]:
        """Extract text within quotes for glyph rendering."""
        prompt = prompt[0] if isinstance(prompt, list) else prompt
        ocr_texts = (
            re.findall(r"'([^']*)'", prompt)
            + re.findall(r"“([^“”]*)”", prompt)
            + re.findall(r'"([^"]*)"', prompt)
            + re.findall(r"「([^「」]*)」", prompt)
        )
        return ocr_texts

    def _get_glyph_embeds(
        self,
        prompt: str | list[str],
        max_sequence_length: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Get glyph embeddings from T5 encoder for text rendering."""
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        glyph_texts = self.get_glyph_texts(prompt)
        input_ids = self.tokenizer(
            glyph_texts if len(glyph_texts) > 0 else [""],
            max_length=max_sequence_length,
            truncation=True,
        ).input_ids

        # Pad to even length
        input_ids = [[self.tokenizer.pad_token_id] * ((len(ids) + 1) % 2) + ids for ids in input_ids]
        max_length = max(len(ids) for ids in input_ids)

        attention_mask = torch.tensor(
            [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in input_ids],
            device=device,
        )
        input_ids = torch.tensor(
            [ids + [self.tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids],
            device=device,
        )

        outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        glyph_embeds = outputs.last_hidden_state[attention_mask.bool()].unsqueeze(0)

        return glyph_embeds.to(device=device, dtype=dtype)

    def encode_prompt(
        self,
        prompt: str | list[str],
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode prompt into glyph embeddings for text rendering."""
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_glyph_embeds(prompt, max_sequence_length, device, dtype)

        seq_len = prompt_embeds.size(1)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = [""] * batch_size
            negative_prompt_embeds = self._get_glyph_embeds(negative_prompt, max_sequence_length, device, dtype)
            seq_len = negative_prompt_embeds.size(1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    # ==================== Latent Preparation ====================

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepare random noise latents."""
        if latents is not None:
            return latents.to(device)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Passed {len(generator)} generators but batch size is {batch_size}.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    # ==================== Main Forward Pass ====================

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Main generation forward pass.

        Args:
            req: OmniDiffusionRequest with generation parameters

        Returns:
            DiffusionOutput containing generated image
        """
        prompt = req.prompt or ""
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""

        height = req.height or self.default_sample_size * self.vae_scale_factor
        width = req.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = req.num_inference_steps or 50
        guidance_scale = req.guidance_scale or 1.5

        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0

        # Set seed if provided
        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.seed)

        # 1. Generate prior tokens with AR model
        logger.info("Generating prior tokens with AR model...")
        prior_token_id, prior_token_image_ids, ar_height, ar_width = self.generate_prior_tokens(
            prompt=prompt,
            image=None,  # Text-to-image for now
            height=height,
            width=width,
        )
        height = ar_height
        width = ar_width

        # 2. Encode prompt for glyph embeddings
        logger.info("Encoding prompt...")
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=1,
            device=self.device,
            dtype=self.transformer.dtype,
        )

        # 3. Prepare latents
        latent_channels = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=latent_channels,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=self.device,
            generator=generator,
        )

        # 4. Prepare timesteps
        image_seq_len = ((height // self.vae_scale_factor) * (width // self.vae_scale_factor)) // (self._patch_size**2)
        timesteps_array = np.linspace(self.scheduler.config.num_train_timesteps, 1.0, num_inference_steps + 1)[:-1]
        timesteps_array = timesteps_array.astype(np.int64).astype(np.float32)
        sigmas = timesteps_array / self.scheduler.config.num_train_timesteps

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("base_shift", 0.25),
            self.scheduler.config.get("max_shift", 0.75),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device, timesteps_array.tolist(), sigmas.tolist(), mu=mu
        )

        # 5. Prepare conditioning tensors
        target_size = torch.tensor([[height, width]], dtype=prompt_embeds.dtype, device=self.device)
        crop_coords = torch.zeros((1, 2), dtype=prompt_embeds.dtype, device=self.device)

        prior_token_drop_cond = torch.full_like(prior_token_id, False, dtype=torch.bool)
        prior_token_drop_uncond = torch.full_like(prior_token_id, True, dtype=torch.bool)

        # 6. Denoising loop
        logger.info(f"Starting denoising loop with {num_inference_steps} steps...")
        transformer_dtype = self.transformer.dtype

        for i, t in enumerate(timesteps):
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0]) - 1

            # Conditional forward pass
            noise_pred_cond = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                prior_token_id=prior_token_id,
                prior_token_drop=prior_token_drop_cond,
                timestep=timestep,
                target_size=target_size,
                crop_coords=crop_coords,
                return_dict=False,
            )[0].float()

            # CFG: Unconditional forward pass
            if do_classifier_free_guidance:
                noise_pred_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=negative_prompt_embeds,
                    prior_token_id=prior_token_id,
                    prior_token_drop=prior_token_drop_uncond,
                    timestep=timestep,
                    target_size=target_size,
                    crop_coords=crop_coords,
                    return_dict=False,
                )[0].float()

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 7. VAE decode
        logger.info("Decoding latents with VAE...")
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

        # 8. Post-process
        image = self.image_processor.postprocess(image, output_type="pil")[0]

        return DiffusionOutput(output=image)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load transformer weights."""
        # Filter weights for transformer only
        transformer_weights = (
            (name.replace("transformer.", ""), weight) for name, weight in weights if name.startswith("transformer.")
        )
        return self.transformer.load_weights(transformer_weights)
