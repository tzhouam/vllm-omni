# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Iterable
from math import prod
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward

from vllm_omni.diffusion.cache.base import CachedTransformer

class GlmImageTransformer2DModel(CachedTransformer):
