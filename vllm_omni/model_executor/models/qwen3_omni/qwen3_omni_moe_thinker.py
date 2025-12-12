# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3-Omni-Moe model (thinker part)."""

from collections.abc import Sequence

import torch
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel as _Qwen3MoeLLMModel
from vllm.sequence import IntermediateTensors

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None


logger = init_logger(__name__)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen3MoeLLMModel(_Qwen3MoeLLMModel):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        *,
        capture_layer_indices: Sequence[int] | None = None,
        return_hidden_states: bool = False,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        capture_set = set(capture_layer_indices) if capture_layer_indices else None
        captured_hidden_states: dict[str, torch.Tensor] | None = {} if return_hidden_states else None

        for layer_idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            layer_idx = layer_idx + self.start_layer

            if captured_hidden_states is not None and capture_set is not None:
                if layer_idx in capture_set:
                    captured_hidden_states[str(layer_idx)] = hidden_states.clone().view(-1, hidden_states.shape[-1])

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(0, len(deepstack_input_embeds)):
                hidden_states = hidden_states + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        if captured_hidden_states is not None:
            return hidden_states, captured_hidden_states
        else:
            return hidden_states, None
