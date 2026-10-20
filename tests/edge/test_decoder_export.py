"""WP4 decoder export: windowed wrapper (CPU, fake decoder) and ExecuTorch lowering (skipped without executorch)."""

from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from vllm_omni.edge import decoder_export as dx

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HOP = 4


class _FakeConfig:
    num_quantizers = 3
    codebook_size = 8
    output_sample_rate = 24000


class _FakeDecoder(nn.Module):
    """Stands in for Qwen3TTSTokenizerV2Decoder: ``_forward_exact`` maps each frame to HOP samples."""

    total_upsample = HOP
    config = _FakeConfig()

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def _forward_exact(self, codes: torch.Tensor) -> torch.Tensor:
        # [1, Q, T] -> [1, 1, T*HOP]; value depends on the frame's codes so slicing is testable
        per_frame = codes.to(torch.float32).sum(dim=1, keepdim=True) * self.scale  # [1, 1, T]
        return per_frame.repeat_interleave(HOP, dim=-1).clamp(min=-1000, max=1000)


def test_windowed_wrapper_returns_only_the_new_chunk():
    module = dx.WindowedCode2Wav(_FakeDecoder(), chunk_frames=2, context_frames=3)
    assert module.window_frames == 5 and module.hop == HOP and module.num_quantizers == 3
    codes = torch.arange(15, dtype=torch.long).reshape(1, 3, 5)
    out = module(codes)
    assert out.shape == (1, 2 * HOP)
    full = _FakeDecoder()._forward_exact(codes).reshape(1, -1)
    assert torch.equal(out, full[:, -2 * HOP :])


def test_torch_export_of_windowed_wrapper_matches_eager():
    module = dx.WindowedCode2Wav(_FakeDecoder(), chunk_frames=2, context_frames=3)
    ep = dx.export_program(module)
    codes = torch.randint(0, 8, module.example_input().shape, dtype=torch.long)
    with torch.inference_mode():
        assert torch.allclose(ep.module()(codes), module(codes))


@pytest.mark.skipif(importlib.util.find_spec("executorch") is None, reason="executorch not installed")
def test_executorch_lowering_of_windowed_wrapper():
    module = dx.WindowedCode2Wav(_FakeDecoder(), chunk_frames=2, context_frames=3)
    buf = dx.lower_executorch(dx.export_program(module), partitioner="xnnpack")
    assert isinstance(buf, (bytes, bytearray)) and len(buf) > 0
