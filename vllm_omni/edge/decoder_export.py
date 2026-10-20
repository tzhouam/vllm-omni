# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Export the Qwen3-TTS Code2Wav decoder for on-device (NPU / phone) runtimes (WP4).

vLLM-Omni cannot run on phone NPUs; the ``npu_phone`` hardware class is served
by the codec-token stream (WP5) plus an exported decoder artifact. This module
exports a *windowed, stateless* variant of ``Qwen3TTSTokenizerV2Decoder``:

* input ``codes``: ``int64[1, num_quantizers, context_frames + chunk_frames]``
  (the last ``context_frames`` frames already decoded plus the new chunk);
* output ``wav``: ``float32[1, chunk_frames * hop]`` (only the new chunk's audio,
  ``hop = total_upsample`` samples per frame, 1920 at 24 kHz / 12.5 Hz).

Fixed shapes (one program per chunk size of the ramp schedule) keep the export
static, which is what ExecuTorch / QNN / CoreML partitioners need; the
72-frame window is explicit input state instead of the decoder's Python
caches, so the artifact carries no mutable state. Backends:

* ``torch_export`` - ``torch.export`` + ``.pt2`` (always available);
* ``executorch`` - XNNPACK-lowered ``.pte`` (needs ``pip install executorch``;
  the QNN partitioner is export-only and listed as unverified).

Status: the ExecuTorch path is written against the executorch 1.x API but was
not executed in this checkout (package not installed; install is permission
gated). ``tests/edge/test_decoder_export.py`` skips the lowering without it and
still checks the windowed wrapper against the eager decoder.

CLI:
  python -m vllm_omni.edge.decoder_export --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
      --ramp 2 4 8 16 25 --context 72 --backend torch_export --out-dir ./code2wav_export
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_RAMP = (2, 4, 8, 16, 25)
DEFAULT_CONTEXT = 72


class WindowedCode2Wav(nn.Module):
    """Stateless windowed decode: full window in, trailing ``chunk_frames`` of audio out."""

    def __init__(self, decoder: nn.Module, chunk_frames: int, context_frames: int = DEFAULT_CONTEXT):
        super().__init__()
        self.decoder = decoder
        self.chunk_frames = int(chunk_frames)
        self.context_frames = int(context_frames)
        self.hop = int(getattr(decoder, "total_upsample", 1920))
        self.num_quantizers = int(getattr(getattr(decoder, "config", None), "num_quantizers", 16))

    @property
    def window_frames(self) -> int:
        return self.context_frames + self.chunk_frames

    def example_input(self) -> torch.Tensor:
        return torch.zeros(1, self.num_quantizers, self.window_frames, dtype=torch.long)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        wav = self.decoder._forward_exact(codes)  # [1, 1, samples] or [1, samples]
        wav = wav.reshape(wav.shape[0], -1)
        return wav[:, -self.chunk_frames * self.hop :]


def resolve_model_dir(model: str) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model))


def load_code2wav_decoder(model: str, dtype: torch.dtype = torch.float32) -> nn.Module:
    """Load ``Qwen3TTSTokenizerV2Decoder`` from ``<model>/speech_tokenizer`` (eager, CPU)."""
    from safetensors.torch import load_file

    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Config,
    )
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Decoder,
    )

    model_dir = resolve_model_dir(model)
    cfg = Qwen3TTSTokenizerV2Config.from_pretrained(str(model_dir), subfolder="speech_tokenizer")
    decoder = Qwen3TTSTokenizerV2Decoder._from_config(cfg.decoder_config)
    state = load_file(str(model_dir / "speech_tokenizer" / "model.safetensors"))
    dec_state = {k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")}
    _, unexpected = decoder.load_state_dict(dec_state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected decoder keys: {unexpected[:5]}")
    decoder.eval().to(dtype=dtype)
    if hasattr(decoder, "precompute_snake_caches"):
        decoder.precompute_snake_caches()
    return decoder


def export_program(module: WindowedCode2Wav) -> Any:
    """``torch.export`` the windowed module with its fixed example input."""
    example = module.example_input()
    with torch.inference_mode():
        return torch.export.export(module.eval(), (example,))


def lower_executorch(exported_program: Any, partitioner: str = "xnnpack") -> bytes:
    """Lower an exported program to an ExecuTorch ``.pte`` buffer (requires executorch)."""
    from executorch.exir import to_edge_transform_and_lower  # type: ignore[import-not-found]

    partitioners = []
    if partitioner == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (  # type: ignore[import-not-found]
            XnnpackPartitioner,
        )

        partitioners = [XnnpackPartitioner()]
    elif partitioner == "qnn":  # unverified: needs the Qualcomm SDK build of executorch
        from executorch.backends.qualcomm.partition.qnn_partitioner import (  # type: ignore[import-not-found]
            QnnPartitioner,
        )

        partitioners = [QnnPartitioner()]
    elif partitioner not in ("none", ""):
        raise ValueError(f"unknown partitioner {partitioner!r}")
    edge = to_edge_transform_and_lower(exported_program, partitioner=partitioners or None)
    return edge.to_executorch().buffer


def export_windowed_decoder(
    model: str,
    out_dir: str | Path,
    *,
    ramp: tuple[int, ...] = DEFAULT_RAMP,
    context_frames: int = DEFAULT_CONTEXT,
    backend: str = "torch_export",
    partitioner: str = "xnnpack",
    check: bool = True,
) -> dict[str, Any]:
    """Export one program per chunk size; returns a manifest (also written as ``manifest.json``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    decoder = load_code2wav_decoder(model)
    manifest: dict[str, Any] = {
        "model": model,
        "backend": backend,
        "partitioner": partitioner if backend == "executorch" else None,
        "context_frames": context_frames,
        "num_quantizers": int(decoder.config.num_quantizers),
        "hop": int(decoder.total_upsample),
        "sample_rate": int(getattr(decoder.config, "output_sample_rate", 24000) or 24000),
        "programs": {},
    }
    for chunk in sorted(set(int(c) for c in ramp)):
        module = WindowedCode2Wav(decoder, chunk, context_frames)
        t0 = time.perf_counter()
        ep = export_program(module)
        entry: dict[str, Any] = {"chunk_frames": chunk, "window_frames": module.window_frames}
        if check:
            codes = torch.randint(0, int(decoder.config.codebook_size), module.example_input().shape, dtype=torch.long)
            with torch.inference_mode():
                ref = module(codes)
                got = ep.module()(codes)
            entry["max_abs_diff_vs_eager"] = float((ref - got).abs().max())
            entry["out_samples"] = int(got.shape[-1])
        if backend == "torch_export":
            path = out / f"code2wav_c{chunk}_ctx{context_frames}.pt2"
            torch.export.save(ep, str(path))
        elif backend == "executorch":
            path = out / f"code2wav_c{chunk}_ctx{context_frames}.pte"
            path.write_bytes(lower_executorch(ep, partitioner))
        else:
            raise ValueError(f"unknown backend {backend!r}")
        entry["path"] = str(path)
        entry["export_s"] = time.perf_counter() - t0
        entry["bytes"] = path.stat().st_size
        manifest["programs"][str(chunk)] = entry
        print(
            f"[decoder_export] chunk={chunk} -> {path} ({entry['bytes'] / 1e6:.1f} MB, {entry['export_s']:.1f} s)",
            flush=True,
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ramp", type=int, nargs="*", default=list(DEFAULT_RAMP))
    p.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    p.add_argument("--backend", choices=["torch_export", "executorch"], default="torch_export")
    p.add_argument("--partitioner", choices=["xnnpack", "qnn", "none"], default="xnnpack")
    p.add_argument("--no-check", action="store_true")
    args = p.parse_args(argv)
    export_windowed_decoder(
        args.model,
        args.out_dir,
        ramp=tuple(args.ramp),
        context_frames=args.context,
        backend=args.backend,
        partitioner=args.partitioner,
        check=not args.no_check,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
