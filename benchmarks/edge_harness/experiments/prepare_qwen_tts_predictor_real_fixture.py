#!/usr/bin/env python3
"""Export a revision-pinned predictor body with an input from real TTS generation.

The result is deliberately a predictor *component* fixture.  The captured
talker hidden state and speech codes were produced locally, so a hosted
predictor job using this fixture is not a hosted complete TTS request.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel


REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class PredictorBody(torch.nn.Module):
    def __init__(self, predictor: torch.nn.Module):
        super().__init__()
        self.predictor = predictor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(inputs_embeds=x, use_cache=False).last_hidden_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text", default="Hello from the edge device.")
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    model_path = args.model.resolve()
    if model_path.name != REVISION:
        parser.error(f"expected exact snapshot {REVISION}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(args.threads)
    wrapper = Qwen3TTSModel.from_pretrained(
        str(model_path), device_map="cpu", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    predictor = wrapper.model.talker.code_predictor
    original_generate = predictor.generate
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(*a: object, **kw: object):
        result = original_generate(*a, **kw)
        if not captured:
            initial = kw.get("inputs_embeds")
            if not isinstance(initial, torch.Tensor):
                raise RuntimeError("predictor generation did not supply input embeddings")
            captured.append((initial.detach().cpu().clone(),
                             result.sequences.detach().cpu().clone()))
        return result

    predictor.generate = capture
    torch.manual_seed(42)
    wavs, sample_rate = wrapper.generate_custom_voice(
        text=args.text, language="English", speaker=args.speaker,
        max_new_tokens=args.max_new_tokens, do_sample=False,
        subtalker_dosample=True,
        non_streaming_mode=True,
    )
    predictor.generate = original_generate
    if len(captured) != 1 or len(wavs) != 1 or sample_rate != 24000:
        raise RuntimeError("expected one complete reference request and predictor call")
    initial, generated = captured[0]
    if initial.shape != (1, 2, 1024) or generated.shape != (1, 15):
        raise RuntimeError(f"unexpected predictor tensors: {initial.shape}, {generated.shape}")
    group_embeds = [predictor.get_input_embeddings()[i](generated[:, i:i + 1])
                    .detach().cpu() for i in range(15)]
    x = torch.cat([initial, *group_embeds], dim=1).float().contiguous()
    if x.shape != (1, 17, 1024):
        raise RuntimeError("predictor body input has unexpected shape")
    body = PredictorBody(predictor.model.float().eval()).eval()
    with torch.inference_mode():
        reference = body(x).detach().cpu().numpy()
    if reference.shape != (1, 17, 1024) or not np.isfinite(reference).all():
        raise RuntimeError("predictor reference output failed shape/finiteness")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = args.output_dir / "real_predictor_input.npz"
    reference_path = args.output_dir / "reference_output.npz"
    export_path = args.output_dir / "predictor_body.onnx"
    np.savez_compressed(fixture_path, x=x.numpy())
    np.savez_compressed(reference_path, hidden=reference)
    export = torch.onnx.export(
        body, (x,), input_names=["x"], output_names=["hidden"],
        opset_version=17, dynamo=True, optimize=True,
    )
    export.save(str(export_path))
    import onnxruntime as ort

    session = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    observed = session.run(["hidden"], {"x": x.numpy()})[0]
    relative_l2 = float(np.linalg.norm((observed - reference).astype(np.float64)) /
                        max(np.linalg.norm(reference.astype(np.float64)), 1e-12))
    report = {
        "scope": "real-input predictor body component only; talker and vocoder ran on local CPU",
        "checkpoint": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "revision": REVISION,
        "text": args.text,
        "speaker": args.speaker,
        "sample_rate": sample_rate,
        "talker_do_sample": False,
        "subtalker_dosample": True,
        "reference_audio_samples": len(wavs[0]),
        "code_groups": generated.numpy().tolist(),
        "source_sha256": sha256(model_path / "model.safetensors"),
        "qwen_tts_source_sha256": sha256(Path(inspect.getfile(Qwen3TTSModel))),
        "artifacts": {p.name: sha256(p) for p in
                      (fixture_path, reference_path, export_path)},
        "onnx_vs_eager_relative_l2": relative_l2,
        "batch_size": 1,
        "concurrency": 1,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if relative_l2 > 1e-4:
        raise RuntimeError(f"export failed numerical gate: {relative_l2}")
    print(json.dumps({"status": "exported", "relative_l2": relative_l2,
                      "onnx_sha256": report["artifacts"][export_path.name]}))


if __name__ == "__main__":
    main()
