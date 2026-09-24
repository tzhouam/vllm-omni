#!/usr/bin/env python3
"""Measure synthetic policy-action sensitivity to the local AMD NPU Cosmos stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


CHECKPOINT_SHA = "586799539888c69c35f84d083e559cc5f0ba887801d000e70e65f58d3f85c313"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "cosmos-dir", "processor-dir", "latents", "downstream-report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    os.environ["INTERNVLA_A1_COSMOS_DIR"] = str(args.cosmos_dir.resolve(strict=True))
    os.environ["INTERNVLA_A1_PROCESSOR_DIR"] = str(args.processor_dir.resolve(strict=True))
    os.environ["HF_HUB_OFFLINE"] = "1"

    import numpy as np
    import torch
    import transformers
    import vllm
    import vllm_omni

    repo_root = Path(__file__).resolve().parents[3]
    if not Path(vllm_omni.__file__).resolve().is_relative_to(repo_root / "vllm_omni"):
        raise RuntimeError("run with PYTHONPATH set to this fork checkout")

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.registry import initialize_model
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    checkpoint = args.model_dir / "model.safetensors"
    if sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("real Place_Markpen policy checkpoint changed")
    downstream = json.loads(args.downstream_report.read_text(encoding="utf-8"))
    if (downstream.get("status") != "encoder_suffix_numeric_reported"
            or downstream.get("latent_output_sha256") != sha256(args.latents)):
        raise ValueError("NPU-injected Cosmos latent lacks pinned downstream evidence")
    with np.load(args.latents, allow_pickle=False) as file:
        source_latent = np.asarray(file["pattern_reference"])
        npu_latent = np.asarray(file["pattern_npu_injected"])
    if (source_latent.shape != npu_latent.shape or source_latent.shape != (6, 16, 32, 32)
            or source_latent.dtype != npu_latent.dtype or source_latent.dtype != np.float32
            or not np.isfinite(source_latent).all() or not np.isfinite(npu_latent).all()):
        raise ValueError("latent contract changed")

    config = OmniDiffusionConfig(
        model=str(args.model_dir.resolve(strict=True)),
        model_class_name="InternVLAA1Pipeline",
        dtype=torch.bfloat16,
        custom_pipeline_args={
            "device": "cpu", "dtype": "bfloat16", "compile_model": False,
            "enable_regional_compile": False, "enable_warmup": False,
            "strict_load": True,
            "processor_model_name": str(args.processor_dir.resolve(strict=True)),
        },
    )
    started = time.perf_counter()
    pipeline = initialize_model(config)
    load_s = time.perf_counter() - started
    if pipeline.runtime_mode() != "real_checkpoint_loaded":
        raise RuntimeError("real policy checkpoint was not loaded")
    if {parameter.device.type for parameter in pipeline.policy.parameters()} != {"cpu"}:
        raise RuntimeError("policy parameters did not stay on CPU")
    inputs = pipeline._build_fake_batch_inputs()
    noise = torch.zeros((1, pipeline.config.chunk_size, pipeline.config.max_action_dim), dtype=torch.float32)
    cosmos = pipeline.policy.model.cosmos
    original_encoder = cosmos._enc_model

    class PinnedLatent(torch.nn.Module):
        def __init__(self, latent: np.ndarray) -> None:
            super().__init__()
            self.latent = torch.from_numpy(latent.copy())
            self.calls = 0

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            if tuple(pixels.shape) != (6, 3, 256, 256) or pixels.device.type != "cpu":
                raise RuntimeError("unexpected policy encoder input")
            self.calls += 1
            return self.latent.to(dtype=pixels.dtype)

    def actions(encoder: PinnedLatent) -> np.ndarray:
        cosmos._enc_model = encoder
        result = pipeline.forward(DiffusionRequestBatch(requests=[
            OmniDiffusionRequest(
                prompt="",
                sampling_params=OmniDiffusionSamplingParams(extra_args={
                    "batch_inputs": inputs, "noise": noise, "decode_image": False,
                }),
                request_id="internvla-hx370-npu-latent-synthetic",
            )
        ]))
        if result.error:
            raise RuntimeError(result.error)
        tensor = result.output["payload"]["actions"]
        if tuple(tensor.shape) != (1, 50, 32) or tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
            raise RuntimeError("policy action tensor contract failed")
        return tensor.detach().float().cpu().contiguous().numpy()

    source_encoder = PinnedLatent(source_latent)
    npu_encoder = PinnedLatent(npu_latent)
    try:
        source_actions = actions(source_encoder)
        npu_actions = actions(npu_encoder)
        source_repeat = actions(source_encoder)
    finally:
        cosmos._enc_model = original_encoder
    if not np.array_equal(source_actions, source_repeat):
        raise RuntimeError("CPU source policy changed across paired calls")
    source = source_actions.astype(np.float64).ravel()
    tested = npu_actions.astype(np.float64).ravel()
    difference = tested - source
    report = {
        "scope": "HX370 AMD NPU Conv13 boundary latent injected into real local CPU InternVLA policy; synthetic input and zero noise only",
        "status": "synthetic_action_sensitivity_measured",
        "policy_checkpoint_sha256": CHECKPOINT_SHA,
        "latent_output_sha256": downstream["latent_output_sha256"],
        "policy_torch": torch.__version__, "policy_vllm": vllm.__version__,
        "policy_transformers": transformers.__version__,
        "loaded_omni_path": str(Path(vllm_omni.__file__).resolve()),
        "policy_load_seconds_not_request_timing": load_s,
        "encoder_calls": {"source": source_encoder.calls, "npu": npu_encoder.calls},
        "action_shape": list(source_actions.shape),
        "source_repeat_bitwise_equal": True,
        "action_relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(source)),
        "action_max_abs": float(np.max(np.abs(difference))),
        "action_cosine": float(np.dot(source, tested) / (np.linalg.norm(source) * np.linalg.norm(tested))),
        "source_actions_sha256": hashlib.sha256(source_actions.tobytes()).hexdigest(),
        "npu_actions_sha256": hashlib.sha256(npu_actions.tobytes()).hexdigest(),
        "limits": [
            "The policy and encoder suffix ran on CPU; only the isolated Conv13 boundary ran on AMD NPU.",
            "The NPU boundary came from a retained synthetic pattern, not the live policy input.",
            "No real-observation reference actions or physical units/time/task tolerance exist.",
            "No complete-request latency, admission, cancellation or sustained power gate was tested.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "action_relative_l2", "action_max_abs", "action_cosine", "source_repeat_bitwise_equal"
    )}, indent=2))


if __name__ == "__main__":
    main()
