#!/usr/bin/env python3
"""Measure policy-action sensitivity to a hosted Cosmos encoder latent.

The full policy runs on local CPU with identical synthetic state/noise. Its
encoder is replaced by pinned source/device latent tensors from one fixture.
This is a numerical handoff probe, not a device-local or robot-task run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


CHECKPOINT_SHA256 = "586799539888c69c35f84d083e559cc5f0ba887801d000e70e65f58d3f85c313"
FIXTURE_SHA256 = "dcbaf1391ca914d4bbccc1cb384daf05666e771d9cdae90c4f069ec8fcf05368"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "cosmos-dir", "processor-dir", "fixture", "device-output", "audit-report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()

    os.environ["INTERNVLA_A1_COSMOS_DIR"] = str(args.cosmos_dir.resolve(strict=True))
    os.environ["INTERNVLA_A1_PROCESSOR_DIR"] = str(args.processor_dir.resolve(strict=True))
    os.environ["HF_HUB_OFFLINE"] = "1"

    import numpy as np
    import torch

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.registry import initialize_model
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    checkpoint = args.model_dir / "model.safetensors"
    if sha256(checkpoint) != CHECKPOINT_SHA256 or sha256(args.fixture) != FIXTURE_SHA256:
        raise ValueError("checkpoint or encoder fixture differs from pinned source")
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    if (audit.get("status") != "component_inference_numeric_measured"
            or audit.get("device", {}).get("name") != "Samsung Galaxy S25"
            or audit.get("files", {}).get("device_output", {}).get("sha256") != sha256(args.device_output)):
        raise ValueError("hosted latent lacks audited S25 provenance")
    with np.load(args.fixture, allow_pickle=False) as fixture:
        source_latent = np.asarray(fixture["pattern_reference"])
    with np.load(args.device_output, allow_pickle=False) as output:
        if output.files != ["output_0__0"]:
            raise ValueError("unexpected hosted output layout")
        device_latent = np.asarray(output["output_0__0"])
    if (source_latent.shape != (6, 16, 32, 32) or device_latent.shape != source_latent.shape
            or source_latent.dtype != np.float32 or device_latent.dtype != np.float32
            or not np.isfinite(source_latent).all() or not np.isfinite(device_latent).all()):
        raise ValueError("latents violate the finite FP32 six-frame contract")

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
    load_seconds = time.perf_counter() - started
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
                request_id="internvla-hosted-latent-synthetic",
            )
        ]))
        if result.error:
            raise RuntimeError(result.error)
        tensor = result.output["payload"]["actions"]
        if tuple(tensor.shape) != (1, 50, 32) or tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
            raise RuntimeError("bad policy action tensor")
        return tensor.detach().float().cpu().contiguous().numpy()

    source_encoder = PinnedLatent(source_latent)
    hosted_encoder = PinnedLatent(device_latent)
    try:
        source_actions = actions(source_encoder)
        hosted_actions = actions(hosted_encoder)
        source_repeat = actions(source_encoder)
    finally:
        cosmos._enc_model = original_encoder
    if not np.array_equal(source_actions, source_repeat):
        raise RuntimeError("CPU source policy changed across the paired calls")
    source = source_actions.astype(np.float64).ravel()
    hosted = hosted_actions.astype(np.float64).ravel()
    delta = source - hosted
    report = {
        "scope": "S25 hosted Cosmos latent injected into local CPU InternVLA policy with synthetic state/noise; no device-local policy or robot-task result",
        "status": "synthetic_action_sensitivity_measured",
        "policy_checkpoint_sha256": CHECKPOINT_SHA256,
        "fixture_sha256": FIXTURE_SHA256,
        "hosted_output_sha256": sha256(args.device_output),
        "hosted_inference_job_id": audit["inference_job_id"],
        "policy_torch": torch.__version__,
        "policy_load_seconds_not_request_timing": load_seconds,
        "encoder_calls": {"source": source_encoder.calls, "hosted": hosted_encoder.calls},
        "action_shape": list(source_actions.shape),
        "source_repeat_bitwise_equal": True,
        "action_relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(source)),
        "action_max_abs": float(np.max(np.abs(delta))),
        "action_cosine": float(np.dot(source, hosted) / (np.linalg.norm(source) * np.linalg.norm(hosted))),
        "source_actions_sha256": hashlib.sha256(source_actions.tobytes()).hexdigest(),
        "hosted_actions_sha256": hashlib.sha256(hosted_actions.tobytes()).hexdigest(),
        "limits": [
            "The policy ran on local CPU; the device only executed the isolated encoder fixture.",
            "The encoder input was a synthetic pattern and policy state/noise were synthetic zeros.",
            "No real-observation reference actions, physical units/time, or robot-task tolerance exist.",
            "No latency, memory admission, cancellation or sustained device power gate was tested.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "action_relative_l2", "action_max_abs", "action_cosine", "source_repeat_bitwise_equal"
    )}, indent=2))


if __name__ == "__main__":
    main()
