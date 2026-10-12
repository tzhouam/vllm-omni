#!/usr/bin/env python3
"""Export one real InternVLA action projection from a synthetic policy pass.

The synthetic camera/state fixture and previously hosted Cosmos latent only
provide numerical input to this *component* export. The Qwen/VLA policy and
flow steps execute locally while capturing the final projection's true hidden
input. This does not establish target-side complete action generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


CHECKPOINT_SHA256 = "586799539888c69c35f84d083e559cc5f0ba887801d000e70e65f58d3f85c313"
COSMOS_FIXTURE_SHA256 = "dcbaf1391ca914d4bbccc1cb384daf05666e771d9cdae90c4f069ec8fcf05368"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--cosmos-dir", required=True, type=Path)
    parser.add_argument("--processor-dir", required=True, type=Path)
    parser.add_argument("--cosmos-fixture", required=True, type=Path)
    parser.add_argument("--hosted-cosmos-output", required=True, type=Path)
    parser.add_argument("--hosted-cosmos-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    for path in (args.model_dir, args.cosmos_dir, args.processor_dir,
                 args.cosmos_fixture, args.hosted_cosmos_output,
                 args.hosted_cosmos_audit):
        path.resolve(strict=True)
    if digest(args.model_dir / "model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("InternVLA Place_Markpen checkpoint SHA-256 changed")
    if digest(args.cosmos_fixture) != COSMOS_FIXTURE_SHA256:
        raise ValueError("synthetic Cosmos fixture changed")
    audit = json.loads(args.hosted_cosmos_audit.read_text(encoding="utf-8"))
    if (audit.get("status") != "component_inference_numeric_and_placement_measured"
            or audit.get("files", {}).get("device_output", {}).get("sha256")
            != digest(args.hosted_cosmos_output)):
        raise ValueError("hosted Cosmos latent lacks pinned numeric/placement audit")

    os.environ["INTERNVLA_A1_COSMOS_DIR"] = str(args.cosmos_dir.resolve())
    os.environ["INTERNVLA_A1_PROCESSOR_DIR"] = str(args.processor_dir.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"

    import numpy as np
    import onnxruntime as ort
    import torch
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.registry import initialize_model
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    with np.load(args.hosted_cosmos_output, allow_pickle=False) as archive:
        if archive.files != ["output_0__0"]:
            raise ValueError("hosted Cosmos output layout changed")
        latent = np.asarray(archive["output_0__0"])
    if latent.shape != (6, 16, 32, 32) or latent.dtype != np.float32 or not np.isfinite(latent).all():
        raise ValueError("hosted synthetic Cosmos latent violates expected contract")

    config = OmniDiffusionConfig(
        model=str(args.model_dir.resolve()), model_class_name="InternVLAA1Pipeline",
        dtype=torch.bfloat16,
        custom_pipeline_args={
            "device": "cpu", "dtype": "bfloat16", "compile_model": False,
            "enable_regional_compile": False, "enable_warmup": False,
            "strict_load": True,
            "processor_model_name": str(args.processor_dir.resolve()),
        },
    )
    pipeline = initialize_model(config)
    if pipeline.runtime_mode() != "real_checkpoint_loaded":
        raise RuntimeError("InternVLA real checkpoint was not loaded")
    policy = pipeline.policy
    inputs = pipeline._build_fake_batch_inputs()
    noise = torch.zeros((1, pipeline.config.chunk_size, pipeline.config.max_action_dim), dtype=torch.float32)
    cosmos = policy.model.cosmos
    original_encoder = cosmos._enc_model
    projection = policy.model.action_out_proj

    class PinnedLatent(torch.nn.Module):
        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            if tuple(pixels.shape) != (6, 3, 256, 256):
                raise RuntimeError("unexpected synthetic camera shape")
            return torch.from_numpy(latent.copy()).to(dtype=pixels.dtype)

    captures: list[tuple[np.ndarray, np.ndarray]] = []

    def capture(_module: torch.nn.Module, tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captures.append((
            tensors[0].detach().float().cpu().contiguous().numpy().copy(),
            output.detach().float().cpu().contiguous().numpy().copy(),
        ))

    hook = projection.register_forward_hook(capture)
    cosmos._enc_model = PinnedLatent()
    try:
        result = pipeline.forward(DiffusionRequestBatch(requests=[
            OmniDiffusionRequest(
                prompt="", request_id="internvla-action-projection-synthetic",
                sampling_params=OmniDiffusionSamplingParams(extra_args={
                    "batch_inputs": inputs, "noise": noise, "decode_image": False,
                }),
            )
        ]))
    finally:
        hook.remove()
        cosmos._enc_model = original_encoder
    if result.error or not captures:
        raise RuntimeError(f"local synthetic policy failed or skipped projection: {result.error}")
    actions = result.output["payload"]["actions"].detach().float().cpu().numpy()
    if actions.shape != (1, 50, 32) or not np.isfinite(actions).all():
        raise ValueError("local synthetic policy action tensor is invalid")
    hidden, expected = captures[-1]
    if hidden.shape != (1, 50, projection.in_features) or expected.shape != (1, 50, projection.out_features):
        raise ValueError("captured action projection shape changed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "internvla_action_out_proj.onnx"
    fixture_path = args.output_dir / "projection_fixture.npz"
    projection.eval().cpu()
    with torch.inference_mode():
        torch.onnx.export(
            projection, (torch.from_numpy(hidden),), str(onnx_path),
            input_names=["suffix_hidden"], output_names=["action_velocity"],
            opset_version=17, dynamo=False,
        )
    np.savez_compressed(fixture_path, suffix_hidden=hidden, expected_action_velocity=expected)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    exported = session.run(None, {"suffix_hidden": hidden})[0]
    delta = exported.astype(np.float64) - expected.astype(np.float64)
    relative_l2 = float(np.linalg.norm(delta) / np.linalg.norm(expected.astype(np.float64)))
    if not np.isfinite(exported).all() or relative_l2 > 1e-5:
        raise ValueError(f"exported action projection differs from checkpoint reference: {relative_l2}")
    report = {
        "status": "real_checkpoint_action_projection_exported_component_only",
        "scope": "one real-weight final policy action projection captured during synthetic local policy; not target-side complete action generation",
        "target_name_of_input_latent": audit["device"]["name"],
        "checkpoint_path": str(args.model_dir.resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_revision": "local-Place_Markpen-sha256:" + CHECKPOINT_SHA256,
        "synthetic_cosmos_fixture_sha256": COSMOS_FIXTURE_SHA256,
        "hosted_cosmos_output_sha256": digest(args.hosted_cosmos_output),
        "hosted_cosmos_job_id": audit["inference_job_id"],
        "local_projection_calls": len(captures),
        "local_final_actions_shape": list(actions.shape),
        "source_onnx": {"path": onnx_path.name, "sha256": digest(onnx_path), "bytes": onnx_path.stat().st_size},
        "fixture": {"path": fixture_path.name, "sha256": digest(fixture_path), "bytes": fixture_path.stat().st_size},
        "projection_input_shape": list(hidden.shape),
        "projection_output_shape": list(expected.shape),
        "onnx_cpu_vs_torch_relative_l2": relative_l2,
        "onnx_cpu_vs_torch_max_abs": float(np.max(np.abs(delta))),
        "missing_for_e2e": [
            "matching real Place_Markpen observation/action references are gated",
            "Qwen3-VL prefix and full action expert were computed on local CPU for fixture capture",
            "no target-side multi-step flow action chain or task-quality validation",
        ],
    }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "onnx_sha256": report["source_onnx"]["sha256"],
                      "relative_l2": relative_l2, "projection_calls": len(captures)}, indent=2))


if __name__ == "__main__":
    main()
