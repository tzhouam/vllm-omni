"""Replace only MiniCPM-o 4.5's loaded KV projection with an Omni stage.

The hook is activated by sitecustomize for a bounded benchmark; it is never
installed during ordinary inference. The pinned graph projects independent
tokens, so larger image/crop tensors are tiled into its exact 1024-token
bucket. Artifact, memory, placement and tensor failures still fail the
request. The vision transformer and resampler suffix retain their original device.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

GRAPH_SHA = "330bbbd0d18caaf6903aa41f836dbeaef57643a8afbaba333df68e3c1e722aeb"
SHARD_SHA = "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31"
REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
INPUTS_SHA = "ab71cc3a6b8461c99cdf9e458b2e8b99cc092dd6adc3d49f92d643ec4b8693de"
TILE_TOKENS = 1024
MAX_LOGICAL_TOKENS = 4096
MAX_CUDA_LOGICAL_TOKENS = 16384


def run_tiled_projection(hidden, run_tile, *, output_width: int = 4096,
                         max_logical_tokens: int = MAX_LOGICAL_TOKENS):
    """Pack independent token rows into fixed graph calls, preserving order."""
    if (hidden.ndim != 3 or hidden.dtype != np.float32 or hidden.shape[-1] != 1152
            or hidden.shape[0] < 1 or hidden.shape[1] < 1):
        raise ValueError("MiniCPM-o KV tiles require nonempty float32 [crops,tokens,1152]")
    logical_tokens = hidden.shape[0] * hidden.shape[1]
    if logical_tokens > max_logical_tokens:
        raise ValueError(f"MiniCPM-o KV tiles exceed {max_logical_tokens} logical tokens")
    flat = np.ascontiguousarray(hidden.reshape(logical_tokens, 1152))
    projected = np.empty((logical_tokens, output_width), dtype=np.float32)
    for tile_index, start in enumerate(range(0, logical_tokens, TILE_TOKENS)):
        valid = min(TILE_TOKENS, logical_tokens - start)
        feed = np.zeros((1, TILE_TOKENS, 1152), dtype=np.float32)
        feed[0, :valid] = flat[start:start + valid]
        output = run_tile(feed, tile_index, valid)
        if (output.shape != (1, TILE_TOKENS, output_width)
                or output.dtype != np.float32 or not np.isfinite(output).all()):
            raise ValueError("MiniCPM-o NPU KV output tensor contract failed")
        projected[start:start + valid] = output[0, :valid]
    return projected.reshape(*hidden.shape[:2], output_width)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _event(phase: str, **fields) -> None:
    path = Path(os.environ["VLLM_OMNI_MINICPMO_KV_EVENT_LOG"])
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"phase": phase, "pid": os.getpid(), "unix_s": time.time(), **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def _check_artifacts() -> Path:
    graph = Path(os.environ["VLLM_OMNI_MINICPMO_KV_GRAPH"])
    model_dir = Path(os.environ["VLLM_OMNI_MINICPMO_KV_MODEL_DIR"])
    shard = model_dir / "model-00004-of-00004.safetensors"
    metadata = model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if (_sha256(graph) != GRAPH_SHA or _sha256(shard) != SHARD_SHA
            or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION):
        raise ValueError("MiniCPM-o NPU KV graph or checkpoint identity changed")
    return graph


def install() -> None:
    import torch
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
        MiniCPMO45OmniLLMForConditionalGeneration,
    )

    from vllm_omni.edge.hardware_probe import load_profile
    from vllm_omni.edge.local.capabilities import FORMAT_ONNX_A16W8, enumerate_devices
    from vllm_omni.edge.local.external.stage import ExternalStage, PlacementRefused, plan_external_stage
    from vllm_omni.edge.local.manifest import build_graph_artifact

    thinker_class = MiniCPMO45OmniLLMForConditionalGeneration
    if getattr(thinker_class.load_weights, "_omni_kv_experiment", False):
        return
    original_load_weights = thinker_class.load_weights

    class ExternalKVProjection(torch.nn.Module):
        def __init__(self, cpu_projection):
            super().__init__()
            # The original module has already loaded its checkpoint weight.
            # Keep it for opt-in parity without adding a new named parameter
            # after vLLM's weight-initialization accounting has begun.
            self.__dict__["_cpu_projection"] = cpu_projection
            self.graph = _check_artifacts()
            artifact = build_graph_artifact(
                self.graph, fmt=FORMAT_ONNX_A16W8, opset=21,
                source_model="openbmb/MiniCPM-o-4_5", source_revision=REVISION,
                component="resampler_kv_projection_32x32",
                exporter="probe_minicpmo_resampler_kv_npu.py quantize",
                calibration={"inputs_sha256": INPUTS_SHA, "cases": ["red", "blue"]},
                parity={"three_image_final_embedding_relative_l2_max": 0.00925311117731196},
            )
            self.plan = plan_external_stage(
                artifact, enumerate_devices(load_profile(use_torch=False)),
                require="npu:amd", min_fraction_on_target=1 / 3,
                worker_peak_rss_hint_bytes=536870912,
            )
            if not self.plan.admitted:
                _event("admission_refused", summary=self.plan.summary())
                raise RuntimeError(self.plan.summary())
            self.stage = ExternalStage(self.plan)
            self.opened = False
            self.calls = 0
            self.requests = 0
            atexit.register(self.close)
            _event(
                "adapter_installed", graph_sha256=GRAPH_SHA, model_revision=REVISION,
                planner_budget_bytes=self.plan.budget_bytes,
                platform=platform.platform(),
            )

        def forward(self, x):
            limit = MAX_CUDA_LOGICAL_TOKENS if x.device.type == "cuda" else MAX_LOGICAL_TOKENS
            if (x.ndim != 3 or x.shape[-1] != 1152 or x.shape[0] < 1
                    or x.shape[1] < 1 or x.shape[0] * x.shape[1] > limit
                    or x.dtype != torch.bfloat16 or x.device.type not in {"cpu", "cuda"}):
                _event("tensor_refused", shape=list(x.shape), dtype=str(x.dtype), device=str(x.device))
                raise ValueError("MiniCPM-o NPU KV stage requires bounded CPU/CUDA BF16 [crops,tokens,1152]")
            input_transfer_started = time.perf_counter()
            hidden = np.ascontiguousarray(x.detach().float().cpu().numpy())
            input_transfer_s = time.perf_counter() - input_transfer_started
            self.requests += 1
            request = self.requests

            def run_tile(tile, tile_index, valid):
                feed = {"hidden": tile}
                if not self.opened:
                    profile_dir = Path(os.environ["VLLM_OMNI_MINICPMO_KV_PROFILE_DIR"])
                    profile_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        placement = self.stage.open(feed, profile_dir=profile_dir)
                    except PlacementRefused as exc:
                        _event("placement_refused", refusal=exc.refusal.to_dict(),
                               report=exc.report.to_dict() if exc.report else None)
                        raise
                    if placement.target_nodes < 1:
                        raise RuntimeError("MiniCPM-o KV stage did not execute on NPU")
                    self.opened = True
                    _event("placement", placement=placement.to_dict())
                output, timing = self.stage.run(feed)
                self.calls += 1
                _event("run", call=self.calls, request=request, tile=tile_index,
                       valid_tokens=valid, timing=timing.to_dict(),
                       input_shape=list(tile.shape), output_shape=list(output["projected"].shape))
                return np.ascontiguousarray(output["projected"])

            projected = run_tiled_projection(hidden, run_tile, max_logical_tokens=limit)
            output_transfer_started = time.perf_counter()
            result = torch.from_numpy(projected).to(dtype=x.dtype, device=x.device)
            if x.device.type == "cuda":
                torch.cuda.synchronize(x.device)
            output_transfer_s = time.perf_counter() - output_transfer_started
            relative_l2 = None
            if os.environ.get("VLLM_OMNI_MINICPMO_KV_PARITY") == "1":
                with torch.inference_mode():
                    reference = self._cpu_projection(x).float()
                    difference = (result.float() - reference).norm()
                    relative_l2 = float(difference / reference.norm().clamp_min(1e-12))
                limit_text = os.environ.get("VLLM_OMNI_MINICPMO_KV_MAX_PROJECTION_REL_L2")
                if limit_text is not None:
                    limit = float(limit_text)
                    if not 0 < limit < 1 or relative_l2 > limit:
                        _event("numeric_refused", request=request, projection_relative_l2=relative_l2,
                               limit=limit)
                        raise ValueError("MiniCPM-o NPU KV projection failed configured numerical gate")
            _event("request_complete", request=request, input_shape=list(hidden.shape),
                   output_shape=list(projected.shape),
                   source_device=str(x.device), input_transfer_s=input_transfer_s,
                   output_transfer_s=output_transfer_s,
                   tiles=(hidden.shape[0] * hidden.shape[1] + TILE_TOKENS - 1) // TILE_TOKENS,
                   projection_relative_l2=relative_l2)
            return result

        def close(self):
            if self.opened:
                try:
                    _event("close", calls=self.calls, worker_stats=self.stage.stats())
                finally:
                    self.stage.__exit__(None, None, None)
                    self.opened = False

    def patched_load_weights(self, weights):
        loaded = original_load_weights(self, weights)
        resampler = getattr(self, "resampler", None)
        if resampler is None:
            raise RuntimeError("MiniCPM-o 4.5 thinker has no resampler for NPU KV stage")
        if isinstance(resampler.kv_proj, ExternalKVProjection):
            return loaded
        # The original weights are loaded before replacement. The exact source
        # shard and graph are pinned, so there is no hidden model substitution.
        resampler.kv_proj = ExternalKVProjection(resampler.kv_proj)
        return loaded

    patched_load_weights._omni_kv_experiment = True
    thinker_class.load_weights = patched_load_weights
