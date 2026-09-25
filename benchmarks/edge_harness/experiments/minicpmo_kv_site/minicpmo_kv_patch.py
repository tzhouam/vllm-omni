"""Replace only MiniCPM-o 4.5's loaded KV projection with an Omni stage.

The hook is activated by sitecustomize for a bounded benchmark; it is never
installed during ordinary inference. It requires the exact fixed 32x32-patch
bucket and fails the request if artifact, memory, placement or tensor checks
fail. The model keeps its vision transformer and resampler suffix on CPU.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import platform
import time
from pathlib import Path

GRAPH_SHA = "330bbbd0d18caaf6903aa41f836dbeaef57643a8afbaba333df68e3c1e722aeb"
SHARD_SHA = "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31"
REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
INPUTS_SHA = "ab71cc3a6b8461c99cdf9e458b2e8b99cc092dd6adc3d49f92d643ec4b8693de"


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
    import numpy as np
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
        def __init__(self):
            super().__init__()
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
            atexit.register(self.close)
            _event(
                "adapter_installed", graph_sha256=GRAPH_SHA, model_revision=REVISION,
                planner_budget_bytes=self.plan.budget_bytes,
                platform=platform.platform(),
            )

        def forward(self, x):
            if (x.shape != (1, 1024, 1152) or x.dtype != torch.bfloat16
                    or x.device.type != "cpu"):
                _event("tensor_refused", shape=list(x.shape), dtype=str(x.dtype), device=str(x.device))
                raise ValueError("MiniCPM-o NPU KV stage requires CPU BF16 [1,1024,1152]")
            hidden = np.ascontiguousarray(x.detach().float().numpy())
            feed = {"hidden": hidden}
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
            projected = np.ascontiguousarray(output["projected"])
            if (projected.shape != (1, 1024, 4096)
                    or projected.dtype != np.float32 or not np.isfinite(projected).all()):
                raise ValueError("MiniCPM-o NPU KV output tensor contract failed")
            self.calls += 1
            _event("run", call=self.calls, timing=timing.to_dict(),
                   input_shape=list(hidden.shape), output_shape=list(projected.shape))
            return torch.from_numpy(projected).to(dtype=x.dtype, device=x.device)

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
        resampler.kv_proj = ExternalKVProjection()
        return loaded

    patched_load_weights._omni_kv_experiment = True
    thinker_class.load_weights = patched_load_weights
