# SPDX-License-Identifier: Apache-2.0
"""Opt-in Spark output-head split for a measured same-machine experiment.

The model still uses vLLM for its decoder and KV. Only the pre-final-norm
single-token activation crosses to an Omni ExternalStage. A failure propagates;
there is no hidden CPU-head fallback or precision/model substitution.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.edge.hardware_probe import load_profile
from vllm_omni.edge.local.capabilities import FORMAT_ONNX_A16W8, enumerate_devices
from vllm_omni.edge.local.external.stage import ExternalStage, plan_external_stage
from vllm_omni.edge.local.manifest import build_graph_artifact


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class SparkExternalOutputHead:
    """One admitted VitisAI graph, kept beside an otherwise ordinary vLLM model."""

    def __init__(self, model_dir: str | Path, spec_path: str | Path) -> None:
        spec = json.loads(Path(spec_path).read_text())
        if spec.get("schema_version") != 1:
            raise ValueError("Spark external-head spec has an unknown schema")
        graph = Path(spec["graph"])
        fixture = Path(spec["fixture"])
        model_index = Path(model_dir) / "model.safetensors.index.json"
        if _sha256(graph) != spec["graph_sha256"] or _sha256(fixture) != spec["fixture_sha256"]:
            raise ValueError("Spark external-head graph/fixture hash changed")
        if _sha256(model_index) != spec["checkpoint_index_sha256"]:
            raise ValueError("Spark CPU checkpoint index changed")
        identity = json.loads(Path(spec["weight_identity_report"]).read_text())
        if (identity.get("status") != "output_head_weights_equal_only"
            or identity.get("source_graph_sha256") != spec["source_graph_sha256"]
            or identity.get("bf16_checkpoint_index_sha256") != spec["checkpoint_index_sha256"]
            or len(identity.get("comparisons", [])) != 2
            or not all(row.get("bitwise_equal_after_conversion") for row in identity["comparisons"])):
            raise ValueError("Spark external head does not match this BF16 checkpoint")
        artifact = build_graph_artifact(
            graph, fmt=FORMAT_ONNX_A16W8, opset=21,
            source_model="XHToken/Spark-X2.5-1.7B",
            source_revision="448e61eb392c00f2c403185c5b56d5e0665bfaab",
            component="spark_output_head", exporter="probe_spark_amd_npu_lm_head.py --composite-with-norm",
            calibration={"fixture_sha256": spec["fixture_sha256"], "samples": 4},
            parity={"previous_top1_matches": 4},
        )
        plan = plan_external_stage(
            artifact, enumerate_devices(load_profile(use_torch=False)),
            require="npu:amd", min_fraction_on_target=0.125,
            worker_peak_rss_hint_bytes=int(spec["worker_peak_rss_hint_bytes"]),
        )
        if not plan.admitted:
            raise RuntimeError(plan.summary())
        with np.load(fixture, allow_pickle=False) as archive:
            example = archive["x"][0].copy()
        if example.shape != (1, 1, 2048):
            raise ValueError("Spark external-head fixture has unexpected shape")
        stage = ExternalStage(plan)
        try:
            placement = stage.open({"x": example}, profile_dir=spec.get("profile_dir"))
        except BaseException:
            stage.close()
            raise
        self.stage = stage
        self.plan = plan
        self.placement = placement
        self.report_path = Path(spec["report_path"])
        self.calls: list[dict[str, float]] = []
        self.reference_comparisons: list[dict[str, float | int | bool]] = []
        self.reference_compare_limit = int(spec.get("reference_compare_limit", 0))
        self.started_unix = time.time()
        self._closed = False
        atexit.register(self.close)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type != "cpu" or tuple(hidden_states.shape) != (1, 2048):
            raise ValueError(
                "the experimental Spark NPU head requires one CPU token with hidden size 2048; "
                f"received {tuple(hidden_states.shape)} on {hidden_states.device}"
            )
        x = hidden_states.detach().float().contiguous().numpy().reshape(1, 1, 2048)
        outputs, timing = self.stage.run({"x": x})
        logits = outputs["logits_concatenated"]
        if logits.shape != (1, 131072) or not np.isfinite(logits).all():
            raise RuntimeError("Spark NPU head returned invalid full logits")
        self.calls.append(timing.to_dict())
        return torch.from_numpy(logits)

    def compare_reference(self, npu_logits: torch.Tensor, cpu_logits: torch.Tensor) -> None:
        """Record numerical parity without changing the sampled NPU output."""
        if len(self.reference_comparisons) >= self.reference_compare_limit:
            return
        actual = npu_logits.detach().float().reshape(-1)
        expected = cpu_logits.detach().float().reshape(-1)
        if actual.shape != expected.shape:
            raise ValueError("Spark reference and NPU logits have different shapes")
        cpu_top1 = int(expected.argmax().item())
        npu_top1 = int(actual.argmax().item())
        error = actual - expected
        self.reference_comparisons.append({
            "call_index": len(self.calls) - 1,
            "cpu_top1": cpu_top1,
            "npu_top1": npu_top1,
            "top1_equal": cpu_top1 == npu_top1,
            "relative_l2": float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected)),
            "cpu_top1_logit": float(expected[cpu_top1]),
            "npu_at_cpu_top1": float(actual[cpu_top1]),
            "cpu_at_npu_top1": float(expected[npu_top1]),
            "npu_top1_logit": float(actual[npu_top1]),
        })

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        report: dict[str, Any] = {
            "scope": "opt-in live vLLM Spark decode output-head split",
            "started_unix": self.started_unix,
            "ended_unix": time.time(),
            "plan": self.plan.to_dict(),
            "placement": self.placement.to_dict(),
            "calls": self.calls,
            "reference_comparisons": self.reference_comparisons,
        }
        try:
            report["worker_stats"] = self.stage.stats()
        except Exception as exc:
            report["stats_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self.stage.close()
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2) + "\n")
