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
        self.input_layout = spec.get("input_layout", "pre_final_norm")
        if self.input_layout not in ("pre_final_norm", "post_final_norm"):
            raise ValueError("Spark external-head input layout is unknown")
        self.input_name = "x" if self.input_layout == "pre_final_norm" else "normalized_x_offset_0"
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
        calibration = {"fixture_sha256": spec["fixture_sha256"], "samples": 4}
        if self.input_layout == "post_final_norm":
            source_fixture = Path(spec["source_calibration_fixture"])
            if _sha256(source_fixture) != spec["source_calibration_sha256"]:
                raise ValueError("Spark normalized-head source calibration changed")
            calibration = {
                "source_fixture_sha256": spec["source_calibration_sha256"],
                "example_fixture_sha256": spec["fixture_sha256"],
                "samples": 4,
            }
        if spec.get("calibration_manifest"):
            manifest_path = Path(spec["calibration_manifest"])
            if _sha256(manifest_path) != spec["calibration_manifest_sha256"]:
                raise ValueError("Spark external-head calibration manifest changed")
            manifest = json.loads(manifest_path.read_text())
            calibration_path = Path(spec["calibration_inputs"])
            if _sha256(calibration_path) != manifest["calibration_sha256"]:
                raise ValueError("Spark external-head calibration inputs changed")
            if self.input_layout == "pre_final_norm":
                if manifest["old_fixture_sha256"] != spec["fixture_sha256"]:
                    raise ValueError("Spark pre-norm calibration fixture changed")
            elif (manifest["input_name"] != self.input_name
                  or manifest["source_sha256"] !=
                  "632ed244cbdeadc99b3d6790bb35e5ce3827682ff8989778914712ca090fd897"):
                raise ValueError("Spark post-norm calibration does not match the source graph")
            calibration = {
                "inputs_sha256": manifest["calibration_sha256"],
                "manifest_sha256": spec["calibration_manifest_sha256"],
                "samples": manifest["shape"][0],
            }
            if spec.get("quantization_report"):
                quantization_path = Path(spec["quantization_report"])
                if _sha256(quantization_path) != spec["quantization_report_sha256"]:
                    raise ValueError("Spark external-head quantization report changed")
                quantization = json.loads(quantization_path.read_text())
                if (quantization["output_sha256"] != spec["graph_sha256"]
                    or quantization["calibration_sha256"] != manifest["calibration_sha256"]
                    or quantization["source_sha256"] != manifest["source_sha256"]):
                    raise ValueError("Spark graph does not match its quantization record")
        artifact = build_graph_artifact(
            graph, fmt=FORMAT_ONNX_A16W8, opset=21,
            source_model="XHToken/Spark-X2.5-1.7B",
            source_revision="448e61eb392c00f2c403185c5b56d5e0665bfaab",
            component="spark_output_head",
            exporter=("probe_spark_amd_npu_lm_head.py --composite-with-norm"
                      if self.input_layout == "pre_final_norm"
                      else "probe_spark_amd_npu_lm_head.py --composite-shards"),
            calibration=calibration,
        )
        plan = plan_external_stage(
            artifact, enumerate_devices(load_profile(use_torch=False)),
            require="npu:amd", min_fraction_on_target=0.125,
            worker_peak_rss_hint_bytes=int(spec["worker_peak_rss_hint_bytes"]),
        )
        if not plan.admitted:
            raise RuntimeError(plan.summary())
        with np.load(fixture, allow_pickle=False) as archive:
            example = (
                archive["x"][0].copy() if self.input_layout == "pre_final_norm"
                else archive[self.input_name].copy()
            )
        expected_shape = (1, 1, 2048) if self.input_layout == "pre_final_norm" else (1, 2048)
        if example.shape != expected_shape:
            raise ValueError("Spark external-head fixture has unexpected shape")
        stage = ExternalStage(plan)
        try:
            placement = stage.open({self.input_name: example}, profile_dir=spec.get("profile_dir"))
        except BaseException:
            stage.close()
            raise
        self.stage = stage
        self.plan = plan
        self.placement = placement
        self.report_path = Path(spec["report_path"])
        self.capture_path = Path(spec["capture_path"]) if spec.get("capture_path") else None
        self.captured_activations: list[np.ndarray] = []
        self.calls: list[dict[str, float]] = []
        self.reference_comparisons: list[dict[str, float | int | bool]] = []
        self.reference_compare_limit = int(spec.get("reference_compare_limit", 0))
        self.cpu_refine_top_k = int(spec.get("cpu_refine_top_k", 0))
        if self.cpu_refine_top_k and not 1 <= self.cpu_refine_top_k <= 256:
            raise ValueError("Spark CPU candidate refinement requires top_k in [1, 256]")
        if self.cpu_refine_top_k and spec.get("sampling_contract") != "greedy-only":
            raise ValueError("Spark CPU candidate refinement requires explicit greedy-only contract")
        self.refinement_calls: list[dict[str, float | int | bool | str]] = []
        self.started_unix = time.time()
        self._closed = False
        atexit.register(self.close)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (hidden_states.device.type not in {"cpu", "cuda"}
                or tuple(hidden_states.shape) != (1, 2048)):
            raise ValueError(
                "the experimental Spark NPU head requires one CPU/CUDA token with hidden size 2048; "
                f"received {tuple(hidden_states.shape)} on {hidden_states.device}"
            )
        input_copy_started = time.perf_counter()
        x = hidden_states.detach().float().contiguous().cpu().numpy().reshape(
            (1, 1, 2048) if self.input_layout == "pre_final_norm" else (1, 2048)
        )
        input_copy_s = time.perf_counter() - input_copy_started
        if self.capture_path is not None:
            self.captured_activations.append(x.copy())
        outputs, timing = self.stage.run({self.input_name: x})
        logits = outputs["logits_concatenated"]
        if logits.shape != (1, 131072) or not np.isfinite(logits).all():
            raise RuntimeError("Spark NPU head returned invalid full logits")
        output_copy_started = time.perf_counter()
        result = torch.from_numpy(logits).to(device=hidden_states.device)
        if hidden_states.device.type == "cuda":
            torch.cuda.synchronize(hidden_states.device)
        output_copy_s = time.perf_counter() - output_copy_started
        self.calls.append({**timing.to_dict(), "input_copy_s": input_copy_s,
                           "output_copy_s": output_copy_s})
        return result

    def refine_candidates(
        self, npu_logits: torch.Tensor, normalized: torch.Tensor,
        head_weight: torch.Tensor, cpu_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use NPU top-k retrieval and the resident BF16 head for greedy re-ranking."""
        if not self.cpu_refine_top_k:
            return npu_logits
        if (npu_logits.device.type not in {"cpu", "cuda"}
                or head_weight.device != npu_logits.device
                or normalized.device != npu_logits.device):
            raise ValueError("Spark candidate refinement requires tensors on the same CPU/CUDA device")
        started = time.perf_counter()
        candidates = torch.topk(npu_logits, self.cpu_refine_top_k, dim=-1).indices.reshape(-1)
        selected_weights = head_weight.index_select(0, candidates)
        scores = torch.nn.functional.linear(
            normalized.to(selected_weights.dtype), selected_weights
        ).float()
        refined = torch.full_like(npu_logits, -torch.inf)
        refined.scatter_(1, candidates.reshape(1, -1), scores)
        row: dict[str, float | int | bool | str] = {
            "call_index": len(self.calls) - 1,
            "candidate_count": self.cpu_refine_top_k,
            "npu_top1": int(npu_logits.argmax()),
            "refined_top1": int(refined.argmax()),
            "cpu_refine_s": time.perf_counter() - started,
            "refine_device": str(npu_logits.device),
        }
        if cpu_reference is not None:
            expected = cpu_reference.float().gather(1, candidates.reshape(1, -1))
            row["max_abs_sparse_vs_full_cpu"] = float((scores - expected).abs().max())
            row["cpu_top1_in_candidates"] = bool(
                (candidates == cpu_reference.argmax()).any()
            )
            row["refined_top1_matches_full_cpu"] = (
                row["refined_top1"] == int(cpu_reference.argmax())
            )
        self.refinement_calls.append(row)
        return refined

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
        cpu_top10 = torch.topk(expected, 10).indices
        npu_top10 = torch.topk(actual, 10).indices
        npu_top64 = torch.topk(actual, 64).indices
        npu_top2_values = torch.topk(actual, 2).values
        cpu_logp = torch.log_softmax(expected, dim=0)
        npu_logp = torch.log_softmax(actual, dim=0)
        error = actual - expected
        self.reference_comparisons.append({
            "call_index": len(self.calls) - 1,
            "cpu_top1": cpu_top1,
            "npu_top1": npu_top1,
            "top1_equal": cpu_top1 == npu_top1,
            "relative_l2": float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected)),
            "max_abs_logit_error": float(error.abs().max()),
            "cpu_to_npu_kl": float(torch.sum(cpu_logp.exp() * (cpu_logp - npu_logp))),
            "top10_overlap": int(torch.isin(cpu_top10, npu_top10).sum()),
            "cpu_top1_in_npu_top64": bool((npu_top64 == cpu_top1).any()),
            "npu_top2_margin": float(npu_top2_values[0] - npu_top2_values[1]),
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
            "input_layout": self.input_layout,
            "calls": self.calls,
            "reference_comparisons": self.reference_comparisons,
            "cpu_refine_top_k": self.cpu_refine_top_k,
            "sampling_contract": "greedy-only" if self.cpu_refine_top_k else None,
            "refinement_calls": self.refinement_calls,
        }
        try:
            report["worker_stats"] = self.stage.stats()
        except Exception as exc:
            report["stats_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self.stage.close()
        if self.capture_path is not None and self.captured_activations:
            self.capture_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.capture_path,
                x=np.stack(self.captured_activations),
            )
            report["captured_activations"] = {
                "path": str(self.capture_path),
                "count": len(self.captured_activations),
                "sha256": _sha256(self.capture_path),
            }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2) + "\n")
