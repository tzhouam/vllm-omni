"""Private whole-model image stage prototype; not registered or qualified by default.

Uses StrataTextStageClient's scheduler, single-active lifecycle, streaming, ACK,
resource lease and cancellation. The host performs PNG validation only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

from vllm_omni.engine.backends import strata, strata_vision

BACKEND_NAME = "external.strata.multimodal.v1"


def _typed_tier_plan(value):
    from vllm_omni.engine.weight_tiers import WeightTierPlan

    return WeightTierPlan.from_dict(value)


def validate_strata_multimodal_load_plan(plan, requested):
    """Experimental complete-model image gate; text proof cannot grant images."""
    try:
        strata._validate_strata_loaded_config(plan, requested, expected_backend=BACKEND_NAME)
        image = plan["image_route"]
        controls = plan["route_controls"]
        preimage = dict(image)
        image_sha = preimage.pop("identity_sha256")
        if (
            image.get("schema") != "omni-strata-image-identity-v1"
            or image_sha != hashlib.sha256(strata_vision.canonical(preimage).encode()).hexdigest()
            or controls.get("image_route") != image
            or plan.get("route_controls_sha256")
            != hashlib.sha256(strata_vision.canonical(controls).encode()).hexdigest()
            or image.get("runtime_manifest_sha256") != plan["runtime_manifest_sha256"]
            or image.get("text_artifact_manifest_sha256") != plan["artifact_manifest_sha256"]
            or controls.get("runtime_manifest_sha256") != plan["runtime_manifest_sha256"]
            or controls.get("artifact_manifest_sha256") != plan["artifact_manifest_sha256"]
            or image.get("vision_bootstrap_sha256") != plan["supervisor_bootstrap_sha256"]
            or image.get("vision_bootstrap_sha256")
            != hashlib.sha256(strata_vision.bootstrap_source(strata._BOOTSTRAP).encode()).hexdigest()
            or image.get("base_text_bootstrap_sha256") != hashlib.sha256(strata._BOOTSTRAP.encode()).hexdigest()
            or image.get("vision_adapter_sha256") != strata_vision.digest(Path(strata_vision.__file__))
            or image.get("encoder_device") != "cpu"
            or image.get("allow_cpu_fallback") is not False
            or image.get("native_startup_warmup_pixels") != 0
            or image.get("encoder_physical_gpu_identity") is not None
        ):
            raise ValueError("image route/source/control identity is incomplete or mismatched")
        observation = plan["observation_runtime"]
        observation_preimage = dict(observation)
        observation_sha = observation_preimage.pop("identity_sha256")
        if (
            observation.get("schema") != "omni-strata-observed-runtime-v1"
            or observation_sha != hashlib.sha256(strata_vision.canonical(observation_preimage).encode()).hexdigest()
            or controls.get("observation_runtime") != observation
            or observation.get("base_revision") != strata.PINNED_STRATA_REVISION
            or observation.get("dependency_revision") != strata_vision.DEPENDENCY_REVISION
            or observation.get("dependency_tree") != strata_vision.DEPENDENCY_TREE
            or observation.get("native_io_schema") != "strata-omni-io-v1"
            or observation.get("native_dll_search_policy") != "isolated_engine_and_system32"
            or observation.get("three_tier_memory_qualified") is not False
            or observation.get("supervisor_bootstrap_sha256") != image["base_text_bootstrap_sha256"]
            or observation.get("io_adapter_sha256") != strata_vision.digest(Path(strata.strata_io.__file__))
            or not isinstance(observation.get("patch_source_hashes"), dict)
            or set(observation["patch_source_hashes"]) != strata.strata_io.PATCH_SOURCES
            or any(
                not isinstance(value, dict)
                or set(value) != {"base_sha256", "patched_sha256"}
                or any(
                    not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    for digest in value.values()
                )
                for value in observation["patch_source_hashes"].values()
            )
            or not isinstance(observation.get("native_dependency_files"), list)
            or not observation["native_dependency_files"]
            or any(
                not isinstance(observation.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", observation[key])
                for key in (
                    "native_executable_sha256",
                    "patch_sha256",
                    "build_receipt_sha256",
                    "dependency_provenance_sha256",
                    "runtime_dependencies_sha256",
                    "io_adapter_sha256",
                )
            )
        ):
            raise ValueError("image route requires a nonempty verified observed native runtime identity")
        owner = plan["owned_encoder_at_load"]
        if set(owner) != {"pid", "creation_filetime_100ns"} or any(
            type(value) is not int or value <= 0 for value in owner.values()
        ):
            raise ValueError("encoder role ownership is unknown")
        audit = plan["selected_encoder_module_audit_at_load"]
        language_audit = plan["selected_native_module_audit_at_load"]
        if (
            audit.get("schema") != "omni-strata-selected-module-audit-v1"
            or audit.get("status") != "verified"
            or audit.get("role") != "encoder"
            or audit.get("pid") != owner["pid"]
            or audit.get("creation_filetime_100ns") != owner["creation_filetime_100ns"]
            or audit.get("observed_non_system_module_closure_verified") is not True
            or audit.get("pe_closure_identity_sha256") != image["encoder_pe_closure_sha256"]
            or plan.get("encoder_selected_loaded_modules_verified") is not True
            or plan.get("encoder_observed_non_system_module_closure_verified_at_load") is not True
            or language_audit.get("status") != "verified"
            or language_audit.get("schema") != "omni-strata-selected-module-audit-v1"
            or any(
                type(language_audit.get(key)) is not int or language_audit[key] <= 0
                for key in ("pid", "creation_filetime_100ns")
            )
            or not any(
                item.get("sha256") == observation["native_executable_sha256"] for item in language_audit["modules"]
            )
            or owner["pid"] == language_audit.get("pid")
            or not any(item.get("sha256") == image["encoder_sha256"] for item in audit["modules"])
        ):
            raise ValueError("separate exact encoder role/module closure is unverified")
        backend = plan["encoder_backend_selection_at_load"]
        if (
            backend.get("schema") != "strata-vision-backend-v1"
            or backend.get("device_type") != "CPU"
            or backend.get("primary_backend") != "CPU"
            or backend.get("gpu_requested") is not False
            or backend.get("cpu_fallback_available") is not True
        ):
            raise ValueError("actual CPU encoder backend selection is unverified")
        bounds = image["bounds_and_declared_budgets"]
        if any(type(value) is not int or value < 0 for value in bounds.values()) or bounds["vision_gpu_bytes"] != 0:
            raise ValueError("invalid explicit CPU image budget")
        host = bounds["vision_host_bytes"]
        scratch = bounds["vision_scratch_bytes"]
        if (
            host < image["projector_size_bytes"] + bounds["max_image_pixels"] * 32 + 2 * bounds["max_image_bytes"]
            or scratch
            < 2 * (20 + 4 * bounds["embedding_width"] * bounds["max_image_tokens"]) + bounds["max_image_bytes"]
            or bounds["max_image_tokens"] + plan["max_new_tokens"] + 8 >= plan["context_tokens"]
        ):
            raise ValueError("image weights/preprocessing/SVE/context budget is omitted")
        tier = _typed_tier_plan(plan["weight_tier_plan"])
        reserved = plan["reserved_bytes"]
        if (
            tier.backend != BACKEND_NAME
            or tier.backend_revision != strata.PINNED_STRATA_REVISION
            or tier.artifact_manifest_sha256 != plan["artifact_manifest_sha256"]
            or tier.budget.host_workspace_bytes < host + scratch
            or plan["host_overhead_bytes"] < host + scratch
            or tier.budget.ssd_temporary_bytes < scratch
            or tier.budget.ssd_artifact_bytes
            < sum(
                image[key]
                for key in (
                    "text_artifact_size_bytes",
                    "prepared_artifact_size_bytes",
                    "runtime_artifact_size_bytes",
                    "projector_size_bytes",
                )
            )
            or reserved
            != tier.budget.resource_demands(
                gpu_pool=plan["gpu_pool"],
                include_wsl="wsl_ram" in reserved,
                include_windows_commit="windows_commit" in reserved,
            )
        ):
            raise ValueError("image stage does not bind the shared typed admission budget")
        if (
            plan.get("declared_modalities") != ["text", "image"]
            or plan.get("qualified_modalities") != []
            or image.get("release_qualified") is not False
            or image.get("full_model_placement") is not None
            or image.get("all_encoder_operators_gpu_verified") is not False
            or plan.get("encoder_all_operators_placement") is not None
            or plan.get("observed_model_placement") is not None
            or plan.get("three_tier_memory_qualified") is not False
            or plan.get("request_capacity") != 1
            or plan.get("batch_size") != 1
            or plan.get("spec_tokens") != 0
            or plan.get("qualification") != "experimental_image_backend_loaded_not_neural_qualified"
            or plan.get("image_resource_budget_scope") != "declared_encoder_and_text_coexistence_not_hard_caps"
        ):
            raise ValueError("experimental image route makes an unsupported capability or qualification claim")
    except (RuntimeError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError("Strata did not verify its experimental CPU image stage load plan") from exc


class StrataMultimodalStageClient(strata.StrataTextStageClient):
    _backend_name = BACKEND_NAME
    _supports_images = True

    def __init__(self, metadata, config, ledger, reservation):
        self._vision = self._encoder_observer = None
        self._encoder_process_owner = None
        self._image_data_url = self._image_input = None
        super().__init__(metadata, config, ledger, reservation)

    def _prepare_extension(
        self, config, runtime, runtime_manifest, runtime_files, artifacts, native, tier_plan, host_overhead, gpu_budget
    ):
        from vllm_omni.engine.weight_tiers import ArtifactManifest

        if os.name != "nt" or self._observation_runtime is None:
            raise ValueError("initial image prototype requires the observed Windows route")
        self._vision = strata_vision.verify_vision_route(
            config.get("image_route"),
            runtime,
            runtime_manifest,
            runtime_files,
            artifacts,
            manifest_type=ArtifactManifest,
        )
        bounds = self._vision["config"]
        if config.get("spec_tokens", 0) != 0:
            raise ValueError("initial image routes require MTP off until independently validated")
        environment = config.get("python_environment")
        if (
            not isinstance(environment, dict)
            or not isinstance(environment.get("dependencies"), dict)
            or not isinstance(environment["dependencies"].get("Pillow"), str)
        ):
            raise ValueError("image route requires its captured Pillow/Python environment")
        if bounds["max_image_tokens"] + self._max_new_tokens + 8 >= self._context_tokens:
            raise ValueError("image tokens/output leave no admitted text context")
        if tier_plan is None:
            raise ValueError("image routes require an explicit shared tier plan")
        if (
            tier_plan.budget.host_workspace_bytes < bounds["vision_host_bytes"] + bounds["vision_scratch_bytes"]
            or tier_plan.budget.gpu_workspace_bytes < bounds["vision_gpu_bytes"]
            or host_overhead < bounds["vision_host_bytes"] + bounds["vision_scratch_bytes"]
            or gpu_budget < bounds["vision_gpu_bytes"]
        ):
            raise ValueError("shared resource budget omits encoder/scratch coexistence")
        prepared = ArtifactManifest.from_dict(config["prepared_pack_manifest"])
        self._vision["prepared_artifact_size_bytes"] = prepared.total_size_bytes
        projector = ArtifactManifest.from_dict(bounds["projector_manifest"])
        if tier_plan.budget.ssd_artifact_bytes < (
            artifacts.total_size_bytes
            + prepared.total_size_bytes
            + runtime_manifest.total_size_bytes
            + projector.total_size_bytes
        ):
            raise ValueError("SSD budget omits encoder/runtime/projector source")
        if tier_plan.budget.ssd_temporary_bytes < bounds["vision_scratch_bytes"]:
            raise ValueError("SSD temporary budget omits image/SVE scratch")
        self._vision["text_first_shard"] = str(native)
        self._vision["bootstrap_sha256"] = hashlib.sha256(self._bootstrap_source().encode()).hexdigest()
        self._vision["adapter_sha256"] = strata_vision.digest(Path(strata_vision.__file__))

    def _configure_extension(self, server_cfg):
        cfg = self._vision["config"]
        server_cfg["args"].append("--vision")
        server_cfg["vision"] = {
            "exe": self._vision["encoder"],
            "mmproj": self._vision["projector"],
            "model": self._vision["text_first_shard"],
            "gpu": cfg["encoder_device"] == "cuda",
            "threads": cfg["encoder_threads"],
            "max_tokens": cfg["max_image_tokens"],
        }

    def _bootstrap_source(self):
        return strata_vision.bootstrap_source(strata._BOOTSTRAP)

    def _extension_environment(self, env):
        cfg = self._vision["config"]
        scratch = Path(self._temporary.name).resolve() / "vision-scratch"
        if " " in str(scratch):
            raise ValueError("pinned GENI needs a space-free owned stage scratch path")
        scratch.mkdir()
        env["OMNI_STRATA_VISION_SCRATCH"] = str(scratch)
        adapter = Path(self._temporary.name) / "strata_vision.py"
        adapter.write_bytes(Path(strata_vision.__file__).read_bytes())
        if strata_vision.digest(adapter) != self._vision["adapter_sha256"]:
            raise RuntimeError("image adapter changed before launch")
        env["OMNI_STRATA_VISION_ADAPTER"] = str(adapter)
        env["OMNI_STRATA_VISION_ADAPTER_SHA256"] = self._vision["adapter_sha256"]
        env["OMNI_STRATA_VISION_ARGS"] = json.dumps(
            [
                self._vision["encoder"],
                "--mmproj",
                self._vision["projector"],
                "--model",
                self._vision["text_first_shard"],
                *(["--gpu"] if cfg["encoder_device"] == "cuda" else []),
                "--threads",
                str(cfg["encoder_threads"]),
                "--max-tokens",
                str(cfg["max_image_tokens"]),
            ]
        )
        env["OMNI_STRATA_VISION_BOUNDS"] = json.dumps(
            {key: cfg[key] for key in ("max_image_bytes", "max_image_pixels", "max_image_tokens", "embedding_width")}
            | {"max_io_bytes": self._max_io_bytes}
        )
        self._encoder_observer = strata_vision.EncoderObserver(
            env["OMNI_STRATA_OBSERVER_NONCE"],
            self._generation,
            allow_cpu_fallback=cfg["allow_cpu_fallback"],
            embedding_width=cfg["embedding_width"],
            is_live=self._encoder_is_live,
            requested_device=cfg["encoder_device"],
        )

    @staticmethod
    def _encoder_is_live(identity):
        return (
            identity is not None
            and strata._windows_process_creation_filetime(identity["pid"]) == identity["creation_filetime_100ns"]
        )

    def _new_diagnostics(self, *args, **kwargs):
        return strata._DiagnosticLog(*args, **kwargs, frame_callback=self._encoder_observer.ingest)

    def _finish_extension_plan(self):
        self._encoder_process_owner = strata_vision.OwnedWindowsProcess(self._encoder_observer.identity)
        if self._encoder_process_owner.state() != "alive":
            raise RuntimeError("owned encoder live process handle unverified")
        self._encoder_observer.check()
        self._encoder_observer.language_identity = self._diagnostics.owned_native_identity()
        if self._encoder_observer.identity == self._diagnostics.owned_native_identity():
            raise RuntimeError("encoder and language worker cannot share an owner identity")
        image_identity = self._vision["identity"] | {
            "vision_bootstrap_sha256": self._vision["bootstrap_sha256"],
            "vision_adapter_sha256": self._vision["adapter_sha256"],
            "base_text_bootstrap_sha256": hashlib.sha256(strata._BOOTSTRAP.encode()).hexdigest(),
            "prepared_artifact_size_bytes": self._vision["prepared_artifact_size_bytes"],
        }
        image_identity.pop("identity_sha256")
        image_identity["identity_sha256"] = hashlib.sha256(strata_vision.canonical(image_identity).encode()).hexdigest()
        controls = self.execution_plan["route_controls"]
        controls["image_route"] = image_identity
        self.execution_plan["route_controls_sha256"] = hashlib.sha256(
            strata_vision.canonical(controls).encode()
        ).hexdigest()
        self.execution_plan["image_route"] = image_identity
        self.execution_plan["owned_encoder_at_load"] = copy.deepcopy(self._encoder_observer.identity)
        self.execution_plan["encoder_backend_selection_at_load"] = copy.deepcopy(self._encoder_observer.backend)
        encoder_module_audit = strata.strata_io.audit_selected_loaded_modules(
            self._encoder_observer.identity,
            self._vision["runtime_root"],
            {
                "native_executable_sha256": self._vision["identity"]["encoder_sha256"],
                "native_dependency_files": self._vision["native_dependency_files"],
            },
            native_file=self._vision["config"]["encoder_file"],
            include_cuda_driver=self._vision["config"]["encoder_device"] == "cuda",
            role="encoder",
            non_system_pe_closure=self._vision["pe_dependency_closure"],
        )
        if encoder_module_audit["status"] != "verified":
            raise RuntimeError("owned encoder selected module binding is unverified")
        self.execution_plan["selected_encoder_module_audit_at_load"] = encoder_module_audit
        self.execution_plan["encoder_selected_loaded_modules_verified"] = True
        self.execution_plan["encoder_observed_non_system_module_closure_verified_at_load"] = True
        self.execution_plan["encoder_all_operators_placement"] = None
        self.execution_plan["declared_modalities"] = ["text", "image"]
        self.execution_plan["qualified_modalities"] = []
        self.execution_plan["qualification"] = "experimental_image_backend_loaded_not_neural_qualified"
        self.execution_plan["image_resource_budget_scope"] = "declared_encoder_and_text_coexistence_not_hard_caps"

    def _prompt_fields(self):
        return super()._prompt_fields() | {"image_data_url"}

    def _prepare_extension_prompt(self, prompt):
        self._image_data_url = prompt.get("image_data_url")
        self._image_input = None
        if self._image_data_url is not None:
            cfg = self._vision["config"]
            self._image_input = strata_vision.png_input(
                self._image_data_url,
                max_bytes=cfg["max_image_bytes"],
                max_pixels=cfg["max_image_pixels"],
                max_io_bytes=self._max_io_bytes,
                text_bytes=len(prompt["text"].encode()),
            )
            # Exact rendered/tokenized prompt check remains in pinned server,
            # which has fit_max_tokens=False. No estimated truncation is used.

    def _http_content(self, text):
        if self._image_data_url is None:
            return text
        return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": self._image_data_url}}]

    def _begin_io_observation(self, request):
        super()._begin_io_observation(request)
        if self._image_input is not None:
            self._encoder_observer.begin(request.request_id, request.epoch, self._image_input)

    def _finish_io_observation(self, request, *, completed, reason=None):
        report = super()._finish_io_observation(request, completed=completed, reason=reason)
        if request is not None and self._encoder_observer is not None and self._encoder_observer.active is not None:
            if completed:
                with self._diagnostics._changed:
                    self._diagnostics._changed.wait_for(
                        lambda: self._encoder_observer.retired or self._encoder_observer.active["dispatch"] is not None,
                        timeout=5,
                    )
            self._encoder_observer.finish(request.request_id, request.epoch, completed=completed, reason=reason)
        return report

    def _extension_telemetry(self, request, telemetry):
        report = self.last_image_observation() if self._image_input is not None else None
        if self._image_input is not None and (
            report is None
            or report["status"] != "complete"
            or report["request_id"] != request.request_id
            or report["epoch"] != request.epoch
            or report["generation"] != request.worker_generation
        ):
            raise RuntimeError("image neural chain lacks complete owned encoder evidence")
        if self._image_input is not None:
            native_io = telemetry.get("native_io_observation") or {}
            dispatch = report["language_dispatch"]
            if (
                native_io.get("status") != "complete"
                or native_io.get("request_id") != request.request_id
                or native_io.get("epoch") != request.epoch
                or native_io.get("generation") != request.worker_generation
                or native_io.get("native_pid") != dispatch["language_pid"]
                or native_io.get("creation_filetime_100ns") != dispatch["language_creation_filetime_100ns"]
            ):
                raise RuntimeError("image GENI completion lacks matching owned native terminal evidence")
        if report is not None:
            report["text_native_io_request_seq"] = (telemetry.get("native_io_observation") or {}).get(
                "native_request_seq"
            )
            report["text_native_io_scope"] = (telemetry.get("native_io_observation") or {}).get("scope")
        telemetry["image_observation"] = report
        telemetry["native_io_excludes_encoder"] = True
        telemetry["encoder_all_operators_placement"] = None

    def last_image_observation(self):
        if self._encoder_observer is None:
            return None
        with self._encoder_observer._lock:
            return copy.deepcopy(self._encoder_observer.last)

    def _check_extension_health(self):
        if self._encoder_observer is not None:
            self._encoder_observer.check()

    def _extension_drained(self):
        if self._encoder_observer is None:
            return True
        identity = self._encoder_observer.identity
        # Unknown/restarted roles cannot silently release a shared lease. The
        # existing Windows job still kills all descendants of this supervisor.
        if identity is None:
            return False
        try:
            owner = getattr(self, "_encoder_process_owner", None)
            if owner is None:
                owner = self._encoder_process_owner = strata_vision.OwnedWindowsProcess(identity)
            if owner.identity != identity:
                return False
            return owner.close_retired()
        except (OSError, ValueError, TypeError, KeyError):
            return False
