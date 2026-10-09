# SPDX-License-Identifier: Apache-2.0
"""Explicit launch controls for a selected, help-verified llama.cpp runtime.

These controls freeze requested native options. Selected-file hashes and CLI
support do not establish loaded-module closure, effective placement, bounded
lazy tensor residency, or a successful model request.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

SCHEMA = "omni-llamacpp-reproducible-launch-v1"
RUNTIME_PROFILE = "b11124-1a679828f-windows-vulkan-v1"
SOURCE_COMMIT = "1a679828f3312ebc53c9285805cc8bd7c7d90366"
HELP_SHA256 = "d4fb490c47587051532f5e7a4b1d10bc40ea5878d4d03324ee6d2bee05700b88"
VERSION_SHA256 = "11816d5466694eb0a1e5abc307ac69b15eedd771eee6e86640558f9f7a643b70"
RUNTIME_MANIFEST_SHA256 = "3a9853534ac5fe93a532571252eae04fa67f937647878f8c41d480ac99b90947"


@dataclass(frozen=True)
class LlamaCppLaunchControls:
    schema: str
    runtime_profile: str
    kv_type_k: str
    kv_type_v: str
    context_shift: bool
    speculative_decoding: str
    load_mode: str
    lazy_mode: str
    enable_thinking: bool

    def arguments(self) -> list[str]:
        return [
            "--cache-type-k",
            self.kv_type_k,
            "--cache-type-v",
            self.kv_type_v,
            "--no-context-shift",
            "--spec-type",
            self.speculative_decoding,
            "--load-mode",
            self.load_mode,
            "--lazy-mode",
            self.lazy_mode,
            "--chat-template-kwargs",
            '{"enable_thinking":false}',
        ]


def parse_launch_controls(value: object) -> LlamaCppLaunchControls:
    """Reject partial controls and unsupported runtimes before payload reads.

    The v1 profile deliberately freezes f16 KV, no shifting/speculation/thinking.
    Lazy tensors map independently of ordinary weight loading in the pinned source.
    No unreviewed option, auto-selection or general CLI passthrough is accepted.
    """
    fields = set(LlamaCppLaunchControls.__dataclass_fields__)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("llama.cpp launch_controls requires its exact complete schema")
    text = fields - {"context_shift", "enable_thinking"}
    if any(type(value[key]) is not str for key in text):
        raise ValueError("llama.cpp launch_controls strings must be exact strings")
    if value["schema"] != SCHEMA or value["runtime_profile"] != RUNTIME_PROFILE:
        raise ValueError("unsupported llama.cpp launch-controls schema or runtime profile")
    if value["kv_type_k"] != "f16" or value["kv_type_v"] != "f16":
        raise ValueError("this llama.cpp baseline profile requires f16 K and V")
    if value["context_shift"] is not False or value["enable_thinking"] is not False:
        raise ValueError("this llama.cpp baseline disables context shifting and thinking")
    if value["speculative_decoding"] != "none":
        raise ValueError("this llama.cpp baseline disables speculative decoding")
    if value["load_mode"] not in {"mmap", "none"} or value["lazy_mode"] not in {"on", "off"}:
        raise ValueError("unsupported explicit llama.cpp loading or lazy control")
    return LlamaCppLaunchControls(**value)


def isolated_launch_environment(inherited: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """Remove only child argument/device overrides; never mutate the parent.

    Windows environment names are case insensitive. Removed names, never their
    potentially private values, may be recorded. Existing explicit adapter GPU
    selection is applied by the caller after this isolation.
    """
    removed = sorted(
        key
        for key in inherited
        if key.upper().startswith("LLAMA_ARG_") or key.upper() in {"GGML_VK_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"}
    )
    removed_set = set(removed)
    return {key: item for key, item in inherited.items() if key not in removed_set}, removed


def launch_controls_metadata(controls: LlamaCppLaunchControls, runtime_manifest) -> dict:
    """Bind supported options to an existing ArtifactManifest fingerprint.

    The Stage constructs/verifies the typed manifest before using this metadata.
    The fingerprint is derived from closed selected-file metadata, not a complete
    PE closure. No private manifest or alternate file-verification API is added.
    """
    if type(controls) is not LlamaCppLaunchControls or controls != parse_launch_controls(asdict(controls)):
        raise ValueError("invalid llama.cpp launch controls")
    if runtime_manifest.manifest_sha256 != RUNTIME_MANIFEST_SHA256:
        raise ValueError("unsupported llama.cpp launch-controls runtime ArtifactManifest")
    return {
        "schema": SCHEMA,
        "requested": asdict(controls),
        "runtime_artifact_manifest_sha256": runtime_manifest.manifest_sha256,
        "selected_runtime_file_count": len(runtime_manifest.files),
        "support_evidence": {
            "installed_help_sha256": HELP_SHA256,
            "installed_version_sha256": VERSION_SHA256,
            "reported_primary_source_commit": SOURCE_COMMIT,
            "scope": "installed CLI metadata and selected deployed ArtifactManifest members",
        },
        "full_dependency_closure_verified": False,
        "loaded_module_identity_verified": False,
        "effective_request_controls_verified": False,
        "model_compatibility_verified": False,
        "lazy_residency_bound_verified": False,
        "memory_claim_discounted": False,
    }


def normalized_launch_controls_bundle(value: object, manifest_value: object) -> tuple[dict, dict]:
    """Detach and validate opt-in config with the existing artifact types."""
    from vllm_omni.engine.weight_tiers import ArtifactManifest

    controls = parse_launch_controls(value)
    if not isinstance(manifest_value, Mapping):
        raise ValueError("llama.cpp launch controls require an ArtifactManifest object")
    manifest = ArtifactManifest.from_dict(manifest_value)
    launch_controls_metadata(controls, manifest)
    return asdict(controls), manifest.to_dict()
