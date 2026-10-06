"""Fail-closed catalog gate for the edge Agent download CLI.

This module admits *artifact transport*, never execution placement, model
quality, or release qualification.  A first-weight research download is a
separate, weaker tier; a DLL architecture marker is never runtime proof.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm_omni.edge.agent.catalog import (
    CANDIDATES, CapacitySnapshot, RuntimeEvidence, preflight_artifact,
)

if TYPE_CHECKING:
    from benchmarks.edge_agent.download import Artifact


class GateError(ValueError):
    """A catalog, capacity, or evidence prerequisite is not satisfied."""


# Content hashes of reviewed, complete manifests.  A different file list,
# revision, projector, byte count, or LFS digest requires a code review.
APPROVED_MANIFEST_SHA256 = {
    "gemma4-31b-qat-q4-0": "875630253838f5bbb9c5fd141178dc5541d326d20d1f19d3a53d192faa61790b",
    "qwen3.6-35b-a3b-iq4-xs": "9188a167b09a8f43e28e9ca15b96f4f093abfb6ab6f3da72fc616b455b8988bc",
    "qwen3-30b-a3b-q4-k-m": "d34aae7a6467cb1332dddeb9a96025481100cff295130c8379294a1a77d5bf81",
}
_RUNTIME_UNVERIFIED = "no probed local runtime; architecture support is unverified"


@dataclass(frozen=True)
class ManifestBinding:
    candidate_key: str
    manifest_sha256: str
    artifact_revision: str
    model: Artifact
    projector: Artifact | None
    files: tuple[Artifact, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bind_reviewed_manifest(candidate_key: str, manifest_path: Path,
                           files: list[Artifact]) -> ManifestBinding:
    """Bind one catalog lineage to exact reviewed manifest bytes and files."""
    candidate = CANDIDATES.get(candidate_key)
    expected_hash = APPROVED_MANIFEST_SHA256.get(candidate_key)
    if candidate is None or expected_hash is None:
        raise GateError(f"candidate has no reviewed download manifest: {candidate_key}")
    actual_hash = _sha256(manifest_path)
    if actual_hash != expected_hash:
        raise GateError("manifest bytes differ from the reviewed candidate binding")
    if not files or len({(item.repo, item.revision) for item in files}) != 1:
        raise GateError("manifest must contain one immutable repository revision")
    repo, revision = files[0].repo, files[0].revision.lower()
    if candidate.artifact_url.split("/tree/")[0] != f"https://huggingface.co/{repo}":
        raise GateError("catalog candidate and manifest repository differ")
    if candidate.artifact_revision is None or candidate.artifact_revision.lower() != revision:
        raise GateError("catalog candidate and manifest revisions differ")
    models = [item for item in files if item.filename.lower().endswith(".gguf")
              and "mmproj" not in item.filename.lower()]
    projectors = [item for item in files if "mmproj" in item.filename.lower()
                  and item.filename.lower().endswith(".gguf")]
    if len(models) != 1 or len(models) + len(projectors) != len(files):
        raise GateError("reviewed GGUF bundle needs exactly one model and only projector extras")
    if len(projectors) != int("image" in candidate.modalities):
        raise GateError("projector set differs from catalog modalities")
    if candidate.size_bytes_estimate != models[0].size:
        raise GateError("catalog model byte count differs from pinned LFS metadata")
    if projectors and candidate.complete_download_bytes_estimate != sum(x.size for x in files):
        raise GateError("catalog projector byte count differs from pinned LFS metadata")
    return ManifestBinding(candidate_key, actual_hash, revision, models[0],
                           projectors[0] if projectors else None, tuple(files))


def _wsl_host_free_ram() -> int:
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
               "([long](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory)*1024"]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=12, check=True)
        value = int(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise GateError("cannot read live Windows host free RAM from WSL") from exc
    if value <= 0:
        raise GateError("Windows host returned no usable free RAM")
    return value


def _wsl_memory_limit(default: int) -> int:
    limit = default
    for path in (Path("/sys/fs/cgroup/memory.max"),
                 Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            raw = path.read_text(encoding="ascii").strip()
            if raw != "max":
                value = int(raw)
                if 0 < value < (1 << 60):
                    limit = min(limit, value)
        except (OSError, ValueError):
            pass
    return limit


def _dedicated_vram() -> tuple[int, int | None]:
    """Count only one discrete NVML pool; never add shared iGPU/NPU RAM."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            pools = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                info = pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(index))
                pools.append((int(info.free), int(info.total)))
            # A route-specific CUDA/Vulkan index is not available at this
            # early gate.  Use the least-free discrete pool rather than sum
            # or select a larger unrelated GPU.
            return min(pools, default=(0, None))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        # NVML absence or a transient device error cannot increase capacity.
        return 0, None


def live_capacity_snapshot(destination: Path) -> CapacitySnapshot:
    """Read current host, process, single-GPU and filesystem availability."""
    try:
        import psutil
    except ImportError as exc:
        raise GateError("psutil is required for a live capacity snapshot") from exc
    vm = psutil.virtual_memory()
    available, total = int(vm.available), int(vm.total)
    if available <= 0 or total <= 0:
        raise GateError("live RAM availability is unavailable")
    parent = destination.resolve()
    while not parent.exists():
        if parent == parent.parent:
            raise GateError("cannot resolve destination filesystem")
        parent = parent.parent
    disk = int(shutil.disk_usage(parent).free)
    vram, dedicated = _dedicated_vram()
    is_wsl = sys.platform == "linux" and "microsoft" in platform.release().lower()
    wsl_limit = None
    if is_wsl:
        available = min(available, _wsl_host_free_ram())
        wsl_limit = _wsl_memory_limit(total)
    return CapacitySnapshot(available, vram, disk,
                            physical_ram_bytes=total if sys.platform == "win32" else None,
                            dedicated_vram_bytes=dedicated,
                            wsl_ram_limit_bytes=wsl_limit,
                            native_windows=sys.platform == "win32")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError("evidence must be a JSON object")
    return value


def native_runtime_evidence(index_path: Path, binding: ManifestBinding) -> RuntimeEvidence:
    """Accept only hash-bound native whole-Agent load and inference evidence.

    A successful fixed fixture proves this exact binary executed the exact
    artifact, not open-world task quality, independent placement, or speed.
    """
    if sys.platform != "win32":
        raise GateError("native runtime proof can only be reused on native Windows")
    index = _load_json(index_path)
    if not str(index.get("conditions", {}).get("os_version", "")).startswith("Windows"):
        raise GateError("runtime index is not a native Windows run")
    if index.get("protocol") not in {"smoke_incomplete", "full_20x3_and_30m"}:
        raise GateError("runtime index is not a recognized native Agent run")
    from vllm_omni.edge.agent.runtime_identity import (
        imported_omni_source_sha256, loaded_runtime_sha256,
    )
    profiled_versions = index.get("conditions", {}).get("runtime_versions", {})
    if (profiled_versions.get("vllm_omni_imported_source_sha256")
            != imported_omni_source_sha256()
            or profiled_versions.get("agent_runtime_identity_sha256")
            != loaded_runtime_sha256()):
        raise GateError("native profile was produced by a different Omni runtime source")
    for path_key, hash_key in (("source_config", "source_config_sha256"),
                               ("lineage_manifest", "lineage_sha256")):
        source = Path(str(index.get(path_key, "")))
        if not source.is_file() or _sha256(source) != index.get(hash_key):
            raise GateError(f"native runtime {path_key} is absent or changed")
    artifact_hashes = {item.sha256.lower() for item in binding.files}
    candidate = CANDIDATES[binding.candidate_key]
    for route_id, provenance in index.get("artifact_provenance", {}).items():
        if not isinstance(provenance, dict):
            continue
        if (provenance.get("model_sha256") != binding.model.sha256.lower()
                or provenance.get("mmproj_sha256") != (binding.projector.sha256.lower()
                                                        if binding.projector else None)
                or candidate.precision not in str(provenance.get("precision", ""))):
            continue
        if not artifact_hashes.issuperset(
                {str(provenance.get("model_sha256")),
                 str(provenance.get("mmproj_sha256"))} - {"None"}):
            continue
        model_file = Path(str(provenance.get("model_file", "")))
        if not model_file.is_file() or _sha256(model_file) != binding.model.sha256.lower():
            continue
        if binding.projector is not None:
            projector_file = Path(str(provenance.get("mmproj_file", "")))
            if (not projector_file.is_file()
                    or _sha256(projector_file) != binding.projector.sha256.lower()):
                continue
        server = Path(str(provenance.get("server_bin", "")))
        server_hash = provenance.get("server_sha256")
        if not server.is_file() or _sha256(server) != server_hash:
            continue
        for result in index.get("results", []):
            if not isinstance(result, dict) or result.get("route_id") != route_id:
                continue
            raw_path, summary_path = Path(str(result.get("raw_jsonl", ""))), Path(str(result.get("summary", "")))
            if (not raw_path.is_file() or not summary_path.is_file()
                    or _sha256(raw_path) != result.get("raw_sha256")):
                continue
            summary = _load_json(summary_path)
            route = summary.get("routes", {}).get(route_id, {})
            if (summary.get("raw_sha256") != result.get("raw_sha256")
                    or summary.get("conditions", {}).get("runtime_versions") != profiled_versions
                    or not route.get("correctness_pass")
                    or not route.get("e2e_trace_pass")
                    or route.get("measured_successes", 0) < 1
                    or route.get("route", {}).get("artifact_sha256") != binding.model.sha256.lower()):
                continue
            manifest_seen = prepare_seen = measured_seen = vision_seen = offload_seen = False
            with raw_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("record_type") == "manifest":
                        manifest_seen = (row.get("batch_size") == 1
                                         and row.get("concurrency") == 1
                                         and row.get("run_id") == summary.get("run_id"))
                    elif (manifest_seen and row.get("record_type") == "route_prepare"
                          and row.get("route_id") == route_id
                          and row.get("run_id") == summary.get("run_id")):
                        prep = row.get("preparation", {})
                        plan = prep.get("details", {}).get("execution_plan", {})
                        pools = plan.get("hybrid_placement_evidence", {}).get("model_buffer_bytes_by_pool", {})
                        prepare_seen = (prep.get("cold_start_confirmed") is True
                                        and plan.get("model_sha256") == binding.model.sha256.lower()
                                        and plan.get("mmproj_sha256") == (binding.projector.sha256.lower()
                                                                          if binding.projector else None)
                                        and plan.get("server_sha256") == server_hash)
                        offload_seen = (prepare_seen
                                        and prep.get("details", {}).get("placement_independently_verified") is True
                                        and pools.get("host_ram", 0) > 0
                                        and pools.get("vram", 0) > 0)
                    elif (manifest_seen and prepare_seen and row.get("record_type") == "request"
                          and row.get("route_id") == route_id
                          and row.get("run_id") == summary.get("run_id")):
                        valid = (row.get("phase") == "measured"
                                 and row.get("batch_size") == 1 and row.get("concurrency") == 1
                                 and row.get("e2e_complete") is True
                                 and row.get("evaluation", {}).get("success") is True
                                 and row.get("result", {}).get("complete_agent_trace") is True
                                 and row.get("result", {}).get("artifact_id") ==
                                     route.get("route", {}).get("artifact_id"))
                        measured_seen |= valid
                        vision_seen |= valid and row.get("case", {}).get("task_class") in {
                            "browser_vision", "screen"}
            if not (manifest_seen and prepare_seen and measured_seen):
                continue
            return RuntimeEvidence(
                executable=str(server), version=str(server_hash), native_windows=True,
                formats=frozenset({candidate.precision}), vision_projector=vision_seen,
                cpu_gpu_offload=offload_seen, runtime_family=candidate.runtime_family,
                verified_candidate_keys=frozenset({candidate.key}), lazy_mmap=False)
    raise GateError("no exact native whole-Agent load and measured inference proof matches this candidate")


def verify_research_probe(path: Path, binding: ManifestBinding) -> str:
    """Verify weak one-file diagnostic; return DLL hash, not runtime proof."""
    if binding.projector is not None or binding.candidate_key != "qwen3-30b-a3b-q4-k-m":
        raise GateError("research tier currently covers only pinned one-file Qwen3-30B")
    record = _load_json(path)
    digest = record.pop("record_sha256", None)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    if digest != hashlib.sha256(encoded).hexdigest():
        raise GateError("architecture diagnostic record hash differs")
    artifact = record.get("artifact", {})
    marker = record.get("runtime_marker", {})
    expected = binding.model
    header = record.get("header_range", {})
    received = header.get("range_received_bytes")
    if (record.get("schema") != "omni-agent-gguf-architecture-probe-v1"
            or record.get("manifest_sha256") != binding.manifest_sha256
            or artifact.get("repo") != expected.repo
            or artifact.get("revision") != expected.revision.lower()
            or artifact.get("filename") != expected.filename
            or artifact.get("published_size_bytes") != expected.size
            or artifact.get("published_lfs_sha256") != expected.sha256.lower()
            or artifact.get("whole_artifact_sha256_verified") is not False
            or header.get("http_status") != 206
            or not isinstance(received, int) or not 64 <= received <= 1024 * 1024
            or header.get("range_requested_bytes") != received
            or not isinstance(header.get("range_sha256"), str)
            or len(header["range_sha256"]) != 64
            or record.get("outcome") != "architecture_marker_observed_diagnostic_only"
            or marker.get("architecture_marker_found") is not True
            or record.get("full_model_load_verified") is not False
            or record.get("inference_verified") is not False
            or record.get("qualification") is not False):
        raise GateError("architecture diagnostic does not match the pinned artifact and weak tier")
    dll = Path(str(marker.get("dll_path", "")))
    if not dll.is_file() or _sha256(dll) != marker.get("dll_sha256"):
        raise GateError("diagnostic runtime DLL is absent or changed")
    from benchmarks.edge_agent.experiments.probe_gguf_architecture import scan_runtime_marker
    architecture = record.get("gguf_metadata", {}).get("general_architecture")
    if (not isinstance(architecture, str)
            or scan_runtime_marker(dll, architecture) != marker):
        raise GateError("diagnostic architecture marker cannot be reproduced")
    return str(marker["dll_sha256"])


def gate_download(binding: ManifestBinding, snapshot: CapacitySnapshot, *,
                  runtime_index: Path | None = None,
                  research_probe: Path | None = None) -> dict[str, Any]:
    """Return audit fields only when all download prerequisites are met."""
    if (runtime_index is None) == (research_probe is None):
        raise GateError("choose exactly one of --runtime-index or --research-download")
    candidate = CANDIDATES[binding.candidate_key]
    exact_download = sum(item.size for item in binding.files)
    if snapshot.disk_free_bytes < 2 * exact_download + 5 * 1024 ** 3:
        raise GateError("live disk free space is below exact artifact staging and headroom")
    if runtime_index is not None:
        runtime = native_runtime_evidence(runtime_index, binding)
        preflight = preflight_artifact(candidate, snapshot, runtime)
        if not preflight.download_eligible:
            raise GateError("catalog measurement preflight refused: " + "; ".join(preflight.reasons))
        tier = "measurement_download_eligible_not_route_qualified"
        proof_hash = _sha256(runtime_index)
    else:
        proof_hash = verify_research_probe(research_probe, binding)
        preflight = preflight_artifact(candidate, snapshot, None)
        remaining = [reason for reason in preflight.reasons if reason != _RUNTIME_UNVERIFIED]
        if remaining:
            raise GateError("catalog capacity/policy preflight refused: " + "; ".join(remaining))
        tier = "quarantined_research_download_architecture_unverified"
    return {
        "candidate_key": binding.candidate_key,
        "manifest_sha256": binding.manifest_sha256,
        "artifact_revision": binding.artifact_revision,
        "tier": tier,
        "capacity_evidence": "E_lower_bound_only",
        "minimum_resident_bytes": preflight.minimum_resident_bytes,
        "available_combined_bytes": preflight.available_combined_bytes,
        "live_available_ram_bytes": snapshot.available_ram_bytes,
        "live_available_discrete_vram_bytes": snapshot.available_vram_bytes,
        "wsl_ram_limit_bytes": snapshot.wsl_ram_limit_bytes,
        "native_windows": snapshot.native_windows,
        "exact_download_bytes": exact_download,
        "live_disk_free_bytes": snapshot.disk_free_bytes,
        "evidence_sha256": proof_hash,
        "release_qualified": False,
    }
