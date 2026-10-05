# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Auditable *candidate* artifacts for the local, batch-one edge Agent.

This is a download gate, not an execution-plan or model-quality gate.  The
published file sizes below are estimates from the user's 2026-10-05 plan
unless a candidate's ``size_basis`` names pinned LFS metadata.  When pinned
file bytes replace a rounded plan estimate, ``declared_plan_size_gb_decimal``
preserves the user's number separately.  A GGUF's bytes
on disk are neither its loading peak nor proof that a Windows runtime
implements the architecture.
The existing Omni stage planner and resource ledger remain responsible for
actual admission once an artifact and runtime have been measured.

All sizes exposed by this module use decimal GB (10**9 bytes).  The one
published 59 GiB gpt-oss estimate is converted to decimal GB explicitly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Literal

GB = 10**9
GiB = 2**30
_SHA_REVISION = re.compile(r"[0-9a-fA-F]{40,64}\Z")


@dataclass(frozen=True)
class ArtifactCandidate:
    """One immutable quantization/pruning lineage, not a model family claim."""

    key: str
    model: str
    checkpoint_url: str
    artifact_url: str
    precision: str
    size_gb_decimal: float
    modalities: tuple[str, ...]
    task_routes: tuple[str, ...]
    declared_plan_size_gb_decimal: float | None = None
    lineage: str = "unpruned"
    status: Literal["measure", "capacity_only"] = "measure"
    resident_gb_decimal: float | None = None
    lazy_disk_gb_decimal: float = 0.0
    projector_gb_decimal: float | None = 0.0
    projector_url: str | None = None
    license_hint: str = "verify source and artifact licenses before distribution"
    size_basis: str = "user plan; not a measured runtime peak"
    runtime_family: str = "llama.cpp/GGUF"
    checkpoint_revision: str | None = None
    artifact_revision: str | None = None
    artifact_bundle_sha256: str | None = None
    base_checkpoint_url: str | None = None

    @property
    def size_bytes_estimate(self) -> int:
        return int(Decimal(str(self.size_gb_decimal)) * GB)

    @property
    def resident_bytes_lower_bound(self) -> int:
        size = self.resident_gb_decimal
        if size is None:
            size = self.size_gb_decimal - self.lazy_disk_gb_decimal
        projector = self.projector_gb_decimal or 0.0
        return int((Decimal(str(size)) + Decimal(str(projector))) * GB)

    @property
    def complete_download_bytes_estimate(self) -> int | None:
        """None if a necessary projector has no size estimate yet."""
        if self.projector_gb_decimal is None:
            return None
        return self.size_bytes_estimate + int(Decimal(str(self.projector_gb_decimal)) * GB)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _candidate(
    key: str,
    model: str,
    checkpoint: str,
    artifact: str,
    precision: str,
    size: float,
    modalities: tuple[str, ...],
    routes: tuple[str, ...],
    **options: object,
) -> ArtifactCandidate:
    return ArtifactCandidate(
        key=key,
        model=model,
        checkpoint_url=f"https://huggingface.co/{checkpoint}",
        artifact_url=f"https://huggingface.co/{artifact}",
        precision=precision,
        size_gb_decimal=size,
        modalities=modalities,
        task_routes=routes,
        **options,
    )


# This registry is a research queue.  A few entries have pinned remote
# revisions and LFS file hashes in reviewed download manifests; only
# ``inspect_local_artifact`` verifies local bytes and binds a bundle hash.
# "capacity_only" entries are intentionally never downloaded on this laptop.
_ENTRIES = (
    _candidate(
        "qwen3.6-35b-a3b-iq4-xs", "Qwen3.6-35B-A3B",
        "Qwen/Qwen3.6-35B-A3B",
        "unsloth/Qwen3.6-35B-A3B-GGUF/tree/a483e9e6cbd595906af30beda3187c2663a1118c",
        "IQ4_XS", 17.730509792, ("text", "image"),
        ("browser", "screen", "general"),
        declared_plan_size_gb_decimal=17.7,
        lineage="Unsloth UD-IQ4_XS dynamic quantization, unpruned",
        projector_gb_decimal=0.89928368,
        projector_url=("https://huggingface.co/unsloth/"
                       "Qwen3.6-35B-A3B-GGUF/blob/"
                       "a483e9e6cbd595906af30beda3187c2663a1118c/mmproj-F16.gguf"),
        artifact_revision="a483e9e6cbd595906af30beda3187c2663a1118c",
        license_hint=("official base and Unsloth GGUF repository metadata: "
                      "Apache-2.0; exact source checkpoint commit is unpublished"),
        size_basis="pinned Hugging Face LFS sizes, decimal GB; not runtime peak",
    ),
    _candidate(
        "qwen3.6-35b-a3b-q4-k-m", "Qwen3.6-35B-A3B",
        "Qwen/Qwen3.6-35B-A3B", "ggml-org/Qwen3.6-35B-A3B-GGUF/tree/main",
        "Q4_K_M", 20.4, ("text", "image"), ("browser", "screen", "general"),
        projector_gb_decimal=0.614,
        projector_url=("https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/"
                       "blob/main/mmproj-Qwen3.6-35B-A3B-Q8_0.gguf"),
        license_hint="base Apache-2.0; verify third-party artifact and projector",
    ),
    _candidate(
        "gemma4-31b-qat-q4-0", "Gemma 4 31B",
        "google/gemma-4-31B-it-qat-q4_0-gguf",
        "google/gemma-4-31B-it-qat-q4_0-gguf/tree/59dde24573e7e61570dba08b18a2e1fe246955ed",
        "Q4_0", 17.651001568,
        ("text", "image"), ("browser", "screen", "general"),
        declared_plan_size_gb_decimal=17.7,
        lineage="official QAT Q4_0 GGUF, unpruned",
        projector_gb_decimal=1.200726368,
        projector_url=("https://huggingface.co/google/"
                       "gemma-4-31B-it-qat-q4_0-gguf/blob/"
                       "59dde24573e7e61570dba08b18a2e1fe246955ed/"
                       "gemma-4-31B-it-mmproj.gguf"),
        checkpoint_revision="59dde24573e7e61570dba08b18a2e1fe246955ed",
        artifact_revision="59dde24573e7e61570dba08b18a2e1fe246955ed",
        license_hint="official Google GGUF repository metadata: Apache-2.0",
        size_basis="pinned Hugging Face LFS sizes, decimal GB; not runtime peak",
    ),
    _candidate(
        "qwen3.5-35b-a3b-q4", "Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-35B-A3B", "Advantech-EIOT/Qwen3.5-35B-A3B-GGUF",
        "Q4_K_M", 21.2, ("text", "image"), ("browser", "screen", "general"),
        projector_gb_decimal=0.514318464,
        projector_url=("https://huggingface.co/Advantech-EIOT/"
                       "Qwen3.5-35B-A3B-GGUF/blob/main/mmproj/"
                       "mmproj-Qwen3.5-35B-A3B-Q4_K_M.gguf"),
    ),
    _candidate(
        "glm4.7-flash-q4-k-m", "GLM-4.7-Flash",
        "zai-org/GLM-4.7-Flash", "bartowski/zai-org_GLM-4.7-Flash-GGUF",
        "Q4_K_M", 18.47, ("text",), ("tools", "bilingual", "general"),
    ),
    _candidate(
        "qwen3-30b-a3b-q4-k-m", "Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B-GGUF", "Qwen/Qwen3-30B-A3B-GGUF",
        "Q4_K_M", 18.6, ("text",), ("general", "tools"),
    ),
    _candidate(
        "qwen3-32b-q4-k-m", "Qwen3-32B",
        "Qwen/Qwen3-32B-GGUF", "Qwen/Qwen3-32B-GGUF",
        "Q4_K_M", 19.8, ("text",), ("general", "tools"),
    ),
    _candidate(
        "qwen3-coder-next-q4-k-m", "Qwen3-Coder-Next",
        "Qwen/Qwen3-Coder-Next", "Qwen/Qwen3-Coder-Next-GGUF",
        "Q4_K_M", 48.4, ("text",), ("code", "tools"),
    ),
    _candidate(
        "qwen3.5-122b-a10b-iq4-xs", "Qwen3.5-122B-A10B",
        "Qwen/Qwen3.5-122B-A10B", "unsloth/Qwen3.5-122B-A10B-GGUF/tree/main",
        "IQ4_XS", 60.2, ("text", "image"), ("complex", "screen"),
        projector_gb_decimal=None,
    ),
    _candidate(
        "qwen3.5-122b-a10b-iq3-xxs", "Qwen3.5-122B-A10B",
        "Qwen/Qwen3.5-122B-A10B", "unsloth/Qwen3.5-122B-A10B-GGUF/tree/main",
        "IQ3_XXS", 44.7, ("text", "image"), ("complex", "screen"),
        projector_gb_decimal=None,
    ),
    _candidate(
        "gpt-oss-120b-mxfp4", "gpt-oss-120B",
        "openai/gpt-oss-120b", "openai/gpt-oss-120b",
        "MXFP4", float(Decimal(59 * GiB) / GB), ("text",),
        ("complex", "tools"),
        size_basis="user plan: approximately 59 GiB, converted to decimal GB; not a measured runtime peak",
        runtime_family="model-specific MXFP4 backend",
        license_hint="base Apache-2.0; verify downloaded artifact",
    ),
    # The GSQ-RCO card lists Q2_0 as 37.6 + 28.8 GB and a 907,543,008 B
    # projector; lazy mapping is needed for the smaller resident estimate.
    # https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
    _candidate(
        "qwen3.8-flash-next-gsq-rco-q2-0", "Qwen3.8-Flash-Next",
        "Qwen/Qwen3.8-Flash-Next",
        "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/Q2_0", "GSQ-RCO_Q2_0",
        66.4, ("text", "image"), ("complex", "screen"),
        resident_gb_decimal=37.6, lazy_disk_gb_decimal=28.8,
        projector_gb_decimal=0.907543008,
        projector_url=("https://huggingface.co/ISTA-DASLab/"
                       "Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/main/"
                       "mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
        license_hint=("base Qwen Community License 1.0; artifact metadata says "
                      "Apache-2.0 while its card says inherited base license; "
                      "resolve before distribution"),
    ),
    _candidate(
        "minimax-m2.7-iq1-m", "MiniMax-M2.7",
        "MiniMaxAI/MiniMax-M2.7", "unsloth/MiniMax-M2.7-GGUF",
        "IQ1_M", 60.7, ("text",), ("complex", "tools"),
        license_hint="source and third-party GGUF licenses require review",
    ),
    _candidate(
        "minimax-m2.7-iq2-xxs", "MiniMax-M2.7",
        "MiniMaxAI/MiniMax-M2.7", "unsloth/MiniMax-M2.7-GGUF",
        "IQ2_XXS", 65.4, ("text",), ("complex", "tools"),
        license_hint="source and third-party GGUF licenses require review",
    ),
    # The coder variant has only the IQ1_M release (29.6 + 28.8 GB),
    # distinct from the unpruned Q2_0 artifact above.
    # https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF
    _candidate(
        "qwen3.8-flash-next-coder-gsq-rco", "Qwen3.8-Flash-Next GSQ-RCO Coder",
        "Qwen/Qwen3.8-Flash-Next",
        "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/tree/main/IQ1_M",
        "GSQ-RCO_IQ1_M", 58.4, ("text", "image"), ("complex", "code", "screen"),
        lineage="pruned: half of routed experts removed",
        resident_gb_decimal=29.6, lazy_disk_gb_decimal=28.8,
        projector_gb_decimal=0.907543008,
        projector_url=("https://huggingface.co/ISTA-DASLab/"
                       "Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/blob/main/"
                       "mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
        license_hint=("base Qwen Community License 1.0; artifact metadata says "
                      "Apache-2.0 while its card says inherited base license; "
                      "resolve before distribution"),
    ),
    # The pruned BF16 source explicitly requires language_model_only.
    # https://huggingface.co/0xSero/Qwen3.5-76B
    _candidate(
        "qwen3.5-122b-reap-76b-q4-k-m", "Qwen3.5-122B REAP 76B",
        "0xSero/Qwen3.5-76B", "0xSero/Qwen3.5-76B-GGUF",
        "Q4_K_M", 46.2, ("text",), ("complex",),
        lineage="pruned: REAP 76B text-only checkpoint from Qwen3.5-122B-A10B",
        base_checkpoint_url="https://huggingface.co/Qwen/Qwen3.5-122B-A10B",
        license_hint="pruned checkpoint says inherited Qwen base license; verify terms",
    ),
    # The GGUF card names lovedheart's pruned checkpoint as its base model.
    # https://huggingface.co/mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-GGUF
    _candidate(
        "qwen3-coder-next-reap-40b-iq4-xs", "Qwen3-Coder-Next REAP 40B",
        "lovedheart/Qwen3-Coder-Next-REAP-40B-A3B",
        "mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-GGUF/tree/main",
        "IQ4_XS", 22.2, ("text",), ("code", "tools"),
        lineage="pruned: REAP 40B from Qwen3-Coder-Next",
        base_checkpoint_url="https://huggingface.co/Qwen/Qwen3-Coder-Next",
    ),
    _candidate(
        "qwen3-coder-next-reap-40b-q4-k-m", "Qwen3-Coder-Next REAP 40B",
        "lovedheart/Qwen3-Coder-Next-REAP-40B-A3B",
        "mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-GGUF/tree/main",
        "Q4_K_M", 25.0, ("text",), ("code", "tools"),
        lineage="pruned: REAP 40B from Qwen3-Coder-Next",
        base_checkpoint_url="https://huggingface.co/Qwen/Qwen3-Coder-Next",
    ),
    # This compact GGUF is built for ds4/DwarfStar; ordinary llama.cpp
    # precision support is not proof of architecture support.
    # https://huggingface.co/ljupco/DeepSeek-V4-Flash-0731-REAP25-GGUF
    _candidate(
        "deepseek-v4-flash-0731-reap25-mixed", "DeepSeek V4 Flash 0731 REAP25",
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "ljupco/DeepSeek-V4-Flash-0731-REAP25-GGUF",
        "mixed_IQ2_Q8", 68.6, ("text",), ("complex", "code"),
        lineage="pruned: REAP25 from DeepSeek V4 Flash 0731",
        size_basis="user plan; Windows runtime support specifically unverified",
        runtime_family="ds4/DwarfStar GGUF",
    ),
    _candidate(
        "deepseek-v4-flash-0731-iq1-s", "DeepSeek V4 Flash 0731",
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "unsloth/DeepSeek-V4-Flash-0731-GGUF/tree/main/UD-IQ1_S",
        "IQ1_S", 82.5, ("text",), ("complex", "code"),
        status="capacity_only",
    ),
    # The 93.1 GB minimum is UD-IQ1_S, not the 97.6 GB UD-IQ1_M.
    # https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF
    _candidate(
        "glm5.3-flash-min-gguf", "GLM-5.3-Flash",
        "zai-org/GLM-5.3-Flash", "unsloth/GLM-5.3-Flash-GGUF/tree/main/UD-IQ1_S",
        "UD-IQ1_S", 93.1, ("text", "image"), ("complex", "screen"),
        status="capacity_only", projector_gb_decimal=None,
    ),
    # The 90.5 GB minimum is IQ1_S; the same card lists IQ1_M at 100.74 GB.
    # https://huggingface.co/bartowski/MiniMax-M3-GGUF
    _candidate(
        "minimax-m3-min-gguf", "MiniMax-M3",
        "MiniMaxAI/MiniMax-M3", "bartowski/MiniMax-M3-GGUF/tree/main/MiniMax-M3-IQ1_S",
        "IQ1_S", 90.5, ("text", "image", "video"),
        ("complex", "screen"),
        status="capacity_only", projector_gb_decimal=None,
        license_hint="MiniMax Community License; verify quantized artifact separately",
    ),
)

CANDIDATES: dict[str, ArtifactCandidate] = {item.key: item for item in _ENTRIES}
assert len(CANDIDATES) == len(_ENTRIES), "duplicate candidate key"


@dataclass(frozen=True)
class CapacitySnapshot:
    """Instantaneous bytes available to *this* process, not installed totals."""

    available_ram_bytes: int
    available_vram_bytes: int
    disk_free_bytes: int
    physical_ram_bytes: int | None = None
    dedicated_vram_bytes: int | None = None
    wsl_ram_limit_bytes: int | None = None
    native_windows: bool = True


@dataclass(frozen=True)
class RuntimeEvidence:
    """A local executable's probed capabilities, never inferred from its name."""

    executable: str
    version: str
    native_windows: bool
    formats: frozenset[str]
    vision_projector: bool
    cpu_gpu_offload: bool
    runtime_family: str
    verified_candidate_keys: frozenset[str]
    lazy_mmap: bool = False


@dataclass(frozen=True)
class PreflightResult:
    candidate_key: str
    download_eligible: bool
    status: str
    reasons: tuple[str, ...]
    estimated_file_bytes: int | None
    minimum_resident_bytes: int
    available_combined_bytes: int
    capacity_evidence: Literal["E"] = "E"


def preflight_artifact(
    candidate: ArtifactCandidate,
    snapshot: CapacitySnapshot,
    runtime: RuntimeEvidence | None,
    *,
    ram_headroom_bytes: int = 8 * GiB,
    vram_headroom_bytes: int = 2 * GiB,
    disk_headroom_bytes: int = 5 * GiB,
    download_staging_multiplier: int = 2,
) -> PreflightResult:
    """Exclude impossible downloads; never approve an execution placement.

    Shared iGPU/NPU memory is already part of RAM and is not added again.  A
    known WSL limit caps usable RAM, but native Windows tests avoid WSL by
    default.  RAM+VRAM may be combined only as a generous *lower-bound test*;
    passing it does not establish a working offload partition or peak memory.
    """
    if min(
        snapshot.available_ram_bytes, snapshot.available_vram_bytes,
        snapshot.disk_free_bytes, ram_headroom_bytes, vram_headroom_bytes,
        disk_headroom_bytes,
    ) < 0:
        raise ValueError("capacity and headroom bytes must be nonnegative")
    if download_staging_multiplier < 1:
        raise ValueError("download_staging_multiplier must be >= 1")

    usable_ram = max(0, snapshot.available_ram_bytes - ram_headroom_bytes)
    if not snapshot.native_windows and snapshot.wsl_ram_limit_bytes is not None:
        usable_ram = min(
            usable_ram, max(0, snapshot.wsl_ram_limit_bytes - ram_headroom_bytes)
        )
    usable_vram = max(0, snapshot.available_vram_bytes - vram_headroom_bytes)
    combined = usable_ram + usable_vram
    minimum = candidate.resident_bytes_lower_bound
    file_bytes = candidate.complete_download_bytes_estimate
    reasons: list[str] = []

    if candidate.status == "capacity_only":
        reasons.append("catalog policy: capacity-only candidate; do not download weights")
    if minimum > combined:
        reasons.append(
            f"resident-weight lower bound {minimum} B exceeds usable RAM+VRAM {combined} B"
        )
    if file_bytes is None:
        reasons.append("required vision projector size and artifact identity are not recorded")
    elif file_bytes * download_staging_multiplier + disk_headroom_bytes > snapshot.disk_free_bytes:
        reasons.append("insufficient disk for the complete artifact and download staging")
    if runtime is None:
        reasons.append("no probed local runtime; architecture support is unverified")
    else:
        if snapshot.native_windows and not runtime.native_windows:
            reasons.append("runtime is not native Windows")
        if runtime.runtime_family != candidate.runtime_family:
            reasons.append(
                f"runtime family {runtime.runtime_family} does not match "
                f"required {candidate.runtime_family}"
            )
        if candidate.key not in runtime.verified_candidate_keys:
            reasons.append("runtime has not verified this exact candidate architecture")
        if candidate.precision not in runtime.formats:
            reasons.append(f"runtime has not verified {candidate.precision} support")
        if "image" in candidate.modalities and not runtime.vision_projector:
            reasons.append("runtime has not verified vision projector support")
        if minimum > usable_ram and not runtime.cpu_gpu_offload:
            reasons.append("runtime has not verified CPU/GPU offload")
        if candidate.lazy_disk_gb_decimal and not runtime.lazy_mmap:
            reasons.append("runtime has not verified lazy mapping of the n-gram shard")
    if (candidate.artifact_revision is None or
            not _SHA_REVISION.fullmatch(candidate.artifact_revision)):
        reasons.append("artifact repository revision is not pinned to a commit")

    if not reasons:
        status = "download_eligible_for_measurement"
    elif candidate.status == "capacity_only":
        status = "capacity_only_refusal"
    elif minimum > combined:
        status = "capacity_lower_bound_refusal"
    else:
        status = "pending_preflight_evidence"
    return PreflightResult(
        candidate_key=candidate.key,
        download_eligible=not reasons,
        status=status,
        reasons=tuple(reasons),
        estimated_file_bytes=file_bytes,
        minimum_resident_bytes=minimum,
        available_combined_bytes=combined,
    )


@dataclass(frozen=True)
class HashedFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ResolvedArtifact:
    """Content identity after download; no claim of correctness or speed."""

    candidate_key: str
    checkpoint_revision: str | None
    artifact_revision: str
    files: tuple[HashedFile, ...]
    bundle_sha256: str
    total_bytes: int
    license_reviewed: bool

    @property
    def fully_traced(self) -> bool:
        return self.checkpoint_revision is not None and self.license_reviewed


def bind_artifact_identity(
    candidate: ArtifactCandidate, resolved: ResolvedArtifact
) -> ArtifactCandidate:
    """Copy a measured content identity into a candidate manifest.

    The candidate's file-size basis remains separate from runtime memory;
    numerical validity and task quality still require separate records.
    """
    if candidate.key != resolved.candidate_key:
        raise ValueError("candidate and resolved artifact refer to different lineages")
    return replace(
        candidate,
        checkpoint_revision=resolved.checkpoint_revision,
        artifact_revision=resolved.artifact_revision,
        artifact_bundle_sha256=resolved.bundle_sha256,
    )


def inspect_local_artifact(
    candidate_key: str,
    paths: tuple[Path, ...],
    *,
    artifact_revision: str,
    checkpoint_revision: str | None = None,
    license_reviewed: bool = False,
) -> ResolvedArtifact:
    """Hash exact downloaded bytes and bind mutable repository names to commits.

    ``paths`` is an explicit artifact file set, including any projector and
    tokenizer files.  The caller must store the returned manifest alongside
    its validation record; a hash cannot be reconstructed from a model name.
    """
    if candidate_key not in CANDIDATES:
        raise KeyError(candidate_key)
    for label, revision in (
        ("artifact_revision", artifact_revision),
        ("checkpoint_revision", checkpoint_revision),
    ):
        if revision is not None and not _SHA_REVISION.fullmatch(revision):
            raise ValueError(f"{label} must be a pinned 40-64 character hex commit")
    if not paths:
        raise ValueError("at least one artifact file is required")

    files: list[HashedFile] = []
    seen: set[str] = set()
    for supplied in paths:
        path = Path(supplied)
        if not path.is_file():
            raise FileNotFoundError(path)
        # Absolute paths make the evidence unambiguous on a particular host.
        resolved = path.resolve(strict=True)
        name = str(resolved)
        if name in seen:
            raise ValueError(f"duplicate artifact file: {name}")
        seen.add(name)
        before = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                digest.update(chunk)
        after = resolved.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"artifact changed while hashing: {name}")
        files.append(HashedFile(name, after.st_size, digest.hexdigest()))

    files.sort(key=lambda item: item.path)
    bundle = hashlib.sha256()
    # A bundle's content identity must survive moving the files to another
    # machine.  Host-specific absolute paths stay in ``files`` for audit only.
    for item in sorted(files, key=lambda entry: (entry.sha256, entry.size_bytes)):
        bundle.update(str(item.size_bytes).encode("ascii"))
        bundle.update(b"\0")
        bundle.update(item.sha256.encode("ascii"))
        bundle.update(b"\n")
    return ResolvedArtifact(
        candidate_key=candidate_key,
        checkpoint_revision=checkpoint_revision.lower() if checkpoint_revision else None,
        artifact_revision=artifact_revision.lower(),
        files=tuple(files),
        bundle_sha256=bundle.hexdigest(),
        total_bytes=sum(item.size_bytes for item in files),
        license_reviewed=license_reviewed,
    )
