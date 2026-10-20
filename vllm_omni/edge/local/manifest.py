# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""What was run, precisely enough to run it again.

A performance number in this repo is worthless without the row that produced
it, and the audit behind the 2026-09-15 proposal found three separate ways the
row goes wrong:

* **A git SHA is not what executed.** ``vllm-omni`` is checked out *and*
  installed, and the installed wheel's ``interfaces.py``, UVA offloader and
  Qwen model files hashed differently from the checkout at the recorded SHA.
  So this records both, plus which file the import actually resolved to.
* **A model name is not a checkpoint.** ``Qwen3.8-27B`` names three artifacts
  (official FP8, NVIDIA NVFP4, community GGUF) that behave differently; Spark
  names nineteen directories under ``models/``. The manifest is keyed on the
  directory contents, not the name.
* **A checkpoint is not its weights alone.** The tokenizer and the chat
  template decide what the model is even asked, and they change independently
  of the weights.

Weights are identified by name, size and count rather than by content hash:
digesting 7.7 GB on every plan costs ~20 s and answers a question nobody asked
at plan time. ``digest_weights=True`` does the full sha256 when the question is
actually being asked -- publishing a number, or chasing a checkpoint that
changed underneath a result.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Files that decide what the model is asked, as opposed to how it answers.
# Hashed in full on every manifest: together they are a few hundred KB.
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RuntimeVersions:
    """The interpreter, the wheels, and the checkouts -- all three."""

    python: str
    platform: str
    torch: str | None
    vllm: str | None
    vllm_omni: str | None
    vllm_omni_file: str
    """Where ``import vllm_omni`` actually landed. An editable install and a
    site-packages copy are different code with the same version string."""
    vllm_file: str
    checkouts: dict[str, str] = field(default_factory=dict)
    """``{repo_path: git describe}`` for the source trees, when they are git
    repositories. Dirty trees are marked; this repo's engine work happens on a
    dirty tree by design and a clean-looking SHA would be a lie."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _module_version(name: str) -> tuple[str | None, str]:
    try:
        mod = importlib.import_module(name)
    except Exception:
        return None, ""
    return getattr(mod, "__version__", None), getattr(mod, "__file__", "") or ""


def _git_describe(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
    except Exception:
        return None
    return f"{sha}{'-dirty' if dirty else ''}"


def runtime_versions() -> RuntimeVersions:
    torch_v, _ = _module_version("torch")
    vllm_v, vllm_file = _module_version("vllm")
    omni_v, omni_file = _module_version("vllm_omni")

    checkouts: dict[str, str] = {}
    for file in (omni_file, vllm_file):
        if not file:
            continue
        # walk up out of the package to the repo root
        for parent in Path(file).resolve().parents:
            described = _git_describe(parent)
            if described is not None:
                checkouts[str(parent)] = described
                break

    return RuntimeVersions(
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch_v,
        vllm=vllm_v,
        vllm_omni=omni_v,
        vllm_omni_file=omni_file,
        vllm_file=vllm_file,
        checkouts=checkouts,
    )


@dataclass(frozen=True)
class ArtifactManifest:
    """One checkpoint, identified well enough to be told from its siblings."""

    model_dir: str
    model_type: str
    architectures: tuple[str, ...]
    dtype: str | None
    weight_format: str
    """One of :mod:`vllm_omni.edge.local.capabilities`'s ``FORMAT_*``."""
    quantization: dict[str, Any] | None
    """The checkpoint's own ``quantization_config``, verbatim. Kept whole
    because ``int-quantized`` vs ``float-quantized`` and the ``ignore`` list
    are what decide whether a device can run it."""
    config_sha256: str
    tokenizer_sha256: str
    """One digest over every tokenizer/template file present, in a fixed order,
    each entry salted with its filename so a rename is not invisible."""
    tokenizer_files: tuple[str, ...]
    weight_files: tuple[str, ...]
    weight_bytes: int
    weight_sha256: str | None
    """Only when ``digest_weights=True``; ``None`` otherwise, never a stand-in."""
    revision: str | None
    """From a sibling HF cache ``refs``/snapshot path when the directory came
    from one. ``None`` for a locally produced checkpoint, which is the honest
    answer -- most of ``models/`` was quantized here."""
    runtime: RuntimeVersions
    notes: list[str] = field(default_factory=list)

    @property
    def artifact_id(self) -> str:
        """A short stable key for a performance profile.

        Deliberately built from config + tokenizer + weight *sizes*, so two
        quantizations of one model never collide, and re-running the same plan
        on the same directory reproduces the key without reading 7.7 GB.
        """
        seed = f"{self.config_sha256}:{self.tokenizer_sha256}:{self.weight_bytes}:{len(self.weight_files)}"
        return _sha256_text(seed)[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["architectures"] = list(self.architectures)
        d["tokenizer_files"] = list(self.tokenizer_files)
        d["weight_files"] = list(self.weight_files)
        d["artifact_id"] = self.artifact_id
        return d


def _detect_weight_format(config: dict[str, Any]) -> str:
    from vllm_omni.edge.local.capabilities import (
        FORMAT_AWQ,
        FORMAT_DENSE,
        FORMAT_FP8,
        FORMAT_GPTQ,
        FORMAT_INT8,
        FORMAT_NVFP4,
    )

    quant = config.get("quantization_config")
    if not quant:
        return FORMAT_DENSE
    method = str(quant.get("quant_method", "")).lower()
    if method == "compressed-tensors":
        # ``format`` is the discriminator: ``int-quantized`` is the int8 W8A8
        # build that SM >= 100 cannot execute, ``float-quantized`` is fp8.
        fmt = str(quant.get("format", "")).lower()
        if "float" in fmt or "fp8" in fmt:
            return FORMAT_FP8
        return FORMAT_INT8
    if method == "gptq":
        return FORMAT_GPTQ
    if method == "awq":
        return FORMAT_AWQ
    if method in ("modelopt", "modelopt_fp4", "nvfp4"):
        return FORMAT_NVFP4
    return method or FORMAT_DENSE


def _hf_revision(model_dir: Path) -> str | None:
    """Recover the revision when the directory is inside an HF cache snapshot.

    Layout is ``.../models--org--name/snapshots/<revision>/``. A directory that
    is not one returns ``None`` rather than a guess.
    """
    parts = model_dir.resolve().parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def build_manifest(model_dir: str | os.PathLike[str], *, digest_weights: bool = False) -> ArtifactManifest:
    """Read a checkpoint directory into an :class:`ArtifactManifest`."""
    root = Path(model_dir)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{root} has no config.json; not a checkpoint directory")
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)

    tokenizer_present = [name for name in _TOKENIZER_FILES if (root / name).is_file()]
    tok_hash = hashlib.sha256()
    for name in tokenizer_present:
        tok_hash.update(name.encode("utf-8"))
        tok_hash.update(sha256_file(root / name).encode("ascii"))

    weight_paths = sorted(root.glob("*.safetensors"))
    weight_bytes = sum(p.stat().st_size for p in weight_paths)
    weight_digest: str | None = None
    if digest_weights and weight_paths:
        h = hashlib.sha256()
        for p in weight_paths:
            h.update(p.name.encode("utf-8"))
            h.update(sha256_file(p).encode("ascii"))
        weight_digest = h.hexdigest()

    notes: list[str] = []
    if not weight_paths:
        notes.append("no *.safetensors in the directory; weight_bytes is 0 and no size budget can be formed")
    if weight_digest is None and weight_paths:
        notes.append("weights identified by name+size, not content; pass digest_weights=True to hash them")

    return ArtifactManifest(
        model_dir=str(root.resolve()),
        model_type=str(config.get("model_type", "")),
        architectures=tuple(config.get("architectures") or ()),
        dtype=config.get("dtype") or config.get("torch_dtype"),
        weight_format=_detect_weight_format(config),
        quantization=config.get("quantization_config"),
        config_sha256=_sha256_text(config_text),
        tokenizer_sha256=tok_hash.hexdigest(),
        tokenizer_files=tuple(tokenizer_present),
        weight_files=tuple(p.name for p in weight_paths),
        weight_bytes=weight_bytes,
        weight_sha256=weight_digest,
        revision=_hf_revision(root),
        runtime=runtime_versions(),
        notes=notes,
    )


# ---- exported graphs -------------------------------------------------------


@dataclass(frozen=True)
class GraphArtifact:
    """An exported graph, with everything needed to trust and to reproduce it.

    :class:`ArtifactManifest` describes a checkpoint; this describes what an
    exporter made *from* one. The extra fields are not bookkeeping. AGENTS.md
    requires a quantized or compiled artifact to carry its version, layout,
    calibration and numeric record, because the same model exported twice is
    two different things -- and this project has the scars: a Qwen3-TTS vocoder
    quantized to int8 measured -4.3 dB with correlation 0.00 while its fp16
    export was faithful at 39.2 dB. A file called ``code2wav.onnx`` says
    nothing about which of those two it is.

    ``parity`` is therefore part of the artifact's identity, not a test result
    filed next to it: an export with no numeric record is not usable, and a
    failure constrains *this artifact* rather than the model or the precision.
    """

    path: str
    sha256: str
    bytes: int
    fmt: str
    """``onnx:fp32`` / ``onnx:fp16`` / ``onnx:a16w8`` -- matched against a
    device's ``weight_formats``."""
    opset: int
    """A16W8 needs >= 21, where 16-bit QDQ is expressible without contrib ops."""
    source_model: str
    """The checkpoint this came from."""
    source_revision: str | None
    component: str
    """Which part of the model: ``code2wav``, ``vision_tower``, ``spark_sliding``."""
    exporter: str
    """Dotted path of the module that produced it, so the export is repeatable."""
    input_spec: tuple[dict[str, Any], ...] = ()
    output_spec: tuple[dict[str, Any], ...] = ()
    calibration: dict[str, Any] | None = None
    """Which data the quantizer saw. ``None`` for an unquantized export; a
    quantized one without this is not reproducible."""
    parity: dict[str, Any] | None = None
    """The numeric record against the reference implementation."""
    notes: str = ""

    @property
    def artifact_id(self) -> str:
        return _sha256_text(f"{self.sha256}:{self.fmt}:{self.opset}:{self.component}")[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def external_data_files(graph: Path) -> list[Path]:
    """Weight files an ONNX graph keeps beside itself.

    Read without materialising the tensors -- the point is to size them, and
    loading a 1.8 GB checkpoint to ask how big it is defeats the purpose.
    """
    try:
        import onnx
    except ImportError:  # pragma: no cover - onnx is a dependency in practice
        sidecar = graph.with_name(graph.name + ".data")
        return [sidecar] if sidecar.is_file() else []

    try:
        model = onnx.load(str(graph), load_external_data=False)
    except Exception:
        return []
    found: dict[str, Path] = {}
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in init.external_data:
            if entry.key == "location":
                candidate = graph.parent / entry.value
                if candidate.is_file():
                    found[entry.value] = candidate
    return sorted(found.values())


def build_graph_artifact(
    path: str | os.PathLike[str],
    *,
    fmt: str,
    opset: int,
    source_model: str,
    component: str,
    exporter: str,
    source_revision: str | None = None,
    input_spec: tuple[dict[str, Any], ...] = (),
    output_spec: tuple[dict[str, Any], ...] = (),
    calibration: dict[str, Any] | None = None,
    parity: dict[str, Any] | None = None,
    notes: str = "",
) -> GraphArtifact:
    """Digest an exported graph into a :class:`GraphArtifact`."""
    graph = Path(path)
    if not graph.is_file():
        raise FileNotFoundError(f"exported graph not found: {graph}")
    sidecars = external_data_files(graph)
    return GraphArtifact(
        path=str(graph),
        sha256=sha256_file(graph),
        # The graph *and* its weights. An ONNX file with external data is a few
        # megabytes of topology pointing at gigabytes of tensors beside it;
        # charging only the .onnx would tell the admission ledger a 1.8 GB
        # vision tower costs 2 MiB.
        bytes=graph.stat().st_size + sum(f.stat().st_size for f in sidecars),
        fmt=fmt,
        opset=int(opset),
        source_model=str(source_model),
        source_revision=source_revision,
        component=component,
        exporter=exporter,
        input_spec=input_spec,
        output_spec=output_spec,
        calibration=calibration,
        parity=parity,
        notes=notes,
    )
