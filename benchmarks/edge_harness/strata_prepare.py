# SPDX-License-Identifier: Apache-2.0
"""Verify downloaded pinned weights, invoke upstream iq_pack, and bind a launch.

No weights are downloaded, copied into a second expert store, or executed by
this tool. An unfinished/unbound existing pack is never adopted automatically.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from .strata_profile import RUNTIME_REVISION, canonical_hash, file_hash, save_json, validate_target
except ImportError:
    from strata_profile import RUNTIME_REVISION, canonical_hash, file_hash, save_json, validate_target

SCHEMA = "omni-strata-preparation-v1"
PACK_REQUIRED = (
    "index.txt",
    "dense.bin",
    "native_experts.txt",
    "conversions.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/token_type.json",
    "tokenizer/chat_template.jinja",
)


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Canonical form of weight_tiers.ArtifactManifest.to_dict(), without vLLM imports."""
    value = copy.deepcopy(manifest)
    value.setdefault("schema", "omni-weight-artifacts-v1")
    value.setdefault("hash_origin", "declared")
    value.setdefault("lineage", [])
    for item in value["files"]:
        item.setdefault("role", "weights")
        item.setdefault("quantization", None)
        item.setdefault("layout", None)
    for item in value["lineage"]:
        item.setdefault("teacher_checkpoint", None)
        item.setdefault("teacher_revision", None)
    value["files"].sort(key=lambda item: item["path"])
    return canonical_hash(value)


def verify_manifest(manifest: dict[str, Any], root: Path, *, reject_extras: bool = False) -> None:
    root = root.resolve(strict=True)
    declared = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in item["path"] or ":" in item["path"]:
            raise ValueError("manifest path must stay inside its root")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("manifest file escapes its root or is not a file")
        if path.stat().st_size != item["size_bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError(f"manifest bytes changed: {item['path']}")
        declared.add(path)
    if reject_extras and any(path.resolve() not in declared for path in root.rglob("*") if path.is_file()):
        raise ValueError("directory contains an unbound file")


def inventory(root: Path, checkpoint: str, revision: str, license_name: str, *, runtime: bool = False) -> dict:
    root = root.resolve(strict=True)
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if runtime and ".git" in relative.parts:
            continue
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"bundle symlink escapes root: {relative}")
        # Include helpers, DLLs, existing pyc, profiles and all source assets.
        role = "runtime" if runtime else "prepared_pack"
        if relative.parts[0] == "tokenizer":
            role = "tokenizer"
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_hash(path),
                "role": role,
                "quantization": None,
                "layout": None,
            }
        )
    if not entries:
        raise ValueError("cannot bind an empty bundle")
    return {
        "schema": "omni-weight-artifacts-v1",
        "checkpoint": checkpoint,
        "revision": revision,
        "license": license_name,
        "hash_origin": "local_observation",
        "files": entries,
        "lineage": [],
    }


def source_for_text(target: dict[str, Any]) -> dict[str, Any]:
    validate_target(target)
    source = copy.deepcopy(target["artifact_manifest"])
    source["files"] = [item for item in source["files"] if item["role"] == "weights"]
    if not source["files"]:
        raise ValueError("target has no text weights")
    return source


def read_conversion_receipt(
    pack: Path, source: dict[str, Any], prepared: dict[str, Any], compat_bf16: bool
) -> dict[str, Any]:
    for relative in PACK_REQUIRED:
        path = pack / relative
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"incomplete prepared pack: {relative}")
    if any(path.name.startswith("experts.bin") or path.suffix == ".tmp" for path in pack.rglob("*")):
        raise ValueError("pack must use source GGUF experts and contain no unfinished files")
    record = json.loads((pack / "conversions.json").read_text(encoding="utf-8"))
    if record.get("schema") != 1 or record.get("tool") != "tools/iq_pack.py":
        raise ValueError("unexpected upstream conversion receipt schema")
    if record.get("compat_bf16") is not compat_bf16 or not isinstance(record.get("tensors"), list):
        raise ValueError("conversion mode differs from the declared preparation")
    expected = {Path(item["path"]).name: item["size_bytes"] for item in source["files"]}
    declared = {item["name"]: item["size"] for item in record.get("source_shards", [])}
    if expected != declared:
        raise ValueError("conversion receipt does not bind every source shard")
    for item in record["tensors"]:
        required = {"name", "src_type", "dst_type", "method", "exact", "max_abs_err", "source_sha256"}
        if not required <= item.keys() or type(item["exact"]) is not bool:
            raise ValueError("conversion tensor record is incomplete")
        if (
            not isinstance(item["source_sha256"], str)
            or len(item["source_sha256"]) != 64
            or not math.isfinite(item["max_abs_err"])
            or item["max_abs_err"] < 0
        ):
            raise ValueError("conversion tensor lacks a valid source hash/error")
    compatibility = None
    if compat_bf16:
        path = pack / "compat-bf16.json"
        if not path.is_file():
            raise ValueError("compatibility conversion summary is missing")
        compatibility = json.loads(path.read_text(encoding="utf-8"))
        if compatibility.get("rounding") != "nearest-even" or not isinstance(compatibility.get("tensors"), list):
            raise ValueError("unexpected compatibility conversion summary")
    return {
        "schema": SCHEMA,
        "complete": True,
        "tool_revision": RUNTIME_REVISION,
        "source_manifest_sha256": manifest_digest(source),
        "prepared_manifest_sha256": manifest_digest(prepared),
        "compat_bf16": compat_bf16,
        "conversions": record["tensors"],
        "upstream_conversions_sha256": file_hash(pack / "conversions.json"),
        "compatibility_summary": compatibility,
        "upstream_compatibility_sha256": file_hash(pack / "compat-bf16.json") if compat_bf16 else None,
        "quality_status": "conversion recorded; numerical and task quality not qualified",
    }


def command_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("PYTHON", "STRATA_")):
            env.pop(name)
    # iq_pack invokes tokenizer export with another Python process.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    return env


def compat_dependency(python: Path, runtime: Path) -> dict[str, Any]:
    """Resolve the packer's actual vendored converter before reading large weights."""
    vendor = runtime / "third_party/llama.cpp/gguf-py"
    if not (vendor / "gguf/__init__.py").is_file():
        raise FileNotFoundError(
            "compat-BF16 packing requires vendored llama.cpp gguf-py at "
            f"{vendor}; an installed gguf distribution does not satisfy tools/_paths.py"
        )
    code = (
        "import json,pathlib,sys;sys.path.insert(0,sys.argv[1]);from _paths import gguf_py;"
        "path=pathlib.Path(gguf_py()).resolve();expected=pathlib.Path(sys.argv[2]).resolve();"
        "assert path==expected,'gguf-py resolved outside the bound runtime';sys.path.insert(0,str(path));"
        "import gguf;from gguf import GGMLQuantizationType as Q,quants;"
        "assert Q.BF16 is not None and Q.Q8_0 is not None;"
        "assert callable(quants.dequantize) and callable(quants.quantize);"
        "module=pathlib.Path(gguf.__file__).resolve();"
        "assert module.is_relative_to(expected),'gguf import escaped the bound vendor';"
        "print(json.dumps({'source_path':str(path),'module_file':str(module)}))"
    )
    command = [str(python), "-I", "-B", "-X", "utf8", "-c", code, str(runtime / "tools"), str(vendor)]
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", env=command_environment(), timeout=60
    )
    if result.returncode:
        raise RuntimeError(f"compat-BF16 vendored gguf-py preflight failed: {result.stderr.strip()}")
    value = json.loads(result.stdout.strip())
    manifest = inventory(vendor, "ggml-org/llama.cpp/gguf-py", "runtime-bound", "MIT", runtime=True)
    value.update(
        probe_sha256=hashlib_sha256_text(code),
        files_count=len(manifest["files"]),
        files_sha256=canonical_hash(manifest["files"]),
        scope="observed converter path and bytes; source revision requires the runtime provenance receipt",
    )
    return value


def detect_ple(python: Path, runtime: Path, files: list[Path]) -> tuple[Path, dict[str, Any]]:
    code = (
        "import json,sys;sys.path.insert(0,sys.argv[1]);import gguf_reader as G;"
        "found=[p for p in sys.argv[2:] if any(t.name=='per_layer_token_embd.weight' "
        "for t in G.GGUFFile(p).tensors)];print(json.dumps(found))"
    )
    command = [str(python), "-I", "-B", "-X", "utf8", "-c", code, str(runtime / "tools"), *map(str, files)]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, encoding="utf-8", env=command_environment(), timeout=120
    )
    matches = json.loads(result.stdout.strip())
    if len(matches) != 1:
        raise ValueError("expected exactly one PLE table shard in the complete source model")
    return Path(matches[0]), {"probe_sha256": hashlib_sha256_text(code), "selected_shard": matches[0]}


def hashlib_sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def python_environment(python: Path) -> dict[str, Any]:
    code = """import importlib.metadata as m,json,sys
versions={}
for name in ('numpy','jinja2','regex','PyYAML','psutil','Pillow','gguf'):
    try: versions[name]=m.version(name)
    except m.PackageNotFoundError: versions[name]=None
print(json.dumps({'sys_version':sys.version,'executable':sys.executable,'dependencies':versions}))
"""
    command = [str(python), "-I", "-B", "-X", "utf8", "-c", code]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, encoding="utf-8", env=command_environment(), timeout=60
    )
    value = json.loads(result.stdout.strip())
    value["executable_sha256"] = file_hash(python)
    value["probe_sha256"] = hashlib_sha256_text(code)
    value["scope"] = "supplied native interpreter and installed distribution versions; not source-checkout assumptions"
    return value


def runtime_origin(runtime: Path, declared: dict[str, Any] | None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "declared_protocol_revision": RUNTIME_REVISION,
        "observed_git_head": None,
        "git_status": None,
        "dirty_patch": None,
        "external_receipt": declared,
        "status": "locally hashed bundle; source/build origin unverified",
    }
    if declared is not None:
        if declared.get("source_revision", RUNTIME_REVISION) != RUNTIME_REVISION:
            raise ValueError("external runtime provenance names another source revision")
        value["external_receipt_sha256"] = canonical_hash(declared)
        value["status"] = "external provenance declared; byte manifests are not an independent rebuild"
    if (runtime / ".git").exists():
        head = subprocess.run(
            ["git", "-C", str(runtime), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()
        if head != RUNTIME_REVISION:
            raise ValueError("observed Strata checkout HEAD differs from the pinned revision")
        status = subprocess.run(
            ["git", "-C", str(runtime), "status", "--porcelain"], check=True, capture_output=True, text=True, timeout=30
        ).stdout
        patch = subprocess.run(
            ["git", "-C", str(runtime), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        value.update(
            observed_git_head=head,
            git_status=status,
            dirty_patch=patch,
            dirty_patch_sha256=hashlib_sha256_text(patch),
            status="checkout HEAD observed; dirty files bound separately; binary build provenance separate",
        )
    return value


def prepare(
    target: dict[str, Any],
    *,
    runtime_root: Path,
    python_bin: Path,
    engine_file: str,
    artifact_root: Path,
    out_pack: Path,
    launch_out: Path,
    expert_ram_bytes: int,
    gpu_total_bytes: int,
    gpu_budget_bytes: int,
    host_overhead_bytes: int,
    host_capacity_bytes: int,
    hardware_snapshot: dict[str, Any],
    component_budget: dict[str, int],
    kv_type: str = "fp16",
    runtime_provenance: dict[str, Any] | None = None,
    context_tokens: int = 4096,
    max_new_tokens: int = 512,
    max_io_bytes: int = 1 << 20,
    expert_profile_file: str = "data/expert-profile.bin",
    ple_file: str | None = None,
    reuse_bound_pack: bool = False,
    windows_commit_capacity_bytes: int | None = None,
    wsl_capacity_bytes: int | None = None,
    gpu_index: int = 0,
    pack_temporary_bytes: int = 4 << 30,
    pack_runner: Callable[[list[str], Path, Path], None] | None = None,
    ple_detector: Callable[[Path, Path, list[Path]], tuple[Path, dict[str, Any]]] = detect_ple,
) -> dict[str, Any]:
    """Build one launch; tests inject pack execution, never fake a production receipt."""
    runtime = runtime_root.resolve(strict=True)
    artifacts = artifact_root.resolve(strict=True)
    python = python_bin.resolve(strict=True)
    pack = out_pack.absolute()
    launch_out = launch_out.absolute()
    if launch_out.exists():
        raise FileExistsError("launch output already exists; choose a new experiment identity")
    if pack.resolve().is_relative_to(runtime) or launch_out.resolve().is_relative_to(pack.resolve()):
        raise ValueError("pack must be outside runtime; launch and receipts must be outside pack")
    for value in (
        expert_ram_bytes,
        gpu_total_bytes,
        gpu_budget_bytes,
        host_overhead_bytes,
        host_capacity_bytes,
        context_tokens,
        max_new_tokens,
        max_io_bytes,
    ):
        if type(value) is not int or value <= 0:
            raise ValueError("all capacities and bounds must be positive integer bytes/counts")
    if gpu_budget_bytes > gpu_total_bytes or context_tokens <= max_new_tokens + 8:
        raise ValueError("GPU/context budget exceeds its declared physical/request bound")
    if type(pack_temporary_bytes) is not int or pack_temporary_bytes <= 0:
        raise ValueError("preparation temporary byte budget must be positive")
    if type(gpu_index) is not int or gpu_index < 0:
        raise ValueError("GPU index must be a nonnegative integer")
    if kv_type not in {"fp16", "int8"}:
        raise ValueError("KV precision must be explicitly fp16 or int8")
    if os.name == "nt" and windows_commit_capacity_bytes is None:
        raise ValueError("native Windows preparation requires an explicit available commit capacity")
    from vllm_omni.engine.weight_tiers import WeightTierBudget, WeightTierPlan

    components = dict(component_budget)
    if components.get("cpu_expert_cache_bytes") != expert_ram_bytes:
        raise ValueError("component budget expert cache must equal the requested expert RAM budget")
    if components.get("host_transfer_bytes", 0) < max_io_bytes:
        raise ValueError("component budget must include the complete admitted transfer/I/O bound")
    if windows_commit_capacity_bytes is not None and components.get("windows_commit_peak_bytes", 0) <= 0:
        raise ValueError("Windows route requires an explicit loading/steady commit peak budget")
    preliminary_budget = WeightTierBudget(**components)
    if (
        max(preliminary_budget.host_steady_bytes, preliminary_budget.host_loading_peak_bytes)
        != expert_ram_bytes + host_overhead_bytes + max_io_bytes
    ):
        raise ValueError("component host peak must equal expert cache + declared overhead + I/O bound")
    if max(preliminary_budget.gpu_steady_bytes, preliminary_budget.gpu_loading_peak_bytes) != gpu_budget_bytes:
        raise ValueError("component GPU peak must equal the declared GPU budget")
    source = source_for_text(target)
    compat = "unsloth" in source["checkpoint"].lower() and "q4" in target["target_id"].lower()
    binding_path = pack.with_name(pack.name + ".omni-binding.json")
    receipt_path = launch_out.with_suffix(".prepare.json")
    if pack.exists() and (not reuse_bound_pack or not binding_path.is_file()):
        raise FileExistsError("existing pack needs an explicit, verified Omni binding; unbound packs are refused")
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "verifying",
        "started_unix": time.time(),
        "target_id": target["target_id"],
        "target_sha256": canonical_hash(target),
        "runtime_revision": RUNTIME_REVISION,
        "artifact_root": str(artifacts),
        "pack": str(pack),
        "hardware_snapshot": hardware_snapshot,
        "future_projectors": [item for item in target["artifact_manifest"]["files"] if item["role"] == "mmproj"],
    }
    save_json(receipt_path, receipt)
    try:
        if compat and pack_runner is None:
            receipt["compat_dependency"] = compat_dependency(python, runtime)
        verify_manifest(source, artifacts)
        source_paths = [artifacts / item["path"] for item in source["files"]]
        original_stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths}
        native = next(path for path in source_paths if "-00001-of-" in path.name)
        runtime_manifest = inventory(runtime, "Niko1221/Strata", RUNTIME_REVISION, "MIT", runtime=True)
        required_runtime = {
            "serve/server.py",
            "tools/iq_pack.py",
            "tools/gguf_reader.py",
            "tools/strata_tokenizer.py",
            engine_file,
            expert_profile_file,
        }
        if not required_runtime <= {item["path"] for item in runtime_manifest["files"]}:
            raise ValueError("runtime bundle lacks entry points, helpers, engine or expert profile")
        receipt["source_manifest_sha256"] = manifest_digest(source)
        receipt["runtime_manifest_sha256"] = manifest_digest(runtime_manifest)
        receipt["runtime_provenance"] = runtime_origin(runtime, runtime_provenance)
        receipt["python_sha256"] = file_hash(python)
        receipt["python_environment"] = python_environment(python)
        if ple_file is None:
            ple, probe = ple_detector(python, runtime, source_paths)
            receipt["ple_probe"] = probe
        else:
            ple = (artifacts / ple_file).resolve(strict=True)
            receipt["ple_probe"] = {"selected_shard": str(ple), "source": "explicit; runtime verifies PLE tensor"}
        if ple.resolve() not in {path.resolve() for path in source_paths}:
            raise ValueError("PLE file is not a verified source shard")
        command = [
            str(python),
            "-B",
            "-X",
            "utf8",
            str(runtime / "tools/iq_pack.py"),
            "--gguf",
            str(native),
            "--out",
            str(pack),
        ]
        if compat:
            command.append("--compat-bf16")
        receipt.update(pack_command=command, compatibility_mode=compat, status="packing")
        save_json(receipt_path, receipt)
        if pack.exists():
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            if (
                binding.get("target_sha256") != canonical_hash(target)
                or binding.get("source_manifest_sha256") != manifest_digest(source)
                or binding.get("runtime_manifest_sha256") != manifest_digest(runtime_manifest)
                or binding.get("artifact_root") != str(artifacts)
            ):
                raise ValueError("existing pack binding belongs to another source/runtime")
            prepared = binding["prepared_pack_manifest"]
            verify_manifest(prepared, pack, reject_extras=True)
            receipt["reused_bound_pack"] = True
        else:
            pack.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(pack.parent).free < pack_temporary_bytes:
                raise MemoryError("insufficient SSD free space for the declared preparation temporary budget")
            log_path = launch_out.with_suffix(".prepare.log")
            if pack_runner is None:
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command,
                        cwd=runtime,
                        env=command_environment(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            else:
                pack_runner(command, runtime, log_path)
            prepared = inventory(
                pack,
                source["checkpoint"] + "/strata-pack",
                source["revision"] + "+strata-" + RUNTIME_REVISION,
                source["license"],
            )
            receipt["pack_log"] = str(log_path)
        conversion = read_conversion_receipt(pack, source, prepared, compat)
        # The pack tool must not replace or rewrite any source shard or runtime helper.
        if original_stats != {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths}:
            raise ValueError("source shard changed while packing")
        verify_manifest(runtime_manifest, runtime)
        if manifest_digest(
            inventory(runtime, "Niko1221/Strata", RUNTIME_REVISION, "MIT", runtime=True)
        ) != manifest_digest(runtime_manifest):
            raise ValueError("runtime bundle changed during preparation")
        binding = {
            "schema": SCHEMA,
            "target_sha256": canonical_hash(target),
            "source_manifest_sha256": manifest_digest(source),
            "artifact_root": str(artifacts),
            "runtime_manifest_sha256": manifest_digest(runtime_manifest),
            "prepared_pack_manifest": prepared,
            "conversion_manifest": conversion,
        }
        if not binding_path.exists():
            save_json(binding_path, binding)
        source_bytes = sum(item["size_bytes"] for item in source["files"])
        prepared_bytes = sum(item["size_bytes"] for item in prepared["files"])
        ssd_bytes = source_bytes + prepared_bytes
        components["ssd_artifact_bytes"] = ssd_bytes
        components["ssd_temporary_bytes"] = pack_temporary_bytes
        budget = WeightTierBudget(**components)
        host_demand = max(budget.host_steady_bytes, budget.host_loading_peak_bytes)
        if host_demand != expert_ram_bytes + host_overhead_bytes + max_io_bytes:
            raise ValueError("component host peak must equal expert cache + declared overhead + I/O bound")
        if max(budget.gpu_steady_bytes, budget.gpu_loading_peak_bytes) != gpu_budget_bytes:
            raise ValueError("component GPU peak must equal the declared GPU budget")
        gpu_pool = f"vram:{gpu_index}"
        plan = WeightTierPlan(
            route_id=target["target_id"] + f"-ram{expert_ram_bytes}-ctx{context_tokens}",
            artifact_manifest_sha256=manifest_digest(source),
            backend="external.strata.text.v1",
            backend_revision=RUNTIME_REVISION,
            budget=budget,
            ssd_experts="ssd-experts" in target.get("storage_route", ""),
            lookup_tables_on_demand=True,
            mtp=False,
            prefetch=False,
        )
        demands = budget.resource_demands(
            gpu_pool=gpu_pool,
            include_wsl=wsl_capacity_bytes is not None,
            include_windows_commit=windows_commit_capacity_bytes is not None,
        )
        # A declared capacity is an upper ceiling; current host availability is a separate bound.
        observed_available = None
        try:
            import psutil

            observed_available = psutil.virtual_memory().available
        except ImportError:
            pass
        effective_host_capacity = (
            min(host_capacity_bytes, observed_available) if observed_available is not None else host_capacity_bytes
        )
        capacities = {
            "host_ram": effective_host_capacity,
            gpu_pool: gpu_total_bytes,
            "ssd": ssd_bytes + min(shutil.disk_usage(artifacts).free, shutil.disk_usage(pack).free),
        }
        if windows_commit_capacity_bytes is not None:
            capacities["windows_commit"] = windows_commit_capacity_bytes
        if wsl_capacity_bytes is not None:
            capacities["wsl_ram"] = wsl_capacity_bytes
        backend = {
            "name": "external.strata.text.v1",
            "runtime_root": str(runtime),
            "runtime_revision": RUNTIME_REVISION,
            "runtime_manifest": runtime_manifest,
            "runtime_provenance": receipt["runtime_provenance"],
            "python_bin": str(python),
            "python_sha256": receipt["python_sha256"],
            "python_environment": receipt["python_environment"],
            "server_script": "serve/server.py",
            "engine_file": engine_file,
            "expert_profile_file": expert_profile_file,
            "artifact_root": str(artifacts),
            "artifact_manifest": source,
            "prepared_model_dir": str(pack),
            "prepared_pack_manifest": prepared,
            "conversion_manifest": conversion,
            "native_file": native.relative_to(artifacts).as_posix(),
            "ple_file": ple.relative_to(artifacts).as_posix(),
            "expert_ram_budget_bytes": expert_ram_bytes,
            "host_overhead_bytes": host_overhead_bytes,
            "gpu_total_bytes": gpu_total_bytes,
            "gpu_budget_bytes": gpu_budget_bytes,
            "gpu_pool": f"vram:{gpu_index}",
            "gpu_index": gpu_index,
            "context_tokens": context_tokens,
            "max_new_tokens": max_new_tokens,
            "max_io_bytes": max_io_bytes,
            "spec_tokens": 0,
            "ple_prefetch": False,
            "routing_prefetch": False,
            "io_prefetch": False,
            "io_mode": "auto",
            "ple_io": "direct",
            "kv_type": kv_type,
            "start_timeout_s": 900,
            "request_timeout_s": 600,
            "route_id": plan.route_id,
            "weight_tier_plan": plan.to_dict(),
        }
        launch = {
            "schema": "omni-strata-launch-v1",
            "backend": backend,
            "resource_budget": {"capacities": capacities, "demands": demands},
            "hardware_snapshot": hardware_snapshot,
            "stage_init_timeout_s": 900,
            "python_environment": receipt["python_environment"],
            "runtime_provenance": receipt["runtime_provenance"],
            "capacity_snapshot": {
                "host_declared_ceiling_bytes": host_capacity_bytes,
                "observed_available_host_ram_bytes": observed_available,
                "captured_unix": time.time(),
                "scope": "preparation-time snapshot; startup must recheck availability",
            },
            "cache_condition": "existing caches; preparation read source shards; cold disk not established",
            "power_condition": hardware_snapshot.get("power_condition", "unknown"),
            "preparation_receipt": str(receipt_path),
            "admission_preflight": {
                "admitted": all(demands[key] <= capacities[key] for key in demands),
                "over_budget_pools": [key for key in demands if demands[key] > capacities[key]],
            },
        }
        save_json(launch_out, launch)
        receipt.update(
            status="completed",
            conversion_manifest=conversion,
            prepared_manifest_sha256=manifest_digest(prepared),
            launch_sha256=canonical_hash(launch),
            launch_out=str(launch_out),
            binding_path=str(binding_path),
            admission_preflight=launch["admission_preflight"],
        )
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["finished_unix"] = time.time()
        save_json(receipt_path, receipt)
    return receipt


def gib(value: str) -> int:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("GiB must be positive and finite")
    return int(number * (1 << 30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("target", "runtime-root", "python-bin", "artifact-root", "out-pack", "launch-out", "hardware-snapshot"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument(
        "--component-budget",
        type=Path,
        required=True,
        help="explicit WeightTierBudget components; declarations, not measured allocations",
    )
    parser.add_argument(
        "--runtime-provenance",
        type=Path,
        help="external source archive/release digest/build receipt; unknown origin remains explicit",
    )
    parser.add_argument("--engine-file", default="engine/strata.exe" if os.name == "nt" else "engine/strata")
    parser.add_argument("--expert-profile-file", default="data/expert-profile.bin")
    parser.add_argument("--ple-file")
    parser.add_argument("--reuse-bound-pack", action="store_true")
    for name, default in (
        ("expert-ram", None),
        ("gpu-total", None),
        ("gpu-budget", None),
        ("host-overhead", None),
        ("host-capacity", None),
        ("windows-commit-capacity", None),
        ("wsl-capacity", None),
    ):
        parser.add_argument(
            "--" + name + "-gib",
            type=gib,
            default=default,
            required=name in {"expert-ram", "gpu-total", "gpu-budget", "host-overhead", "host-capacity"},
        )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--kv-type", choices=("fp16", "int8"), default="fp16")
    args = parser.parse_args()
    receipt = prepare(
        json.loads(args.target.read_text(encoding="utf-8")),
        runtime_root=args.runtime_root,
        python_bin=args.python_bin,
        engine_file=args.engine_file,
        artifact_root=args.artifact_root,
        out_pack=args.out_pack,
        launch_out=args.launch_out,
        expert_ram_bytes=args.expert_ram_gib,
        gpu_total_bytes=args.gpu_total_gib,
        gpu_budget_bytes=args.gpu_budget_gib,
        host_overhead_bytes=args.host_overhead_gib,
        host_capacity_bytes=args.host_capacity_gib,
        hardware_snapshot=json.loads(args.hardware_snapshot.read_text(encoding="utf-8")),
        component_budget=json.loads(args.component_budget.read_text(encoding="utf-8")),
        kv_type=args.kv_type,
        runtime_provenance=json.loads(args.runtime_provenance.read_text(encoding="utf-8"))
        if args.runtime_provenance
        else None,
        context_tokens=args.context_tokens,
        max_new_tokens=args.max_new_tokens,
        expert_profile_file=args.expert_profile_file,
        ple_file=args.ple_file,
        reuse_bound_pack=args.reuse_bound_pack,
        windows_commit_capacity_bytes=args.windows_commit_capacity_gib,
        wsl_capacity_bytes=args.wsl_capacity_gib,
        gpu_index=args.gpu_index,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
