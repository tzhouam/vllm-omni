# SPDX-License-Identifier: Apache-2.0
"""Resumable batch-one complete-request experiments through Omni's Strata stage.

The standard-library analysis/validation path does not import vLLM. Actual
execution imports StageRuntime lazily and always uses its admission and ACKs.
Chunk delivery times are not silently converted into per-token timings.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import re
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

SCHEMA = "omni-strata-profile-v1"
BANDS = ("short", "medium", "long")
RUNTIME_REVISION = "d5ea7133741e67743c0e886bb426c0ce8d69cf6c"
RAM_BUDGETS_GIB = (24, 32, 40)
UNKNOWN_IO = {
    "logical_weight_read_bytes": None,
    "physical_ssd_read_bytes": None,
    "io_wait_s": None,
    "cache_hits": None,
    "expert_cache_hit_ratio": None,
    "reason": "not exposed by the backend; absent counters are not zero",
}


def normalize_io(telemetry: dict[str, Any] | None) -> dict[str, Any]:
    raw = telemetry or {}
    result = {**UNKNOWN_IO, **raw}
    logical = raw.get("logical_file_read_bytes")
    if type(logical) is int and logical >= 0:
        result["logical_weight_read_bytes"] = logical
    ratio = raw.get("hit_rate")
    if type(ratio) in (int, float) and math.isfinite(ratio) and 0 <= ratio <= 1:
        result["expert_cache_hit_ratio"] = ratio
    wait = raw.get("ssd_wait_s")
    if type(wait) in (int, float) and math.isfinite(wait) and wait >= 0:
        result["io_wait_s"] = wait
    result["source_scope"] = "backend file-tier/status counters; physical SSD attribution remains separate"
    if logical is not None or ratio is not None:
        result["reason"] = "logical reads/cache ratio exposed; physical SSD bytes and missing counters remain unknown"
    return result


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def nearest_rank(values: list[float], fraction: float) -> float | None:
    if not 0 < fraction <= 1:
        raise ValueError("percentile fraction must be in (0, 1]")
    if not values:
        return None
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latencies must be finite nonnegative values")
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def validate_target(target: dict[str, Any]) -> None:
    manifest = target["artifact_manifest"]
    if manifest.get("schema") != "omni-weight-artifacts-v1":
        raise ValueError("target requires the Omni weight artifact manifest")
    if not re.fullmatch(r"[a-f0-9]{40}", manifest.get("revision", "")):
        raise ValueError("checkpoint revision must be a full immutable SHA")
    if target.get("runtime", {}).get("revision") != RUNTIME_REVISION:
        raise ValueError("target must use the pinned Strata revision")
    files = manifest["files"]
    paths: set[str] = set()
    groups: dict[str, set[int]] = {}
    totals: dict[str, int] = {}
    for item in files:
        name = item["path"]
        path = Path(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or re.match(r"^[A-Za-z]:", name)
            or name in paths
        ):
            raise ValueError("artifact paths must be unique relative POSIX paths")
        paths.add(name)
        if type(item["size_bytes"]) is not int or item["size_bytes"] <= 0:
            raise ValueError("artifact size must be a positive integer")
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
            raise ValueError("artifact requires an authoritative SHA256")
        match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", name)
        if match:
            prefix, index, total = match[1], int(match[2]), int(match[3])
            if total < 1 or not 1 <= index <= total or prefix in totals and totals[prefix] != total:
                raise ValueError("inconsistent GGUF split numbering")
            totals[prefix] = total
            groups.setdefault(prefix, set()).add(index)
    if not files or not groups:
        raise ValueError("target requires a complete pinned GGUF split set")
    if any(indices != set(range(1, totals[prefix] + 1)) for prefix, indices in groups.items()):
        raise ValueError("target is missing a GGUF shard")


def verify_target_files(target: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    """Check all upstream files, including lookup-table and projector shards."""
    validate_target(target)
    root = model_dir.resolve(strict=True)
    verified = []
    for item in target["artifact_manifest"]["files"]:
        path = (root / item["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"artifact escapes root or is not a file: {item['path']}")
        if path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"artifact size mismatch: {item['path']}")
        actual = file_hash(path)
        if actual != item["sha256"]:
            raise ValueError(f"artifact hash mismatch: {item['path']}")
        verified.append(dict(item, observed_sha256=actual))
    return {
        "schema": SCHEMA,
        "target_id": target["target_id"],
        "manifest_sha256": canonical_hash(target["artifact_manifest"]),
        "verified_unix": time.time(),
        "model_dir": str(root),
        "files": verified,
        "total_bytes": sum(item["size_bytes"] for item in verified),
    }


def validate_launch(target: dict[str, Any], launch: dict[str, Any]) -> None:
    backend = launch["backend"]
    if backend.get("name") != "external.strata.text.v1" or backend.get("runtime_revision") != RUNTIME_REVISION:
        raise ValueError("launch must select the pinned registered Strata backend")
    expected = target["artifact_manifest"]
    declared = backend["artifact_manifest"]
    if (declared.get("checkpoint"), declared.get("revision")) != (expected["checkpoint"], expected["revision"]):
        raise ValueError("launch checkpoint/revision differs from pinned target")
    expected_files = {item["path"]: item for item in expected["files"]}
    declared_files = {item["path"]: item for item in declared["files"]}
    required = {item["path"] for item in expected["files"] if item["role"] == "weights"}
    if not required <= declared_files.keys():
        raise ValueError("launch does not declare every pinned weight shard")
    for name, item in declared_files.items():
        pinned = expected_files.get(name)
        if pinned is None or (item["size_bytes"], item["sha256"]) != (pinned["size_bytes"], pinned["sha256"]):
            raise ValueError(f"launch artifact differs from target: {name}")


def cache_variant_launches(launch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Generate budget variants without silently enlarging a machine capacity."""
    backend = launch["backend"]
    current = backend["expert_ram_budget_bytes"]
    host_demand = launch["resource_budget"]["demands"]["host_ram"]
    if type(current) is not int or type(host_demand) is not int or host_demand < current:
        raise ValueError("declared host demand must include the expert cache budget")
    variants = {}
    for gib in RAM_BUDGETS_GIB:
        variant = json.loads(json.dumps(launch))
        variant["backend"]["expert_ram_budget_bytes"] = gib << 30
        delta = (gib << 30) - current
        variant["resource_budget"]["demands"]["host_ram"] = host_demand + delta
        for pool in ("wsl_ram", "windows_commit"):
            if pool in variant["resource_budget"]["demands"]:
                variant["resource_budget"]["demands"][pool] += delta
        if "weight_tier_plan" in variant["backend"]:
            plan = variant["backend"]["weight_tier_plan"]
            plan["budget"]["cpu_expert_cache_bytes"] = gib << 30
            for key in ("host_loading_peak_bytes", "windows_commit_peak_bytes"):
                if plan["budget"].get(key, 0):
                    plan["budget"][key] += delta
            plan["route_id"] += f"-cache{gib}gib"
            variant["backend"]["route_id"] = plan["route_id"]
        variant["experiment_variant"] = {
            "expert_ram_budget_gib": gib,
            "scope": "same artifact, runtime, MTP and prefetch settings",
        }
        demands = variant["resource_budget"]["demands"]
        capacities = variant["resource_budget"]["capacities"]
        variant["admission_preflight"] = {
            "admitted": all(demands[key] <= capacities[key] for key in demands),
            "over_budget_pools": [key for key in demands if demands[key] > capacities[key]],
        }
        variants[f"ram-{gib}gib"] = variant
    return variants


def default_suite() -> dict[str, Any]:
    """A smoke retrieval task, not an Agent or multimodal qualification suite."""
    cases = []
    for band, count in zip(BANDS, (2, 24, 96), strict=True):
        context = "The archive contains trees, rivers, and blue notebooks. 档案包含树木、河流和蓝色笔记本。\n" * count
        text = (
            context + "The final signed record says answer_code=36.\n"
            "仅回答最后签名记录的 answer_code 数字。Reply with that number only, without reasoning."
        )
        cases.append(
            {
                "id": f"retrieval-{band}",
                "length_band": band,
                "prompt": {"text": text, "max_tokens": 128, "temperature": 0},
                "quality": {"kind": "exact_text", "expected": "36"},
            }
        )
    return {
        "schema": "omni-strata-cases-v1",
        "coverage": "synthetic bilingual retrieval only",
        "full_agent_quality_coverage": False,
        "cases": cases,
    }


def validate_suite(suite: dict[str, Any]) -> None:
    seen = set()
    if suite.get("schema") != "omni-strata-cases-v1":
        raise ValueError("unsupported task suite schema")
    for case in suite["cases"]:
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case["id"])
            or case["id"] in seen
            or case["length_band"] not in BANDS
        ):
            raise ValueError("case IDs must be unique safe names and length bands must be declared")
        seen.add(case["id"])
        prompt = case["prompt"]
        if not isinstance(prompt.get("text"), str) or not prompt["text"]:
            raise ValueError("each case requires real prompt text")
        if type(prompt.get("max_tokens", 128)) is not int or prompt.get("max_tokens", 128) < 1:
            raise ValueError("max_tokens must be positive")
        if case.get("quality", {}).get("kind") not in {None, "exact_text", "json_equal", "regex"}:
            raise ValueError("unsupported task check")
    if not seen:
        raise ValueError("task suite cannot be empty")


def check_quality(
    text: str, check: dict[str, Any] | None, reference: dict[str, Any] | None = None, token_ids: list[int] | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {"passed": None, "kind": None, "reference": None}
    if check:
        kind = check["kind"]
        if kind == "exact_text":
            passed = text.strip() == check["expected"].strip()
        elif kind == "json_equal":
            try:
                passed = json.loads(text) == check["expected"]
            except json.JSONDecodeError:
                passed = False
        elif kind == "regex":
            passed = re.fullmatch(check["pattern"], text.strip()) is not None
        else:
            raise ValueError("unsupported task quality check")
        result.update(passed=passed, kind=kind)
    if reference is not None:
        result["reference"] = {
            "output_text_equal": text == reference.get("output_text"),
            "output_sha256_equal": hashlib.sha256(text.encode()).hexdigest() == reference.get("output_sha256"),
            "token_ids_equal": (
                token_ids == reference["output_token_ids"]
                if token_ids is not None and "output_token_ids" in reference
                else None
            ),
            "scope": "paired output comparison; different quantized checkpoints need task-quality review",
        }
    return result


@dataclass(frozen=True)
class ProfileOptions:
    repeats: int = 20
    warmups: int = 1
    sustained_seconds: float = 1800
    request_timeout_s: float = 600
    batch_size: int = 1
    concurrency: int = 1
    bands: tuple[str, ...] = BANDS
    cancel_after_deltas: int = 8
    cancel_after_seconds: float = 0.25
    telemetry_interval_s: float = 0.5

    def validate(self) -> None:
        if (
            type(self.batch_size) is not int
            or type(self.concurrency) is not int
            or self.batch_size != 1
            or self.concurrency != 1
        ):
            raise ValueError("only batch size 1 and one active request are supported")
        if type(self.repeats) is not int or self.repeats < 1 or type(self.warmups) is not int or self.warmups < 1:
            raise ValueError("positive repeats and separate warmups are required")
        if not self.bands or len(set(self.bands)) != len(self.bands) or any(band not in BANDS for band in self.bands):
            raise ValueError("length bands must be unique short/medium/long entries")
        for value in (self.sustained_seconds, self.telemetry_interval_s, self.cancel_after_seconds):
            if not math.isfinite(value) or value < 0:
                raise ValueError("durations must be finite and nonnegative")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("request timeout must be positive and finite")
        if type(self.cancel_after_deltas) is not int or self.cancel_after_deltas < 1:
            raise ValueError("cancellation delta threshold must be positive")


class Driver(Protocol):
    async def start(self) -> dict[str, Any]: ...
    def stream(self, request_id: str, prompt: dict[str, Any]) -> AsyncIterator[dict[str, Any]]: ...
    async def cancel(self, request_id: str) -> dict[str, Any]: ...
    def usage(self) -> dict[str, Any]: ...
    async def close(self) -> dict[str, Any]: ...


class OmniStageDriver:
    """Use the production StageRuntime, resource ledger and credited stream."""

    def __init__(self, launch: dict[str, Any], *, resource_ledger: Any) -> None:
        self.launch = launch
        self.resource_ledger = resource_ledger
        self.runtime = None
        self.pool = None
        self.client = None

    async def start(self) -> dict[str, Any]:
        from vllm_omni.config.stage_config import (
            DeployConfig,
            PipelineConfig,
            StageDeployConfig,
            StageExecutionType,
            StagePipelineConfig,
            merge_pipeline_deploy,
        )
        from vllm_omni.engine.stage_runtime import StageRuntime

        if self.launch["backend"].get("name") != "external.strata.text.v1":
            raise ValueError("Strata harness requires the registered external.strata.text.v1 backend")
        pipeline = PipelineConfig(
            model_type="strata_whole_model",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="text",
                    execution_type=StageExecutionType.GRAPH,
                    final_output=True,
                    final_output_type="text",
                ),
            ),
        )
        deploy = DeployConfig(
            async_chunk=False,
            stages=[
                StageDeployConfig(
                    stage_id=0,
                    backend=self.launch["backend"],
                    resource_budget=self.launch["resource_budget"],
                )
            ],
        )
        configs = [stage.to_omegaconf() for stage in merge_pipeline_deploy(pipeline, deploy)]
        self.runtime = StageRuntime(
            configs,
            "local-strata-profile",
            "",
            async_chunk=False,
            stage_init_timeout=int(self.launch.get("stage_init_timeout_s", 600)),
            resource_ledger=self.resource_ledger,
        )
        # Initialization is synchronous in Omni; keep it off the event loop.
        await asyncio.to_thread(self.runtime.initialize)
        self.pool = self.runtime.stage_pools[0]
        self.client = self.pool.stage_client
        return {
            "execution_plan": getattr(self.client, "execution_plan", None),
            "ledger": self.runtime.resource_ledger.snapshot(),
            "loaded_client_module": type(self.client).__module__,
        }

    async def stream(self, request_id: str, prompt: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        state = SimpleNamespace(sampling_params_list=[None])
        await self.pool.submit_initial(request_id, state, dict(prompt, stream_agent=True))
        sequence = 0
        try:
            while True:
                delta = await self.client.receive_agent_delta(request_id)
                if delta is None:
                    break
                sequence += 1
                text, produced_monotonic_s = delta
                yield {
                    "kind": "delta",
                    "request_id": request_id,
                    "sequence": sequence,
                    "text": text,
                    "produced_monotonic_s": produced_monotonic_s,
                    "timing_scope": "credited text delivery; may contain multiple tokens",
                }
            while (output := self.pool.poll_graph_output(0)) is None:
                await asyncio.sleep(0.005)
            try:
                if output.request_id != request_id:
                    raise RuntimeError("cross-request terminal output")
                metrics = output.metrics or {}
                custom = output.custom_output or {}
                text = output.outputs[0].text if output.outputs else ""
                raw_ids = output.outputs[0].token_ids if output.outputs else None
                yield {
                    "kind": "terminal",
                    "request_id": request_id,
                    "text": text,
                    "finished": not bool(output.error),
                    "error": output.error,
                    "reasoning_content": custom.get("reasoning_content", ""),
                    "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
                    "stage_event": custom.get("stage_event"),
                    "metrics": metrics,
                    "output_token_ids": raw_ids if raw_ids else None,
                }
            finally:
                output.release_stage_buffers()
                # ACK uses call_soon_threadsafe; yield before the next request.
                await asyncio.sleep(0)
        finally:
            self.pool.release_binding(request_id)

    async def cancel(self, request_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        await self.client.abort_requests_async([request_id])
        self.pool.release_binding(request_id)
        # Retiring the complete route is required to prove native I/O/DMA drained.
        closed = await self.close()
        snapshot = self.resource_ledger.snapshot()
        require_drained_ledger(snapshot)
        return {
            "wall_s": time.perf_counter() - started,
            "requires_route_reload": True,
            "drained": True,
            "close": closed,
            "ledger": snapshot,
        }

    def usage(self) -> dict[str, Any]:
        return {
            "ledger": self.runtime.resource_ledger.snapshot() if self.runtime else None,
            "placement": getattr(self.client, "execution_plan", None),
        }

    async def close(self) -> dict[str, Any]:
        if self.runtime is not None:
            await asyncio.to_thread(self.runtime.shutdown)
            return {"ledger_after_shutdown": self.runtime.resource_ledger.snapshot()}
        return {"ledger_after_shutdown": None}


def require_drained_ledger(snapshot: dict[str, Any]) -> None:
    """A quarantined or retained old route cannot be replaced with fresh ceilings."""
    if (
        not isinstance(snapshot, dict)
        or not isinstance(snapshot.get("owners"), list)
        or not isinstance(snapshot.get("quarantined"), list)
        or not isinstance(snapshot.get("reserved"), dict)
        or any(type(value) is not int or value < 0 for value in snapshot["reserved"].values())
    ):
        raise RuntimeError("retired route has no complete resource drain proof; recovery is refused")
    if snapshot["quarantined"] or snapshot["owners"] or any(snapshot["reserved"].values()):
        raise RuntimeError("retired route retains or quarantines resource claims; recovery is refused")


class Telemetry:
    """Sample all visible GPUs and host counters without choosing GPU index 0."""

    def __init__(self, path: Path, interval_s: float) -> None:
        self.path, self.interval_s = path, interval_s
        self.stop_event = threading.Event()
        self.thread = None
        self.samples = 0
        self.errors: dict[str, str] = {}

    def start(self) -> None:
        if not self.interval_s:
            return

        def worker() -> None:
            psutil = nvml = None
            handles = []
            try:
                import psutil as psutil_module

                psutil = psutil_module
            except ImportError as exc:
                self.errors["host"] = str(exc)
            try:
                import pynvml

                pynvml.nvmlInit()
                nvml = pynvml
                handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())]
            except Exception as exc:
                self.errors["gpu"] = f"{type(exc).__name__}: {exc}"
            try:
                with self.path.open("a", encoding="utf-8") as output:
                    while not self.stop_event.is_set():
                        row: dict[str, Any] = {"unix": time.time(), "host": None, "gpus": []}
                        if psutil:
                            try:
                                memory = psutil.virtual_memory()
                                tree = [psutil.Process()] + psutil.Process().children(recursive=True)
                                rss = sum(process.memory_info().rss for process in tree if process.is_running())
                                disk = psutil.disk_io_counters()
                                row["host"] = {
                                    "total_ram_bytes": memory.total,
                                    "available_ram_bytes": memory.available,
                                    "used_ram_bytes": memory.total - memory.available,
                                    "process_tree_rss_sum_bytes": rss,
                                    "system_disk_read_bytes": disk.read_bytes if disk else None,
                                    "swap_used_bytes": psutil.swap_memory().used,
                                }
                            except Exception as exc:
                                self.errors["host_sample"] = f"{type(exc).__name__}: {exc}"
                        for handle in handles:
                            gpu: dict[str, Any] = {}
                            getters = {
                                "uuid": lambda: nvml.nvmlDeviceGetUUID(handle),
                                "name": lambda: nvml.nvmlDeviceGetName(handle),
                                "used_vram_bytes": lambda: nvml.nvmlDeviceGetMemoryInfo(handle).used,
                                "total_vram_bytes": lambda: nvml.nvmlDeviceGetMemoryInfo(handle).total,
                                "power_mw": lambda: nvml.nvmlDeviceGetPowerUsage(handle),
                                "temperature_c": lambda: nvml.nvmlDeviceGetTemperature(
                                    handle, nvml.NVML_TEMPERATURE_GPU
                                ),
                            }
                            for key, getter in getters.items():
                                try:
                                    value = getter()
                                    gpu[key] = value.decode() if isinstance(value, bytes) else value
                                except Exception:
                                    gpu[key] = None
                            row["gpus"].append(gpu)
                        output.write(json.dumps(row) + "\n")
                        output.flush()
                        self.samples += 1
                        self.stop_event.wait(self.interval_s)
            finally:
                if nvml:
                    nvml.nvmlShutdown()

        self.thread = threading.Thread(target=worker, name="strata-profile-telemetry", daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        return {
            "path": self.path.name,
            "samples_this_attempt": self.samples,
            "errors": self.errors,
            "interval_s": self.interval_s,
            "scope": "whole host and each whole visible GPU",
            "limitations": [
                "sampled peaks may miss transients",
                "tree RSS can count shared pages twice",
                "system disk reads include unrelated processes and storage devices",
                "system disk reads do not establish model-attributable physical SSD bytes",
                "GPU power is not whole-device power",
                "WSL RAM is not an extra RAM pool",
            ],
        }


def read_records(out: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((out / "requests").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA:
            raise ValueError(f"unknown request evidence schema: {path}")
        if (
            record.get("input_sha256") != canonical_hash(record["input"])
            or record.get("output_sha256") != hashlib.sha256(record["output_text"].encode()).hexdigest()
        ):
            raise ValueError(f"request evidence hash mismatch: {path}")
        if record.get("batch_size") != 1 or record.get("concurrency") != 1:
            raise ValueError(f"request evidence is outside the batch-one protocol: {path}")
        records.append(record)
    return records


def summarize(records: list[dict[str, Any]], options: ProfileOptions, suite: dict[str, Any]) -> dict[str, Any]:
    completed = [row for row in records if row.get("status") == "completed"]
    keys = [row["sample_key"] for row in completed]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate successful sample keys in evidence")
    groups = {}
    for band in BANDS:
        rows = [row for row in completed if row["phase"] == "measured" and row["length_band"] == band]
        walls = [row["full_response_s"] for row in rows]
        first = [row["first_output_s"] for row in rows if row["first_output_s"] is not None]
        groups[band] = {
            "n": len(rows),
            "full_response_p50_s": nearest_rank(walls, 0.5),
            "full_response_p95_s": nearest_rank(walls, 0.95),
            "first_output_n": len(first),
            "first_output_p50_s": nearest_rank(first, 0.5),
            "first_output_p95_s": nearest_rank(first, 0.95),
            "raw_full_response_s": walls,
            "raw_first_output_s": first,
        }
    sustained = [row for row in completed if row["phase"] == "sustained"]
    # Request spans are real elapsed work; idle downtime between resumed runs does not count.
    sustained_s = sum(row["full_response_s"] for row in sustained)
    segments: dict[str, float] = {}
    for row in sustained:
        segment = row.get("sustained_segment_id", "legacy-unidentified")
        segments[segment] = segments.get(segment, 0) + row["full_response_s"]
    longest_segment_s = max(segments.values(), default=0)
    warm_bands = {row["length_band"] for row in completed if row["phase"] == "warmup"}
    measured = [row for row in completed if row["phase"] in {"measured", "sustained"}]
    task_pass = bool(measured) and all(row["quality"]["passed"] is True for row in measured)
    timing_complete = (
        all(groups[band]["n"] >= 20 for band in BANDS) and warm_bands == set(BANDS) and longest_segment_s >= 1800
    )
    return {
        "groups": groups,
        "sustained_active_request_s": sustained_s,
        "sustained_segments_active_s": segments,
        "longest_uninterrupted_sustained_active_s": longest_segment_s,
        "sustained_requests": len(sustained),
        "warmup_bands": sorted(warm_bands),
        "timing_protocol_complete": timing_complete,
        "task_checks_passed": task_pass,
        "quality_coverage": suite.get("coverage", "unspecified"),
        "qualification": "not release-qualified by profiling alone",
        "remaining_gates": [
            "review reference/task quality including complete declared modalities",
            "verify executed placement and memory admission",
            "review cancellation, state isolation and stability evidence",
            "compare same-input unsplit baseline and power conditions",
        ],
        "successful_complete_requests": len(completed),
        "natural_completion_count": sum(row.get("finish_reason") == "stop" for row in completed),
        "output_budget_exhausted_count": sum(row.get("finish_reason") == "length" for row in completed),
        "failed_attempts": sum(row.get("status") == "failed" for row in records),
    }


async def collect_request(
    driver: Driver,
    case: dict[str, Any],
    phase: str,
    index: int,
    options: ProfileOptions,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_id = "strata-" + uuid.uuid4().hex
    started = time.perf_counter()
    row: dict[str, Any] = {
        "schema": SCHEMA,
        "sample_key": f"{phase}-{case['id']}-{index:06d}",
        "request_id": request_id,
        "case_id": case["id"],
        "phase": phase,
        "length_band": case["length_band"],
        "batch_size": 1,
        "concurrency": 1,
        "input": case["prompt"],
        "input_sha256": canonical_hash(case["prompt"]),
        "started_unix": time.time(),
        "deltas": [],
        "stage_events": [],
        "status": "failed",
        "first_output_s": None,
        "output_text": "",
        "output_token_ids": None,
        "output_token_timestamps_s": None,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_content": "",
        "finish_reason": None,
        "io": dict(UNKNOWN_IO),
        "metrics": {},
    }
    terminal = None
    previous_sequence = 0
    iterator = driver.stream(request_id, case["prompt"])
    try:
        async with asyncio.timeout(options.request_timeout_s):
            async for event in iterator:
                elapsed = time.perf_counter() - started
                if event.get("request_id") != request_id:
                    raise RuntimeError("cross-request stream output")
                if terminal is not None:
                    raise RuntimeError("output after terminal event")
                if event["kind"] == "delta":
                    if event["sequence"] != previous_sequence + 1:
                        raise RuntimeError("duplicate or out-of-order stream delta")
                    previous_sequence = event["sequence"]
                    row["deltas"].append(dict(event, elapsed_s=elapsed))
                    row["output_text"] += event["text"]
                    if event["text"] and row["first_output_s"] is None:
                        row["first_output_s"] = elapsed
                elif event["kind"] == "terminal":
                    terminal = event
                    row["full_response_s"] = elapsed
                    if event.get("error") or not event.get("finished"):
                        raise RuntimeError(event.get("error") or "incomplete request")
                    if row["deltas"] and row["output_text"] != event["text"]:
                        raise RuntimeError("terminal text differs from ordered stream deltas")
                    row["output_text"] = event["text"]
                    row["reasoning_content"] = event.get("reasoning_content", "")
                    row["finish_reason"] = event.get("finish_reason")
                    row["output_token_ids"] = event.get("output_token_ids")
                    metrics = event.get("metrics") or {}
                    row["metrics"] = metrics
                    if event.get("stage_event"):
                        stage_event = event["stage_event"]
                        if stage_event.get("request_id") != request_id or not stage_event.get("terminal"):
                            raise RuntimeError("invalid terminal stage identity")
                        row["stage_events"].append(stage_event)
                    backend_events = metrics.get("stage_events", [])
                    if backend_events:
                        expected_epoch = backend_events[0]["epoch"]
                        expected_generation = backend_events[0]["worker_generation"]
                        for index, backend_event in enumerate(backend_events + row["stage_events"], 1):
                            if (
                                backend_event.get("request_id") != request_id
                                or backend_event.get("seq") != index
                                or backend_event.get("epoch") != expected_epoch
                                or backend_event.get("worker_generation") != expected_generation
                            ):
                                raise RuntimeError("invalid backend event order, epoch or generation")
                        row["stage_events"] = backend_events + row["stage_events"]
                    usage = metrics.get("usage") or {}
                    row["input_tokens"] = usage.get("prompt_tokens")
                    row["output_tokens"] = usage.get("completion_tokens")
                    # These must be actual backend counters, never estimated from text length.
                    row["output_token_timestamps_s"] = metrics.get("token_timestamps_s")
                    row["io"] = normalize_io(metrics.get("runtime_telemetry"))
                else:
                    raise RuntimeError(f"unsupported stream event kind: {event['kind']}")
        if terminal is None:
            raise RuntimeError("stream ended without terminal output")
        row["status"] = "completed"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["full_response_s"] = time.perf_counter() - started
        try:
            row["cancel_cleanup"] = await driver.cancel(request_id)
        except Exception as cleanup_error:
            row["cancel_cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
    finally:
        await iterator.aclose()
    row["output_sha256"] = hashlib.sha256(row["output_text"].encode()).hexdigest()
    row["reasoning_sha256"] = hashlib.sha256(row["reasoning_content"].encode()).hexdigest()
    row["quality"] = check_quality(row["output_text"], case.get("quality"), reference, row["output_token_ids"])
    row["delivery_timing_scope"] = "API text chunks; not per-token/kernel latency"
    row["finished_unix"] = time.time()
    return row


async def cancellation_probe(
    driver: Driver, case: dict[str, Any], options: ProfileOptions, admitted_max_tokens: int = 512
) -> dict[str, Any]:
    request_id = "strata-cancel-" + uuid.uuid4().hex
    prompt = dict(case["prompt"], max_tokens=admitted_max_tokens)
    prompt["text"] += "\nCancellation probe: continue listing the integers from 1 through 10000, one per line."
    iterator = driver.stream(request_id, prompt)
    probe: dict[str, Any] = {
        "request_id": request_id,
        "status": "failed",
        "deltas_before_cancel": 0,
        "input": prompt,
        "input_sha256": canonical_hash(prompt),
    }
    terminal = False

    async def consume() -> None:
        nonlocal terminal
        async for event in iterator:
            if event.get("request_id") != request_id:
                raise RuntimeError("cross-request output during cancellation")
            if event["kind"] == "terminal":
                terminal = True
            if event["kind"] == "delta":
                probe["deltas_before_cancel"] += 1
                if probe["deltas_before_cancel"] >= options.cancel_after_deltas:
                    return

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait({task}, timeout=options.cancel_after_seconds)
        if task.done():
            task.result()
        if terminal:
            probe.update(status="inconclusive", reason="request completed before cancellation")
            return probe
        probe["cancel"] = await driver.cancel(request_id)
        if probe["cancel"].get("drained") is not True:
            raise RuntimeError("cancellation did not prove route drain")
        if probe["cancel"].get("ledger") is not None:
            require_drained_ledger(probe["cancel"]["ledger"])
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        probe.update(status="cancelled", reason="route drained; recovery is a separate fresh-route request")
    except Exception as exc:
        probe["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await iterator.aclose()
    return probe


async def run_profile(
    target: dict[str, Any],
    launch: dict[str, Any],
    suite: dict[str, Any],
    out: Path,
    options: ProfileOptions,
    *,
    resume: bool = False,
    references: dict[str, Any] | None = None,
    driver_factory: Callable[[], Driver] | None = None,
) -> dict[str, Any]:
    options.validate()
    validate_target(target)
    validate_suite(suite)
    if driver_factory is None:
        validate_launch(target, launch)
    cases = [case for case in suite["cases"] if case["length_band"] in options.bands]
    if {case["length_band"] for case in cases} != set(options.bands):
        raise ValueError("suite is missing a selected length band")
    identity = {
        "schema": SCHEMA,
        "target": target,
        "launch": launch,
        "suite": suite,
        "options": asdict(options),
        "references": references,
    }
    identity_sha256 = canonical_hash(identity)
    if out.exists():
        if not resume:
            raise FileExistsError("output directory already exists; use --resume for the same immutable run")
        old = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        if old["identity_sha256"] != identity_sha256:
            raise ValueError("resume identity changed: artifact/runtime/config/task/protocol mismatch")
    else:
        out.mkdir(parents=True)
        save_json(out / "run_manifest.json", {"identity_sha256": identity_sha256, **identity})
    records = read_records(out)
    if any(row.get("identity_sha256") != identity_sha256 for row in records):
        raise ValueError("raw requests belong to another immutable experiment identity")
    successful = {row["sample_key"] for row in records if row["status"] == "completed"}
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "identity_sha256": identity_sha256,
        "status": "running",
        "platform": platform.platform(),
        "python": sys.executable,
        "started_unix": time.time(),
        "profile_protocol": "single_request_batch1_v1",
        "batch_size": 1,
        "concurrency": 1,
        "cache_condition": launch.get("cache_condition", "existing caches; cold disk not established"),
        "power_condition": launch.get("power_condition", "unknown; must be recorded for qualification"),
        "startups": [],
        "io": dict(UNKNOWN_IO),
    }
    report["route_controls"] = {
        key: launch.get("backend", {}).get(key)
        for key in (
            "expert_ram_budget_bytes",
            "context_tokens",
            "gpu_budget_bytes",
            "spec_tokens",
            "ple_prefetch",
            "routing_prefetch",
            "io_prefetch",
            "io_mode",
            "ple_io",
            "kv_type",
        )
    }
    old_report_path = out / "report.json"
    if resume and old_report_path.exists():
        previous_report = json.loads(old_report_path.read_text(encoding="utf-8"))
        previous_ledger = previous_report.get("shared_ledger_final")
        if previous_ledger is not None:
            require_drained_ledger(previous_ledger)
        elif previous_report.get("status") == "running" and any(
            startup.get("metadata", {}).get("ledger", {}).get("owners")
            for startup in previous_report.get("startups", [])
        ):
            raise RuntimeError("interrupted run has no proven route drain; verify release before resuming")
        report["startups"] = previous_report.get("startups", [])
    save_json(old_report_path, report)
    telemetry = Telemetry(out / "telemetry.jsonl", options.telemetry_interval_s)
    telemetry.start()
    shared_ledger = None
    if driver_factory is None:
        from vllm_omni.engine.resource_ledger import ResourceLedger

        shared_ledger = ResourceLedger(launch["resource_budget"]["capacities"])
    factory = driver_factory or (lambda: OmniStageDriver(launch, resource_ledger=shared_ledger))
    driver: Driver | None = None
    run_instance_id = uuid.uuid4().hex

    async def start_driver(reason: str) -> Driver:
        if shared_ledger is not None:
            require_drained_ledger(shared_ledger.snapshot())
        value = factory()
        began = time.perf_counter()
        startup = {
            "reason": reason,
            "started_unix": time.time(),
            "cache_condition": "process cold start; filesystem cache state is not inferred",
        }
        try:
            startup["metadata"] = await value.start()
            startup["status"] = "completed"
        except Exception as exc:
            startup.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            try:
                startup["cleanup"] = await value.close()
            except Exception as cleanup_error:
                startup["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            raise
        finally:
            startup["wall_s"] = time.perf_counter() - began
            report["startups"].append(startup)
            save_json(old_report_path, report)
        return value

    async def one(case: dict[str, Any], phase: str, index: int) -> None:
        key = f"{phase}-{case['id']}-{index:06d}"
        if key in successful:
            return
        row = await collect_request(driver, case, phase, index, options, (references or {}).get(case["id"]))
        row["identity_sha256"] = identity_sha256
        row["run_instance_id"] = run_instance_id
        if phase == "sustained":
            row["sustained_segment_id"] = run_instance_id
        save_json(out / "requests" / f"{key}-{row['request_id']}.json", row)
        records.append(row)
        report["summary"] = summarize(records, options, suite)
        save_json(old_report_path, report)
        if row["status"] != "completed":
            raise RuntimeError(row.get("error", "request failed"))
        successful.add(key)
        print(
            json.dumps(
                {"sample_key": key, "wall_s": row["full_response_s"], "quality_passed": row["quality"]["passed"]}
            ),
            flush=True,
        )

    try:
        driver = await start_driver("resume" if resume else "initial")
        for case in cases:
            # A resumed process has new caches/JIT state. Do not reuse old warmup records.
            warmup_base = max(
                (
                    int(row["sample_key"].rsplit("-", 1)[1]) + 1
                    for row in records
                    if row["phase"] == "warmup" and row["case_id"] == case["id"]
                ),
                default=0,
            )
            for index in range(options.warmups):
                await one(case, "warmup", warmup_base + index)
            for index in range(options.repeats):
                await one(case, "measured", index)
        medium = next((case for case in cases if case["length_band"] == "medium"), cases[0])
        previous_segment_s = summarize(records, options, suite)["longest_uninterrupted_sustained_active_s"]
        sustained_s = options.sustained_seconds if previous_segment_s >= options.sustained_seconds else 0
        index = sum(row["phase"] == "sustained" and row["status"] == "completed" for row in records)
        while sustained_s < options.sustained_seconds:
            await one(medium, "sustained", index)
            sustained_s = sum(
                row["full_response_s"]
                for row in records
                if row["phase"] == "sustained"
                and row["status"] == "completed"
                and row.get("sustained_segment_id") == run_instance_id
            )
            index += 1
        report["usage_before_cancel"] = driver.usage()
        report["cancel"] = await cancellation_probe(
            driver, cases[0], options, launch.get("backend", {}).get("max_new_tokens", 512)
        )
        report["cancel_route_close"] = await driver.close()
        driver = None
        save_json(old_report_path, report)
        if report["cancel"]["status"] == "failed":
            raise RuntimeError("cancellation failed; recovery is refused")
        driver = await start_driver("after_cancel_recovery")
        await one(cases[0], "after_cancel", len(report["startups"]))
        report["usage"] = driver.usage()
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if driver is not None:
            try:
                report["close"] = await driver.close()
            except Exception as exc:
                report.update(status="failed", close_error=f"{type(exc).__name__}: {exc}")
        report["telemetry"] = telemetry.stop()
        report["summary"] = summarize(records, options, suite)
        if shared_ledger is not None:
            report["shared_ledger_final"] = shared_ledger.snapshot()
        report["finished_unix"] = time.time()
        save_json(old_report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-target", help="hash every upstream model and auxiliary file")
    verify.add_argument("--target", type=Path, required=True)
    verify.add_argument("--model-dir", type=Path, required=True)
    verify.add_argument("--out", type=Path)
    run = commands.add_parser("run", help="execute through Omni; short runs remain instrumentation only")
    run.add_argument("--target", type=Path, required=True)
    run.add_argument("--launch", type=Path, required=True)
    run.add_argument("--suite", type=Path)
    run.add_argument("--reference", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--repeats", type=int, default=20)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--sustained-seconds", type=float, default=1800)
    run.add_argument("--request-timeout-s", type=float, default=600)
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--length-band", choices=("all", *BANDS), default="all")
    run.add_argument("--telemetry-interval-s", type=float, default=0.5)
    run.add_argument("--cancel-after-deltas", type=int, default=8)
    run.add_argument("--cancel-after-seconds", type=float, default=0.25)
    analyze = commands.add_parser("summarize", help="recompute nearest-rank summaries from durable raw requests")
    analyze.add_argument("run_dir", type=Path)
    variants = commands.add_parser("cache-variants", help="create explicit 24/32/40 GiB cache launch variants")
    variants.add_argument("--launch", type=Path, required=True)
    variants.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cache-variants":
        if args.out.exists():
            raise FileExistsError("variant output directory must be new")
        values = cache_variant_launches(json.loads(args.launch.read_text(encoding="utf-8")))
        args.out.mkdir(parents=True)
        for name, value in values.items():
            save_json(args.out / (name + ".json"), value)
        result = {
            "status": "completed",
            "variants": list(values),
            "note": "capacities unchanged; each variant still needs admission and actual measurement",
        }
    elif args.command == "summarize":
        manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        options = ProfileOptions(**manifest["options"])
        result = summarize(read_records(args.run_dir), options, manifest["suite"])
    else:
        target = json.loads(args.target.read_text(encoding="utf-8"))
        if args.command == "verify-target":
            result = verify_target_files(target, args.model_dir)
            if args.out:
                save_json(args.out, result)
        else:
            launch = json.loads(args.launch.read_text(encoding="utf-8"))
            suite = json.loads(args.suite.read_text(encoding="utf-8")) if args.suite else default_suite()
            references = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
            options = ProfileOptions(
                repeats=args.repeats,
                warmups=args.warmups,
                sustained_seconds=args.sustained_seconds,
                request_timeout_s=args.request_timeout_s,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                bands=BANDS if args.length_band == "all" else (args.length_band,),
                telemetry_interval_s=args.telemetry_interval_s,
                cancel_after_deltas=args.cancel_after_deltas,
                cancel_after_seconds=args.cancel_after_seconds,
            )
            result = asyncio.run(
                run_profile(target, launch, suite, args.out, options, resume=args.resume, references=references)
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status", "completed") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
