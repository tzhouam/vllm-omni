# SPDX-License-Identifier: Apache-2.0
"""Configuration/qualification tests; fixture manifests are not model evidence."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from vllm_omni.edge.agent.omni_backend import OmniStrataBackend, OmniStrataConfig
from vllm_omni.edge.agent.router import Admission, Route, select_route
from vllm_omni.edge.agent.strata_route import (
    BACKEND,
    RUNTIME_REVISION,
    agent_config_from_launch,
    agent_entry_from_launch,
)
from vllm_omni.engine.resource_ledger import ResourceLedger
from vllm_omni.engine.weight_tiers import ArtifactFile, ArtifactManifest, WeightTierBudget, WeightTierPlan


@pytest.fixture
def launch():
    source = ArtifactManifest(
        "fixture/Qwen",
        "a" * 40,
        "MIT",
        (
            ArtifactFile("weights.gguf", 300, "b" * 64),
            ArtifactFile("ple.gguf", 300, "c" * 64),
        ),
    )
    runtime = ArtifactManifest(
        "fixture/Strata", RUNTIME_REVISION, "MIT", (ArtifactFile("engine/strata.exe", 100, "d" * 64, role="runtime"),)
    )
    prepared = ArtifactManifest(
        "fixture/pack",
        source.revision + "+pack",
        "MIT",
        (ArtifactFile("dense.bin", 200, "e" * 64, role="prepared_pack"),),
    )
    budget = WeightTierBudget(
        cpu_expert_cache_bytes=1024,
        host_workspace_bytes=128,
        host_transfer_bytes=64,
        windows_commit_peak_bytes=1280,
        gpu_weights_bytes=256,
        gpu_expert_cache_bytes=128,
        gpu_workspace_bytes=64,
        ssd_artifact_bytes=800,
        ssd_temporary_bytes=64,
    )
    plan = WeightTierPlan(
        "fixture-q4",
        source.manifest_sha256,
        BACKEND,
        RUNTIME_REVISION,
        budget,
        ssd_experts=True,
        lookup_tables_on_demand=True,
    )
    backend = {
        "name": BACKEND,
        "runtime_revision": RUNTIME_REVISION,
        "route_id": plan.route_id,
        "gpu_index": 0,
        "gpu_pool": "vram:0",
        "gpu_total_bytes": 2048,
        "gpu_budget_bytes": 448,
        "artifact_manifest": source.to_dict(),
        "runtime_manifest": runtime.to_dict(),
        "prepared_pack_manifest": prepared.to_dict(),
        "weight_tier_plan": plan.to_dict(),
        "conversion_manifest": {
            "complete": True,
            "source_manifest_sha256": source.manifest_sha256,
            "prepared_manifest_sha256": prepared.manifest_sha256,
            "tool_revision": RUNTIME_REVISION,
            "conversions": [],
        },
        "runtime_root": "C:/fixture/runtime",
        "artifact_root": "C:/fixture/models",
        "prepared_model_dir": "C:/fixture/pack",
        "python_bin": "C:/fixture/python.exe",
        "python_sha256": "f" * 64,
        "python_environment": {"fixture": True},
        "native_file": "weights.gguf",
        "ple_file": "ple.gguf",
        "expert_ram_budget_bytes": 1024,
        "host_overhead_bytes": 128,
        "context_tokens": 4096,
        "max_new_tokens": 128,
        "max_io_bytes": 64,
        "request_timeout_s": 600,
        "start_timeout_s": 900,
        "spec_tokens": 0,
        "ple_prefetch": False,
        "routing_prefetch": False,
        "io_prefetch": False,
    }
    return {
        "schema": "omni-strata-launch-v1",
        "backend": backend,
        "resource_budget": {
            "demands": budget.resource_demands(include_windows_commit=True),
            "capacities": {"host_ram": 4096, "vram:0": 2048, "windows_commit": 4096, "ssd": 8192},
        },
    }


def _entry(launch):
    return agent_entry_from_launch(launch, expected_device_name="fixture NVIDIA GPU 0")


def test_converter_preserves_identity_and_exact_bytes_without_mutating_launch(launch):
    original = copy.deepcopy(launch)
    entry = _entry(launch)
    assert launch == original
    assert entry["backend_config"]["gpu_pool"] == "vram"
    assert entry["memory_demands"] == {"host_ram": 1216, "vram": 448, "windows_commit": 1280, "ssd": 864}
    for key in (
        "artifact_manifest",
        "runtime_manifest",
        "prepared_pack_manifest",
        "conversion_manifest",
        "weight_tier_plan",
        "python_sha256",
        "python_environment",
    ):
        assert entry["backend_config"][key] == launch["backend"][key]
    assert entry["model"] == "fixture/Qwen" and entry["modalities"] == ["text"]
    assert entry["requires_nvidia"] is True
    assert "unqualified" in entry["experimental_status"]


def test_converted_adapter_accepts_same_engine_lease_and_still_requires_placement(launch):
    entry = _entry(launch)
    capacities = {"host_ram": 4096, "vram": 2048, "windows_commit": 4096, "ssd": 8192}
    config = OmniStrataConfig(
        route_id=entry["route_id"],
        backend_config=entry["backend_config"],
        placement=entry["placement"],
        capacities=capacities,
        demands=entry["memory_demands"],
        context_tokens=entry["context_tokens"],
        max_new_tokens=entry["max_new_tokens"],
        max_io_bytes=entry["max_io_bytes"],
    )
    adapter = OmniStrataBackend(config)
    ledger = ResourceLedger(capacities)
    lease = ledger.reserve(config.route_id, config.demands)
    adapter.bind_resource_lease(ledger, lease)
    assert adapter._shared_reservation is lease and adapter._shared_ledger is ledger
    assert adapter._stage_backend_config()["gpu_pool"] == "vram"
    with pytest.raises(RuntimeError, match="verify its loaded backend execution configuration"):
        adapter._validate_loaded_plan({"requested_device": entry["placement"], "observed_model_placement": None})


def test_translation_never_qualifies_route_or_implicitly_enables_bootstrap(launch):
    config = agent_config_from_launch(launch, expected_device_name="fixture GPU")
    assert "experimental_bootstrap_route_id" not in config
    assert config["qualification_bundles"] == [] and config["trusted_review_keys"] == {}
    assert config["limits"]["max_answer_tokens"] == 128
    entry = config["routes"][0]
    route = Route(
        entry["route_id"],
        entry["artifact_id"],
        entry["model"],
        entry["backend"],
        frozenset(entry["modalities"]),
        entry["placement"],
        entry["memory_demands"],
        True,
    )
    options = {
        "suite_id": "test",
        "environment_fingerprint": "fixture",
        "power_condition": "AC",
        "admit": lambda _route: Admission(True, "provisional; stage must prove placement"),
    }
    assert select_route("basic", [route], [], **options).route is None
    explicit = agent_config_from_launch(launch, expected_device_name="fixture GPU", experimental_bootstrap=True)
    decision = select_route(
        "basic", [route], [], **options, bootstrap_route_id=explicit["experimental_bootstrap_route_id"]
    )
    assert decision.route is route and decision.experimental and decision.qualification is None
    assert select_route("browser_vision", [route], [], **options, bootstrap_route_id=route.route_id).route is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["backend"].update(gpu_index=1),
        lambda data: data["backend"].update(gpu_pool="vram:1"),
        lambda data: data["backend"].update(runtime_revision="0" * 40),
        lambda data: data["backend"]["conversion_manifest"].update(prepared_manifest_sha256="0" * 64),
        lambda data: data["backend"].update(native_file="unbound.gguf"),
        lambda data: data["resource_budget"]["demands"].update(host_ram=1215),
        lambda data: data["resource_budget"]["capacities"].update(host_ram=1),
        lambda data: data["resource_budget"]["demands"].pop("windows_commit"),
        lambda data: data["resource_budget"]["demands"].update(wsl_ram=1216),
        lambda data: data["backend"].update(spec_tokens=4, mtp_directory="draft"),
        lambda data: data["backend"].update(args=["--mcp"]),
        lambda data: data["backend"].update(request_timeout_s=float("nan")),
    ],
)
def test_converter_refuses_changed_boundaries(launch, mutation):
    mutation(launch)
    with pytest.raises(ValueError):
        _entry(launch)


@pytest.mark.parametrize("maximum,expected", [(16, 16), (512, 512), (1024, 512)])
def test_agent_answer_limit_respects_the_prepared_stage_bound(launch, maximum, expected):
    launch["backend"]["max_new_tokens"] = maximum
    config = agent_config_from_launch(launch, expected_device_name="fixture GPU", experimental_bootstrap=True)
    assert config["limits"]["max_answer_tokens"] == expected
    assert config["limits"]["max_answer_tokens"] <= config["routes"][0]["max_new_tokens"]
    assert config["experimental_bootstrap_route_id"] == config["routes"][0]["route_id"]
    assert config["qualification_bundles"] == []


def test_budget_change_creates_different_artifact_route_identity(launch):
    before = _entry(launch)
    backend = launch["backend"]
    plan = WeightTierPlan.from_dict(backend["weight_tier_plan"])
    plan = replace(plan, budget=replace(plan.budget, cpu_expert_cache_bytes=1536, windows_commit_peak_bytes=1792))
    backend["weight_tier_plan"] = plan.to_dict()
    backend["expert_ram_budget_bytes"] = 1536
    launch["resource_budget"]["demands"] = plan.budget.resource_demands(include_windows_commit=True)
    after = _entry(launch)
    assert before["artifact_id"] != after["artifact_id"]
    assert before["backend_config"]["artifact_manifest"] == after["backend_config"]["artifact_manifest"]
