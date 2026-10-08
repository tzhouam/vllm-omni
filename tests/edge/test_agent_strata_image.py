"""Protocol-only checks. Fixtures do not establish an actual image run."""

from __future__ import annotations

import ast
import asyncio
import base64
import copy
import importlib.util
import io
import json
import os
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.edge.agent.omni_backend import (
    OmniStrataConfig,
    OmniStrataImageBackend,
    OmniStrataImageConfig,
)
from vllm_omni.edge.agent.placement import preparation_placement_matches, result_placement_matches
from vllm_omni.edge.agent.strata_image_evidence import (
    IMAGE_BACKEND,
    digest,
    image_input_identity,
    validate_image_load_plan_for_config,
    validate_image_terminal,
)
from vllm_omni.edge.agent.strata_route import agent_config_from_launch, agent_entry_from_launch
from vllm_omni.engine.resource_ledger import ResourceLedger
from vllm_omni.engine.weight_tiers import ArtifactFile, ArtifactManifest, WeightTierPlan

REPO = Path(os.environ.get("OMNI_AGENT_TEST_REPO", str(Path(__file__).resolve().parents[2])))
PATCH_ROOT = Path(os.environ.get("OMNI_AGENT_IMAGE_PATCH_ROOT", str(REPO)))
VISION_ROOT = Path(os.environ.get("OMNI_STRATA_VISION_PROTOTYPE_ROOT", str(REPO)))
vision_fixture = sys.modules.get("vision_fixtures")
if vision_fixture is None:
    _spec = importlib.util.spec_from_file_location(
        "vision_fixtures", VISION_ROOT / "tests/edge/test_strata_vision_prototype.py"
    )
    vision_fixture = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = vision_fixture
    _spec.loader.exec_module(vision_fixture)


def load_test(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "tests/edge" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


text_tests = load_test("original_text_route_tests", "test_agent_strata_route.py")


@pytest.fixture
def launch():
    value = text_tests.launch.__wrapped__()
    backend = value["backend"]
    runtime = ArtifactManifest.from_dict(backend["runtime_manifest"])
    backend["runtime_manifest"] = replace(
        runtime,
        files=runtime.files
        + (
            ArtifactFile("engine/strata-vision.exe", 100, "1" * 64, role="runtime"),
            ArtifactFile("vision-build/build.json", 200, "2" * 64, role="runtime"),
        ),
    ).to_dict()
    projector = ArtifactManifest(
        "fixture/projector", "a" * 40, "MIT", (ArtifactFile("mmproj.gguf", 100, "3" * 64, role="vision_projector"),)
    )
    backend.update(
        name=IMAGE_BACKEND,
        image_route={
            "schema": "omni-strata-image-route-v1",
            "projector_root": "C:/fixture/projector",
            "projector_manifest": projector.to_dict(),
            "projector_file": "mmproj.gguf",
            "text_artifact_manifest_sha256": ArtifactManifest.from_dict(backend["artifact_manifest"]).manifest_sha256,
            "encoder_file": "engine/strata-vision.exe",
            "encoder_build_manifest_file": "vision-build/build.json",
            "embedding_width": 2,
            "max_image_bytes": 256,
            "max_image_pixels": 16,
            "max_image_tokens": 4,
            "encoder_threads": 2,
            "encoder_device": "cpu",
            "allow_cpu_fallback": False,
            "vision_host_bytes": 1124,
            "vision_gpu_bytes": 0,
            "vision_scratch_bytes": 360,
        },
    )
    tier = WeightTierPlan.from_dict(backend["weight_tier_plan"])
    budget = replace(
        tier.budget,
        host_workspace_bytes=2000,
        windows_commit_peak_bytes=4096,
        ssd_artifact_bytes=1400,
        ssd_temporary_bytes=512,
    )
    tier = replace(tier, backend=IMAGE_BACKEND, budget=budget)
    backend["weight_tier_plan"] = tier.to_dict()
    backend["host_overhead_bytes"] = 2000
    value["resource_budget"] = {
        "capacities": {"host_ram": 8192, "vram:0": 2048, "windows_commit": 8192, "ssd": 8192},
        "demands": budget.resource_demands(include_windows_commit=True),
    }
    return value


def entry(launch):
    return agent_entry_from_launch(launch, expected_device_name="fixture NVIDIA GPU 0")


def config(launch):
    route = entry(launch)
    return OmniStrataImageConfig(
        route_id=route["route_id"],
        backend_config=route["backend_config"],
        placement=route["placement"],
        capacities={"host_ram": 8192, "vram": 2048, "windows_commit": 8192, "ssd": 8192},
        demands=route["memory_demands"],
        context_tokens=route["context_tokens"],
        max_new_tokens=route["max_new_tokens"],
        max_io_bytes=route["max_io_bytes"],
        mmproj_file=route["mmproj_file"],
        image_capability=route["image_capability"],
    )


def png(width=2, height=2):
    stream = io.BytesIO()
    Image.new("RGB", (width, height), "blue").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def proof_fixture(launch):
    cfg = config(launch)
    capability = cfg.image_capability
    identity = image_input_identity(png(), capability)
    selection = {
        "schema": "strata-vision-backend-v1",
        "primary_backend": "CPU",
        "device_type": "CPU",
        "gpu_requested": False,
        "cpu_fallback_available": True,
    }
    bounds = {
        key: cfg.backend_config["image_route"][key]
        for key in (
            "embedding_width",
            "max_image_bytes",
            "max_image_pixels",
            "max_image_tokens",
            "encoder_threads",
            "vision_host_bytes",
            "vision_gpu_bytes",
            "vision_scratch_bytes",
        )
    }
    plan = {
        "image_route": {"identity_sha256": "4" * 64, "bounds_and_declared_budgets": bounds},
        "owned_encoder_at_load": {"pid": 11, "creation_filetime_100ns": 111},
        "encoder_backend_selection_at_load": selection,
        "selected_native_module_audit_at_load": {"pid": 22, "creation_filetime_100ns": 222},
        "observation_runtime": {"identity_sha256": "5" * 64},
        "stage_id": 0,
        "worker_generation": "generation-a",
    }
    stage = {
        "terminal": True,
        "kind": "text",
        "request_id": "task-step-0",
        "stage_id": 0,
        "worker_generation": "generation-a",
        "epoch": 1,
        "seq": 1,
    }
    io_observation = {
        "schema": "omni-strata-request-io-observation-v1",
        "status": "complete",
        "request_id": "task-step-0",
        "epoch": 1,
        "generation": "generation-a",
        "native_pid": 22,
        "creation_filetime_100ns": 222,
        "native_request_seq": 1,
        "runtime_identity_sha256": "5" * 64,
        "scope": "native_FileExpertSource_and_PLE_counters_excludes_loading",
        "physical_ssd_read_bytes": None,
        "loading_covered": False,
        "three_tier_memory_qualified": False,
    }
    report = {
        "schema": "omni-strata-image-request-observation-v1",
        "status": "complete",
        "generation": "generation-a",
        "request_id": "task-step-0",
        "epoch": 1,
        "owned_encoder": copy.deepcopy(plan["owned_encoder_at_load"]),
        "native_encoder_request_seq": 1,
        "backend_selection": copy.deepcopy(selection),
        "input_sha256": identity["sha256"],
        "encode_result": {
            "image_tokens": 4,
            "nx": 2,
            "ny": 2,
            "sve_bytes": 52,
            "sve_sha256": "6" * 64,
            "native_encoder_ms": 2.0,
        },
        "language_dispatch": {
            "command": "GENI",
            "language_pid": 22,
            "language_creation_filetime_100ns": 222,
            "sve_sha256": "6" * 64,
        },
        "scope": "owned_native_encoder_then_pinned_server_SVE_GENI_chain",
        "all_encoder_operators_gpu_verified": False,
        "whole_model_placement": None,
        "physical_ssd_read_bytes": None,
        "release_qualified": False,
        "reasons": [],
        "text_native_io_request_seq": 1,
        "text_native_io_scope": io_observation["scope"],
    }
    metrics = {
        "runtime_telemetry": {
            "image_observation": report,
            "native_io_observation": io_observation,
            "native_io_excludes_encoder": True,
            "encoder_all_operators_placement": None,
        }
    }
    return cfg, plan, metrics, stage, identity


def test_explicit_image_route_preserves_artifacts_budget_and_is_opt_in(launch):
    original = copy.deepcopy(launch)
    route = entry(launch)
    assert launch == original
    assert route["backend"] == IMAGE_BACKEND and route["modalities"] == ["text", "image"]
    assert route["image_capability"]["encoder_device"] == "cpu"
    assert route["backend_config"]["image_route"] == launch["backend"]["image_route"]
    assert "experimental_bootstrap_route_id" not in agent_config_from_launch(launch, expected_device_name="GPU")
    opted = agent_config_from_launch(launch, expected_device_name="GPU", experimental_bootstrap=True)
    assert opted["experimental_bootstrap_route_id"] == route["route_id"] and not opted["qualification_bundles"]
    assert config(launch).mmproj_file == "C:/fixture/projector/mmproj.gguf"


@pytest.mark.parametrize("mutation", ["text_spoof", "projector", "source", "encoder", "cuda", "budget", "context"])
def test_route_refuses_mismatched_image_identity_or_budget(launch, mutation):
    backend = launch["backend"]
    if mutation == "text_spoof":
        backend["name"] = "external.strata.text.v1"
    elif mutation == "projector":
        backend["image_route"]["projector_file"] = "other.gguf"
    elif mutation == "source":
        backend["image_route"]["text_artifact_manifest_sha256"] = "0" * 64
    elif mutation == "encoder":
        backend["image_route"]["encoder_file"] = "outside.exe"
    elif mutation == "cuda":
        backend["image_route"]["encoder_device"] = "cuda"
    elif mutation == "budget":
        backend["image_route"]["vision_host_bytes"] = 1
    else:
        backend["image_route"]["max_image_tokens"] = 4000
    with pytest.raises(ValueError):
        entry(launch)


def test_text_config_and_route_stay_strict_and_unchanged(launch):
    text = text_tests.launch.__wrapped__()
    text_tests.test_converter_preserves_identity_and_exact_bytes_without_mutating_launch(text)
    with pytest.raises(ValueError):
        OmniStrataConfig(
            "x", launch["backend"], "cpu+cuda:0", {"host_ram": 8192, "vram": 2048}, {"host_ram": 1, "vram": 1}
        )
    with pytest.raises(TypeError):
        OmniStrataImageBackend(
            OmniStrataConfig(
                "x",
                {"name": "external.strata.text.v1"},
                "cpu+cuda:0",
                {"host_ram": 2, "vram": 2},
                {"host_ram": 1, "vram": 1},
            )
        )


@pytest.mark.parametrize("mutation", ["pixels", "bytes", "jpeg", "noncanonical"])
def test_png_transport_refuses_unadmitted_inputs(launch, mutation):
    capability = entry(launch)["image_capability"]
    data = png()
    if mutation == "pixels":
        data = png(5, 5)
    elif mutation == "bytes":
        capability["max_image_bytes"] = 10
    elif mutation == "jpeg":
        data = data.replace("image/png", "image/jpeg")
    else:
        data += "\n"
    with pytest.raises((ValueError, TypeError)):
        image_input_identity(data, capability)


def test_terminal_proof_binds_all_roles_input_and_final_output_detached(launch):
    cfg, plan, metrics, stage, identity = proof_fixture(launch)
    seen = set()
    proof = validate_image_terminal(
        plan, metrics, stage, request_id="task-step-0", input_identity=identity, seen_states=seen
    )
    assert proof["loaded_plan_sha256"] == digest(plan) and proof["release_qualified"] is False
    proof["image_observation"]["owned_encoder"]["pid"] = 999
    assert metrics["runtime_telemetry"]["image_observation"]["owned_encoder"]["pid"] == 11
    with pytest.raises(ValueError, match="reuses"):
        validate_image_terminal(
            plan, metrics, stage, request_id="task-step-0", input_identity=identity, seen_states=seen
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "request",
        "epoch",
        "generation",
        "terminal",
        "input",
        "owner",
        "cpu",
        "sve",
        "size",
        "grid",
        "language_owner",
        "native_terminal",
        "native_seq",
        "runtime",
        "scope",
        "qualification",
    ],
)
def test_terminal_refuses_cross_request_incomplete_or_fabricated_compute(launch, mutation):
    _cfg, plan, metrics, stage, identity = proof_fixture(launch)
    telemetry = metrics["runtime_telemetry"]
    report = telemetry["image_observation"]
    native = telemetry["native_io_observation"]
    if mutation == "request":
        report["request_id"] = "other"
    elif mutation == "epoch":
        report["epoch"] = 2
    elif mutation == "generation":
        report["generation"] = "other"
    elif mutation == "terminal":
        stage["terminal"] = False
    elif mutation == "input":
        report["input_sha256"] = "f" * 64
    elif mutation == "owner":
        report["owned_encoder"]["pid"] = 999
    elif mutation == "cpu":
        report["backend_selection"]["device_type"] = "GPU"
    elif mutation == "sve":
        report["language_dispatch"]["sve_sha256"] = "f" * 64
    elif mutation == "size":
        report["encode_result"]["sve_bytes"] = 1
    elif mutation == "grid":
        report["encode_result"]["nx"] = 1
    elif mutation == "language_owner":
        report["language_dispatch"]["language_pid"] = 999
    elif mutation == "native_terminal":
        native["status"] = "incomplete"
    elif mutation == "native_seq":
        report["text_native_io_request_seq"] = 2
    elif mutation == "runtime":
        native["runtime_identity_sha256"] = "f" * 64
    elif mutation == "scope":
        report["physical_ssd_read_bytes"] = 123
    else:
        report["release_qualified"] = True
    seen = set()
    with pytest.raises(ValueError):
        validate_image_terminal(
            plan, metrics, stage, request_id="task-step-0", input_identity=identity, seen_states=seen
        )
    assert not seen


def test_text_followup_cannot_borrow_image_receipt(launch):
    _cfg, plan, metrics, stage, _identity = proof_fixture(launch)
    with pytest.raises(ValueError, match="text-only"):
        validate_image_terminal(plan, metrics, stage, request_id="task-step-0", input_identity=None, seen_states=set())
    metrics["runtime_telemetry"]["image_observation"] = None
    assert (
        validate_image_terminal(plan, metrics, stage, request_id="task-step-0", input_identity=None, seen_states=set())
        is None
    )


def test_image_backend_uses_actual_image_gate_and_checks_exact_lease(launch, monkeypatch):
    checked = []
    import vllm_omni.edge.agent.strata_image_evidence as helper

    monkeypatch.setattr(helper, "validate_image_load_plan_for_config", lambda *args: checked.append(args))
    cfg = config(launch)
    backend = OmniStrataImageBackend(cfg)
    plan = {"reserved_bytes": dict(cfg.demands)}
    backend._validate_loaded_plan(plan)
    assert checked == [(plan, cfg.backend_config, "cpu+cuda:0")]
    plan["reserved_bytes"]["host_ram"] -= 1
    with pytest.raises(RuntimeError, match="exact shared"):
        backend._validate_loaded_plan(plan)
    assert backend._stage_backend_config()["name"] == IMAGE_BACKEND


def test_image_cancel_retains_same_quarantined_lease_and_blocks_replacement(launch):
    cfg = config(launch)
    backend = OmniStrataImageBackend(cfg)
    ledger = ResourceLedger(cfg.capacities)
    claim = ledger.reserve("owned-image", cfg.demands)
    backend.bind_resource_lease(ledger, claim)
    runtime = SimpleNamespace(
        resource_ledger=ledger,
        _resource_reservations={(0, 0): claim},
        shutdown=lambda: ledger.release(claim, drained=False),
    )
    backend._runtime = runtime
    backend._pool = SimpleNamespace(stage_client=SimpleNamespace(_proc=SimpleNamespace(pid=123, poll=lambda: 0)))
    assert backend.close() is False and ledger.owns(claim)
    assert "owned-image" in ledger.snapshot()["quarantined"]
    with pytest.raises(RuntimeError, match="reserved or quarantined"):
        backend.start()
    assert ledger.owns(claim)


def test_image_never_inherits_generic_nullable_placement_qualification():
    route = {"backend": IMAGE_BACKEND, "artifact_id": "a", "model_id": "m", "expected_placement": None}
    result = {"backend": IMAGE_BACKEND, "artifact_id": "a", "model_id": "m", "actual_placement": None}
    assert result_placement_matches(result, route) is False
    assert (
        preparation_placement_matches(
            {"cold_start_confirmed": True, "artifact_id": "a", "actual_placement": None}, route
        )
        is False
    )


def test_ui_names_cpu_encoder_separately_without_whole_compute_claim():
    from vllm_omni.edge.agent.desktop import _placement_text

    payload = {
        "backend": IMAGE_BACKEND,
        "actual_placement": None,
        "verified_execution_configuration": "cpu+cuda:0",
        "execution_configuration_evidence": {"status": "verified"},
        "image_configuration": {"encoder_device": "cpu", "release_qualified": False},
        "encoder_backend_selection": {"primary_backend": "CPU", "device_type": "CPU", "gpu_requested": False},
    }
    text = _placement_text(payload)
    assert "image encoder selected CPU" in text and "language loaded CPU + GPU 0" in text
    assert "Complete compute placement is not verified" in text


def test_real_v3_image_validator_then_exact_selected_configuration_binding(tmp_path, monkeypatch):
    # Small synthetic files, actual v3 validation functions. No model/device run.
    route = vision_fixture.route.__wrapped__(tmp_path, monkeypatch)
    plan = vision_fixture.loaded_image_plan.__wrapped__(route, monkeypatch)
    image_cfg, _runtime, runtime_manifest, _files, source = route
    prepared = ArtifactManifest(
        "fixture/pack", "test", "test", (ArtifactFile("dense.bin", 100, "e" * 64, role="prepared_pack"),)
    )
    cfg = {
        "name": IMAGE_BACKEND,
        "route_id": "test-image",
        "image_route": image_cfg,
        "artifact_manifest": source.to_dict(),
        "runtime_manifest": runtime_manifest.to_dict(),
        "prepared_pack_manifest": prepared.to_dict(),
        "weight_tier_plan": plan["weight_tier_plan"],
        "context_tokens": 4096,
        "max_new_tokens": 64,
        "max_io_bytes": 1,
        "host_overhead_bytes": plan["host_overhead_bytes"],
        "gpu_budget_bytes": 1024,
        "gpu_total_bytes": 2048,
        "gpu_pool": "vram:0",
        "expert_ram_budget_bytes": 0,
        "conversion_manifest": {"fixture": "bound conversion"},
    }
    for key in (
        "context_tokens",
        "max_new_tokens",
        "max_io_bytes",
        "host_overhead_bytes",
        "gpu_budget_bytes",
        "gpu_total_bytes",
        "gpu_pool",
        "expert_ram_budget_bytes",
        "conversion_manifest",
    ):
        plan[key] = cfg[key]
    plan["prepared_manifest_sha256"] = prepared.manifest_sha256
    validate_image_load_plan_for_config(plan, cfg, "cpu+cuda:0")
    # A self-consistent loaded plan for another configuration is still refused.
    altered = copy.deepcopy(cfg)
    altered["context_tokens"] = 8192
    with pytest.raises(RuntimeError, match="selected Agent"):
        validate_image_load_plan_for_config(plan, altered, "cpu+cuda:0")
    altered = copy.deepcopy(cfg)
    altered["image_route"]["projector_manifest"]["files"][0]["sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="selected Agent"):
        validate_image_load_plan_for_config(plan, altered, "cpu+cuda:0")


def test_stream_does_not_yield_terminal_before_image_proof_validates(launch):
    cfg, plan, metrics, stage, _identity = proof_fixture(launch)
    backend = OmniStrataImageBackend(cfg)
    backend.execution_plan = plan
    emitted = iter([("answer", 1.0), None])
    released = []
    aborted = []

    async def receive(_request):
        return next(emitted)

    async def submit(*args):
        return 0

    async def abort(requests):
        aborted.extend(requests)

    output = SimpleNamespace(
        request_id="task-step-0",
        error=None,
        outputs=[SimpleNamespace(text="answer")],
        custom_output={"stage_event": stage},
        metrics=metrics,
        release_stage_buffers=lambda: released.append(True),
    )
    backend._pool = SimpleNamespace(
        stage_client=SimpleNamespace(receive_agent_delta=receive),
        submit_initial=submit,
        poll_graph_output=lambda _: output,
        abort_requests=abort,
    )
    backend.close = lambda: True

    async def collect():
        return [
            chunk
            async for chunk in backend.generate("prompt", request_id="task-step-0", max_tokens=8, image_data_url=png())
        ]

    chunks = asyncio.run(collect())
    assert chunks[-1].terminal and chunks[-1].metrics["image_chain_evidence"]["request_id"] == "task-step-0"
    assert released == [True] and not aborted


def test_stream_invalid_image_terminal_aborts_before_tool_eligible_completion(launch):
    cfg, plan, metrics, stage, _identity = proof_fixture(launch)
    metrics["runtime_telemetry"]["image_observation"]["status"] = "incomplete"
    backend = OmniStrataImageBackend(cfg)
    backend.execution_plan = plan
    emitted = iter([("answer", 1.0), None])
    aborted = []
    terminal = []

    async def receive(_request):
        return next(emitted)

    async def submit(*args):
        return 0

    async def abort(requests):
        aborted.extend(requests)

    output = SimpleNamespace(
        request_id="task-step-0",
        error=None,
        outputs=[SimpleNamespace(text="answer")],
        custom_output={"stage_event": stage},
        metrics=metrics,
        release_stage_buffers=lambda: None,
    )
    backend._pool = SimpleNamespace(
        stage_client=SimpleNamespace(receive_agent_delta=receive),
        submit_initial=submit,
        poll_graph_output=lambda _: output,
        abort_requests=abort,
    )
    backend.close = lambda: True

    async def collect():
        async for chunk in backend.generate("prompt", request_id="task-step-0", max_tokens=8, image_data_url=png()):
            terminal.append(chunk.terminal)

    with pytest.raises(ValueError, match="incomplete"):
        asyncio.run(collect())
    assert terminal == [False] and aborted == ["task-step-0"]


def test_controller_dispatches_distinct_image_gate_and_keeps_actual_placement_unknown(tmp_path, launch, monkeypatch):
    tests = load_test("original_controller_tests", "test_agent_strata_controller.py")
    backend = tests.LeaseBackend(tests._plan())
    backend.config = config(launch)
    backend.execution_plan.update(
        image_route={"identity_sha256": "a" * 64, "encoder_device": "cpu"},
        encoder_backend_selection_at_load={"primary_backend": "CPU", "device_type": "CPU"},
    )
    controller = tests._controller(tmp_path, backend, backend_name=IMAGE_BACKEND)
    controller.routes = [replace(controller.routes[0], modalities=frozenset({"text", "image"}))]
    checked = []
    import vllm_omni.engine.backends.strata_multimodal as image_stage

    monkeypatch.setattr(image_stage, "validate_strata_multimodal_load_plan", lambda *args: checked.append(args))
    events = []
    controller.add_listener(events.append)
    try:
        assert controller.submit("Say ready").result(timeout=20) == "ready"
        route_event = next(event["payload"] for event in events if event["kind"] == "route")
        assert checked == [(backend.execution_plan, "cpu+cuda:0")]
        assert route_event["actual_placement"] is None and route_event["experimental"] is True
        assert route_event["encoder_backend_selection"]["device_type"] == "CPU"
    finally:
        controller.close()


def test_formal_profile_refuses_image_backend_before_llama_or_nullable_placement_fallback():
    # Execute only the actual small functions, excluding profiler imports/sensors.
    path = PATCH_ROOT / "benchmarks/edge_agent/native_profile.py"
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"load_profile_routes", "_trace_complete"}
    ]
    module = types.ModuleType("private_image_profile_guard")
    module.__dict__.update(STRATA_BACKEND="external.strata.text.v1")
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), module.__dict__)
    with pytest.raises(ValueError, match="dedicated reviewed image"):
        module.load_profile_routes(
            {"routes": [{"route_id": "image", "backend": IMAGE_BACKEND}]}, {"routes": {"image": {}}}
        )
    profile = SimpleNamespace(
        route_id="image", model_id="model", artifact_id="artifact", backend=IMAGE_BACKEND, expected_placement=None
    )
    events = [
        {"seq": 1, "epoch": 1, "request_id": "r", "kind": "user_observation"},
        {
            "seq": 2,
            "epoch": 1,
            "request_id": "r",
            "kind": "route",
            "payload": {
                "route_id": "image",
                "model": "model",
                "artifact_id": "artifact",
                "backend": IMAGE_BACKEND,
                "actual_placement": None,
            },
        },
        {"seq": 3, "epoch": 1, "request_id": "r", "kind": "model_metrics"},
        {"seq": 4, "epoch": 1, "request_id": "r", "kind": "final", "payload": {"answer": "yes"}},
    ]
    assert module._trace_complete(events, "yes", profile) is False


@pytest.mark.parametrize(
    "new_encoder,new_language,new_epoch,new_id",
    [
        (3, 2, 2, "task-step-1"),
        (2, 1, 2, "task-step-1"),
        (2, 2, 1, "task-step-1"),
        (2, 2, 2, "task-step-0"),
    ],
)
def test_terminal_native_sequence_cannot_skip_replay_or_reorder(launch, new_encoder, new_language, new_epoch, new_id):
    _cfg, plan, metrics, stage, identity = proof_fixture(launch)
    seen = set()
    validate_image_terminal(plan, metrics, stage, request_id="task-step-0", input_identity=identity, seen_states=seen)
    stage.update(request_id=new_id, epoch=new_epoch)
    report = metrics["runtime_telemetry"]["image_observation"]
    report.update(
        request_id=new_id,
        epoch=new_epoch,
        native_encoder_request_seq=new_encoder,
        text_native_io_request_seq=new_language,
    )
    metrics["runtime_telemetry"]["native_io_observation"].update(
        request_id=new_id, epoch=new_epoch, native_request_seq=new_language
    )
    before = copy.deepcopy(seen)
    with pytest.raises(ValueError):
        validate_image_terminal(plan, metrics, stage, request_id=new_id, input_identity=identity, seen_states=seen)
    assert seen == before


def test_native_app_selects_explicit_image_stage_and_refuses_implicit_default(tmp_path, launch, monkeypatch):
    from vllm_omni.edge.agent import native_app

    configured = agent_config_from_launch(launch, expected_device_name="GPU", experimental_bootstrap=True)
    path = tmp_path / "config.json"
    hardware = {
        "host_ram_available_bytes": 8192,
        "vram_available_bytes": 2048,
        "windows_commit_available_bytes": 8192,
        "gpu_name": "GPU",
        "vram_total_bytes": 2048,
        "power_condition": "AC",
    }
    monkeypatch.setattr(native_app.sys, "platform", "win32")
    monkeypatch.setattr(native_app, "_hardware_snapshot", lambda **kwargs: hardware)
    monkeypatch.setattr(native_app, "_artifact_disk_capacity", lambda routes: 8192)
    monkeypatch.setattr(native_app, "_gpu_pool_refusal", lambda *args: None)
    monkeypatch.setattr(native_app, "_fingerprint", lambda hardware: "fixture-only")
    monkeypatch.setattr(native_app, "_qualifications", lambda *args, **kwargs: [])
    monkeypatch.setattr(native_app, "EncryptedMemoryStore", lambda *args: None)
    monkeypatch.setattr(native_app, "WindowsToolBoundary", lambda: None)
    monkeypatch.setattr(native_app, "AgentController", lambda **kwargs: SimpleNamespace(**kwargs))
    path.write_text(json.dumps(configured))
    controller, _ = native_app.build_controller(path)
    wrapper = controller.backends[configured["routes"][0]["route_id"]]
    # The memory manager wraps the exact image backend; it does not start it.
    owned_backend = wrapper._coordinator._backends[wrapper._route_id]
    assert owned_backend.__class__ is OmniStrataImageBackend
    assert owned_backend.config.image_capability["encoder_device"] == "cpu"
    configured.pop("experimental_bootstrap_route_id")
    path.write_text(json.dumps(configured))
    with pytest.raises(ValueError, match="explicit experimental"):
        native_app.build_controller(path)


def test_lower_unseen_epoch_is_rejected_without_consuming_new_native_sequences(launch):
    _cfg, plan, metrics, stage, identity = proof_fixture(launch)
    stage["epoch"] = 3
    metrics["runtime_telemetry"]["image_observation"]["epoch"] = 3
    metrics["runtime_telemetry"]["native_io_observation"]["epoch"] = 3
    seen = set()
    validate_image_terminal(plan, metrics, stage, request_id="task-step-0", input_identity=identity, seen_states=seen)
    stage.update(request_id="task-step-1", epoch=2)
    metrics["runtime_telemetry"]["image_observation"].update(
        request_id="task-step-1", epoch=2, native_encoder_request_seq=2, text_native_io_request_seq=2
    )
    metrics["runtime_telemetry"]["native_io_observation"].update(
        request_id="task-step-1", epoch=2, native_request_seq=2
    )
    before = copy.deepcopy(seen)
    with pytest.raises(ValueError, match="epoch is stale"):
        validate_image_terminal(
            plan, metrics, stage, request_id="task-step-1", input_identity=identity, seen_states=seen
        )
    assert seen == before
