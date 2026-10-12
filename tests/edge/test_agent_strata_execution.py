# SPDX-License-Identifier: Apache-2.0
"""Synthetic route/report compatibility; no runtime, device or model proof."""

from __future__ import annotations

import copy
from dataclasses import asdict

import pytest

from benchmarks.edge_agent.native_profile import load_profile_routes
from tests.edge.test_agent_strata_profile import fixture  # noqa: F401
from tests.edge.test_strata_execution_parser import Harness
from vllm_omni.edge.agent.placement import (
    evidence_sha256,
    strata_route_binding,
    validate_strata_profile_plan,
    validate_strata_request_evidence,
)
from vllm_omni.engine.backends import strata_io as io
from vllm_omni.engine.weight_tiers import ArtifactFile, ArtifactManifest, WeightTierPlan

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def synthetic_static(runtime_manifest, engine_sha):
    """A valid metadata specimen; this does not attest any archived bytes."""
    value = dict.fromkeys(io.COMBINED_STATIC_KEYS, False)
    value.update(
        schema=io.COMBINED_STATIC_SCHEMA,
        status="static_archived_bytes_and_build_records_verified_not_live_runtime_eligible",
        base_revision=io.BASE_REVISION,
        base_tree="332979d72ea7c5fae7f00bec6f2c79234292bc3b",
        dependency_revision=io.DEPENDENCY_REVISION,
        dependency_tree=io.DEPENDENCY_TREE,
        native_io_schema="strata-omni-io-v1",
        native_execution_schema="strata-omni-exec-v1",
        combined_patch_manifest_sha256=io.COMBINED_PATCH_MANIFEST_SHA256,
        header_sha256=io.COMBINED_HEADER_SHA256,
        schema_sha256=io.COMBINED_SCHEMA_SHA256,
        parser_sha256=io.COMBINED_PARSER_SHA256,
        runtime_manifest_sha256=runtime_manifest,
        descriptor_sha256="b" * 64,
        native_executable_sha256=engine_sha,
        source_hashes={},
        build={},
        pe={},
        runtime_binding=None,
        physical_ssd_read_bytes=None,
        observer_layout_scope="compiled_standalone_fixture_reference_only",
    )
    value["patch_chain"] = [
        {"kind": kind, "file": f"patches/{index}.patch", "sha256": pin}
        for index, (kind, pin) in enumerate(
            (
                ("existing_io_patch", io.PATCH_SHA256),
                ("incremental_execution_patch", io.COMBINED_EXEC_PATCH_SHA256),
                ("boundary_safety_v2", io.COMBINED_BOUNDARY_PATCH_SHA256),
            )
        )
    ]
    for name, pins in io.COMBINED_SOURCE_CHAIN.items():
        rows = []
        for index, pin in enumerate(pins):
            if pin is None:
                rows.append(None)
                continue
            row = {"path": f"stages/{index}/source/{name}", "size_bytes": 1, "sha256": pin}
            if index == 0 and name in io.PATCH_SOURCES:
                row.update(canonical_sha256=pin, pin_scope="base_IO_canonical_Git_blob_raw_checkout_separately_bound")
            rows.append(row)
        value["source_hashes"][name] = rows
    value["observer_layout"] = {
        "state_bytes": 8000,
        "frame_buffer_bytes": 32768,
        "frame_state_bytes": 32784,
        "ticket_bytes": 16,
        "concurrent_ticket_count_verified": False,
        "root_registry_capacity": 16,
        "overhead_scope": "fixed_Cpp_state_and_one_owner_frame_plus_per_call_stack_ticket_excludes_CRT_stdio",
    }
    value["ABI_reference"] = {
        "schema": "omni-strata-compiled-fixture-ABI-reference-v1",
        "observer_layout": copy.deepcopy(value["observer_layout"]),
        "observer_layout_scope": "compiled_standalone_fixture_reference_only",
        "compiled_engine_ABI_verified": False,
        "runtime_binding": None,
    }
    value.pop("identity_sha256", None)
    value["identity_sha256"] = evidence_sha256(value, ascii=True)
    return value


@pytest.fixture
def combined(fixture):  # noqa: F811 - pytest injects the imported shared fixture
    entry = copy.deepcopy(fixture.entry)
    config = entry["backend_config"]
    runtime = ArtifactManifest.from_dict(config["runtime_manifest"])
    extras = (
        ArtifactFile("proof/combined.json", 1, "b" * 64, role="runtime"),
        ArtifactFile("proof/source-context.json", 1, "e" * 64, role="runtime"),
        ArtifactFile("proof/members.json", 1, "9" * 64, role="runtime"),
    )
    extras += tuple(
        ArtifactFile(
            f"adapter/{name}.py", 1, io.COMBINED_PARSER_SHA256 if name == "strata_exec" else "f" * 64, role="runtime"
        )
        for name in ("strata_exec", "strata_exec_bridge", "strata_exec_runtime", "strata_exec_live")
    )
    runtime = ArtifactManifest(runtime.checkpoint, runtime.revision, runtime.license, runtime.files + extras)
    config["runtime_manifest"] = runtime.to_dict()
    tier = copy.deepcopy(config["weight_tier_plan"])
    tier["budget"].update(host_workspace_bytes=512 << 20, ssd_temporary_bytes=16 << 20)
    config["weight_tier_plan"] = WeightTierPlan.from_dict(tier).to_dict()
    static = synthetic_static("9" * 64, "d" * 64)
    config["runtime_provenance"] = {"combined_member_manifest": {"file": "proof/members.json", "sha256": "9" * 64}}
    config["execution_observation"] = {
        "schema": "omni-strata-execution-observation-config-v1",
        "descriptor_file": "proof/combined.json",
        "source_context_file": "proof/source-context.json",
        "static_identity_sha256": static["identity_sha256"],
        "receipt_root": "synthetic-receipt-root",
        "workspace_bytes": 512 << 20,
        "receipt_storage_bytes": 16 << 20,
    }
    entry["artifact_id"] = "strata:" + evidence_sha256(config)
    routes, _ = load_profile_routes({"routes": [entry]}, fixture.lineage)
    route = asdict(routes[0])
    binding = route["backend_identity"]
    plan = copy.deepcopy(fixture.plan)
    plan.update(
        runtime_manifest_sha256=runtime.manifest_sha256,
        weight_tier_plan=config["weight_tier_plan"],
        observation_runtime=static,
    )
    execution = binding["execution_observation"]
    controls = {
        **binding["expected_controls"],
        "gpu_expert_cache": plan["gpu_expert_cache_control"],
        "observation_runtime": static,
        "execution_observation": {
            "schema": "omni-strata-execution-observation-config-v1",
            **{
                key: copy.deepcopy(execution[key])
                for key in ("static_identity_sha256", "adapter_sha256", "workspace_bytes", "receipt_storage_bytes")
            },
            "workspace_scope": "separate_declared_observer_allowance_not_a_measured_heap_hard_cap",
            "Agent_consumer_workspace_reused": False,
        },
    }
    plan.update(route_controls=controls, route_controls_sha256=evidence_sha256(controls, ascii=True))
    harness = Harness()
    native = harness.complete()
    native["runtime_binding"].update(
        runtime_manifest_sha256="9" * 64, native_executable_sha256="d" * 64, owner_adapter_sha256="f" * 64
    )
    native["runtime_binding_sha256"] = evidence_sha256(native["runtime_binding"], ascii=True)
    owner = native["owner_binding"]
    plan.update(
        worker_generation=owner["worker_generation"],
        gpu_observer_identity={
            "status": "verified",
            **{key: owner[key] for key in ("worker_generation", "pid", "creation_filetime_100ns", "gpu")},
        },
    )
    evidence = copy.deepcopy(fixture.evidence)
    evidence.update(execution_plan=plan, loaded_plan_sha256=evidence_sha256(plan))
    metrics = evidence["terminal_model_metrics"][0]["metrics"]
    stage = metrics["stage_event"]
    stage.update(request_id=native["request_id"], worker_generation=owner["worker_generation"])
    telemetry = metrics["backend_metrics"]["runtime_telemetry"]
    telemetry["native_io_observation"] = {
        "schema": io.COMBINED_IO_REPORT_SCHEMA,
        "runtime_identity_schema": io.COMBINED_STATIC_SCHEMA,
        "combined_io_binding": io.validate_combined_io_identity(static)
        | {"io_adapter_sha256": io.adapter_source_sha256()},
        "runtime_identity_sha256": static["identity_sha256"],
        "generation": stage["worker_generation"],
        "request_id": stage["request_id"],
        "epoch": stage["epoch"],
        "scope": io.SCOPE,
        "status": "complete",
        "physical_ssd_read_bytes": None,
        "loading_covered": False,
        "three_tier_memory_qualified": False,
        "reasons": [],
        "native_terminal": "stop",
        "native_pid": owner["pid"],
        "creation_filetime_100ns": owner["creation_filetime_100ns"],
        "native_request_seq": native["dispatch"]["native_request_seq"],
    }
    telemetry["native_execution_observation"] = {
        "schema": "omni-private-strata-owned-execution-report-v1",
        "status": "complete_scoped_observation",
        "complete": True,
        "channel_errors": [],
        "engine_abi_binding_established": True,
        "request_binding": {"request_id": native["request_id"], "epoch": native["epoch"]},
        "static_runtime_identity_sha256": static["identity_sha256"],
        "lifecycle_outcome": "normal",
        "reader_healthy": True,
        "omni_completed": True,
        "cancellation_requested": False,
        "pending_dispatch_count_at_finish": 0,
        "native_observation": native,
        "runtime_binding_sha256": native["runtime_binding_sha256"],
        "physical_ssd_read_bytes": None,
        "actual_whole_model_placement": None,
        "aggregate_gpu_hard_cap_verified": False,
        "aggregate_ram_hard_cap_verified": False,
        "runtime_qualification": False,
        "default_eligible": False,
    }
    return entry, route, evidence, telemetry


def test_combined_route_accepts_separate_runtime_without_legacy_conversion(combined):
    _, route, evidence, _ = combined
    assert route["backend_identity"]["observation_runtime"] is None
    validate_strata_request_evidence(evidence, route)
    assert (
        route["backend_identity"]["runtime_manifest_sha256"]
        != (route["backend_identity"]["execution_observation"]["member_manifest_sha256"])
    )
    assert evidence["execution_plan"]["observation_runtime"]["compiled_engine_ABI_verified"] is False


@pytest.mark.parametrize("actual", ["GEN", "GENI"])
@pytest.mark.parametrize("expected", ["GEN", "GENI", "BGEN", "", None, True])
def test_execution_command_is_selected_by_caller_not_report(combined, actual, expected):
    from vllm_omni.edge.agent.strata_execution_evidence import validate_execution_report

    _, route, evidence, telemetry = combined
    report = telemetry["native_execution_observation"]
    report["native_observation"]["dispatch"]["command"] = actual
    args = (
        report,
        evidence["terminal_model_metrics"][0]["metrics"]["stage_event"],
        evidence["execution_plan"],
        route["backend_identity"],
        telemetry["native_io_observation"],
    )
    if expected == actual:
        validate_execution_report(*args, expected_command=expected)
    else:
        with pytest.raises(ValueError):
            validate_execution_report(*args, expected_command=expected)
    if actual == "GENI":
        with pytest.raises(ValueError):
            validate_execution_report(*args)
        # The public text qualification consumer cannot acquire image support
        # just because the shared report validator understands GENI.
        with pytest.raises(ValueError):
            validate_strata_request_evidence(evidence, route)


@pytest.mark.parametrize(
    "mutation", [None, "missing_binding", "missing_report", "wrong_command", "io_request", "stage_request"]
)
def test_image_stage_text_followup_requires_combined_execution_proof(combined, mutation):
    from vllm_omni.edge.agent.strata_image_evidence import validate_image_terminal

    _, route, evidence, telemetry = combined
    stage = evidence["terminal_model_metrics"][0]["metrics"]["stage_event"]
    request_id = stage["request_id"]
    binding = route["backend_identity"]
    if mutation == "missing_binding":
        binding = None
    elif mutation == "missing_report":
        telemetry.pop("native_execution_observation")
    elif mutation == "wrong_command":
        telemetry["native_execution_observation"]["native_observation"]["dispatch"]["command"] = "GENI"
    elif mutation == "io_request":
        telemetry["native_io_observation"]["request_id"] = "other"
    elif mutation == "stage_request":
        stage["request_id"] = "other"
    seen = set()
    args = (evidence["execution_plan"], {"runtime_telemetry": telemetry}, stage)
    kwargs = dict(request_id=request_id, input_identity=None, seen_states=seen, execution_binding=binding)
    if mutation is None:
        assert validate_image_terminal(*args, **kwargs) is None
    else:
        with pytest.raises(ValueError):
            validate_image_terminal(*args, **kwargs)
    assert seen == set()


@pytest.mark.parametrize("mutation", [None, "command", "missing_execution", "image_input", "io_sequence"])
def test_combined_image_chain_checks_both_reports_before_consuming_state(combined, mutation):
    from tests.edge import test_agent_strata_image as images
    from vllm_omni.edge.agent.strata_image_evidence import validate_image_terminal

    _, route, evidence, telemetry = combined
    _, image_plan, image_metrics, _, identity = images.proof_fixture(images.launch.__wrapped__())
    plan = evidence["execution_plan"]
    for key in ("image_route", "owned_encoder_at_load", "encoder_backend_selection_at_load"):
        plan[key] = image_plan[key]
    stage = evidence["terminal_model_metrics"][0]["metrics"]["stage_event"]
    native_io = telemetry["native_io_observation"]
    plan["selected_native_module_audit_at_load"] = {
        "pid": native_io["native_pid"], "creation_filetime_100ns": native_io["creation_filetime_100ns"]
    }
    image_report = image_metrics["runtime_telemetry"]["image_observation"]
    image_report.update(request_id=stage["request_id"], epoch=stage["epoch"], generation=stage["worker_generation"])
    image_report["language_dispatch"].update(
        language_pid=native_io["native_pid"], language_creation_filetime_100ns=native_io["creation_filetime_100ns"]
    )
    telemetry.update(image_observation=image_report, native_io_excludes_encoder=True,
                     encoder_all_operators_placement=None)
    report = telemetry["native_execution_observation"]
    report["native_observation"]["dispatch"]["command"] = "GENI"
    if mutation == "command":
        report["native_observation"]["dispatch"]["command"] = "GEN"
    elif mutation == "missing_execution":
        telemetry.pop("native_execution_observation")
    elif mutation == "image_input":
        image_report["input_sha256"] = "0" * 64
    elif mutation == "io_sequence":
        native_io["native_request_seq"] += 1
    seen = set()
    args = (plan, {"runtime_telemetry": telemetry}, stage)
    kwargs = dict(request_id=stage["request_id"], input_identity=identity, seen_states=seen,
                  execution_binding=route["backend_identity"])
    if mutation is not None:
        with pytest.raises(ValueError):
            validate_image_terminal(*args, **kwargs)
        assert seen == set()
    else:
        proof = validate_image_terminal(*args, **kwargs)
        assert len(seen) == 4
        assert proof["native_execution_observation"] == report
        report["complete"] = False
        assert proof["native_execution_observation"]["complete"] is True
        report["complete"] = True
        with pytest.raises(ValueError, match="reuses"):
            validate_image_terminal(*args, **kwargs)


@pytest.mark.parametrize("mutation", [None, "shared_workspace", "shared_scratch", "missing_projector_disk"])
def test_combined_image_route_adds_independent_observer_and_encoder_budgets(combined, mutation):
    from dataclasses import replace

    from tests.edge import test_agent_strata_image as images

    launch = images.launch.__wrapped__()
    config = launch["backend"]
    combined_config = combined[0]["backend_config"]
    old_runtime = ArtifactManifest.from_dict(config["runtime_manifest"])
    combined_runtime = ArtifactManifest.from_dict(combined_config["runtime_manifest"])
    old_paths = {row.path for row in old_runtime.files}
    runtime = replace(old_runtime, files=old_runtime.files + tuple(
        row for row in combined_runtime.files if row.path not in old_paths
    ))
    config["runtime_manifest"] = runtime.to_dict()
    for key in ("execution_observation", "runtime_provenance"):
        config[key] = copy.deepcopy(combined_config[key])
    image = config["image_route"]
    observer = config["execution_observation"]
    tier = WeightTierPlan.from_dict(config["weight_tier_plan"])
    disk = sum(ArtifactManifest.from_dict(config[key]).total_size_bytes
               for key in ("artifact_manifest", "prepared_pack_manifest", "runtime_manifest"))
    projector_bytes = ArtifactManifest.from_dict(image["projector_manifest"]).total_size_bytes
    workspace = image["vision_host_bytes"] + image["vision_scratch_bytes"] + observer["workspace_bytes"]
    budget = replace(tier.budget, host_workspace_bytes=workspace,
                     ssd_temporary_bytes=image["vision_scratch_bytes"] + observer["receipt_storage_bytes"],
                     ssd_artifact_bytes=disk + projector_bytes, windows_commit_peak_bytes=1 << 30)
    if mutation == "shared_workspace":
        budget = replace(budget, host_workspace_bytes=observer["workspace_bytes"])
    elif mutation == "shared_scratch":
        budget = replace(budget, ssd_temporary_bytes=observer["receipt_storage_bytes"])
    elif mutation == "missing_projector_disk":
        budget = replace(budget, ssd_artifact_bytes=disk)
    config["weight_tier_plan"] = replace(tier, budget=budget).to_dict()
    config["host_overhead_bytes"] = budget.host_workspace_bytes - observer["workspace_bytes"]
    demands = budget.resource_demands(include_windows_commit=True)
    launch["resource_budget"] = {"demands": demands, "capacities": {key: 2 << 30 for key in demands}}
    if mutation is None:
        route = images.entry(launch)
        assert route["modalities"] == ["text", "image"]
        assert route["memory_demands"]["host_ram"] == demands["host_ram"]
    else:
        with pytest.raises(ValueError):
            images.entry(launch)


@pytest.mark.parametrize("mutation", ["missing_adapter", "alias", "unbudgeted", "legacy_mixed", "raw_manifest"])
def test_combined_route_rejects_unbound_or_unadmitted_metadata(combined, mutation):
    entry, _, _, _ = combined
    config = entry["backend_config"]
    if mutation == "missing_adapter":
        config["execution_observation"]["descriptor_file"] = "absent.json"
    elif mutation == "alias":
        config["execution_observation"]["source_context_file"] = "../proof/source-context.json"
    elif mutation == "unbudgeted":
        config["weight_tier_plan"]["budget"]["host_workspace_bytes"] = 0
    elif mutation == "legacy_mixed":
        config["observation_runtime_identity_sha256"] = "a" * 64
    else:
        config["runtime_provenance"]["combined_member_manifest"]["sha256"] = "a" * 64
    entry["artifact_id"] = "strata:" + evidence_sha256(config)
    with pytest.raises(ValueError):
        strata_route_binding(entry)


@pytest.mark.parametrize("mutation", ["controls", "static", "fake_qualification"])
def test_combined_plan_rejects_changed_identity_budget_or_scope(combined, mutation):
    _, route, evidence, _ = combined
    plan = evidence["execution_plan"]
    if mutation == "controls":
        plan["route_controls"]["execution_observation"]["workspace_bytes"] += 1
    elif mutation == "static":
        plan["observation_runtime"]["descriptor_sha256"] = "a" * 64
    else:
        plan["observation_runtime"]["runtime_qualification"] = True
    plan["route_controls_sha256"] = evidence_sha256(plan["route_controls"], ascii=True)
    with pytest.raises(ValueError):
        validate_strata_profile_plan(plan, route)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "request",
        "bool_epoch",
        "generation",
        "birth",
        "snapshot",
        "missing_done",
        "done_bool_sequence",
        "done_float_sequence",
        "physical_ssd",
        "qualification",
        "legacy_io",
        "io_adapter",
        "io_sequence",
        "io_incomplete",
    ],
)
def test_combined_request_rejects_cross_request_incomplete_or_overclaimed_reports(combined, mutation):
    _, route, evidence, telemetry = combined
    report = telemetry["native_execution_observation"]
    if mutation == "missing":
        telemetry.pop("native_execution_observation")
    elif mutation == "request":
        report["request_binding"]["request_id"] = "other"
    elif mutation == "bool_epoch":
        report["request_binding"]["epoch"] = True
    elif mutation in {"generation", "birth"}:
        owner = report["native_observation"]["owner_binding"]
        owner["worker_generation" if mutation == "generation" else "creation_filetime_100ns"] = (
            "other" if mutation == "generation" else 1
        )
        report["native_observation"]["owner_binding_sha256"] = evidence_sha256(owner, ascii=True)
    elif mutation == "snapshot":
        report["native_observation"]["snapshots"].reverse()
    elif mutation == "missing_done":
        report["native_observation"]["native_done"] = None
    elif mutation in {"done_bool_sequence", "done_float_sequence"}:
        done = report["native_observation"]["native_done"]
        assert done["native_request_seq"] == 1
        done["native_request_seq"] = True if mutation == "done_bool_sequence" else 1.0
    elif mutation == "physical_ssd":
        report["physical_ssd_read_bytes"] = 100
    elif mutation == "qualification":
        report["runtime_qualification"] = True
    elif mutation == "legacy_io":
        telemetry["native_io_observation"]["schema"] = "omni-strata-request-io-observation-v1"
    elif mutation == "io_adapter":
        telemetry["native_io_observation"]["combined_io_binding"]["io_adapter_sha256"] = "a" * 64
    elif mutation == "io_sequence":
        telemetry["native_io_observation"]["native_request_seq"] += 1
    else:
        telemetry["native_io_observation"]["status"] = "incomplete"
    with pytest.raises(ValueError):
        validate_strata_request_evidence(evidence, route)
