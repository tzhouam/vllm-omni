# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Closed synthetic files exercise the real registrar chain, never neural code.

Only the native compile/static-observation proof boundary and live sensors are
synthetic. Parent preparation, public registration, typed manifests, conversion
receipt, PE parser, projector/build/file verification and budget logic are real.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = Path(os.environ.get("OMNI_STRATA_PRODUCTION_ROOT", str(ROOT)))
sys.path.insert(0, str(PRODUCTION))

from benchmarks.edge_harness import strata_runtime
from benchmarks.edge_harness.strata_prepare import inventory, prepare
from benchmarks.edge_harness.strata_profile import RUNTIME_REVISION, canonical_hash, file_hash, save_json
from benchmarks.edge_harness.test_strata_prepare import fake_packer
from benchmarks.edge_harness.test_strata_prepare import fixture as preparation_fixture


def direct(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helper_tests = Path(
    os.environ.get("OMNI_STRATA_VISION_HELPER_TESTS", str(PRODUCTION / "tests/edge/test_strata_vision_prototype.py"))
)
helpers = direct("vision_registrar_fixture_helpers", helper_tests)
registration = direct("private_cpu_vision_registration", Path(__file__).with_name("strata_vision_register.py"))


@pytest.fixture
def case(tmp_path, monkeypatch):
    vision, tiers, base = helpers.vision, helpers.tiers, helpers.base
    io_module = helpers.direct_module("image_registration_actual_io", helpers.BACKENDS / "strata_io.py")
    for name in ("vllm_omni", "vllm_omni.engine", "vllm_omni.engine.backends"):
        package = types.ModuleType(name)
        package.__path__ = [str(PRODUCTION / name.replace(".", "/"))]
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "vllm_omni.engine.weight_tiers", tiers)
    monkeypatch.setitem(sys.modules, "vllm_omni.engine.backends.strata_io", io_module)
    monkeypatch.setattr(
        registration, "_helpers", lambda: (base, io_module, vision, tiers.ArtifactManifest, tiers.WeightTierPlan)
    )
    base.__file__ = str(helpers.BACKENDS / "strata.py")
    target, args = preparation_fixture(tmp_path)
    args.update(host_capacity_bytes=1 << 30, windows_commit_capacity_bytes=1 << 30, max_io_bytes=1 << 20)
    args["host_overhead_bytes"] = 2 << 20
    args["component_budget"].update(
        gpu_expert_cache_bytes=512,
        gpu_workspace_bytes=488,
        host_transfer_bytes=1 << 20,
        host_workspace_bytes=1 << 20,
        host_loading_peak_bytes=(3 << 20) + 100,
        windows_commit_peak_bytes=4 << 20,
    )

    def complete_fixture_packer(command, runtime, log):
        fake_packer(command, runtime, log)
        pack = Path(command[command.index("--out") + 1])
        (pack / "native_experts.txt").write_text(
            "# strata native experts v4: fixture\n0 1 1 0 64 0 0 0 test-00001-of-00002.gguf\n",
            encoding="utf-8",
        )

    prepare(target, **args, pack_runner=complete_fixture_packer)
    runtime = tmp_path / "image-runtime"
    shutil.copytree(args["runtime_root"], runtime)
    (runtime / "engine/strata.exe").write_bytes(b"synthetic observed native, no neural code")
    (runtime / "engine/strata-vision.exe").write_bytes(helpers.pe_bytes())
    proof = []
    reviewed = {}
    for role in (*vision.REVIEWED_VISION_EVIDENCE, "configure_log", "build_log"):
        path = runtime / "vision-build" / (role + ".txt")
        path.parent.mkdir(exist_ok=True)
        path.write_text("SYNTHETIC COMPILE RECEIPT ONLY: " + role, encoding="utf-8")
        row = dict(role=role, path=path.relative_to(runtime).as_posix(), sha256=file_hash(path))
        proof.append(row)
        if role in vision.REVIEWED_VISION_EVIDENCE:
            reviewed[role] = row["sha256"]
    monkeypatch.setattr(vision, "REVIEWED_VISION_EVIDENCE", reviewed)
    system = tmp_path / "System32"
    system.mkdir()
    monkeypatch.setattr(vision, "_windows_system_directory", lambda: system)
    closure = vision.scan_pe_closure(
        runtime / "engine/strata-vision.exe", runtime, [runtime / "engine/strata-vision.exe"]
    )
    build = dict(
        schema="omni-strata-vision-build-v1",
        base_revision=vision.BASE_REVISION,
        dependency_revision=vision.DEPENDENCY_REVISION,
        dependency_tree=vision.DEPENDENCY_TREE,
        encoder_sha256=file_hash(runtime / "engine/strata-vision.exe"),
        native_observation_schema="strata-vision-backend-v1",
        configure_exit_code=0,
        build_exit_code=0,
        evidence_files=proof,
        native_dependency_files=[],
        pe_dependency_closure=closure,
        cmake_options=dict(STRATA_VISION_CUDA="OFF", STRATA_PORTABLE="ON", CMAKE_BUILD_TYPE="Release"),
    )
    save_json(runtime / "vision-build/build.json", build)
    runtime_manifest = tmp_path / "runtime-manifest.json"
    save_json(runtime_manifest, inventory(runtime, "test/Strata", RUNTIME_REVISION, "fixture", runtime=True))
    descriptor = dict(engine_file="engine/strata.exe", synthetic_compile_proof_boundary=True)
    real_verifier = io_module.verify_observation_runtime
    calls = []

    def observed(value, root, files, engine, *bootstrap):
        assert value == descriptor
        assert root == runtime.resolve() and engine == (runtime / "engine/strata.exe").resolve()
        assert files == {path.resolve() for path in runtime.rglob("*") if path.is_file()}
        assert not bootstrap or bootstrap == (base._BOOTSTRAP,)
        calls.append(True)
        identity = dict(
            schema="synthetic-static-observation-proof-for-file-tests",
            native_executable_sha256=file_hash(engine),
            io_adapter_sha256=file_hash(Path(io_module.__file__)),
            three_tier_memory_qualified=False,
        )
        return identity | {"identity_sha256": canonical_hash(identity)}

    monkeypatch.setattr(io_module, "verify_observation_runtime", observed)
    text = tmp_path / "intermediate/launch.json"
    strata_runtime.register_runtime_variant(
        args["launch_out"],
        runtime_root=runtime,
        runtime_manifest_path=runtime_manifest,
        observation_runtime=descriptor,
        launch_out=text,
    )
    launch = json.loads(text.read_text(encoding="utf-8"))
    source = tiers.ArtifactManifest.from_dict(launch["backend"]["artifact_manifest"])
    projector = tmp_path / "projector"
    projector.mkdir()
    (projector / "mmproj.gguf").write_bytes(b"synthetic projector" * 128)
    projector_manifest = helpers.manifest(
        projector, [("mmproj.gguf", "vision_projector")], "fixture-projector", "b" * 40
    )
    config = dict(
        schema="omni-strata-image-route-v1",
        projector_root=str(projector),
        projector_manifest=projector_manifest.to_dict(),
        projector_file="mmproj.gguf",
        text_artifact_manifest_sha256=source.manifest_sha256,
        encoder_file="engine/strata-vision.exe",
        encoder_build_manifest_file="vision-build/build.json",
        embedding_width=4,
        max_image_bytes=4096,
        max_image_pixels=16,
        max_image_tokens=4,
        encoder_threads=1,
        encoder_device="cpu",
        allow_cpu_fallback=False,
        vision_host_bytes=65536,
        vision_gpu_bytes=0,
        vision_scratch_bytes=8192,
    )
    memory = dict(
        host_ram_available_bytes=1 << 30,
        gpu_total_bytes=2000,
        gpu_free_bytes=2000,
        windows_commit_available_bytes=1 << 30,
        is_wsl=False,
    )
    monkeypatch.setattr(base, "_probe_memory", lambda gpu: copy.deepcopy(memory))
    return dict(
        args=args,
        text=text,
        receipt=text.with_suffix(".registration.json"),
        runtime=runtime,
        runtime_manifest=runtime_manifest,
        projector=projector,
        config=config,
        memory=memory,
        tiers=tiers,
        base=base,
        vision=vision,
        io=io_module,
        real_verifier=real_verifier,
        calls=calls,
        out=tmp_path / "image/launch.json",
    )


def run(case, **kwargs):
    return registration.register_cpu_image_route(
        case["text"],
        text_registration_path=case["receipt"],
        image_route=case["config"],
        launch_out=case["out"],
        route_id="test-image-cpu-v1",
        **kwargs,
    )


def test_image_registration_replays_real_parent_files_and_adds_coexistence_budget(case):
    paths = [case["text"], case["receipt"], case["args"]["launch_out"], *case["args"]["out_pack"].rglob("*")]
    before = {path: path.read_bytes() for path in paths if path.is_file()}
    text = json.loads(case["text"].read_text(encoding="utf-8"))
    receipt = run(case)
    image = json.loads(case["out"].read_text(encoding="utf-8"))
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert len(case["calls"]) == 2  # initial public registration and real metadata replay
    assert image["backend"]["name"] == registration.BACKEND
    assert image["backend"]["weight_tier_plan"]["backend"] == registration.BACKEND
    assert image["backend"]["runtime_manifest"] == text["backend"]["runtime_manifest"]
    assert image["backend"]["artifact_manifest"] == text["backend"]["artifact_manifest"]
    old, new = (launch["backend"]["weight_tier_plan"]["budget"] for launch in (text, image))
    extra = 65536 + 8192
    for field in ("host_workspace_bytes", "host_loading_peak_bytes", "windows_commit_peak_bytes"):
        assert new[field] == old[field] + extra
    assert image["backend"]["host_overhead_bytes"] == text["backend"]["host_overhead_bytes"] + extra
    assert new["host_transfer_bytes"] == old["host_transfer_bytes"]
    assert new["ssd_temporary_bytes"] == old["ssd_temporary_bytes"] + 8192
    assert new["ssd_artifact_bytes"] == old["ssd_artifact_bytes"] + (case["projector"] / "mmproj.gguf").stat().st_size
    assert image["resource_budget"]["demands"]["vram:0"] == text["resource_budget"]["demands"]["vram:0"]
    assert receipt["launch_canonical_sha256"] == canonical_hash(image)
    assert receipt["qualification_created"] is False and receipt["default_route_created"] is False
    assert receipt["fresh_admission"]["admitted"] is True
    source = case["tiers"].ArtifactManifest.from_dict(image["backend"]["artifact_manifest"])
    assert receipt["source_manifest_sha256"] == source.manifest_sha256
    assert receipt["source_manifest_input_canonical_sha256"] == canonical_hash(image["backend"]["artifact_manifest"])


def test_inline_image_bound_needs_explicit_transport_expansion_and_charges_all_host_pools(case):
    case["config"]["max_image_bytes"] = 4 << 20
    case["config"]["vision_host_bytes"] = 12 << 20
    case["config"]["vision_scratch_bytes"] = 5 << 20
    with pytest.raises(ValueError, match="transport bytes"):
        run(case)
    before = json.loads(case["text"].read_text(encoding="utf-8"))
    receipt = run(case, max_io_bytes=8 << 20)
    after = json.loads(case["out"].read_text(encoding="utf-8"))
    delta = 7 << 20
    old, new = (launch["backend"]["weight_tier_plan"]["budget"] for launch in (before, after))
    assert new["host_transfer_bytes"] == old["host_transfer_bytes"] + delta
    assert new["host_loading_peak_bytes"] == old["host_loading_peak_bytes"] + (17 << 20) + delta
    assert new["windows_commit_peak_bytes"] == old["windows_commit_peak_bytes"] + (17 << 20) + delta
    assert after["backend"]["host_overhead_bytes"] == before["backend"]["host_overhead_bytes"] + (17 << 20)
    assert receipt["additional_host_transport_buffer_bytes"] == delta
    assert after["backend"]["max_io_bytes"] == 8 << 20
    assert before["backend"]["max_io_bytes"] == 1 << 20


@pytest.mark.parametrize(
    "which", ["text_receipt", "parent", "prepare", "binding", "pack", "packer", "source", "runtime", "projector"]
)
def test_changed_linked_metadata_or_asset_bytes_refused_without_outputs(case, which):
    launch = json.loads(case["text"].read_text(encoding="utf-8"))
    paths = dict(
        text_receipt=case["receipt"],
        parent=case["args"]["launch_out"],
        prepare=Path(launch["preparation_receipt"]),
        binding=case["args"]["out_pack"].with_name(case["args"]["out_pack"].name + ".omni-binding.json"),
        pack=case["args"]["out_pack"] / "dense.bin",
        packer=case["args"]["runtime_root"] / "tools/iq_pack.py",
        source=case["args"]["artifact_root"] / "test-00001-of-00002.gguf",
        runtime=case["runtime"] / "vision-build/build.json",
        projector=case["projector"] / "mmproj.gguf",
    )
    path = paths[which]
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["tampered"] = True
        save_json(path, value)
    else:
        path.write_bytes(b"changed closed file")
    with pytest.raises((ValueError, KeyError)):
        run(case)
    assert not case["out"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("encoder_device", "cuda"),
        ("vision_gpu_bytes", 1),
        ("vision_host_bytes", 1),
        ("vision_scratch_bytes", 1),
        ("text_artifact_manifest_sha256", "0" * 64),
        ("max_image_tokens", 4096),
        ("allow_cpu_fallback", True),
    ],
)
def test_image_route_capacity_or_source_or_cpu_claim_changes_refused(case, field, value):
    case["config"][field] = value
    with pytest.raises(ValueError):
        run(case)
    assert not case["out"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("host_ram_available_bytes", 1),
        ("gpu_free_bytes", 1),
        ("gpu_total_bytes", None),
        ("windows_commit_available_bytes", 1),
    ],
)
def test_registration_requires_fresh_ram_vram_commit_not_old_preparation_snapshot(case, field, value):
    case["memory"][field] = value
    with pytest.raises(Exception, match="admission refused"):
        run(case)
    assert not case["out"].exists()


def test_real_static_verifier_does_not_accept_syntax_only_build_proof(case, monkeypatch):
    monkeypatch.setattr(case["io"], "verify_observation_runtime", case["real_verifier"])
    with pytest.raises(ValueError, match="descriptor"):
        run(case)
    assert not case["out"].exists()


def test_unbound_runtime_code_is_not_adopted(case):
    (case["runtime"] / "surprise.pyc").write_bytes(b"unbound code")
    with pytest.raises(ValueError, match="unbound"):
        run(case)
    assert not case["out"].exists()


def test_registration_rejects_existing_output_and_immutable_bundle_destination(case):
    case["out"].parent.mkdir()
    case["out"].write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        run(case)
    assert case["out"].read_bytes() == b"preserve"
    case["out"] = case["projector"] / "new.json"
    with pytest.raises(ValueError, match="outside immutable"):
        run(case)


def test_missing_python_pillow_or_changed_native_environment_refuses(case, monkeypatch):
    monkeypatch.setattr(
        case["base"],
        "_verify_python_environment",
        lambda *_: (_ for _ in ()).throw(ValueError("changed Python/Pillow")),
    )
    with pytest.raises(ValueError, match="Python/Pillow"):
        run(case)
    assert not case["out"].exists()


def test_mutation_during_fresh_probe_cannot_publish_stale_projector_or_receipt(case, monkeypatch):
    def changed(_):
        (case["projector"] / "mmproj.gguf").write_bytes(b"changed after verified hash")
        return copy.deepcopy(case["memory"])

    monkeypatch.setattr(case["base"], "_probe_memory", changed)
    with pytest.raises(ValueError, match="changed after verification"):
        run(case)
    assert not case["out"].exists()


def test_disk_capacity_and_unverified_volume_are_fail_closed(case, monkeypatch):
    monkeypatch.setattr(
        registration, "_disk_snapshot", lambda roots: [dict(total_bytes=100, free_bytes=100, filesystem_device=1)]
    )
    with pytest.raises(ValueError, match="SSD artifact/temporary claim"):
        run(case)
    monkeypatch.setattr(
        registration,
        "_disk_snapshot",
        lambda roots: [dict(total_bytes=1 << 40, free_bytes=1 << 40, filesystem_device=i) for i in (1, 2)],
    )
    with pytest.raises(ValueError, match="multiple/unverified"):
        run(case)
    monkeypatch.setattr(registration, "_disk_snapshot", lambda roots: [dict(total_bytes=1 << 40, free_bytes=1 << 40)])
    with pytest.raises(ValueError, match="multiple/unverified"):
        run(case)


@pytest.mark.parametrize("transport", [True, 0, 1024, (64 << 20) + 1])
def test_invalid_or_reduced_transport_cannot_silently_change_parent_budget(case, transport):
    with pytest.raises(ValueError):
        run(case, max_io_bytes=transport)
    assert not case["out"].exists()


def test_inherited_resource_ceilings_are_not_silently_increased(case):
    case["config"]["vision_host_bytes"] = 2 << 30
    with pytest.raises(ValueError, match="inherited resource ceilings"):
        run(case)
    assert not case["out"].exists()


@pytest.mark.skipif(os.name != "nt", reason="initial CPU image production preflight is Windows only")
@pytest.mark.parametrize("reintroduce_v1_double_count", [False, True])
def test_derived_image_passes_actual_constructor_through_native_launch_boundary(
    case, monkeypatch, reintroduce_v1_double_count
):
    """Run the real v3 constructor/file/budget prefix, stopping before Popen.

    Files and typed budgets are real tiny fixtures. Native static compile proof
    remains the explicitly mocked fixture boundary; no process/model is run.
    """
    case["config"].update(max_image_bytes=4 << 20, vision_host_bytes=12 << 20, vision_scratch_bytes=5 << 20)
    run(case, max_io_bytes=8 << 20)
    launch = json.loads(case["out"].read_text())
    config = launch["backend"]
    if reintroduce_v1_double_count:
        config["host_overhead_bytes"] += 7 << 20
    base = case["base"]
    monkeypatch.setattr(base, "strata_io", case["io"])
    monkeypatch.setattr(base, "OmegaConf", types.SimpleNamespace(is_config=lambda _: False))
    verified_python = base._verify_python_environment(Path(config["python_bin"]), config["python_environment"])
    # This exact isolated interpreter was just observed; do not let its repeat
    # version probe hit the separately prohibited native-launch boundary.
    monkeypatch.setattr(base, "_verify_python_environment", lambda *_: verified_python)

    class NativeLaunchBoundaryError(RuntimeError):
        pass

    calls = []

    def refuse_process(*args, **kwargs):
        calls.append((args, kwargs))
        raise NativeLaunchBoundaryError("all real constructor prefix gates passed; native execution prohibited")

    def close_prefix_only(self):
        assert self._proc is None
        if self._temporary is not None:
            self._temporary.cleanup()

    monkeypatch.setattr(base.subprocess, "Popen", refuse_process)
    monkeypatch.setattr(helpers.multimodal.StrataMultimodalStageClient, "shutdown", close_prefix_only)
    error = helpers.ResourceUnavailableError if reintroduce_v1_double_count else NativeLaunchBoundaryError
    with pytest.raises(error):
        helpers.multimodal.StrataMultimodalStageClient(
            types.SimpleNamespace(stage_id=0),
            config,
            types.SimpleNamespace(),
            types.SimpleNamespace(demands=launch["resource_budget"]["demands"]),
        )
    assert bool(calls) is not reintroduce_v1_double_count
