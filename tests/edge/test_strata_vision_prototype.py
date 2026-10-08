"""Private source/protocol tests. No neural model, device qualification or build."""

from __future__ import annotations

import ast
import asyncio
import base64
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ROOT / "vllm_omni/engine/backends"
PRODUCTION = Path(os.environ.get("OMNI_STRATA_PRODUCTION_ROOT", str(ROOT)))


def direct_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vision = direct_module("private_strata_vision", BACKENDS / "strata_vision.py")


def load_ast(name, path, injected, excluded_modules):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        or not any((node.module or "").startswith(prefix) for prefix in excluded_modules)
    ]
    module = types.ModuleType(name)
    module.__dict__.update(injected)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@dataclasses.dataclass
class StageRequest:
    request_id: str
    stage_id: int
    epoch: int
    worker_generation: str


class ResourceUnavailableError(RuntimeError):
    pass


base = load_ast(
    "private_strata_base",
    BACKENDS / "strata.py",
    {
        "StageClientBase": object,
        "StageRequest": StageRequest,
        "StageEvent": object,
        "OmniRequestOutput": object,
        "CompletionOutput": object,
        "ResourceUnavailable": ResourceUnavailableError,
        "strata_io": types.SimpleNamespace(),
        "OmegaConf": types.SimpleNamespace(),
    },
    ("vllm", "omni_stage_contracts", "omegaconf"),
)
multimodal = load_ast(
    "private_strata_multimodal",
    BACKENDS / "strata_multimodal.py",
    {
        "strata": base,
        "strata_vision": vision,
    },
    ("vllm",),
)
tiers = load_ast(
    "private_weight_tiers",
    PRODUCTION / "vllm_omni/engine/weight_tiers.py",
    {
        "Reservation": object,
        "ResourceLedger": object,
    },
    ("vllm",),
)


def png(width=2, height=2):
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (20, 100, 30)).save(stream, format="PNG")
    return stream.getvalue()


def url(data):
    return "data:image/png;base64," + base64.b64encode(data).decode()


def image_input(data=None, **overrides):
    bounds = {"max_bytes": 4096, "max_pixels": 16, "max_io_bytes": 8192}
    bounds.update(overrides)
    return vision.png_input(url(png() if data is None else data), **bounds)


@pytest.mark.parametrize(
    "source",
    [
        "http://localhost/image.png",
        "file:///x.png",
        "data:image/jpeg;base64,eA==",
        "data:image/png;base64,@@@",
        "data:image/png;base64,eA==",
    ],
)
def test_image_payload_refuses_remote_wrong_format_or_invalid_bytes(source):
    with pytest.raises(ValueError):
        vision.png_input(source, max_bytes=4096, max_pixels=16, max_io_bytes=8192)


def test_png_decode_and_transport_have_independent_bounds():
    result = image_input()
    assert result["sha256"] == hashlib.sha256(png()).hexdigest()
    assert (result["width"], result["height"], result["pixels"]) == (2, 2, 4)
    for limits in ({"max_bytes": 4}, {"max_pixels": 3}, {"max_io_bytes": 20}, {"text_bytes": 8192}):
        with pytest.raises(ValueError):
            image_input(**limits)


def test_pixel_header_checked_before_decoder(monkeypatch):
    oversized = bytearray(png())
    oversized[16:24] = struct.pack(">II", 100_000, 100_000)
    monkeypatch.setattr(Image, "open", lambda *_: pytest.fail("decoder allocated before pixel admission"))
    with pytest.raises(ValueError, match="pixel"):
        image_input(bytes(oversized))


def test_corrupt_truncated_and_animated_png_refused():
    data = png()
    with pytest.raises((ValueError, OSError, SyntaxError)):
        image_input(data[:-15])
    stream = io.BytesIO()
    images = [Image.new("RGB", (2, 2), color) for color in ("red", "blue")]
    images[0].save(stream, format="PNG", save_all=True, append_images=images[1:], duration=1)
    with pytest.raises(ValueError, match="animated"):
        image_input(stream.getvalue())


def manifest(root, files, checkpoint="fixture", revision="fixture-revision"):
    return tiers.ArtifactManifest(
        checkpoint=checkpoint,
        revision=revision,
        license="fixture-test-only",
        files=tuple(
            tiers.ArtifactFile(
                path=relative,
                size_bytes=(root / relative).stat().st_size,
                sha256=vision.digest(root / relative),
                role=role,
            )
            for relative, role in files
        ),
    )


def pe_bytes(normal=(), delay=()):
    """Synthetic import-only x64 PE bytes; never an executable neural fixture."""
    data = bytearray(8192)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<HH", data, 132, 0x8664, 1)
    struct.pack_into("<H", data, 148, 240)
    optional = 152
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into("<Q", data, optional + 24, 0x140000000)
    struct.pack_into("<I", data, optional + 60, 512)
    struct.pack_into("<I", data, optional + 108, 16)
    section = optional + 240
    struct.pack_into("<IIII", data, section + 8, len(data) - 512, 0x1000, len(data) - 512, 512)
    cursor, name_cursor = 512, 2048
    for names, directory, row_size in ((normal, 1, 20), (delay, 13, 32)):
        if not names:
            continue
        struct.pack_into(
            "<II", data, optional + 112 + directory * 8, cursor - 512 + 0x1000, row_size * (len(names) + 1)
        )
        for index, library in enumerate(names):
            encoded = library.encode("ascii") + b"\0"
            data[name_cursor : name_cursor + len(encoded)] = encoded
            row = cursor + index * row_size
            if directory == 1:
                struct.pack_into("<I", data, row + 12, name_cursor - 512 + 0x1000)
            else:
                struct.pack_into("<II", data, row, 1, name_cursor - 512 + 0x1000)
            name_cursor += len(encoded)
        cursor += row_size * (len(names) + 1)
    return bytes(data)


@pytest.fixture
def route(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    projector_root = tmp_path / "projector"
    projector_root.mkdir()
    (runtime / "encoder.exe").write_bytes(pe_bytes())
    system = tmp_path / "system32"
    system.mkdir()
    monkeypatch.setattr(vision, "_windows_system_directory", lambda: system)
    (projector_root / "mmproj.gguf").write_bytes(b"fixture projector")
    proof = []
    native_files = {role: (role + " recorded test-only evidence") for role in vision.REVIEWED_VISION_EVIDENCE}
    fixture_reviewed = {}
    for role in (*native_files, "configure_log", "build_log"):
        file = runtime / (role + ".txt")
        file.write_text(native_files.get(role, role), encoding="utf-8")
        record = {"role": role, "path": file.name, "sha256": vision.digest(file)}
        proof.append(record)
        if role in native_files:
            fixture_reviewed[role] = record["sha256"]
    # Synthetic compile receipts exercise strict manifest/budget rejection. The
    # production reviewed constants are separately checked against delivered
    # source patches; no real native build or model is claimed by this fixture.
    monkeypatch.setattr(vision, "REVIEWED_VISION_EVIDENCE", fixture_reviewed)
    build = {
        "schema": "omni-strata-vision-build-v1",
        "base_revision": vision.BASE_REVISION,
        "dependency_revision": vision.DEPENDENCY_REVISION,
        "dependency_tree": vision.DEPENDENCY_TREE,
        "encoder_sha256": vision.digest(runtime / "encoder.exe"),
        "native_observation_schema": "strata-vision-backend-v1",
        "configure_exit_code": 0,
        "build_exit_code": 0,
        "evidence_files": proof,
        "native_dependency_files": [],
        "pe_dependency_closure": vision.scan_pe_closure(runtime / "encoder.exe", runtime, [runtime / "encoder.exe"]),
        "cmake_options": {
            "STRATA_VISION_CUDA": "OFF",
            "STRATA_PORTABLE": "ON",
            "CMAKE_BUILD_TYPE": "Release",
        },
    }
    (runtime / "build.json").write_text(json.dumps(build), encoding="utf-8")
    artifacts = manifest(projector_root, [("mmproj.gguf", "weights")])
    projector = manifest(projector_root, [("mmproj.gguf", "vision_projector")], "fixture-projector")
    runtime_manifest = manifest(runtime, [(p.name, "runtime") for p in runtime.iterdir()])
    config = {
        "schema": "omni-strata-image-route-v1",
        "projector_root": str(projector_root),
        "projector_manifest": projector.to_dict(),
        "projector_file": "mmproj.gguf",
        "text_artifact_manifest_sha256": artifacts.manifest_sha256,
        "encoder_file": "encoder.exe",
        "encoder_build_manifest_file": "build.json",
        "embedding_width": 4,
        "max_image_bytes": 4096,
        "max_image_pixels": 16,
        "max_image_tokens": 4,
        "encoder_threads": 1,
        "allow_cpu_fallback": False,
        "encoder_device": "cpu",
        "vision_host_bytes": 16384,
        "vision_gpu_bytes": 0,
        "vision_scratch_bytes": 8192,
    }
    return config, runtime, runtime_manifest, set(runtime_manifest.verify(runtime)), artifacts


def verify_route(route):
    return vision.verify_vision_route(*route, manifest_type=tiers.ArtifactManifest)


def test_projector_encoder_sources_bound_and_budgets_truthful(route):
    verified = verify_route(route)
    identity = verified["identity"]
    assert identity["cache_policy"] == "disabled_reencode_each_image"
    assert identity["release_qualified"] is False
    assert identity["full_model_placement"] is None
    original = identity.copy()
    digest = original.pop("identity_sha256")
    assert digest == hashlib.sha256(vision.canonical(original).encode()).hexdigest()


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("text_artifact_manifest_sha256", "0" * 64, "another text"),
        ("projector_file", "../mmproj.gguf", None),
        ("vision_scratch_bytes", 1, "scratch"),
        ("max_image_tokens", 4097, "reviewed limit"),
        ("allow_cpu_fallback", "yes", "boolean"),
    ],
)
def test_projector_route_mismatch_or_missing_capacity_refused(route, field, value, reason):
    route[0][field] = value
    with pytest.raises((ValueError, FileNotFoundError), match=reason):
        verify_route(route)


def test_changed_projector_and_unbound_encoder_fail(route):
    projector = Path(route[0]["projector_root"]) / "mmproj.gguf"
    projector.write_bytes(b"different")
    with pytest.raises(ValueError):
        verify_route(route)
    route[3].remove(route[1] / "encoder.exe")
    with pytest.raises(ValueError, match="unbound"):
        verify_route(route)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dependency_revision", "wrong"),
        ("encoder_sha256", "0" * 64),
        ("configure_exit_code", 1),
        ("evidence_files", []),
    ],
)
def test_native_provenance_requires_actual_bound_recorded_bytes(route, field, value):
    path = route[1] / "build.json"
    build = json.loads(path.read_text())
    build[field] = value
    path.write_text(json.dumps(build), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_route(route)


NONCE, GENERATION = "a" * 32, "b" * 32
ENCODER = {"pid": 11, "creation_filetime_100ns": 22}
LANGUAGE = {"pid": 33, "creation_filetime_100ns": 44}
BOUNDS = {
    "max_image_bytes": 4096,
    "max_image_pixels": 16,
    "max_io_bytes": 8192,
    "embedding_width": 4,
    "max_image_tokens": 4,
}
BACKEND = {
    "schema": "strata-vision-backend-v1",
    "primary_backend": "CPU",
    "device_type": "CPU",
    "gpu_requested": False,
    "cpu_fallback_available": True,
}


def pair(tmp_path, *, cpu=True, allow=False, live=lambda _: True):
    observer = vision.EncoderObserver(NONCE, GENERATION, allow_cpu_fallback=allow, embedding_width=4, is_live=live)
    observer.language_identity = LANGUAGE.copy()
    frames = []

    def emit(frame):
        frames.append(frame)
        observer.ingest(frame)

    writer = vision.OwnedEncoderWriter(NONCE, GENERATION, 11, 22, emit, cwd=tmp_path, bounds=BOUNDS, contained=True)
    backend = BACKEND | ({"device_type": "CPU", "primary_backend": "CPU"} if cpu else {})
    assert writer.observe_output(vision.NATIVE_PREFIX + json.dumps(backend) + "\n") is True
    writer.observe_output("READY 4\n")
    return writer, observer, frames


def encoded(tmp_path, writer, observer, *, request="r1", epoch=1):
    image = png()
    (tmp_path / "x.png").write_bytes(image)
    observer.begin(request, epoch, image_input(image))
    writer.observe_input("ENC x.png x.sve\n")
    sve = struct.pack("<5i", 0x31455653, 4, 2, 2, 4) + struct.pack("<16f", *range(16))
    (tmp_path / "x.sve").write_bytes(sve)
    writer.observe_output("OK 4 2 2 1.5\n")
    return sve


def test_owned_image_to_exact_geni_is_required_and_recoverable_state_is_bounded(tmp_path):
    writer, observer, frames = pair(tmp_path)
    encoded(tmp_path, writer, observer)
    writer.observe_language_input(f"GENI 16 {tmp_path / 'x.sve'} 1,2,3\n", 33, 44)
    first = observer.finish("r1", 1, completed=True)
    assert first["status"] == "complete"
    assert first["language_dispatch"]["sve_sha256"] == first["encode_result"]["sve_sha256"]
    assert first["whole_model_placement"] is None and first["all_encoder_operators_gpu_verified"] is False
    first["owned_encoder"]["pid"] = 99
    assert observer.last["owned_encoder"]["pid"] == 11
    encoded(tmp_path, writer, observer, request="r2", epoch=2)
    writer.observe_language_input(f"GENI 16 {tmp_path / 'x.sve'} 1,2\n", 33, 44)
    assert observer.finish("r2", 2, completed=True)["native_encoder_request_seq"] == 2
    assert len(frames) == 9
    assert all(len(row.encode()) <= 4096 for row in frames)


@pytest.mark.parametrize("mutation", ["pid", "birth", "generation", "sequence", "role", "owner", "duplicate_json"])
def test_spoof_replay_and_restarted_owners_refused(tmp_path, mutation):
    writer, observer, frames = pair(tmp_path)
    raw = json.loads(frames[-1].split(NONCE + " ", 1)[1])
    raw["sequence"] = observer.sequence + 1
    if mutation == "pid":
        raw["pid"] = 99
    elif mutation == "birth":
        raw["creation_filetime_100ns"] += 1
    elif mutation == "generation":
        raw["generation"] = "c" * 32
    elif mutation == "sequence":
        raw["sequence"] -= 1
    elif mutation == "role":
        raw["role"] = "language"
    elif mutation == "owner":
        raw["event"], raw["payload"] = "owner", {"contained": True}
    line = vision.PREFIX + NONCE + " " + json.dumps(raw) + "\n"
    if mutation == "duplicate_json":
        line = line.replace('"pid": 11', '"pid": 11, "pid": 11')
    observer.ingest(line)
    assert observer.retired
    with pytest.raises(RuntimeError):
        observer.check()


def test_foreign_nonce_does_not_acquire_owner(tmp_path):
    writer, observer, frames = pair(tmp_path)
    observer.ingest(frames[0].replace(NONCE, "f" * 32))
    assert not observer.retired and observer.sequence == 3


def test_cuda_and_fallback_observers_are_disabled_until_physical_binding():
    for requested, fallback in (("cuda", False), ("cuda", True), ("cpu", True)):
        with pytest.raises(ValueError, match="only an explicit CPU"):
            vision.EncoderObserver(
                NONCE,
                GENERATION,
                allow_cpu_fallback=fallback,
                embedding_width=4,
                is_live=lambda _: True,
                requested_device=requested,
            )


def test_missing_geni_changed_sve_and_wrong_language_owner_refuse_completion(tmp_path):
    writer, observer, _ = pair(tmp_path)
    encoded(tmp_path, writer, observer)
    (tmp_path / "x.sve").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SVE differs"):
        writer.observe_language_input(f"GENI 16 {tmp_path / 'x.sve'} 1,2\n", 33, 44)
    assert observer.finish("r1", 1, completed=True)["status"] == "incomplete"
    writer, observer, _ = pair(tmp_path)
    encoded(tmp_path, writer, observer)
    writer.observe_language_input(f"GENI 16 {tmp_path / 'x.sve'} 1,2\n", 99, 44)
    assert observer.retired and "GENI_dispatch_owner_or_SVE_mismatch" in observer.reasons


def test_encoder_wrong_token_shape_or_ready_width_fails_before_language(tmp_path):
    writer, observer, _ = pair(tmp_path)
    (tmp_path / "x.png").write_bytes(png())
    observer.begin("r1", 1, image_input())
    writer.observe_input("ENC x.png x.sve\n")
    (tmp_path / "x.sve").write_bytes(b"bad")
    for line in ("OK 8 4 2 1.0\n", "OK 4 3 2 1.0\n", "OK 4 2 2 nan\n", "OK 4 2 2 1.0\n"):
        with pytest.raises(ValueError):
            writer.observe_output(line)
    with pytest.raises(ValueError):
        writer.observe_output("READY 4\n")


@pytest.mark.skipif(os.name != "nt", reason="exact owned FILETIME test uses Windows")
def test_actual_mock_encoder_death_is_not_health_images_true(tmp_path):
    child = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", "import time;time.sleep(30)"], creationflags=subprocess.CREATE_NO_WINDOW
    )
    try:
        birth = base._windows_process_creation_filetime(child.pid)
        writer, observer, frames = pair(
            tmp_path,
            live=lambda identity: base._windows_process_creation_filetime(identity["pid"])
            == identity["creation_filetime_100ns"],
        )
        observer.identity = {"pid": child.pid, "creation_filetime_100ns": birth}
        observer.check()
        child.kill()
        child.wait(timeout=5)
        with pytest.raises(RuntimeError):
            observer.check()
        assert observer.retired
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="exact owned FILETIME and Windows job semantics")
@pytest.mark.parametrize("unknown_retirement", [False, True])
def test_stage_cancel_retires_real_mock_encoder_and_preserves_incomplete_observation(tmp_path, unknown_retirement):
    child = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", "import time;time.sleep(30)"], creationflags=subprocess.CREATE_NO_WINDOW
    )
    writer, observer, _ = pair(tmp_path)
    observer.identity = {
        "pid": child.pid,
        "creation_filetime_100ns": base._windows_process_creation_filetime(child.pid),
    }
    observer.is_live = multimodal.StrataMultimodalStageClient._encoder_is_live
    observer.begin("cancel", 1, image_input())
    stage = object.__new__(multimodal.StrataMultimodalStageClient)
    stage._encoder_observer = observer
    stage._encoder_process_owner = (
        types.SimpleNamespace(identity=observer.identity.copy(), close_retired=lambda: False)
        if unknown_retirement
        else vision.OwnedWindowsProcess(observer.identity)
    )
    stage._proc, stage._active, stage._epoch, stage._generation = child, "cancel", 1, GENERATION
    stage._cancel, stage._retirement_lock = threading.Event(), threading.Lock()
    stage._known_children, stage._temporary, stage._diagnostics = {}, None, None
    stage._agent_stream, stage._task, stage._output, stage._ack_pending = None, None, None, None
    stage._observation_runtime = None
    stage._io_request = StageRequest("cancel", 0, 1, GENERATION)
    stage._io_report_lock = threading.RLock()
    stage._last_io_report = None
    releases = []
    stage._ledger = types.SimpleNamespace(release=lambda reservation, **kw: releases.append(kw))
    stage._reservation = object()
    try:
        asyncio.run(stage.abort_requests_async(["cancel"]))
        assert child.poll() is not None and stage._closed and stage._epoch == 2
        assert releases == [{"drained": not unknown_retirement}]
        report = stage.last_image_observation()
        assert report["status"] == "incomplete" and "request_cancelled" in report["reasons"]
        assert report["physical_ssd_read_bytes"] is None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_composite_bootstrap_preserves_text_literal_and_has_separate_identity():
    original_tree = ast.parse((PRODUCTION / "vllm_omni/engine/backends/strata.py").read_text())
    original = next(
        node.value.value
        for node in original_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_BOOTSTRAP" for target in node.targets)
    )
    assert base._BOOTSTRAP == original
    assert (
        hashlib.sha256(original.encode()).hexdigest()
        == "ad1e4b4a7b530bebc63bc991b4f0f0874bd452e690085bb77b149859c53df30e"
    )
    combined = vision.bootstrap_source(original)
    ast.parse(combined)
    assert "_v_owned_encoder.observe_language_input(text, p.pid, created)" in combined
    assert "_v_class.encode = _v_uncached_encode" in combined
    assert "_v_class.work_dir = staticmethod" in combined
    assert "os.path.dirname(_v_scratch) != os.path.realpath(os.path.dirname(__file__))" in combined
    assert hashlib.sha256(combined.encode()).hexdigest() != hashlib.sha256(original.encode()).hexdigest()


def test_text_route_controls_and_payload_hooks_remain_identical():
    stage = object.__new__(base.StrataTextStageClient)
    assert stage._backend_name == "external.strata.text.v1" and stage._supports_images is False
    assert stage._bootstrap_source() == base._BOOTSTRAP
    assert stage._http_content("text") == "text" and stage._extension_drained() is True
    assert stage._prompt_fields() == {"text", "max_tokens", "temperature", "stream_agent"}
    with pytest.raises(ValueError, match="text stage refuses"):
        stage._prepare_extension({"image_route": {}}, *([None] * 8))
    assert (
        multimodal.StrataMultimodalStageClient.abort_requests_async is base.StrataTextStageClient.abort_requests_async
    )
    assert multimodal.StrataMultimodalStageClient.acknowledge is base.StrataTextStageClient.acknowledge


def test_multimodal_openai_payload_contains_only_validated_inline_image():
    stage = object.__new__(multimodal.StrataMultimodalStageClient)
    stage._vision = {"config": BOUNDS}
    stage._max_io_bytes = 8192
    stage._prepare_extension_prompt({"text": "Describe", "image_data_url": url(png())})
    content = stage._http_content("Describe")
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert stage._image_input["sha256"] == hashlib.sha256(png()).hexdigest()
    stage._prepare_extension_prompt({"text": "Followup"})
    assert stage._http_content("Followup") == "Followup" and stage._image_input is None


@pytest.mark.skipif(os.name != "nt", reason="fixture ownership uses exact Windows creation time")
def test_pinned_upstream_vision_protocol_against_actual_mock_child(tmp_path):
    """Real Vision class/ENC/SVE protocol; mock child is explicitly not neural evidence."""
    source_root = os.environ.get("OMNI_STRATA_PINNED_SOURCE_ROOT")
    if not source_root:
        pytest.skip("set pinned source root to exercise upstream protocol fixture")
    server_path = Path(source_root) / "serve/server.py"
    tree = ast.parse(server_path.read_text(encoding="utf-8"))
    vision_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Vision")
    frames, writers, children = [], [], []
    observer = vision.EncoderObserver(
        NONCE,
        GENERATION,
        allow_cpu_fallback=False,
        embedding_width=4,
        is_live=multimodal.StrataMultimodalStageClient._encoder_is_live,
    )
    observer.language_identity = LANGUAGE.copy()
    child_code = """import json,sys,struct
record=dict(schema='strata-vision-backend-v1',primary_backend='CPU',device_type='CPU',
            gpu_requested=False,cpu_fallback_available=True)
print('OMNI_VISION_V1 '+json.dumps(record),flush=True)
print('READY 4',flush=True)
for line in sys.stdin:
 if line.strip()=='QUIT':break
 fields=line.strip().split(' ')
 if fields[0]=='ENC':
  open(fields[2],'wb').write(struct.pack('<5i',0x31455653,4,2,2,4)+struct.pack('<16f',*range(16)))
  print('OK 4 2 2 1.5',flush=True)
"""

    def mock_popen(what, args, **kw):
        assert what == "the image encoder" and "--max-tokens" in args and "--gpu" not in args
        child = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", child_code], creationflags=subprocess.CREATE_NO_WINDOW, **kw
        )
        children.append(child)
        writer = vision.OwnedEncoderWriter(
            NONCE,
            GENERATION,
            child.pid,
            base._windows_process_creation_filetime(child.pid),
            lambda row: (frames.append(row), observer.ingest(row)),
            cwd=kw["cwd"],
            bounds=BOUNDS,
            contained=True,
        )
        writers.append(writer)

        class Input:
            def __init__(self, pipe):
                self.pipe = pipe

            def write(self, line):
                writer.observe_input(line)
                return self.pipe.write(line)

            def __getattr__(self, name):
                return getattr(self.pipe, name)

        class Output:
            def __init__(self, pipe):
                self.pipe = pipe

            def readline(self):
                while True:
                    line = self.pipe.readline(4097)
                    if not writer.observe_output(line):
                        return line

            def __getattr__(self, name):
                return getattr(self.pipe, name)

        child.stdin, child.stdout = Input(child.stdin), Output(child.stdout)
        return child

    namespace = dict(
        Path=Path,
        os=os,
        subprocess=subprocess,
        tempfile=tempfile,
        threading=threading,
        queue=queue,
        shutil=shutil,
        base64=base64,
        hashlib=hashlib,
        re=__import__("re"),
        popen=mock_popen,
        contain=lambda _: True,
        VISION_READY_S=5,
        VISION_ENCODE_S=5,
    )
    exec(compile(ast.Module(body=[vision_class], type_ignores=[]), str(server_path), "exec"), namespace)
    upstream = namespace["Vision"](
        {
            "exe": "mock-native-encoder",
            "mmproj": "fixture.gguf",
            "model": "fixture-text.gguf",
            "gpu": False,
            "threads": 1,
            "max_tokens": 4,
        }
    )
    try:
        observer.check()
        observer.begin("actual-protocol-fixture", 1, image_input())
        path, tokens = upstream.encode(url(png()))
        assert tokens == 4 and path.stat().st_size == 84
        writers[0].observe_language_input(f"GENI 16 {path} 1,2,3\n", 33, 44)
        report = observer.finish("actual-protocol-fixture", 1, completed=True)
        assert report["status"] == "complete" and report["native_encoder_request_seq"] == 1
        assert report["encode_result"]["sve_sha256"] == vision.digest(path)
        assert len(frames) == 6 and report["release_qualified"] is False
    finally:
        upstream.shutdown()
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_explicit_cpu_encoder_route_has_no_gpu_allocation_or_implicit_fallback(route):
    route[0]["encoder_device"] = "cpu"
    route[0]["vision_gpu_bytes"] = 0
    path = route[1] / "build.json"
    build = json.loads(path.read_text())
    build["cmake_options"] = {"STRATA_VISION_CUDA": "OFF", "STRATA_PORTABLE": "ON", "CMAKE_BUILD_TYPE": "Release"}
    path.write_text(json.dumps(build), encoding="utf-8")
    assert verify_route(route)["identity"]["encoder_device"] == "cpu"
    stage = object.__new__(multimodal.StrataMultimodalStageClient)
    stage._vision = verify_route(route) | {"text_first_shard": "text.gguf"}
    config = {"args": []}
    stage._configure_extension(config)
    assert config["vision"]["gpu"] is False and config["args"] == ["--vision"]
    route[0]["vision_gpu_bytes"] = 1
    with pytest.raises(ValueError, match="GPU claim must be zero"):
        verify_route(route)


def test_cpu_request_requires_actual_cpu_backend_even_if_gpu_is_available(tmp_path):
    frames = []
    observer = vision.EncoderObserver(
        NONCE, GENERATION, allow_cpu_fallback=False, embedding_width=4, is_live=lambda _: True, requested_device="cpu"
    )
    writer = vision.OwnedEncoderWriter(
        NONCE,
        GENERATION,
        11,
        22,
        lambda row: (frames.append(row), observer.ingest(row)),
        cwd=tmp_path,
        bounds=BOUNDS,
        contained=True,
    )
    writer.observe_output(
        vision.NATIVE_PREFIX
        + json.dumps(BACKEND | {"gpu_requested": False, "primary_backend": "CPU", "device_type": "CPU"})
        + "\n"
    )
    writer.observe_output("READY 4\n")
    observer.check()
    assert observer.backend["gpu_requested"] is False


@pytest.mark.skipif(os.name != "nt", reason="selected modules require Windows process observation")
def test_generalized_selected_module_audit_binds_actual_mock_exe_and_rejects_stale_owner(tmp_path):
    io_module = direct_module("private_strata_io_audit", BACKENDS / "strata_io.py")
    python = Path(sys.executable).resolve()
    child = subprocess.Popen(
        [str(python), "-I", "-B", "-c", "import time;print('READY',flush=True);time.sleep(30)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout.readline() == "READY\n"
    try:
        owner = {"pid": child.pid, "creation_filetime_100ns": base._windows_process_creation_filetime(child.pid)}
        identity = {"native_executable_sha256": vision.digest(python), "native_dependency_files": []}
        report = io_module.audit_selected_loaded_modules(
            owner, python.parent, identity, native_file=python.name, include_cuda_driver=False, role="encoder"
        )
        assert report["status"] == "verified" and report["role"] == "encoder", report
        assert report["pid"] == child.pid and report["modules"][0]["sha256"] == identity["native_executable_sha256"]
        bad = io_module.audit_selected_loaded_modules(
            owner | {"creation_filetime_100ns": owner["creation_filetime_100ns"] + 1},
            python.parent,
            identity,
            native_file=python.name,
            include_cuda_driver=False,
            role="encoder",
        )
        assert bad["status"] == "unverified"
        wrong = io_module.audit_selected_loaded_modules(
            owner,
            python.parent,
            identity | {"native_executable_sha256": "0" * 64},
            native_file=python.name,
            include_cuda_driver=False,
            role="encoder",
        )
        assert wrong["status"] == "unverified"
    finally:
        child.kill()
        child.wait(timeout=5)


def test_text_followup_never_reuses_prior_image_observation():
    stage = object.__new__(multimodal.StrataMultimodalStageClient)
    stage._image_input = None
    stage._encoder_observer = types.SimpleNamespace(last={"request_id": "old-image"}, _lock=threading.RLock())
    telemetry = {}
    stage._extension_telemetry(StageRequest("new-text", 0, 2, GENERATION), telemetry)
    assert telemetry["image_observation"] is None


def test_image_completion_cannot_borrow_prior_request_terminal():
    stage = object.__new__(multimodal.StrataMultimodalStageClient)
    stage._image_input = {"sha256": "0" * 64}
    stage._encoder_observer = types.SimpleNamespace(
        last={"status": "complete", "request_id": "old-image", "epoch": 1, "generation": GENERATION},
        _lock=threading.RLock(),
    )
    with pytest.raises(RuntimeError, match="complete owned encoder"):
        stage._extension_telemetry(StageRequest("new-image", 0, 2, GENERATION), {})


def test_default_reviewed_native_patch_constants_match_delivered_patch_bytes():
    patch_root = ROOT / "benchmarks/edge_harness/runtime_patches"
    assert (
        vision.digest(patch_root / "strata_vision_backend_v1.patch") == vision.REVIEWED_VISION_EVIDENCE["strata_patch"]
    )
    assert (
        vision.digest(patch_root / "strata_vision_clip_v1.patch") == vision.REVIEWED_VISION_EVIDENCE["dependency_patch"]
    )


def test_wrong_reviewed_native_source_cannot_be_registered(route, monkeypatch):
    monkeypatch.setattr(vision, "REVIEWED_VISION_EVIDENCE", {"patched_vision_source": "0" * 64})
    with pytest.raises(ValueError, match="reviewed prototype"):
        verify_route(route)


def test_actual_pe_normal_delay_recursive_closure_is_bound_and_load_time_scoped(tmp_path, monkeypatch):
    runtime, system = tmp_path / "runtime", tmp_path / "System32"
    runtime.mkdir()
    system.mkdir()
    monkeypatch.setattr(vision, "_windows_system_directory", lambda: system)
    (system / "kernel32.dll").write_bytes(b"OS path fixture only")
    (runtime / "encoder.exe").write_bytes(pe_bytes(("app.dll", "KERNEL32.dll"), ("later.dll",)))
    (runtime / "app.dll").write_bytes(pe_bytes(("transitive.dll",)))
    (runtime / "transitive.dll").write_bytes(pe_bytes())
    (runtime / "later.dll").write_bytes(pe_bytes(("latechild.dll",)))
    (runtime / "latechild.dll").write_bytes(pe_bytes())
    verified = list(runtime.iterdir())
    closure = vision.scan_pe_closure(runtime / "encoder.exe", runtime, verified)
    assert closure["required_at_load"] == ["app.dll", "encoder.exe", "transitive.dll"]
    assert len(closure["files"]) == 5 and closure["all_dynamic_loads_covered"] is False
    root = next(row for row in closure["files"] if row["path"] == "encoder.exe")
    assert root["system"] == ["kernel32.dll"] and root["delay"] == ["later.dll"]
    preimage = closure.copy()
    sha = preimage.pop("identity_sha256")
    assert sha == hashlib.sha256(vision.canonical(preimage).encode()).hexdigest()
    verified.remove(runtime / "latechild.dll")
    with pytest.raises(ValueError, match="unbound"):
        vision.scan_pe_closure(runtime / "encoder.exe", runtime, verified)


@pytest.mark.parametrize("malformed", [b"x" * 128, b"MZ" + b"\0" * 126])
def test_pe_closure_refuses_invalid_executable_bytes(tmp_path, malformed):
    file = tmp_path / "bad.exe"
    file.write_bytes(malformed)
    with pytest.raises(ValueError):
        vision.pe_imports(file)


def test_missing_pe_proof_is_not_an_empty_verified_dependency_closure(route):
    file = route[1] / "build.json"
    build = json.loads(file.read_text())
    build.pop("pe_dependency_closure")
    file.write_text(json.dumps(build), encoding="utf-8")
    with pytest.raises(ValueError, match="PE normal/delay closure"):
        verify_route(route)


def test_empty_declared_dlls_refused_when_actual_pe_requires_non_system(route):
    runtime = route[1]
    (runtime / "encoder.exe").write_bytes(pe_bytes(("needed.dll",)))
    (runtime / "needed.dll").write_bytes(pe_bytes())
    route[3].add(runtime / "needed.dll")
    file = runtime / "build.json"
    build = json.loads(file.read_text())
    build["encoder_sha256"] = vision.digest(runtime / "encoder.exe")
    build["pe_dependency_closure"] = vision.scan_pe_closure(runtime / "encoder.exe", runtime, route[3])
    file.write_text(json.dumps(build), encoding="utf-8")
    with pytest.raises(ValueError, match="do not equal actual recursive"):
        verify_route(route)


def cuda_config(route, host):
    route[0].update(encoder_device="cuda", allow_cpu_fallback=True, vision_gpu_bytes=1 << 20, vision_host_bytes=host)
    file = route[1] / "build.json"
    build = json.loads(file.read_text())
    build["cmake_options"] = dict(
        STRATA_VISION_CUDA="ON", STRATA_PORTABLE="ON", CMAKE_BUILD_TYPE="Release", CMAKE_CUDA_ARCHITECTURES="120"
    )
    file.write_text(json.dumps(build), encoding="utf-8")


def test_cuda_startup_warmup_cannot_borrow_tiny_input_bounds(route):
    cuda_config(route, 16384)
    with pytest.raises(ValueError, match="decode/preprocessing"):
        verify_route(route)


def test_cuda_fallback_needs_simultaneous_cpu_projector_budget_and_remains_disabled(route):
    base = 2048 * 2048 * 35 + 2 * route[0]["max_image_bytes"]
    projector_size = len(b"fixture projector")
    cuda_config(route, base + projector_size - 1)
    with pytest.raises(ValueError, match="simultaneous host"):
        verify_route(route)
    route[0]["vision_host_bytes"] += 1
    with pytest.raises(ValueError, match="CUDA encoder route disabled"):
        verify_route(route)


def test_cpu_observer_refuses_generic_gpu_or_requested_cuda_frames(tmp_path):
    for gpu_requested, device, primary in ((False, "GPU", "Vulkan0"), (True, "GPU", "CUDA0")):
        observer = vision.EncoderObserver(
            NONCE, GENERATION, allow_cpu_fallback=False, embedding_width=4, is_live=lambda _: True
        )
        writer = vision.OwnedEncoderWriter(
            NONCE, GENERATION, 11, 22, observer.ingest, cwd=tmp_path, bounds=BOUNDS, contained=True
        )
        writer.observe_output(
            vision.NATIVE_PREFIX
            + json.dumps(BACKEND | dict(gpu_requested=gpu_requested, device_type=device, primary_backend=primary))
            + "\n"
        )
        assert observer.retired


def test_unknown_encoder_retirement_quarantines_instead_of_using_false_liveness(monkeypatch):
    stage = object.__new__(multimodal.StrataMultimodalStageClient)
    stage._encoder_observer = types.SimpleNamespace(identity=ENCODER.copy())
    stage._encoder_process_owner = types.SimpleNamespace(identity=ENCODER.copy(), close_retired=lambda: False)
    monkeypatch.setattr(base, "_windows_process_creation_filetime", lambda _: None)
    assert stage._encoder_is_live(ENCODER) is False
    assert stage._extension_drained() is False
    stage._encoder_process_owner.identity = LANGUAGE.copy()
    assert stage._extension_drained() is False
    stage._encoder_process_owner = None
    monkeypatch.setattr(
        vision, "OwnedWindowsProcess", lambda _: (_ for _ in ()).throw(PermissionError("access denied"))
    )
    assert stage._extension_drained() is False


@pytest.mark.skipif(os.name != "nt", reason="retained Windows process object")
def test_owned_process_handle_proves_exit_and_query_failure_remains_unknown():
    child = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", "import time;time.sleep(30)"], creationflags=subprocess.CREATE_NO_WINDOW
    )
    owner = vision.OwnedWindowsProcess(
        dict(pid=child.pid, creation_filetime_100ns=base._windows_process_creation_filetime(child.pid))
    )
    try:
        assert owner.state() == "alive" and owner.close_retired() is False
        kernel = owner._kernel
        owner._kernel = types.SimpleNamespace(GetProcessTimes=lambda *_: 0)
        assert owner.state() == "unknown" and owner.close_retired() is False
        owner._kernel = kernel
        child.kill()
        child.wait(timeout=5)
        assert owner.state() == "retired" and owner.close_retired() is True
        assert owner.state() == "retired"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        owner.close_retired()


@pytest.mark.skipif(os.name != "nt", reason="actual Windows module reconciliation")
def test_actual_non_system_module_cannot_hide_behind_empty_selected_dependency_list():
    io_module = direct_module("private_strata_io_complete_audit", BACKENDS / "strata_io.py")
    python = Path(sys._base_executable).resolve()  # venv launcher may otherwise own a separate Python child
    child = subprocess.Popen(
        [str(python), "-I", "-B", "-c", "import time;print('READY',flush=True);time.sleep(30)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout.readline() == "READY\n"
    try:
        owner = dict(pid=child.pid, creation_filetime_100ns=base._windows_process_creation_filetime(child.pid))
        identity = dict(native_executable_sha256=vision.digest(python), native_dependency_files=[])
        # Deliberately incomplete closure exercises live reconciliation, not PE
        # verification: real Python also loads non-System32 python312.dll.
        closure = dict(
            schema="omni-strata-pe-closure-v1",
            files=[
                dict(path=python.name, size_bytes=python.stat().st_size, sha256=identity["native_executable_sha256"])
            ],
            required_at_load=[python.name.lower()],
            all_dynamic_loads_covered=False,
            system_dependencies_pinned=False,
        )
        closure["identity_sha256"] = hashlib.sha256(vision.canonical(closure).encode()).hexdigest()
        report = io_module.audit_selected_loaded_modules(
            owner,
            python.parent,
            identity,
            native_file=python.name,
            include_cuda_driver=False,
            role="encoder",
            non_system_pe_closure=closure,
        )
        assert report["status"] == "unverified"
        assert "absent from the bound PE closure" in report["reasons"][0]
    finally:
        child.kill()
        child.wait(timeout=5)


@pytest.fixture
def loaded_image_plan(route, monkeypatch):
    import copy

    io_module = direct_module("private_strata_io_validator", BACKENDS / "strata_io.py")
    monkeypatch.setattr(base, "strata_io", io_module)
    monkeypatch.setattr(multimodal, "_typed_tier_plan", tiers.WeightTierPlan.from_dict)
    image = verify_route(route)["identity"]
    image.pop("identity_sha256")
    image.update(
        vision_bootstrap_sha256=hashlib.sha256(vision.bootstrap_source(base._BOOTSTRAP).encode()).hexdigest(),
        vision_adapter_sha256=vision.digest(BACKENDS / "strata_vision.py"),
        base_text_bootstrap_sha256=hashlib.sha256(base._BOOTSTRAP.encode()).hexdigest(),
        prepared_artifact_size_bytes=100,
    )
    image["identity_sha256"] = hashlib.sha256(vision.canonical(image).encode()).hexdigest()
    observation = dict(
        schema="omni-strata-observed-runtime-v1",
        base_revision=base.PINNED_STRATA_REVISION,
        dependency_revision=vision.DEPENDENCY_REVISION,
        dependency_tree=vision.DEPENDENCY_TREE,
        native_executable_sha256="1" * 64,
        patch_sha256="2" * 64,
        build_receipt_sha256="3" * 64,
        dependency_provenance_sha256="4" * 64,
        runtime_dependencies_sha256="5" * 64,
        supervisor_bootstrap_sha256=image["base_text_bootstrap_sha256"],
        io_adapter_sha256=vision.digest(BACKENDS / "strata_io.py"),
        native_io_schema="strata-omni-io-v1",
        native_dll_search_policy="isolated_engine_and_system32",
        three_tier_memory_qualified=False,
        patch_source_hashes={
            path: dict(base_sha256="0" * 64, patched_sha256="1" * 64) for path in io_module.PATCH_SOURCES
        },
        native_dependency_files=[dict(path="engine/test.dll")],
    )
    observation["identity_sha256"] = hashlib.sha256(vision.canonical(observation).encode()).hexdigest()
    controls = dict(
        image_route=image,
        observation_runtime=observation,
        runtime_manifest_sha256=image["runtime_manifest_sha256"],
        artifact_manifest_sha256=image["text_artifact_manifest_sha256"],
    )
    host = image["bounds_and_declared_budgets"]["vision_host_bytes"]
    scratch = image["bounds_and_declared_budgets"]["vision_scratch_bytes"]
    budget = tiers.WeightTierBudget(
        host_workspace_bytes=host + scratch,
        ssd_temporary_bytes=scratch,
        ssd_artifact_bytes=sum(
            image[key]
            for key in (
                "text_artifact_size_bytes",
                "prepared_artifact_size_bytes",
                "runtime_artifact_size_bytes",
                "projector_size_bytes",
            )
        ),
        gpu_weights_bytes=1024,
        host_loading_peak_bytes=host + scratch,
    )
    tier = tiers.WeightTierPlan(
        "test-image",
        image["text_artifact_manifest_sha256"],
        multimodal.BACKEND_NAME,
        base.PINNED_STRATA_REVISION,
        budget,
    )
    memory = dict(gpu_name_sha256="a" * 64)
    info = dict(
        engine=base.PINNED_STRATA_VERSION,
        context=4096,
        kv="fp16",
        spec=2,
        lookup=0,
        conversation_cache_mib=0,
        conversation_cache_slots=0,
        pool_workers=1,
        expert_slots=1,
        expert_cache_mib=1,
        arena_mib=1,
    )
    pool = dict(workers=1, host_thread=True, tasks_per_phase=1, participating_threads=2)
    evidence = dict(
        gpu=dict(local_index=0, name_sha256="a" * 64),
        cpu_pool=pool,
        expert_workers=pool,
        native_pack=True,
        native_starts=1,
    )
    proof = base._verify_load_configuration(evidence, info, memory, gpu=0, context=4096, kv="fp16", verify_window=2)
    plan = dict(
        backend=multimodal.BACKEND_NAME,
        runtime_revision=base.PINNED_STRATA_REVISION,
        requested_device="cpu+cuda:0",
        verified_execution_configuration="cpu+cuda:0",
        placement_evidence_level="native_loaded_configuration",
        execution_configuration_evidence=proof,
        fresh_memory_admission=memory,
        context_tokens=4096,
        max_new_tokens=64,
        kv_type="fp16",
        native_verify_window=2,
        image_route=image,
        route_controls=controls,
        route_controls_sha256=hashlib.sha256(vision.canonical(controls).encode()).hexdigest(),
        runtime_manifest_sha256=image["runtime_manifest_sha256"],
        artifact_manifest_sha256=image["text_artifact_manifest_sha256"],
        supervisor_bootstrap_sha256=image["vision_bootstrap_sha256"],
        observation_runtime=observation,
        owned_encoder_at_load=ENCODER.copy(),
        selected_encoder_module_audit_at_load=dict(
            schema="omni-strata-selected-module-audit-v1",
            status="verified",
            role="encoder",
            **ENCODER,
            observed_non_system_module_closure_verified=True,
            pe_closure_identity_sha256=image["encoder_pe_closure_sha256"],
            modules=[dict(sha256=image["encoder_sha256"])],
        ),
        selected_native_module_audit_at_load=dict(
            schema="omni-strata-selected-module-audit-v1",
            status="verified",
            modules=[dict(sha256=observation["native_executable_sha256"])],
            **LANGUAGE,
        ),
        encoder_selected_loaded_modules_verified=True,
        encoder_observed_non_system_module_closure_verified_at_load=True,
        encoder_backend_selection_at_load=BACKEND.copy(),
        weight_tier_plan=tier.to_dict(),
        host_overhead_bytes=host + scratch,
        reserved_bytes=budget.resource_demands(gpu_pool="vram:0"),
        gpu_pool="vram:0",
        declared_modalities=["text", "image"],
        qualified_modalities=[],
        encoder_all_operators_placement=None,
        observed_model_placement=None,
        three_tier_memory_qualified=False,
        request_capacity=1,
        batch_size=1,
        spec_tokens=0,
        qualification="experimental_image_backend_loaded_not_neural_qualified",
        image_resource_budget_scope="declared_encoder_and_text_coexistence_not_hard_caps",
    )
    return copy.deepcopy(plan)


def test_distinct_multimodal_load_gate_and_strict_public_text_gate(loaded_image_plan):
    import copy

    before = copy.deepcopy(loaded_image_plan)
    multimodal.validate_strata_multimodal_load_plan(loaded_image_plan, "cpu+cuda:0")
    assert loaded_image_plan == before
    with pytest.raises(RuntimeError):
        base.validate_strata_load_plan(loaded_image_plan, "cpu+cuda:0")
    loaded_image_plan["backend"] = base.BACKEND_NAME
    with pytest.raises(RuntimeError):
        base.validate_strata_load_plan(loaded_image_plan, "cpu+cuda:0")
    with pytest.raises(RuntimeError):
        multimodal.validate_strata_multimodal_load_plan(loaded_image_plan, "cpu+cuda:0")
    loaded_image_plan.pop("image_route")
    loaded_image_plan.pop("owned_encoder_at_load")
    loaded_image_plan["declared_modalities"] = ["text"]
    loaded_image_plan["route_controls"].pop("image_route")
    base.validate_strata_load_plan(loaded_image_plan, "cpu+cuda:0")


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_owner",
        "same_owner",
        "role",
        "closure",
        "backend_cpu",
        "cuda",
        "image_digest",
        "controls_digest",
        "runtime",
        "empty_runtime",
        "host_workspace",
        "host_overhead",
        "ssd",
        "reserved",
        "qualified",
        "gpu_claim",
        "context",
        "source_adapter",
        "false_module",
        "batch",
    ],
)
def test_multimodal_load_gate_refuses_tampered_missing_or_broadened_evidence(loaded_image_plan, mutation):
    p = loaded_image_plan
    if mutation == "unknown_owner":
        p["owned_encoder_at_load"]["creation_filetime_100ns"] = None
    elif mutation == "same_owner":
        p["selected_native_module_audit_at_load"]["pid"] = p["owned_encoder_at_load"]["pid"]
    elif mutation == "role":
        p["selected_encoder_module_audit_at_load"]["role"] = "language"
    elif mutation == "closure":
        p["selected_encoder_module_audit_at_load"]["observed_non_system_module_closure_verified"] = False
    elif mutation == "backend_cpu":
        p["encoder_backend_selection_at_load"]["device_type"] = "GPU"
    elif mutation == "cuda":
        p["image_route"]["encoder_device"] = "cuda"
    elif mutation == "image_digest":
        p["image_route"]["projector_sha256"] = "0" * 64
    elif mutation == "controls_digest":
        p["route_controls_sha256"] = "0" * 64
    elif mutation == "runtime":
        p["observation_runtime"] = False
    elif mutation == "empty_runtime":
        p["observation_runtime"] = {}
    elif mutation == "host_workspace":
        p["weight_tier_plan"]["budget"]["host_workspace_bytes"] = 1
    elif mutation == "host_overhead":
        p["host_overhead_bytes"] = 1
    elif mutation == "ssd":
        p["weight_tier_plan"]["budget"]["ssd_artifact_bytes"] = 1
    elif mutation == "reserved":
        p["reserved_bytes"]["host_ram"] += 1
    elif mutation == "qualified":
        p["qualified_modalities"] = ["image"]
    elif mutation == "gpu_claim":
        p["observed_model_placement"] = "cpu+cuda:0"
    elif mutation == "context":
        p["max_new_tokens"] = 4096
    elif mutation == "source_adapter":
        p["image_route"]["vision_adapter_sha256"] = "0" * 64
    elif mutation == "false_module":
        p["selected_encoder_module_audit_at_load"]["status"] = "unverified"
    elif mutation == "batch":
        p["batch_size"] = 2
    if mutation in {"cuda", "source_adapter"}:
        image = p["image_route"]
        preimage = dict(image)
        preimage.pop("identity_sha256")
        image["identity_sha256"] = hashlib.sha256(vision.canonical(preimage).encode()).hexdigest()
        p["route_controls_sha256"] = hashlib.sha256(vision.canonical(p["route_controls"]).encode()).hexdigest()
    with pytest.raises(RuntimeError, match="experimental CPU image"):
        multimodal.validate_strata_multimodal_load_plan(p, "cpu+cuda:0")


@pytest.mark.skipif(os.name != "nt", reason="actual Windows complete selected appdir closure")
def test_actual_mock_process_full_pe_closure_and_all_observed_non_system_modules_reconcile():
    io_module = direct_module("private_strata_io_positive_closure", BACKENDS / "strata_io.py")
    python = Path(sys._base_executable).resolve()
    files = [python, *python.parent.glob("*.dll")]
    closure = vision.scan_pe_closure(python, python.parent, files)
    deps = [
        {key: row[key] for key in ("path", "sha256", "size_bytes")}
        for row in closure["files"]
        if row["path"] != python.name
    ]
    assert deps, "actual interpreter fixture should import a non-system Python DLL"
    child = subprocess.Popen(
        [str(python), "-I", "-B", "-c", "import time;print('READY',flush=True);time.sleep(30)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout.readline() == "READY\n"
    try:
        owner = dict(pid=child.pid, creation_filetime_100ns=base._windows_process_creation_filetime(child.pid))
        identity = dict(native_executable_sha256=vision.digest(python), native_dependency_files=deps)
        report = io_module.audit_selected_loaded_modules(
            owner,
            python.parent,
            identity,
            native_file=python.name,
            include_cuda_driver=False,
            role="encoder",
            non_system_pe_closure=closure,
        )
        assert report["status"] == "verified", report
        assert report["observed_non_system_module_closure_verified"] is True
        assert report["all_future_dynamic_loads_covered"] is False
        assert report["pe_closure_identity_sha256"] == closure["identity_sha256"]
    finally:
        child.kill()
        child.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows OpenProcess error semantics")
@pytest.mark.parametrize("error,state", [(5, "unknown"), (6, "unknown"), (87, "retired")])
def test_open_process_denial_is_unknown_and_only_missing_pid_error_proves_retirement(monkeypatch, error, state):
    import ctypes

    class Function:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    def denied(*_):
        ctypes.set_last_error(error)
        return 0

    kernel = types.SimpleNamespace(
        OpenProcess=Function(denied),
        GetProcessTimes=Function(lambda *_: 0),
        WaitForSingleObject=Function(lambda *_: 0xFFFFFFFF),
        GetExitCodeProcess=Function(lambda *_: 0),
        CloseHandle=Function(lambda *_: 1),
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kw: kernel)
    owner = vision.OwnedWindowsProcess(ENCODER)
    assert owner.state() == state
    assert owner.close_retired() is (state == "retired")
    with pytest.raises(ValueError, match="DWORD"):
        vision.OwnedWindowsProcess(ENCODER | {"pid": 2**32 + 1})


@pytest.mark.parametrize("mutation", ["flattened", "missing_path", "extra_path", "wrong_fields", "bad_digest"])
def test_real_shaped_patch_source_lineage_is_required_even_when_resigned(loaded_image_plan, mutation):
    observation = loaded_image_plan["observation_runtime"]
    hashes = observation["patch_source_hashes"]
    first = next(iter(hashes))
    if mutation == "flattened":
        hashes[first] = "0" * 64
    elif mutation == "missing_path":
        hashes.pop(first)
    elif mutation == "extra_path":
        hashes["unexpected.cpp"] = dict(base_sha256="0" * 64, patched_sha256="1" * 64)
    elif mutation == "wrong_fields":
        hashes[first] = dict(base_sha256="0" * 64, wrong="1" * 64)
    elif mutation == "bad_digest":
        hashes[first]["patched_sha256"] = "not-a-hash"
    preimage = dict(observation)
    preimage.pop("identity_sha256")
    observation["identity_sha256"] = hashlib.sha256(vision.canonical(preimage).encode()).hexdigest()
    loaded_image_plan["route_controls_sha256"] = hashlib.sha256(
        vision.canonical(loaded_image_plan["route_controls"]).encode()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="experimental CPU image"):
        multimodal.validate_strata_multimodal_load_plan(loaded_image_plan, "cpu+cuda:0")
