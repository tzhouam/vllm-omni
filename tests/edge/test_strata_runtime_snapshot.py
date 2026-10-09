# SPDX-License-Identifier: Apache-2.0
"""Same-generation verification reuse with tiny files, never native eligibility.

The real loader, Bundle, typed manifest, snapshot and Stage cleanup methods run
here. Only the full archive/build verifier and native/diagnostic boundaries are
mocked. Live construction stops at the owned-module audit boundary; reaching it
does not establish actual process, GPU, module or first-frame ABI verification.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import pickle
import shutil
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm_omni.engine.backends import strata
from vllm_omni.engine.resource_ledger import ResourceLedger
from vllm_omni.engine.weight_tiers import ArtifactFile, ArtifactManifest

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def pe_fixture_bytes(normal=(), delay=()):
    """Synthetic import-only x64 PE, adapted from the existing vision fixture.

    These bytes are parsed as data only; no PE or tool is loaded or executed.
    """
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
            raw_name = library.encode("ascii") + b"\0"
            data[name_cursor : name_cursor + len(raw_name)] = raw_name
            row = cursor + index * row_size
            if directory == 1:
                struct.pack_into("<I", data, row + 12, name_cursor - 512 + 0x1000)
            else:
                struct.pack_into("<II", data, row, 1, name_cursor - 512 + 0x1000)
            name_cursor += len(raw_name)
        cursor += row_size * (len(names) + 1)
    return bytes(data)


@pytest.fixture
def adapters(tmp_path):
    """Use the production loader and current pins, without a runtime install."""
    root = tmp_path / "snapshot-adapters"
    target = root / "adapter"
    target.mkdir(parents=True)
    source = Path(strata.__file__).with_name("strata_execution")
    names = tuple(strata._EXECUTION_ADAPTER_SHA256)
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
            shutil.copyfile(source / (name + ".py"), target / (name + ".py"))
        monkeypatch.setattr(strata, "_EXECUTION_LOADED_MODULES", {})
        try:
            modules = strata._load_execution_adapters(root, set(target.iterdir()))
            yield modules
        finally:
            for name in names:
                sys.modules.pop(name, None)


@pytest.fixture
def harness(tmp_path, adapters):
    sut = adapters["strata_exec_runtime"]
    root = tmp_path / "runtime"
    root.mkdir()
    members = {
        "descriptor.json": encoded({"manifest_file": "members.json"}),
        "context.json": encoded({"fixture": "bounded metadata"}),
        "serve/server.py": b"# selected server fixture\n",
        "serve/lazy/helper.pyc": b"bounded-bytecode-preimage",
        "tools/helper.py": b"# selected tool fixture\n",
    }
    rows = []
    for name, raw in members.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        rows.append({"path": name, "size_bytes": len(raw), "sha256": sut.sha(raw)})
    raw_manifest = encoded({"schema": "omni-strata-combined-members-v2", "files": rows})
    (root / "members.json").write_bytes(raw_manifest)
    bundle = sut.Bundle(root, "members.json")
    typed = ArtifactManifest(
        checkpoint="snapshot-fixture-中文",
        revision="test-revision",
        license="test-only",
        hash_origin="local_observation",
        files=tuple(ArtifactFile(**row, role="runtime") for row in rows)
        + (ArtifactFile("members.json", len(raw_manifest), sut.sha(raw_manifest), role="runtime"),),
    )
    stage = strata.StrataTextStageClient.__new__(strata.StrataTextStageClient)
    stage._execution_modules = adapters
    stage._execution_runtime_manifest = typed
    stage._verified_execution_snapshot = None
    stage._execution_snapshot_retirement_failed = False
    stage._generation, stage._closed = "generation-1", False
    stage.execution_plan = {"runtime_manifest_sha256": typed.manifest_sha256}
    identity = {
        "schema": "test-only-strict-boundary-result",
        "runtime_manifest_sha256": bundle.manifest_sha256,
        "descriptor_sha256": sut.sha(members["descriptor.json"]),
        "runtime_binding": None,
        "compiled_engine_ABI_verified": False,
        "observer_layout_scope": "compiled_standalone_fixture_reference_only",
    }
    identity["identity_sha256"] = sut.sha(sut.canonical(identity))
    value = SimpleNamespace(
        sut=sut,
        live=adapters["strata_exec_live"],
        root=root,
        bundle=bundle,
        typed=typed,
        stage=stage,
        identity=identity,
        directory=tmp_path / "receipts",
    )
    value.directory.mkdir()
    yield value
    snapshot = stage._verified_execution_snapshot
    if type(snapshot) is sut._VerifiedCombinedRuntimeSnapshot:
        snapshot.invalidate()


def issue(harness):
    """Replace only whole-installation proof; keep actual capability validation."""
    h = harness
    with patch.object(
        h.sut, "_verify_combined_runtime_strict", return_value=(copy.deepcopy(h.identity), h.bundle)
    ) as strict:
        snapshot = h.sut._verify_combined_runtime_snapshot(
            "descriptor.json",
            h.root,
            owner_stage=h.stage,
            worker_generation=h.stage._generation,
            typed_runtime_manifest_sha256=h.typed.manifest_sha256,
            source_context_file="context.json",
        )
    strict.assert_called_once_with("descriptor.json", h.root)
    h.stage._verified_execution_snapshot = snapshot
    return snapshot


def claim(harness, snapshot, **changes):
    return snapshot.claim(
        changes.get("stage", harness.stage),
        changes.get("descriptor", "descriptor.json"),
        changes.get("root", harness.root),
        changes.get("context", "context.json"),
    )


def prepare_cleanup(harness):
    """Actual lease accounting; only OS drain and report closure are mocked."""
    stage = harness.stage
    stage._ledger = ResourceLedger({"ram": 128})
    stage._reservation = stage._ledger.reserve("snapshot-fixture", {"ram": 64})
    stage._active, stage._epoch = "request", 1
    stage._cancel = threading.Event()
    stage._agent_stream = stage._output = stage._task = stage._ack_pending = stage._io_request = None
    stage._terminate = Mock(return_value=True)
    stage._finish_failure_observations = Mock(side_effect=lambda *args, **kwargs: (None, None, kwargs["drained"]))
    return stage


def assert_quarantined(stage):
    state = stage._ledger.snapshot()
    assert stage._ledger.owns(stage._reservation)
    assert state["reserved"] == {"ram": 64}
    assert state["quarantined"] == ["snapshot-fixture"]


def test_public_static_entry_preserves_dict_contract(harness):
    h = harness
    with patch.object(h.sut, "_verify_combined_runtime_strict", return_value=(h.identity, h.bundle)) as strict:
        assert h.sut.verify_combined_runtime("descriptor.json", h.root) is h.identity
    strict.assert_called_once_with("descriptor.json", h.root)


def test_real_pe_verifier_preserves_dict_identity_and_static_scope(tmp_path, adapters):
    """Exercise the actual Bundle, PE parser and public helper without mocks."""
    sut = adapters["strata_exec_runtime"]
    root = tmp_path / "pe-contract"
    root.mkdir()
    specifications = [
        (
            "engine/strata.exe",
            ("cublas64_13.dll", "KERNEL32.dll"),
            ("cublaslt64_13.dll",),
            ["cublas64_13.dll", "cublaslt64_13.dll"],
            ["kernel32.dll"],
        ),
        ("engine/cublas64_13.dll", ("ADVAPI32.dll",), (), [], ["advapi32.dll"]),
        ("engine/cublaslt64_13.dll", (), (), [], []),
    ]
    members = {"tools/import-tool": b"synthetic tool identity; never executed\n"}
    pe_rows, logs = [], []
    for index, (name, normal, delay, non_system, system) in enumerate(specifications):
        raw = pe_fixture_bytes(normal, delay)
        members[name] = raw
        pe_rows.append(
            {
                "path": name,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "normal": sorted(library.lower() for library in normal),
                "delay": sorted(library.lower() for library in delay),
                "non_system": non_system,
                "system": system,
            }
        )
        log_name = f"logs/import-{index}.txt"
        members[log_name] = b"synthetic receipt preimage; no tool invocation\n"
        logs.append(
            {
                "binary_file": name,
                "log_file": log_name,
                "log_sha256": hashlib.sha256(members[log_name]).hexdigest(),
                "exit_code": 0,
                "tool_file": "tools/import-tool",
                "tool_sha256": hashlib.sha256(members["tools/import-tool"]).hexdigest(),
            }
        )
    members["pe-receipt.json"] = encoded(
        {
            "schema": "omni-strata-combined-pe-evidence-v2",
            "engine_file": "engine/strata.exe",
            "files": pe_rows,
            "raw_import_logs": logs,
        }
    )
    manifest_rows = []
    for name, raw in sorted(members.items()):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest_rows.append({"path": name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    (root / "members.json").write_bytes(encoded({"schema": "omni-strata-combined-members-v2", "files": manifest_rows}))
    bundle = sut.Bundle(root, "members.json")

    result = sut.verify_pe_evidence(bundle, "pe-receipt.json", "engine/strata.exe")

    assert type(result) is dict
    assert set(result) == {
        "schema",
        "files",
        "all_dynamic_loads_covered",
        "current_system_dependencies_verified",
        "identity_sha256",
    }
    assert result["schema"] == "omni-strata-static-PE-closure-v2"
    assert result["files"] == sorted(pe_rows, key=lambda row: row["path"])
    assert result["all_dynamic_loads_covered"] is False
    assert result["current_system_dependencies_verified"] is False
    identity = {key: value for key, value in result.items() if key != "identity_sha256"}
    assert result["identity_sha256"] == hashlib.sha256(encoded(identity)).hexdigest()


def test_one_claim_and_detached_static_identity(harness):
    h = harness
    snapshot = issue(h)
    detached = snapshot.detached_identity()
    detached["runtime_binding"] = {"forged": True}
    identity, bundle = claim(h, snapshot)
    assert bundle is h.bundle
    assert identity == h.identity and identity is not h.identity
    assert identity["runtime_binding"] is None
    with pytest.raises(h.sut.EvidenceError, match="snapshot_already_claimed"):
        claim(h, snapshot)
    assert snapshot._valid is False


@pytest.mark.parametrize("operation", [copy.copy, copy.deepcopy, pickle.dumps])
def test_capability_cannot_be_copied_or_serialized(harness, operation):
    with pytest.raises(TypeError):
        operation(issue(harness))


def test_unissued_snapshot_cannot_be_constructed(harness):
    h = harness
    with pytest.raises(h.sut.EvidenceError, match="snapshot_private_issuer_required"):
        h.sut._VerifiedCombinedRuntimeSnapshot(
            object(),
            h.stage,
            h.stage._generation,
            h.root,
            "descriptor.json",
            "context.json",
            h.typed.manifest_sha256,
            h.identity,
            h.bundle,
            (),
        )


@pytest.mark.parametrize("mutation", ["stage", "generation", "closed", "field", "module"])
def test_stage_generation_lifetime_and_actual_issuer_module_are_bound(harness, mutation):
    h = harness
    snapshot = issue(h)
    kwargs = {}
    if mutation == "stage":
        kwargs["stage"] = strata.StrataTextStageClient.__new__(strata.StrataTextStageClient)
    elif mutation == "generation":
        h.stage._generation = "generation-2"
    elif mutation == "closed":
        h.stage._closed = True
    elif mutation == "field":
        h.stage._verified_execution_snapshot = {"verified": True}
    else:
        h.stage._execution_modules = dict(h.stage._execution_modules, strata_exec_runtime=object())
    with pytest.raises(h.sut.EvidenceError):
        claim(h, snapshot, **kwargs)
    assert snapshot._valid is False


@pytest.mark.parametrize("mutation", ["root", "descriptor", "context", "typed_object", "loaded_outer_digest"])
def test_input_paths_and_typed_outer_manifest_are_bound(harness, mutation):
    h = harness
    snapshot = issue(h)
    kwargs = {}
    if mutation == "root":
        kwargs["root"] = h.root.parent
    elif mutation in ("descriptor", "context"):
        kwargs[mutation] = "different.json"
    elif mutation == "typed_object":
        h.stage._execution_runtime_manifest = ArtifactManifest.from_dict(h.typed.to_dict())
    else:
        h.stage.execution_plan["runtime_manifest_sha256"] = "f" * 64
    with pytest.raises(h.sut.EvidenceError):
        claim(h, snapshot, **kwargs)
    assert snapshot._valid is False


def test_wrong_typed_digest_refuses_before_whole_installation_verification(harness):
    h = harness
    with patch.object(h.sut, "_verify_combined_runtime_strict") as strict:
        with pytest.raises(h.sut.EvidenceError, match="snapshot_typed_outer_manifest_binding"):
            h.sut._verify_combined_runtime_snapshot(
                "descriptor.json",
                h.root,
                owner_stage=h.stage,
                worker_generation=h.stage._generation,
                typed_runtime_manifest_sha256="f" * 64,
                source_context_file="context.json",
            )
    strict.assert_not_called()


def test_sticky_failure_alone_refuses_reissuance_before_strict_boundary(harness):
    h = harness
    h.stage._execution_snapshot_retirement_failed = True
    assert h.stage._closed is False and h.stage._verified_execution_snapshot is None
    with patch.object(h.sut, "_verify_combined_runtime_strict") as strict:
        with pytest.raises(h.sut.EvidenceError, match="snapshot_fresh_Stage_generation_required"):
            h.sut._verify_combined_runtime_snapshot(
                "descriptor.json",
                h.root,
                owner_stage=h.stage,
                worker_generation=h.stage._generation,
                typed_runtime_manifest_sha256=h.typed.manifest_sha256,
                source_context_file="context.json",
            )
    strict.assert_not_called()


@pytest.mark.parametrize(
    "member",
    ["members.json", "descriptor.json", "context.json", "serve/server.py", "serve/lazy/helper.pyc", "tools/helper.py"],
)
def test_selected_bytes_are_freshly_reread_after_issuance(harness, member):
    h = harness
    snapshot = issue(h)
    path = h.root / member
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(h.sut.EvidenceError):
        claim(h, snapshot)
    assert snapshot._valid is False


def test_raw_manifest_and_canonical_outer_manifest_are_separate_bindings(harness):
    h = harness
    raw = (h.root / "members.json").read_bytes()
    assert h.typed.manifest_sha256 != hashlib.sha256(raw).hexdigest()
    h.stage._execution_runtime_manifest = ArtifactManifest(
        checkpoint=h.typed.checkpoint,
        revision=h.typed.revision,
        license=h.typed.license,
        hash_origin=h.typed.hash_origin,
        files=tuple(
            ArtifactFile(row.path, row.size_bytes, "f" * 64, role=row.role) if row.path == "members.json" else row
            for row in h.typed.files
        ),
    )
    h.typed = h.stage._execution_runtime_manifest
    h.stage.execution_plan["runtime_manifest_sha256"] = h.typed.manifest_sha256
    snapshot = issue(h)
    with pytest.raises(h.sut.EvidenceError, match="snapshot_outer_raw_manifest_binding_changed"):
        claim(h, snapshot)


class StopAtOwnedModuleAuditError(Exception):
    """An explicit boundary, not a successful live verification result."""


@pytest.mark.parametrize("mode", ["snapshot", "legacy", "invalid"])
def test_actual_live_constructor_reuse_legacy_and_invalid_refusal(harness, mode):
    h = harness
    snapshot = issue(h) if mode == "snapshot" else {"verified": True} if mode == "invalid" else None
    instance = h.live.LiveExecutionBindingVerifier.__new__(h.live.LiveExecutionBindingVerifier)
    with (
        patch.object(h.live, "os", SimpleNamespace(name="nt")),
        patch.object(h.sut, "verify_combined_runtime", return_value=h.identity) as verify,
        patch.object(h.live, "_reopen_static_bundle", return_value=h.bundle) as reopen,
        patch.object(h.live, "_module_preimages", side_effect=StopAtOwnedModuleAuditError) as native_boundary,
    ):
        expected = h.live.LiveBindingError if mode == "invalid" else StopAtOwnedModuleAuditError
        with pytest.raises(expected):
            instance.__init__(
                h.stage, "descriptor.json", h.root, "context.json", h.directory, verified_snapshot=snapshot
            )
    assert verify.call_count == reopen.call_count == int(mode == "legacy")
    assert native_boundary.call_count == int(mode != "invalid")
    if mode == "snapshot":
        assert instance._bundle is h.bundle


@pytest.mark.asyncio
async def test_failed_abort_then_repeated_shutdown_retains_metadata_and_quarantine(harness):
    h = harness
    snapshot = issue(h)
    stage = prepare_cleanup(h)
    with patch.object(type(snapshot), "invalidate", side_effect=RuntimeError("fixture retirement failure")):
        await stage.abort_requests_async(["request"])
        stage.shutdown()
        stage.shutdown()
    assert stage._verified_execution_snapshot is snapshot
    assert stage._execution_runtime_manifest is h.typed
    assert stage._execution_snapshot_retirement_failed is True
    assert stage._terminate.call_count == 3
    assert_quarantined(stage)
    with patch.object(h.sut, "_verify_combined_runtime_strict") as strict:
        with pytest.raises(h.sut.EvidenceError, match="snapshot_fresh_Stage_generation_required"):
            h.sut._verify_combined_runtime_snapshot(
                "descriptor.json",
                h.root,
                owner_stage=stage,
                worker_generation=stage._generation,
                typed_runtime_manifest_sha256=h.typed.manifest_sha256,
                source_context_file="context.json",
            )
    strict.assert_not_called()


def test_unknown_then_missing_snapshot_cannot_erase_retirement_uncertainty(harness):
    h = harness
    stage = prepare_cleanup(h)
    stage._verified_execution_snapshot = object()
    stage.shutdown()
    stage._verified_execution_snapshot = None
    stage.shutdown()
    assert stage._execution_snapshot_retirement_failed is True
    assert stage._execution_runtime_manifest is h.typed
    assert stage._terminate.call_count == 2
    assert_quarantined(stage)


def test_successful_retirement_retry_releases_exact_lease_only_after_drain(harness):
    h = harness
    snapshot = issue(h)
    stage = prepare_cleanup(h)
    invalidate = type(snapshot).invalidate
    attempts = []

    def first_failure_then_real_invalidation(actual):
        attempts.append(actual)
        if len(attempts) == 1:
            raise RuntimeError("fixture retirement failure")
        invalidate(actual)

    with patch.object(type(snapshot), "invalidate", autospec=True, side_effect=first_failure_then_real_invalidation):
        stage.shutdown()
        assert_quarantined(stage)
        stage.shutdown()
    assert stage._terminate.call_count == 2
    assert stage._ledger.was_released(stage._reservation)
    assert stage._ledger.snapshot()["reserved"] == {"ram": 0}
    assert stage._ledger.snapshot()["quarantined"] == []
    assert stage._verified_execution_snapshot is stage._execution_runtime_manifest is None
    assert stage._execution_snapshot_retirement_failed is False


@pytest.mark.asyncio
async def test_actual_request_cancelled_error_preserves_failed_retirement(harness):
    h = harness
    snapshot = issue(h)
    stage = prepare_cleanup(h)
    stage.stage_id = 0
    stage._snapshot_children = Mock()
    stage._base_url, stage._model_alias, stage._token = "http://fixture.invalid", "fixture", "fixture"
    stage._timeout, stage._max_io_bytes = 1, 1024
    stage._http_content = Mock(return_value="fixture")
    request = strata.StageRequest("request", stage.stage_id, stage._epoch, stage._generation)
    with (
        patch.object(type(snapshot), "invalidate", side_effect=RuntimeError("fixture retirement failure")),
        patch.object(strata, "_stream_request", side_effect=asyncio.CancelledError) as native_request,
    ):
        with pytest.raises(asyncio.CancelledError):
            await stage._run(request, "fixture", 8, None)
        assert stage._closed is True
        assert stage._execution_snapshot_retirement_failed is True
        assert stage._verified_execution_snapshot is snapshot
        assert stage._ledger.owns(stage._reservation)
        stage.shutdown()
    native_request.assert_called_once()
    assert stage._terminate.call_count == 1
    assert_quarantined(stage)


def test_absent_legacy_snapshot_allows_existing_successful_shutdown(harness):
    stage = prepare_cleanup(harness)
    stage.shutdown()
    stage.shutdown()
    assert stage._execution_snapshot_retirement_failed is False
    assert stage._execution_runtime_manifest is None
    assert stage._terminate.call_count == 2
    assert stage._ledger.was_released(stage._reservation)
    assert stage._ledger.snapshot()["reserved"] == {"ram": 0}
