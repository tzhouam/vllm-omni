# SPDX-License-Identifier: Apache-2.0
"""Bounded image metadata and exact text registration replay; no model execution."""

import copy
import json
from types import SimpleNamespace

import pytest

from benchmarks.edge_harness import strata_vision_register as registrar


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', "[]"])
def test_image_metadata_retains_strict_json_rejection(tmp_path, raw):
    path = tmp_path / "input.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        registrar._object(path)


def test_only_explicit_text_manifest_read_accepts_larger_bounded_metadata(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"padding": "x" * (9 << 20)}))
    with pytest.raises(ValueError, match="bounded metadata"):
        registrar._object(path)
    assert len(registrar._object(path, max_bytes=16 << 20)["padding"]) == 9 << 20
    with pytest.raises(ValueError, match="bounded metadata"):
        registrar._object(path, max_bytes=path.stat().st_size - 1)


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("damage", [None, "mixed", "missing", "replayed_launch_changed"])
def test_image_registrar_replays_exact_observation_mode(tmp_path, monkeypatch, combined, damage):
    parent = tmp_path / "parent.json"
    parent.write_text("{}")
    launch_path, receipt_path = tmp_path / "launch.json", tmp_path / "receipt.json"
    backend = {
        "name": registrar.TEXT_BACKEND,
        "runtime_revision": registrar.RUNTIME_REVISION,
        "route_id": "text",
        "runtime_root": str(tmp_path),
        "artifact_manifest": {"identity": "source"},
        "runtime_manifest": {"identity": "runtime"},
        "prepared_pack_manifest": {"identity": "prepared"},
    }
    key = "execution_observation" if combined else "observation_runtime"
    backend[key] = {"schema": "fixture-bound-configuration"}
    if damage == "mixed":
        backend["observation_runtime" if combined else "execution_observation"] = {"mixed": True}
    if damage == "missing":
        backend.pop(key)
    launch = {"schema": "omni-strata-launch-v1", "backend": backend}
    receipt = {
        "schema": registrar.strata_runtime.SCHEMA,
        "status": "registered_not_executed",
        "qualification_created": False,
        "parent_binding_rewritten": False,
        "launch_sha256": registrar.canonical_hash(launch),
        "launch_out": str(launch_path),
        "parent_launch": str(parent),
        "route_id": "text",
        "source_manifest_sha256": "source",
        "runtime_manifest_sha256": "runtime",
        "prepared_manifest_sha256": "prepared",
    }
    launch_path.write_text(json.dumps(launch))
    receipt_path.write_text(json.dumps(receipt))
    calls = []

    def replay(original, **kwargs):
        calls.append(kwargs)
        assert original == parent
        assert kwargs[key] == backend[key]
        assert ("observation_runtime" if combined else "execution_observation") not in kwargs
        reproduced = copy.deepcopy(launch)
        if damage == "replayed_launch_changed":
            reproduced["backend"]["route_id"] = "different"
        kwargs["launch_out"].write_text(json.dumps(reproduced))
        return receipt | {"launch_out": str(kwargs["launch_out"])}

    monkeypatch.setattr(registrar.strata_runtime, "register_runtime_variant", replay)
    manifest = SimpleNamespace(from_dict=lambda value: SimpleNamespace(manifest_sha256=value["identity"]))
    if damage:
        with pytest.raises(ValueError):
            registrar._verified_text_intermediate(launch_path, receipt_path, manifest_type=manifest)
        if damage in {"mixed", "missing"}:
            assert not calls
    else:
        assert registrar._verified_text_intermediate(launch_path, receipt_path, manifest_type=manifest) == (
            launch,
            receipt,
            parent,
        )
        assert len(calls) == 1
