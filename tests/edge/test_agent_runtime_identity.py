# SPDX-License-Identifier: Apache-2.0
"""A reviewed route must expire when its actual runtime code changes."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from vllm_omni.edge.agent import runtime_identity


def _bridged_package(tmp_path: Path) -> tuple[ModuleType, Path, Path]:
    bridge = tmp_path / "bridge" / "omni_stage_contracts"
    implementation = tmp_path / "packages" / "omni_stage_contracts"
    bridge.mkdir(parents=True)
    implementation.mkdir(parents=True)
    (bridge / "__init__.py").write_text("__path__ = ['implementation']\n")
    source = implementation / "types.py"
    source.write_text("PROTOCOL_VERSION = 1\n")
    module = ModuleType("omni_stage_contracts")
    module.__file__ = str(bridge / "__init__.py")
    module.__path__ = [str(implementation)]
    return module, bridge / "__init__.py", source


def test_imported_package_hash_covers_bridge_and_implementation(tmp_path):
    module, bridge, implementation = _bridged_package(tmp_path)
    original = runtime_identity._package_source_sha256(module)
    assert len(original) == 64

    implementation.write_text("PROTOCOL_VERSION = 2\n")
    after_contract_edit = runtime_identity._package_source_sha256(module)
    assert after_contract_edit != original

    bridge.write_text("__path__ = ['other implementation']\n")
    assert runtime_identity._package_source_sha256(module) != after_contract_edit


def test_imported_package_hash_uses_relative_names(tmp_path):
    left, _, _ = _bridged_package(tmp_path / "left")
    right, _, _ = _bridged_package(tmp_path / "right")
    assert runtime_identity._package_source_sha256(left) == runtime_identity._package_source_sha256(right)


def test_imported_package_hash_refuses_missing_source(tmp_path):
    module = ModuleType("vllm")
    module.__path__ = [str(tmp_path / "missing")]
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime_identity._package_source_sha256(module)


def test_runtime_identity_changes_with_vllm_or_stage_contract(monkeypatch):
    monkeypatch.setattr(runtime_identity, "imported_omni_source_sha256", lambda: "a" * 64)
    monkeypatch.setattr(runtime_identity, "imported_vllm_source_sha256", lambda: "b" * 64)
    monkeypatch.setattr(runtime_identity, "imported_stage_contract_source_sha256", lambda: "c" * 64)
    baseline = runtime_identity.loaded_runtime_sha256()
    identity = runtime_identity.loaded_runtime_identity()
    assert identity["vllm_source_sha256"] == "b" * 64
    assert identity["stage_contract_source_sha256"] == "c" * 64

    monkeypatch.setattr(runtime_identity, "imported_vllm_source_sha256", lambda: "d" * 64)
    vllm_edit = runtime_identity.loaded_runtime_sha256()
    assert vllm_edit != baseline
    monkeypatch.setattr(runtime_identity, "imported_vllm_source_sha256", lambda: "b" * 64)
    monkeypatch.setattr(runtime_identity, "imported_stage_contract_source_sha256", lambda: "e" * 64)
    assert runtime_identity.loaded_runtime_sha256() != baseline
