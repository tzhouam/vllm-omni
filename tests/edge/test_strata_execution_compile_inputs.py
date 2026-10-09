# SPDX-License-Identifier: Apache-2.0
"""Exercise only the bound compile-record reader and its first shape gate.

Real temporary Bundle members are hashed and reread. Deliberately wrong-shape
JSON stops the whole validator before any graph or build evidence is needed;
these fixtures do not establish complete runtime or compile qualification.
"""

import hashlib
import json

import pytest

from vllm_omni.engine.backends.strata_execution import strata_exec_runtime as sut

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def compile_bundle(tmp_path, raw):
    member = "c/compile-inputs.json"
    (tmp_path / "c").mkdir()
    (tmp_path / member).write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    manifest = {
        "schema": "omni-strata-combined-members-v2",
        "files": [{"path": member, "size_bytes": len(raw), "sha256": digest}],
    }
    (tmp_path / "members.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = sut.Bundle(tmp_path, "members.json")
    # This is an archived native-path field, not a path opened by the verifier.
    receipt = {"path": "C:/synthetic-closed-build/compile-inputs.json", "size_bytes": len(raw), "sha256": digest}
    return bundle, receipt


def validate_compile_record(bundle, receipt, prefix="c"):
    # Wrong-shape input must stop before these deliberately unused contexts.
    return sut.verify_target_compile_inputs(bundle, {}, {}, {"compile_closure": receipt}, {}, prefix)


def json_string_of_size(size):
    return b'"' + b"x" * (size - 2) + b'"'


def test_scoped_reader_accepts_over_two_mib_before_refusing_wrong_shape(tmp_path):
    raw = json_string_of_size((2 << 20) + 1)
    bundle, receipt = compile_bundle(tmp_path, raw)
    with pytest.raises(sut.EvidenceError, match="^file_bound$"):
        bundle.load("c/compile-inputs.json")
    with pytest.raises(sut.EvidenceError, match="^target_compile_shape$"):
        validate_compile_record(bundle, receipt)


@pytest.mark.parametrize("size", [2 << 20, 4 << 20])
def test_exact_reader_boundaries_reach_shape_gate(tmp_path, size):
    bundle, receipt = compile_bundle(tmp_path, json_string_of_size(size))
    with pytest.raises(sut.EvidenceError, match="^target_compile_shape$"):
        validate_compile_record(bundle, receipt)
    if size == 2 << 20:
        assert bundle.load("c/compile-inputs.json") == "x" * (size - 2)
    else:
        with pytest.raises(sut.EvidenceError, match="^file_bound$"):
            bundle.load("c/compile-inputs.json")


def test_over_four_mib_refuses_at_file_read(tmp_path, monkeypatch):
    bundle, receipt = compile_bundle(tmp_path, json_string_of_size((4 << 20) + 1))

    def unexpected_decode(*args, **kwargs):
        pytest.fail("over-limit compile record reached decoding")

    monkeypatch.setattr(sut, "json_", unexpected_decode)
    with pytest.raises(sut.EvidenceError, match="^file_bound$"):
        validate_compile_record(bundle, receipt)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("size_bytes", 999, "target_closure_file_bytes"),
        ("sha256", "0" * 64, "target_closure_file_bytes"),
        ("path", "../escape.json", "native_relative_evidence_path"),
        ("path", "C:/synthetic/invalid\rpath.json", "native_evidence_path"),
    ],
)
def test_mismatched_receipt_refuses_before_read_or_decode(tmp_path, monkeypatch, field, value, reason):
    bundle, receipt = compile_bundle(tmp_path, b"{}")
    receipt[field] = value

    def unexpected_read_or_decode(*args, **kwargs):
        pytest.fail("mismatched receipt reached member read or JSON decoding")

    monkeypatch.setattr(bundle, "read", unexpected_read_or_decode)
    monkeypatch.setattr(sut, "json_", unexpected_read_or_decode)
    with pytest.raises(sut.EvidenceError, match=f"^{reason}$"):
        validate_compile_record(bundle, receipt)


def test_unbound_member_prefix_refuses_before_read_or_decode(tmp_path, monkeypatch):
    bundle, receipt = compile_bundle(tmp_path, b"{}")

    def unexpected_read_or_decode(*args, **kwargs):
        pytest.fail("unbound member prefix reached member read or JSON decoding")

    monkeypatch.setattr(bundle, "read", unexpected_read_or_decode)
    monkeypatch.setattr(sut, "json_", unexpected_read_or_decode)
    with pytest.raises(sut.EvidenceError, match="^target_closure_file_bytes$"):
        validate_compile_record(bundle, receipt, prefix="not-bound")
