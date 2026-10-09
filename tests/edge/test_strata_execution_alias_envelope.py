# SPDX-License-Identifier: Apache-2.0
"""Public alias-envelope regressions without SDK headers or runtime grants."""

import pytest

from vllm_omni.engine.backends.strata_execution import strata_exec_runtime as sut

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]

WITNESS = "external_header_alias_witness_file"
COLLECTOR = "external_header_alias_collector_file"


def test_legacy_descriptor_does_not_read_or_normalize_aliases():
    assert sut.alias_descriptor_keys({}) == set()
    assert sut.verified_external_header_aliases(object(), {}, None, None, "unused") == ({}, None)


@pytest.mark.parametrize("key", [WITNESS, COLLECTOR])
def test_partial_witness_descriptor_is_rejected(key):
    with pytest.raises(sut.EvidenceError, match="alias_descriptor_pair_required"):
        sut.alias_descriptor_keys({key: "synthetic/member"})


def test_witness_requires_target_closure_supplement():
    with pytest.raises(sut.EvidenceError, match="alias_requires_target_supplement"):
        sut.alias_descriptor_keys({WITNESS: "synthetic/receipt.json", COLLECTOR: "synthetic/collect.py"})
    assert sut.alias_descriptor_keys(
        {
            WITNESS: "synthetic/receipt.json",
            COLLECTOR: "synthetic/collect.py",
            "build_closure_supplement_file": "synthetic/closure.json",
        }
    ) == {WITNESS, COLLECTOR}


def test_witness_duration_keeps_decimal_lexeme_without_changing_generic_json():
    raw = b'{"elapsed_s":0.047,"pairs":[{"size_bytes":123}]}'
    value = sut.alias_witness_json(raw)
    assert value == {"elapsed_s": "0.047", "pairs": [{"size_bytes": 123}]}
    assert type(value["pairs"][0]["size_bytes"]) is int
    with pytest.raises(sut.EvidenceError):
        sut.json_(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"elapsed_s":0.047,"size_bytes":123.0}',
        b'{"elapsed_s":0.047,"nested":[0.1]}',
        b'{"elapsed_s":true}',
        b'{"elapsed_s":"0.047"}',
        b'{"elapsed_s":-0.1}',
        b'{"elapsed_s":60.1}',
        b'{"elapsed_s":NaN}',
        b'{"elapsed_s":Infinity}',
        b'{"elapsed_s":0.047,"elapsed_s":0.047}',
        b'{"pairs":[]}',
    ],
)
def test_witness_decoder_rejects_other_decimals_invalid_duration_and_duplicates(raw):
    with pytest.raises(sut.EvidenceError):
        sut.alias_witness_json(raw)
