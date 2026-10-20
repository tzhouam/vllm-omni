# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Identifying a checkpoint well enough to tell it from its siblings."""

import json

import pytest

from vllm_omni.edge.local.capabilities import FORMAT_DENSE, FORMAT_FP8, FORMAT_INT8
from vllm_omni.edge.local.manifest import build_manifest, runtime_versions

from .conftest import FP8_QUANT_CONFIG, INT8_QUANT_CONFIG, write_checkpoint

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_quantization_format_is_read_from_the_config(tmp_path):
    assert build_manifest(write_checkpoint(tmp_path / "d")).weight_format == FORMAT_DENSE
    assert build_manifest(
        write_checkpoint(tmp_path / "i", quantization=INT8_QUANT_CONFIG)
    ).weight_format == FORMAT_INT8
    assert build_manifest(
        write_checkpoint(tmp_path / "f", quantization=FP8_QUANT_CONFIG)
    ).weight_format == FORMAT_FP8


def test_compressed_tensors_int8_and_fp8_are_distinguished_by_format_not_method(tmp_path):
    """Both say ``compressed-tensors``; only ``format`` separates the one
    Blackwell can execute from the one it cannot."""
    int8 = build_manifest(write_checkpoint(tmp_path / "i", quantization=INT8_QUANT_CONFIG))
    fp8 = build_manifest(write_checkpoint(tmp_path / "f", quantization=FP8_QUANT_CONFIG))
    assert int8.quantization["quant_method"] == fp8.quantization["quant_method"]
    assert int8.weight_format != fp8.weight_format


def test_two_quantizations_of_one_model_get_different_ids(tmp_path):
    dense = build_manifest(write_checkpoint(tmp_path / "d"))
    int8 = build_manifest(write_checkpoint(tmp_path / "i", quantization=INT8_QUANT_CONFIG))
    assert dense.artifact_id != int8.artifact_id


def test_the_id_is_stable_across_reads(tmp_path):
    path = write_checkpoint(tmp_path / "d")
    assert build_manifest(path).artifact_id == build_manifest(path).artifact_id


def test_a_changed_tokenizer_changes_the_id(tmp_path):
    """The tokenizer decides what the model is asked, and moves independently
    of the weights."""
    path = write_checkpoint(tmp_path / "d")
    before = build_manifest(path).artifact_id
    (path / "chat_template.jinja").write_text("{{ different }}", encoding="utf-8")
    assert build_manifest(path).artifact_id != before


def test_weights_are_not_hashed_unless_asked(tmp_path):
    path = write_checkpoint(tmp_path / "d")
    assert build_manifest(path).weight_sha256 is None
    assert any("digest_weights" in n for n in build_manifest(path).notes)
    assert build_manifest(path, digest_weights=True).weight_sha256 is not None


def test_weight_bytes_and_shards_are_recorded(tmp_path):
    path = write_checkpoint(tmp_path / "d", weight_bytes=8192, shards=4)
    manifest = build_manifest(path)
    assert len(manifest.weight_files) == 4
    assert manifest.weight_bytes == 8192


def test_a_directory_without_a_config_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path / "empty")


def test_an_hf_snapshot_path_yields_its_revision(tmp_path):
    root = tmp_path / "models--org--name" / "snapshots" / "deadbeef1234"
    write_checkpoint(root)
    assert build_manifest(root).revision == "deadbeef1234"


def test_a_local_checkpoint_reports_no_revision_rather_than_guessing(tmp_path):
    assert build_manifest(write_checkpoint(tmp_path / "d")).revision is None


def test_the_runtime_records_both_the_wheel_and_the_checkout():
    versions = runtime_versions()
    assert versions.python
    # The import path, not just the version string: an editable install and a
    # site-packages copy report the same version and are different code.
    assert versions.vllm_omni_file.endswith("__init__.py")
    assert isinstance(versions.checkouts, dict)


def test_the_manifest_is_json_serialisable(tmp_path):
    manifest = build_manifest(write_checkpoint(tmp_path / "d"))
    back = json.loads(json.dumps(manifest.to_dict(), default=str))
    assert back["artifact_id"] == manifest.artifact_id
    assert back["architectures"] == ["Spark2_5ForCausalLM"]
