# SPDX-License-Identifier: Apache-2.0
"""The one-case visual evidence runner must stay narrow and non-qualifying."""

from __future__ import annotations

import hashlib
import json

import pytest

from benchmarks.edge_agent.experiments.native_browser_screenshot_one import (
    CASE_ID,
    PINNED_DOWNLOAD_MANIFEST,
    _request_passed,
    _select_case,
    _validate_pinned_bundle,
    _write_new_json,
)
from benchmarks.edge_agent.paired_suite import FixtureSite


def test_selected_case_uses_pixels_without_desktop_capture() -> None:
    with FixtureSite() as fixture:
        case = _select_case(fixture.origin, 10)
        assert case.case_id == CASE_ID
        assert case.reference == "ORBIT-7391"
        assert case.metadata["kind"] == "screen_vision"
        assert case.metadata["source"] == fixture.origin + "/visual/en"
        assert case.metadata["required_operations"] == [
            "browser_open", "browser_screenshot",
        ]
        assert "screen_capture" not in case.prompt


def test_one_request_result_never_promotes_incomplete_or_unsafe_trace() -> None:
    row = {
        "error": None,
        "e2e_complete": True,
        "placement_matches": True,
        "evaluation": {"success": True, "tool_safe": True},
    }
    assert _request_passed(row)
    assert not _request_passed({**row, "e2e_complete": False})
    assert not _request_passed({**row, "placement_matches": False})
    assert not _request_passed({**row, "evaluation": {"success": True, "tool_safe": False}})
    assert not _request_passed(None)


def test_hash_bound_json_is_created_only_once(tmp_path) -> None:
    path = tmp_path / "index.json"
    digest = _write_new_json(path, {"protocol_compliant": False, "release_qualified": False})
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text())["release_qualified"] is False
    with pytest.raises(FileExistsError):
        _write_new_json(path, {"protocol_compliant": True})


def test_qwen_route_must_match_pinned_model_and_projector() -> None:
    download = json.loads(PINNED_DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
    model, projector = download["files"]
    entry = {
        "model_file": rf"C:\models\{model['filename']}",
        "model_sha256": model["sha256"],
        "mmproj_file": rf"C:\models\{projector['filename']}",
        "mmproj_sha256": projector["sha256"],
    }
    lineage = {"artifact_revision": download["revision"]}
    _validate_pinned_bundle(entry, lineage, download)
    with pytest.raises(ValueError, match="mmproj_file"):
        _validate_pinned_bundle({**entry, "mmproj_sha256": "0" * 64}, lineage, download)
    with pytest.raises(ValueError, match="revision"):
        _validate_pinned_bundle(entry, {"artifact_revision": "0" * 40}, download)
