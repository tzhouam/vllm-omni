# SPDX-License-Identifier: Apache-2.0
"""Native owned-browser observer integration; no inference or user-browser attach."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from benchmarks.edge_agent.paired_suite import FixtureSite
from vllm_omni.edge.agent.tools import ManagedEdgeBrowser
from vllm_omni.edge.windows_process_memory import WindowsProcessMemoryRegistry


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Edge")
def test_native_owned_browser_memory_and_close(tmp_path: Path) -> None:
    registry = WindowsProcessMemoryRegistry("pytest-owned-browser-native-memory-v1")
    assert registry.bind_current_agent()
    browser = ManagedEdgeBrowser(
        tmp_path / "owned-profile", headless=True, process_observer=registry.browser_checkpoint,
    )
    failure: BaseException | None = None
    observed = None
    text_hash = None
    close = None
    try:
        with FixtureSite() as fixture:
            browser.open(fixture.origin + "/text/en")
            page = browser.read()
            assert page["url"] == fixture.origin + "/text/en"
            assert "CEDAR-4827" in page["text"]
            text_hash = hashlib.sha256(page["text"].encode("utf-8")).hexdigest()
            observed = registry.sample()
            coverage = observed["coverage"]
            assert coverage["agent_identity_verified"]
            assert coverage["node_spawn_identity_verified"]
            assert coverage["browser_root_identity_verified"]
            assert coverage["all_descendants_covered"] is False
            assert not coverage["binding_registry_overflow"]
            assert not coverage["owner_checkpoint_queue_overflow"]
            rows = {row["role"]: row for row in observed["processes"]}
            for role in ("agent", "playwright_node", "edge_browser"):
                row = rows[role]
                assert row["available"] and not row["signaled"]
                assert row["creation_filetime_100ns"] > 0
                assert row["working_set_bytes"] > 0 and row["private_commit_bytes"] >= 0
            assert rows["playwright_node"]["identity_source"] == "playwright_owned_spawn_handle"
            checkpoints = {row["checkpoint"] for row in observed["owner_checkpoint_samples"]}
            assert {"before_node_start", "before_browser_launch", "after_browser_launch"} <= checkpoints
    except BaseException as exc:
        failure = exc
    finally:
        try:
            browser.close()
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note("secondary native browser close failure:" + type(exc).__name__)
        try:
            close = registry.close()
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note("secondary native observer close failure:" + type(exc).__name__)
        if failure is None:
            try:
                assert browser.process_memory_close_receipt["worker_joined"]
                assert browser.process_memory_close_receipt["observer_callback_error_count"] == 0
                assert close["agent_alive_observed"] and close["observer_handles_closed"]
                assert close["model_retirement_verified"] is False  # This test creates no model.
                assert close["children"]["bound_set_drain_verified"]
                assert close["all_descendants_retired"] is False
                assert any(item["checkpoint"] == "before_browser_close"
                           for item in close["children"]["owner_checkpoint_samples"])
            except BaseException as exc:
                failure = exc
        (tmp_path / "native-browser-memory-receipt.json").write_bytes((json.dumps({
            "schema": "omni-native-browser-memory-integration-v1",
            "scope": "owned bound set, no model, no all-descendant or memory-cap qualification",
            "text_sha256": text_hash, "sample": observed,
            "browser_close": browser.process_memory_close_receipt, "registry_close": close,
            "failure_type": type(failure).__name__ if failure is not None else None,
        }, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    if failure is not None:
        raise failure

