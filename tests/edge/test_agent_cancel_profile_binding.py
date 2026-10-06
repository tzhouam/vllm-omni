"""Native cancellation diagnostics refuse mismatched profiles and log clobber."""

from __future__ import annotations

import json
import sys

import pytest

from benchmarks.edge_agent.experiments import native_cancel_recovery as cancel_probe


def test_cancel_log_copy_is_exclusive(tmp_path):
    source = tmp_path / "source.log"
    target = tmp_path / "record.route.log"
    source.write_bytes(b"sanitized startup one\n")
    cancel_probe.copy_log_no_clobber(source, target)
    assert target.read_bytes() == source.read_bytes()
    source.write_bytes(b"later run must not replace the original\n")
    with pytest.raises(FileExistsError):
        cancel_probe.copy_log_no_clobber(source, target)
    assert target.read_bytes() == b"sanitized startup one\n"


def test_cancel_profile_mismatch_is_recorded_before_any_request(tmp_path, monkeypatch):
    config = tmp_path / "source.json"
    config.write_text(json.dumps({"routes": [{"route_id": "gemma",
                                             "log_file": "original.log"}]}), encoding="utf-8")
    profile = tmp_path / "profile-index.json"
    profile.write_text("{}", encoding="utf-8")
    record = tmp_path / "private" / "cancel.jsonl"
    seen = []

    class Controller:
        def close(self):
            seen.append("closed")

        def submit(self, _prompt):
            raise AssertionError("profile mismatch must refuse before submitting")

    def reject(_path, **kwargs):
        seen.append("binding_checked")
        assert kwargs["route_id"] == "gemma"
        raise ValueError("profile hardware mismatch")

    monkeypatch.setattr(cancel_probe, "build_controller",
                        lambda _path: (Controller(), {"os": "test"}))
    monkeypatch.setattr(cancel_probe, "bind_profile_to_live", reject)
    monkeypatch.setattr(sys, "argv", ["native_cancel_recovery.py", "--config", str(config),
                                      "--record", str(record), "--profile-index", str(profile)])
    with pytest.raises(ValueError, match="profile hardware mismatch"):
        cancel_probe.main()
    assert seen == ["binding_checked", "closed"]
    rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["record_type"] == "manifest"
    assert rows[0]["result"] == "failed"
    assert rows[0]["profile_binding_requested"] is True
    assert rows[0]["profile_binding"] is None


def test_existing_cancel_outputs_refuse_before_controller_build(tmp_path, monkeypatch):
    config = tmp_path / "source.json"
    config.write_text(json.dumps({"routes": [{"route_id": "gemma",
                                             "log_file": "original.log"}]}), encoding="utf-8")
    record = tmp_path / "cancel.jsonl"
    record.write_bytes(b"earlier evidence\n")
    monkeypatch.setattr(cancel_probe, "build_controller",
                        lambda _path: pytest.fail("existing record must refuse before model setup"))
    monkeypatch.setattr(sys, "argv", ["native_cancel_recovery.py", "--config", str(config),
                                      "--record", str(record)])
    with pytest.raises(FileExistsError):
        cancel_probe.main()
    assert record.read_bytes() == b"earlier evidence\n"
