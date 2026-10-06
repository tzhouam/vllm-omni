"""Native quality/profile binding orchestration without loading a model."""

from __future__ import annotations

import json
import sys

import pytest

from benchmarks.edge_agent.profile import ProfileRoute


def _setup_native(tmp_path, monkeypatch):
    import benchmarks.edge_agent.experiments.native_memory_quality as quality
    import benchmarks.edge_agent.experiments.profile_binding as binding_module
    import benchmarks.edge_agent.native_profile as native_profile
    import vllm_omni.edge.agent.native_app as native_app
    import vllm_omni.edge.agent.runtime_identity as runtime_identity

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"routes": [{"route_id": "gemma-test"}]}),
                      encoding="utf-8")
    lineage = tmp_path / "lineage.json"
    lineage.write_text("{}", encoding="utf-8")
    route = ProfileRoute(
        "gemma-test", "Gemma 4 31B", "model-artifact", "revision",
        "a" * 64, "QAT Q4", "external.llamacpp.text.v1", "cpu+Vulkan0",
    )
    monkeypatch.setattr(native_profile, "load_profile_routes", lambda *_: (
        [route], {route.route_id: {"lineage_verified": True}},
    ))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(runtime_identity, "loaded_runtime_sha256", lambda: "runtime")
    monkeypatch.setattr(quality, "PRIVATE_RESULTS", tmp_path / "private")

    built = []

    class Controller:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    hardware = {"os": "Windows", "power_condition": "AC"}

    def build_controller(_path):
        controller = Controller()
        built.append(controller)
        return controller, hardware

    monkeypatch.setattr(native_app, "build_controller", build_controller)
    submitted = []

    def fake_suite(_cases, *, controller_factory, timeout_s, on_record):
        seed = controller_factory("seed")
        try:
            replay = controller_factory("replay")
            replay.close()
        finally:
            seed.close()
        submitted.append(timeout_s)
        return {"cases": 4, "passed": 4, "qualifies_default": False}

    monkeypatch.setattr(quality, "run_suite", fake_suite)
    return quality, binding_module, config, lineage, route, built, submitted, hardware


@pytest.mark.parametrize("profile_bound", [False, True])
def test_native_quality_manifest_binds_before_requests_and_rechecks_replay(
    tmp_path, monkeypatch, profile_bound,
):
    quality, binding_module, config, lineage, route, built, submitted, hardware = (
        _setup_native(tmp_path, monkeypatch)
    )
    profile = tmp_path / "profile.json" if profile_bound else None
    binding = {"schema": "omni-agent-profile-binding-v1",
               "profile_raw_sha256": "b" * 64}
    checked = []

    def bind(path, *, source_config_bytes, route_id, hardware: object):
        checked.append((path, source_config_bytes, route_id, hardware))
        return binding

    monkeypatch.setattr(binding_module, "bind_profile_to_live", bind)
    index_path = quality.run_native(config, lineage, timeout_s=7,
                                    seed=b"fixed-private-seed-0123456789",
                                    profile_index=profile)
    rows = [json.loads(row) for row in
            (index_path.parent / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["record_type"] for row in rows] == ["manifest", "summary"]
    assert rows[0]["profile_binding_requested"] is profile_bound
    assert rows[0]["profile_binding"] == (binding if profile_bound else None)
    assert len(checked) == (2 if profile_bound else 0)
    assert all(check == (profile, config.read_bytes(), route.route_id, hardware)
               for check in checked)
    assert len(built) == 2  # Preflight seed controller is reused by the suite.
    assert all(controller.closed for controller in built)
    assert submitted == [7]
    assert json.loads(index_path.read_text(encoding="utf-8"))["status"] == "completed"


def test_profile_mismatch_records_failure_and_never_submits_request(
    tmp_path, monkeypatch,
):
    quality, binding_module, config, lineage, _, built, submitted, _ = (
        _setup_native(tmp_path, monkeypatch)
    )
    checked = []

    def reject(*_args, **_kwargs):
        checked.append("binding_checked")
        raise ValueError("profile runtime mismatch")

    monkeypatch.setattr(binding_module, "bind_profile_to_live", reject)
    with pytest.raises(RuntimeError, match="memory-quality experiment failed"):
        quality.run_native(config, lineage, timeout_s=7,
                           seed=b"fixed-private-seed-0123456789",
                           profile_index=tmp_path / "profile.json")
    run_dir, = (tmp_path / "private").glob("memory_quality_*")
    rows = [json.loads(row) for row in
            (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["record_type"] for row in rows] == ["manifest", "failure"]
    assert rows[0]["profile_binding_requested"] is True
    assert rows[0]["profile_binding"] is None
    assert rows[1]["error_type"] == "ValueError"
    assert "profile runtime mismatch" in rows[1]["message"]
    assert checked == ["binding_checked"]
    assert len(built) == 1 and built[0].closed
    assert submitted == []
    assert json.loads((run_dir / "index.json").read_text(encoding="utf-8"))["status"] == "failed"
