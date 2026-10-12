"""No network: contract tests for the audited AI Hub replay journal."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.edge_harness.aihub_replay import (
    AmbiguousSubmission,
    HubArtifact,
    HubJobFailed,
    HubReplay,
    audit_run,
)


TARGET = "Unit Test Qualcomm Target"
CHECKPOINT = "test/checkpoint"
REVISION = "0000001"


class FakeJob:
    def __init__(self, job_id, device, model_id, dataset_id, inputs, *, status="SUCCESS"):
        self.job_id = job_id
        self.device = device
        self.model = SimpleNamespace(model_id=model_id)
        self.inputs = SimpleNamespace(dataset_id=dataset_id)
        self._inputs = inputs
        self._status = status
        self.options = "--compute_unit npu"
        self.profiling = False

    def wait(self):
        return SimpleNamespace(code=self._status, message="" if self._status == "SUCCESS" else "failed")

    def download_output_data(self):
        return {"y": [self._inputs["x"][0] * 2]}

    def download_profile(self):
        return {"execution_summary": {"all_inference_times": [1]}, "execution_detail": []}


class FakeProfileJob:
    def __init__(self, job_id, device, model_id, profile):
        self.job_id = job_id
        self.device = device
        self.model = SimpleNamespace(model_id=model_id)
        self._profile = profile

    def get_status(self):
        return SimpleNamespace(code="SUCCESS")

    def download_profile(self):
        return self._profile


class FakeClient:
    def __init__(self):
        self.device = SimpleNamespace(name=TARGET, os="15", attributes=["os:android", "framework:qnn"])
        self.jobs = {}
        self.submissions = []
        self.fail_status = None
        self.throw_after_submit = False

    def get_devices(self, *, name):
        return [self.device] if name == TARGET else []

    def get_model(self, model_id):
        return SimpleNamespace(model_id=model_id)

    def submit_inference_job(self, *, model, device, inputs, name, options, profile):
        assert profile is False
        assert "--max_profiler_iterations 1" in options
        assert device is self.device
        assert list(inputs) == ["x"]
        assert len(inputs["x"]) == 1
        job_id = f"j{len(self.jobs) + 1}"
        job = FakeJob(job_id, device, model.model_id, f"d{len(self.jobs) + 1}", inputs,
                      status=self.fail_status or "SUCCESS")
        self.jobs[job_id] = job
        self.submissions.append({"job_id": job_id, "name": name, "profile": profile})
        if self.throw_after_submit:
            raise ConnectionError("response lost after server accepted the job")
        return job

    def get_job(self, job_id):
        return self.jobs[job_id]


class OldClient(FakeClient):
    def submit_inference_job(self, *, model, device, inputs, name, options):
        raise AssertionError("old client must not submit")


def artifact():
    return HubArtifact(
        artifact_id="compiled-block", target_model_id="m123", sha256="a" * 64,
        checkpoint=CHECKPOINT, revision=REVISION, layout_version=2,
        precision="fp16", target_name=TARGET, runtime="qnn_context_binary",
        shape_bucket="batch1-token1", hash_source="compiled_artifact",
    )


def replay(tmp_path, client, *, topology=None):
    return HubReplay(
        tmp_path, target_name=TARGET, model="Spark-X2.5", checkpoint=CHECKPOINT,
        revision=REVISION, request_id="request-1", epoch=7,
        stage_topology=topology or {"encode": [], "decode": ["encode"]},
        client=client, hub_module=SimpleNamespace(__version__="0.56.0"),
    )


def first_stage(run):
    return run.execute(
        "encode-0", stage_id=0, stage_name="encode", artifact=artifact(),
        inputs={"x": np.array([1, 2], dtype=np.float32)},
        input_provenance={"x": "request:real-input"},
        options="--compute_unit npu",
    )


def second_stage(run, previous):
    return run.execute(
        "decode-0", stage_id=1, stage_name="decode", artifact=artifact(),
        inputs={"x": previous.outputs["y"]},
        input_provenance={"x": "step:encode-0:y"},
        options="--compute_unit npu",
    )


def quality():
    return {"passed": True, "reference": "test/checkpoint@0000001",
            "checks": [{"name": "exact", "passed": True}]}


def test_sequential_real_input_jobs_and_resume(tmp_path):
    client = FakeClient()
    with replay(tmp_path, client) as run:
        first = first_stage(run)
        second = second_stage(run, first)
        assert second.outputs["y"].tolist() == [4.0, 8.0]
        run.finish(final_output={"token_ids": [4, 8]},
                   final_output_source="step:decode-0",
                   quality=quality(), required_stages=["encode", "decode"])
        assert run.record["status"] == "functional_complete_placement_unverified"
    assert [row["profile"] for row in client.submissions] == [False, False]
    assert [row["job_id"] for row in json.loads((tmp_path / "run.json").read_text())["steps"]] == ["j1", "j2"]
    with replay(tmp_path, client) as run:
        assert first_stage(run).job_id == "j1"
        assert second_stage(run, first_stage(run)).job_id == "j2"
    assert len(client.submissions) == 2
    assert audit_run(tmp_path)["step_count"] == 2


def test_complete_requires_verified_profile_and_audit_detects_tampering(tmp_path):
    client = FakeClient()
    raw_profile = {"execution_detail": [
        {"name": "Input", "type": "Input", "compute_unit": "NPU"},
        {"name": "MatMul", "type": "MatMul", "compute_unit": "NPU"},
    ]}
    with replay(tmp_path, client) as run:
        first = first_stage(run)
        second_stage(run, first)
        for index, step_id in enumerate(("encode-0", "decode-0"), start=1):
            profile_id = f"jp{index}"
            client.jobs[profile_id] = FakeProfileJob(profile_id, client.device, "m123", raw_profile)
            raw_path = tmp_path / f"{step_id}.profile.json"
            raw_path.write_text(json.dumps(raw_profile), encoding="utf-8")
            run.record_placement(step_id, actual_compute_units=["NPU"],
                                 profile_job_id=profile_id, raw_profile_path=raw_path)
        run.finish(final_output={"token_ids": [4, 8]},
                   final_output_source="step:decode-0",
                   quality=quality(), required_stages=["encode", "decode"])
        assert run.record["status"] == "complete"
    assert audit_run(tmp_path)["status"] == "complete"
    (tmp_path / "encode-0.outputs.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="tensor file changed"):
        audit_run(tmp_path)


def test_submission_gap_is_ambiguous_and_never_automatically_retried(tmp_path):
    client = FakeClient()
    client.throw_after_submit = True
    with replay(tmp_path, client) as run:
        with pytest.raises(ConnectionError):
            first_stage(run)
    client.throw_after_submit = False
    with replay(tmp_path, client) as run:
        with pytest.raises(AmbiguousSubmission):
            first_stage(run)
    assert len(client.submissions) == 1


def test_failed_job_and_bad_lineage_are_not_silently_replaced(tmp_path):
    client = FakeClient()
    client.fail_status = "FAILED"
    with replay(tmp_path, client) as run:
        with pytest.raises(HubJobFailed):
            first_stage(run)
    with replay(tmp_path, client) as run:
        with pytest.raises(HubJobFailed, match="existing job"):
            first_stage(run)
    assert len(client.submissions) == 1
    other = tmp_path / "other"
    client.fail_status = None
    with replay(other, client) as run:
        first_stage(run)
        with pytest.raises(ValueError, match="does not match named predecessor"):
            run.execute(
                "decode-0", stage_id=1, stage_name="decode", artifact=artifact(),
                inputs={"x": np.array([9, 9], dtype=np.float32)},
                input_provenance={"x": "step:encode-0:y"},
            )


def test_pending_job_reattaches_and_target_lock_is_single_request(tmp_path):
    client = FakeClient()
    client.fail_status = "RUNNING"
    with replay(tmp_path, client) as run:
        with pytest.raises(HubJobFailed, match="RUNNING"):
            first_stage(run)
        with pytest.raises(RuntimeError, match="another batch-1 replay"):
            replay(tmp_path / "concurrent", client)
    client.jobs["j1"]._status = "SUCCESS"
    with replay(tmp_path, client) as run:
        assert first_stage(run).job_id == "j1"
    assert len(client.submissions) == 1


def test_failed_stage_successor_preserves_evidence_without_replaying_predecessor(tmp_path):
    client = FakeClient()
    with replay(tmp_path, client) as run:
        first = first_stage(run)
        client.fail_status = "FAILED"
        with pytest.raises(HubJobFailed):
            second_stage(run, first)
    client.fail_status = None
    with replay(tmp_path, client) as run:
        first = first_stage(run)
        replacement = run.execute(
            "decode-corrected", stage_id=1, stage_name="decode", artifact=artifact(),
            inputs={"x": first.outputs["y"]},
            input_provenance={"x": "step:encode-0:y"},
            options="--compute_unit npu",
        )
        assert replacement.outputs["y"].tolist() == [4.0, 8.0]
        run.supersede_failed_step(
            "decode-0", successor_step_id="decode-corrected",
            reason="recompiled corrected decoder artifact after failed service job",
        )
        run.finish(final_output={"token_ids": [4, 8]},
                   final_output_source="step:decode-corrected",
                   quality=quality(), required_stages=["encode", "decode"])
    audit = audit_run(tmp_path)
    assert audit["status"] == "functional_complete_placement_unverified"
    assert audit["job_ids"] == ["j1", "j3"]
    assert audit["failed_job_ids"] == ["j2"]
    assert len(client.submissions) == 3
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["steps"][1]["superseded_by"] == "decode-corrected"
    assert record["completion"]["superseded_failed_job_ids"] == ["j2"]


def test_old_sdk_and_wrong_target_rejected_before_submission(tmp_path):
    with pytest.raises(RuntimeError, match="0.56.0"):
        replay(tmp_path, OldClient())
    client = FakeClient()
    with replay(tmp_path, client) as run:
        wrong = HubArtifact(**{**artifact().__dict__, "target_name": "Samsung Galaxy S24"})
        with pytest.raises(ValueError, match="another registration"):
            run.execute(
                "encode-0", stage_id=0, stage_name="encode", artifact=wrong,
                inputs={"x": np.array([1], dtype=np.float32)},
                input_provenance={"x": "request:real-input"},
            )
    assert not client.submissions
