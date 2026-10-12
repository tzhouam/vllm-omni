"""Auditable, resumable single-request Qualcomm AI Hub stage replay.

AI Hub releases a leased device after each job.  This is a host-controlled
functional replay on one *target registration*, never a resident device
session or an end-to-end device latency measurement.  Model adapters own the
stage graph, pre/postprocessing, sampling, state layout and quality checks.
This module only submits real tensor inputs to target-model artifacts and
preserves each dependent inference result.

Use qai-hub >= 0.56.0: earlier clients cannot request ``profile=False``.
Credentials are read by the SDK and are never serialized here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shlex
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


SCHEMA_VERSION = 1
_STEP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HEX256 = re.compile(r"^[0-9a-f]{64}$")
_HOST_TRANSFORMS = frozenset({
    "tokenize", "preprocess", "sample", "state_pack", "cache_update",
    "mask", "position", "reshape", "concat", "cast", "resample",
    "postprocess", "format",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _arrays(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("a stage needs named, nonempty tensor inputs or outputs")
    result: dict[str, np.ndarray] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tensor names must be nonempty strings")
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind not in "biufc":
            raise ValueError(f"tensor {name!r} must have a numeric or boolean dtype")
        result[name] = np.ascontiguousarray(array)
    return result


def _tensor_record(array: np.ndarray) -> dict[str, Any]:
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape), "layout": "C"},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header)
    data = memoryview(array).cast("B")
    for start in range(0, len(data), 1 << 20):
        digest.update(data[start:start + (1 << 20)])
    return {
        "sha256": digest.hexdigest(), "dtype": str(array.dtype),
        "shape": list(array.shape), "nbytes": int(array.nbytes), "layout": "C",
    }


def _tensor_records(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {name: _tensor_record(array) for name, array in arrays.items()}


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_npz(
    path: Path, expected: Mapping[str, dict[str, Any]], expected_file_sha256: str,
) -> dict[str, np.ndarray]:
    if _file_sha256(path) != expected_file_sha256:
        raise ValueError(f"saved tensor file changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        result = _arrays({name: archive[name] for name in archive.files})
    if _tensor_records(result) != expected:
        raise ValueError(f"saved tensor snapshot changed: {path}")
    return result


def _validate_lineage(
    source: str, tensor: dict[str, Any], prior_steps: Mapping[str, Mapping[str, Any]],
    *, transformed: bool = False,
) -> None:
    if source.startswith("request:") and source[8:]:
        return
    if source.startswith("step:"):
        parts = source.split(":", 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise ValueError(f"invalid step tensor provenance: {source}")
        predecessor = prior_steps.get(parts[1])
        if predecessor is None or predecessor["status"] != "completed":
            raise ValueError(f"unavailable predecessor in input provenance: {source}")
        output = predecessor["output_records"].get(parts[2])
        if output is None or (not transformed and output["sha256"] != tensor["sha256"]):
            raise ValueError(f"input does not match named predecessor output: {source}")
        return
    if source.startswith("host:"):
        parts = source.split(":", 2)
        if len(parts) != 3 or parts[1] not in _HOST_TRANSFORMS:
            raise ValueError(f"undeclared host transform in input provenance: {source}")
        _validate_lineage(parts[2], tensor, prior_steps, transformed=True)
        return
    raise ValueError(
        "input provenance must be request:<name>, step:<id>:<tensor>, "
        "or host:<allowed-transform>:<source>"
    )


@dataclass(frozen=True)
class HubArtifact:
    """One target-specific, compiled neural stage artifact.

    ``sha256`` is the digest of the downloaded compiled artifact or a hashed
    manifest that identifies it; the caller must record which in provenance.
    A source ONNX digest alone is not a target artifact identity.
    """

    artifact_id: str
    target_model_id: str
    sha256: str
    checkpoint: str
    revision: str
    layout_version: int
    precision: str
    target_name: str
    runtime: str
    shape_bucket: str
    hash_source: str = "compiled_artifact"

    def __post_init__(self) -> None:
        for name in (
            "artifact_id", "target_model_id", "checkpoint", "revision",
            "precision", "target_name", "runtime", "shape_bucket", "hash_source",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"artifact {name} is required")
        if not _HEX256.fullmatch(self.sha256):
            raise ValueError("artifact sha256 must be a lowercase SHA-256 digest")
        if type(self.layout_version) is not int or self.layout_version < 1:
            raise ValueError("artifact layout_version must be positive")
        if self.hash_source not in {"compiled_artifact", "compiled_manifest"}:
            raise ValueError("artifact hash_source must identify compiled target content")


@dataclass(frozen=True)
class HubStepResult:
    step_id: str
    job_id: str
    outputs: dict[str, np.ndarray]
    output_records: dict[str, dict[str, Any]]


class AmbiguousSubmission(RuntimeError):
    """A process stopped between SDK submission and durable job-ID capture."""


class HubJobFailed(RuntimeError):
    """An identified AI Hub inference job failed; its evidence is retained."""


class HubChainAdapter(Protocol):
    """Model-owned semantics; only ``HubReplay.execute`` performs neural work."""

    required_stages: Sequence[str]

    def run(self, replay: HubReplay, request: Any) -> Mapping[str, Any]: ...

    def verify(self, request: Any, final_output: Mapping[str, Any]) -> Mapping[str, Any]: ...


class HubReplay:
    """One batch-1 request, one target registration, serial dependent jobs.

    Construct one instance per evidence directory and close it after the chain.
    A per-target local-process lock prevents two local replay requests from
    overlapping. The remote service does not promise the same physical leased
    unit across jobs, so neither residency nor whole-request latency is claimed.
    """

    def __init__(
        self, evidence_dir: str | Path, *, target_name: str, model: str,
        checkpoint: str, revision: str, request_id: str, epoch: int = 0,
        stage_topology: Mapping[str, Sequence[str]] | None = None,
        client: Any = None, hub_module: Any = None,
    ) -> None:
        if not all(isinstance(x, str) and x for x in
                   (target_name, model, checkpoint, revision, request_id)):
            raise ValueError("target, model, checkpoint, revision and request ID are required")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if hub_module is None:
            import qai_hub as hub_module  # optional, only needed for live replay
        self.hub = hub_module
        self.client = client or hub_module.Client()
        if "profile" not in inspect.signature(self.client.submit_inference_job).parameters:
            raise RuntimeError("qai-hub >= 0.56.0 is required to submit with profile=False")
        exact = [device for device in self.client.get_devices(name=target_name)
                 if device.name == target_name]
        if len(exact) != 1:
            raise ValueError(f"target registration {target_name!r} has {len(exact)} exact matches")
        self.device = exact[0]
        self.evidence_dir = Path(evidence_dir).resolve()
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.run_path = self.evidence_dir / "run.json"
        identity = {
            "target_name": target_name, "model": model,
            "checkpoint": checkpoint, "revision": revision,
            "request_id": request_id, "epoch": epoch,
        }
        if stage_topology is None:
            stage_topology = {}
        topology = {name: list(dependencies) for name, dependencies in stage_topology.items()}
        if any(not isinstance(name, str) or not name or not isinstance(dependencies, list)
               or any(not isinstance(dep, str) or not dep for dep in dependencies)
               for name, dependencies in topology.items()):
            raise ValueError("stage topology needs named stages and dependencies")
        if any(dep not in topology for dependencies in topology.values() for dep in dependencies):
            raise ValueError("stage topology refers to an undeclared dependency")
        run_id = _sha256_json(identity)
        self._closed = False
        self._lock_file = None
        self._acquire_target_lock(target_name)
        try:
            if self.run_path.exists():
                self.record = json.loads(self.run_path.read_text(encoding="utf-8"))
                if self.record.get("schema_version") != SCHEMA_VERSION or self.record.get("run_id") != run_id:
                    raise ValueError("evidence directory belongs to another replay request")
                if self.record.get("identity") != identity:
                    raise ValueError("replay identity changed")
                if self.record.get("target_registration") != self._device_record():
                    raise ValueError("target registration changed since this replay began")
                if self.record.get("stage_topology") != topology:
                    raise ValueError("stage topology changed since this replay began")
            else:
                self.record = {
                    "schema_version": SCHEMA_VERSION, "run_id": run_id,
                    "tier": "aihub_hosted_functional_replay",
                    "facility": "Qualcomm AI Hub Workbench",
                    "identity": identity,
                    "stage_topology": topology,
                    "target_registration": self._device_record(),
                    "qai_hub_version": str(getattr(hub_module, "__version__", "unknown")),
                    "batch_size": 1, "active_requests": 1,
                    "profile": False,
                    "state_location": "versioned_host_snapshots_between_jobs",
                    "same_physical_device_across_jobs": False,
                    "whole_request_device_latency_measured": False,
                    "status": "running", "created_utc": _utc_now(),
                    "steps": [], "completion": None,
                }
                self._save()
        except BaseException:
            self.close()
            raise

    def _acquire_target_lock(self, target_name: str) -> None:
        try:
            import fcntl
        except ImportError as exc:
            raise RuntimeError("AI Hub replay target locking currently requires POSIX") from exc
        name = hashlib.sha256(target_name.encode("utf-8")).hexdigest()[:20]
        lock_path = Path(tempfile.gettempdir()) / f"omni-aihub-{name}.lock"
        self._lock_file = lock_path.open("a+b")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            self._lock_file = None
            raise RuntimeError(f"another batch-1 replay is active for {target_name}") from exc

    def _device_record(self) -> dict[str, Any]:
        return {
            "name": self.device.name, "os": self.device.os,
            "attributes": list(self.device.attributes),
        }

    def _save(self) -> None:
        self.record["updated_utc"] = _utc_now()
        _atomic_json(self.run_path, self.record)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._lock_file is not None:
                import fcntl
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
                self._lock_file = None

    def __enter__(self) -> HubReplay:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()

    def _remote_job(self, job: Any, step: Mapping[str, Any]) -> None:
        if job.job_id != step["job_id"] or job.device.name != self.device.name:
            raise ValueError("reattached inference job has another ID or target")
        if getattr(job.device, "os", None) != self.device.os:
            raise ValueError("reattached inference job has another target OS")
        if job.model.model_id != step["artifact"]["target_model_id"]:
            raise ValueError("reattached inference job has another target artifact")
        if step.get("input_dataset_id") and job.inputs.dataset_id != step["input_dataset_id"]:
            raise ValueError("reattached inference job has another input dataset")
        if getattr(job, "profiling", None) is not False:
            raise ValueError("AI Hub did not confirm profiling is disabled for this job")

    def execute(
        self, step_id: str, *, stage_id: int, stage_name: str,
        artifact: HubArtifact, inputs: Mapping[str, np.ndarray],
        input_provenance: Mapping[str, str], options: str = "",
    ) -> HubStepResult:
        """Submit or reattach one real-input job and download all output tensors.

        Repeating a completed step verifies its saved input and output hashes
        before returning. A submission with no durably captured job ID is
        deliberately not retried: automatic resubmission could duplicate work.
        """
        if self._closed:
            raise RuntimeError("replay is closed")
        if not _STEP_ID.fullmatch(step_id) or type(stage_id) is not int or stage_id < 0:
            raise ValueError("step ID or stage ID is invalid")
        if not isinstance(stage_name, str) or not stage_name or not isinstance(options, str):
            raise ValueError("stage name and options are required")
        if any(token.startswith("--max_profiler_iterations") for token in shlex.split(options)):
            raise ValueError("the replay runner owns --max_profiler_iterations")
        options = (options.strip() + " --max_profiler_iterations 1").strip()
        if self.record["stage_topology"] and stage_name not in self.record["stage_topology"]:
            raise ValueError("stage is not declared in the model topology")
        identity = self.record["identity"]
        if artifact.target_name != identity["target_name"]:
            raise ValueError("stage artifact targets another registration")
        if artifact.checkpoint != identity["checkpoint"] or artifact.revision != identity["revision"]:
            raise ValueError("stage artifact uses another checkpoint or revision")
        arrays = _arrays(inputs)
        if set(input_provenance) != set(arrays) or any(
            not isinstance(v, str) or not v for v in input_provenance.values()
        ):
            raise ValueError("each input needs a provenance string")
        input_records = _tensor_records(arrays)
        prior_steps = {row["step_id"]: row for row in self.record["steps"]}
        for name, source in input_provenance.items():
            # A step cannot cite its own output as an input on first submission.
            _validate_lineage(source, input_records[name],
                              {key: row for key, row in prior_steps.items() if key != step_id})
        specification = {
            "step_id": step_id, "stage_id": stage_id, "stage_name": stage_name,
            "artifact": asdict(artifact), "input_records": input_records,
            "input_provenance": dict(input_provenance), "options": options,
        }
        existing = next((step for step in self.record["steps"]
                         if step["step_id"] == step_id), None)
        if (self.record["status"] not in {"running", "job_failed"}
                and (existing is None or existing["status"] != "completed")):
            raise RuntimeError("replay is already finalized")
        if existing is None:
            step = dict(specification)
            step.update({
                "sequence": len(self.record["steps"]), "status": "prepared",
                "profile": False, "batch_size": 1, "active_requests": 1,
                "input_npz": f"{step_id}.inputs.npz",
                "output_npz": f"{step_id}.outputs.npz",
                "input_file_sha256": None, "output_file_sha256": None,
                "job_id": None, "input_dataset_id": None,
                "output_records": None,
                "placement": {"requested_options": options,
                              "actual_compute_units": None,
                              "source": "not verified by inference job"},
            })
            _atomic_npz(self.evidence_dir / step["input_npz"], arrays)
            step["input_file_sha256"] = _file_sha256(self.evidence_dir / step["input_npz"])
            self.record["steps"].append(step)
            self._save()
        else:
            step = existing
            if any(step.get(name) != value for name, value in specification.items()):
                raise ValueError(f"step {step_id} changed artifact, inputs, stage or options")
            _verified_npz(
                self.evidence_dir / step["input_npz"], input_records,
                step["input_file_sha256"],
            )
            if step["status"] == "completed":
                outputs = _verified_npz(
                    self.evidence_dir / step["output_npz"], step["output_records"],
                    step["output_file_sha256"],
                )
                return HubStepResult(step_id, step["job_id"], outputs, step["output_records"])
            if step["status"] == "submitting":
                raise AmbiguousSubmission(
                    f"{step_id}: submission outcome unknown; reconcile AI Hub jobs before resuming"
                )
            if step["status"] == "failed":
                raise HubJobFailed(f"{step_id}: existing job {step['job_id']} failed")
        if step["status"] == "prepared":
            target_model = self.client.get_model(artifact.target_model_id)
            if target_model.model_id != artifact.target_model_id:
                raise ValueError("AI Hub returned a different target-model ID")
            step["status"] = "submitting"
            self._save()
            # The SDK's default is profile=True. Never omit this argument.
            job = self.client.submit_inference_job(
                model=target_model, device=self.device,
                inputs={name: [array] for name, array in arrays.items()},
                name=f"omni-e2e-{self.record['run_id'][:12]}-{step_id}",
                options=options, profile=False,
            )
            if isinstance(job, list):
                raise RuntimeError("a single target registration returned multiple jobs")
            step["job_id"] = job.job_id
            step["input_dataset_id"] = job.inputs.dataset_id
            step["status"] = "submitted"
            step["submitted_utc"] = _utc_now()
            self._save()
        elif step["status"] == "submitted":
            job = self.client.get_job(step["job_id"])
        else:
            raise RuntimeError(f"unexpected replay step state: {step['status']}")
        self._remote_job(job, step)
        step["remote_profiling_enabled"] = job.profiling
        status = job.wait()
        step["job_status"] = status.code
        step["job_status_message"] = status.message or ""
        if status.code != "SUCCESS":
            step["status"] = "failed" if status.code in {"FAILED", "CANCELED"} else "submitted"
            self.record["status"] = "job_failed" if step["status"] == "failed" else "running"
            self._save()
            raise HubJobFailed(f"{step_id}: AI Hub job {step['job_id']} returned {status.code}")
        returned_profile = job.download_profile()
        if not isinstance(returned_profile, Mapping):
            raise ValueError("AI Hub returned an invalid inference profile payload")
        response_path = self.evidence_dir / f"{step_id}.inference_profile_response.json"
        _atomic_json(response_path, returned_profile)
        step["remote_profile_response_path"] = response_path.name
        step["remote_profile_response_sha256"] = _file_sha256(response_path)
        summary = returned_profile.get("execution_summary", {})
        times = summary.get("all_inference_times", []) if isinstance(summary, Mapping) else []
        details = returned_profile.get("execution_detail", [])
        step["remote_profile_iteration_samples"] = len(times) if isinstance(times, list) else None
        step["remote_profile_detail_rows"] = len(details) if isinstance(details, list) else None
        downloaded = job.download_output_data()
        if not isinstance(downloaded, Mapping) or not downloaded:
            raise RuntimeError(f"{step_id}: successful job returned no downloadable output")
        outputs: dict[str, np.ndarray] = {}
        for name, samples in downloaded.items():
            if not isinstance(samples, (tuple, list)) or len(samples) != 1:
                raise ValueError(f"{step_id}: output {name!r} is not a batch-1 tensor")
            outputs[name] = samples[0]
        outputs = _arrays(outputs)
        output_records = _tensor_records(outputs)
        _atomic_npz(self.evidence_dir / step["output_npz"], outputs)
        step["output_file_sha256"] = _file_sha256(self.evidence_dir / step["output_npz"])
        step["output_records"] = output_records
        step["status"] = "completed"
        step["completed_utc"] = _utc_now()
        self.record["status"] = "running"
        self._save()
        return HubStepResult(step_id, step["job_id"], outputs, output_records)

    def record_placement(
        self, step_id: str, *, actual_compute_units: Sequence[str],
        profile_job_id: str, raw_profile_path: str | Path,
    ) -> None:
        """Attach separately verified placement; a requested unit is not evidence."""
        step = next((row for row in self.record["steps"] if row["step_id"] == step_id), None)
        if step is None or step["status"] != "completed":
            raise ValueError("placement requires a completed inference step")
        if not actual_compute_units or any(not isinstance(unit, str) or not unit
                                           for unit in actual_compute_units):
            raise ValueError("actual compute-unit names are required")
        path = Path(raw_profile_path).resolve()
        if not profile_job_id or not path.is_file():
            raise ValueError("a saved raw profile and its job ID are required")
        if not path.is_relative_to(self.evidence_dir):
            raise ValueError("raw placement profile must be archived in this evidence directory")
        profile_job = self.client.get_job(profile_job_id)
        if (
            profile_job.job_id != profile_job_id
            or profile_job.device.name != self.device.name
            or profile_job.model.model_id != step["artifact"]["target_model_id"]
            or profile_job.get_status().code != "SUCCESS"
        ):
            raise ValueError("placement profile is not a successful job for this target artifact")
        raw_profile = json.loads(path.read_text(encoding="utf-8"))
        if raw_profile != profile_job.download_profile():
            raise ValueError("saved placement profile differs from AI Hub job output")
        details = raw_profile.get("execution_detail")
        if not isinstance(details, list):
            raise ValueError("raw placement profile has no per-operation detail")
        observed_units = {
            row.get("compute_unit") for row in details
            if isinstance(row, dict) and row.get("compute_unit")
            and row.get("type") not in {"Input", "Output"}
        }
        if not observed_units or observed_units != set(actual_compute_units):
            raise ValueError("reported compute units differ from raw profile operations")
        step["placement"] = {
            "requested_options": step["options"],
            "actual_compute_units": list(actual_compute_units),
            "source": "separate_aihub_profile",
            "profile_job_id": profile_job_id,
            "raw_profile_path": str(path),
            "raw_profile_sha256": _file_sha256(path),
            "input_shapes": {name: row["shape"] for name, row in step["input_records"].items()},
        }
        self._save()

    def supersede_failed_step(
        self, failed_step_id: str, *, successor_step_id: str, reason: str,
    ) -> None:
        """Retain a failed job but replace its logical stage with a new result.

        The successor must be a separately named, completed job for the same
        stage and request. The caller should derive its inputs from saved
        request data or completed predecessor output; no failed output is used.
        """
        if self._closed or not isinstance(reason, str) or not reason:
            raise ValueError("a live replay and a named correction reason are required")
        by_id = {step["step_id"]: step for step in self.record["steps"]}
        failed = by_id.get(failed_step_id)
        successor = by_id.get(successor_step_id)
        if failed is None or successor is None or failed_step_id == successor_step_id:
            raise ValueError("failed and successor step IDs must be distinct and recorded")
        if failed["status"] != "failed" or successor["status"] != "completed":
            raise ValueError("only a failed job can be superseded by a completed job")
        if (failed["stage_id"] != successor["stage_id"]
                or failed["stage_name"] != successor["stage_name"]
                or successor["sequence"] <= failed["sequence"]):
            raise ValueError("successor must be a later job for the same neural stage")
        if failed.get("superseded_by") is not None:
            if failed["superseded_by"] == successor_step_id and failed.get("supersession_reason") == reason:
                return
            raise ValueError("failed job was already superseded by another step")
        if successor.get("replaces_failed_step") is not None:
            raise ValueError("successor already replaces another failed step")
        failed["superseded_by"] = successor_step_id
        failed["supersession_reason"] = reason
        successor["replaces_failed_step"] = failed_step_id
        self._save()

    def finish(
        self, *, final_output: Mapping[str, Any], final_output_source: str,
        quality: Mapping[str, Any], required_stages: Sequence[str],
    ) -> dict[str, Any]:
        """Finalize only after every declared neural stage and quality gate.

        The adapter supplies the model-specific required stage set. A full
        chain pass is only as strong as that adapter's declared topology and
        reference checks; both remain visible in evidence for audit.
        """
        if self._closed or self.record["status"] != "running":
            raise RuntimeError("replay cannot be finalized from this state")
        all_steps = self.record["steps"]
        if not all_steps or any(
            step["status"] != "completed"
            and not (step["status"] == "failed" and step.get("superseded_by"))
            for step in all_steps
        ):
            raise ValueError("all neural stages must complete or have an explicit failed-job successor")
        steps = [step for step in all_steps if step["status"] == "completed"]
        if any("--max_profiler_iterations 1" not in step["options"] for step in steps):
            raise ValueError("functional replay requires one profiler iteration per job")
        if any(step.get("remote_profile_iteration_samples") != 1 for step in steps):
            raise ValueError("AI Hub did not confirm one inference timing sample per job")
        required = list(required_stages)
        if not required or len(required) != len(set(required)) or any(
            not isinstance(name, str) or not name for name in required
        ):
            raise ValueError("unique required neural stage names are needed")
        completed = {step["stage_name"] for step in steps}
        if not set(required).issubset(completed):
            raise ValueError(f"missing neural stages: {sorted(set(required) - completed)}")
        declared = set(self.record["stage_topology"])
        if not declared or set(required) != declared:
            raise ValueError("required stages must equal the recorded model topology")
        for step in steps:
            dependencies = self.record["stage_topology"][step["stage_name"]]
            predecessors = {row["step_id"]: row for row in steps
                            if row["sequence"] < step["sequence"]}
            for dependency in dependencies:
                matching = [row for row in predecessors.values() if row["stage_name"] == dependency]
                if not matching:
                    raise ValueError(f"{step['step_id']} has no predecessor stage {dependency}")
            if dependencies and not any("step:" in source for source in step["input_provenance"].values()):
                raise ValueError(f"{step['step_id']} lacks input lineage from an earlier neural job")
        if not isinstance(final_output_source, str) or not final_output_source:
            raise ValueError("final output needs source step provenance")
        if not any(step["step_id"] in final_output_source for step in steps):
            raise ValueError("final output source must name a completed step")
        if not isinstance(quality, Mapping) or type(quality.get("passed")) is not bool:
            raise ValueError("quality report requires an explicit boolean pass")
        if not isinstance(quality.get("reference"), str) or not quality["reference"]:
            raise ValueError("quality report needs a named reference")
        if not isinstance(quality.get("checks"), list) or not quality["checks"]:
            raise ValueError("quality report needs nonempty model-specific checks")
        # JSON is used for text/action metadata; large PCM arrays may be passed
        # as lists of numbers but adapters should save them as separate media.
        payload = dict(final_output)
        try:
            json.dumps(payload, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("final output must be finite JSON; save large media separately") from exc
        output_path = self.evidence_dir / "final_output.json"
        _atomic_json(output_path, payload)
        completion = {
            "required_stages": required, "completed_stages": sorted(completed),
            "neural_job_ids": [step["job_id"] for step in steps],
            "superseded_failed_job_ids": [step["job_id"] for step in all_steps
                                          if step["status"] == "failed"],
            "final_output_path": output_path.name,
            "final_output_sha256": _file_sha256(output_path),
            "final_output_source": final_output_source,
            "quality": dict(quality), "completed_utc": _utc_now(),
        }
        self.record["completion"] = completion
        if quality["passed"]:
            placement_verified = all(
                step["placement"]["actual_compute_units"] is not None
                for step in steps
            )
            self.record["status"] = (
                "complete" if placement_verified
                else "functional_complete_placement_unverified"
            )
        else:
            self.record["status"] = "quality_failed"
        self._save()
        return completion

    def run_adapter(self, adapter: HubChainAdapter, request: Any) -> dict[str, Any]:
        """Execute a model-owned chain and its reference checks in order."""
        final_output = adapter.run(self, request)
        quality = adapter.verify(request, final_output)
        last_step = self.record["steps"][-1]["step_id"]
        return self.finish(
            final_output=final_output, final_output_source=f"step:{last_step}",
            quality=quality, required_stages=adapter.required_stages,
        )


def audit_run(evidence_dir: str | Path) -> dict[str, Any]:
    """Offline integrity and status check for matrix promotion.

    This does not independently validate model semantics or remote server
    authenticity. It does reject missing/changed local raw tensors, broken
    direct tensor lineage, and unverified placement masquerading as complete.
    """
    root = Path(evidence_dir).resolve()
    record = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA_VERSION or record.get("batch_size") != 1:
        raise ValueError("invalid batch-1 replay record")
    if record.get("active_requests") != 1 or record.get("profile") is not False:
        raise ValueError("replay used an unsupported concurrency or profile setting")
    identity = record["identity"]
    if record["run_id"] != _sha256_json(identity):
        raise ValueError("replay run ID does not match its identity")
    topology = record.get("stage_topology")
    if not isinstance(topology, dict) or any(
        not isinstance(name, str) or not name or not isinstance(dependencies, list)
        or any(not isinstance(dep, str) or dep not in topology for dep in dependencies)
        for name, dependencies in topology.items()
    ):
        raise ValueError("replay stage topology is invalid")
    prior: dict[str, dict[str, Any]] = {}
    for index, step in enumerate(record["steps"]):
        if step["sequence"] != index or step["step_id"] in prior:
            raise ValueError("replay step order or identity is invalid")
        if step["artifact"]["checkpoint"] != identity["checkpoint"] or \
                step["artifact"]["revision"] != identity["revision"] or \
                step["artifact"]["target_name"] != identity["target_name"]:
            raise ValueError("step artifact differs from replay identity")
        stage_name = step["stage_name"]
        if stage_name not in topology or any(
            dep not in {row["stage_name"] for row in prior.values()
                        if row["status"] == "completed"}
            for dep in topology[stage_name]
        ):
            if step["status"] != "failed":
                raise ValueError("replay step violates declared stage topology")
        if topology[stage_name] and not any(
            "step:" in source for source in step["input_provenance"].values()
        ):
            raise ValueError("dependent neural stage lacks predecessor tensor lineage")
        _verified_npz(root / step["input_npz"], step["input_records"],
                      step["input_file_sha256"])
        for name, source in step["input_provenance"].items():
            _validate_lineage(source, step["input_records"][name], prior)
        if step["status"] == "completed":
            if not step["job_id"] or not step["input_dataset_id"]:
                raise ValueError("completed step lacks AI Hub job or dataset ID")
            if step.get("remote_profiling_enabled") is not False:
                raise ValueError("completed step lacks service confirmation of profile=False")
            profile_response = root / step["remote_profile_response_path"]
            if _file_sha256(profile_response) != step["remote_profile_response_sha256"]:
                raise ValueError("inference profile response changed")
            response = json.loads(profile_response.read_text(encoding="utf-8"))
            summary = response.get("execution_summary", {})
            times = summary.get("all_inference_times", []) if isinstance(summary, dict) else []
            details = response.get("execution_detail", [])
            if step["remote_profile_iteration_samples"] != (len(times) if isinstance(times, list) else None) \
                    or step["remote_profile_detail_rows"] != (len(details) if isinstance(details, list) else None):
                raise ValueError("inference profile counts differ from raw response")
            _verified_npz(root / step["output_npz"], step["output_records"],
                          step["output_file_sha256"])
            placement = step["placement"]
            if placement["actual_compute_units"] is not None:
                raw_profile = Path(placement["raw_profile_path"])
                if not raw_profile.is_file() or _file_sha256(raw_profile) != placement["raw_profile_sha256"]:
                    raise ValueError("raw placement profile changed or disappeared")
                if not placement.get("profile_job_id"):
                    raise ValueError("placement profile lacks a job ID")
                profile = json.loads(raw_profile.read_text(encoding="utf-8"))
                units = {
                    row.get("compute_unit") for row in profile.get("execution_detail", [])
                    if isinstance(row, dict) and row.get("compute_unit")
                    and row.get("type") not in {"Input", "Output"}
                }
                if units != set(placement["actual_compute_units"]):
                    raise ValueError("recorded placement differs from raw profile")
        elif step["status"] == "failed":
            if not step.get("job_id") or step.get("job_status") not in {"FAILED", "CANCELED"}:
                raise ValueError("failed step lacks a recorded terminal AI Hub job")
        prior[step["step_id"]] = step
    for failed in (step for step in prior.values() if step["status"] == "failed"):
        successor = prior.get(failed.get("superseded_by"))
        if successor is None:
            if record["status"] in {"complete", "functional_complete_placement_unverified"}:
                raise ValueError("completed replay contains an unresolved failed job")
            continue
        if (successor["status"] != "completed"
                or successor.get("replaces_failed_step") != failed["step_id"]
                or successor["stage_id"] != failed["stage_id"]
                or successor["stage_name"] != failed["stage_name"]
                or successor["sequence"] <= failed["sequence"]
                or not failed.get("supersession_reason")):
            raise ValueError("failed-job supersession record is inconsistent")
    if record["status"] in {"complete", "functional_complete_placement_unverified"}:
        completion = record.get("completion")
        if not completion or completion["quality"].get("passed") is not True:
            raise ValueError("completed replay lacks passing quality evidence")
        if set(completion["required_stages"]) != set(topology):
            raise ValueError("completed replay omits a model neural stage")
        if completion["neural_job_ids"] != [step["job_id"] for step in record["steps"]
                                            if step["status"] == "completed"]:
            raise ValueError("completion job list differs from neural steps")
        if completion.get("superseded_failed_job_ids") != [
            step["job_id"] for step in record["steps"] if step["status"] == "failed"
        ]:
            raise ValueError("completion failed-job list differs from archived attempts")
        if not any(step_id in completion["final_output_source"] for step_id in prior):
            raise ValueError("final output lacks completed-step provenance")
        if any(step["status"] not in {"completed", "failed"} for step in record["steps"]):
            raise ValueError("completed replay contains an unresolved neural submission")
        if any("--max_profiler_iterations 1" not in step["options"] for step in record["steps"]):
            raise ValueError("completed replay used repeated inference iterations")
        if any(step.get("remote_profile_iteration_samples") != 1
               for step in record["steps"] if step["status"] == "completed"):
            raise ValueError("completed replay used multiple inference iterations")
        placement_verified = all(
            step["placement"]["actual_compute_units"] is not None
            for step in record["steps"] if step["status"] == "completed"
        )
        if (record["status"] == "complete") != placement_verified:
            raise ValueError("replay status disagrees with placement evidence")
        output = root / completion["final_output_path"]
        if _file_sha256(output) != completion["final_output_sha256"]:
            raise ValueError("final output changed")
    return {
        "run_id": record["run_id"], "status": record["status"],
        "model": identity["model"], "target_name": identity["target_name"],
        "step_count": len(record["steps"]),
        "job_ids": [step["job_id"] for step in record["steps"]
                    if step["status"] == "completed"],
        "failed_job_ids": [step["job_id"] for step in record["steps"]
                           if step["status"] == "failed"],
    }
