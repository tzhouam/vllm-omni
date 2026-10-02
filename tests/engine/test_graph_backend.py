"""The real StageRuntime/StagePool with a subprocess graph worker."""

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omni_stage_contracts import BufferRef, StateHandle, negotiate
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    merge_pipeline_deploy,
)
from vllm_omni.engine.resource_ledger import ResourceLedger, ResourceUnavailable
from vllm_omni.engine.stage_runtime import StageRuntime

ROOT = Path(__file__).resolve().parents[2]


def artifact(tmp_path, *, schema_version=1, qualified=True):
    (tmp_path / "graph.onnx").write_bytes(b"synthetic graph, not model evidence")
    np.savez(tmp_path / "inputs.npz", x=np.ones((2, 3), dtype=np.float32))
    files = {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in ("graph.onnx", "inputs.npz")}
    metadata = {"graph_file": "graph.onnx", "example_inputs_file": "inputs.npz"}
    if schema_version == 2:
        descriptor = {
            "checkpoint_id": "synthetic/test", "checkpoint_revision": "fixed-test",
            "precision": "float32", "runtime": "ort-cpu", "runtime_version": "test",
            "target_abi": "linux-x86_64", "adapter_version": "test",
            "exporter_version": "test", "compiler_version": "none",
            "state_layout_version": 1,
            "shape_bucket": "2x3", "calibration_file": None,
            "numerical_validation": {"file": "numeric.json", "passed": qualified},
            "task_validation": {"file": "task.json", "passed": qualified},
        }
        for name in ("numeric.json", "task.json"):
            kind = "numerical" if name == "numeric.json" else "task"
            (tmp_path / name).write_text(json.dumps({
                "schema_version": 1, "kind": kind,
                "checkpoint_id": descriptor["checkpoint_id"],
                "checkpoint_revision": descriptor["checkpoint_revision"],
                "precision": descriptor["precision"],
                "shape_bucket": descriptor["shape_bucket"],
                "target_abi": descriptor["target_abi"],
                "checks": [{"name": "test_error", "observed": 0.0 if qualified else 1.0,
                            "comparison": "<=", "limit": 0.1}],
            }))
            files[name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        metadata["artifact"] = descriptor
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "component": "test",
                "files": files,
                "metadata": metadata,
            }
        )
    )
    return path


@pytest.fixture
def graph_runtime(tmp_path, monkeypatch):
    config_path = tmp_path / "worker.json"
    config_path.write_text("{}")
    monkeypatch.setenv("VLLM_OMNI_FAKE_WORKER_JSON", str(config_path))
    monkeypatch.setenv("VLLM_OMNI_EXTERNAL_WORKER_ORT_CPU", str(ROOT / "tests/edge/local/external/fake_worker.py"))
    monkeypatch.setenv("VLLM_OMNI_EXTERNAL_PYTHON_ORT_CPU", sys.executable)
    runtimes = []

    def create(
        *, stages=1, capacity=1024 << 20, worker_config=None, start=True, chain=False, real_ort=False,
        entrypoint=None, manifest_schema_version=1, artifact_qualified=True
    ):
        config_path.write_text(json.dumps(worker_config or {}))
        manifest = artifact(tmp_path, schema_version=manifest_schema_version, qualified=artifact_qualified)
        if real_ort:
            onnx = pytest.importorskip("onnx")
            pytest.importorskip("onnxruntime")
            from onnx import TensorProto, helper

            graph = helper.make_graph(
                [helper.make_node("Add", ["x", "x"], ["x_out"])],
                "double",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
                [helper.make_tensor_value_info("x_out", TensorProto.FLOAT, [2, 3])],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
            onnx.save(model, tmp_path / "graph.onnx")
            data = json.loads(manifest.read_text())
            data["files"]["graph.onnx"] = hashlib.sha256((tmp_path / "graph.onnx").read_bytes()).hexdigest()
            manifest.write_text(json.dumps(data))
            monkeypatch.delenv("VLLM_OMNI_EXTERNAL_WORKER_ORT_CPU", raising=False)
        pipeline = PipelineConfig(
            model_type="test_graph",
            stages=tuple(
                StagePipelineConfig(
                    stage_id=i,
                    model_stage="graph",
                    execution_type=StageExecutionType.GRAPH,
                    final_output=(not chain or i == stages - 1),
                    final_output_type="latent",
                    input_sources=(i - 1,) if chain and i else (),
                    custom_process_input_func=(
                        "tests.engine.test_graph_backend.graph_handoff" if chain and i else None
                    ),
                )
                for i in range(stages)
            ),
        )
        deploy = DeployConfig(
            async_chunk=False,
            stages=[
                StageDeployConfig(
                    stage_id=i,
                    backend={
                        "name": "external.graph.v1",
                        "route": "ort-cpu",
                        "manifest": str(manifest),
                        "start_timeout_s": 5,
                    },
                    resource_budget={"capacities": {"host_ram": capacity}, "demands": {"host_ram": 512 << 20}},
                )
                for i in range(stages)
            ],
        )
        if entrypoint is not None:
            import yaml

            from vllm_omni.config.pipeline_registry import OMNI_PIPELINES

            monkeypatch.setitem(OMNI_PIPELINES, pipeline.model_type, pipeline)
            (tmp_path / "config.json").write_text('{"model_type":"bert"}')
            deploy_path = tmp_path / "deploy.yaml"
            deploy_path.write_text(
                yaml.safe_dump(
                    {
                        "pipeline": pipeline.model_type,
                        "async_chunk": False,
                        "stages": [
                            {"stage_id": s.stage_id, "backend": s.backend, "resource_budget": s.resource_budget}
                            for s in deploy.stages
                        ],
                    }
                )
            )
            engine = entrypoint(
                model=str(tmp_path), deploy_config=str(deploy_path), stage_init_timeout=10, init_timeout=30
            )
            runtimes.append(engine)
            return engine
        configs = [s.to_omegaconf() for s in merge_pipeline_deploy(pipeline, deploy)]
        runtime = StageRuntime(configs, "local-graph", "", stage_init_timeout=5, async_chunk=False)
        runtimes.append(runtime)
        if start:
            runtime.initialize()
        return runtime

    yield create
    for runtime in runtimes:
        runtime.shutdown()


def test_contract_import_does_not_import_engine():
    code = "import omni_stage_contracts; import sys; assert not ({'torch','vllm','vllm_omni'} & set(sys.modules))"
    subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True)


def test_device_init_lock_is_host_local_and_preserves_exclusion(tmp_path, monkeypatch):
    import os

    from vllm_omni import host
    from vllm_omni._filelock_compat import flock_exclusive_nb, funlock
    from vllm_omni.engine.stage_init_utils import device_init_lock_path, open_device_lock_file
    from vllm_omni.host import paths

    assert host is not None
    monkeypatch.setattr(paths, "device_lock_directory", lambda: str(tmp_path))
    lock_path = device_init_lock_path(0)
    assert Path(lock_path).parent == tmp_path
    first, writable = open_device_lock_file(lock_path)
    second = None
    try:
        assert writable
        flock_exclusive_nb(first)
        second, _ = open_device_lock_file(lock_path)
        with pytest.raises(BlockingIOError):
            flock_exclusive_nb(second)
        funlock(first)
        flock_exclusive_nb(second)
        funlock(second)
    finally:
        os.close(first)
        if second is not None:
            os.close(second)


def test_graph_metadata_matches_structured_and_legacy_config():
    from vllm_omni.config import VllmOmniConfig, VllmOmniGraphStageConfig
    from vllm_omni.engine.stage_init_utils import (
        extract_legacy_stage_metadata,
        extract_stage_metadata_from_omni_stage_config,
    )

    pipeline = PipelineConfig(
        model_type="test_graph",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="graph",
                execution_type=StageExecutionType.GRAPH,
                final_output=True,
                final_output_type="latent",
            ),
        ),
    )
    deploy = DeployConfig(
        async_chunk=False,
        stages=[
            StageDeployConfig(
                stage_id=0,
                backend={"name": "external.graph.v1"},
                resource_budget={"capacities": {"ram": 1024}, "demands": {"ram": 512}},
            )
        ],
    )
    config = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_by_id(0)
    assert isinstance(config, VllmOmniGraphStageConfig)
    structured = extract_stage_metadata_from_omni_stage_config(config)
    legacy = extract_legacy_stage_metadata(merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf())
    assert structured.stage_type == legacy.stage_type == "graph"
    assert type(structured.default_sampling_params) is type(legacy.default_sampling_params)


def test_language_independent_contract_fixtures():
    from omni_stage_contracts import StageEvent, StageRequest

    fixture = json.loads((ROOT / "packages/omni-stage-contracts/conformance/v1.json").read_text())
    negotiate(fixture["protocol_version"], fixture["required_features"])
    BufferRef(**fixture["buffer"])
    StageRequest(**fixture["request"])
    StageEvent(**fixture["event"])
    state = StateHandle(**fixture["state"])
    for override in fixture["reject_buffer_overrides"]:
        with pytest.raises(ValueError):
            BufferRef(**{**fixture["buffer"], **override})
    for override in fixture["stale_state_overrides"]:
        assert not state.accepts(StateHandle(**{**fixture["state"], **override}))


def test_external_weight_payload_must_be_in_manifest(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    from vllm_omni.edge.local.external.worker_ort import _verify_artifact_members

    weight = numpy_helper.from_array(np.ones((32, 32), dtype=np.float32), "w")
    graph = helper.make_graph([helper.make_node("Add", ["x", "w"], ["y"])], "external", [], [], [weight])
    path = tmp_path / "model.onnx"
    onnx.save_model(
        helper.make_model(graph),
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    with pytest.raises(ValueError, match="absent from verified manifest"):
        _verify_artifact_members(str(path), [str(path)])
    _verify_artifact_members(str(path), [str(path), str(tmp_path / "weights.bin")])


def test_contract_rejects_shape_and_generation_mismatch():
    with pytest.raises(ValueError):
        BufferRef("x", "a", "1", "float32", (2, 3), 23)
    with pytest.raises(ValueError):
        negotiate(3)
    with pytest.raises(ValueError):
        negotiate(1, ["zero-copy"])
    a = StateHandle("s", "ort", "sha", worker_generation="old")
    b = StateHandle("s", "ort", "sha", worker_generation="new")
    assert not a.accepts(b)


def test_graph_v2_artifact_validation_gate(graph_runtime):
    runtime = graph_runtime(manifest_schema_version=2, start=False)
    with pytest.raises(ValueError, match="cannot bind v2 artifact ABI"):
        runtime.initialize()
    assert runtime.resource_ledger.snapshot()["owners"] == []
    rejected = graph_runtime(manifest_schema_version=2, artifact_qualified=False, start=False)
    with pytest.raises(ValueError, match="failed numerical or task validation"):
        rejected.initialize()
    assert rejected.resource_ledger.snapshot()["owners"] == []


def test_atomic_shared_constraints_and_quarantine():
    ledger = ResourceLedger({"ram": 100, "wsl": 60})
    first = ledger.reserve("wsl", {"ram": 50, "wsl": 50})
    with pytest.raises(ResourceUnavailable):
        ledger.reserve("igpu+npu", {"ram": 60})
    with pytest.raises(ResourceUnavailable):
        ledger.reserve("second-wsl", {"ram": 10, "wsl": 20})
    assert ledger.snapshot()["reserved"] == {"ram": 50, "wsl": 50}
    assert not ledger.release(first, drained=False)
    assert ledger.snapshot()["quarantined"] == ["wsl"]
    assert ledger.release(first, drained=True)
    second = ledger.reserve("wsl", {"ram": 5})
    assert not ledger.release(first, drained=True)
    assert ledger.snapshot()["reserved"]["ram"] == 5
    ledger.release(second, drained=True)


@pytest.mark.parametrize(
    "kinds",
    [
        ("cpu",),
        ("cpu", "gpu_integrated"),
        ("cpu", "npu"),
        ("cpu", "gpu_integrated", "npu"),
        ("cpu", "gpu_integrated", "npu", "gpu_discrete"),
    ],
)
def test_integrated_topology_has_one_physical_ram(kinds):
    from vllm_omni.edge.topology import describe_topology

    devices = [
        SimpleNamespace(device_id=k, kind=k, extra={}, memory_pool="vram" if k == "gpu_discrete" else "host_ram")
        for k in kinds
    ]
    descriptions = describe_topology(devices, machine_id="pc", domain_id="windows")
    for descriptor in descriptions:
        if descriptor.integration != "discrete":
            assert descriptor.memory_pool_ids == ("pc:ram",)
        assert descriptor.power_domain_ids == ()  # unknown is not invented


async def wait_output(pool):
    async def poll():
        while (out := pool.poll_graph_output(0)) is None:
            await asyncio.sleep(0.005)
        return out

    return await asyncio.wait_for(poll(), timeout=10)


@pytest.mark.asyncio
async def test_runtime_load_submit_lease_and_shutdown(graph_runtime):
    runtime = graph_runtime()
    pool = runtime.stage_pools[0]
    state = SimpleNamespace(sampling_params_list=[None])
    prompt = {"tensors": {"x": np.ones((2, 3), dtype=np.float32)}}
    await pool.submit_initial("a", state, prompt)
    output = await wait_output(pool)
    np.testing.assert_array_equal(output.custom_output["tensors"]["x_out"], 2)
    with pytest.raises(ResourceUnavailable):
        await pool.submit_initial("b", state, prompt)
    output.release_stage_buffers()
    await asyncio.sleep(0)
    await pool.submit_initial("b", state, prompt)
    (await wait_output(pool)).release_stage_buffers()
    runtime.shutdown()
    assert runtime.resource_ledger.snapshot()["reserved"] == {"host_ram": 0}


def test_joint_admission_before_any_worker(graph_runtime):
    runtime = graph_runtime(stages=2, chain=True, capacity=600 << 20, start=False)
    with pytest.raises(ResourceUnavailable):
        runtime.initialize()
    assert runtime.resource_ledger.snapshot()["owners"] == []
    assert runtime.stage_pools == []


@pytest.mark.parametrize("fanout", [False, True])
def test_unimplemented_graph_topologies_refused_before_loading(graph_runtime, fanout):
    runtime = graph_runtime(stages=3, chain=fanout, start=False)
    if fanout:
        runtime._stage_configs[2].engine_input_source = [0]
    with pytest.raises(ValueError, match="linear chain with exactly one final output"):
        runtime.initialize()
    assert runtime.resource_ledger is None
    assert runtime.stage_pools == []


def test_placement_failure_releases_load(graph_runtime):
    runtime = graph_runtime(worker_config={"load": {"fraction_on_target": 0.0}}, start=False)
    with pytest.raises(RuntimeError, match="REFUSE_EP_PLACEMENT"):
        runtime.initialize()
    assert runtime.resource_ledger.snapshot()["owners"] == []


@pytest.mark.asyncio
async def test_worker_failure_produces_error_not_success(graph_runtime):
    runtime = graph_runtime(worker_config={"die_on": "run"})
    pool = runtime.stage_pools[0]
    await pool.submit_initial(
        "a", SimpleNamespace(sampling_params_list=[None]), {"tensors": {"x": np.ones((2, 3), dtype=np.float32)}}
    )
    output = await wait_output(pool)
    assert output.finished and output.error


@pytest.mark.asyncio
async def test_cancel_fences_inflight_graph(graph_runtime):
    runtime = graph_runtime(worker_config={"run": {"sleep_s": 30}})
    pool = runtime.stage_pools[0]
    await pool.submit_initial(
        "a", SimpleNamespace(sampling_params_list=[None]), {"tensors": {"x": np.ones((2, 3), dtype=np.float32)}}
    )
    await asyncio.sleep(0.05)
    await asyncio.wait_for(pool.abort_requests(["a"]), timeout=8)
    assert pool.poll_graph_output(0) is None
    assert runtime.resource_ledger.snapshot()["reserved"]["host_ram"] == 0


def graph_handoff(outputs, prompt, requires_multimodal_data):
    return {"tensors": {"x": outputs[0].custom_output["tensors"]["x_out"]}}


@pytest.mark.asyncio
@pytest.mark.parametrize("event_driven", [False, True])
async def test_two_graphs_through_existing_orchestrator(graph_runtime, event_driven):
    from vllm_omni.engine.messages import StageSubmissionMessage
    from vllm_omni.engine.orchestrator import Orchestrator

    runtime = graph_runtime(stages=2, chain=True)
    request_queue, output_queue, rpc_queue = asyncio.Queue(), asyncio.Queue(), asyncio.Queue()
    orch = Orchestrator(request_queue, output_queue, rpc_queue, runtime.stage_pools)
    loop = asyncio.create_task(orch._orchestration_loop_event_driven() if event_driven else orch._orchestration_loop())
    prompt = {"tensors": {"x": np.ones((2, 3), dtype=np.float32)}}
    try:
        await orch._handle_add_request(
            StageSubmissionMessage(
                type="add_request",
                request_id="pipeline",
                prompt=prompt,
                original_prompt=prompt,
                output_prompt_text=None,
                sampling_params_list=[None, None],
                final_stage_id=1,
                final_output_stage_ids=[1],
                preprocess_ms=0,
                request_timestamp=0,
                enqueue_ts=0,
            )
        )
        from vllm_omni.engine.messages import StageMetricsMessage

        while True:
            result = await asyncio.wait_for(output_queue.get(), timeout=10)
            if not isinstance(result, StageMetricsMessage):
                break
        assert result.finished
        np.testing.assert_array_equal(result.engine_outputs.custom_output["tensors"]["x_out"], 4)
        result.engine_outputs.release_stage_buffers()
        await asyncio.sleep(0)
        assert not orch.request_states
        assert all(pool.stage_client._active is None for pool in runtime.stage_pools)
    finally:
        orch._shutdown_event.set()
        await asyncio.wait_for(loop, timeout=5)


@pytest.mark.asyncio
async def test_stale_ack_cannot_release_reused_request_id(graph_runtime):
    runtime = graph_runtime()
    pool = runtime.stage_pools[0]
    state = SimpleNamespace(sampling_params_list=[None])
    prompt = {"tensors": {"x": np.ones((2, 3), dtype=np.float32)}}
    await pool.submit_initial("same", state, prompt)
    output = await wait_output(pool)
    stale_ack = output._stage_release
    output.release_stage_buffers()
    await asyncio.sleep(0)
    await pool.submit_initial("same", state, prompt)
    next_output = await wait_output(pool)
    stale_ack()
    await asyncio.sleep(0)
    assert pool.stage_client._active == "same"
    next_output.release_stage_buffers()


@pytest.mark.asyncio
async def test_real_ort_cpu_graph_through_stage_runtime(graph_runtime):
    runtime = graph_runtime(real_ort=True)
    pool = runtime.stage_pools[0]
    await pool.submit_initial(
        "ort", SimpleNamespace(sampling_params_list=[None]), {"tensors": {"x": np.ones((2, 3), dtype=np.float32)}}
    )
    output = await wait_output(pool)
    assert not output.error
    np.testing.assert_array_equal(output.custom_output["tensors"]["x_out"], 2)
    assert pool.stage_client.load_report.target_nodes > 0
    output.release_stage_buffers()


@pytest.mark.asyncio
async def test_public_async_omni_graph_requests(graph_runtime):
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    engine = graph_runtime(entrypoint=AsyncOmni)
    for request_id in ("first", "second"):
        outputs = [
            out
            async for out in engine.generate(
                {"tensors": {"x": np.ones((2, 3), dtype=np.float32)}}, request_id=request_id
            )
        ]
        assert len(outputs) == 1
        assert outputs[0].request_id == request_id
        assert outputs[0].custom_output["tensors"]["x_out"].shape == (2, 3)


def test_public_sync_omni_graph_requests(graph_runtime):
    from vllm_omni.entrypoints.omni import Omni

    engine = graph_runtime(entrypoint=Omni)
    for _ in range(2):
        outputs = engine.generate({"tensors": {"x": np.ones((2, 3), dtype=np.float32)}}, use_tqdm=False)
        assert len(outputs) == 1
        assert outputs[0].custom_output["tensors"]["x_out"].shape == (2, 3)


@pytest.mark.asyncio
async def test_graph_ingress_stays_bounded_until_consumer_ack(graph_runtime):
    from vllm_omni.engine.messages import OutputMessage
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    frontend = graph_runtime(entrypoint=AsyncOmni)
    engine = frontend.engine
    prompt = {"tensors": {"x": np.ones((2, 3), dtype=np.float32)}}
    engine.add_request("slow", prompt)
    for i in range(100):
        with pytest.raises(ResourceUnavailable, match="capacity is one"):
            engine.add_request(f"blocked-{i}", prompt)

    async def take():
        while True:
            msg = await engine.try_get_output_async()
            if isinstance(msg, OutputMessage):
                return msg.engine_outputs
            await asyncio.sleep(0.005)

    output = await asyncio.wait_for(take(), 5)
    with pytest.raises(ResourceUnavailable):
        engine.add_request("still-blocked", prompt)
    stale_release = output._stage_release
    output.release_stage_buffers()
    await asyncio.sleep(0.01)
    engine.add_request("next", prompt)
    stale_release()
    with pytest.raises(ResourceUnavailable):
        engine.add_request("late-ack-must-not-release-next", prompt)
    (await asyncio.wait_for(take(), 5)).release_stage_buffers()
