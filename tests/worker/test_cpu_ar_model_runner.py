"""CPU Omni runners/workers: MRO and override resolution (import-only, CPU)."""

from __future__ import annotations

import pytest
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.cpu_worker import CPUWorker
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.platforms.cpu.worker.cpu_ar_model_runner import (
    CPUARModelRunner,
    CPUGenerationModelRunner,
    _CPUOmniRunnerMixin,
)
from vllm_omni.platforms.cpu.worker.cpu_ar_worker import CPUARWorker, CPUGenerationWorker
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ar_runner_mro():
    mro = CPUARModelRunner.__mro__
    assert (
        mro.index(_CPUOmniRunnerMixin)
        < mro.index(GPUARModelRunner)
        < mro.index(CPUModelRunner)
        < mro.index(GPUModelRunner)
    )
    # CPU overrides win over the GPU runner implementations
    assert CPUARModelRunner._init_device_properties is _CPUOmniRunnerMixin._init_device_properties
    assert CPUARModelRunner._sync_device is _CPUOmniRunnerMixin._sync_device
    assert CPUARModelRunner.capture_model is _CPUOmniRunnerMixin.capture_model
    assert CPUARModelRunner._capture_talker_mtp_graphs is _CPUOmniRunnerMixin._capture_talker_mtp_graphs
    # vLLM CPU shims (triton/tensor post-processing, warm-up) are reachable
    assert CPUARModelRunner.warming_up_model is CPUModelRunner.warming_up_model
    assert CPUARModelRunner._postprocess_triton is CPUModelRunner._postprocess_triton
    # omni connector mixin is present for the worker-class validation
    assert issubclass(CPUARModelRunner, OmniConnectorModelRunnerMixin)


def test_generation_runner_mro():
    mro = CPUGenerationModelRunner.__mro__
    assert mro.index(GPUGenerationModelRunner) < mro.index(CPUModelRunner)
    assert CPUGenerationModelRunner.capture_model is _CPUOmniRunnerMixin.capture_model


def test_workers_compose_cpu_worker_and_omni_mixin():
    for worker_cls, runner_cls in ((CPUARWorker, CPUARModelRunner), (CPUGenerationWorker, CPUGenerationModelRunner)):
        assert issubclass(worker_cls, CPUWorker) and issubclass(worker_cls, OmniWorkerMixin)
        assert worker_cls.model_runner_cls is runner_cls
        assert hasattr(worker_cls, "handle_sleep_task") and hasattr(worker_cls, "handle_wake_task")


def test_worker_cls_validation_accepts_cpu_workers():
    from vllm_omni.worker.omni_connector_validation import validate_worker_omni_connector

    validate_worker_omni_connector("vllm_omni.platforms.cpu.worker.cpu_ar_worker.CPUARWorker", required=True)
    validate_worker_omni_connector(
        "vllm_omni.platforms.cpu.worker.cpu_generation_worker.CPUGenerationWorker", required=True
    )


def test_sleep_and_wake_return_error_acks():
    class _Task:
        task_id = "t1"
        level = 1

    ack = CPUARWorker.handle_sleep_task(object.__new__(CPUARWorker), _Task())
    assert ack.status == "ERROR" and ack.task_id == "t1"
    ack = CPUARWorker.handle_wake_task(object.__new__(CPUARWorker), _Task())
    assert ack.status == "ERROR"
