# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""M0: an auditable local text mode on the Omni stage runtime.

One model, one autoregressive stage, on the device this process can actually
drive -- with the execution plan, the memory budget, the state handles and the
cancellation semantics made explicit instead of implicit.

    from vllm_omni.edge.local import plan_text_session, LocalTextEngine

    plan = plan_text_session("models/Spark-X2.5-4B", max_model_len=4096)
    print(plan.summary())          # says yes with a budget, or no with a reason
    async with LocalTextEngine(plan) as engine:
        session = engine.open_session()
        rid, stream = await engine.submit(session, "Hello", max_tokens=128)
        async for event in stream:
            ...

The command line is ``python -m vllm_omni.edge.local`` (``devices``, ``plan``,
``run``, ``accept``).
"""

from vllm_omni.edge.local.capabilities import (
    DeviceCapability,
    describe as describe_devices,
    enumerate_devices,
)
from vllm_omni.edge.local.engine import LocalTextEngine, PlanNotAdmitted, RequestRecord
from vllm_omni.edge.local.manifest import ArtifactManifest, build_manifest, runtime_versions
from vllm_omni.edge.local.plan import ExecutionPlan, MemoryReservation, Refusal, plan_text_session
from vllm_omni.edge.local.prompts import acceptance_prompts
from vllm_omni.edge.local.session import BoundedEventStream, ChunkEvent, StateHandle

__all__ = [
    "ArtifactManifest",
    "BoundedEventStream",
    "ChunkEvent",
    "DeviceCapability",
    "ExecutionPlan",
    "LocalTextEngine",
    "MemoryReservation",
    "PlanNotAdmitted",
    "Refusal",
    "RequestRecord",
    "StateHandle",
    "acceptance_prompts",
    "build_manifest",
    "describe_devices",
    "enumerate_devices",
    "plan_text_session",
    "runtime_versions",
]
