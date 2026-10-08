# SPDX-License-Identifier: Apache-2.0
"""Explicit Agent consumer profile metadata and timing; no model execution."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import threading
import unittest
from dataclasses import asdict
from unittest.mock import patch

from benchmarks.edge_agent import native_profile, profile
from vllm_omni.edge.agent import consumer_trace as c
from vllm_omni.edge.agent import model_output as m
from vllm_omni.edge.agent import placement as p
from vllm_omni.edge.agent.controller import _permitted_tools
from vllm_omni.edge.agent.router import classify_task
from vllm_omni.engine import weight_tiers as w

ns = vars(native_profile)

def entry(explicit=False):
    source = w.ArtifactManifest("fixture/Qwen", "a" * 40, "MIT", (w.ArtifactFile("model.gguf", 100, "b" * 64),))
    runtime = w.ArtifactManifest(
        "fixture/Strata",
        p.STRATA_REVISION,
        "MIT",
        (w.ArtifactFile("engine/strata.exe", 100, "c" * 64, role="runtime"),),
    )
    packed = w.ArtifactManifest(
        "fixture/pack",
        "a" * 40 + "+pack",
        "MIT",
        (w.ArtifactFile("native_experts.txt", 100, "d" * 64, role="prepared_pack"),),
    )
    budget = w.WeightTierBudget(
        cpu_expert_cache_bytes=8 << 20,
        gpu_expert_cache_bytes=4 << 20,
        host_workspace_bytes=2 << 20,
        host_transfer_bytes=1 << 20,
    )
    tier = w.WeightTierPlan("fixture", source.manifest_sha256, p.STRATA_BACKEND, p.STRATA_REVISION, budget)
    config = dict(
        name=p.STRATA_BACKEND,
        runtime_revision=p.STRATA_REVISION,
        artifact_manifest=source.to_dict(),
        runtime_manifest=runtime.to_dict(),
        prepared_pack_manifest=packed.to_dict(),
        weight_tier_plan=tier.to_dict(),
        engine_file="engine/strata.exe",
        python_sha256="e" * 64,
        python_environment={
            "executable_sha256": "e" * 64,
            "sys_version": "fixture Python",
            "dependencies": {key: "1" for key in ("numpy", "jinja2", "regex", "PyYAML", "psutil", "Pillow", "gguf")},
        },
        conversion_manifest={
            "complete": True,
            "source_manifest_sha256": source.manifest_sha256,
            "prepared_manifest_sha256": packed.manifest_sha256,
            "tool_revision": p.STRATA_REVISION,
            "conversions": [],
        },
        expert_ram_budget_bytes=8 << 20,
        gpu_budget_bytes=16 << 20,
        gpu_total_bytes=32 << 20,
        context_tokens=4096,
        kv_type="fp16",
        expert_profile_file=None,
    )
    value = dict(
        route_id="fixture",
        backend=p.STRATA_BACKEND,
        backend_config=config,
        artifact_id="strata:" + p.evidence_sha256(config),
        model=source.checkpoint,
        placement="cpu+cuda:0",
        max_io_bytes=1 << 20,
    )
    if explicit:
        contract = m.AgentOutputContract("strict_outer_json_fence_agent_v1")
        identity = contract.consumer_identity(value["artifact_id"])
        value.update(
            base_artifact_id=value["artifact_id"],
            artifact_id="strata-agent:" + identity["identity_sha256"],
            model_output_contract=contract.to_dict(),
            model_output_consumer_identity=identity,
            model_output_workspace_bytes=contract.workspace_budget_bytes,
        )
    return value


def route_and_events():
    value = entry(True)
    binding = p.strata_route_binding(value)
    route = profile.ProfileRoute(
        value["route_id"],
        value["model"],
        value["artifact_id"],
        "a" * 40,
        binding["artifact_manifest_sha256"],
        "fixture Q4",
        p.STRATA_BACKEND,
        value["placement"],
        binding,
    )
    consumer = binding["model_output_consumer_identity"]
    stage = dict(
        request_id="outer-step-0",
        worker_generation="generation",
        stage_id=0,
        epoch=1,
        seq=2,
        kind="text",
        terminal=True,
    )
    terminal_stage = {
        **stage,
        "buffers": [],
        "error": None,
        "state": None,
        "started_monotonic_ns": 0,
        "emitted_monotonic_ns": 0,
        "input_watermark": 0,
        "payload_nbytes": 0,
        "release_token": "",
    }
    proof = dict(
        schema="omni-agent-output-interpretation-v1",
        consumer_identity_sha256=consumer["identity_sha256"],
        contract_sha256=p.evidence_sha256(consumer["contract"]),
        mode=consumer["contract"]["mode"],
        canonical_output_sha256=p.evidence_sha256({"final": "accepted"}),
        raw_output_sha256="f" * 64,
        model_request_id="outer-step-0",
        step=0,
        stage_event=stage,
        constrained_decoding=False,
        extraction_used=False,
        retry_count=0,
    )
    payloads = [
        (
            "route",
            {"artifact_id": route.artifact_id, "backend": route.backend, "model_output_consumer_identity": consumer},
        ),
        (
            "model_metrics",
            {
                "step": 0,
                "metrics": {
                    "stage_event": terminal_stage,
                    "finish_reason": "stop",
                    "raw_model_output_sha256": "f" * 64,
                },
            },
        ),
        ("model_output_contract", proof),
        ("final", {"answer": "accepted", "model_step": 0, "streamed": False}),
    ]
    events = [
        dict(request_id="outer", epoch=1, seq=index + 1, kind=kind, payload=payload)
        for index, (kind, payload) in enumerate(payloads)
    ]
    return route, events


def complete_consumer_trace():
    """Actual consumer event/identity shapes; native load proof is isolated below."""
    route, original = route_and_events()
    task = "Compute 1 + 1."
    contract = m.AgentOutputContract.from_dict(route.backend_identity["model_output_consumer_identity"]["contract"])
    prompt = "model input"
    record = dict(
        step=0,
        model_request_id="outer-step-0",
        sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        utf8_bytes=len(prompt.encode()),
        chars=len(prompt),
    )
    events = [{"kind": "user_observation", "payload": {"text": task}}, *copy.deepcopy(original)]
    events[1]["payload"].update(route_id=route.route_id, model=route.model_id, model_output_contract=contract.to_dict())
    events[2]["payload"]["visibility"] = "withheld_until_validated_terminal"
    events[3]["payload"].update(
        input_sha256=record["sha256"],
        normalization="none",
        directive_sha256=p.evidence_sha256(contract.directive(_permitted_tools(classify_task(task), task))),
    )
    for seq, event in enumerate(events, 1):
        event.update(request_id="outer", session_id="session", epoch=1, seq=seq)
    evidence = {
        "execution_plan": {"worker_generation": "generation", "stage_id": 0},
        "model_step_identities": [record],
        "model_step_identity_policy": c.model_step_capture_policy(contract, 6),
        "first_model_prompt_identity": {key: value for key, value in record.items() if key != "model_request_id"},
        "model_prompt_step_count": 1,
    }
    return route, events, evidence, {"text": task, "read_url": None, "mode": "ordinary"}


class IdentityTests(unittest.TestCase):
    def test_legacy_binding_keeps_original_schema_and_no_consumer_fields(self):
        binding = p.strata_route_binding(entry())
        self.assertEqual(binding["schema"], "omni-strata-profile-binding-v1")
        self.assertNotIn("base_engine_artifact_id", binding)
        self.assertNotIn("model_output_consumer_identity", binding)
        self.assertEqual(binding["backend_config_sha256"], p.evidence_sha256(entry()["backend_config"]))

    def test_explicit_binding_uses_actual_validator_and_existing_profile_field(self):
        value = entry(True)
        metadata = {
            "routes": {
                "fixture": {
                    "artifact_manifest_sha256": p.strata_route_binding(value)["artifact_manifest_sha256"],
                    "checkpoint_revision": "a" * 40,
                    "precision": "fixture Q4",
                }
            }
        }
        routes, provenance = ns["load_profile_routes"]({"routes": [value]}, metadata)
        self.assertEqual(
            routes[0].backend_identity["model_output_consumer_identity"], value["model_output_consumer_identity"]
        )
        self.assertEqual(provenance["fixture"]["base_engine_artifact_id"], value["base_artifact_id"])
        self.assertEqual(set(asdict(routes[0])), set(profile.ProfileRoute.__dataclass_fields__))

    def test_tampered_entries_refuse(self):
        for field in (
            "artifact_id",
            "base_artifact_id",
            "model_output_consumer_identity",
            "model_output_workspace_bytes",
        ):
            value = entry(True)
            value[field] = "tampered"
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.strata_route_binding(value)
        value = entry()
        value["base_artifact_id"] = "orphan"
        with self.assertRaises(ValueError):
            p.strata_route_binding(value)

    def test_source_or_base_tamper_fails_before_native_import(self):
        route, _ = route_and_events()
        for field, val in (("adapter_source_sha256", "0" * 64), ("base_artifact_id", "different")):
            value = asdict(route)
            value["backend_identity"]["model_output_consumer_identity"][field] = val
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.validate_strata_profile_plan({}, value)

    def test_consumer_visibility_fixture_is_not_a_complete_trace(self):
        route, events = route_and_events()
        self.assertFalse(ns["_trace_complete"](events, "accepted", route, {}))

    def test_complete_consumer_trace_requires_external_task_and_all_step_capture(self):
        route, events, evidence, trusted = complete_consumer_trace()
        with (
            patch.object(native_profile, "validate_strata_request_evidence") as native,
            patch.object(c, "validate_strata_request_evidence") as shared_native,
        ):
            self.assertTrue(
                native_profile._trace_complete(
                    events,
                    "accepted",
                    route,
                    evidence,
                    trusted_task_binding=trusted,
                )
            )
            native.assert_not_called()  # Shared consumer validation owns the one mandatory native gate.
            shared_native.assert_called_once()
            self.assertFalse(native_profile._trace_complete(events, "accepted", route, evidence))
            self.assertFalse(
                native_profile._trace_complete(
                    events,
                    "accepted",
                    route,
                    evidence,
                    trusted_task_binding={**trusted, "text": "different user instruction"},
                )
            )
            malformed = copy.deepcopy(evidence)
            malformed["model_step_identities"][0]["model_request_id"] = "other-step-0"
            self.assertFalse(
                native_profile._trace_complete(
                    events,
                    "accepted",
                    route,
                    malformed,
                    trusted_task_binding=trusted,
                )
            )

    def test_native_failure_and_image_route_cannot_fall_back_to_consumer_trace(self):
        route, events, evidence, trusted = complete_consumer_trace()
        with patch.object(c, "validate_strata_request_evidence", side_effect=ValueError("unowned")) as native:
            self.assertFalse(
                native_profile._trace_complete(
                    events,
                    "accepted",
                    route,
                    evidence,
                    trusted_task_binding=trusted,
                )
            )
            native.assert_called_once()
        image = profile.ProfileRoute(**{**asdict(route), "backend": "external.strata.multimodal.v1"})
        with patch.object(native_profile, "validate_consumer_trace") as shared:
            self.assertFalse(
                native_profile._trace_complete(
                    events,
                    "accepted",
                    image,
                    evidence,
                    trusted_task_binding=trusted,
                )
            )
            shared.assert_not_called()

    def test_partial_consumer_identity_is_not_generic_trace_or_visible_delta(self):
        route, events = route_and_events()
        route = profile.ProfileRoute(
            **{
                **asdict(route),
                "backend_identity": {
                    "model_output_consumer_identity": route.backend_identity["model_output_consumer_identity"],
                },
            }
        )
        self.assertFalse(native_profile._trace_complete(events, "accepted", route, {}))
        bridge = object.__new__(native_profile.NativeProfileBridge)
        emitted = []
        bridge._lock, bridge._events = threading.RLock(), []
        bridge._emitter = lambda kind, payload: emitted.append((kind, payload))
        bridge.route, bridge.controller = route, None
        bridge._listen(dict(kind="text_delta", payload={"text": "must stay hidden"}))
        self.assertEqual([kind for kind, _ in emitted], ["agent_event"])


class VisibilityTests(unittest.TestCase):
    def test_full_terminal_event_matches_exact_consumer_identity_projection(self):
        route, events = route_and_events()
        proof_stage = events[2]["payload"]["stage_event"]
        terminal_stage = events[1]["payload"]["metrics"]["stage_event"]
        self.assertEqual(len(proof_stage), 7)
        self.assertEqual(len(terminal_stage), 15)
        self.assertNotEqual(proof_stage, terminal_stage)
        visible = ns["_validated_consumer_final"](events[-1], route, events)
        self.assertEqual(visible["text"], "accepted")

    def test_failed_or_unreleased_full_terminal_never_becomes_visible(self):
        route, original = route_and_events()
        for field, value in (
            ("error", "failed"),
            ("state", {"handle": "retained"}),
            ("buffers", [{"handle": "retained"}]),
            ("release_token", "pending"),
        ):
            events = copy.deepcopy(original)
            events[1]["payload"]["metrics"]["stage_event"][field] = value
            with self.subTest(field=field):
                self.assertIsNone(ns["_validated_consumer_final"](events[-1], route, events))
        for field in ("error", "state", "buffers", "release_token"):
            events = copy.deepcopy(original)
            events[1]["payload"]["metrics"]["stage_event"].pop(field)
            with self.subTest(missing=field):
                self.assertIsNone(ns["_validated_consumer_final"](events[-1], route, events))

    def test_stage_projection_rejects_extra_or_invalid_identity_fields(self):
        route, original = route_and_events()
        events = copy.deepcopy(original)
        events[2]["payload"]["stage_event"]["error"] = None
        self.assertIsNone(ns["_validated_consumer_final"](events[-1], route, events))
        for field, value in (("worker_generation", ""), ("stage_id", -1),
                             ("epoch", 0), ("epoch", True), ("seq", 0)):
            events = copy.deepcopy(original)
            events[2]["payload"]["stage_event"][field] = value
            events[1]["payload"]["metrics"]["stage_event"][field] = value
            with self.subTest(field=field, value=value):
                self.assertIsNone(ns["_validated_consumer_final"](events[-1], route, events))
        events = copy.deepcopy(original)
        events[1]["payload"]["metrics"]["stage_event"]["epoch"] += 1
        self.assertIsNone(ns["_validated_consumer_final"](events[-1], route, events))

    def test_full_terminal_projection_preserves_identity_types(self):
        route, original = route_and_events()
        for field, value in (("epoch", True), ("epoch", 1.0), ("seq", 2.0),
                             ("stage_id", False), ("stage_id", 0.0), ("terminal", 1)):
            events = copy.deepcopy(original)
            events[1]["payload"]["metrics"]["stage_event"][field] = value
            with self.subTest(field=field, value=value):
                self.assertIsNone(ns["_validated_consumer_final"](events[-1], route, events))

    def test_only_bound_final_becomes_visible(self):
        route, events = route_and_events()
        visible = ns["_validated_consumer_final"](events[-1], route, events)
        self.assertEqual(visible["text"], "accepted")
        self.assertEqual(visible["visibility"], "validated_final_full_response")
        self.assertIsNone(ns["_validated_consumer_final"](events[1], route, events))

    def test_tampered_final_proof_or_terminal_never_visible(self):
        route, original = route_and_events()
        changes = (
            (0, "artifact_id", "different"),
            (2, "consumer_identity_sha256", "0" * 64),
            (2, "model_request_id", "other-step-0"),
            (2, "canonical_output_sha256", "0" * 64),
            (2, "raw_output_sha256", None),
            (3, "answer", "unproved"),
            (3, "streamed", True),
        )
        for index, field, val in changes:
            events = copy.deepcopy(original)
            events[index]["payload"][field] = val
            with self.subTest(field=field):
                self.assertIsNone(ns["_validated_consumer_final"](events[-1], route, events))
        events = copy.deepcopy(original)
        events[1]["payload"]["metrics"]["finish_reason"] = "length"
        self.assertIsNone(ns["_validated_consumer_final"](events[-1], route, events))

    def test_listener_withholds_hidden_delta_and_emits_distinct_final(self):
        route, events = route_and_events()
        bridge = object.__new__(ns["NativeProfileBridge"])
        emitted = []
        bridge._lock, bridge._events = threading.RLock(), []
        bridge._emitter = lambda kind, payload: emitted.append((kind, payload))
        bridge.route, bridge.controller = route, None
        bridge._listen(dict(request_id="outer", epoch=1, seq=0, kind="text_delta", payload={"text": "hidden"}))
        for event in events:
            bridge._listen(event)
        visible = [(kind, data) for kind, data in emitted if kind != "agent_event"]
        self.assertEqual([kind for kind, _ in visible], ["assistant_final"])
        self.assertEqual(visible[0][1]["text"], "accepted")
        bridge.route = profile.ProfileRoute("legacy", "m", "a", "r", "s", "p", "llama", "cpu")
        bridge._listen(dict(kind="text_delta", payload={"text": "legacy"}))
        self.assertEqual(emitted[-1], ("assistant_text_delta", "legacy"))


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_wrapper_closes_nested_generator_on_consumer_error(self):
        closed = []

        class Backend:
            async def generate(self, *args, **kwargs):
                try:
                    yield "first"
                    raise AssertionError("consumer should close here")
                finally:
                    closed.append(True)

        wrapper = ns["_PromptIdentityBackend"](Backend())
        chunks = wrapper.generate("actual prompt", request_id="inner", max_tokens=1)
        self.assertEqual(await anext(chunks), "first")
        await chunks.aclose()
        self.assertEqual(closed, [True])
        self.assertEqual(wrapper.identities()[0]["sha256"], hashlib.sha256(b"actual prompt").hexdigest())

    async def test_profile_timer_distinguishes_validated_final_visibility(self):
        route, _ = route_and_events()
        # Use a generic timing route to keep this test independent of native placement imports.
        route = profile.ProfileRoute(
            route.route_id,
            route.model_id,
            route.artifact_id,
            route.checkpoint_revision,
            route.artifact_sha256,
            route.precision,
            "fixture",
            "cpu",
            route.backend_identity,
        )
        case = profile.AgentCase("case", "basic", "en", "short", "task")

        async def runner(selected, task, emit):
            emit("agent_event", {"kind": "model_metrics", "ttft_s": 0.001})
            emit("assistant_text_delta", "hidden delta cannot set this consumer's visibility timer")
            await asyncio.sleep(0.01)
            emit(
                "assistant_final",
                {
                    "text": "accepted",
                    "visibility": "validated_final_full_response",
                    "consumer_identity_sha256": route.backend_identity["model_output_consumer_identity"][
                        "identity_sha256"
                    ],
                },
            )
            return profile.AgentRunResult("accepted", False, route.model_id, route.artifact_id, "cpu", "fixture")

        row = await profile._one_request(
            run_id="r",
            phase="smoke",
            repetition=0,
            route=route,
            case=case,
            runner=runner,
            evaluator=lambda *_: profile.Evaluation(True, True, 1, True),
            telemetry=None,
            telemetry_interval_s=1,
        )
        self.assertGreater(row["ttft_s"], 0.005)
        self.assertEqual(row["first_visible_event_kind"], "assistant_final")
        self.assertEqual(row["ttft_scope"], "time_to_validated_final_visibility")
        self.assertFalse(row["e2e_complete"])
