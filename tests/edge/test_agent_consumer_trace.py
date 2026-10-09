# SPDX-License-Identifier: Apache-2.0
"""Pure trace contracts; synthetic metadata does not claim a runnable model.

The existing engine/load validator has its own integration tests. These tests
isolate that gate, assert it is called/fails closed, and exercise the new trace
state machine with proofs produced by the actual unchanged output parser.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from vllm_omni.edge.agent import consumer_trace as trace
from vllm_omni.edge.agent.controller import _permitted_tools
from vllm_omni.edge.agent.model_output import AgentOutputBuffer, AgentOutputContract


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def scenario(kind="final", *, observed=True, mode="strict_outer_json_fence_agent_v1", max_steps=6, fenced=False):
    contract = AgentOutputContract(mode)
    base = "strata:" + "a" * 64
    consumer = contract.consumer_identity(base)
    route = {
        "route_id": "fixture",
        "model_id": "fixture/Qwen",
        "backend": "external.strata.text.v1",
        "artifact_id": "strata-agent:" + consumer["identity_sha256"],
        "expected_placement": "cpu+cuda:0",
        "backend_identity": {
            "backend_config_sha256": "a" * 64,
            "base_engine_artifact_id": base,
            "model_output_consumer_identity": consumer,
            "observation_runtime": {"identity_sha256": "b" * 64} if observed else None,
        },
    }
    owner = {"status": "verified", "pid": 123, "creation_filetime_100ns": 321, "worker_generation": "generation"}
    plan = {"stage_id": 0, "worker_generation": "generation", "gpu_observer_identity": owner}
    url = "https://example.test/"
    if kind == "read_url":
        trusted = {"text": "Read the heading.", "read_url": url, "mode": "read_url"}
        initial = dict(trusted)
        task = f"Read this URL: {url}\nInstruction: {trusted['text']}"
        task_class = "browser_text"
    else:
        trusted = {"text": "Open " + url if kind == "tool" else "Return READY.", "read_url": None, "mode": "ordinary"}
        initial, task = {"text": trusted["text"]}, trusted["text"]
        task_class = "browser_text" if kind == "tool" else "basic"
    permitted = _permitted_tools(task_class, task)
    events = []

    def emit(kind, payload):
        events.append(
            {
                "kind": kind,
                "payload": payload,
                "request_id": "outer",
                "session_id": "session",
                "epoch": 1,
                "seq": len(events) + 1,
            }
        )

    def tools():
        emit("tool_proposed", {"operation": "browser_open", "arguments": {"url": url}})
        emit(
            "tool_result",
            {
                "operation": "browser_open",
                "data": {"url": url},
                "source": url,
                "untrusted": True,
                "completed_during_cancel": False,
            },
        )
        emit(
            "tool_proposed",
            {"operation": "browser_read", "arguments": {}, "automatic_after_navigation": "browser_open"},
        )
        emit(
            "tool_result",
            {
                "operation": "browser_read",
                "data": {"text": "untrusted page text"},
                "source": url,
                "untrusted": True,
                "completed_during_cancel": False,
            },
        )

    emit("user_observation", initial)
    emit(
        "route",
        {
            "route_id": route["route_id"],
            "model": route["model_id"],
            "backend": route["backend"],
            "artifact_id": route["artifact_id"],
            "model_output_consumer_identity": consumer,
            "model_output_contract": contract.to_dict(),
            "actual_placement": None,
        },
    )
    if kind == "read_url":
        tools()
    records, terminals = [], []
    for step in range(2 if kind == "tool" else 1):
        prompt = f"synthetic model input {step}"
        record = {
            "step": step,
            "model_request_id": f"outer-step-{step}",
            "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "utf8_bytes": len(prompt.encode()),
            "chars": len(prompt),
        }
        records.append(record)
        stage = {
            "request_id": record["model_request_id"],
            "worker_generation": "generation",
            "stage_id": 0,
            "epoch": 11 + step,
            "seq": 3,
            "kind": "text",
            "terminal": True,
            "buffers": [],
            "error": None,
            "state": None,
            "started_monotonic_ns": 0,
            "emitted_monotonic_ns": 0,
            "input_watermark": 0,
            "payload_nbytes": 0,
            "release_token": "",
        }
        command = {"tool": "browser_open", "args": {"url": url}} if kind == "tool" and step == 0 else {"final": "READY"}
        raw = json.dumps(command, ensure_ascii=False)
        if fenced:
            raw = "```json\n" + raw + "\n```"
        metrics = {
            "stage_event": stage,
            "finish_reason": "stop",
            "raw_model_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        }
        if observed:
            metrics["backend_metrics"] = {
                "runtime_telemetry": {
                    "native_io_observation": {
                        "schema": "omni-strata-request-io-observation-v1",
                        "status": "complete",
                        "reasons": [],
                        "native_terminal": "stop",
                        "request_id": stage["request_id"],
                        "epoch": stage["epoch"],
                        "generation": "generation",
                        "native_pid": 123,
                        "creation_filetime_100ns": 321,
                        "runtime_identity_sha256": "b" * 64,
                        "native_request_seq": 7 + step,
                        "scope": "native_FileExpertSource_and_PLE_counters_excludes_loading",
                        "physical_ssd_read_bytes": None,
                        "loading_covered": False,
                        "three_tier_memory_qualified": False,
                    }
                }
            }
        buffer = AgentOutputBuffer(
            contract,
            request_id=stage["request_id"],
            worker_generation="generation",
            stage_id=0,
            permitted_tools=permitted,
            previous_epoch=step,
        )
        buffer.append(raw)
        buffer.terminal(metrics)
        _, _, proof = buffer.finish()
        proof.update(
            step=step,
            model_request_id=stage["request_id"],
            input_sha256=record["sha256"],
            consumer_identity_sha256=consumer["identity_sha256"],
        )
        terminal = {"step": step, "metrics": metrics, "visibility": "withheld_until_validated_terminal"}
        terminals.append(terminal)
        emit("model_metrics", terminal)
        emit("model_output_contract", proof)
        if kind == "tool" and step == 0:
            tools()
    emit("final", {"answer": "READY", "model_step": len(records) - 1, "streamed": False})
    evidence = {
        "execution_plan": plan,
        "terminal_model_metrics": terminals,
        "model_prompt_step_count": len(records),
        "model_step_identities": records,
        "model_step_identity_policy": trace.model_step_capture_policy(contract, max_steps),
        "first_model_prompt_identity": {k: v for k, v in records[0].items() if k != "model_request_id"},
    }
    return events, route, evidence, trusted


def rebind_tool_command(sample, operation, arguments):
    """Produce a real parser proof for a mutated action, keeping every hash matched."""
    events, route, _, trusted = sample
    metrics, old_proof = events[2]["payload"]["metrics"], events[3]["payload"]
    raw = json.dumps({"tool": operation, "args": arguments}, ensure_ascii=False)
    metrics["raw_model_output_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    contract = AgentOutputContract.from_dict(route["backend_identity"]["model_output_consumer_identity"]["contract"])
    buffer = AgentOutputBuffer(
        contract,
        request_id=old_proof["model_request_id"],
        worker_generation="generation",
        stage_id=0,
        permitted_tools=_permitted_tools("browser_text", trusted["text"]),
        previous_epoch=0,
    )
    buffer.append(raw)
    buffer.terminal(metrics)
    _, _, proof = buffer.finish()
    proof.update(
        {key: old_proof[key] for key in ("step", "model_request_id", "input_sha256", "consumer_identity_sha256")}
    )
    events[3]["payload"] = proof
    events[4]["payload"].update(operation=operation, arguments=arguments)
    events[5]["payload"]["operation"] = operation
    if operation not in {"browser_open", "browser_follow"}:
        del events[6:8]
    else:
        events[6]["payload"]["automatic_after_navigation"] = operation
    for index, event in enumerate(events, 1):
        event["seq"] = index
    return proof


class ConsumerTraceTests(unittest.TestCase):
    def setUp(self):
        self.engine = patch.object(trace, "validate_strata_request_evidence").start()
        self.addCleanup(patch.stopall)

    def validate(self, value):
        events, route, evidence, trusted = value
        trace.validate_consumer_trace(events, "READY", route, evidence, trusted_task_binding=trusted)

    def test_actual_parser_final_tool_and_structured_chains(self):
        for kind in ("final", "tool", "read_url"):
            with self.subTest(kind=kind):
                self.validate(scenario(kind))
        self.assertEqual(self.engine.call_count, 3)

    def test_supported_fence_and_strict_raw_consumers_remain_distinct(self):
        self.validate(scenario(fenced=True))
        self.validate(scenario(mode="strict_raw_agent_json_v1"))
        with self.assertRaises(ValueError):
            scenario(mode="strict_raw_agent_json_v1", fenced=True)

    def test_engine_gate_is_required_and_errors_propagate(self):
        self.engine.side_effect = ValueError("engine identity refused")
        with self.assertRaisesRegex(ValueError, "engine identity refused"):
            self.validate(scenario())

    def test_legacy_and_image_detection_stay_separate(self):
        self.assertFalse(
            trace.consumer_trace_requested({"artifact_id": "model:legacy", "backend": "external.llamacpp.text.v1"})
        )
        self.assertFalse(trace.consumer_trace_requested({"artifact_id": "legacy", "backend_identity": None}))
        for marker in (
            {"artifact_id": "strata-agent"},
            {"base_engine_artifact_id": "strata:" + "a" * 64},
            {"backend_identity": {"model_output_contract": {}}},
        ):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                trace.consumer_trace_requested(marker)
        _, route, _, _ = scenario()
        route["backend"] = "external.strata.multimodal.v1"
        with self.assertRaises(ValueError):
            trace.consumer_trace_requested(route)

    def test_partial_or_conflicting_route_identity_refuses(self):
        _, original, _, _ = scenario()
        mutations = [
            lambda r: r["backend_identity"].pop("base_engine_artifact_id"),
            lambda r: r["backend_identity"].pop("model_output_consumer_identity"),
            lambda r: r["backend_identity"].update(base_engine_artifact_id="strata:" + "f" * 64),
            lambda r: r["backend_identity"].update(backend_config_sha256="f" * 64),
            lambda r: r.update(model_output_contract={}),
            lambda r: r.update(artifact_id="strata-agent:" + "f" * 64),
            lambda r: r.update(backend_identity={}),
            lambda r: r["backend_identity"]["model_output_consumer_identity"].update(adapter_source_sha256="f" * 64),
        ]
        for mutate in mutations:
            route = copy.deepcopy(original)
            mutate(route)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                trace.consumer_trace_requested(route)

    def test_complete_top_representation_must_match_nested(self):
        _, route, _, _ = scenario()
        identity = route["backend_identity"]["model_output_consumer_identity"]
        route.update(
            model_output_contract=identity["contract"],
            model_output_consumer_identity=identity,
            base_artifact_id=identity["base_artifact_id"],
            model_output_workspace_bytes=2 << 20,
        )
        self.assertTrue(trace.consumer_trace_requested(route))
        route["base_artifact_id"] = "strata:" + "f" * 64
        with self.assertRaises(ValueError):
            trace.consumer_trace_requested(route)

    def test_policy_accounts_for_both_copies_without_growing_workspace(self):
        contract = AgentOutputContract("strict_outer_json_fence_agent_v1")
        policy = trace.model_step_capture_policy(contract, 6)
        self.assertEqual(
            policy,
            {
                "schema": "omni-agent-model-step-identities-v1",
                "max_model_steps": 6,
                "max_record_bytes": 16384,
                "transient_copies": 2,
                "declared_metadata_bytes": 196608,
            },
        )
        trace.model_step_capture_policy(contract.to_dict(), 13)
        for value in (True, 1.0, 0, -1, 14, 1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                trace.model_step_capture_policy(contract, value)

    def test_capture_metadata_missing_duplicate_overflow_or_tampered_refuses(self):
        original = scenario("tool")
        mutations = [
            lambda e: e.pop("model_step_identities"),
            lambda e: e["model_step_identities"].reverse(),
            lambda e: e["model_step_identities"][1].update(step=0),
            lambda e: e["model_step_identities"][0].update(model_request_id="different"),
            lambda e: e["model_step_identities"][0].update(sha256="f" * 64),
            lambda e: e["model_step_identities"][0].update(chars=True),
            lambda e: e["model_step_identities"][0].update(extra="hidden prompt"),
            lambda e: e["model_step_identity_policy"].update(max_model_steps=1),
            lambda e: e["model_step_identity_policy"].update(declared_metadata_bytes=1),
            lambda e: e["model_step_identity_policy"].update(max_record_bytes=16384.0),
            lambda e: e["model_step_identity_policy"].update(transient_copies=2.0),
            lambda e: e["model_step_identity_policy"].update(declared_metadata_bytes=196608.0),
            lambda e: e.update(first_model_prompt_identity=e["model_step_identities"][0]),
            lambda e: e.update(model_prompt_step_count=True),
            lambda e: e["first_model_prompt_identity"].update(step=0.0),
            lambda e: e.update(model_step_identities=[None]),
        ]
        for mutate in mutations:
            value = copy.deepcopy(original)
            mutate(value[2])
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.validate(value)

    def test_final_visibility_does_not_alias_boolean_or_float_step_identities(self):
        for kind, value in (("model_metrics", False), ("model_output_contract", 0.0)):
            sample = scenario()
            event = next(event for event in sample[0] if event["kind"] == kind)
            event["payload"]["step"] = value
            with self.subTest(kind=kind):
                self.assertIsNone(trace.validated_consumer_final(sample[0][-1], sample[1], sample[0]))
                with self.assertRaises(ValueError):
                    self.validate(sample)

    def test_trusted_input_and_permission_directive_are_not_page_controlled(self):
        original = scenario("tool")
        for change in [
            lambda v: v[3].update(text="Different trusted request"),
            lambda v: v[3].update(mode="unknown"),
            lambda v: v[3].update(read_url="https://other.test/"),
            lambda v: v[0][3]["payload"].update(directive_sha256="f" * 64),
        ]:
            value = copy.deepcopy(original)
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(value)

    def test_outer_sequence_session_request_and_extra_events_refuse(self):
        for field, value in [
            ("seq", True),
            ("seq", 99),
            ("epoch", True),
            ("session_id", "other"),
            ("request_id", "other"),
        ]:
            sample = scenario()
            sample[0][2][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.validate(sample)
        for kind in ("text_delta", "error", "approval_required", "final"):
            sample = scenario()
            event = copy.deepcopy(sample[0][-1])
            event.update(kind=kind, seq=len(sample[0]) + 1)
            sample[0].append(event)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(sample)

    def test_full_terminal_projection_checks_identity_types_and_resource_release(self):
        original = scenario()
        mutations = [
            ("epoch", True),
            ("epoch", 11.0),
            ("stage_id", False),
            ("seq", 3.0),
            ("error", {"message": "failed"}),
            ("state", {"id": "retained"}),
            ("buffers", [{"id": "retained"}]),
            ("release_token", "owned"),
        ]
        for key, value in mutations:
            sample = copy.deepcopy(original)
            sample[0][2]["payload"]["metrics"]["stage_event"][key] = value
            with self.subTest(key=key, value=value):
                self.assertIsNone(trace.validated_consumer_final(sample[0][-1], sample[1], sample[0]))
                with self.assertRaises(ValueError):
                    self.validate(sample)

    def test_final_visibility_is_scoped_and_full_trace_can_still_fail(self):
        sample = scenario()
        visible = trace.validated_consumer_final(sample[0][-1], sample[1], sample[0])
        self.assertEqual(visible["visibility"], "validated_final_full_response")
        self.assertEqual(visible["text"], "READY")
        sample[2].pop("model_step_identities")
        self.assertIsNotNone(trace.validated_consumer_final(sample[0][-1], sample[1], sample[0]))
        with self.assertRaises(ValueError):
            self.validate(sample)

    def test_parser_flags_normalization_raw_and_canonical_hashes_refuse(self):
        for field, value in [
            ("normalization", "extracted"),
            ("constrained_decoding", True),
            ("extraction_used", True),
            ("retry_count", True),
            ("retry_count", 1),
            ("raw_output_sha256", "f" * 64),
            ("canonical_output_sha256", "f" * 64),
        ]:
            sample = scenario()
            sample[0][3]["payload"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(sample)

    def test_each_model_epoch_and_observed_native_dispatch_must_advance(self):
        for target, field, value in [
            ("stage", "epoch", 11),
            ("io", "native_request_seq", 7),
            ("io", "native_request_seq", 9),
            ("io", "generation", "other"),
        ]:
            sample = scenario("tool")
            metrics = [x["payload"]["metrics"] for x in sample[0] if x["kind"] == "model_metrics"][-1]
            proof = [x["payload"] for x in sample[0] if x["kind"] == "model_output_contract"][-1]
            if target == "stage":
                metrics["stage_event"][field] = proof["stage_event"][field] = value
            else:
                metrics["backend_metrics"]["runtime_telemetry"]["native_io_observation"][field] = value
            with self.subTest(target=target, field=field, value=value), self.assertRaises(ValueError):
                self.validate(sample)

    def test_observed_runtime_requires_exact_native_owner_complete_stop(self):
        for field, value in [
            ("status", "incomplete"),
            ("reasons", ["unknown"]),
            ("native_terminal", "length"),
            ("native_pid", 999),
            ("creation_filetime_100ns", True),
            ("request_id", "other"),
            ("epoch", True),
            ("runtime_identity_sha256", "f" * 64),
            ("physical_ssd_read_bytes", 0),
            ("loading_covered", True),
            ("three_tier_memory_qualified", True),
        ]:
            sample = scenario()
            sample[0][2]["payload"]["metrics"]["backend_metrics"]["runtime_telemetry"]["native_io_observation"][
                field
            ] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(sample)
        sample = scenario(observed=False)
        self.validate(sample)

    def test_combined_runtime_preserves_owned_complete_sequential_io_gates(self):
        def combined(kind="final"):
            sample = scenario(kind)
            binding = sample[1]["backend_identity"]
            binding["observation_runtime"] = None
            binding["execution_observation"] = {"static_identity_sha256": "b" * 64}
            for event in sample[0]:
                if event["kind"] == "model_metrics":
                    io = event["payload"]["metrics"]["backend_metrics"]["runtime_telemetry"]["native_io_observation"]
                    io["schema"] = "omni-strata-combined-request-io-observation-v1"
                    io["runtime_identity_schema"] = "omni-strata-combined-static-runtime-identity-v2"
            return sample

        self.validate(combined("tool"))
        for field, value in (
            ("schema", "omni-strata-request-io-observation-v1"),
            ("runtime_identity_schema", "legacy"),
            ("status", "incomplete"),
            ("reasons", ["missing_native_snapshot"]),
            ("native_terminal", "length"),
            ("native_pid", 999),
            ("creation_filetime_100ns", True),
            ("request_id", "other"),
            ("epoch", True),
            ("generation", "other"),
            ("runtime_identity_sha256", "f" * 64),
            ("physical_ssd_read_bytes", 0),
            ("loading_covered", True),
            ("three_tier_memory_qualified", True),
            ("native_request_seq", 7),
            ("native_request_seq", 9),
            ("native_request_seq", True),
        ):
            sample = combined("tool")
            metrics = [event["payload"]["metrics"] for event in sample[0] if event["kind"] == "model_metrics"][-1]
            metrics["backend_metrics"]["runtime_telemetry"]["native_io_observation"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.validate(sample)
        sample = combined()
        sample[1]["backend_identity"]["observation_runtime"] = {"identity_sha256": "b" * 64}
        with self.assertRaises(ValueError):
            self.validate(sample)

    def test_model_proof_must_precede_tool_and_unknown_automatic_action_refuses(self):
        for change in [
            lambda e: e[4]["payload"].update(automatic_after_navigation="browser_open"),
            lambda e: e[6]["payload"].update(automatic_after_navigation="browser_read"),
            lambda e: e[6]["payload"].update(operation="browser_click"),
            lambda e: e[7]["payload"].update(source="https://other.test/"),
            lambda e: e[5]["payload"].update(untrusted=False),
            lambda e: e[5]["payload"].update(completed_during_cancel=True),
            lambda e: e[3]["payload"].update(canonical_output_sha256=digest({"tool": "browser_read", "args": {}})),
        ]:
            sample = scenario("tool")
            change(sample[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(sample)
        sample = scenario("tool")
        sample[0][2], sample[0][4] = sample[0][4], sample[0][2]
        for index, event in enumerate(sample[0], 1):
            event["seq"] = index
        with self.assertRaises(ValueError):
            self.validate(sample)

    def test_recomputed_parser_proof_cannot_make_invalid_tool_arguments_complete(self):
        for arguments in ({"url": "https://example.test/", "extra": 1}, {}, {"url": "file:///private"}):
            sample = scenario("tool")
            proof = rebind_tool_command(sample, "browser_open", arguments)
            self.assertEqual(proof["canonical_output_sha256"], digest({"tool": "browser_open", "args": arguments}))
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.validate(sample)

    def test_matched_proof_requires_exact_low_impact_user_navigation(self):
        for url, register in (("https://other.test/", False), ("https://example.test/delete-account", True)):
            sample = scenario("tool")
            if register:
                sample[3]["text"] = "Open " + url
                sample[0][0]["payload"]["text"] = sample[3]["text"]
            rebind_tool_command(sample, "browser_open", {"url": url})
            sample[0][5]["payload"].update(data={"url": url}, source=url)
            sample[0][7]["payload"]["source"] = url
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "navigation is not authorized"):
                self.validate(sample)

    def test_matched_write_or_follow_proofs_require_unavailable_approval_or_dom_evidence(self):
        for operation, arguments in (
            ("browser_click", {"selector": "#submit"}),
            ("browser_fill", {"selector": "#message", "value": "text"}),
            ("browser_post", {"url": "https://example.test/", "body_b64": "e30=", "content_type": "application/json"}),
            ("browser_follow", {"selector": "a"}),
        ):
            sample = scenario("tool")
            rebind_tool_command(sample, operation, arguments)
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "approval or DOM"):
                self.validate(sample)

    def test_structured_prefix_cannot_be_presented_as_model_planning(self):
        sample = scenario("read_url")
        sample[3].update(text="Read the heading.", read_url=None, mode="ordinary")
        sample[0][0]["payload"] = {"text": "Read the heading."}
        with self.assertRaises(ValueError):
            self.validate(sample)
        sample = scenario("read_url")
        sample[0][2]["payload"]["arguments"]["url"] = "https://other.test/"
        with self.assertRaises(ValueError):
            self.validate(sample)

    def test_image_task_and_legacy_route_do_not_open_consumer_gate(self):
        sample = scenario()
        sample[3]["text"] = "Understand this image."
        sample[0][0]["payload"]["text"] = sample[3]["text"]
        with self.assertRaises(ValueError):
            self.validate(sample)
        sample = scenario()
        sample[1].clear()
        sample[1].update(backend="external.llamacpp.text.v1", artifact_id="legacy")
        with self.assertRaises(ValueError):
            self.validate(sample)


if __name__ == "__main__":
    unittest.main()
