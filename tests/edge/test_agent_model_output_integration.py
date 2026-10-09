"""Agent output contract integration: real controllers and lease code, synthetic model pool."""

from __future__ import annotations

import copy
import types
import unittest
from dataclasses import replace

from vllm_omni.edge.agent import controller as c
from vllm_omni.edge.agent import model_output as m
from vllm_omni.edge.agent import omni_backend as b
from vllm_omni.edge.agent import router as r
from vllm_omni.edge.agent import strata_route as sr
from vllm_omni.edge.agent import tools as t
from vllm_omni.engine import local_plan as lp
from vllm_omni.engine import weight_tiers as w


class Memory:
    def __init__(self):
        self.events = []

    def append_event(self, **kwargs):
        self.events.append(kwargs)
        return types.SimpleNamespace(event_id=str(len(self.events)))

    def search(self, *args, **kwargs):
        return []


class Browser:
    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1
        return {"url": "https://example.com", "text": "observed"}

    def current_url(self):
        return "https://example.com"

    def close(self):
        pass


class Settings:
    def __init__(self):
        self.writes = []

    def read(self, setting):
        return {"setting": setting, "value": 8}

    def set(self, *args):
        self.writes.append(args)
        return {}


class Backend:
    def __init__(self, replies, contract, *, replay=False, finish_reason="stop"):
        self.replies = list(replies)
        self.cancelled = []
        self.closed = 0
        self.prompts = []
        self.finished = []
        self.epoch = 0
        self.replay = replay
        self.finish_reason = finish_reason
        self.execution_plan = {
            "requested_device": "cpu",
            "observed_model_placement": "cpu",
            "stage_id": 0,
            "worker_generation": "fixture-generation",
        }
        self.model_output_contract_identity = contract.consumer_identity("fixture-base") if contract else None

    def start(self):
        pass

    async def generate(self, prompt, *, request_id, max_tokens, image_data_url=None):
        self.prompts.append(prompt)
        self.epoch += 1
        try:
            for text in self.replies.pop(0):
                yield b.BackendChunk(text)
            event = {
                "request_id": request_id,
                "worker_generation": "fixture-generation",
                "stage_id": 0,
                "epoch": 1 if self.replay else self.epoch,
                "seq": 10,
                "kind": "text",
                "terminal": True,
            }
            yield b.BackendChunk("", terminal=True, metrics={"finish_reason": self.finish_reason, "stage_event": event})
        finally:
            self.finished.append(request_id)

    async def cancel(self, request_id):
        self.cancelled.append(request_id)

    def close(self):
        self.closed += 1
        return True


def fixture(replies, *, contract=None, **backend_kw):
    contract = contract or m.AgentOutputContract("strict_outer_json_fence_agent_v1")
    identity = contract.consumer_identity("fixture-base")
    route = r.Route(
        "explicit-json",
        "llamacpp-agent:" + identity["identity_sha256"],
        "fixture",
        "external.llamacpp.text.v1",
        frozenset({"text"}),
        "cpu",
        {"host_ram": contract.workspace_budget_bytes + 65536},
        model_output_contract=contract,
        base_artifact_id="fixture-base",
    )
    backend = Backend(replies, contract, **backend_kw)
    browser, settings = Browser(), Settings()
    controller = c.AgentController(
        routes=[route],
        qualifications=[],
        backends={route.route_id: backend},
        memory=Memory(),
        tools=t.WindowsToolBoundary(browser=browser, settings=settings),
        admit=lambda _: r.Admission(True, "fixture"),
        environment_fingerprint="fixture",
        power_condition="AC",
        qualification_suite_id="fixture",
        bootstrap_route_id=route.route_id,
    )
    events = []
    controller.add_listener(events.append)
    return controller, backend, browser, settings, events, route


class AgentIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_fenced_read_is_hidden_then_runs_guarded_tool_and_exact_final(self):
        controller, backend, browser, _, events, _ = fixture(
            [["```json\n", '{"tool":"browser_read","args":{}}', "\n```"], ['{"final":"done"}']]
        )
        try:
            answer = await controller.run_turn("Read the page at https://example.com", request_id="turn", epoch=1)
            self.assertEqual(answer, "done")
            self.assertEqual(browser.reads, 1)
            self.assertFalse(any(x["kind"] == "text_delta" for x in events))
            kinds = [x["kind"] for x in events]
            self.assertLess(kinds.index("model_output_contract"), kinds.index("tool_proposed"))
            self.assertEqual(kinds[-1], "final")
            self.assertEqual(len(backend.finished), 2)
            for prompt in backend.prompts:
                self.assertIn("output_contract", __import__("json").loads(prompt))
                self.assertNotIn("For a final answer, write plain text.", prompt)
        finally:
            controller._tool_executor.shutdown(wait=True)

    async def test_invalid_formats_and_length_never_become_tool_or_final(self):
        for reply, finish in (
            (['```json\n{"tool":"browser_read","args":{}}\n``` prose'], "stop"),
            (['{"final":"a","final":"b"}'], "stop"),
            (['{"tool":"browser_read","args":{}}'], "length"),
        ):
            controller, backend, browser, _, events, _ = fixture([reply], finish_reason=finish)
            try:
                with self.subTest(finish=finish), self.assertRaises(ValueError):
                    await controller.run_turn("Read the page at https://example.com", request_id="turn", epoch=1)
                self.assertEqual(browser.reads, 0)
                self.assertFalse(any(x["kind"] in {"text_delta", "tool_proposed", "final"} for x in events))
                self.assertEqual(backend.finished, ["turn-step-0"])
            finally:
                controller._tool_executor.shutdown(wait=True)

    async def test_model_cannot_grant_tool_or_bypass_existing_write_approval(self):
        controller, _, browser, _, events, _ = fixture([['{"tool":"screen_capture","args":{}}']])
        try:
            with self.assertRaises(ValueError):
                await controller.run_turn("Read the page at https://example.com", request_id="turn", epoch=1)
            self.assertEqual(browser.reads, 0)
            self.assertFalse(any(x["kind"] == "tool_proposed" for x in events))
        finally:
            controller._tool_executor.shutdown(wait=True)
        controller, _, _, settings, events, _ = fixture(
            [['```json\n{"tool":"settings_set","args":{"setting":"mouse_speed","value":10}}\n```']]
        )

        def deny(event):
            if event["kind"] == "approval_required":
                controller._approval[1].set_result(False)

        controller.add_listener(deny)
        try:
            with self.assertRaises(PermissionError):
                await controller.run_turn("Set mouse speed to 10", request_id="write", epoch=1)
            self.assertEqual(settings.writes, [])
            self.assertTrue(any(x["kind"] == "approval_required" for x in events))
        finally:
            controller._tool_executor.shutdown(wait=True)

    async def test_replayed_epoch_is_rejected_before_second_action(self):
        controller, _, browser, _, events, _ = fixture(
            [['{"tool":"browser_read","args":{}}'], ['{"tool":"browser_read","args":{}}']], replay=True
        )
        try:
            with self.assertRaises(ValueError):
                await controller.run_turn("Read the page at https://example.com", request_id="turn", epoch=1)
            self.assertEqual(browser.reads, 1)
            self.assertEqual(sum(x["kind"] == "tool_proposed" for x in events), 1)
        finally:
            controller._tool_executor.shutdown(wait=True)

    async def test_legacy_text_route_preserves_original_prompt_and_visibility(self):
        controller, backend, _, _, events, route = fixture([["plain ", "answer"]])
        legacy = replace(route, model_output_contract=None, base_artifact_id=None, artifact_id="legacy")
        controller.routes = [legacy]
        backend.model_output_contract_identity = None
        try:
            answer = await controller.run_turn("Say hello", request_id="legacy", epoch=1)
            self.assertEqual(answer, "plain answer")
            self.assertEqual([x["payload"]["text"] for x in events if x["kind"] == "text_delta"], ["plain ", "answer"])
            self.assertFalse(any(x["kind"] == "model_output_contract" for x in events))
            self.assertNotIn("output_contract", __import__("json").loads(backend.prompts[0]))
        finally:
            controller._tool_executor.shutdown(wait=True)

    async def test_managed_wrapper_closes_underlying_generator_on_consumer_error(self):
        contract = m.AgentOutputContract("strict_raw_agent_json_v1", max_response_bytes=8)
        backend = Backend([["x" * 9]], contract)
        manager = types.SimpleNamespace(_backends={"r": backend}, _companion=None)
        wrapped = lp.ManagedLocalBackend(manager, "r")
        buffer = m.AgentOutputBuffer(
            contract, request_id="id", worker_generation="fixture-generation", stage_id=0, permitted_tools=frozenset()
        )
        with self.assertRaises(ValueError):
            await m.collect_agent_command(
                wrapped.generate("prompt", request_id="id", max_tokens=8), buffer=buffer, cancel=wrapped.cancel
            )
        self.assertEqual(backend.cancelled, ["id"])
        self.assertEqual(backend.finished, ["id"])


class ActualAdapterProtocol(unittest.IsolatedAsyncioTestCase):
    def adapter(self, text, finish_reason="stop"):
        config = types.SimpleNamespace(
            max_new_tokens=128,
            max_io_bytes=65536,
            mmproj_file=None,
            request_timeout_s=2,
            backend_config={"weight_tier_plan": {"budget": {"host_workspace_bytes": 2 << 20}}},
        )
        adapter = b.OmniCompleteModelBackend(config)
        contract = m.AgentOutputContract("strict_outer_json_fence_agent_v1")
        adapter.bind_output_contract(contract, base_artifact_id="strata:" + m._hash(config.backend_config))
        event = {
            "request_id": "id",
            "worker_generation": "generation",
            "stage_id": 0,
            "seq": 1,
            "epoch": 1,
            "kind": "text",
            "terminal": True,
        }
        counts = {"ack": 0, "abort": 0}
        output = types.SimpleNamespace(
            request_id="id",
            error=None,
            outputs=[types.SimpleNamespace(text=text, finish_reason=finish_reason)],
            metrics={},
            custom_output={"stage_event": event},
            release_stage_buffers=lambda: counts.update(ack=counts["ack"] + 1),
        )

        class Pool:
            def __init__(self):
                self.stage_client = self
                self.items = iter([(text, __import__("time").perf_counter()), None])

            async def submit_initial(self, *_):
                pass

            async def receive_agent_delta(self, *_):
                return next(self.items)

            def poll_graph_output(self, *_):
                return output

            async def abort_requests(self, *_):
                counts["abort"] += 1

        adapter._pool = Pool()
        return adapter, contract, output, counts

    async def test_actual_adapter_retains_exact_raw_without_copy_and_acks(self):
        raw = '```json\n{"final":"done"}\n```'
        adapter, contract, output, counts = self.adapter(raw)
        buffer = m.AgentOutputBuffer(
            contract, request_id="id", worker_generation="generation", stage_id=0, permitted_tools=frozenset()
        )
        command, value, proof, metrics = await m.collect_agent_command(
            adapter.generate("prompt", request_id="id", max_tokens=128), buffer=buffer, cancel=adapter.cancel
        )
        self.assertEqual((command, value), ("final", "done"))
        self.assertEqual(counts, {"ack": 1, "abort": 0})
        record = adapter.last_model_output()
        self.assertIs(record["text"], output.outputs[0].text)
        self.assertEqual(record["raw_output_sha256"], proof["raw_output_sha256"])
        self.assertEqual(metrics["raw_model_output_sha256"], proof["raw_output_sha256"])
        record["stage_event"]["epoch"] = 999
        self.assertEqual(adapter.last_model_output()["stage_event"]["epoch"], 1)
        self.assertTrue(adapter.close())
        self.assertIsNone(adapter.last_model_output())

    async def test_actual_length_is_never_interpreted_and_cleanup_clears_retained_raw(self):
        adapter, contract, _, counts = self.adapter('{"final":"done"}', "length")
        buffer = m.AgentOutputBuffer(
            contract, request_id="id", worker_generation="generation", stage_id=0, permitted_tools=frozenset()
        )
        with self.assertRaises(ValueError):
            await m.collect_agent_command(
                adapter.generate("prompt", request_id="id", max_tokens=128), buffer=buffer, cancel=adapter.cancel
            )
        # Early close retires the actual adapter after failed terminal validation.
        self.assertEqual(counts["ack"], 1)
        self.assertGreaterEqual(counts["abort"], 1)
        self.assertIsNone(adapter._active)
        self.assertIsNone(adapter.last_model_output())

    async def test_malformed_natural_stop_preserves_raw_until_explicit_close(self):
        raw = '{"tool":"browser_read","args":{},"extra":true}'
        adapter, contract, _, counts = self.adapter(raw)
        buffer = m.AgentOutputBuffer(
            contract,
            request_id="id",
            worker_generation="generation",
            stage_id=0,
            permitted_tools=frozenset({"browser_read"}),
        )
        with self.assertRaises(ValueError):
            await m.collect_agent_command(
                adapter.generate("prompt", request_id="id", max_tokens=128), buffer=buffer, cancel=adapter.cancel
            )
        self.assertEqual(counts, {"ack": 1, "abort": 0})
        self.assertEqual(adapter.last_model_output()["text"], raw)
        adapter.close()
        self.assertIsNone(adapter.last_model_output())

    async def test_early_consumer_failure_aborts_actual_adapter_without_output_ack(self):
        adapter, _, _, counts = self.adapter("x" * 9)
        contract = m.AgentOutputContract("strict_raw_agent_json_v1", max_response_bytes=8)
        buffer = m.AgentOutputBuffer(
            contract, request_id="id", worker_generation="generation", stage_id=0, permitted_tools=frozenset()
        )
        with self.assertRaises(ValueError):
            await m.collect_agent_command(
                adapter.generate("prompt", request_id="id", max_tokens=128), buffer=buffer, cancel=adapter.cancel
            )
        self.assertEqual(counts["ack"], 0)
        self.assertGreaterEqual(counts["abort"], 1)
        self.assertIsNone(adapter._active)
        self.assertIsNone(adapter._pool)

    async def test_consumer_failure_keeps_unverified_shared_lease_quarantined(self):
        adapter, _, _, counts = self.adapter("x" * 9)
        ledger = lp.ResourceLedger({"host_ram": 4 << 20})
        reservation = ledger.reserve("owned-route", {"host_ram": 2 << 20})
        adapter.config.demands = dict(reservation.demands)
        adapter.bind_resource_lease(ledger, reservation)

        class Runtime:
            resource_ledger = ledger
            _resource_reservations = {(0, 0): reservation}
            drained = False

            def shutdown(self):
                ledger.release(reservation, drained=self.drained)

        runtime = Runtime()
        adapter._runtime = runtime
        contract = m.AgentOutputContract("strict_raw_agent_json_v1", max_response_bytes=8)
        buffer = m.AgentOutputBuffer(
            contract, request_id="id", worker_generation="generation", stage_id=0, permitted_tools=frozenset()
        )
        with self.assertRaises(ValueError):
            await m.collect_agent_command(
                adapter.generate("prompt", request_id="id", max_tokens=128), buffer=buffer, cancel=adapter.cancel
            )
        self.assertTrue(ledger.owns(reservation))
        self.assertEqual(ledger.snapshot()["quarantined"], ["owned-route"])
        self.assertFalse(adapter.close())
        self.assertIsNone(adapter.last_model_output())
        runtime.drained = True
        self.assertTrue(adapter.close())
        self.assertFalse(ledger.owns(reservation))
        self.assertTrue(ledger.was_released(reservation))
        self.assertEqual(counts["ack"], 0)


class IdentityAndAdmission(unittest.TestCase):
    def launch(self):
        source = w.ArtifactManifest(
            "fixture/Qwen",
            "a" * 40,
            "MIT",
            (w.ArtifactFile("weights.gguf", 300, "b" * 64), w.ArtifactFile("ple.gguf", 300, "c" * 64)),
        )
        runtime = w.ArtifactManifest(
            "fixture/Strata",
            sr.RUNTIME_REVISION,
            "MIT",
            (w.ArtifactFile("engine/strata.exe", 100, "d" * 64, role="runtime"),),
        )
        pack = w.ArtifactManifest(
            "fixture/pack",
            source.revision + "+pack",
            "MIT",
            (w.ArtifactFile("dense.bin", 200, "e" * 64, role="prepared_pack"),),
        )
        budget = w.WeightTierBudget(
            cpu_expert_cache_bytes=1024,
            host_workspace_bytes=128,
            host_transfer_bytes=65536,
            windows_commit_peak_bytes=128000,
            gpu_weights_bytes=256,
            gpu_expert_cache_bytes=128,
            gpu_workspace_bytes=64,
            ssd_artifact_bytes=800,
            ssd_temporary_bytes=64,
        )
        plan = w.WeightTierPlan(
            "parent",
            source.manifest_sha256,
            sr.BACKEND,
            sr.RUNTIME_REVISION,
            budget,
            ssd_experts=True,
            lookup_tables_on_demand=True,
        )
        backend = {
            "name": sr.BACKEND,
            "runtime_revision": sr.RUNTIME_REVISION,
            "route_id": "parent",
            "gpu_index": 0,
            "gpu_pool": "vram:0",
            "gpu_total_bytes": 2048,
            "gpu_budget_bytes": 448,
            "artifact_manifest": source.to_dict(),
            "runtime_manifest": runtime.to_dict(),
            "prepared_pack_manifest": pack.to_dict(),
            "weight_tier_plan": plan.to_dict(),
            "conversion_manifest": {
                "complete": True,
                "source_manifest_sha256": source.manifest_sha256,
                "prepared_manifest_sha256": pack.manifest_sha256,
                "tool_revision": sr.RUNTIME_REVISION,
                "conversions": [],
            },
            "runtime_root": "C:/fixture/runtime",
            "artifact_root": "C:/fixture/source",
            "prepared_model_dir": "C:/fixture/pack",
            "python_bin": "C:/fixture/python.exe",
            "python_sha256": "f" * 64,
            "python_environment": {"fixture": True},
            "native_file": "weights.gguf",
            "ple_file": "ple.gguf",
            "expert_ram_budget_bytes": 1024,
            "host_overhead_bytes": 128,
            "context_tokens": 4096,
            "max_new_tokens": 128,
            "max_io_bytes": 65536,
            "request_timeout_s": 600,
            "start_timeout_s": 900,
            "spec_tokens": 0,
            "ple_prefetch": False,
            "routing_prefetch": False,
            "io_prefetch": False,
        }
        return {
            "schema": "omni-strata-launch-v1",
            "backend": backend,
            "resource_budget": {
                "demands": budget.resource_demands(include_windows_commit=True),
                "capacities": {"host_ram": 8 << 20, "vram:0": 2048, "windows_commit": 8 << 20, "ssd": 8192},
            },
        }

    def test_derived_one_lease_charges_workspace_once_and_preserves_neural_controls(self):
        original = self.launch()
        before = copy.deepcopy(original)
        contract = m.AgentOutputContract("strict_outer_json_fence_agent_v1")
        config = sr.agent_config_with_output_contract_from_launch(
            original,
            expected_device_name="fixture NVIDIA",
            output_contract=contract.to_dict(),
            route_id="new-consumer",
            experimental_bootstrap=True,
        )
        entry = config["routes"][0]
        backend = entry["backend_config"]
        delta = contract.workspace_budget_bytes
        self.assertEqual(original, before)
        self.assertEqual(entry["memory_demands"]["host_ram"], before["resource_budget"]["demands"]["host_ram"] + delta)
        self.assertEqual(
            entry["memory_demands"]["windows_commit"], before["resource_budget"]["demands"]["windows_commit"] + delta
        )
        self.assertEqual(entry["memory_demands"]["vram"], 448)
        self.assertEqual(backend["host_overhead_bytes"], 128 + delta)
        self.assertEqual(backend["weight_tier_plan"]["budget"]["host_transfer_bytes"], 65536)
        self.assertEqual(
            entry["memory_demands"]["host_ram"],
            backend["expert_ram_budget_bytes"] + backend["host_overhead_bytes"] + backend["max_io_bytes"],
        )
        for key in (
            "artifact_manifest",
            "runtime_manifest",
            "prepared_pack_manifest",
            "conversion_manifest",
            "spec_tokens",
            "gpu_budget_bytes",
            "context_tokens",
            "max_new_tokens",
            "max_io_bytes",
        ):
            self.assertEqual(backend[key], before["backend"][key])
        self.assertEqual(m.validate_output_contract_entry(entry), contract)
        adapter = b.OmniStrataBackend(
            b.OmniStrataConfig(
                route_id=entry["route_id"],
                backend_config=backend,
                placement=entry["placement"],
                capacities={"host_ram": 8 << 20, "vram": 2048, "windows_commit": 8 << 20, "ssd": 8192},
                demands=entry["memory_demands"],
                max_io_bytes=65536,
            )
        )
        adapter.bind_output_contract(contract, base_artifact_id=entry["base_artifact_id"])
        with self.assertRaises(ValueError):
            adapter.bind_output_contract(contract, base_artifact_id="wrong-engine")
        self.assertEqual(adapter.model_output_contract_identity, entry["model_output_consumer_identity"])
        for key in (
            "artifact_id",
            "base_artifact_id",
            "model_output_consumer_identity",
            "model_output_workspace_bytes",
        ):
            bad = copy.deepcopy(entry)
            bad[key] = "tampered"
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_output_contract_entry(bad)
        tight = self.launch()
        tight["resource_budget"]["capacities"]["host_ram"] = tight["resource_budget"]["demands"]["host_ram"]
        with self.assertRaises(ValueError):
            sr.agent_config_with_output_contract_from_launch(
                tight, expected_device_name="fixture NVIDIA", output_contract=contract.to_dict(), route_id="new"
            )

    def test_old_qualification_cannot_select_new_parser_even_matching_route_artifact(self):
        controller, _, _, _, _, route = fixture([])
        try:
            samples = {name: (1.0,) * 20 for name in ("short", "medium", "long")}
            profile = r.Qualification(
                route.route_id,
                route.artifact_id,
                "basic",
                "fixture",
                "fixture",
                "AC",
                20,
                20,
                samples,
                samples,
                {name: 1 for name in samples},
                1800,
                True,
                True,
                True,
                True,
                True,
                True,
                raw_evidence="fixture",
            )
            decision = r.select_route(
                "basic",
                [route],
                [profile],
                suite_id="fixture",
                environment_fingerprint="fixture",
                power_condition="AC",
                admit=lambda _: r.Admission(True, "fixture"),
            )
            self.assertIsNone(decision.route)
            current = replace(profile, output_contract_sha256=route.model_output_contract.identity_sha256)
            self.assertIs(
                r.select_route(
                    "basic",
                    [route],
                    [current],
                    suite_id="fixture",
                    environment_fingerprint="fixture",
                    power_condition="AC",
                    admit=lambda _: r.Admission(True, "fixture"),
                ).route,
                route,
            )
        finally:
            controller._tool_executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
