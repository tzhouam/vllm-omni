"""Bounded Agent output-contract checks without engine initialization."""

import hashlib
import importlib.util
import pathlib
import sys
import unittest

FILE = pathlib.Path(__file__).parents[2] / "vllm_omni/edge/agent/model_output.py"
spec = importlib.util.spec_from_file_location("candidate_agent_output", FILE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class ContractTests(unittest.TestCase):
    def buffer(self, mode="strict_outer_json_fence_agent_v1", **kw):
        return m.AgentOutputBuffer(
            m.AgentOutputContract(mode, **kw),
            request_id="model-step-0",
            worker_generation="generation-a",
            stage_id=0,
            permitted_tools=frozenset({"browser_read"}),
        )

    def terminal(self, **kw):
        event = dict(
            request_id="model-step-0",
            worker_generation="generation-a",
            stage_id=0,
            epoch=1,
            seq=3,
            kind="text",
            terminal=True,
        )
        metrics = {"finish_reason": "stop", "stage_event": event}
        for key, value in kw.items():
            if key == "finish_reason":
                metrics[key] = value
            else:
                event[key] = value
        return metrics

    def parse(self, text, mode="strict_outer_json_fence_agent_v1"):
        b = self.buffer(mode)
        for x in text:
            b.append(x)
        b.terminal(self.terminal())
        return b.finish()

    def test_raw_and_single_fence_preserve_distinct_hashes(self):
        raw = '{"tool":"browser_read","args":{}}'
        a = self.parse(raw)
        b = self.parse("```json\n" + raw + "\n```")
        self.assertEqual(a[:2], b[:2])
        self.assertNotEqual(a[2]["raw_output_sha256"], b[2]["raw_output_sha256"])
        self.assertEqual(a[2]["canonical_output_sha256"], b[2]["canonical_output_sha256"])
        self.assertEqual(b[2]["normalization"], "single_outer_json_fence")
        self.assertFalse(b[2]["extraction_used"])

    def test_strict_raw_refuses_fence(self):
        with self.assertRaises(ValueError):
            self.parse('```json\n{"final":"x"}\n```', "strict_raw_agent_json_v1")

    def test_no_prose_search_or_multiple_object_acceptance(self):
        for text in (
            'prefix {"final":"x"}',
            '{"final":"x"} suffix',
            '```json\n{"final":"x"}\n``` prose',
            '```\n{"final":"x"}\n```',
            '```JSON\n{"final":"x"}\n```',
            '```json\n{"final":"x"}\n```\n```json\n{}\n```',
            '{"final":"x"} {"final":"x"}',
            '[ {"final":"x"} ]',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

    def test_duplicates_nonfinite_unknown_tool_or_envelope_refused(self):
        for text in (
            '{"final":"x","final":"y"}',
            '{"tool":"browser_read","args":{"x":1,"x":2}}',
            '{"tool":"browser_read","args":{"x":NaN}}',
            '{"tool":"browser_read","args":{"x":1e999}}',
            '{"tool":"settings_set","args":{}}',
            '{"tool":"browser_read","arguments":{}}',
            '{"final":"x","extra":1}',
            '{"final":1}',
            '{"tool":"browser_read","args":[]}',
            '{"final":"\\ud800"}',
        ):
            with self.subTest(text=text), self.assertRaises((ValueError, UnicodeError)):
                self.parse(text)

    def test_no_completion_without_owned_stop_and_one_terminal(self):
        for change in (
            {"finish_reason": "length"},
            {"request_id": "other"},
            {"worker_generation": "old"},
            {"stage_id": 1},
            {"stage_id": False},
            {"epoch": 0},
            {"epoch": True},
            {"seq": 0},
            {"kind": "error"},
            {"terminal": False},
        ):
            b = self.buffer()
            b.append('{"final":"x"}')
            with self.subTest(change=change), self.assertRaises(ValueError):
                b.terminal(self.terminal(**change))
        b = self.buffer()
        b.append('{"final":"x"}')
        with self.assertRaises(ValueError):
            b.finish()
        b.terminal(self.terminal())
        with self.assertRaises(ValueError):
            b.terminal(self.terminal())
        with self.assertRaises(ValueError):
            b.append(" ")
        self.assertEqual(b.finish()[0], "final")
        with self.assertRaises(ValueError):
            b.finish()

    def test_admission_bounds_and_identity(self):
        c = m.AgentOutputContract("strict_raw_agent_json_v1")
        c.admit(max_io_bytes=65536, workspace_reserved_bytes=c.workspace_budget_bytes)
        for bounds in (
            {"max_io_bytes": 65535, "workspace_reserved_bytes": c.workspace_budget_bytes},
            {"max_io_bytes": 65536, "workspace_reserved_bytes": c.workspace_budget_bytes - 1},
        ):
            with self.assertRaises(ValueError):
                c.admit(**bounds)
        for kw in ({"workspace_budget_bytes": 1}, {"max_depth": True}, {"mode": "extract_first_object"}):
            value = c.to_dict()
            value.update(kw)
            with self.assertRaises(ValueError):
                m.AgentOutputContract.from_dict(value)
        identity = c.consumer_identity("base-pinned-artifact")
        self.assertEqual(identity["adapter_source_sha256"], hashlib.sha256(FILE.read_bytes()).hexdigest())
        d = m.AgentOutputContract("strict_outer_json_fence_agent_v1")
        self.assertNotEqual(c.consumer_identity("a")["identity_sha256"], d.consumer_identity("a")["identity_sha256"])

    def test_response_depth_node_and_utf8_limits(self):
        b = self.buffer(max_response_bytes=8)
        with self.assertRaises(ValueError):
            b.append("x" * 9)
        c = self.buffer(max_depth=2)
        c.append('{"tool":"browser_read","args":{"x":{}}}')
        c.terminal(self.terminal())
        with self.assertRaises(ValueError):
            c.finish()
        c = self.buffer(max_nodes=2)
        c.append('{"final":"x"}')
        c.terminal(self.terminal())
        self.assertEqual(c.finish()[1], "x")
        c = self.buffer(max_nodes=1)
        c.append('{"final":"x"}')
        c.terminal(self.terminal())
        with self.assertRaises(ValueError):
            c.finish()


class CollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_contract_releases_iterator_without_cancel(self):
        from types import SimpleNamespace

        fixture = ContractTests()
        b = fixture.buffer()
        closed, cancelled = [], []

        async def chunks():
            try:
                for part in ("```json\n", '{"final":"accepted"}', "\n```"):
                    yield SimpleNamespace(text=part, terminal=False, metrics={})
                yield SimpleNamespace(text="", terminal=True, metrics=fixture.terminal())
            finally:
                closed.append(True)

        async def cancel(request_id):
            cancelled.append(request_id)

        result = await m.collect_agent_command(chunks(), buffer=b, cancel=cancel)
        self.assertEqual(result[:2], ("final", "accepted"))
        self.assertEqual(closed, [True])
        self.assertEqual(cancelled, [])

    async def test_consumer_limit_failure_cancels_and_closes(self):
        from types import SimpleNamespace

        fixture = ContractTests()
        b = fixture.buffer(max_response_bytes=8)
        closed, cancelled = [], []

        async def chunks():
            try:
                yield SimpleNamespace(text="xxxxxxxxx", terminal=False, metrics={})
                self.fail("consumer must stop before another chunk")
            finally:
                closed.append(True)

        async def cancel(request_id):
            self.assertEqual(bytes(b._raw), b"")
            self.assertEqual(b.directive, {})
            cancelled.append(request_id)

        with self.assertRaises(ValueError):
            await m.collect_agent_command(chunks(), buffer=b, cancel=cancel)
        self.assertEqual(closed, [True])
        self.assertEqual(cancelled, ["model-step-0"])

    async def test_task_cancellation_never_returns_command_and_closes(self):
        import asyncio
        from types import SimpleNamespace

        fixture = ContractTests()
        cancelled, closed = [], []
        started = asyncio.Event()

        async def chunks():
            try:
                yield SimpleNamespace(text='{"tool":"browser_read","args":', terminal=False, metrics={})
                started.set()
                await asyncio.Event().wait()
            finally:
                closed.append(True)

        async def cancel(request_id):
            cancelled.append(request_id)

        task = asyncio.create_task(m.collect_agent_command(chunks(), buffer=fixture.buffer(), cancel=cancel))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(cancelled, ["model-step-0"])
        self.assertEqual(closed, [True])

    async def test_incomplete_or_length_never_returns_command(self):
        from types import SimpleNamespace

        fixture = ContractTests()
        for stop in (None, "length"):
            cancelled, closed = [], []

            async def chunks():
                try:
                    yield SimpleNamespace(text='{"tool":"browser_read","args":{}}', terminal=False, metrics={})
                    if stop is not None:
                        yield SimpleNamespace(text="", terminal=True, metrics=fixture.terminal(finish_reason=stop))
                finally:
                    closed.append(True)

            async def cancel(request_id):
                cancelled.append(request_id)

            with self.subTest(stop=stop), self.assertRaises(ValueError):
                await m.collect_agent_command(chunks(), buffer=fixture.buffer(), cancel=cancel)
            self.assertEqual(closed, [True])
            self.assertEqual(cancelled, ["model-step-0"])


if __name__ == "__main__":
    unittest.main()
