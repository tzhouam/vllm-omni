# SPDX-License-Identifier: Apache-2.0
"""Exact summary parity and completed-request release for batch-one profiles."""

from __future__ import annotations

import asyncio
import copy
import gc
import hashlib
import io
import json
import math
import tempfile
import unittest
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from benchmarks.edge_agent import profile as profile_module
from benchmarks.edge_agent.profile import (
    LENGTH_BUCKETS,
    AgentCase,
    AgentRunResult,
    Evaluation,
    Preparation,
    ProfileConditions,
    ProfileConfig,
    ProfileRoute,
    RouteProfile,
    _route_profile,
    _RouteProfileAccumulator,
    _sha256_file,
    run_profile,
)


def _legacy_nearest_rank(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


# Frozen pre-streaming summary from profile.py SHA
# f86f3e3f547ba7ad175af07492972309e074a432837eac21a178892369a2a363.
# This independent oracle preserves all prior gates, including their scopes.
def _legacy_route_profile(
    route: ProfileRoute,
    samples: Sequence[Mapping[str, Any]],
    *,
    cold_start_s: float | None,
    sustained_seconds: float,
    endurance_wall_seconds: float,
    config: ProfileConfig,
) -> RouteProfile:
    measured = [row for row in samples if row["phase"] == "measured"]
    warmups = [row for row in samples if row["phase"] == "warmup"]
    endurance = [row for row in samples if row["phase"] == "endurance"]
    by_length = {bucket: [row for row in measured if row["case"]["length"] == bucket]
                 for bucket in LENGTH_BUCKETS}
    answer = {bucket: tuple(row["answer_latency_s"] for row in rows)
              for bucket, rows in by_length.items()}
    first = {bucket: tuple(row["ttft_s"] for row in rows if row["ttft_s"] is not None)
             for bucket, rows in by_length.items()}
    answer_p50 = {bucket: _legacy_nearest_rank(answer[bucket], .5) for bucket in LENGTH_BUCKETS}
    answer_p95 = {bucket: _legacy_nearest_rank(answer[bucket], .95) for bucket in LENGTH_BUCKETS}
    first_p50 = {bucket: _legacy_nearest_rank(first[bucket], .5) for bucket in LENGTH_BUCKETS}
    first_p95 = {bucket: _legacy_nearest_rank(first[bucket], .95) for bucket in LENGTH_BUCKETS}
    warmup_counts = {bucket: sum(row["case"]["length"] == bucket for row in warmups)
                     for bucket in LENGTH_BUCKETS}
    success = sum(bool(row["e2e_complete"] and row["placement_matches"]
                       and row["evaluation"] and row["evaluation"]["success"])
                  for row in measured)
    all_rows = [*measured, *endurance]
    e2e = all(bool(row["e2e_complete"]) for row in all_rows)
    placement_matches = all(bool(row["placement_matches"]) for row in all_rows)
    placement_evidence = all(bool(row["result"] and
                                  row["result"]["placement_evidence"])
                             for row in all_rows)
    correctness = all(bool(row["evaluation"] and row["evaluation"]["quality_pass"]
                           and row["evaluation"]["success"])
                      for row in measured)
    tool_safety = all(bool(row["evaluation"] and row["evaluation"]["tool_safe"])
                      for row in all_rows)
    stable = bool(endurance) and sustained_seconds >= config.endurance_seconds and all(
        row["e2e_complete"] and row["error"] is None
        and row["evaluation"] and row["evaluation"]["success"]
        and row["evaluation"]["quality_pass"]
        and row["evaluation"]["tool_safe"] for row in endurance
    )
    telemetry_present = all(bool(row["telemetry"]["raw_samples"])
                            and not row["telemetry"]["errors"] for row in all_rows)
    sample_counts_ok = all(len(by_length[bucket]) >= 20 and
                           len(first[bucket]) == len(by_length[bucket]) and
                           warmup_counts[bucket] >= 1 for bucket in LENGTH_BUCKETS)
    compliant = bool(config.protocol_minimums and sample_counts_ok and
                     sustained_seconds >= 1800 and e2e and placement_matches
                     and telemetry_present and cold_start_s is not None)
    return RouteProfile(
        route=route,
        task_class=measured[0]["case"]["task_class"] if measured else "",
        measured_attempts=len(measured),
        measured_successes=success,
        answer_latency_s=answer,
        ttft_s=first,
        answer_p50_s=answer_p50,
        answer_p95_s=answer_p95,
        ttft_p50_s=first_p50,
        ttft_p95_s=first_p95,
        warmups_per_length=warmup_counts,
        sustained_seconds=sustained_seconds,
        endurance_wall_seconds=endurance_wall_seconds,
        endurance_requests=len(endurance),
        correctness_pass=correctness,
        tool_safety_pass=tool_safety,
        stability_pass=stable,
        e2e_trace_pass=e2e,
        actual_placement_matches=placement_matches,
        placement_evidence_present=placement_evidence,
        telemetry_present=telemetry_present,
        cold_start_s=cold_start_s,
        protocol_compliant=compliant,
    )



def _route(backend="llama.cpp"):
    return ProfileRoute("route", "model", "artifact", "revision", "a" * 64,
                        "Q4", backend, "CPU")


def _row(phase="measured", bucket="short", latency=1.0, ttft=0.1):
    return {
        "phase": phase,
        "case": {"length": bucket, "task_class": "browser_text"},
        "answer_latency_s": latency,
        "ttft_s": ttft,
        "e2e_complete": True,
        "placement_matches": True,
        "result": {"placement_evidence": {"actual": "CPU"}, "final_answer": "done"},
        "evaluation": {"success": True, "quality_pass": True, "tool_safe": True},
        "telemetry": {"raw_samples": [{"ram_used_bytes": 1024}], "errors": []},
        "error": None,
    }


def _complete_rows():
    rows = []
    for bucket in LENGTH_BUCKETS:
        rows.append(_row("warmup", bucket, latency=9999.0, ttft=999.0))
        rows.extend(_row("measured", bucket, latency=float(i), ttft=i / 10)
                    for i in range(20, 0, -1))
    rows.extend((_row("endurance", "short", 901.0),
                 _row("endurance", "long", 902.0)))
    return rows


def _arguments(**updates):
    result = {"cold_start_s": 12.5, "sustained_seconds": 1803.0,
              "endurance_wall_seconds": 1850.0, "config": ProfileConfig()}
    result.update(updates)
    return result


class _WeakPayload(dict):
    pass


class _WeakRow(dict):
    pass


class StreamingRouteProfileTests(unittest.TestCase):
    def assert_parity(self, rows, **updates):
        arguments = _arguments(**updates)
        route = _route()
        expected = _legacy_route_profile(route, rows, **arguments)
        self.assertEqual(_route_profile(route, rows, **arguments), expected)
        accumulator = _RouteProfileAccumulator(route)
        for row in rows:
            accumulator.add(row)
        actual = accumulator.finish(**arguments)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.qualification_fields(
            conditions=ProfileConditions("CPU", "Windows", {}, {}, "AC", "suite", "env"),
            raw_evidence="samples.jsonl", task_class=actual.task_class,
            memory_pass=True, cancel_recovery_pass=True, actual_placement_verified=True,
        ), expected.qualification_fields(
            conditions=ProfileConditions("CPU", "Windows", {}, {}, "AC", "suite", "env"),
            raw_evidence="samples.jsonl", task_class=expected.task_class,
            memory_pass=True, cancel_recovery_pass=True, actual_placement_verified=True,
        ))
        return actual

    def test_complete_protocol_exact_nearest_rank_and_order(self):
        result = self.assert_parity(_complete_rows())
        self.assertTrue(result.protocol_compliant)
        self.assertTrue(result.stability_pass)
        self.assertEqual(result.measured_attempts, 60)
        self.assertEqual(result.measured_successes, 60)
        for bucket in LENGTH_BUCKETS:
            self.assertEqual(result.answer_latency_s[bucket], tuple(float(i) for i in range(20, 0, -1)))
            self.assertEqual(result.answer_p50_s[bucket], 10.0)
            self.assertEqual(result.answer_p95_s[bucket], 19.0)
            self.assertEqual(result.ttft_p50_s[bucket], 1.0)
            self.assertEqual(result.ttft_p95_s[bucket], 1.9)
            self.assertEqual(result.warmups_per_length[bucket], 1)

    def test_all_quality_placement_telemetry_gates_and_failure_phases_match(self):
        changes = (
            ("e2e_false", lambda row: row.update(e2e_complete=False)),
            ("placement_false", lambda row: row.update(placement_matches=False)),
            ("missing_result", lambda row: row.update(result=None)),
            ("missing_placement_evidence", lambda row: row["result"].update(placement_evidence={})),
            ("missing_evaluation", lambda row: row.update(evaluation=None)),
            ("unsuccessful", lambda row: row["evaluation"].update(success=False)),
            ("quality_failed", lambda row: row["evaluation"].update(quality_pass=False)),
            ("unsafe", lambda row: row["evaluation"].update(tool_safe=False)),
            ("missing_telemetry", lambda row: row["telemetry"].update(raw_samples=[])),
            ("telemetry_error", lambda row: row["telemetry"].update(errors=["query failed"])),
            ("request_error", lambda row: row.update(error="runner failed")),
            ("no_visible_output", lambda row: row.update(ttft_s=None)),
        )
        for phase in ("warmup", "measured", "endurance"):
            for position in (0, -1):
                for name, change in changes:
                    with self.subTest(phase=phase, position=position, failure=name):
                        rows = _complete_rows()
                        candidates = [row for row in rows if row["phase"] == phase]
                        change(candidates[position])
                        self.assert_parity(rows)

    def test_failed_measured_rows_remain_in_samples_and_attempts(self):
        rows = [_row(latency=5.0), _row(latency=500.0, ttft=None)]
        rows[-1].update(error="runner failed", evaluation=None, result=None,
                        e2e_complete=False, placement_matches=False)
        result = self.assert_parity(rows, sustained_seconds=0, endurance_wall_seconds=0)
        self.assertEqual(result.measured_attempts, 2)
        self.assertEqual(result.measured_successes, 1)
        self.assertEqual(result.answer_latency_s["short"], (5.0, 500.0))
        self.assertEqual(result.answer_p95_s["short"], 500.0)
        self.assertEqual(result.ttft_s["short"], (0.1,))
        self.assertFalse(result.correctness_pass)
        self.assertFalse(result.e2e_trace_pass)
        self.assertFalse(result.actual_placement_matches)
        self.assertFalse(result.placement_evidence_present)
        self.assertFalse(result.tool_safety_pass)

    def test_endurance_failure_is_not_filtered_or_counted_as_measured(self):
        rows = _complete_rows()
        rows[-1]["evaluation"]["success"] = False
        result = self.assert_parity(rows)
        self.assertEqual(result.endurance_requests, 2)
        self.assertEqual(result.measured_attempts, 60)
        self.assertTrue(result.correctness_pass)
        self.assertFalse(result.stability_pass)
        self.assertNotIn(902.0, result.answer_latency_s["long"])

    def test_warmup_failures_do_not_affect_measured_or_endurance_gates(self):
        rows = _complete_rows()
        for row in rows:
            if row["phase"] == "warmup":
                row.update(e2e_complete=False, placement_matches=False, evaluation=None,
                           result=None, telemetry={"raw_samples": [], "errors": ["bad"]},
                           error="warmup failed", ttft_s=None)
        result = self.assert_parity(rows)
        self.assertTrue(result.protocol_compliant)
        self.assertTrue(result.stability_pass)
        self.assertEqual(result.measured_successes, 60)
        self.assertTrue(all(9999.0 not in values for values in result.answer_latency_s.values()))

    def test_protocol_thresholds_and_missing_visible_outputs_match(self):
        for missing in ("short_measured", "short_warmup", "short_ttft"):
            with self.subTest(missing=missing):
                rows = _complete_rows()
                if missing == "short_measured":
                    rows.pop(1)
                elif missing == "short_warmup":
                    rows.pop(0)
                else:
                    rows[1]["ttft_s"] = None
                self.assertFalse(self.assert_parity(rows).protocol_compliant)
        for updates in (
            {"cold_start_s": None}, {"sustained_seconds": 1799.99},
            {"config": ProfileConfig(warmups_per_length=0)},
            {"config": ProfileConfig(measured_per_length=19)},
            {"config": ProfileConfig(endurance_seconds=1799)},
        ):
            with self.subTest(updates=updates):
                self.assertFalse(self.assert_parity(_complete_rows(), **updates).protocol_compliant)
        self.assertTrue(self.assert_parity(_complete_rows(), cold_start_s=0.0).protocol_compliant)

    def test_empty_no_endurance_and_ignored_phases_preserve_prior_semantics(self):
        for rows in ([], [_row("warmup")], [_row()], [_row("ignored")]):
            with self.subTest(phases=[row["phase"] for row in rows]):
                result = self.assert_parity(rows, sustained_seconds=0.0,
                                            config=ProfileConfig(endurance_seconds=0))
                self.assertFalse(result.stability_pass)
                self.assertFalse(result.protocol_compliant)
        rows = _complete_rows()
        rows.insert(0, {"phase": "ignored"})
        self.assert_parity(rows)

    def test_task_class_comes_from_first_measured_and_missing_lengths_match(self):
        rows = [_row("warmup", "long"), _row("endurance", "long"),
                _row("measured", "medium"), _row("measured", "unknown")]
        rows[0]["case"]["task_class"] = "warmup_class"
        rows[1]["case"]["task_class"] = "endurance_class"
        rows[2]["case"]["task_class"] = "first_measured"
        rows[3]["case"]["task_class"] = "later_measured"
        result = self.assert_parity(rows)
        self.assertEqual(result.task_class, "first_measured")
        self.assertEqual(result.measured_attempts, 2)
        self.assertEqual(sum(len(values) for values in result.answer_latency_s.values()), 1)

    def test_snapshot_does_not_alias_accumulator_lists_or_warmup_counts(self):
        accumulator = _RouteProfileAccumulator(_route())
        accumulator.add(_row())
        accumulator.add(_row("warmup"))
        before = accumulator.finish(**_arguments())
        accumulator.add(_row(latency=2.0))
        accumulator.add(_row("warmup"))
        self.assertEqual(before.answer_latency_s["short"], (1.0,))
        self.assertEqual(before.warmups_per_length["short"], 1)

    def test_accumulator_does_not_retain_request_or_nested_payloads(self):
        accumulator = _RouteProfileAccumulator(_route())
        for phase in ("warmup", "measured", "endurance"):
            with self.subTest(phase=phase):
                row = _WeakRow(_row(phase))
                payload = _WeakPayload(private_payload="x" * 65536)
                row["events"] = [{"payload": payload}]
                row["telemetry"]["raw_samples"][0]["nested"] = payload
                row["fixture_setup"] = payload
                row["result"]["private_output"] = payload
                row_ref, payload_ref = weakref.ref(row), weakref.ref(payload)
                accumulator.add(row)
                del row, payload
                gc.collect()
                self.assertIsNone(row_ref())
                self.assertIsNone(payload_ref())
        self.assertEqual(accumulator.answer["short"], [1.0])
        self.assertEqual(accumulator.first["short"], [0.1])

    def test_endurance_count_grows_without_latency_or_payload_history(self):
        accumulator = _RouteProfileAccumulator(_route())
        for row in _complete_rows():
            accumulator.add(row)
        del row
        state_before = copy.deepcopy(vars(accumulator))
        for _ in range(2000):
            row = _WeakRow(_row("endurance", latency=1000.0, ttft=100.0))
            payload = _WeakPayload(event="x" * 4096)
            row["events"] = [payload]
            reference = weakref.ref(payload)
            accumulator.add(row)
            del row, payload
            self.assertIsNone(reference())
        after = dict(vars(accumulator))
        self.assertEqual(after.pop("endurance_requests"), 2002)
        state_before.pop("endurance_requests")
        self.assertEqual(after, state_before)

    def test_strata_qualification_still_refuses_manufactured_actual_placement(self):
        result = _route_profile(_route("external.strata.text.v1"), _complete_rows(), **_arguments())
        fields = result.qualification_fields(
            conditions=ProfileConditions("CPU", "Windows", {}, {}, "AC", "suite", "env"),
            raw_evidence="samples.jsonl", task_class="browser_text",
            memory_pass=True, cancel_recovery_pass=True, actual_placement_verified=True,
        )
        self.assertFalse(fields["actual_placement_verified"])


class StreamingRawHashTests(unittest.TestCase):
    def test_exact_binary_hash_for_empty_lf_crlf_and_multiple_chunks(self):
        for raw in (b"", b'{"a":1}\n', b'{"a":1}\r\n',
                    (b'{"a":1}\r\n' * 300000) + b"tail\x00"):
            with self.subTest(size=len(raw)), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "samples.jsonl"
                path.write_bytes(raw)
                with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read forbidden")):
                    actual = _sha256_file(path)
                self.assertEqual(actual, hashlib.sha256(raw).hexdigest())
        self.assertNotEqual(hashlib.sha256(b"row\n").hexdigest(), hashlib.sha256(b"row\r\n").hexdigest())

    def test_hash_reads_in_finite_chunks_and_closes_binary_reader(self):
        reads = []
        raw = b"abcdef\r\n" * 300000

        class Reader(io.BytesIO):
            def read(self, size=-1):
                reads.append(size)
                if not 0 < size <= 1 << 20:
                    raise AssertionError("unbounded binary read")
                return super().read(size)

        reader = Reader(raw)

        class File:
            def open(self, mode):
                self.mode = mode
                return reader

        file = File()
        self.assertEqual(_sha256_file(file), hashlib.sha256(raw).hexdigest())
        self.assertEqual(file.mode, "rb")
        self.assertGreater(len(reads), 2)
        self.assertTrue(reader.closed)


class StreamingProfileIntegrationTests(unittest.TestCase):
    def test_run_profile_releases_rows_before_next_request_and_preserves_raw_evidence(self):
        route = _route()
        cases = {bucket: [AgentCase(bucket, "browser_text", "zh-CN", bucket,
                                   "真实输入 " + bucket, reference="done")]
                 for bucket in LENGTH_BUCKETS}
        conditions = ProfileConditions("CPU", "Windows", {}, {}, "AC", "suite", "env")
        config = ProfileConfig(warmups_per_length=1, measured_per_length=2,
                               endurance_seconds=.5)
        previous_refs = []
        calls, setups = [], []

        async def one_request(**arguments):
            gc.collect()
            self.assertTrue(all(reference() is None for reference in previous_refs))
            row = _WeakRow(_row(arguments["phase"], arguments["case"].length, .25, .05))
            payload = _WeakPayload(full_private_event="raw evidence " + "x" * 4096)
            row["events"] = [{"kind": "fixture_event", "payload": payload}]
            row["telemetry"]["raw_samples"][0]["detail"] = payload
            row["result"]["final_answer"] = "done"
            row["case"] = {"length": arguments["case"].length,
                           "task_class": arguments["case"].task_class,
                           "prompt": arguments["case"].prompt}
            calls.append((arguments["phase"], arguments["case"].length,
                          arguments["repetition"]))
            previous_refs[:] = [weakref.ref(row), weakref.ref(payload)]
            return row

        def before_request(_route, case, phase, repetition):
            gc.collect()
            self.assertTrue(all(reference() is None for reference in previous_refs))
            setups.append((phase, case.length, repetition))
            return {"fixture_seeded": True, "phase": phase, "repetition": repetition}

        async def forbidden_runner(*_):
            raise AssertionError("stubbed request must not invoke a backend")

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(profile_module, "_one_request", one_request), patch.object(
                Path, "read_bytes", side_effect=AssertionError("whole-file read forbidden")
            ):
                summary = asyncio.run(run_profile(
                    routes=[route], cases_by_length=cases, runner=forbidden_runner,
                    evaluator=lambda *_: None, conditions=conditions, output_dir=temporary,
                    config=config, before_request=before_request,
                    prepare=lambda current: Preparation(True, current.artifact_id, "CPU", {"cold": True}),
                ))
            gc.collect()
            self.assertTrue(all(reference() is None for reference in previous_refs))
            raw_bytes = summary.raw_jsonl.read_bytes()
            self.assertEqual(summary.raw_sha256, hashlib.sha256(raw_bytes).hexdigest())
            records = [json.loads(line) for line in raw_bytes.splitlines()]
            self.assertEqual(records[0]["record_type"], "manifest")
            self.assertEqual(records[1]["record_type"], "route_prepare")
            rows = records[2:]
            self.assertEqual(len(rows), 11)
            self.assertEqual(calls, setups)
            self.assertEqual([(row["phase"], row["case"]["length"], row["fixture_setup"]["repetition"])
                              for row in rows], calls)
            self.assertTrue(all(row["fixture_setup"]["fixture_seeded"] for row in rows))
            self.assertTrue(all(row["events"][0]["payload"]["full_private_event"].startswith("raw evidence")
                                for row in rows))
            self.assertTrue(all(row["telemetry"]["raw_samples"][0]["detail"]
                                == row["events"][0]["payload"] for row in rows))
            actual = summary.routes[route.route_id]
            expected = _legacy_route_profile(
                route, rows, cold_start_s=actual.cold_start_s,
                sustained_seconds=.5, endurance_wall_seconds=actual.endurance_wall_seconds,
                config=config,
            )
            self.assertEqual(actual, expected)
            self.assertEqual(json.loads((summary.run_directory / "summary.json").read_text(encoding="utf-8"))
                             ["raw_sha256"], summary.raw_sha256)

    def test_real_request_failures_remain_in_raw_rows_and_summary(self):
        for failure in ("runner", "evaluator", "placement", "quality", "tool", "telemetry", "incomplete", "no_visible"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                route = _route()
                cases = {bucket: [AgentCase(bucket, "basic", "en", bucket, "input")]
                         for bucket in LENGTH_BUCKETS}

                async def runner(current, _case, emit):
                    if failure == "runner":
                        raise RuntimeError("fixture runner failure")
                    if failure != "no_visible":
                        emit("visible_token", "done")
                    return AgentRunResult(
                        "done", failure != "incomplete", current.model_id,
                        current.artifact_id, "wrong" if failure == "placement" else "CPU",
                        current.backend, placement_evidence={"actual": "CPU"},
                    )

                def evaluate(_case, _result):
                    if failure == "evaluator":
                        raise RuntimeError("fixture evaluator failure")
                    return Evaluation(True, failure != "quality", 1.0, failure != "tool")

                def telemetry():
                    if failure == "telemetry":
                        raise RuntimeError("fixture telemetry failure")
                    return {"ram_used_bytes": 1024}

                config = ProfileConfig(warmups_per_length=1, measured_per_length=1,
                                       endurance_seconds=0)
                summary = asyncio.run(run_profile(
                    routes=[route], cases_by_length=cases, runner=runner, evaluator=evaluate,
                    telemetry=telemetry,
                    conditions=ProfileConditions("CPU", "Windows", {}, {}, "AC", "suite", "env"),
                    output_dir=temporary, config=config,
                ))
                rows = [json.loads(line) for line in summary.raw_jsonl.read_text(encoding="utf-8").splitlines()
                        if json.loads(line).get("record_type") == "request"]
                actual = summary.routes[route.route_id]
                self.assertEqual(actual, _legacy_route_profile(
                    route, rows, cold_start_s=None, sustained_seconds=0,
                    endurance_wall_seconds=actual.endurance_wall_seconds, config=config,
                ))
                self.assertEqual(actual.measured_attempts, 3)
                self.assertEqual(sum(len(values) for values in actual.answer_latency_s.values()), 3)
                self.assertEqual(len(rows), 6)
                if failure in {"runner", "evaluator"}:
                    self.assertTrue(all(row["error"] for row in rows))
                    self.assertEqual(actual.measured_successes, 0)
                if failure == "no_visible":
                    self.assertTrue(all(row["ttft_s"] is None for row in rows))

    def test_raw_fsync_completes_before_each_row_is_accumulated(self):
        writes = []
        original_fsync = profile_module.os.fsync
        original_add = _RouteProfileAccumulator.add

        def fsync(fd):
            original_fsync(fd)
            writes.append(fd)

        def add(current, row):
            # One manifest and one persisted row precede the first accumulation.
            self.assertEqual(len(writes), current.measured_attempts + 2)
            original_add(current, row)

        async def runner(route, _case, emit):
            emit("visible_token", "done")
            return AgentRunResult("done", True, route.model_id, route.artifact_id,
                                  "CPU", route.backend, placement_evidence={"actual": "CPU"})

        with tempfile.TemporaryDirectory() as temporary, patch.object(profile_module.os, "fsync", fsync), patch.object(
            _RouteProfileAccumulator, "add", add
        ):
            asyncio.run(run_profile(
                routes=[_route()], cases_by_length={bucket: [AgentCase(bucket, "basic", "en", bucket, "input")]
                                                   for bucket in LENGTH_BUCKETS},
                runner=runner, evaluator=lambda *_: Evaluation(True, True, 1.0, True),
                conditions=ProfileConditions("CPU", "Windows", {}, {}, "AC", "suite", "env"),
                output_dir=temporary,
                config=ProfileConfig(warmups_per_length=0, measured_per_length=1, endurance_seconds=0),
            ))
        self.assertEqual(len(writes), 4)


if __name__ == "__main__":
    unittest.main()
