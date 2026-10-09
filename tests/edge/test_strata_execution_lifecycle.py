# SPDX-License-Identifier: Apache-2.0
"""Production Stage helpers/lifecycle tests; no native engine or model execution."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

import pytest
from vllm_omni.engine.backends import strata

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]
namespace = vars(strata)
Stage = strata.StrataTextStageClient


def bare_stage():
    stage = Stage.__new__(Stage)
    stage._execution_bridge = object()
    stage._execution_report_lock = threading.RLock()
    stage._execution_report_request = None
    stage._execution_active_request = ("request", 1, "generation")
    stage._last_execution_report = None
    stage._execution_verifier = None
    stage._cancel = threading.Event()
    return stage


def request(epoch=1):
    return types.SimpleNamespace(request_id="request", epoch=epoch, worker_generation="generation")


class TestRequestLifecycle(unittest.TestCase):
    def test_real_active_request_only_and_exactly_once_detached_result(self):
        stage = bare_stage()
        calls = []

        def finish(actual, **flags):
            calls.append((actual.epoch, flags))
            return {"status": "synthetic", "items": [1]}

        stage._diagnostics = types.SimpleNamespace(finish_execution_request=finish)
        self.assertIsNone(stage._finish_execution_observation(request(2), completed=True, lifecycle_outcome="normal"))
        first = stage._finish_execution_observation(request(), completed=True, lifecycle_outcome="normal")
        first["items"].append(2)
        self.assertEqual(stage.last_execution_observation()["items"], [1])
        self.assertEqual(
            stage._finish_execution_observation(request(), completed=False, lifecycle_outcome="drained")["items"], [1]
        )
        self.assertEqual(len(calls), 1)
        self.assertIsNone(stage._execution_active_request)

    def test_failed_normal_detach_retires_through_error_and_preserves_unavailable(self):
        stage = bare_stage()
        calls = []

        def finish(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("private fixture value")

        stage._diagnostics = types.SimpleNamespace(finish_execution_request=finish)
        with self.assertRaisesRegex(RuntimeError, "could not be detached"):
            stage._finish_execution_observation(request(), completed=True, lifecycle_outcome="normal")
        report = stage._finish_execution_observation(request(), completed=False, lifecycle_outcome="error")
        self.assertEqual(report["status"], "unavailable")
        self.assertFalse(report["complete"])
        self.assertIsNone(report["native_observation"])
        self.assertNotIn("private fixture value", json.dumps(report))
        self.assertEqual(len(calls), 1)

    def test_both_report_failures_still_attempt_close_and_preserve_drain_failure(self):
        stage = bare_stage()
        events = []

        def fail_io(*args, **kwargs):
            events.append("io")
            raise RuntimeError("private fixture")

        def fail_exec(*args, **kwargs):
            events.append("execution")
            raise RuntimeError("private fixture")

        def fail_close():
            events.append("close")
            raise RuntimeError("private fixture")

        stage._finish_io_observation = fail_io
        stage._finish_execution_observation = fail_exec
        verifier = types.SimpleNamespace(close=fail_close)
        stage._execution_verifier = verifier
        io_report, execution_report, drained = stage._finish_failure_observations(
            request(), reason="request_cancelled", lifecycle_outcome="cancelled", drained=True
        )
        self.assertEqual(events, ["io", "execution", "close"])
        self.assertEqual(io_report["status"], "unavailable")
        self.assertEqual(execution_report["status"], "unavailable")
        self.assertFalse(drained)
        self.assertIs(stage._execution_verifier, verifier)

    def test_unjoined_reader_cannot_clear_verifier(self):
        stage = bare_stage()
        calls = []
        stage._execution_verifier = types.SimpleNamespace(close=lambda: calls.append(1))
        self.assertFalse(stage._retire_execution_observer(False))
        self.assertEqual(calls, [])
        self.assertTrue(stage._retire_execution_observer(True))
        self.assertEqual(calls, [1])
        self.assertIsNone(stage._execution_verifier)


class TestAdapterLoading(unittest.TestCase):
    def test_verified_loader_reuses_only_its_own_module_and_rejects_source_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "adapter").mkdir()
            name = "omni_stage_source_fixture"
            path = root / "adapter" / (name + ".py")
            data = b"value = 7\n"
            path.write_bytes(data)
            old_pins = namespace["_EXECUTION_ADAPTER_SHA256"]
            namespace["_EXECUTION_ADAPTER_SHA256"] = {name: hashlib.sha256(data).hexdigest()}
            try:
                first = namespace["_load_execution_adapters"](root, {path})[name]
                self.assertEqual(first.value, 7)
                self.assertIs(namespace["_load_execution_adapters"](root, {path})[name], first)
                path.write_bytes(b"value = 9\n")
                with self.assertRaisesRegex(ValueError, "reviewed source"):
                    namespace["_load_execution_adapters"](root, {path})
            finally:
                namespace["_EXECUTION_ADAPTER_SHA256"] = old_pins
                namespace["_EXECUTION_LOADED_MODULES"].pop(name, None)
                sys.modules.pop(name, None)

    def test_unowned_cached_module_with_matching_file_cannot_bypass_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "adapter").mkdir()
            name = "omni_stage_collision_fixture"
            path = root / "adapter" / (name + ".py")
            path.write_bytes(b"value = 7\n")
            old_pins = namespace["_EXECUTION_ADAPTER_SHA256"]
            namespace["_EXECUTION_ADAPTER_SHA256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()}
            sys.modules[name] = types.SimpleNamespace(__file__=str(path), value=99)
            try:
                with self.assertRaisesRegex(ValueError, "not owned"):
                    namespace["_load_execution_adapters"](root, {path})
            finally:
                namespace["_EXECUTION_ADAPTER_SHA256"] = old_pins
                sys.modules.pop(name, None)


class TestReceiptAndAdmission(unittest.TestCase):
    def test_stable_root_allows_fresh_generation_and_rejects_missing_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "runtime"
            root.mkdir()
            receipts = parent / "receipts"
            receipts.mkdir()
            (receipts / "generation-old").mkdir()
            for name in ("descriptor.json", "context.json"):
                (root / name).write_text("{}")
            config = dict(
                schema=namespace["_EXECUTION_CONFIG_SCHEMA"],
                descriptor_file="descriptor.json",
                static_identity_sha256="a" * 64,
                source_context_file="context.json",
                receipt_root=str(receipts),
                workspace_bytes=512 << 20,
                receipt_storage_bytes=16 << 20,
            )
            result = namespace["_execution_settings"](config, root)
            self.assertEqual(Path(result["receipt_root"]), receipts)
            self.assertTrue((receipts / "generation-old").exists())
            for field, value in (
                ("workspace_bytes", True),
                ("workspace_bytes", 2 << 20),
                ("receipt_storage_bytes", 0),
                ("receipt_root", str(root)),
            ):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    namespace["_execution_settings"](config | {field: value}, root)
