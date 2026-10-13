# SPDX-License-Identifier: Apache-2.0
"""Closed controllers must not own failed-turn traceback payloads."""

import asyncio
import gc
import threading
import unittest
import weakref
from concurrent.futures import Future
from unittest.mock import Mock

from vllm_omni.edge.agent.controller import AgentController


class Payload:
    pass


def failed_turn():
    future = Future()
    payload = Payload()
    reference = weakref.ref(payload)
    try:
        raise RuntimeError("model failed after observing an image")
    except RuntimeError as error:
        future.set_exception(error)
    return future, reference


class CloseFutureMemoryTests(unittest.TestCase):
    def controller(self, future):
        controller = AgentController.__new__(AgentController)
        controller._current = future
        controller._lock = threading.RLock()
        controller._turn_done = threading.Event()
        controller._turn_done.set()
        controller.cancel = Mock()
        controller.backends = {}
        controller._tool_executor = Mock()
        controller.tools = Mock()
        controller.memory = Mock()
        controller._loop = None
        controller._thread = None
        controller._resource_finalize = None
        controller._resource_unload = None
        controller._turn_owner = None
        controller._epoch = 0
        return controller

    def test_idle_release_preserves_memory_and_tools(self):
        controller = self.controller(None)
        proof = {"resident_route": None, "ledger": {
            "owners": [], "quarantined": [], "reserved": {"host_ram": 0}}}
        controller._resource_unload = Mock(return_value=proof)
        self.assertEqual(controller.release_idle_models(), proof)
        controller.memory.close.assert_not_called()
        controller.memory.delete_all.assert_not_called()
        controller.tools.close.assert_not_called()
        self.assertTrue(controller._turn_done.is_set())
        self.assertIsNone(controller._turn_owner)

    def test_idle_release_rejects_active_turn_and_retained_lease(self):
        controller = self.controller(None)
        controller._resource_unload = Mock(return_value={"resident_route": None, "ledger": {
            "owners": ["model"], "quarantined": [], "reserved": {"host_ram": 1}}})
        controller._turn_done.clear()
        with self.assertRaisesRegex(RuntimeError, "active request"):
            controller.release_idle_models()
        controller._resource_unload.assert_not_called()
        controller._turn_done.set()
        with self.assertRaisesRegex(RuntimeError, "empty resource ledger"):
            controller.release_idle_models()
        self.assertTrue(controller._turn_done.is_set())

    def test_idle_release_blocks_submission_until_coordinator_drains(self):
        controller = self.controller(None)
        entered, proceed = threading.Event(), threading.Event()
        errors = []

        def drain():
            entered.set()
            if not proceed.wait(5):
                raise TimeoutError("test did not release coordinator")
            raise RuntimeError("release failed; lease quarantined")

        controller._resource_unload = drain

        def release():
            try:
                controller.release_idle_models()
            except RuntimeError as error:
                errors.append(str(error))

        worker = threading.Thread(target=release)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaisesRegex(RuntimeError, "already active"):
                controller.submit("new request")
            with self.assertRaises(RuntimeError):
                controller.delete_all_memory()
            with self.assertRaisesRegex(RuntimeError, "already active"):
                asyncio.run(controller.run_turn(
                    "direct request", request_id="idle-model-release", epoch=controller._epoch))
        finally:
            proceed.set()
            worker.join(timeout=5)
        self.assertEqual(errors, ["release failed; lease quarantined"])
        self.assertTrue(controller._turn_done.is_set())

    def test_close_releases_unowned_failed_turn_payload(self):
        future, reference = failed_turn()
        controller = self.controller(future)
        del future
        gc.collect()
        self.assertIsNotNone(reference())
        controller.close()
        gc.collect()
        self.assertIsNone(controller._current)
        self.assertIsNone(reference())

    def test_caller_keeps_original_failure_until_it_releases_future(self):
        future, reference = failed_turn()
        controller = self.controller(future)
        controller.close()
        self.assertIsNone(controller._current)
        self.assertIsInstance(future.exception(), RuntimeError)
        self.assertEqual(str(future.exception()), "model failed after observing an image")
        gc.collect()
        self.assertIsNotNone(reference())
        del future
        gc.collect()
        self.assertIsNone(reference())

    def test_cancelled_future_does_not_skip_turn_cleanup(self):
        future = Future()
        future.cancel()
        controller = self.controller(future)
        controller._turn_done.clear()
        finished = threading.Event()

        def close():
            controller.close()
            finished.set()

        worker = threading.Thread(target=close)
        worker.start()
        try:
            self.assertFalse(finished.wait(0.05))
            self.assertIs(controller._current, future)
            controller.tools.close.assert_not_called()
        finally:
            controller._turn_done.set()
            worker.join(timeout=5)
        self.assertTrue(finished.is_set())
        self.assertIsNone(controller._current)

    def test_close_waits_for_idle_release_without_a_request_future(self):
        controller = self.controller(None)
        controller._turn_done.clear()
        finished = threading.Event()

        def close():
            controller.close()
            finished.set()

        worker = threading.Thread(target=close)
        worker.start()
        try:
            self.assertFalse(finished.wait(.05))
            controller.tools.close.assert_not_called()
            controller.memory.close.assert_not_called()
        finally:
            controller._turn_done.set()
            worker.join(timeout=5)
        self.assertTrue(finished.is_set())


if __name__ == "__main__":
    unittest.main()
