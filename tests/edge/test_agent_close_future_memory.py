# SPDX-License-Identifier: Apache-2.0
"""Closed controllers must not own failed-turn traceback payloads."""

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
        return controller

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


if __name__ == "__main__":
    unittest.main()
