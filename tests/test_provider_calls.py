"""Requests that ignore socket timeouts cannot block later Streamlit reruns."""

import threading
import time
import unittest

from adfm_core.provider_calls import BoundedProvider


class BoundedProviderTests(unittest.TestCase):
    def test_timeout_discards_late_result_and_does_not_spawn_more_workers(self):
        source = BoundedProvider()
        released = threading.Event()
        entered = threading.Event()

        def stalled():
            entered.set()
            released.wait(2)
            return [1]

        try:
            started = time.perf_counter()
            with self.assertRaises(TimeoutError):
                source.call("first", stalled, deadline=started + .03)
            self.assertTrue(entered.is_set())
            self.assertLess(time.perf_counter() - started, .2)
            with self.assertRaises(TimeoutError):
                source.call("second", lambda: self.fail("extra worker"), deadline=time.perf_counter() + .02)
        finally:
            released.set()
        self.assertTrue(source.wait_idle(1))
        self.assertEqual(source.call("first", lambda: [2], deadline=time.perf_counter() + 1), [2])

    def test_success_cache_returns_independent_copies_even_after_budget_elapsed(self):
        source = BoundedProvider()
        first = source.call("key", lambda: [1], deadline=time.perf_counter() + 1)
        first.append(99)
        self.assertEqual(source.call("key", lambda: self.fail("cached"), deadline=0), [1])
        source.clear()
        self.assertEqual(source.call("key", lambda: [2], deadline=time.perf_counter() + 1), [2])

    def test_empty_response_and_exception_are_retried(self):
        source = BoundedProvider()
        self.assertEqual(source.call("key", lambda: [], deadline=time.perf_counter() + 1, valid=bool), [])
        with self.assertRaises(ValueError):
            source.call("key", lambda: (_ for _ in ()).throw(ValueError("offline")), deadline=time.perf_counter() + 1)
        self.assertEqual(source.call("key", lambda: [1], deadline=time.perf_counter() + 1), [1])

    def test_cache_limit_evicts_oldest_success(self):
        source = BoundedProvider(max_entries=1)
        source.call("a", lambda: [1], deadline=time.perf_counter() + 1)
        source.call("b", lambda: [2], deadline=time.perf_counter() + 1)
        self.assertEqual(source.call("a", lambda: [3], deadline=time.perf_counter() + 1), [3])
